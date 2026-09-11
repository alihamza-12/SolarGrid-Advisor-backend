"""FAISS + BM25 hybrid index store — ported from app.py lines 381-609."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
from typing import Optional

import numpy as np

from .chunking import BM25, chunk_pages
from .config import (
    CHUNKS_FILE, DEFAULT_EMB_MODEL, HF_CACHE, MANIFEST_FILE, VECTORS_FILE,
    load_json, save_json,
)
from .utils import norm_title, parse_date

try:
    import faiss
    from faiss import IndexFlatIP
    FAISS_OK = True
except Exception:
    FAISS_OK = False


class IndexStore:
    def __init__(self):
        self.manifest = load_json(MANIFEST_FILE, {})
        self.chunks: list[dict] = load_json(CHUNKS_FILE, [])
        self._bm25: Optional[BM25] = None
        self._bm25_key: Optional[str] = None
        self._embedder = None

    # -- embeddings ---------------------------------------------------------
    def emb_model_name(self) -> str:
        return self.manifest.get("model", DEFAULT_EMB_MODEL)

    def dim(self) -> int:
        return int(self.manifest.get("dim", 384))

    def get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        from sentence_transformers import SentenceTransformer
        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        name = self.emb_model_name()
        self._embedder = SentenceTransformer(name)
        return self._embedder

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self.get_embedder()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        return vecs.astype("float32")

    def _load_vectors(self) -> Optional[np.ndarray]:
        if VECTORS_FILE.exists() and self.chunks:
            try:
                return np.load(VECTORS_FILE)["arr"]
            except Exception:
                return None
        return None

    def _save_vectors(self, arr: np.ndarray) -> None:
        np.savez_compressed(VECTORS_FILE, arr=arr)
        self.manifest = {
            "model": self.emb_model_name(),
            "dim": int(arr.shape[1]) if arr.size else 0,
            "n": int(arr.shape[0]),
            "updated": dt.datetime.now().isoformat(timespec="seconds"),
        }
        save_json(MANIFEST_FILE, self.manifest)

    def bm25(self) -> BM25:
        key = hashlib.md5(
            ("|".join(c["text"][:60] for c in self.chunks) + str(len(self.chunks))).encode()
        ).hexdigest()
        if self._bm25 is None or self._bm25_key != key:
            self._bm25 = BM25([c["text"] for c in self.chunks])
            self._bm25_key = key
        return self._bm25

    # -- document ops --------------------------------------------------------
    def add_document(self, doc: dict, pages: list[str]) -> int:
        """doc: metadata dict. Embeds chunks, appends to index. Returns n_chunks."""
        chunks = chunk_pages(pages)
        base = len(self.chunks)
        texts = [c["text"] for c in chunks]
        vectors: Optional[np.ndarray] = None
        arr: Optional[np.ndarray] = None
        if texts:
            try:
                vectors = self.embed(texts) if FAISS_OK else None
            except Exception as e:
                print(f"[SGA] embedding unavailable ({e}) — indexing with keyword search only")
                vectors = None
        if vectors is not None:
            old = self._load_vectors()
            if old is not None and old.size and old.shape[0] == len(self.chunks):
                arr = np.vstack([old, vectors])
            else:
                arr = self.embed([c["text"] for c in self.chunks]) if self.chunks else vectors
        else:
            arr = self._load_vectors()
        for i, c in enumerate(chunks):
            self.chunks.append({
                "id": f'{doc["id"]}:{base + i}',
                "doc_id": doc["id"],
                "doc_title": doc["title"],
                "disco": doc.get("disco", "Other"),
                "doc_type": doc.get("doc_type", "Circular / Other"),
                "issue_date": doc.get("issue_date", ""),
                "effective_date": doc.get("effective_date", ""),
                "status": doc.get("status", "official"),
                "page": c["page"],
                "text": c["text"],
            })
        save_json(CHUNKS_FILE, self.chunks)
        if arr is not None and arr.size:
            self._save_vectors(arr)
        self._bm25 = None
        return len(chunks)

    def remove_document(self, doc_id: str) -> None:
        old = self._load_vectors()
        keep = [c for c in self.chunks if c["doc_id"] != doc_id]
        if old is not None and old.shape[0] == len(self.chunks):
            keep_ids = {c["id"] for c in keep}
            arr = old[[i for i, c in enumerate(self.chunks) if c["id"] in keep_ids]]
        else:
            arr = None
        self.chunks = keep
        save_json(CHUNKS_FILE, self.chunks)
        if arr is not None:
            self._save_vectors(arr)
        self._bm25 = None

    def rebuild_all(self) -> None:
        """Re-embed everything (used after switching the embedding model)."""
        texts = [c["text"] for c in self.chunks]
        if not texts:
            self._save_vectors(np.zeros((0, self.dim()), dtype="float32"))
            return
        try:
            self.get_embedder()
        except Exception as e:
            raise RuntimeError(
                f"Embedding model unavailable ({e}). `pip install -r requirements.txt` "
                "(sentence-transformers) is needed for vector search."
            )
        vecs = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            vecs.append(self.embed(batch))
        self._save_vectors(np.vstack(vecs))
        self._bm25 = None

    # -- search --------------------------------------------------------------
    def hybrid_search(self, query: str, top_k: int, use_vector: bool, use_bm25: bool,
                       filters: dict) -> tuple[list[dict], dict]:
        """Returns (ranked_chunks, diagnostics). filters:
        discos[], statuses[], min_effective(date|None), latest_only(bool)"""
        if not self.chunks:
            return [], {"candidates": 0, "top_score": 0.0}
        cand_idx = [i for i, c in enumerate(self.chunks) if self._pass_filters(c, filters)]
        if not cand_idx:
            return [], {"candidates": 0, "top_score": 0.0}
        cand = [self.chunks[i] for i in cand_idx]

        scores_vec = np.zeros(len(cand))
        scores_kw = np.zeros(len(cand))

        if use_vector and FAISS_OK and VECTORS_FILE.exists():
            arr = self._load_vectors()
            if arr is not None and arr.shape[0] == len(self.chunks):
                try:
                    qv = self.embed([query])
                    sub = arr[[i for i in cand_idx]].astype("float32")
                    idx = IndexFlatIP(sub.shape[1])
                    idx.add(sub)
                    sims, ids = idx.search(qv, min(len(cand), 60))
                    for s, i in zip(sims[0], ids[0]):
                        if i >= 0:
                            scores_vec[i] = max(float(s), 0.0)
                except Exception:
                    pass

        if use_bm25:
            try:
                bm = self.bm25()
                full = bm.score(query)
                for local_pos, gi in enumerate(cand_idx):
                    scores_kw[local_pos] = float(full[gi])
            except Exception:
                pass

        def norm(a: np.ndarray) -> np.ndarray:
            m = a.max()
            return a / m if m > 0 else a

        wv = 0.55 if (use_vector and scores_vec.max() > 0) else 0.0
        wk = 0.45 if (use_bm25 and scores_kw.max() > 0) else 0.0
        if wv == 0 and wk == 0:
            return [], {"candidates": len(cand), "top_score": 0.0}
        wv, wk = wv / (wv + wk), wk / (wv + wk)
        fused = wv * norm(scores_vec) + wk * norm(scores_kw)
        order = np.argsort(-fused)[:top_k]
        results = []
        for p in order:
            item = dict(cand[p])
            item["score"] = float(fused[p])
            item["vec_score"] = float(scores_vec[p])
            item["kw_score"] = float(scores_kw[p])
            results.append(item)
        top = max((r["score"] for r in results), default=0.0)
        return results, {"candidates": len(cand), "top_score": top}

    @staticmethod
    def _pass_filters(c: dict, f: dict) -> bool:
        if f.get("discos") and c["disco"] not in f["discos"]:
            return False
        if f.get("statuses") and c["status"] not in f["statuses"]:
            return False
        min_eff = f.get("min_effective")
        if min_eff:
            ce = parse_date(c.get("effective_date", ""))
            if ce is None or ce < min_eff:
                return False
        if f.get("latest_only"):
            best = f.get("latest_eff_map") or {}
            key = norm_title(c["doc_title"])
            be = best.get(key)
            if be and parse_date(c.get("effective_date", "")) != be:
                return False
        return True

    # -- versioning helper ----------------------------------------------------
    def latest_effective_map(self) -> dict:
        m: dict[str, dt.date] = {}
        for c in self.chunks:
            d = parse_date(c.get("effective_date", ""))
            if not d:
                continue
            key = norm_title(c["doc_title"])
            if key not in m or d > m[key]:
                m[key] = d
        return m
