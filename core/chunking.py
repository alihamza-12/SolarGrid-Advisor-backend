"""Text chunking + tiny BM25 — ported from app.py lines 264-334."""
from __future__ import annotations

import math
import re

import numpy as np

from .config import CHUNK_CHARS, CHUNK_OVERLAP, STOPWORDS


def chunk_pages(pages: list[str]) -> list[dict]:
    """Split per-page text into overlapping chunks (~900 chars)."""
    out = []
    for pno, raw in enumerate(pages, start=1):
        t = re.sub(r"[ \t]+", " ", raw or "")
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
        if not t:
            continue
        if len(t) <= CHUNK_CHARS:
            out.append({"page": pno, "text": t})
            continue
        start = 0
        while start < len(t):
            end = min(start + CHUNK_CHARS, len(t))
            if end < len(t):
                cut = t.rfind("\n", start + CHUNK_CHARS - CHUNK_OVERLAP, end)
                if cut == -1:
                    cut = t.rfind(". ", start + CHUNK_CHARS - CHUNK_OVERLAP, end)
                if cut > start:
                    end = cut
            piece = t[start:end].strip()
            if piece:
                out.append({"page": pno, "text": piece})
            if end >= len(t):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return out


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", (text or "").lower()) if len(w) > 1 and w not in STOPWORDS]


class BM25:
    """Tiny BM25 — no sklearn needed."""

    def __init__(self, docs: list[str]):
        self.n = len(docs)
        self.df: dict[str, int] = {}
        self.docs_tokens = []
        total = 0
        for d in docs:
            toks = tokenize(d)
            self.docs_tokens.append(toks)
            total += len(toks)
            for w in set(toks):
                self.df[w] = self.df.get(w, 0) + 1
        self.avgdl = (total / max(self.n, 1)) or 1.0
        self.k1, self.b = 1.5, 0.75

    def score(self, query: str) -> np.ndarray:
        if self.n == 0:
            return np.zeros(0)
        q = tokenize(query)
        scores = np.zeros(self.n)
        for w in set(q):
            df = self.df.get(w, 0)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, toks in enumerate(self.docs_tokens):
                tf = toks.count(w)
                if tf:
                    dl = len(toks) or 1
                    scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return scores
