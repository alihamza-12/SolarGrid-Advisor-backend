"""
SolarGrid Advisor — FastAPI backend
Ported RAG + calculators from the original Streamlit app; see core/ for pure logic.
"""
from __future__ import annotations

import datetime as dt
import gc
import hashlib
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.bill import BILL_PATTERNS, LABEL_MAP, absence_zeros, crosscheck_energy, extract_bill, llm_extract_bill, meter_units, normalize_bill_text, sanitize_llm_bill, strong_periods
from core.calculators import (
    backup_hours, heuristic_plan, inverter_size, llm_plan, payback,
    savings_calc, solar_sizing,
)
from core.config import (
    CHUNKS_FILE, DISCOS, DOC_TYPES, EMB_MODELS, LLM_PROVIDERS, META_FILE, RATES_FILE,
    EXTRACT_CACHE, UPLOADS_DIR, load_json, save_json,
)
from core.index_store import IndexStore
from core.llm import effective_key, get_llm, llm_ok
from core.pdf_utils import BILL_OCR_DPI, BILL_OCR_LANG, BILL_OCR_PSM, OCR_OK, extract_pdf_pages, is_pdf_bytes
from core.rag import build_memory, rag_answer
from core.rates import rates as get_rates
from core.utils import auto_detect_dates, norm_title, parse_date, pkrs

app = FastAPI(title="SolarGrid Advisor API")

# ---------------------------------------------------------------------------
# Upload limits & CORS (overridable via env for deploys behind other hosts)
# ---------------------------------------------------------------------------
MAX_PDF_MB = float(os.environ.get("SGA_MAX_PDF_MB", "50"))
MAX_PDF_BYTES = int(MAX_PDF_MB * 1024 * 1024)

# Zero-PDF-footprint mode (default): an uploaded PDF is read once, indexed as
# tiny text chunks, and never stored on the server — no heavy files stay on
# disk or in RAM. Set SGA_KEEP_PDFS=1 to archive originals under data/uploads/.
KEEP_PDFS = os.environ.get("SGA_KEEP_PDFS", "0") == "1"

_cors_env = [o.strip() for o in os.environ.get("SGA_CORS_ORIGINS", "").split(",") if o.strip()]
_allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173", *{o for o in _cors_env}]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = IndexStore()


def _purge_server_footprint() -> None:
    """Startup cleanup so nothing heavy lingers on the server.

    - data/uploads/*.pdf are never read back by any endpoint (indexing and
      chat use the text chunks only), so in default mode they are dead weight.
    - data/cache/huggingface may hold a partial embedding-model download left
      by the old full build; the light build never downloads models, so in
      light mode that space can be reclaimed safely.
    """
    try:
        if not KEEP_PDFS and UPLOADS_DIR.exists():
            n = 0
            for p in UPLOADS_DIR.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                        n += 1
                except Exception:
                    pass
            if n:
                print(f"[startup] removed {n} archived PDF(s) from uploads/ (zero-footprint mode)")
        import importlib.util
        full_build = importlib.util.find_spec("sentence_transformers") is not None
        if not full_build:
            hf_cache = UPLOADS_DIR.parent / "cache" / "huggingface"
            if hf_cache.exists():
                import shutil
                shutil.rmtree(hf_cache, ignore_errors=True)
                print("[startup] cleared stale partial model download (light build needs no models)")
    except Exception as e:
        print(f"[startup] footprint cleanup skipped: {e}")


_purge_server_footprint()


async def _read_upload_pdf(file: UploadFile) -> bytes:
    """Read + validate an uploaded PDF. Raises HTTPException with a clear message."""
    data = await file.read()
    name = file.filename or "upload.pdf"
    if not data:
        raise HTTPException(400, f"“{name}” is empty (0 bytes). Please choose a valid PDF file.")
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(
            413,
            f"“{name}” is {len(data) / 1024 / 1024:.1f} MB — the limit is {MAX_PDF_MB:g} MB. "
            "Split the PDF or compress it and try again.",
        )
    if not is_pdf_bytes(data):
        raise HTTPException(
            400,
            f"“{name}” doesn't look like a PDF file (only .pdf uploads are supported).",
        )
    return data


def _extract_or_400(data: bytes, filename: str, ocr_kwargs: dict | None = None) -> tuple[list[str], int]:
    """Run PDF extraction, mapping failures to actionable 400 errors."""
    try:
        return extract_pdf_pages(data, **(ocr_kwargs or {}))
    except ValueError as e:
        if str(e) == "encrypted":
            raise HTTPException(
                400,
                f"“{filename}” is password-protected. Remove the password and upload again.",
            )
        raise HTTPException(400, f"“{filename}” could not be read as a PDF — the file may be corrupt.")
    except Exception as e:
        raise HTTPException(400, f"Could not read “{filename}”: {str(e)[:200]}")


EXTRACT_CACHE_MAX = 30  # max cached extractions kept on disk (each is a few 100 KB)


def _extract_cached(data: bytes, filename: str, cache_variant: str = "",
                    ocr_kwargs: dict | None = None) -> tuple[list[str], int]:
    """extract_pdf_pages + content-addressed disk cache.

    The UI calls preview first and upload right after with the same file; for a
    96-page scan, OCR takes minutes — without the cache that work would run twice.
    cache_variant namespaces callers that extract with different OCR settings
    (bills use high-DPI English-only OCR) so they never share cache entries.
    """
    key = hashlib.sha1(data).hexdigest()
    cache_file = EXTRACT_CACHE / f"{key}{cache_variant}.json"
    try:
        if cache_file.exists():
            cached = load_json(cache_file, None)
            if isinstance(cached, dict) and isinstance(cached.get("pages"), list):
                return cached["pages"], int(cached.get("ocr_pages", 0))
    except Exception:
        pass
    pages, ocr_n = _extract_or_400(data, filename, ocr_kwargs)
    try:
        save_json(cache_file, {"pages": pages, "ocr_pages": ocr_n})
        files = sorted(EXTRACT_CACHE.glob("*.json"), key=lambda f: f.stat().st_mtime)
        for old in files[:-EXTRACT_CACHE_MAX]:
            try:
                old.unlink()
            except Exception:
                pass
    except Exception:
        pass
    return pages, ocr_n


def _repeated_pages(pages: list[str]) -> set[int]:
    """Indices of pages whose full text duplicates another page's text.

    A repeated full-page text means a template artifact (e-signature stamp,
    copied header) rather than real content — the signature of a scanned PDF
    whose native text layer carries nothing useful.
    """
    first_seen: dict[str, int] = {}
    dups: set[int] = set()
    for i, t in enumerate(pages):
        k = re.sub(r"\s+", " ", (t or "").strip().lower())
        if len(k) < 20:
            continue
        if k in first_seen:
            dups.add(i)
            dups.add(first_seen[k])
        else:
            first_seen[k] = i
    return dups


# ---------------------------------------------------------------------------
# Health & meta
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "n_docs": len(load_json(META_FILE, [])),
        "n_chunks": len(store.chunks),
        "ocr_available": OCR_OK,
        "max_pdf_mb": MAX_PDF_MB,
        "keep_pdfs": KEEP_PDFS,
        "discos": DISCOS,
        "doc_types": DOC_TYPES,
        "emb_models": list(EMB_MODELS.keys()),
        "llm_providers": {k: v for k, v in LLM_PROVIDERS.items()},
    }


# ---------------------------------------------------------------------------
# LLM connection test
# ---------------------------------------------------------------------------
class LLMConfig(BaseModel):
    provider: str = "Offline (no LLM — retrieval only)"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2


@app.post("/api/llm/test")
def test_llm(cfg: LLMConfig):
    pack = get_llm(cfg.provider, cfg.base_url, cfg.api_key, cfg.model, cfg.temperature)
    if pack is None:
        return {"ok": False, "message": "Could not build client — check the base URL."}
    ok = llm_ok(pack)
    return {"ok": bool(ok), "message": "Connected — model responded." if ok else "No response — check key/URL/model."}


def _llm_pack(cfg: LLMConfig):
    return get_llm(cfg.provider, cfg.base_url, cfg.api_key, cfg.model, cfg.temperature)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@app.post("/api/documents/preview")
async def preview_document(file: UploadFile = File(...)):
    data = await _read_upload_pdf(file)
    pages, ocr_n = _extract_cached(data, file.filename or "upload.pdf")
    total_chars = sum(len(p) for p in pages)
    dup_idx = _repeated_pages(pages)
    looks_scanned = len(dup_idx) >= 3 and len(dup_idx) >= len(pages) // 2
    usable = total_chars > 200 and not (looks_scanned and ocr_n == 0)
    dates = auto_detect_dates("\n".join(pages)) if total_chars > 200 else {"issue": "", "effective": ""}
    size_kb = round(len(data) / 1024)
    warning = None
    if looks_scanned and ocr_n == 0:
        if OCR_OK:
            warning = (f"Only a repeated stamp/header was readable on {len(dup_idx)} pages and OCR found nothing more — try a clearer scan.")
        else:
            warning = (f"This looks like a scanned PDF (only a repeated stamp/header repeats on {len(dup_idx)} pages) and OCR is not available on the server, so its real content cannot be read.")
    del data
    gc.collect()
    return {
        "filename": file.filename,
        "size_kb": size_kb,
        "n_pages": len(pages),
        "total_chars": total_chars,
        "ocr_pages": ocr_n,
        "ocr_available": OCR_OK,
        "usable": usable,
        "repeated_pages": len(dup_idx),
        "warning": warning,
        "detected_issue_date": dates["issue"],
        "detected_effective_date": dates["effective"],
    }


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(""),
    disco: str = Form("Other"),
    doc_type: str = Form("Circular / Other"),
    status: str = Form("official"),
    issue_date: str = Form(""),
    effective_date: str = Form(""),
    notes: str = Form(""),
):
    data = await _read_upload_pdf(file)
    filename = file.filename or "upload.pdf"
    pages, ocr_n = _extract_cached(data, filename)
    dup_idx = _repeated_pages(pages)
    looks_scanned = len(dup_idx) >= 3 and len(dup_idx) >= len(pages) // 2
    if sum(len(p) for p in pages) <= 200 or (looks_scanned and ocr_n == 0):
        if looks_scanned and ocr_n == 0:
            if OCR_OK:
                detail = (f"This PDF looks scanned — only a repeated stamp/header repeats on {len(dup_idx)} pages "
                          "and OCR could not read the pages. Try a clearer scan.")
            else:
                detail = (f"This PDF looks scanned — only a repeated stamp/header repeats on {len(dup_idx)} pages. "
                          "OCR is not available on the server, so its content cannot be read.")
        elif OCR_OK:
            detail = ("Could not extract usable text from this PDF — it may be a scanned image. "
                      "OCR ran but found too little text; try a clearer scan.")
        else:
            detail = ("Could not extract usable text from this PDF. If it's a scanned document, "
                      "install tesseract-ocr on the server and retry.")
        raise HTTPException(400, detail)

    doc_id = hashlib.sha1(data).hexdigest()[:12]
    docs = load_json(META_FILE, [])
    existing = next((d for d in docs if d["id"] == doc_id), None)

    clean_title = title.strip() or Path(filename).stem.replace("_", " ").replace("-", " ").title()

    if existing is not None:
        # Idempotent re-upload: same file → refresh metadata instead of duplicating.
        existing.update({
            "title": clean_title,
            "disco": disco or existing.get("disco", "Other"),
            "doc_type": doc_type or existing.get("doc_type", "Circular / Other"),
            "status": status or existing.get("status", "official"),
            "issue_date": issue_date, "effective_date": effective_date,
            "notes": notes.strip(),
        })
        save_json(META_FILE, docs)
        for c in store.chunks:
            if c["doc_id"] == doc_id:
                c.update({
                    "doc_title": existing["title"], "disco": existing["disco"],
                    "status": existing["status"], "issue_date": existing["issue_date"],
                    "effective_date": existing["effective_date"],
                })
        save_json(CHUNKS_FILE, store.chunks)
        nch = sum(1 for c in store.chunks if c["doc_id"] == doc_id)
        del data
        gc.collect()
        return {"doc": existing, "n_chunks": nch, "duplicate": True}

    # Zero-footprint default: the PDF bytes were indexed above and are dropped —
    # only tiny text chunks stay on the server. Archive the original only if asked.
    if KEEP_PDFS:
        try:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(ch for ch in Path(filename).name if ch.isalnum() or ch in "._-") or "upload.pdf"
            (UPLOADS_DIR / f"{doc_id}_{safe_name}").write_bytes(data)
        except Exception:
            pass  # indexing must not fail just because archival failed

    doc = {
        "id": doc_id,
        "filename": filename,
        "title": clean_title,
        "disco": disco, "doc_type": doc_type, "status": status,
        "issue_date": issue_date, "effective_date": effective_date,
        "notes": notes.strip(),
        "added": dt.datetime.now().isoformat(timespec="seconds"),
        "ocr_pages": ocr_n,
        "pages": len(pages),
    }
    try:
        nch = store.add_document(doc, pages)
    except Exception as e:
        raise HTTPException(500, f"Indexing failed: {str(e)[:300]}")
    docs.append(doc)
    save_json(META_FILE, docs)
    del data
    gc.collect()
    return {"doc": doc, "n_chunks": nch, "duplicate": False}


@app.get("/api/documents")
def list_documents():
    docs = load_json(META_FILE, [])
    latest_map = store.latest_effective_map()
    out = []
    for d in docs:
        superseded = any(
            norm_title(x["title"]) == norm_title(d["title"]) and x["id"] != d["id"]
            and (parse_date(x.get("effective_date", "")) or dt.date(1900, 1, 1)) > (parse_date(d.get("effective_date", "")) or dt.date(1900, 1, 1))
            for x in docs
        )
        out.append({**d, "superseded": superseded})
    return {
        "documents": out,
        "n_chunks": len(store.chunks),
        "manifest": store.manifest,
        "latest_effective": {k: v.isoformat() for k, v in latest_map.items()},
        "ocr_available": OCR_OK,
    }


class DocMetaUpdate(BaseModel):
    issue_date: Optional[str] = None
    effective_date: Optional[str] = None
    disco: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


@app.patch("/api/documents/{doc_id}")
def update_document(doc_id: str, upd: DocMetaUpdate):
    docs = load_json(META_FILE, [])
    found = None
    for d in docs:
        if d["id"] == doc_id:
            for field in ("issue_date", "effective_date", "disco", "status", "notes"):
                val = getattr(upd, field)
                if val is not None:
                    d[field] = val
            found = d
    if not found:
        raise HTTPException(404, "Document not found")
    save_json(META_FILE, docs)
    for c in store.chunks:
        if c["doc_id"] == doc_id:
            for field in ("disco", "status", "issue_date", "effective_date"):
                val = getattr(upd, field)
                if val is not None:
                    c[field] = val
    save_json(CHUNKS_FILE, store.chunks)
    return {"doc": found}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    docs = load_json(META_FILE, [])
    match = next((d for d in docs if d["id"] == doc_id), None)
    if not match:
        raise HTTPException(404, "Document not found")
    store.remove_document(doc_id)
    docs = [d for d in docs if d["id"] != doc_id]
    save_json(META_FILE, docs)
    for p in UPLOADS_DIR.glob(f"{doc_id}_*"):
        try:
            p.unlink()
        except Exception:
            pass
    # Evict the cached extraction so a re-upload re-reads the file with current code.
    for p in EXTRACT_CACHE.glob(f"{doc_id}*.json"):
        try:
            p.unlink()
        except Exception:
            pass
    return {"deleted": doc_id}


@app.post("/api/documents/rebuild")
def rebuild_index():
    try:
        store.rebuild_all()
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True, "manifest": store.manifest}


# ---------------------------------------------------------------------------
# Chat (RAG)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    question: str
    discos: list[str] = []
    statuses: list[str] = ["official"]
    min_effective: str = ""
    latest_only: bool = True
    top_k: int = 5
    vector: bool = True
    bm25: bool = True
    rewrite: bool = False
    rerank: bool = False
    memory_enabled: bool = True
    history: list[ChatMessage] = []
    llm: LLMConfig = LLMConfig()


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(400, "Question is empty.")
    opts = {
        "discos": req.discos, "statuses": req.statuses,
        "min_effective": req.min_effective, "latest_only": req.latest_only,
        "top_k": max(1, min(req.top_k, 20)), "vector": req.vector, "bm25": req.bm25,
        "rewrite": req.rewrite, "rerank": req.rerank, "memory_enabled": req.memory_enabled,
        "memory": build_memory([m.model_dump() for m in req.history[-7:-1]]) if req.memory_enabled else "",
        "docs": load_json(META_FILE, []),
    }
    pack = _llm_pack(req.llm)
    try:
        res = rag_answer(req.question, store, pack, opts)
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)[:300]}")
    score, label = res["confidence"]
    return {
        "answer": res["answer"],
        "confidence": {"score": score, "label": label},
        "verified": res["verified"],
        "rewritten": res["rewritten"],
        "sources": [
            {
                "doc_title": c["doc_title"], "disco": c["disco"], "page": c["page"],
                "effective_date": c.get("effective_date", ""), "text": c["text"][:600],
                "score": c.get("score", 0),
            }
            for c in res["context_chunks"][:8]
        ],
    }


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------
@app.get("/api/rates")
def rates_endpoint():
    return get_rates()


class RateUpdate(BaseModel):
    peak: float
    offpeak: float
    fixed: float
    buyback: float


@app.post("/api/rates/{disco}")
def update_rate(disco: str, upd: RateUpdate):
    if disco not in DISCOS:
        raise HTTPException(400, "Unknown DISCO")
    r = get_rates()
    r[disco] = {"peak": upd.peak, "offpeak": upd.offpeak, "fixed": upd.fixed,
                "buyback": upd.buyback, "note": "Updated in-app"}
    save_json(RATES_FILE, r)
    return {"disco": disco, "rate": r[disco]}


# ---------------------------------------------------------------------------
# Savings dashboard
# ---------------------------------------------------------------------------
class DashboardRequest(BaseModel):
    disco: str
    units: float = 640
    peak_share: float = 0.25
    shift: float = 0.4
    solar_kwp: float = 5.0
    sun_hours: float = 4.5
    self_use_share: float = 0.7


@app.post("/api/dashboard/calc")
def dashboard_calc(req: DashboardRequest):
    r = get_rates().get(req.disco)
    if not r:
        raise HTTPException(400, "Unknown DISCO")
    rr = {"peak": float(r["peak"]), "offpeak": float(r["offpeak"]),
          "fixed": float(r["fixed"]), "buyback": float(r["buyback"])}
    res = savings_calc(req.units, req.peak_share, req.shift, req.solar_kwp, req.sun_hours, rr, req.self_use_share)
    return {"rates": rr, "result": res}


# ---------------------------------------------------------------------------
# Solar toolkit
# ---------------------------------------------------------------------------
class SolarCalcRequest(BaseModel):
    disco: str
    daily_kwh: float = 20.0
    sun_hours: float = 4.5
    peak_load_kw: float = 6.0
    battery_kwh: float = 10.0
    cost_per_kwp: float = 150000
    save_share: float = 0.6
    export_share: float = 0.2


@app.post("/api/solar/calc")
def solar_calc(req: SolarCalcRequest):
    r = get_rates().get(req.disco)
    if not r:
        raise HTTPException(400, "Unknown DISCO")
    rr = {"peak": float(r["peak"]), "offpeak": float(r["offpeak"]),
          "fixed": float(r["fixed"]), "buyback": float(r["buyback"])}
    kwp = solar_sizing(req.daily_kwh, req.sun_hours)
    inv = inverter_size(kwp, req.peak_load_kw)
    b_hours = backup_hours(req.battery_kwh, req.peak_load_kw) if req.battery_kwh else 0.0
    gen = kwp * req.sun_hours * 30 * 0.8

    solar_month = gen * req.save_share
    self_used = solar_month * (1 - req.export_share)
    exported = solar_month * req.export_share
    monthly_saving = self_used * rr["offpeak"] * 0.9 + exported * rr["buyback"]
    cost = kwp * req.cost_per_kwp
    yrs = payback(cost, monthly_saving)

    return {
        "rates": rr,
        "kwp": kwp, "inverter_kw": inv, "backup_hours": b_hours, "monthly_generation_kwh": round(gen),
        "system_cost": cost, "monthly_saving": monthly_saving, "payback_years": yrs,
        "self_use_value": self_used * rr["offpeak"] * 0.9,
        "export_value": exported * rr["buyback"],
    }


class SolarPlanRequest(BaseModel):
    disco: str
    monthly_units: float = 640
    solar_kwp: float = 5.0
    battery_kwh: float = 10.0
    appliances: list[str] = []
    peak_appliances: list[str] = []
    avg_monthly_bill: float = 18000
    use_llm: bool = True
    llm: LLMConfig = LLMConfig()


@app.post("/api/solar/plan")
def solar_plan(req: SolarPlanRequest):
    r = get_rates().get(req.disco)
    if not r:
        raise HTTPException(400, "Unknown DISCO")
    rr = {"peak": float(r["peak"]), "offpeak": float(r["offpeak"]),
          "fixed": float(r["fixed"]), "buyback": float(r["buyback"])}
    profile = {
        "disco": req.disco, "monthly_units": req.monthly_units, "solar_kwp": req.solar_kwp,
        "battery_kwh": req.battery_kwh, "appliances": req.appliances,
        "peak_appliances": req.peak_appliances, "avg_monthly_bill_pkrs": req.avg_monthly_bill,
        "peak_rate": rr["peak"], "offpeak_rate": rr["offpeak"], "buyback": rr["buyback"],
    }
    rows = heuristic_plan(profile, rr)
    llm_text = None
    if req.use_llm:
        pack = _llm_pack(req.llm)
        llm_text = llm_plan(pack, profile, rr)

    estimate = None
    if req.peak_appliances:
        estimate = req.monthly_units * 0.2 * 0.5 * (rr["peak"] - rr["offpeak"])

    return {"rows": rows, "llm_plan": llm_text, "quick_estimate": estimate}


# ---------------------------------------------------------------------------
# Bill analyzer
# ---------------------------------------------------------------------------
@app.post("/api/bills/extract")
async def bills_extract(
    file: UploadFile = File(...),
    use_llm: bool = Form(True),
    provider: str = Form("Offline (no LLM — retrieval only)"),
    base_url: str = Form(""),
    api_key: str = Form(""),
    model: str = Form(""),
):
    data = await _read_upload_pdf(file)
    filename = file.filename or "bill.pdf"
    # Bills get their own high-DPI English-only OCR pass (cached separately),
    # which reads meter/charge digits far more reliably than the chat default.
    pages, _ = _extract_cached(
        data, filename, cache_variant="-bill300",
        ocr_kwargs={"ocr_dpi": BILL_OCR_DPI, "ocr_lang": BILL_OCR_LANG, "ocr_psm": BILL_OCR_PSM},
    )
    text = "\n".join(pages)
    clean = normalize_bill_text(text)
    if len(clean) < 100:
        if OCR_OK:
            detail = "No text found in this bill — OCR ran but found too little text. Try a clearer scan."
        else:
            detail = "No text found — this bill appears to be scanned. OCR requires tesseract."
        raise HTTPException(400, detail)

    # All 11 labels, always in order; missing stays None ("not on this bill").
    found: dict = {label: None for label in BILL_PATTERNS}
    llm_used = False
    llm_note = None
    if use_llm:
        pack = get_llm(provider, base_url, api_key, model)
        if pack is None:
            llm_note = "AI extraction skipped: set the provider and paste your API key in Chat, then re-extract."
        else:
            llm_f = llm_extract_bill(pack, clean)
            if not llm_f:
                if (api_key or "").strip():
                    llm_note = "AI extraction failed (check the API key in Chat) — showing regex results only."
                elif effective_key(provider, ""):
                    llm_note = "AI extraction failed (the server Groq key looks invalid) — showing regex results only."
                else:
                    llm_note = "AI extraction skipped: paste your API key in Chat (or set GROQ_API_KEY on the server), then re-extract."
            else:
                llm_used = True
                for k, v in sanitize_llm_bill(llm_f).items():
                    if v is not None and k in LABEL_MAP:
                        found[LABEL_MAP[k]] = v
    rx = extract_bill(clean)
    for k, v in rx.items():
        if found.get(k) is None:
            found[k] = v
    mu = meter_units(clean)
    if mu is not None:
        found["Total units"] = mu
    # Cross-checks: the bill's own arithmetic / strong labels beat a mis-copied
    # figure (OCR-garbled or LLM-misattributed) whenever they disagree.
    found["Energy charge (Rs)"] = crosscheck_energy(found.get("Energy charge (Rs)"), clean)
    rx_total = rx.get("Total amount (Rs)")
    if (rx_total is not None and found.get("Total amount (Rs)") is not None
            and rx_total != found["Total amount (Rs)"]):
        found["Total amount (Rs)"] = rx_total
    # Explicit BILL MONTH / billing-period labels overrule the LLM (weak models
    # often substitute reading/issue/due dates for the period).
    # Zero-fill: export/fixed are 0 (not missing) when the bill shows no such
    # concept at all. Only fills gaps — never overrides extracted values.
    for k, v in absence_zeros(clean).items():
        if found.get(k) is None:
            found[k] = v
    sf, st = strong_periods(clean)
    if sf is not None:
        found["Billing period (from)"] = sf
    if st is not None:
        found["Billing period (to)"] = st

    # quick insights
    insights = []
    tu = found.get("Total units")
    pk = found.get("Peak units")
    tot = found.get("Total amount (Rs)")
    exp = found.get("Export units (net metering)")
    if isinstance(tu, float) and tu:
        if isinstance(pk, float):
            share = pk / tu
            insights.append(f"{round(share * 100)}% of your units are used in the peak window (6–10 PM). Shifting half of them to off-peak saves roughly {pkrs(pk * 0.5 * 12)}/month (at ~Rs 13 spread).")
        else:
            insights.append("Peak units not detected — if your DISCO bills TOU, check the “peak”/“high rate” line.")
    if isinstance(exp, float) and exp:
        r = get_rates().get("LESCO", {"buyback": 0.85})
        insights.append(f"You exported {exp:,.0f} units — at a typical buyback of {pkrs(r['buyback'])}/unit that's only {pkrs(exp * r['buyback'])}/month. Self-consuming those units is worth ~3-4x more.")
    if isinstance(tot, float) and isinstance(tu, float) and tu:
        insights.append(f"Your effective rate is {pkrs(tot / tu)}/unit all-in — compare it against your DISCO tariff to spot anomalies.")

    bill_id = hashlib.sha1(data).hexdigest()[:10]
    del data
    gc.collect()
    return {
        "bill_id": bill_id, "filename": filename, "fields": found,
        "insights": insights, "raw_text_preview": clean[:3500],
        "llm_used": llm_used, "llm_note": llm_note,
        "total_fields": len(BILL_PATTERNS),
    }


class SaveBillRequest(BaseModel):
    bill_id: str
    filename: str
    fields: dict


@app.post("/api/bills/save")
def bills_save(req: SaveBillRequest):
    from core.config import BILLS_FILE
    bills = load_json(BILLS_FILE, [])
    bills.append({
        "id": req.bill_id, "name": req.filename, "fields": req.fields,
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
    })
    save_json(BILLS_FILE, bills[-60:])
    return {"ok": True}


@app.get("/api/bills/history")
def bills_history():
    from core.config import BILLS_FILE
    return {"bills": load_json(BILLS_FILE, [])}


@app.delete("/api/bills/history")
def bills_clear():
    from core.config import BILLS_FILE
    save_json(BILLS_FILE, [])
    return {"ok": True}
