"""PDF text extraction (+ OCR for scanned pages) — ported from app.py lines 335-380.

Two-pass strategy:
  1. Fast text pass (pdfplumber) over all pages.
  2. OCR pass (Tesseract) for pages with almost no native text AND for pages
     whose text is a short duplicate of another page's text — that pattern
     means a template artifact (e-signature stamp, repeated header) sitting on
     a scanned image, not real content. OCR runs on several Tesseract workers
     in parallel, so even a 96-page scan finishes in a few minutes in ~1 GB RAM.

Env knobs: SGA_OCR_DPI (default 150), SGA_OCR_WORKERS (default 3),
SGA_OCR_LANG (default "" = eng+urd when Urdu data is installed, else eng).
"""
from __future__ import annotations

import io
import logging
import os
import re
import shutil

# pdfminer/pdfplumber are chatty: real-world PDFs (common in DISCO bills and
# NEPRA circulars) often carry malformed font metadata, which triggers
# per-page warnings like "Could not get FontBBox from font descriptor ...".
# Those are noise — extraction continues and our character-count checks
# validate the result — so keep only real errors from these loggers.
for _log_name in ("pdfminer", "pdfplumber"):
    logging.getLogger(_log_name).setLevel(logging.ERROR)

try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False

try:
    import pytesseract
    OCR_OK = shutil.which("tesseract") is not None
except Exception:
    pytesseract = None  # type: ignore[assignment]
    OCR_OK = False

try:
    import pypdfium2 as pdfium
    PDFIUM_OK = True
except Exception:
    PDFIUM_OK = False


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))  # type: ignore[arg-type]
    except Exception:
        return default


OCR_DPI = _int_env("SGA_OCR_DPI", 150)
OCR_WORKERS = max(1, _int_env("SGA_OCR_WORKERS", 3))
OCR_LANG_ENV = os.environ.get("SGA_OCR_LANG", "").strip()

_ocr_lang_cache: str | None = None


def _ocr_lang() -> str:
    """Tesseract language(s): explicit env override, else eng+urd if available."""
    global _ocr_lang_cache
    if OCR_LANG_ENV:
        return OCR_LANG_ENV
    if _ocr_lang_cache is None:
        try:
            have = list(pytesseract.get_languages())
        except Exception:
            have = []
        _ocr_lang_cache = "eng+urd" if "urd" in have else "eng"
    return _ocr_lang_cache


def is_pdf_bytes(data: bytes) -> bool:
    """Cheap magic-byte check so we can reject non-PDF uploads with a clear error."""
    return bool(data) and data[:5] == b"%PDF-"


def _is_encrypted(data: bytes) -> bool:
    try:
        from pypdf import PdfReader
        return bool(getattr(PdfReader(io.BytesIO(data)), "is_encrypted", False))
    except Exception:
        return False


def _norm_dup(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def _ocr_targets(pages: list[str]) -> list[int]:
    """Return 0-based indices of pages that need OCR.

    A page needs OCR when it has almost no native text (<40 chars) — the
    classic scanned page — or when its short text duplicates another page's
    text, which marks it as a template artifact (e-signature stamp, repeated
    header) on top of a scanned image rather than real content.
    """
    first_seen: dict[str, int] = {}
    dup_idx: set[int] = set()
    for i, t in enumerate(pages):
        k = _norm_dup(t)
        if len(k) < 20:
            continue
        if k in first_seen:
            dup_idx.add(i)
            dup_idx.add(first_seen[k])
        else:
            first_seen[k] = i
    return [i for i, t in enumerate(pages) if len(t) < 40 or (i in dup_idx and len(t) < 500)]


def _tess_image(img, lang: str) -> str:
    try:
        return (pytesseract.image_to_string(img, lang=lang) or "").strip()
    except Exception:
        return ""


def _ocr_pages_fast(data: bytes, indices: list[int]) -> dict[int, str]:
    """Raster + OCR the given 0-based page indices with parallel workers.

    Pages are rasterized one at a time (cheap) while a small pool of Tesseract
    subprocesses (the slow part) runs concurrently; a semaphore bounds how many
    page images wait in memory so RAM stays flat regardless of page count.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lang = _ocr_lang()
    scale = max(1.0, OCR_DPI / 72.0)
    out: dict[int, str] = {}
    sem = threading.Semaphore(max(1, OCR_WORKERS * 2))

    def _run(img):
        try:
            return _tess_image(img, lang)
        finally:
            sem.release()

    try:
        doc = pdfium.PdfDocument(data)
    except Exception:
        return {}
    try:
        with ThreadPoolExecutor(max_workers=OCR_WORKERS, thread_name_prefix="ocr") as pool:
            futs = {}
            for idx in indices:
                try:
                    page = doc[idx]
                    bitmap = page.render(scale=scale)
                    img = bitmap.to_pil().convert("L")
                except Exception:
                    continue
                sem.acquire()
                try:
                    futs[pool.submit(_run, img)] = idx
                except Exception:
                    sem.release()
            for fut in as_completed(futs):
                try:
                    out[futs[fut]] = fut.result()
                except Exception:
                    pass
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out


def _ocr_pages_legacy(data: bytes, indices: list[int]) -> dict[int, str]:
    """Sequential OCR via pdfplumber rendering (used only if pypdfium2 is missing)."""
    out: dict[int, str] = {}
    lang = _ocr_lang()
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i in indices:
                try:
                    img = pdf.pages[i].to_image(resolution=170).original.convert("L")
                    out[i] = _tess_image(img, lang)
                except Exception:
                    pass
    except Exception:
        pass
    return out


def extract_pdf_pages(data: bytes) -> tuple[list[str], int]:
    """Return (page_texts, n_ocr_pages). Text pass first, OCR only for scanned pages.

    Raises:
        ValueError: if the bytes are not a PDF or the PDF is password-protected.
    """
    if not is_pdf_bytes(data):
        raise ValueError("not-a-pdf")
    if _is_encrypted(data):
        # Try the common empty-password case before giving up.
        try:
            from pypdf import PdfReader
            r = PdfReader(io.BytesIO(data))
            if r.decrypt("") == 0:
                raise ValueError("encrypted")
        except ValueError:
            raise
        except Exception:
            raise ValueError("encrypted")

    pages: list[str] = []
    if PDF_OK:
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    # One bad page must not kill the whole document.
                    try:
                        pages.append((page.extract_text() or "").strip())
                    except Exception:
                        pages.append("")
        except Exception:
            pages = []
    if not pages:
        # Fallback: pypdf (no OCR)
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            for p in reader.pages:
                try:
                    pages.append((p.extract_text() or "").strip())
                except Exception:
                    pages.append("")
            return pages, 0
        except Exception:
            return [""], 0

    # OCR pass for scanned pages (near-empty or stamp-duplicate text).
    ocr_n = 0
    if OCR_OK:
        need = _ocr_targets(pages)
        if need:
            try:
                results = _ocr_pages_fast(data, need) if PDFIUM_OK else _ocr_pages_legacy(data, need)
            except Exception:
                results = {}
            for i, t2 in results.items():
                # Only trust OCR when it found substantially more than the
                # native text — keeps clean native text and filters OCR noise.
                if t2 and len(t2) > max(len(pages[i]), 40) * 1.5:
                    pages[i] = t2
                    ocr_n += 1
    return pages, ocr_n
