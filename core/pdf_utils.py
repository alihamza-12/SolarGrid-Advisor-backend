"""PDF text extraction (+ optional OCR) — ported from app.py lines 335-380."""
from __future__ import annotations

import io
import shutil

try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False

try:
    import pytesseract
    TESS_BIN = shutil.which("tesseract") is not None
    OCR_OK = bool(TESS_BIN)
except Exception:
    OCR_OK = False


def extract_pdf_pages(data: bytes) -> tuple[list[str], int]:
    """Return (page_texts, n_ocr_pages). Uses pdfplumber; OCR fallback for scanned pages."""
    pages: list[str] = []
    ocr_pages = 0
    if PDF_OK:
        try:
            buf = io.BytesIO(data)
            with pdfplumber.open(buf) as pdf:
                for page in pdf.pages:
                    t = (page.extract_text() or "").strip()
                    if len(t) < 40 and OCR_OK:
                        t2 = _ocr_page(page)
                        if t2 and len(t2) > len(t):
                            t = t2
                            ocr_pages += 1
                    pages.append(t)
            return pages, ocr_pages
        except Exception:
            pass
    # Fallback: pypdf (no OCR)
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        for p in reader.pages:
            pages.append((p.extract_text() or "").strip())
        return pages, 0
    except Exception:
        return [""], 0


def _ocr_page(page) -> str:
    try:
        img = page.to_image(resolution=170).original
        langs = None
        try:
            langs = " ".join(pytesseract.get_languages())
        except Exception:
            pass
        lang = "eng+urd" if langs and "urd" in langs else "eng"
        return (pytesseract.image_to_string(img, lang=lang) or "").strip()
    except Exception:
        return ""
