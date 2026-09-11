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


def is_pdf_bytes(data: bytes) -> bool:
    """Cheap magic-byte check so we can reject non-PDF uploads with a clear error."""
    return bool(data) and data[:5] == b"%PDF-"


def _is_encrypted(data: bytes) -> bool:
    try:
        from pypdf import PdfReader
        return bool(getattr(PdfReader(io.BytesIO(data)), "is_encrypted", False))
    except Exception:
        return False


def extract_pdf_pages(data: bytes) -> tuple[list[str], int]:
    """Return (page_texts, n_ocr_pages). Uses pdfplumber; OCR fallback for scanned pages.

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
    ocr_pages = 0
    if PDF_OK:
        try:
            buf = io.BytesIO(data)
            with pdfplumber.open(buf) as pdf:
                for page in pdf.pages:
                    # One bad page must not kill the whole document.
                    try:
                        t = (page.extract_text() or "").strip()
                    except Exception:
                        t = ""
                    if len(t) < 40 and OCR_OK:
                        t2 = _ocr_page(page)
                        if t2 and len(t2) > len(t):
                            t = t2
                            ocr_pages += 1
                    pages.append(t)
            if pages:
                return pages, ocr_pages
        except Exception:
            pages = []
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


def _ocr_page(page) -> str:
    try:
        # Needs pypdfium2 (see requirements.txt) to rasterize the page.
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
