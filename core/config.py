"""
Paths, constants and shared config for SolarGrid Advisor backend.
Ported from the original Streamlit app.py (lines 90-201) — same values, no Streamlit deps.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths & data root
# ---------------------------------------------------------------------------
def _data_root() -> Path:
    env = os.environ.get("SGA_DATA_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path(__file__).resolve().parent.parent / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


DATA = _data_root()
INDEX_DIR = DATA / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = DATA / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
META_FILE = DATA / "metadata.json"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
VECTORS_FILE = INDEX_DIR / "vectors.npz"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
RATES_FILE = DATA / "rates.json"
BILLS_FILE = DATA / "bills.json"
HF_CACHE = DATA / "cache" / "huggingface"
EXTRACT_CACHE = DATA / "cache" / "extract"
EXTRACT_CACHE.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
TOP_K_DEFAULT = 5

DISCOS = ["LESCO", "IESCO", "HESCO", "NESCO", "MEPCO", "GEPCO", "QESCO", "K-Electric", "Other"]

DOC_TYPES = ["NEPRA Notification", "DISCO Tariff", "Net-Metering Circular", "Circular / Other", "Bill"]

EMB_MODELS = {
    "Multilingual MiniLM — Urdu + English (recommended)": "paraphrase-multilingual-MiniLM-L12-v2",
    "MiniLM L6 — English only, fastest": "all-MiniLM-L6-v2",
}
DEFAULT_EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

LLM_PROVIDERS = {
    "Groq (free, recommended)": {
        "base": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b",
        "hint": "Free key at console.groq.com — paste it as the API key.",
    },
    "gpt-oss-120b (Modal — free)": {
        "base": "", "model": "openai/gpt-oss-120b",
        "hint": "Paste your Modal endpoint URL, e.g. https://<app>-<user>.modal.live",
    },
    "Grok (xAI API)": {
        "base": "https://api.x.ai/v1", "model": "grok-4-1-fast",
        "hint": "Get a key at console.x.ai — cheapest/fastest Grok model by default.",
    },
    "Custom / Ollama (any OpenAI-compatible)": {
        "base": "http://localhost:11434/v1", "model": "gpt-oss:20b",
        "hint": "Works with local Ollama too (ollama run gpt-oss:20b).",
    },
    "Offline (no LLM — retrieval only)": {"base": "", "model": ""},
}

# Sample editable defaults — ALWAYS verify against the latest circular
SEED_RATES = {
    d: {"peak": 38.5, "offpeak": 25.5, "fixed": 300.0, "buyback": 0.85}
    for d in DISCOS
}

STOPWORDS = set("""
a an the is are was were be been being to of in on for with and or not no do does
did what which who whom when where why how can could should would will shall may
might this that these those it its as at by from about please tell me tell us my
me i you your our their he she they we us ka hai kya kaise kab kahan kyun kya se
mein par aur ya ham aap tum us wo ise use un unka kya
""".split())

BUYBACK_RE = re.compile(
    r"(buy[\s-]?back|feed[\s-]?in|purchas\w+|credit)[^\n]{0,140}?Rs\.?\s*([0-9]{1,4}(?:[.,][0-9]{1,3}){0,3}(?:\.[0-9]{1,2})?)",
    re.I,
)
