# SolarGrid Advisor — Backend API

FastAPI backend for **SolarGrid Advisor** (سولر گرڈ ایڈوائزر), an expert energy assistant for Pakistan.
It answers questions about DISCO tariffs, peak/off-peak hours, net-metering and buyback rates —
grounded in NEPRA notifications and DISCO circulars the user uploads — and also analyzes
electricity bills and runs solar savings calculators.

## Features

- **RAG chat** over uploaded PDFs with cited sources, confidence score and filters
  (DISCO, status, effective date, latest-only). Asking *"which PDFs do you have?"*
  lists the indexed documents instead of searching.
- **PDF ingestion**: native-text extraction plus **OCR for scanned PDFs** (Tesseract,
  English + Urdu, parallel workers). Detects stamp-only scans and rejects them with a
  clear message instead of indexing garbage.
- **Document management**: upload / preview / list / update metadata / delete / rebuild index.
- **Bill analyzer**: extracts 11 fields from DISCO bills (units, peak/off-peak, charges,
  taxes, totals…) with regex + optional LLM, plus savings insights.
- **Rates store + calculators**: editable DISCO rates, savings dashboard and solar
  sizing/payback toolkit endpoints.
- **Light build**: idles at ~65 MB RAM (BM25 keyword retrieval, no torch) — fits a 1 GB host.
  Optional full build adds vector search + cross-encoder rerank.

## Tech stack

Python 3.12 · FastAPI · Uvicorn · pdfplumber / pypdf / pypdfium2 · Tesseract OCR ·
OpenAI-compatible LLM client (Groq default) · NumPy · built-in BM25 ·
optional: `sentence-transformers`, `faiss-cpu`

## Project structure

```
backend/
├── main.py            # FastAPI app: routes, upload limits, startup cleanup
├── requirements.txt   # Light build (no torch). See "Full build" below.
├── nixpacks.toml      # Railway/Nixpacks: installs tesseract-ocr + eng/urd data
├── railway.json       # Railway: NIXPACKS builder, start command, healthcheck
├── Procfile           # Start command (Railway/Heroku-style)
├── core/
│   ├── config.py      # Paths, constants, LLM provider presets, rates seed data
│   ├── pdf_utils.py   # Text extraction + parallel OCR + scan detection
│   ├── chunking.py    # Overlapping chunks + tiny built-in BM25
│   ├── index_store.py # Chunk store, filters, hybrid (vector + BM25) search
│   ├── rag.py         # RAG pipeline: rewrite → retrieve → answer → verify cites
│   ├── llm.py         # OpenAI-compatible client (chat completions)
│   ├── bill.py        # Bill text normalization + 11-field extraction
│   ├── rates.py       # DISCO rate table access
│   ├── calculators.py # Savings / sizing / payback math
│   └── utils.py       # Dates, titles, formatting helpers
└── data/              # Created at runtime (or SGA_DATA_DIR, see below)
```

Runtime data layout (`data/` by default):

```
data/
├── metadata.json        # one record per indexed document
├── index/chunks.json    # text chunks used for retrieval
├── index/vectors.npz    # embedding vectors (full build only)
├── index/manifest.json  # embedding model info (full build only)
├── rates.json           # editable DISCO rates
├── bills.json           # saved bill history (last 60)
├── uploads/             # original PDFs — only if SGA_KEEP_PDFS=1
└── cache/extract-v2/    # content-addressed extraction cache (max 30)
```

## Quickstart (local)

Requirements: Python 3.12. Tesseract is optional locally (only needed for scanned PDFs).

```bash
cd backend
pip install -r requirements.txt
# optional, for scanned-PDF OCR (Ubuntu/Debian):
# sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-urd
uvicorn main:app --host 0.0.0.0 --port 8000
```

- API root: `http://localhost:8000/api/…`
- Health: `GET /api/health` → shows `n_docs`, `n_chunks`, `ocr_available`,
  `keep_pdfs`, LLM provider presets, etc.
- Interactive docs: `http://localhost:8000/docs` (Swagger UI)

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SGA_DATA_DIR` | `backend/data/` | Data root. Point at a volume mount (e.g. `/data`) for persistence across deploys; otherwise documents are wiped on restart. |
| `SGA_CORS_ORIGINS` | _(localhost only)_ | Comma-separated extra allowed origins, e.g. `https://solargrid-frontend.up.railway.app` |
| `SGA_MAX_PDF_MB` | `50` | Max upload size; larger files are rejected with 413. |
| `SGA_KEEP_PDFS` | `0` | `1` = archive original PDFs under `uploads/`. Default `0`: PDFs are indexed as text and never stored (zero-footprint mode). |
| `SGA_OCR_DPI` | `150` | Raster resolution for OCR pages. |
| `SGA_OCR_WORKERS` | `3` | Parallel Tesseract workers (RAM-bounded). |
| `SGA_OCR_LANG` | _(auto)_ | Tesseract languages; auto = `eng+urd` when Urdu data is installed, else `eng`. |
| `PORT` | `8000` | Set automatically by Railway; used by the start command. |

## API overview

Base path `/api`. All responses are JSON; errors use `{"detail": "…"}`.

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness + stats + LLM provider presets (drives the frontend dropdown). |
| `POST /llm/test` | Test an LLM connection `{provider, base_url, api_key, model, temperature}`. |
| `POST /documents/preview` | Upload a PDF (multipart `file`) → pages, chars, OCR info, `usable`, scan `warning`. Doesn't index. |
| `POST /documents/upload` | Index a PDF + metadata form fields (`title, disco, doc_type, status, issue_date, effective_date, notes`). Idempotent: re-uploading the same file refreshes metadata. |
| `GET /documents` | List documents (with `superseded` flags), chunk count, manifest. |
| `PATCH /documents/{doc_id}` | Update `issue_date, effective_date, disco, status, notes`. |
| `DELETE /documents/{doc_id}` | Remove a document, its chunks and cached extraction. |
| `POST /documents/rebuild` | Re-embed all chunks (full build only). |
| `POST /chat` | RAG answer `{question, discos, statuses, min_effective, latest_only, top_k, vector, bm25, rewrite, rerank, memory_enabled, history, llm}` → `answer, confidence, verified, sources[]`. |
| `GET /rates` | DISCO rate table. |
| `POST /rates/{disco}` | Update one DISCO's `{peak, offpeak, fixed, buyback}`. |
| `POST /dashboard/calc` | Savings-dashboard computation. |
| `POST /solar/calc` | Solar sizing / generation / payback computation. |
| `POST /solar/plan` | Heuristic + optional LLM solar plan. |
| `POST /bills/extract` | Extract 11 fields + insights from a bill PDF (`file, use_llm, provider, base_url, api_key, model`). |
| `POST /bills/save` | Save a bill to history. |
| `GET /bills/history` | Saved bills. |
| `DELETE /bills/history` | Clear bill history. |

LLM providers (presets served by `/api/health`, keyed by name):

| Provider | Base URL | Model |
|---|---|---|
| Groq (free, recommended) | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` |
| gpt-oss-120b (Modal — free) | _(your endpoint)_ | `openai/gpt-oss-120b` |
| Grok (xAI API) | `https://api.x.ai/v1` | `grok-4-1-fast` |
| Custom / Ollama | `http://localhost:11434/v1` | `gpt-oss:20b` |
| Offline (no LLM) | — | retrieval only |

Any OpenAI-compatible endpoint works — the app only uses `chat.completions`.

## Deployment (Railway)

1. Push this folder as the backend service. `railway.json` selects the NIXPACKS
   builder with start command `uvicorn main:app --host 0.0.0.0 --port $PORT`
   and healthcheck path `/api/health`.
2. `nixpacks.toml` installs `tesseract-ocr` + English/Urdu data during the build
   (needed for scanned PDFs). No action required.
3. Set `SGA_CORS_ORIGINS` to the frontend URL.
4. Optional: attach a small volume and set `SGA_DATA_DIR` to its mount path so
   uploads survive restarts. Without a volume, re-upload documents after redeploys.
5. Verify: open `https://<backend>/api/health` → `"ok": true`,
   `"ocr_available": true`, `"keep_pdfs": false`.

## Light vs full build

The default `requirements.txt` is the **light build**: keyword (BM25) retrieval,
~65 MB idle RAM, no model downloads. Everything works except vector (semantic)
search and cross-encoder reranking, which degrade gracefully.

For the **full build** (local machine or a bigger server):

```bash
pip install "sentence-transformers>=3.0" "faiss-cpu>=1.8"
```

then restart and call `POST /api/documents/rebuild` once to embed existing chunks.
`SGA_DATA_DIR/cache/huggingface` holds the downloaded model (~500 MB).

## Notes & limits

- Uploads: PDF only, ≤ `SGA_MAX_PDF_MB`, must yield real text (native or OCR).
  Scanned PDFs without readable content are rejected with an actionable 400 message.
- A 96-page scan OCRs in ~2–4 minutes per file (preview does the work; the
  follow-up upload reuses the extraction cache and finishes instantly).
- Default chat filters: `statuses=["official"]`, `latest_only=true`, `top_k=5`.
  `rewrite`/`rerank` default off (they need an LLM key / the full build).
- On startup the app deletes orphaned archived PDFs and stale partial model
  downloads (light mode only) and logs what it removed.
- API keys are never stored server-side — the frontend sends the key per request.
