"""RAG pipeline — ported from app.py lines 656-834."""
from __future__ import annotations

import datetime as dt
import json
import os
import re

import numpy as np

from .chunking import tokenize
from .config import HF_CACHE, TOP_K_DEFAULT
from .index_store import IndexStore
from .llm import llm_chat
from .utils import parse_date

_DOC_LIST_PATTERNS = [
    # "list/show/display ... documents/pdfs/files"
    re.compile(r"\b(list|show|display)\b.{0,30}\b(pdfs?|documents?|files?|circulars?|notifications?)\b", re.I),
    # "which/what pdf/document ... have/you/uploaded/indexed/available"
    re.compile(r"\b(which|what)\b.{0,25}\b(pdfs?|documents?|files?|circulars?)\b.{0,35}\b(have|you|uploaded|indexed|available|stored|there|got)\b", re.I),
    # Roman Urdu: "kon se/si documents/pdfs hain"
    re.compile(r"\bkon\s?s[aeiou]\b.{0,25}\b(documents?|pdfs?|files?)\b", re.I),
    # Urdu script: دستاویز / فہرست + question word
    re.compile(r"(دستاویز|فہرست).{0,30}(کون|کیا|کتنی|ہیں|ہے|بتا|دکھا)"),
]


def _is_doc_list_query(question: str) -> bool:
    q = question or ""
    return any(rx.search(q) for rx in _DOC_LIST_PATTERNS)


def _doc_list_answer(docs: list[dict], question: str) -> dict:
    """Answer 'which PDFs do you have' directly from the document metadata."""
    urdu = bool(re.search(r"[\u0600-\u06FF]", question or ""))
    if not docs:
        if urdu:
            answer = "📂 **ابھی کوئی دستاویز موجود نہیں۔** پہلے Documents صفحے پر PDF اپ لوڈ کریں۔"
        else:
            answer = ("📂 **No documents are indexed yet.** Upload a PDF on the Documents page first, "
                      "then ask me about tariffs, peak hours, or net metering.")
        return {"answer": answer, "context_chunks": [], "confidence": (0.5, "No documents yet"),
                "verified": True, "rewritten": None}
    lines = []
    for i, d in enumerate(docs, 1):
        title = (d.get("title") or d.get("filename") or "doc").strip()
        meta = []
        if d.get("disco"):
            meta.append(str(d["disco"]))
        if d.get("doc_type"):
            meta.append(str(d["doc_type"]))
        if d.get("pages"):
            meta.append(f"{d['pages']} pages")
        eff = d.get("effective_date") or ""
        meta.append(f"effective {eff}" if eff else "effective n.d.")
        lines.append(f"{i}. **{title}** — {', '.join(meta)}")
    if urdu:
        header = f"📂 **چیٹ کے لیے میرے پاس {len(docs)} دستاویز ہیں:**"
        footer = "ان دستاویزات سے ٹیرف، پیک آورز یا نیٹ میٹرنگ کے بارے میں پوچھیں۔"
    else:
        header = f"📂 **I have {len(docs)} document(s) for chat:**"
        footer = "Ask me about tariffs, peak hours, net metering, or buyback rates from these documents."
    answer = header + "\n\n" + "\n".join(lines) + "\n\n" + footer
    return {"answer": answer, "context_chunks": [], "confidence": (0.97, "High confidence"),
            "verified": True, "rewritten": None}


SYSTEM_PROMPT = """You are SolarGrid Advisor (سولر گرڈ ایڈوائزر), an expert energy assistant for Pakistan.
You help solar panel owners and electricity customers understand DISCO billing, time-of-use tariffs,
peak/off-peak hours, net-metering, and buyback rates.

GROUNDING RULES — follow strictly:
1. Answer ONLY from the numbered document excerpts provided. Never invent rates, dates or rules.
2. Cite the excerpt number after every factual claim, e.g. [1], [3].
3. If the excerpts do not contain the answer, reply exactly:
   "❌ This is not covered in the uploaded documents. Please upload the relevant NEPRA/DISCO circular and try again."
4. When you use a rate or rule, mention its effective date (from the excerpt header) and remind that
   rates change by circular — the latest effective notification wins.
5. Answer in the user's language (Urdu or English). Be practical, concise and specific.

Current date: {today}
{memory}
"""


def query_rewrite(client_pack, question: str) -> list[str] | None:
    if client_pack is None:
        return None
    try:
        out = llm_chat(client_pack, [
            {"role": "system", "content": "You rewrite user questions about Pakistan electricity tariffs into search queries. Return ONLY a JSON array of 2-3 short strings (English + Urdu keywords where helpful). No other text."},
            {"role": "user", "content": question},
        ])
        m = re.search(r"\[[\s\S]*\]", out or "")
        if m:
            qs = json.loads(m.group(0))
            if isinstance(qs, list) and all(isinstance(q, str) for q in qs):
                return [q.strip() for q in qs if q.strip()][:3]
    except Exception:
        return None
    return None


def rerank_chunks(query: str, chunks: list[dict], k: int) -> list[dict]:
    """Optional cross-encoder rerank (lazy load). Falls back to input order on error."""
    if len(chunks) <= k:
        return chunks[:k]
    try:
        from sentence_transformers import CrossEncoder
        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6")
        pairs = [(query, c["text"][:400]) for c in chunks]
        scores = model.predict(pairs)
        order = np.argsort(-scores)[:k]
        return [dict(chunks[i], score=float(scores[i])) for i in order]
    except Exception:
        return chunks[:k]


def build_context(chunks: list[dict], max_chars: int = 6500) -> tuple[str, list[dict]]:
    parts, used = [], 0
    for i, c in enumerate(chunks, 1):
        eff = c.get("effective_date") or "n.d."
        hdr = f"[{i}] ({c.get('doc_title', 'doc')} | {c.get('disco', '—')} | effective {eff} | p.{c.get('page', '?')})"
        block = f"{hdr}\n{c['text']}\n"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n---\n".join(parts), parts


def coverage(query: str, context: str) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    ctx = (context or "").lower()
    hit = sum(1 for w in q if w in ctx)
    return hit / len(q)


def confidence(top_score: float, cov: float, n_cites: int, verified: bool) -> tuple[float, str]:
    if n_cites == 0:
        base = 0.25 * min(top_score, 1.0) + 0.2 * cov
    else:
        base = 0.4 * min(top_score, 1.0) + 0.3 * cov + (0.3 if verified else 0.15)
    score = max(0.03, min(0.97, base))
    label = "High confidence" if score >= 0.7 else ("Medium confidence" if score >= 0.45 else "Low confidence — check the source")
    return round(score, 2), label


def validate_citations(answer: str, n_ctx: int) -> tuple[int, bool]:
    cites = set(int(x) for x in re.findall(r"\[(\d+)\]", answer or ""))
    unknown = {c for c in cites if c > n_ctx or c < 1}
    return (len(cites) - len(unknown)), (len(unknown) == 0 and len(cites) > 0)


def build_memory(msgs: list[dict]) -> str:
    if not msgs:
        return ""
    lines = []
    for m in msgs[-6:]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['text'][:400]}")
    return "Recent conversation (for context only — still ground new answers in the excerpts):\n" + "\n".join(lines)


def rag_answer(question: str, store: IndexStore, client_pack, opts: dict) -> dict:
    """Full RAG pipeline. Returns dict(answer, context_chunks, confidence, verified, rewritten)."""
    if _is_doc_list_query(question):
        return _doc_list_answer(opts.get("docs") or [], question)
    filters = {
        "discos": opts.get("discos") or [],
        "statuses": opts.get("statuses") or [],
        "min_effective": parse_date(opts.get("min_effective", "")) if opts.get("min_effective") else None,
        "latest_only": bool(opts.get("latest_only")),
        "latest_eff_map": store.latest_effective_map(),
    }
    top_k = int(opts.get("top_k", TOP_K_DEFAULT))
    queries = [question]
    rewritten = None
    if opts.get("rewrite") and client_pack is not None:
        rewritten = query_rewrite(client_pack, question) or None
        if rewritten:
            queries = rewritten

    best: list[dict] = []
    diag = {"candidates": 0, "top_score": 0.0}
    seen = set()
    for q in queries:
        res, d = store.hybrid_search(
            q, top_k=max(top_k, 8),
            use_vector=opts.get("vector", True),
            use_bm25=opts.get("bm25", True),
            filters=filters,
        )
        diag = d
        for item in res:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            existing = next((b for b in best if b["id"] == item["id"]), None)
            if existing:
                existing["score"] = max(existing["score"], item["score"])
            else:
                best.append(item)
        best.sort(key=lambda x: -x["score"])
    best = best[: max(top_k * 2, 10)]

    if opts.get("rerank"):
        best = rerank_chunks(question, best, top_k)
    else:
        best = best[:top_k]

    if not best:
        return {
            "answer": ("🔍 **No matching passages found** in the current filters. "
                       "Try widening the DISCO/status/date filters, or upload the relevant circular "
                       "on the Documents page."),
            "context_chunks": [], "confidence": (0.1, "No sources retrieved"),
            "verified": False, "rewritten": rewritten,
        }

    context, _ = build_context(best)
    cov = coverage(question, context)

    if client_pack is None:
        lines = ["🔌 **Offline mode (no LLM configured)** — here are the most relevant document passages:\n"]
        for i, c in enumerate(best, 1):
            lines.append(f"**[{i}] {c['doc_title']}** ({c['disco']}, effective {c.get('effective_date') or 'n.d.'}, p.{c['page']})\n> {c['text'][:500]}\n")
        conf = confidence(diag.get("top_score", 0.3), cov, 0, False)
        return {"answer": "\n".join(lines), "context_chunks": best,
                "confidence": conf, "verified": True, "rewritten": rewritten}

    memory = opts.get("memory", "")
    system = SYSTEM_PROMPT.format(today=dt.date.today().isoformat(), memory=memory)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Document excerpts (numbered sources):\n{context}\n\n"
        f"Answer the question now. Use [n] citations. If the answer is not in the excerpts, follow rule 3."
    )
    messages = [{"role": "system", "content": system}]
    try:
        answer = llm_chat(client_pack, messages + [{"role": "user", "content": user_prompt}])
        if not answer:
            answer = "⚠️ The model returned an empty response — try again or switch model/provider."
    except Exception as e:
        answer = (f"⚠️ **LLM request failed:** {str(e)[:300]}\n\n"
                  f"Meanwhile, here are the retrieved passages:\n\n" +
                  "\n".join(f"**[{i}]** {c['doc_title']} (p.{c['page']}): {c['text'][:300]}" for i, c in enumerate(best, 1)))

    n_cites, verified = validate_citations(answer, len(best))
    conf = confidence(diag.get("top_score", 0.0), cov, n_cites, verified)
    return {"answer": answer, "context_chunks": best, "confidence": conf, "verified": verified, "rewritten": rewritten}
