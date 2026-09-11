"""LLM client (OpenAI-compatible: xAI / Modal / Ollama / …) — ported from app.py lines 610-655."""
from __future__ import annotations

from typing import Optional


def get_llm(provider: str, base_url: str, api_key: str, model: str, temperature: float = 0.2):
    if "Offline" in (provider or ""):
        return None
    try:
        from openai import OpenAI
        base = (base_url or "").strip().rstrip("/")
        if not base:
            raise ValueError("Base URL is required for this provider")
        client = OpenAI(base_url=base, api_key=(api_key or "").strip() or "not-set", timeout=180.0)
        return client, (model or "").strip() or "gpt-oss-120b", temperature
    except Exception:
        return None


def llm_chat(client_pack, messages: list[dict]) -> str:
    """client_pack from get_llm(). Returns the model's reply text."""
    client, model, temperature = client_pack
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return (resp.choices[0].message.content or "").strip()


def llm_ok(client_pack) -> Optional[bool]:
    if client_pack is None:
        return None
    try:
        out = llm_chat(client_pack, [{"role": "user", "content": "Reply with the single word: ok"}])
        return bool(out)
    except Exception:
        return False
