"""Bill field extraction (regex + optional LLM) — ported from app.py lines 952-993."""
from __future__ import annotations

import json
import re
from typing import Optional

from .llm import llm_chat

BILL_PATTERNS = {
    "Billing period (from)": r"billing\s+period[^\n]{0,60}?from\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
    "Billing period (to)": r"billing\s+period[^\n]{0,60}?to\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
    "Total units": r"total\s*units?\s*[:\-]?\s*([0-9]{1,6}(?:[.,][0-9]{1,3})*)",
    "Peak units": r"peak\s*(?:hour\s*)?units?\s*[:\-]?\s*([0-9]{1,6}(?:[.,][0-9]{1,3})*)",
    "Off-peak units": r"off[\s-]?peak\s*(?:hour\s*)?units?\s*[:\-]?\s*([0-9]{1,6}(?:[.,][0-9]{1,3})*)",
    "Export units (net metering)": r"(?:export|excess|net[\s-]?metering)\s*units?\s*[:\-]?\s*([0-9]{1,6}(?:[.,][0-9]{1,3})*)",
    "Fixed charge (Rs)": r"fixed\s*charges?\s*[:\-]?\s*(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?",
    "Total amount (Rs)": r"total\s*(?:bill|amount|payable|due)\s*[:\-]?\s*(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?",
    "Energy charge (Rs)": r"energy\s*charges?\s*[:\-]?\s*(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?",
    "WAPDA / surcharge (Rs)": r"wapda[^\n]{0,40}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?",
    "Taxes (FBR/PTA) (Rs)": r"(?:fbr|pta|taxes?|levies?)[^\n]{0,40}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?",
}


def extract_bill(text: str) -> dict:
    found = {}
    for label, pat in BILL_PATTERNS.items():
        m = re.search(pat, text, re.I)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                found[label] = float(raw) if label.endswith("(Rs)") or "units" in label.lower() or "Export" in label else raw
            except ValueError:
                found[label] = raw
    return found


def llm_extract_bill(client_pack, text: str) -> Optional[dict]:
    if client_pack is None:
        return None
    try:
        out = llm_chat(client_pack, [
            {"role": "system", "content": 'You extract fields from Pakistani electricity bills. Return ONLY JSON with keys: "billing_period_from","billing_period_to","total_units","peak_units","offpeak_units","export_units","fixed_charge","total_amount","energy_charge","wapda","taxes". Numbers as numbers, unknown fields as null.'},
            {"role": "user", "content": text[:6000]},
        ])
        m = re.search(r"\{[\s\S]*\}", out or "")
        if m:
            return json.loads(m.group(0))
    except Exception:
        return None
    return None


LABEL_MAP = {
    "billing_period_from": "Billing period (from)", "billing_period_to": "Billing period (to)",
    "total_units": "Total units", "peak_units": "Peak units", "offpeak_units": "Off-peak units",
    "export_units": "Export units (net metering)", "fixed_charge": "Fixed charge (Rs)",
    "total_amount": "Total amount (Rs)", "energy_charge": "Energy charge (Rs)",
    "wapda": "WAPDA / surcharge (Rs)", "taxes": "Taxes (FBR/PTA) (Rs)",
}
