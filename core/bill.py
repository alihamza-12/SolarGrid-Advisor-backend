"""Bill field extraction (regex + optional LLM) — ported from app.py lines 952-993.

Pakistani DISCO bills (IESCO/LESCO/…) mix English labels with Urdu (RTL) fragments
and print labels/reference numbers with letter-spacing ("F I X E D", "08 E 1 R 4").
normalize_bill_text() cleans both problems before any pattern runs: Urdu-script
runs are dropped (they carry no field values — amounts/dates are Latin digits)
and letter-spaced runs are re-joined. No extra dependencies needed.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .llm import llm_chat

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]+")
# 3+ single alphanumerics separated by single spaces: "F I X E D", "2 5 0".
# No leading \b on purpose, so runs starting mid-token ("08 E 1 R …") join too.
_DESPACE_RE = re.compile(r"([A-Za-z0-9](?: [A-Za-z0-9]){2,})\b")


def normalize_bill_text(text: str) -> str:
    """Clean raw bill text so label regexes can match.

    1. Drop Urdu-script fragments (RTL noise between English labels).
    2. Re-join letter-spaced print ("F I X E D C H A R G E S" -> "FIXEDCHARGES",
       "08 E 1 R 4 I 5 D" -> "08E1R4I5D").
    3. Collapse stray whitespace.
    """
    t = _ARABIC_RE.sub(" ", text or "")
    t = _DESPACE_RE.sub(lambda m: m.group(1).replace(" ", ""), t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


BILL_PATTERNS = {
    "Billing period (from)": (
        r"billing\s+period[^\n]{0,60}?from\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})"
        r"|bill\s*month[^\n]{0,80}?([a-z]{3,9})\s*[- ]?(\d{2,4})"
    ),
    "Billing period (to)": (
        r"billing\s+period[^\n]{0,60}?to\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})"
        r"|bill\s*month[^\n]{0,80}?([a-z]{3,9})\s*[- ]?(\d{2,4})"
    ),
    "Total units": (
        r"(?<!sub)(?<!sub )(?:total|billed)\s*units?[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|units?\s*consumed[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|consumption[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)\s*units?"
  
        r"|(?<!sub)(?<!sub )total[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)\s*units?"
    ),
    "Peak units": (
        r"(?<!off[\s-])(?<!off)peak\s*(?:hour\s*)?units?[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|units?\s*\(?peak\)?[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
    ),
    "Off-peak units": (
        r"off[\s-]?peak\s*(?:hour\s*)?units?[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|units?\s*\(?off[\s-]?peak\)?[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
    ),
    "Export units (net metering)": (
        r"(?:export|excess|net[\s-]?metering)\s*units?[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|units?\s*exported[^\d]{0,10}?([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
    ),
    "Fixed charge (Rs)": (
        r"fixed\s*charges?[^\n\d]{0,30}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    "Total amount (Rs)": (
        r"(?:grand\s*total|total\s*(?:bill|amount|payable|due|current\s*bill)|net\s*amount\s*payable|amount\s*(?:payable|due))"
        r"[^\n\d]{0,30}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    "Energy charge (Rs)": (
        r"(?:energy|variable)\s*charges?[^\n\d]{0,30}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    "WAPDA / surcharge (Rs)": (
        r"wapda[^\n]{0,40}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
        r"|(?:fc|fuel|late\s*payment)\s*surcharge[^\n]{0,20}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    "Taxes (FBR/PTA) (Rs)": r"(?:fbr|pta|tax(?:es)?|lev(?:y|ies))[^\n]{0,40}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?",
}


def extract_bill(text: str) -> dict:
    text = normalize_bill_text(text)
    found = {}
    for label, pat in BILL_PATTERNS.items():
        m = re.search(pat, text, re.I)
        if m:
            groups = [g for g in m.groups() if g]
            if not groups:
                continue
            raw = " ".join(groups) if label.startswith("Billing period") else groups[0]
            raw = raw.replace(",", "")
            try:
                found[label] = float(raw) if label.endswith("(Rs)") or "units" in label.lower() or "Export" in label else raw
            except ValueError:
                found[label] = raw
    if "Total units" not in found:
        # IESCO-style fallback: derive units from meter readings.
        rm = re.search(r"previous[^\d]{0,30}?(\d{1,7})[^\d]{0,60}?present[^\d]{0,30}?(\d{1,7})", text, re.I)
        if rm:
            try:
                prev, pres = int(rm.group(1)), int(rm.group(2))
                if 100 < pres and 0 < pres - prev < 100000:
                    found["Total units"] = float(pres - prev)
            except ValueError:
                pass
    return found


def llm_extract_bill(client_pack, text: str) -> Optional[dict]:
    if client_pack is None:
        return None
    try:
        out = llm_chat(client_pack, [
            {"role": "system", "content": 'You extract fields from Pakistani electricity bills. Return ONLY JSON with keys: "billing_period_from","billing_period_to","total_units","peak_units","offpeak_units","export_units","fixed_charge","total_amount","energy_charge","wapda","taxes". Numbers as numbers, unknown fields as null.'},
            {"role": "user", "content": normalize_bill_text(text)[:6000]},
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
