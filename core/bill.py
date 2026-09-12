"""Bill field extraction (LLM-first + regex backup) — ported from app.py lines 952-993.

Pakistani DISCO bills (IESCO/LESCO/…) mix English labels with Urdu (RTL) fragments,
print labels/reference numbers with letter-spacing ("F I X E D", "08 E 1 R 4 I 5 D")
and lay values out in columns. normalize_bill_text() cleans the first two problems
before any pattern runs; main.py runs the LLM first (it reads messy layouts best)
and uses these patterns to backfill whatever the LLM missed.

Missing fields stay None ("not on this bill") — values are never invented.
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
        r"|(?:meter|reading)[^\n]{0,80}?units?\b\s*:?\s*([0-9]{1,6}(?:,[0-9]{3})*)"
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
        r"(?:grand\s*total|total\s*(?:bill|amount|payable|due|current\s*bill)|net\s*amount\s*payable|amount\s*(?:payable|due)|payable\s*within\s*due\s*date)"
        r"[^\n\d]{0,30}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    "Energy charge (Rs)": (
        # (?<!net ) keeps "Net Electricity Charges 84.01 %" (a percentage row) out.
        r"(?<!net\s)(?:energy|variable|total\s*electricity|electricity)\s*charges?"
        r"[^\n\d]{0,30}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    "WAPDA / surcharge (Rs)": (
        r"wapda[^\n]{0,40}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
        r"|(?:fc|fuel|late\s*payment|l\.?\s*p\.?)\s*surcharge[^\n]{0,20}?(?:rs\.?\s*)?([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    # Taxes are matched by _tax_amount() (skips percentages like "15.99 %").
    "Taxes (FBR/PTA) (Rs)": "",
}


_TAX_LABEL_RE = re.compile(
    r"fbr|pta|tax(?:es)?|lev(?:y|ies)|gst(?!\s*no\.?)|income\s*tax|electricity\s*duty", re.I
)
_TAX_NUM_RE = re.compile(r"\d{1,6}(?:[.,]\d{1,3})*(?:\.\d{1,2})?")


def _tax_amount(text: str) -> Optional[float]:
    """Tax amount after a tax label on the same line, skipping %-figures.

    E.g. "Taxes 15.99 % 215" -> 215.0 (not 15.99); "FBR PTA TAXES 16.29" -> 16.29.
    Never matches registration numbers like "GST NO: 26-00-…".
    """
    text = normalize_bill_text(text)
    m = _TAX_LABEL_RE.search(text)
    if not m:
        return None
    line = text[m.start():].split("\n", 1)[0]
    for nm in _TAX_NUM_RE.finditer(line):
        if line[nm.end():nm.end() + 2].strip().startswith("%"):
            continue
        try:
            return float(nm.group(0).replace(",", ""))
        except ValueError:
            continue
    return None


def meter_units(text: str) -> Optional[float]:
    """Billed units from meter readings: (present - previous) x MF.

    Returns a value only when exactly one reading pair is found (multi-register
    TOU bills are left to the LLM/regex). Guards reject page numbers and
    reversed/zero diffs.
    """
    text = normalize_bill_text(text)
    pairs = re.findall(
        r"previous[^\d]{0,30}?(\d{1,7})[^\d]{0,60}?present[^\d]{0,30}?(\d{1,7})", text, re.I
    )
    good = []
    for a, b in pairs:
        try:
            prev, pres = int(a), int(b)
        except ValueError:
            continue
        if 100 < pres and 0 < pres - prev < 100000:
            good.append(pres - prev)
    if len(good) != 1:
        return None
    mf = 1
    mm = re.search(r"\bm\.?\s*f\.?\s*:?\s*(\d{1,2})\b", text, re.I)
    if mm:
        try:
            mf = int(mm.group(1)) or 1
        except ValueError:
            pass
    return float(good[0] * mf)


def extract_bill(text: str) -> dict:
    text = normalize_bill_text(text)
    found = {}
    for label, pat in BILL_PATTERNS.items():
        if label == "Taxes (FBR/PTA) (Rs)":
            v = _tax_amount(text)
            if v is not None:
                found[label] = v
            continue
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
        mu = meter_units(text)
        if mu is not None:
            found["Total units"] = mu
    return found


def llm_extract_bill(client_pack, text: str) -> Optional[dict]:
    if client_pack is None:
        return None
    try:
        out = llm_chat(client_pack, [
            {"role": "system", "content": (
                'You extract fields from Pakistani DISCO electricity bills (IESCO, LESCO, K-Electric, etc.). '
                'Return ONLY JSON with exactly these keys: "billing_period_from","billing_period_to",'
                '"total_units","peak_units","offpeak_units","export_units","fixed_charge","total_amount",'
                '"energy_charge","wapda","taxes". Numbers as plain numbers (no commas), unknown/absent fields as null. '
                'Rules: total_units = current billed units only (UNITS / UNITS CONSUMED / TOTAL UNITS) — ignore meter '
                'READING values themselves and ignore the 12-month BILL HISTORY table. peak/offpeak/export_units = null '
                'unless TOU peak/off-peak or net-metering export lines are printed. total_amount = Grand Total / Payable '
                'WITHIN due date, never Payable AFTER due date. fixed_charge = FIXED CHARGES line only, else null. '
                'energy_charge = Total/Variable/Energy Electricity Charges. wapda = WAPDA / L.P. / FC / fuel surcharge '
                'amount, else null. taxes = tax amount (FBR/GST/income tax); ignore percentages like 15.99%. '
                'Billing periods from BILL MONTH (e.g. JUN 26) or reading/issue dates. NEVER guess, compute or copy '
                'history-table values.'
            )},
            {"role": "user", "content": normalize_bill_text(text)[:8000]},
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
