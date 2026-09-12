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
# Urdu / Arabic-Indic digits -> ASCII (applied BEFORE the Arabic-script strip, so a
# value written as "۱۳۴۵" survives as 1345 instead of being deleted).
_DIGIT_MAP = str.maketrans(
    "".join(chr(c) for c in list(range(0x06F0, 0x06FA)) + list(range(0x0660, 0x066A))),
    "0123456789" * 2,
)


def normalize_bill_text(text: str) -> str:
    """Clean raw bill text so label regexes can match.

    1. Map Urdu/Arabic-Indic digits to ASCII.
    2. Drop Urdu-script fragments (RTL noise between English labels).
    3. Re-join letter-spaced print ("F I X E D C H A R G E S" -> "FIXEDCHARGES",
       "08 E 1 R 4 I 5 D" -> "08E1R4I5D").
    4. Collapse stray whitespace.
    """
    t = (text or "").translate(_DIGIT_MAP)
    t = _ARABIC_RE.sub(" ", t)
    t = _DESPACE_RE.sub(lambda m: m.group(1).replace(" ", ""), t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


_MONEY = r"(?<![A-Za-z0-9])(\d{1,6}(?:[.,]\d{1,3})*(?:\.\d{1,2})?)"
# Junk between a same-line label and its value: non-digit chars (OCR debris like
# "zint", "cu'", "Rs.") or parenthetical remarks ("(see note 2)", "(Rs.)").
_JUNK = r"(?:\([^)\n]{0,30}\)|[^\d\n])"
_MONTHS = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
# Gap between "BILL MONTH" and its value: OCR/column debris ("5 _-"), the
# neighbouring column's words ("REFERENCE NO") and stray single digits. Real month
# names only (never a bare [a-z]+, which would match words like "reference").
_MONTH_GAP = r"(?:[^\d]|\b\d\b){0,80}?"


BILL_PATTERNS = {
    "Billing period (from)": (
        r"billing\s+period[^\n]{0,60}?from\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})"
        r"|bill(?:ing)?\s*month\b" + _MONTH_GAP + _MONTHS + r"\s*[- ]?(\d{2,4})"
    ),
    "Billing period (to)": (
        r"billing\s+period[^\n]{0,60}?to\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})"
        r"|bill(?:ing)?\s*month\b" + _MONTH_GAP + _MONTHS + r"\s*[- ]?(\d{2,4})"
    ),
    "Total units": (
        r"(?<!sub)(?<!sub )(?:total|billed)\s*units?[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|units?\s*consumed[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|consumption[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)\s*units?"
        r"|(?<!sub)(?<!sub )total[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)\s*units?"
        r"|(?:meter|reading)[^\n]{0,80}?units?\b\s*:?\s*(?<![A-Za-z0-9])([0-9]{1,6}(?:,[0-9]{3})*)"
        # Bare "UNITS 64" (incl. wrapped "UNITS\n64"); the lookbehinds keep TOU /
        # net-metering registers (PEAK [HOUR] UNITS, OFF-PEAK UNITS, EXPORT UNITS)
        # from being mistaken for the total.
        r"|(?<!peak\s)(?<!peak\shour\s)"
        r"(?<!offpeak\s)(?<!off\speak\s)(?<!off-peak\s)"
        r"(?<!offpeak\shour\s)(?<!off\speak\shour\s)(?<!off-peak\shour\s)"
        r"(?<!export\s)(?<!metering\s)(?<!excess\s)"
        r"(?<!subtotal\s)(?<!sub\stotal\s)(?<!sub-total\s)"
        r"units?\b\s*:?\s*(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    "Peak units": (
        r"(?<!off[\s-])(?<!off)peak\s*(?:hour\s*)?units?[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|units?\s*\(?peak\)?[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
    ),
    "Off-peak units": (
        r"off[\s-]?peak\s*(?:hour\s*)?units?[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|units?\s*\(?off[\s-]?peak\)?[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
    ),
    "Export units (net metering)": (
        r"(?:export|excess|net[\s-]?metering)\s*units?[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
        r"|units?\s*exported[^\d]{0,10}?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)"
    ),
    "Fixed charge (Rs)": (
        r"fixed\s*charges?[^\n\d]{0,30}?(?:rs\.?\s*)?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    # Total is matched by _total_amount() (strong labels first, then generic ones).
    "Total amount (Rs)": "",
    "Energy charge (Rs)": (
        # (?<!net ) keeps "Net Electricity Charges 84.01 %" (a percentage row) out.
        r"(?<!net\s)(?:energy|variable|total\s*electricity|electricity)\s*charges?"
        + _JUNK + r"{0,40}(?:rs\.?\s*)?" + _MONEY
    ),
    "WAPDA / surcharge (Rs)": (
        r"wapda[^\n]{0,40}?(?:rs\.?\s*)?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
        r"|(?:fc|fuel|late\s*payment|l\.?\s*p\.?)\s*surcharge[^\n]{0,20}?(?:rs\.?\s*)?(?<![A-Za-z0-9])([0-9]{1,6}(?:[.,][0-9]{1,3})*)(?:\.[0-9]{1,2})?"
    ),
    # Taxes are matched by _tax_amount() (skips percentages like "15.99 %").
    "Taxes (FBR/PTA) (Rs)": "",
}

# Grand Total / Payable-within-due-date are THE payable total; generic "total …"
# lines (incl. "Total Current Bill", which differs from the grand total when FPA
# or arrears apply) are only a fallback.
_TOTAL_STRONG_RE = re.compile(
    r"(?:grand\s*total|payable\s*within\s*due\s*date)" + _JUNK + r"{0,40}(?:rs\.?\s*)?" + _MONEY,
    re.I,
)
_TOTAL_WEAK_RE = re.compile(
    r"(?:total\s*(?:current\s*bill|bill|amount|payable|due)|net\s*amount\s*payable|amount\s*(?:payable|due))"
    + _JUNK + r"{0,40}(?:rs\.?\s*)?" + _MONEY,
    re.I,
)


def _total_amount(text: str) -> Optional[float]:
    """Payable total: Grand Total / Payable-within first, generic totals fallback.

    Never matches "Payable AFTER due date" (no alternative contains bare
    "payable"). Values under Rs 10 are rejected as debris grabs ("2nd copy").
    """
    text = normalize_bill_text(text)
    for rx in (_TOTAL_STRONG_RE, _TOTAL_WEAK_RE):
        m = rx.search(text)
        if not m:
            continue
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if 0 < v < 10:
            return None
        return v
    return None


_TAX_LABEL_RE = re.compile(
    r"fbr|pta|tax(?:es)?|lev(?:y|ies)|gst(?!\s*no\.?)|income\s*tax|electricity\s*duty", re.I
)
_NET_LABEL_RE = re.compile(r"\bnet\s+(?:electricity\s+)?charges?\b", re.I)
_SUB_LABEL_RE = re.compile(r"\bsubsid(?:y|ies)\b", re.I)
_TAX_NUM_RE = re.compile(r"\b\d{1,6}(?:[.,]\d{1,3})*(?:\.\d{1,2})?\b")
_PARENS_RE = re.compile(r"\([^)\n]{0,60}\)")


def _labeled_amount(text: str, label_re) -> Optional[float]:
    """First number on a label's line, skipping %-figures and (remarks).

    E.g. "Taxes 15.99 % 215" -> 215.0 (not 15.99); "Net Electricity Charges
    84.01 % 1130" -> 1130.0; "Taxes (see note 5) 215" -> 215.0 (not 5).
    """
    m = label_re.search(text)
    if not m:
        return None
    line = text[m.start():].split("\n", 1)[0]
    line = _PARENS_RE.sub(" ", line)
    for nm in _TAX_NUM_RE.finditer(line):
        if line[nm.end():nm.end() + 2].strip().startswith("%"):
            continue
        try:
            return float(nm.group(0).replace(",", ""))
        except ValueError:
            continue
    return None


def _tax_amount(text: str) -> Optional[float]:
    """Tax amount after a tax label on the same line, skipping %-figures.

    E.g. "Taxes 15.99 % 215" -> 215.0 (not 15.99); "FBR PTA TAXES 16.29" -> 16.29.
    Never matches registration numbers like "GST NO: 26-00-…".
    """
    return _labeled_amount(normalize_bill_text(text), _TAX_LABEL_RE)


def energy_derived(text: str) -> Optional[float]:
    """Net Electricity Charges + Subsidies — the bill's own arithmetic.

    Used when the printed energy figure is missing or OCR-garbled. Both labels
    must match explicitly; a tiny subsidy (<1% of net, i.e. leftover debris such
    as a note number) vetoes the derivation instead of corrupting a good figure.
    """
    t = normalize_bill_text(text)
    net = _labeled_amount(t, _NET_LABEL_RE)
    sub = _labeled_amount(t, _SUB_LABEL_RE)
    if net is None or sub is None:
        return None
    if 0 < net < 10 or 0 < sub < 0.01 * net:
        return None
    return net + sub


def crosscheck_energy(value, text: str):
    """Apply the net+subsidies identity to an energy figure.

    Fills a missing value; replaces a present one only when they disagree by
    more than rounding (2%) — i.e. the printed figure was OCR-garbled.
    """
    try:
        der = energy_derived(text)
    except Exception:
        return value
    if der is None:
        return value
    if value is None:
        return float(der)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(der)
    if abs(v - der) > max(1.0, 0.02 * der):
        return float(der)
    return value


# One reading pair: "PREVIOUS [READING] … 899 … 963". The gap between the keyword
# and the first reading tolerates OCR debris and a stray single digit (the MF
# value, which OCR often drops onto the readings row); the readings themselves
# must be multi-digit. Between the two readings only whitespace or present /
# current keywords are allowed (strict on purpose: it keeps comma-separated
# prose and "previous balance" money-figures from forming false pairs — those
# bills fall back to the UNITS label / LLM instead).
_PAIR_RE = re.compile(
    r"previous\s*(?:reading\s*)?:?\s*(?:[^\d]|\b\d\b){0,80}?(?<![A-Za-z0-9])(\d{2,7})"
    r"(?:[^\d]{0,10}?(?:present|pres\.?|curr(?:ent)?)\s*(?:reading\s*)?:?\s*|\s+)"
    r"(?<![A-Za-z0-9])(\d{2,7})",
    re.I,
)


def meter_units(text: str) -> Optional[float]:
    """Billed units from meter readings: (present - previous) x MF.

    Returns a value only when exactly one reading pair is found (multi-register
    TOU bills are left to the LLM/regex). Guards reject page numbers and
    reversed/zero diffs.
    """
    text = normalize_bill_text(text)
    good = []
    for m in _PAIR_RE.finditer(text):
        try:
            prev, pres = int(m.group(1)), int(m.group(2))
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
        if label == "Total amount (Rs)":
            v = _total_amount(text)
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
    ce = crosscheck_energy(found.get("Energy charge (Rs)"), text)
    if ce is not None:
        found["Energy charge (Rs)"] = ce
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
                'Rules: read the CURRENT bill only — NEVER use the 12-month BILL HISTORY table, footer stubs, barcode '
                'numbers, or Urdu-paragraph figures (subsidy text) for any field. '
                'billing periods: from BILL MONTH (e.g. JUN 26 goes in both from and to); only an explicit from/to date '
                'range otherwise. NEVER use reading/issue/due dates as the period. '
                'total_units = current billed UNITS only (label UNITS / UNITS CONSUMED / TOTAL UNITS, or present-minus-'
                'previous meter readings times MF) — never meter READING values themselves. '
                'peak/offpeak/export_units = null unless TOU peak/off-peak or net-metering export lines are printed. '
                'total_amount = Grand Total, or Payable WITHIN due date when the grand-total figure is missing. NEVER '
                'Payable AFTER due date, never history/Urdu-paragraph numbers. '
                'energy_charge = Total/Variable Electricity Charges figure (the same-row number — ignore junk words '
                'between the label and the number). If that figure is garbled but Net Electricity Charges and Subsidies '
                'are printed, return their sum. '
                'wapda = the first (lower / within-due-tier) L.P. / WAPDA / F.C. / fuel surcharge amount, else null. '
                'taxes = tax amount (FBR/GST/income tax); ignore percentages like 15.99% and registration numbers like '
                'GST NO. fixed_charge = FIXED CHARGES line only, else null. '
                'Copy digits exactly as printed. NEVER guess or invent values — use null when a field is not printed.'
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
