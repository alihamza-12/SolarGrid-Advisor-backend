"""Bill field extraction (LLM-first + regex backup) — ported from app.py lines 952-993.

Pakistani DISCO bills (IESCO/LESCO/…) mix English labels with Urdu (RTL) fragments,
print labels/reference numbers with letter-spacing ("F I X E D", "08 E 1 R 4 I 5 D")
and lay values out in columns. normalize_bill_text() cleans the first two problems
before any pattern runs; main.py runs the LLM first (it reads messy layouts best)
and uses these patterns to backfill whatever the LLM missed.

The LLM prompt is deliberately small-model friendly (short rules + a worked
example instead of long prose) because bill extraction usually runs on free /
low-tier models; sanitize_llm_bill() then coerces whatever JSON comes back into
clean values, and strong_periods() / cross-checks overrule the model whenever
the bill's own explicit labels disagree with it.

Missing fields stay None ("not on this bill") — values are never invented.
Only exception: export/fixed become 0 when the bill shows no such concept at
all (single unidirectional meter exports nothing; no fixed line means Rs 0
fixed) — see absence_zeros(). A TOU peak/off-peak split is never defaulted.
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
# Junk between a same-line label and its value: parenthetical remarks first
# ("(see note 2)", "(Rs.)" — tried before single chars so digits inside parens
# are skipped as a whole), then any non-digit chars (OCR debris like "zint",
# "cu'", "Rs.").
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
        # and sub-totals from being mistaken for the total.
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


# Absence detectors for zero-filling: when the bill contains NO mention at all
# of net-metering/export (a single unidirectional meter exports nothing) or of
# fixed/service charges (not charged), the honest extracted value is 0 — not a
# miss. Any mention (even value-less) disables the zero, so garbled-but-present
# lines still surface as null for the LLM/regex to resolve instead.
_ABSENCE_EXPORT_RE = re.compile(
    r"\bexport\w*|\bnet[\s-]*metering|\bnet\s*meter\b|bidirectional|"
    r"\bexcess\s*(?:units|kwh)|\bunits?\s*exported|\bimport\s*(?:units|kwh)",
    re.I,
)
_ABSENCE_FIXED_RE = re.compile(
    r"\bfixed\b|\bservice\s*charges?\b|\bdemand\s*charges?\b", re.I
)


def absence_zeros(text: str) -> dict:
    """0.0 for export/fixed when the bill shows no such concept at all.

    Peak/off-peak are deliberately NEVER zero-filled: a TOU split cannot be
    defaulted (0 + 0 != total units), so single-rate bills keep honest nulls.
    """
    t = normalize_bill_text(text)
    out: dict = {}
    if not _ABSENCE_EXPORT_RE.search(t):
        out["Export units (net metering)"] = 0.0
    if not _ABSENCE_FIXED_RE.search(t):
        out["Fixed charge (Rs)"] = 0.0
    return out


# Explicit period labels. When one of these matches, its value overrules the LLM
# (small models love to substitute reading/issue/due dates for the period).
_STRONG_MONTH_RE = re.compile(
    r"bill(?:ing)?\s*month\b" + _MONTH_GAP + _MONTHS + r"\s*[- ]?(\d{2,4})", re.I
)
_STRONG_RANGE_FROM_RE = re.compile(
    r"billing\s+period[^\n]{0,60}?from\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})", re.I
)
_STRONG_RANGE_TO_RE = re.compile(
    r"billing\s+period[^\n]{0,60}?to\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})", re.I
)


# Full day-dates: "17 JUN 26" or "17/06/2026" (real month names only).
_FULLDAY = r"\d{1,2}\s+" + _MONTHS + r"\s*\d{2,4}"
_FULLSLASH = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
# Gap between a DATE label and its value: debris, stray singles (column spill)
# and long ID runs (consumer/reference numbers the next line starts with) —
# but never across a "due date" (the due date must not become the period).
_DATE_GAP = r"(?:(?!\bdue[\s\-\u2013\u2014]*date\b)(?:[^\d]|\b\d\b|\b\d{3,}\b)){0,120}?"
_STRONG_DATEVAL_RE = re.compile(
    r"(?<!due[\s\-\u2013\u2014])(?<!update[\s\-\u2013\u2014])\bdate\b" + _DATE_GAP
    + r"((?:" + _FULLDAY + r")|(?:" + _FULLSLASH + r"))",
    re.I,
)
_FULLDATE_RE = re.compile(r"(?:" + _FULLDAY + r")|(?:" + _FULLSLASH + r")", re.I)


def strong_periods(text: str) -> tuple[Optional[str], Optional[str]]:
    """Billing period from the bill's own date labels, else (None, None).

    Precedence: an explicit "billing period from X to Y" range first, then the
    READING DATE -> next date (normally the ISSUE DATE), then BILL MONTH in
    both slots. The due date is never used as a period bound.
    """
    t = normalize_bill_text(text)
    f = _STRONG_RANGE_FROM_RE.search(t)
    e = _STRONG_RANGE_TO_RE.search(t)
    if f and e:
        return f.group(1), e.group(1)
    m = _STRONG_DATEVAL_RE.search(t)
    if m:
        frm = m.group(1)
        m2 = _FULLDATE_RE.search(t, m.end())
        return frm, m2.group(0) if m2 else frm
    m = _STRONG_MONTH_RE.search(t)
    if m:
        v = f"{m.group(1)} {m.group(2)}"
        return v, v
    return None, None


# Few-shot system prompt, tuned for small / free-tier models: short numbered
# rules plus one worked example (examples teach weak models far more reliably
# than long rule prose). Kept compact so it also fits small context windows.
_BILL_SYSTEM = """You read Pakistani electricity bills and output JSON. Output ONLY a raw JSON object (no markdown fences, no explanation) with EXACTLY these keys:
{"billing_period_from": "...", "billing_period_to": "...", "total_units": ..., "peak_units": ..., "offpeak_units": ..., "export_units": ..., "fixed_charge": ..., "total_amount": ..., "energy_charge": ..., "wapda": ..., "taxes": ...}
Rules:
1. Numbers are plain (1345, not "1,345"). Never invent values.
2. A field not printed on the bill is null — except export_units and fixed_charge, which are 0 when the bill has no export section / fixed-charge line at all.
3. billing_period_from is the READING DATE and billing_period_to is the ISSUE DATE when the bill prints them (example: 17 JUN 26 and 27 JUN 26). Otherwise BILL MONTH goes in both fields. Never use the due date as a period.
4. total_amount is the Grand Total (never Payable AFTER due date).
5. total_units is the billed UNITS (never meter READING numbers, never the BILL HISTORY table). peak_units and offpeak_units stay null unless the bill prints TOU peak/off-peak lines.
6. Ignore the 12-month BILL HISTORY table, the barcode line, and Urdu paragraphs.
Example bill:
BILL MONTH JUN 26
READING DATE 17 JUN 26 ISSUE DATE 27 JUN 26 DUE DATE 07 JUL 26
MF 1 PREVIOUS READING 899 PRESENT READING 963 UNITS 64
Total Electricity Charges 2635 Subsidies 1505
Net Electricity Charges 84.01 % 1130
Taxes 15.99 % 215 Total FPA 98 Current Bill 1247
Grand Total 1345
L.P. SURCHARGE 53 PAYABLE AFTER DUE DATE Till 1398 After 1450
Example output:
{"billing_period_from": "17 JUN 26", "billing_period_to": "27 JUN 26", "total_units": 64, "peak_units": null, "offpeak_units": null, "export_units": 0, "fixed_charge": 0, "total_amount": 1345, "energy_charge": 2635, "wapda": 53, "taxes": 215}"""

_BILL_USER_PREFIX = (
    "Extract the 11 fields from this bill. billing_period_from is the READING DATE, billing_period_to is the ISSUE DATE "
    "fields. Output ONLY the JSON object, no other text:\n\n"
)


def _parse_llm_json(out: str) -> Optional[dict]:
    """Parse a model's reply into a dict, tolerating weak-model formatting.

    Handles markdown fences, trailing commas and all-single-quote JSON; gives
    up (None) on anything else so the regex layer takes over.
    """
    if not out:
        return None
    t = out.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    frag = m.group(0)
    try:
        d = json.loads(frag)
        return d if isinstance(d, dict) else None
    except Exception:
        pass
    try:
        d = json.loads(re.sub(r",\s*([}\]])", r"\1", frag))
        return d if isinstance(d, dict) else None
    except Exception:
        pass
    if "'" in frag and '"' not in frag:
        try:
            py = frag.replace("'", '"')
            py = re.sub(r"\bNone\b", "null", py)
            py = re.sub(r"\bTrue\b", "true", py)
            py = re.sub(r"\bFalse\b", "false", py)
            d = json.loads(py)
            return d if isinstance(d, dict) else None
        except Exception:
            pass
    return None


_LLM_NUM_KEYS = {
    "total_units", "peak_units", "offpeak_units", "export_units",
    "fixed_charge", "total_amount", "energy_charge", "wapda", "taxes",
}
_LLM_PERIOD_KEYS = ("billing_period_from", "billing_period_to")
_PERIOD_RE = re.compile(
    r"^(?:[A-Za-z]{3,9}\s+\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})$"
)
_RSCURRENCY_RE = re.compile(r"\b(rs\.?|pkr|rupees?)")


def sanitize_llm_bill(d: dict) -> dict:
    """Coerce a (possibly weak) model's JSON into clean field values.

    Numbers: accepts real numbers plus strings like "1,345 Rs" / "Rs1345";
    anything with other words ("about 200"), booleans, negatives or NaN becomes
    missing so the regex layer backfills it. Periods: only strict shapes
    ("JUN 26", "17 JUN 26", "01/08/2026") survive, tidied to upper case.
    Unknown keys are dropped.
    """
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    for k in _LLM_NUM_KEYS:
        v = d.get(k)
        if v is None or v == "" or isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            f = float(v)
        elif isinstance(v, str):
            s = _RSCURRENCY_RE.sub("", v.strip().lower())
            if re.search(r"[a-z]", s):
                continue
            m = re.search(r"-?\d[\d,]*(?:\.\d+)?", s)
            if not m:
                continue
            try:
                f = float(m.group(0).replace(",", ""))
            except ValueError:
                continue
        else:
            continue
        if f != f or f in (float("inf"), float("-inf")) or f < 0:
            continue
        out[k] = f
    for k in _LLM_PERIOD_KEYS:
        v = d.get(k)
        if not isinstance(v, str):
            continue
        s = re.sub(r"([A-Za-z])\s*-\s*(\d)", r"\1 \2", v.strip().upper())
        s = re.sub(r"\s+", " ", s).strip(" .,;:")
        if _PERIOD_RE.match(s):
            out[k] = s
    return out


def llm_extract_bill(client_pack, text: str) -> Optional[dict]:
    if client_pack is None:
        return None
    try:
        # Temperature 0: extraction must be deterministic, never creative.
        # (get_llm packs are (client, model, temperature) tuples.)
        pack = (client_pack[0], client_pack[1], 0)
    except Exception:
        pack = client_pack
    try:
        out = llm_chat(pack, [
            {"role": "system", "content": _BILL_SYSTEM},
            {"role": "user", "content": _BILL_USER_PREFIX + normalize_bill_text(text)[:8000]},
        ])
        return _parse_llm_json(out)
    except Exception:
        return None


LABEL_MAP = {
    "billing_period_from": "Billing period (from)", "billing_period_to": "Billing period (to)",
    "total_units": "Total units", "peak_units": "Peak units", "offpeak_units": "Off-peak units",
    "export_units": "Export units (net metering)", "fixed_charge": "Fixed charge (Rs)",
    "total_amount": "Total amount (Rs)", "energy_charge": "Energy charge (Rs)",
    "wapda": "WAPDA / surcharge (Rs)", "taxes": "Taxes (FBR/PTA) (Rs)",
}
