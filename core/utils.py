"""Small pure utilities — ported from app.py lines 202-263."""
from __future__ import annotations

import datetime as dt
import re
from typing import Optional


def pkrs(x: float) -> str:
    return f"Rs {x:,.2f}"


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def parse_date(s: str) -> Optional[dt.date]:
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except Exception:
            pass
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None
    return None


MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def auto_detect_dates(text: str) -> dict:
    """Heuristically find issue / effective dates inside a PDF's text."""
    head = text[:6000]
    dates = []
    pat = re.compile(
        r"(?:(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}\s*,?\s*\d{4}"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
        re.I,
    )
    for m in pat.finditer(head):
        d = parse_date(m.group(0))
        if d and 2000 <= d.year <= dt.date.today().year + 1:
            dates.append((m.start(), m.group(0), d))

    issue = effective = None
    for idx, raw, d in dates:
        window = head[max(0, idx - 120):idx]
        low = window.lower()
        if effective is None and re.search(r"effect\w*\s+(from|with)|with\s+effect\s+from|effective\s+date", low):
            effective = raw
        elif issue is None and re.search(r"\bdate[ds]?\s*(of|:)?\s*$|dated|issue[dn]?", low[-40:]):
            issue = raw
    if not issue and dates:
        issue = dates[0][1]
    return {"issue": issue or "", "effective": effective or ""}
