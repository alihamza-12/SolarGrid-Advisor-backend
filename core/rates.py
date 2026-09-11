"""DISCO tariff rates (seeded, editable, persisted) — ported from app.py lines 888-896."""
from __future__ import annotations

from .config import DISCOS, RATES_FILE, SEED_RATES, load_json, save_json


def rates() -> dict:
    r = load_json(RATES_FILE, None)
    if not r:
        r = {d: dict(SEED_RATES[d], note="Sample default — verify against latest circular") for d in DISCOS}
        save_json(RATES_FILE, r)
    for d in DISCOS:
        if d not in r:
            r[d] = dict(SEED_RATES[d])
    return r
