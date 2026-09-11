"""Solar & savings calculators (pure functions) — ported from app.py lines 999-1070."""
from __future__ import annotations

import json
from typing import Optional

from .llm import llm_chat
from .utils import pkrs


def savings_calc(units: float, peak_share: float, shift_frac: float, solar_kwp: float,
                  sun_hours: float, r: dict, self_use_share: float) -> dict:
    pr, opr, fixed, bb = r["peak"], r["offpeak"], r["fixed"], r["buyback"]
    peak_u = units * peak_share
    off_u = units - peak_u
    shifted = peak_u * shift_frac
    current = peak_u * pr + off_u * opr + fixed
    shifted_bill = (peak_u - shifted) * pr + (off_u + shifted) * opr + fixed
    savings_shift = current - shifted_bill
    solar_monthly = solar_kwp * sun_hours * 30 * 0.8
    self_use = min(solar_monthly, units) * (self_use_share if units else 1.0)
    export = max(solar_monthly - min(solar_monthly, units), 0.0)
    credit = export * bb
    bill_with_solar = max(current - self_use * opr - credit, fixed)
    return {
        "peak_u": peak_u, "off_u": off_u, "shifted": shifted,
        "current": current, "shifted_bill": shifted_bill, "savings_shift": savings_shift,
        "solar_monthly": solar_monthly, "self_use": self_use, "export": export,
        "credit": credit, "bill_with_solar": bill_with_solar,
        "annual_savings": (savings_shift + (self_use * opr + credit)) * 12,
    }


def solar_sizing(daily_kwh: float, sun_hours: float) -> float:
    return max(0.5, round((daily_kwh * 1.25) / max(sun_hours, 2.0), 1))


def inverter_size(panel_kwp: float, peak_load_kw: float) -> float:
    return round(max(peak_load_kw * 1.25, panel_kwp * 0.85, 0.5), 1)


def backup_hours(battery_kwh: float, load_kw: float, dod: float = 0.8) -> float:
    return round((battery_kwh * dod) / max(load_kw, 0.1), 1)


def payback(total_cost: float, monthly_savings: float) -> float:
    """Payback period in YEARS. Returns None for infinite (never pays back)."""
    if monthly_savings <= 0:
        return None
    return round((total_cost / monthly_savings) / 12.0, 1)


def heuristic_plan(profile: dict, r: dict) -> list[dict]:
    kwp = profile.get("solar_kwp", 0) or 0
    batt = profile.get("battery_kwh", 0) or 0
    peak_appl = profile.get("peak_appliances", []) or []
    rows = [
        {"window": "6:00 – 9:30 AM", "period": "Off-peak", "action": "Run water heater, RO, washing machine, charging.", "why": f"Off-peak rate ≈ {pkrs(r['offpeak'])}/unit — cheapest grid window."},
        {"window": "10:00 AM – 2:00 PM", "period": "Solar peak", "action": "Run heavy loads (AC pre-cool, pool pump, EV, ironing) while the sun is out.", "why": f"Your {kwp} kWp system generates ≈ {pkrs((kwp * 4.5 * 0.8) * r['peak'])}/day worth of avoided peak charges." if kwp else "If you have solar, run heavy loads under the sun — you avoid the most expensive units."},
        {"window": "4:00 – 6:00 PM", "period": "Shoulder", "action": "Pre-cool rooms / start AC 30 min before peak. If battery < 80%, self-consume solar instead of exporting." if batt else "Pre-cool rooms / start AC 30 min before peak.", "why": "Avoids starting compressors at 6 PM when the rate is highest."},
        {"window": "6:00 – 10:00 PM", "period": "PEAK (expensive)", "action": "Essentials only. " + (f"Move these off-peak: {', '.join(peak_appl)}." if peak_appl else "No water heater, no iron, no laundry, no EV charging."), "why": f"Peak rate ≈ {pkrs(r['peak'])}/unit — every unit here costs ~{pkrs(max(r['peak'] - r['offpeak'], 0))} more than off-peak."},
        {"window": "10:00 PM – 6:00 AM", "period": "Off-peak", "action": "Laundry, water heater top-up, EV/battery charge from grid.", "why": "Cheapest window — ideal for timer-based appliances."},
    ]
    if batt:
        rows.append({"window": "Anytime", "period": "Battery", "action": "Keep battery discharge for peak hours; charge at night.", "why": f"{batt} kWh battery covers ~{backup_hours(batt, 1.0)} h at 1 kW load — use it to buy peak units at storage cost."})
    return rows


def llm_plan(client_pack, profile: dict, r: dict) -> Optional[str]:
    if client_pack is None:
        return None
    try:
        out = llm_chat(client_pack, [
            {"role": "system", "content": "You are a Pakistan energy-optimization advisor. Given a household profile, produce a concise Markdown plan: (1) a table of time windows (6-9:30 AM off-peak, 10-2 solar peak, 4-6 PM shoulder, 6-10 PM peak, 10 PM-6 AM off-peak) with specific appliance actions, (2) top 3 concrete savings actions, (3) an estimated monthly saving in PKR. Be practical. Rates: peak " + str(r['peak']) + " PKR/unit, off-peak " + str(r['offpeak']) + " PKR/unit, buyback " + str(r['buyback']) + " PKR/unit."},
            {"role": "user", "content": json.dumps(profile, ensure_ascii=False, indent=1)},
        ])
        return out or None
    except Exception:
        return None
