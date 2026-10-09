"""Deterministic fact functions over FarmData.

Every function takes the farm first and returns plain JSON-safe dicts/lists
(floats, ints, bools, ISO date strings), so they can be handed to an LLM tool loop as-is.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from pashuplan import domain
from pashuplan.data_gen import FarmData

RECENT_DAYS_FOR_SIGNS = 3
EARLY_LACTATION_DIM = 100
LATE_LACTATION_DIM = 250
PREG_CHECK_WINDOW_DAYS = (35, 70)
CALVING_LOOKAHEAD_DAYS = 60
MAX_DETAIL_DAYS = 60


def _r(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d is not None and not pd.isna(d) else None


def _window_dates(farm: FarmData, window: int, periods_back: int = 0) -> list[date]:
    dates = sorted(farm.milk["date"].unique())
    end = len(dates) - periods_back * window
    return dates[max(0, end - window):end]


def _per_cow_avg(farm: FarmData, dates: list[date]) -> pd.Series:
    sub = farm.milk[farm.milk["date"].isin(dates)]
    return sub.groupby("cow_id")["litres"].mean()


def _pct(new: float, old: float) -> float:
    return 0.0 if old == 0 else (new - old) / old * 100


def milk_trend(farm: FarmData, window: int = 7) -> dict:
    daily = farm.milk.groupby("date")["litres"].sum()
    this = daily.loc[_window_dates(farm, window)].mean()
    prior = daily.loc[_window_dates(farm, window, 1)].mean()
    return {
        "window_days": window,
        "this_period_avg_l_per_day": _r(this),
        "prior_period_avg_l_per_day": _r(prior),
        "change_l_per_day": _r(this - prior),
        "change_pct": _r(_pct(this, prior)),
    }


def top_decliners(farm: FarmData, window: int = 7, n: int = 5) -> list[dict]:
    this = _per_cow_avg(farm, _window_dates(farm, window))
    prior = _per_cow_avg(farm, _window_dates(farm, window, 1))
    rows = [{"cow_id": cid, "prior_avg_litres": _r(prior[cid]), "this_avg_litres": _r(this[cid]),
             "change_pct": _r(_pct(this[cid], prior[cid]))} for cid in this.index]
    return sorted(rows, key=lambda r: r["change_pct"])[:max(1, n)]


def _treatment_for(farm: FarmData, cow_id: str) -> dict | None:
    t = farm.treatments
    if t.empty:
        return None
    rows = t[t["cow_id"] == cow_id]
    if rows.empty:
        return None
    last = rows.sort_values("date").iloc[-1]
    return {"date": _iso(last["date"]), "drug": last["drug"], "discard_until": _iso(last["discard_until"]),
            "under_treatment": bool(last["date"] <= farm.today < last["discard_until"])}


def _recent_signs(farm: FarmData, cow_id: str) -> tuple[float, float, float]:
    dates = sorted(farm.milk["date"].unique())[-RECENT_DAYS_FOR_SIGNS:]
    milk = farm.milk[(farm.milk["cow_id"] == cow_id) & farm.milk["date"].isin(dates)]
    vit = farm.vitals[(farm.vitals["cow_id"] == cow_id) & farm.vitals["date"].isin(dates)]
    return milk["scc"].mean(), vit["temp_c"].max(), vit["rumination_min"].mean()


def mastitis_suspects(farm: FarmData) -> list[dict]:
    out = []
    for cow_id in farm.cows["cow_id"]:
        scc, max_temp, rumination = _recent_signs(farm, cow_id)
        if not (domain.is_high_scc(scc) and domain.is_fever(max_temp)):
            continue
        tr = _treatment_for(farm, cow_id)
        out.append({
            "cow_id": cow_id,
            "avg_scc_3d": _r(scc),
            "max_temp_3d": _r(max_temp, 2),
            "avg_rumination_3d": _r(rumination, 0),
            "under_treatment": bool(tr and tr["under_treatment"]),
            "discard_until": tr["discard_until"] if tr else None,
        })
    return sorted(out, key=lambda r: -r["avg_scc_3d"])


def _stage(dim: int) -> str:
    if dim < EARLY_LACTATION_DIM:
        return "early"
    return "late" if dim > LATE_LACTATION_DIM else "mid"


def cow_detail(farm: FarmData, cow_id: str, days: int = 14) -> dict:
    match = farm.cows[farm.cows["cow_id"] == cow_id]
    if match.empty:
        raise ValueError(f"unknown cow_id {cow_id!r}")
    cow = match.iloc[0]
    days = max(1, min(int(days), MAX_DETAIL_DAYS))
    dim = int(cow["dim"])
    milk = farm.milk[farm.milk["cow_id"] == cow_id].set_index("date")
    vit = farm.vitals[farm.vitals["cow_id"] == cow_id].set_index("date")
    dates = sorted(milk.index)[-days:]
    daily = [{"date": _iso(d), "litres": _r(milk.loc[d, "litres"]), "scc": _r(milk.loc[d, "scc"]),
              "temp_c": _r(vit.loc[d, "temp_c"], 2), "rumination_min": _r(vit.loc[d, "rumination_min"], 0)}
             for d in dates]
    scc, max_temp, _ = _recent_signs(farm, cow_id)
    this = _per_cow_avg(farm, _window_dates(farm, 7))[cow_id]
    prior = _per_cow_avg(farm, _window_dates(farm, 7, 1))[cow_id]
    weekly_decline = max(0.0, (1 - domain.wood_yield(dim + 7) / domain.wood_yield(dim)) * 100)
    return {
        "cow_id": cow_id,
        "breed": cow["breed"],
        "parity": int(cow["parity"]),
        "days_in_milk": dim,
        "lactation_stage": _stage(dim),
        "pregnant": bool(cow["pregnant"]),
        "expected_weekly_decline_pct": _r(weekly_decline),
        "avg_litres_this_week": _r(this),
        "avg_litres_prior_week": _r(prior),
        "change_pct": _r(_pct(this, prior)),
        "has_mastitis_signs": bool(domain.is_high_scc(scc) and domain.is_fever(max_temp)),
        "treatment": _treatment_for(farm, cow_id),
        "daily": daily,
    }


def feed_quality(farm: FarmData) -> dict:
    feed = farm.feed.sort_values("date").reset_index(drop=True)
    changed = feed["batch_id"] != feed["batch_id"].shift()
    changed.iloc[0] = False
    out = {"target_cp": domain.TARGET_CRUDE_PROTEIN_PCT, "change_detected": bool(changed.any())}
    if not changed.any():
        cp = feed["crude_protein_pct"].mean()
        return {**out, "change_date": None, "cp_before": _r(cp, 2), "cp_after": _r(cp, 2), "cp_delta": 0.0,
                "below_target": bool(cp < domain.TARGET_CRUDE_PROTEIN_PCT)}
    idx = int(changed.idxmax())
    before, after = feed.iloc[:idx], feed.iloc[idx:]
    return {
        **out,
        "change_date": _iso(feed.loc[idx, "date"]),
        "batch_before": before.iloc[-1]["batch_id"],
        "batch_after": after.iloc[0]["batch_id"],
        "cp_before": _r(before["crude_protein_pct"].mean(), 2),
        "cp_after": _r(after["crude_protein_pct"].mean(), 2),
        "cp_delta": _r(after["crude_protein_pct"].mean() - before["crude_protein_pct"].mean(), 2),
        "below_target": bool(after["crude_protein_pct"].mean() < domain.TARGET_CRUDE_PROTEIN_PCT),
        "cost_per_kg_before": _r(before["cost_per_kg"].mean(), 2),
        "cost_per_kg_after": _r(after["cost_per_kg"].mean(), 2),
        "days_since_change": len(after),
    }


def heat_stress(farm: FarmData, window: int = 7) -> dict:
    recent = farm.weather.sort_values("date").tail(window)
    hot = recent[recent["thi"] >= domain.THI_MILD]
    return {
        "window_days": window,
        "days_at_or_above_mild": len(hot),
        "hot_dates": [_iso(d) for d in hot["date"]],
        "max_thi": _r(recent["thi"].max()),
        "avg_thi": _r(recent["thi"].mean()),
        "avg_temp_c": _r(recent["temp_c"].mean()),
        "worst_level": domain.heat_stress_level(recent["thi"].max()),
    }


def breeding_status(farm: FarmData) -> dict:
    cows, today = farm.cows, farm.today
    open_ = cows[~cows["pregnant"]]
    lo, hi = PREG_CHECK_WINDOW_DAYS
    since_ai = open_["last_ai_date"].map(lambda d: None if d is None or pd.isna(d) else (today - d).days)
    due_check = open_[since_ai.map(lambda n: n is not None and lo <= n <= hi)]
    calving = cows[cows["pregnant"]]
    soon = calving[calving["expected_calving_date"].map(lambda d: (d - today).days <= CALVING_LOOKAHEAD_DAYS)]
    return {
        "pregnant_count": int(cows["pregnant"].sum()),
        "open_count": len(open_),
        "overdue_open_cows": sorted(open_.loc[open_["dim"] > domain.MAX_DAYS_OPEN, "cow_id"]),
        "due_for_pregnancy_check": sorted(due_check["cow_id"]),
        "not_yet_bred": sorted(open_.loc[(open_["dim"] > domain.VOLUNTARY_WAIT_DAYS)
                                         & open_["last_ai_date"].isna(), "cow_id"]),
        "calving_next_60_days": sorted(soon["cow_id"]),
    }


def inventory_status(farm: FarmData) -> dict:
    items = []
    for _, r in farm.inventory.iterrows():
        items.append({
            "item": r["item"], "unit": r["unit"], "qty": _r(r["qty"]),
            "daily_use": _r(r["daily_use"]), "days_remaining": _r(r["qty"] / r["daily_use"]),
            "reorder_level": _r(r["reorder_level"]), "below_reorder_level": bool(r["qty"] < r["reorder_level"]),
            "unit_cost_inr": _r(r["unit_cost_inr"]),
        })
    return {"items": items, "low_stock": [i["item"] for i in items if i["below_reorder_level"]]}


def finance_summary(farm: FarmData, window: int = 7) -> dict:
    price = farm.config["milk_price_per_l"]
    n_cows = len(farm.cows)

    def period(periods_back: int) -> dict:
        dates = _window_dates(farm, window, periods_back)
        milk = farm.milk[farm.milk["date"].isin(dates)]
        sold = milk.loc[~milk["discarded"], "litres"].sum()
        discarded = milk.loc[milk["discarded"], "litres"].sum()
        feed = farm.feed[farm.feed["date"].isin(dates)]
        feed_cost = (feed["kg_per_cow"] * feed["cost_per_kg"]).sum() * n_cows
        vet = 0 if farm.treatments.empty else farm.treatments["date"].isin(dates).sum() * farm.config["vet_cost_per_treatment"]
        revenue = round(sold, 1) * price
        margin = revenue - feed_cost - vet - farm.config["labour_cost_per_day"] * len(dates)
        return {"revenue": revenue, "discarded": round(discarded, 1), "feed_cost": feed_cost,
                "vet_cost": float(vet), "margin": margin}

    now, before = period(0), period(1)
    return {
        "window_days": window,
        "milk_price_per_l": price,
        "milk_revenue_this_period_inr": _r(now["revenue"], 0),
        "milk_revenue_prior_period_inr": _r(before["revenue"], 0),
        "revenue_change_inr": _r(now["revenue"] - before["revenue"], 0),
        "discarded_milk_litres": now["discarded"],
        "discarded_milk_value_inr": _r(now["discarded"] * price, 0),
        "feed_cost_this_period_inr": _r(now["feed_cost"], 0),
        "feed_cost_prior_period_inr": _r(before["feed_cost"], 0),
        "vet_cost_this_period_inr": _r(now["vet_cost"], 0),
        "margin_this_period_inr": _r(now["margin"], 0),
        "margin_prior_period_inr": _r(before["margin"], 0),
    }
