"""Seeded synthetic dairy farm with planted causes for a milk-yield drop.

Hero scenario (last 8 days of a 60-day window):
  * mastitis in 3 mid-lactation cows (high SCC, fever, lower rumination, treated later)
  * a cheaper, lower-protein feed batch from FEED_CHANGE_DAYS_AGO
  * a heat wave over the final HEAT_WAVE_DAYS days
  * red herring: late-lactation cows whose low output is normal, not a problem
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from pashuplan import domain

TODAY = date(2026, 10, 8)
FEED_CHANGE_DAYS_AGO = 8
HEAT_WAVE_DAYS = 7
MASTITIS_ONSET_DAYS_AGO = (8, 6, 5)  # onset offsets counted back from the last day + 1
LATE_LACTATION_COUNT = 6
TREATMENT_DELAY_DAYS = 3

CP_BEFORE = 16.6
CP_AFTER = 13.8
FEED_YIELD_LOSS_PER_CP_POINT = 0.025
HEAT_YIELD_LOSS_PER_THI_POINT = 0.005
MASTITIS_MAX_YIELD_LOSS = 0.40

BREEDS = ("HF cross", "Gir cross", "Jersey cross")


@dataclass(frozen=True)
class FarmData:
    today: date
    cows: pd.DataFrame
    milk: pd.DataFrame
    vitals: pd.DataFrame
    weather: pd.DataFrame
    feed: pd.DataFrame
    treatments: pd.DataFrame
    inventory: pd.DataFrame
    config: dict
    ground_truth: dict


def _make_cows(rng: np.random.Generator, n_cows: int, days: int, today: date) -> pd.DataFrame:
    late_idx = set(rng.choice(n_cows, size=LATE_LACTATION_COUNT, replace=False).tolist())
    rows = []
    for i in range(n_cows):
        dim = int(rng.integers(260, 331)) if i in late_idx else int(rng.integers(max(days + 10, 70), 251))
        pregnant = dim > 110 and rng.random() < 0.8
        last_ai = None
        calving = None
        if pregnant:
            last_ai = today - timedelta(days=int(rng.integers(5, max(6, dim - 70))))
            calving = last_ai + timedelta(days=283)
        elif dim > domain.VOLUNTARY_WAIT_DAYS and rng.random() < 0.5:
            last_ai = today - timedelta(days=int(rng.integers(21, 60)))
        parity = int(rng.integers(1, 6))
        factor = float(np.clip(rng.lognormal(0, 0.12) * (1 + 0.04 * (parity - 2)), 0.7, 1.35))
        rows.append({
            "cow_id": f"C{i + 1:02d}",
            "breed": BREEDS[int(rng.integers(0, len(BREEDS)))],
            "parity": parity,
            "dim": dim,
            "calving_date": today - timedelta(days=dim),
            "yield_factor": round(factor, 3),
            "pregnant": bool(pregnant),
            "last_ai_date": last_ai,
            "expected_calving_date": calving,
        })
    return pd.DataFrame(rows)


def _make_weather(rng: np.random.Generator, dates: list[date], wave: set[date]) -> pd.DataFrame:
    rows = []
    for d in dates:
        if d in wave:
            temp, rh = rng.normal(32.5, 0.8), rng.normal(58, 3)
        else:
            temp, rh = rng.normal(23.0, 1.8), rng.normal(55, 4)
        rows.append({"date": d, "temp_c": round(temp, 1), "rh_pct": round(rh, 1),
                     "thi": round(domain.thi(temp, rh), 1)})
    return pd.DataFrame(rows)


def _make_feed(rng: np.random.Generator, dates: list[date], change: date) -> pd.DataFrame:
    rows = []
    for d in dates:
        after = d >= change
        rows.append({
            "date": d,
            "batch_id": "FB-18" if after else "FB-17",
            "crude_protein_pct": round(rng.normal(CP_AFTER if after else CP_BEFORE, 0.08), 2),
            "kg_per_cow": 22.0,
            "cost_per_kg": 24.0 if after else 26.0,
        })
    return pd.DataFrame(rows)


def _make_inventory() -> pd.DataFrame:
    return pd.DataFrame([
        {"item": "concentrate_feed", "unit": "kg", "qty": 9000.0, "daily_use": 880.0,
         "reorder_level": 4400.0, "unit_cost_inr": 24.0},
        {"item": "mineral_mix", "unit": "kg", "qty": 60.0, "daily_use": 4.0,
         "reorder_level": 40.0, "unit_cost_inr": 90.0},
        {"item": "mastitis_tubes", "unit": "tubes", "qty": 12.0, "daily_use": 3.0,
         "reorder_level": 20.0, "unit_cost_inr": 150.0},
        {"item": "teat_dip", "unit": "litres", "qty": 20.0, "daily_use": 1.5,
         "reorder_level": 10.0, "unit_cost_inr": 120.0},
    ])


def generate_farm(seed: int = 42, n_cows: int = 40, days: int = 60) -> FarmData:
    rng = np.random.default_rng(seed)
    today = TODAY
    dates = [today - timedelta(days=days - 1 - i) for i in range(days)]
    feed_change = today - timedelta(days=FEED_CHANGE_DAYS_AGO - 1)
    wave = {today - timedelta(days=k) for k in range(HEAT_WAVE_DAYS)}

    cows = _make_cows(rng, n_cows, days, today)
    late = cows.loc[cows["dim"] > 250, "cow_id"].tolist()
    candidates = cows.loc[(cows["dim"] < 200) & (cows["yield_factor"] > 0.9), "cow_id"].tolist()
    ill = sorted(rng.choice(candidates, size=len(MASTITIS_ONSET_DAYS_AGO), replace=False).tolist())
    onset = {c: today - timedelta(days=k - 1) for c, k in zip(ill, MASTITIS_ONSET_DAYS_AGO)}

    weather = _make_weather(rng, dates, wave)
    thi_by_date = weather.set_index("date")["thi"].to_dict()
    feed = _make_feed(rng, dates, feed_change)
    feed_gap = (CP_BEFORE - CP_AFTER) * FEED_YIELD_LOSS_PER_CP_POINT

    treatments = pd.DataFrame([
        {"date": onset[c] + timedelta(days=TREATMENT_DELAY_DAYS), "cow_id": c,
         "drug": "intramammary antibiotic",
         "discard_until": domain.milk_discard_until(onset[c] + timedelta(days=TREATMENT_DELAY_DAYS))}
        for c in ill if onset[c] + timedelta(days=TREATMENT_DELAY_DAYS) <= today
    ])

    milk_rows, vital_rows = [], []
    for _, cow in cows.iterrows():
        cid = cow["cow_id"]
        for d in dates:
            dim = int(cow["dim"]) - (today - d).days
            since_onset = (d - onset[cid]).days if cid in onset else -1
            sick = since_onset >= 0
            heat_loss = HEAT_YIELD_LOSS_PER_THI_POINT * max(0.0, thi_by_date[d] - domain.THI_MILD)
            feed_ramp = min(1.0, max(0, (d - feed_change).days + 1) / 3)
            sick_loss = min(MASTITIS_MAX_YIELD_LOSS, 0.15 + 0.08 * since_onset) if sick else 0.0
            litres = (domain.wood_yield(dim) * cow["yield_factor"] * (1 + rng.normal(0, 0.03))
                      * (1 - heat_loss) * (1 - feed_gap * feed_ramp) * (1 - sick_loss))
            scc = float(np.exp(rng.normal(np.log(70), 0.35)))
            if sick:
                scc *= 1 + 6 * (1 - np.exp(-(since_onset + 1) / 2))
            discarded = bool(len(treatments) and (
                (treatments["cow_id"] == cid) & (treatments["date"] <= d) & (d < treatments["discard_until"])
            ).any())
            temp = domain.BASELINE_TEMP_C + rng.normal(0, 0.2) + (0.15 if d in wave else 0)
            rumination = domain.BASELINE_RUMINATION_MIN + rng.normal(0, 25) - (40 if d in wave else 0)
            if sick:
                temp += min(1.3, 0.6 + 0.2 * since_onset)
                rumination -= min(160, 60 + 25 * since_onset)
            milk_rows.append({"date": d, "cow_id": cid, "litres": round(max(litres, 0.0), 1),
                              "scc": round(scc, 1), "discarded": discarded})
            vital_rows.append({"date": d, "cow_id": cid, "temp_c": round(temp, 2),
                               "rumination_min": round(rumination, 0)})

    open_cows = cows.loc[(~cows["pregnant"]) & (cows["dim"] > domain.MAX_DAYS_OPEN), "cow_id"].tolist()
    return FarmData(
        today=today,
        cows=cows,
        milk=pd.DataFrame(milk_rows),
        vitals=pd.DataFrame(vital_rows),
        weather=weather,
        feed=feed,
        treatments=treatments,
        inventory=_make_inventory(),
        config={"milk_price_per_l": 42.0, "labour_cost_per_day": 2000.0, "vet_cost_per_treatment": 600.0},
        ground_truth={
            "mastitis_cows": ill,
            "mastitis_onset": onset,
            "feed_change_date": feed_change,
            "heat_wave_dates": sorted(wave),
            "late_lactation_cows": late,
            "overdue_breeding_cows": open_cows,
        },
    )
