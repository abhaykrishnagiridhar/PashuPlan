"""Feature tables shared by the two models. One row per cow per day.

`cow_day_frame` holds what the loss-attribution model needs. `watch_frame` holds the
early-warning features, which use only the days before each row (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pashuplan import domain
from pashuplan.data_gen import FarmData

FEATURE_WINDOW = 14  # days of history for each cow's own baseline
LITRES_WINDOW = 7
RISK_FEATURES = ["log_scc_ratio", "temp_delta", "rum_delta", "litres_ratio",
                 "log_scc_ratio_2d", "temp_delta_2d", "rum_delta_2d"]


def cow_day_frame(farm: FarmData) -> pd.DataFrame:
    frame = (farm.milk[["date", "cow_id", "litres", "scc"]]
             .merge(farm.vitals[["date", "cow_id", "temp_c", "rumination_min"]], on=["date", "cow_id"])
             .merge(farm.weather[["date", "thi"]], on="date")
             .merge(farm.feed[["date", "crude_protein_pct"]].rename(columns={"crude_protein_pct": "cp"}), on="date"))
    dim_today = farm.cows.set_index("cow_id")["dim"]
    frame["dim"] = frame["cow_id"].map(dim_today) - frame["date"].map(lambda d: (farm.today - d).days)
    frame["expected_litres"] = [domain.wood_yield(int(d)) for d in frame["dim"]]
    frame["heat_excess"] = (frame["thi"] - domain.THI_MILD).clip(lower=0)
    frame["cp_deficit"] = (domain.TARGET_CRUDE_PROTEIN_PCT - frame["cp"]).clip(lower=0)
    frame["sick_signal"] = (frame["scc"] > domain.SCC_MASTITIS_THRESHOLD).astype(int)
    return frame.sort_values(["cow_id", "date"]).reset_index(drop=True)


def _past_baseline(series: pd.Series, window: int, how: str) -> pd.Series:
    rolled = series.shift(1).rolling(window, min_periods=window)
    return rolled.median() if how == "median" else rolled.mean()


def watch_frame(farm: FarmData) -> pd.DataFrame:
    frame = cow_day_frame(farm)
    by_cow = frame.groupby("cow_id", sort=False)
    frame["log_scc_ratio"] = np.log(frame["scc"] / by_cow["scc"].transform(lambda s: _past_baseline(s, FEATURE_WINDOW, "median")))
    frame["temp_delta"] = frame["temp_c"] - by_cow["temp_c"].transform(lambda s: _past_baseline(s, FEATURE_WINDOW, "median"))
    frame["rum_delta"] = frame["rumination_min"] - by_cow["rumination_min"].transform(
        lambda s: _past_baseline(s, FEATURE_WINDOW, "median"))
    frame["litres_ratio"] = frame["litres"] / by_cow["litres"].transform(lambda s: _past_baseline(s, LITRES_WINDOW, "mean"))
    ready = frame.dropna(subset=["log_scc_ratio", "temp_delta", "rum_delta", "litres_ratio"]).copy()
    # Compare each cow with her herd mates the same day, so heat waves and feed changes cancel out
    for name in ("log_scc_ratio", "temp_delta", "rum_delta", "litres_ratio"):
        ready[name] = ready[name] - ready.groupby("date")[name].transform("median")
    by_cow = ready.groupby("cow_id", sort=False)
    for name in ("log_scc_ratio", "temp_delta", "rum_delta"):
        ready[f"{name}_2d"] = by_cow[name].transform(lambda s: s.rolling(2, min_periods=1).mean())
    return ready.reset_index(drop=True)
