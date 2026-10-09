"""Loss attribution: how many litres a day does each cause cost?

A small regression written from scratch with NumPy. For every cow-day it explains
log(litres / lactation-curve expectation) with three observable signals (heat above the THI
threshold, protein below target, high SCC) plus one intercept per cow, so a high or low producer
is not mistaken for a sick one. The fitted effects then answer a counterfactual: how much more
milk would there have been without this cause?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pashuplan import features
from pashuplan.data_gen import FarmData

FACTORS = {"heat": "heat_excess", "feed": "cp_deficit", "sick": "sick_signal"}
RIDGE = 1e-6


def _design(frame: pd.DataFrame, cows: list[str]) -> np.ndarray:
    cow_dummies = (frame["cow_id"].to_numpy()[:, None] == np.array(cows)[None, :]).astype(float)
    signals = frame[list(FACTORS.values())].to_numpy(dtype=float)
    return np.hstack([signals, cow_dummies])


class LossAttribution:
    def __init__(self, coefs: dict[str, float]):
        self.coefs = coefs

    @classmethod
    def fit(cls, farms: list[FarmData]) -> "LossAttribution":
        if not farms:
            raise ValueError("need at least one farm to train on")
        rows = []
        for i, farm in enumerate(farms):
            frame = features.cow_day_frame(farm)
            frame["cow_id"] = f"{i}:" + frame["cow_id"]
            rows.append(frame)
        data = pd.concat(rows, ignore_index=True)
        cows = sorted(data["cow_id"].unique())
        x = _design(data, cows)
        y = np.log(data["litres"].clip(lower=0.1) / data["expected_litres"]).to_numpy()
        gram = x.T @ x + RIDGE * np.eye(x.shape[1])
        beta = np.linalg.solve(gram, x.T @ y)
        # the model is log(litres) = ... - effect * signal, so a cost is the negative coefficient
        return cls({name: float(-beta[i]) for i, name in enumerate(FACTORS)})

    def _loss_per_day(self, farm_frame: pd.DataFrame, dates: list, factor: str) -> float:
        rows = farm_frame[farm_frame["date"].isin(dates)]
        signal = rows[FACTORS[factor]].to_numpy(dtype=float)
        without = rows["litres"].to_numpy() * np.exp(self.coefs[factor] * signal)
        return float((without - rows["litres"].to_numpy()).sum() / len(dates))

    def explain(self, farm: FarmData, window: int = 7) -> dict:
        frame = features.cow_day_frame(farm)
        dates = sorted(frame["date"].unique())
        this, prior = dates[-window:], dates[-2 * window:-window]
        causes = {f: round(self._loss_per_day(frame, this, f) - self._loss_per_day(frame, prior, f), 1)
                  for f in FACTORS}
        daily = frame.groupby("date")["litres"].sum()
        observed = float(daily.loc[prior].mean() - daily.loc[this].mean())
        explained = round(sum(causes.values()), 1)
        return {"window_days": window, "observed_drop_l_per_day": round(observed, 1), "causes": causes,
                "explained_l_per_day": explained, "unexplained_l_per_day": round(observed - explained, 1)}

    def to_dict(self) -> dict:
        return {"kind": "loss_attribution", "coefs": self.coefs}

    @classmethod
    def from_dict(cls, data: dict) -> "LossAttribution":
        return cls({k: float(v) for k, v in data["coefs"].items()})
