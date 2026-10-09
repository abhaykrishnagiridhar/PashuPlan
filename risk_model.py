"""Mastitis early-warning model: which cows are about to fall sick?

Logistic regression written from scratch with NumPy. Each cow-day is described by how far that
cow's SCC, temperature, rumination and milk have moved from her OWN recent baseline. The label
is "she will show mastitis within the next two days", learned only from cow-days that look
healthy today, so it has to find the quiet early signs the plain SCC-plus-fever rule misses.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from pashuplan import features
from pashuplan.data_gen import FarmData

HORIZON_DAYS = 2
LEARNING_RATE = 0.5
ITERATIONS = 1500
L2 = 1e-3
MAX_POSITIVE_WEIGHT = 25.0
WATCH_LOOKBACK_DAYS = 14
ALERT_BUDGET_PER_COW_WEEK = 0.10  # about four udder checks a week on a 40-cow farm


def _onsets(farm: FarmData) -> dict[str, date]:
    gt = farm.ground_truth
    return {**gt["mastitis_onset"], **gt.get("at_risk_onset", {})}


def _labelled(farm: FarmData) -> pd.DataFrame:
    """Warning-feature rows with label 1 (onset within HORIZON days) or 0, minus cows already sick."""
    frame = features.watch_frame(farm)
    onsets = _onsets(farm)
    days_to_onset = frame.apply(lambda r: (onsets[r["cow_id"]] - r["date"]).days if r["cow_id"] in onsets else 999,
                                axis=1)
    frame = frame[days_to_onset >= 1].copy()
    frame["label"] = (days_to_onset[days_to_onset >= 1] <= HORIZON_DAYS).astype(int)
    return frame


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class MastitisRiskModel:
    def __init__(self, weights, bias, mean, std, threshold):
        self.weights, self.bias = np.asarray(weights, float), float(bias)
        self.mean, self.std, self.threshold = np.asarray(mean, float), np.asarray(std, float), float(threshold)

    @classmethod
    def fit(cls, farms: list[FarmData]) -> "MastitisRiskModel":
        if not farms:
            raise ValueError("need at least one farm to train on")
        data = pd.concat([_labelled(f) for f in farms], ignore_index=True)
        x_raw, y = data[features.RISK_FEATURES].to_numpy(float), data["label"].to_numpy(float)
        mean, std = x_raw.mean(axis=0), x_raw.std(axis=0) + 1e-9
        x = (x_raw - mean) / std
        pos_weight = min(MAX_POSITIVE_WEIGHT, (y == 0).sum() / max((y == 1).sum(), 1))
        sample_w = np.where(y == 1, pos_weight, 1.0)
        w, b = np.zeros(x.shape[1]), 0.0
        for _ in range(ITERATIONS):
            err = (_sigmoid(x @ w + b) - y) * sample_w
            w -= LEARNING_RATE * (x.T @ err / sample_w.sum() + L2 * w)
            b -= LEARNING_RATE * err.sum() / sample_w.sum()
        model = cls(w, b, mean, std, 0.5)
        model.threshold = model._budget_threshold(x_raw)
        return model

    def _score(self, x_raw: np.ndarray) -> np.ndarray:
        return _sigmoid(((x_raw - self.mean) / self.std) @ self.weights + self.bias)

    def _budget_threshold(self, x_raw: np.ndarray) -> float:
        """Lowest threshold whose alert rate on the training data stays within the farmer's alert budget.

        A flag costs a farmer a udder check, so the budget is a farm decision: catch as many
        early cases as possible without sending them to check more than about one cow in ten a week.
        """
        scores, cow_weeks = self._score(x_raw), len(x_raw) / 7
        for t in np.linspace(0.05, 0.99, 95):
            if (scores >= t).sum() / cow_weeks <= ALERT_BUDGET_PER_COW_WEEK:
                return float(t)
        return 0.99

    def risk_table(self, farm: FarmData) -> pd.DataFrame:
        frame = features.watch_frame(farm)
        frame["risk"] = self._score(frame[features.RISK_FEATURES].to_numpy(float))
        return frame[["date", "cow_id", "risk", "sick_signal"]]

    def watch_list(self, farm: FarmData) -> list[dict]:
        """Cows that look healthy today but carry the early signs. Already-sick cows are left out."""
        table = self.risk_table(farm)
        today = table[(table["date"] == farm.today) & (table["sick_signal"] == 0)]
        flagged = today[today["risk"] >= self.threshold].sort_values("risk", ascending=False)
        return [{"cow_id": r.cow_id, "risk": round(float(r.risk), 3)} for r in flagged.itertuples()]

    def early_warnings(self, farm: FarmData) -> list[dict]:
        """For every cow that fell sick: the first day in the lookback the model raised the flag."""
        table = self.risk_table(farm)
        out = []
        for cow, onset in farm.ground_truth["mastitis_onset"].items():
            rows = table[(table["cow_id"] == cow) & (table["date"] < onset)
                         & (table["date"] >= onset - timedelta(days=WATCH_LOOKBACK_DAYS))]
            hits = rows[(rows["risk"] >= self.threshold) & (rows["sick_signal"] == 0)]
            first = hits["date"].min() if len(hits) else None
            out.append({"cow_id": cow, "onset": onset, "warned_on": first,
                        "lead_days": (onset - first).days if first is not None else 0})
        return out

    def evaluate(self, farms: list[FarmData]) -> dict:
        data = pd.concat([_labelled(f) for f in farms], ignore_index=True)
        scores, y = self._score(data[features.RISK_FEATURES].to_numpy(float)), data["label"].to_numpy()
        flagged = scores >= self.threshold
        tp = int((flagged & (y == 1)).sum())
        cow_weeks = len(data) / 7
        return {"auc": round(_auc(scores, y), 3), "n_positive": int((y == 1).sum()),
                "precision": round(tp / max(int(flagged.sum()), 1), 3),
                "recall": round(tp / max(int((y == 1).sum()), 1), 3),
                "alerts_per_cow_week": round(float(flagged.sum()) / cow_weeks, 3)}

    def to_dict(self) -> dict:
        return {"kind": "mastitis_risk", "weights": self.weights.tolist(), "bias": self.bias,
                "mean": self.mean.tolist(), "std": self.std.tolist(), "threshold": self.threshold,
                "features": features.RISK_FEATURES}

    @classmethod
    def from_dict(cls, data: dict) -> "MastitisRiskModel":
        return cls(data["weights"], data["bias"], data["mean"], data["std"], data["threshold"])
