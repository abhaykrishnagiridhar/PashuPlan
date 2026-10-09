"""Dairy domain constants and formulas shared by the data generator and the analytics."""
from __future__ import annotations

import math
from datetime import date, timedelta

# Temperature-humidity index bands (dairy cattle heat stress)
THI_MILD = 72
THI_MODERATE = 79
THI_SEVERE = 89

# Health thresholds
FEVER_C = 39.5
SCC_MASTITIS_THRESHOLD = 200  # thousand cells per mL
BASELINE_TEMP_C = 38.6
BASELINE_RUMINATION_MIN = 450

# Antibiotic withdrawal: milk is discarded for this many days after treatment
MILK_WITHDRAWAL_DAYS = 5

# Feed
TARGET_CRUDE_PROTEIN_PCT = 16.0

# Breeding
VOLUNTARY_WAIT_DAYS = 60
MAX_DAYS_OPEN = 120

# Wood's lactation curve: y = a * t^b * exp(-c * t), t = days in milk
WOOD_A = 12.6
WOOD_B = 0.20
WOOD_C = 0.003


def thi(temp_c: float, rel_humidity_pct: float) -> float:
    """Temperature-humidity index from air temperature (C) and relative humidity (%)."""
    return (1.8 * temp_c + 32) - (0.55 - 0.0055 * rel_humidity_pct) * (1.8 * temp_c - 26)


def heat_stress_level(thi_value: float) -> str:
    if thi_value >= THI_SEVERE:
        return "severe"
    if thi_value >= THI_MODERATE:
        return "moderate"
    if thi_value >= THI_MILD:
        return "mild"
    return "none"


def wood_yield(dim: float, a: float = WOOD_A, b: float = WOOD_B, c: float = WOOD_C) -> float:
    """Expected daily litres for an average cow at the given days in milk."""
    if dim <= 0:
        raise ValueError("days in milk must be positive")
    return a * dim**b * math.exp(-c * dim)


def is_fever(temp_c: float) -> bool:
    return temp_c >= FEVER_C


def is_high_scc(scc_thousand_per_ml: float) -> bool:
    return scc_thousand_per_ml > SCC_MASTITIS_THRESHOLD


def milk_discard_until(treatment_date: date) -> date:
    return treatment_date + timedelta(days=MILK_WITHDRAWAL_DAYS)
