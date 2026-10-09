from datetime import date

import pytest

from pashuplan import domain


def test_thi_known_value():
    # (1.8*30+32) - (0.55-0.0055*60)*(1.8*30-26) = 86 - 0.22*28
    assert domain.thi(30, 60) == pytest.approx(79.84, abs=0.01)


def test_thi_rises_with_temperature_and_humidity():
    assert domain.thi(35, 50) > domain.thi(25, 50)
    assert domain.thi(30, 80) > domain.thi(30, 40)


@pytest.mark.parametrize(
    "value,level",
    [(60, "none"), (71.9, "none"), (72, "mild"), (78.9, "mild"),
     (79, "moderate"), (88.9, "moderate"), (89, "severe")],
)
def test_heat_stress_level(value, level):
    assert domain.heat_stress_level(value) == level


def test_wood_curve_peaks_early_and_declines():
    peak = max(range(1, 400), key=domain.wood_yield)
    assert 40 <= peak <= 100
    assert domain.wood_yield(300) < domain.wood_yield(peak)
    assert domain.wood_yield(300) > 0


def test_wood_curve_rejects_non_positive_dim():
    with pytest.raises(ValueError):
        domain.wood_yield(0)


def test_fever_and_scc_thresholds():
    assert domain.is_fever(39.6)
    assert not domain.is_fever(38.6)
    assert domain.is_high_scc(250)
    assert not domain.is_high_scc(150)


def test_milk_discard_until_uses_withdrawal_period():
    out = domain.milk_discard_until(date(2026, 10, 5))
    assert (out - date(2026, 10, 5)).days == domain.MILK_WITHDRAWAL_DAYS
