import pandas as pd
import pytest

from pashuplan import domain
from pashuplan.data_gen import generate_farm


@pytest.fixture(scope="module")
def farm():
    return generate_farm(seed=42)


def _daily(farm):
    return farm.milk.groupby("date")["litres"].sum().sort_index()


def test_deterministic_for_same_seed():
    a, b = generate_farm(seed=7), generate_farm(seed=7)
    pd.testing.assert_frame_equal(a.milk, b.milk)
    assert a.ground_truth == b.ground_truth


def test_different_seed_changes_data():
    assert not generate_farm(seed=1).milk["litres"].equals(generate_farm(seed=2).milk["litres"])


def test_shapes(farm):
    assert len(farm.cows) == 40
    assert farm.milk["date"].nunique() == 60
    assert len(farm.milk) == 40 * 60
    assert farm.milk["date"].max() == farm.today
    for table in (farm.vitals, farm.weather, farm.feed, farm.treatments, farm.inventory):
        assert isinstance(table, pd.DataFrame)


def test_ground_truth_describes_planted_causes(farm):
    gt = farm.ground_truth
    assert len(gt["mastitis_cows"]) == 3
    assert set(gt["mastitis_cows"]) <= set(farm.cows["cow_id"])
    assert {"feed_change_date", "heat_wave_dates", "late_lactation_cows"} <= set(gt)


def test_herd_yield_drops_8_to_25_percent_week_on_week(farm):
    daily = _daily(farm)
    this_week, prior = daily.iloc[-7:].mean(), daily.iloc[-14:-7].mean()
    drop = (prior - this_week) / prior
    assert 0.08 <= drop <= 0.25


def test_mastitis_cows_have_high_scc_recently(farm):
    dates = sorted(farm.milk["date"].unique())[-3:]
    scc = farm.milk[farm.milk["date"].isin(dates)].groupby("cow_id")["scc"].mean()
    ill = farm.ground_truth["mastitis_cows"]
    assert all(scc[c] > domain.SCC_MASTITIS_THRESHOLD for c in ill)
    healthy = scc.drop(ill)
    assert (healthy <= domain.SCC_MASTITIS_THRESHOLD).mean() >= 0.95


def test_feed_protein_drops_at_change_date(farm):
    change = farm.ground_truth["feed_change_date"]
    before = farm.feed[farm.feed["date"] < change]["crude_protein_pct"].mean()
    after = farm.feed[farm.feed["date"] >= change]["crude_protein_pct"].mean()
    assert before - after >= 2.0


def test_heat_wave_is_hot_and_earlier_days_are_not(farm):
    wave = farm.ground_truth["heat_wave_dates"]
    w = farm.weather.set_index("date")
    assert all(w.loc[d, "thi"] >= 79 for d in wave)
    earlier = w[w.index < min(wave)]["thi"]
    assert (earlier < domain.THI_MILD).mean() >= 0.7


def test_late_lactation_red_herring_exists(farm):
    late = farm.ground_truth["late_lactation_cows"]
    assert len(late) >= 4
    dim = farm.cows.set_index("cow_id").loc[late, "dim"]
    assert (dim > 250).all()


def test_antibiotic_stock_is_low(farm):
    row = farm.inventory.set_index("item").loc["mastitis_tubes"]
    assert row["qty"] / row["daily_use"] < 7
