import dataclasses
from datetime import timedelta

import numpy as np
import pytest

from pashuplan import domain, features


def test_cow_day_frame_has_one_row_per_cow_and_day(farm):
    f = features.cow_day_frame(farm)
    assert len(f) == 40 * 60
    assert {"date", "cow_id", "litres", "thi", "cp", "heat_excess", "cp_deficit", "sick_signal"} <= set(f.columns)
    assert not f[["litres", "thi", "cp"]].isna().any().any()


def test_heat_and_protein_features_are_zero_when_conditions_are_fine(farm):
    f = features.cow_day_frame(farm)
    cool = f[f["thi"] < domain.THI_MILD]
    assert (cool["heat_excess"] == 0).all()
    assert (f[f["cp"] >= domain.TARGET_CRUDE_PROTEIN_PCT]["cp_deficit"] == 0).all()
    hot = f[f["thi"] >= 80]
    assert (hot["heat_excess"] > 0).all()
    assert (f[f["date"] >= farm.ground_truth["feed_change_date"]]["cp_deficit"] > 1.5).all()


def test_sick_signal_marks_the_sick_cows_recently(farm):
    f = features.cow_day_frame(farm)
    last = f[f["date"] == farm.today].set_index("cow_id")["sick_signal"]
    assert set(last[last == 1].index) == set(farm.ground_truth["mastitis_cows"])


def test_watch_frame_skips_the_warmup_and_has_no_gaps(farm):
    w = features.watch_frame(farm)
    assert len(w) == 40 * (60 - features.FEATURE_WINDOW)
    assert not w[features.RISK_FEATURES].isna().any().any()
    assert np.isfinite(w[features.RISK_FEATURES].to_numpy()).all()


def test_watch_features_never_look_into_the_future(farm):
    cut = farm.today - timedelta(days=10)
    past = dataclasses.replace(
        farm,
        milk=farm.milk[farm.milk["date"] <= cut],
        vitals=farm.vitals[farm.vitals["date"] <= cut],
        weather=farm.weather[farm.weather["date"] <= cut],
        feed=farm.feed[farm.feed["date"] <= cut],
    )
    full = features.watch_frame(farm)
    trunc = features.watch_frame(past)
    a = full[full["date"] <= cut].reset_index(drop=True)
    b = trunc.reset_index(drop=True)
    assert len(a) == len(b)
    np.testing.assert_allclose(a[features.RISK_FEATURES].to_numpy(), b[features.RISK_FEATURES].to_numpy())


def test_herd_wide_shocks_cancel_out(farm):
    """A heat wave moves every cow's rumination and temperature; only the cow vs her herd mates counts."""
    w = features.watch_frame(farm)
    for column in ("log_scc_ratio", "temp_delta", "rum_delta", "litres_ratio"):
        assert w.groupby("date")[column].median().abs().max() < 1e-9


def test_warning_signal_shows_up_in_the_features_for_the_at_risk_cow(farm):
    w = features.watch_frame(farm)
    cow = farm.ground_truth["at_risk_cows"][0]
    today = w[(w["date"] == farm.today) & (w["cow_id"] == cow)].iloc[0]
    others = w[(w["date"] == farm.today) & (w["cow_id"] != cow)]
    assert today["rum_delta"] < others["rum_delta"].quantile(0.25)
