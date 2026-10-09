import json

import pytest

from pashuplan import analytics
from pashuplan.attribution import LossAttribution


def test_fit_needs_data():
    with pytest.raises(ValueError):
        LossAttribution.fit([])


def test_every_cause_costs_milk(attribution):
    assert set(attribution.coefs) == {"heat", "feed", "sick"}
    assert all(c > 0 for c in attribution.coefs.values())


def test_hero_farm_blames_all_three_causes(farm, attribution):
    out = attribution.explain(farm)
    assert set(out["causes"]) == {"heat", "feed", "sick"}
    assert all(v > 5 for v in out["causes"].values())


def test_explained_loss_adds_up_to_the_observed_drop(farm, attribution):
    out = attribution.explain(farm)
    observed = -analytics.milk_trend(farm)["change_l_per_day"]
    assert out["observed_drop_l_per_day"] == pytest.approx(observed, abs=1.0)
    assert out["explained_l_per_day"] == pytest.approx(sum(out["causes"].values()), abs=0.2)
    assert abs(out["unexplained_l_per_day"]) < 0.25 * observed


def test_feed_is_the_biggest_single_cause_and_sickness_the_smallest(farm, attribution):
    c = attribution.explain(farm)["causes"]
    assert c["feed"] > c["heat"] > c["sick"]


def test_sick_cows_are_not_overblamed(farm, attribution):
    # three sick cows out of 40 can cost only a few percent of the herd's milk
    assert attribution.explain(farm)["causes"]["sick"] < 0.12 * analytics.milk_trend(farm)["prior_period_avg_l_per_day"]


def test_roundtrips_through_json(farm, attribution):
    clone = LossAttribution.from_dict(json.loads(json.dumps(attribution.to_dict())))
    assert clone.explain(farm) == attribution.explain(farm)


def test_training_is_deterministic(train_farms, attribution):
    again = LossAttribution.fit(train_farms)
    assert again.coefs == attribution.coefs
