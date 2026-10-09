import json
from datetime import timedelta

import pytest

from pashuplan.data_gen import generate_farm
from pashuplan.risk_model import MastitisRiskModel

HELD_OUT_SEEDS = (201, 202, 203)


@pytest.fixture(scope="module")
def held_out_farms():
    return [generate_farm(seed=s, randomize_events=True) for s in HELD_OUT_SEEDS]


def test_fit_needs_data():
    with pytest.raises(ValueError):
        MastitisRiskModel.fit([])


def test_separates_cows_about_to_fall_sick_on_unseen_farms(risk_model, held_out_farms):
    metrics = risk_model.evaluate(held_out_farms)
    assert metrics["n_positive"] >= 15
    assert metrics["auc"] >= 0.85


def test_flags_a_useful_share_without_flooding_the_farmer(risk_model, held_out_farms):
    m = risk_model.evaluate(held_out_farms)
    assert m["recall"] >= 0.5
    assert m["precision"] >= 0.2
    assert m["alerts_per_cow_week"] <= 0.25


def test_hero_watch_list_catches_the_cow_one_day_from_sickness(farm, risk_model):
    watch = risk_model.watch_list(farm)
    ids = [w["cow_id"] for w in watch]
    assert farm.ground_truth["at_risk_cows"][0] in ids
    assert not set(ids) & set(farm.ground_truth["mastitis_cows"])  # already sick, not "at risk"
    assert len(ids) <= 4


def test_watch_list_is_sorted_by_risk(farm, risk_model):
    risks = [w["risk"] for w in risk_model.watch_list(farm)]
    assert risks == sorted(risks, reverse=True)
    assert all(0 <= r <= 1 for r in risks)


def test_early_warning_beats_the_plain_rule_on_the_hero_farm(farm, risk_model):
    warnings = risk_model.early_warnings(farm)
    caught = [w for w in warnings if w["lead_days"] >= 1]
    assert len(caught) >= 2
    for w in caught:
        assert w["warned_on"] < w["onset"]
        assert w["onset"] == farm.ground_truth["mastitis_onset"][w["cow_id"]]


def test_never_uses_future_days(farm, risk_model):
    first = risk_model.risk_table(farm)
    day = farm.today - timedelta(days=15)
    assert first[first["date"] == day]["risk"].notna().all()


def test_roundtrips_through_json(farm, risk_model):
    clone = MastitisRiskModel.from_dict(json.loads(json.dumps(risk_model.to_dict())))
    assert clone.watch_list(farm) == risk_model.watch_list(farm)
    assert clone.threshold == risk_model.threshold


def test_training_is_deterministic(train_farms, risk_model):
    again = MastitisRiskModel.fit(train_farms)
    assert again.to_dict() == risk_model.to_dict()
