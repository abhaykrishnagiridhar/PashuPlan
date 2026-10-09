import json

import pytest

from pashuplan import analytics


def test_milk_trend_reports_the_drop(farm):
    t = analytics.milk_trend(farm)
    assert t["this_period_avg_l_per_day"] < t["prior_period_avg_l_per_day"]
    assert -25 <= t["change_pct"] <= -8
    assert t["change_l_per_day"] < 0


def test_top_decliners_include_all_mastitis_cows(farm):
    top = analytics.top_decliners(farm, n=5)
    assert len(top) == 5
    changes = [r["change_pct"] for r in top]
    assert changes == sorted(changes)  # most negative first
    assert set(farm.ground_truth["mastitis_cows"]) <= {r["cow_id"] for r in top}


def test_mastitis_suspects_match_ground_truth_exactly(farm):
    suspects = analytics.mastitis_suspects(farm)
    assert {s["cow_id"] for s in suspects} == set(farm.ground_truth["mastitis_cows"])
    assert all(s["avg_scc_3d"] > 200 and s["max_temp_3d"] >= 39.5 for s in suspects)


def test_mastitis_suspects_flag_treatment_status(farm):
    by_id = {s["cow_id"]: s for s in analytics.mastitis_suspects(farm)}
    assert all("under_treatment" in s and "discard_until" in s for s in by_id.values())
    assert any(s["under_treatment"] for s in by_id.values())


def test_cow_detail_for_mastitis_cow(farm):
    cid = farm.ground_truth["mastitis_cows"][0]
    d = analytics.cow_detail(farm, cid)
    assert d["cow_id"] == cid
    assert d["has_mastitis_signs"] is True
    assert len(d["daily"]) == 14
    assert {"date", "litres", "scc", "temp_c", "rumination_min"} <= set(d["daily"][0])


def test_cow_detail_late_lactation_is_stage_late_not_sick(farm):
    healthy_late = next(c for c in farm.ground_truth["late_lactation_cows"]
                        if c not in farm.ground_truth["mastitis_cows"])
    d = analytics.cow_detail(farm, healthy_late)
    assert d["lactation_stage"] == "late"
    assert d["has_mastitis_signs"] is False
    assert 0 <= d["expected_weekly_decline_pct"] < 3


def test_cow_detail_unknown_cow_raises(farm):
    with pytest.raises(ValueError, match="C99"):
        analytics.cow_detail(farm, "C99")


def test_feed_quality_detects_the_change(farm):
    f = analytics.feed_quality(farm)
    assert f["change_detected"] is True
    assert f["change_date"] == farm.ground_truth["feed_change_date"].isoformat()
    assert f["cp_delta"] <= -2.0
    assert f["cp_after"] < f["target_cp"] <= f["cp_before"]
    assert f["cost_per_kg_after"] < f["cost_per_kg_before"]


def test_heat_stress_summary(farm):
    h = analytics.heat_stress(farm)
    assert h["days_at_or_above_mild"] >= 6
    assert h["max_thi"] >= 82
    assert h["worst_level"] in {"moderate", "severe"}
    assert len(h["hot_dates"]) == h["days_at_or_above_mild"]


def test_breeding_status_lists_overdue_cows(farm):
    b = analytics.breeding_status(farm)
    assert set(b["overdue_open_cows"]) == set(farm.ground_truth["overdue_breeding_cows"])
    assert b["pregnant_count"] + b["open_count"] == 40


def test_inventory_flags_antibiotic_tubes(farm):
    inv = analytics.inventory_status(farm)
    assert "mastitis_tubes" in inv["low_stock"]
    tubes = next(i for i in inv["items"] if i["item"] == "mastitis_tubes")
    assert tubes["days_remaining"] == pytest.approx(4.0)
    assert tubes["below_reorder_level"] is True


def test_finance_summary_quantifies_the_loss(farm):
    f = analytics.finance_summary(farm)
    assert f["revenue_change_inr"] < 0
    assert f["milk_revenue_this_period_inr"] < f["milk_revenue_prior_period_inr"]
    assert f["discarded_milk_litres"] > 0
    assert f["discarded_milk_value_inr"] == pytest.approx(f["discarded_milk_litres"] * farm.config["milk_price_per_l"], rel=1e-3)
    assert f["margin_this_period_inr"] < f["margin_prior_period_inr"]


@pytest.mark.parametrize("call", [
    lambda f: analytics.milk_trend(f),
    lambda f: analytics.top_decliners(f),
    lambda f: analytics.mastitis_suspects(f),
    lambda f: analytics.cow_detail(f, "C01"),
    lambda f: analytics.feed_quality(f),
    lambda f: analytics.heat_stress(f),
    lambda f: analytics.breeding_status(f),
    lambda f: analytics.inventory_status(f),
    lambda f: analytics.finance_summary(f),
])
def test_outputs_are_json_safe(farm, call):
    json.dumps(call(farm))
