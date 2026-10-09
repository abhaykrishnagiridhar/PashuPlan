import dataclasses
import json

import pytest

from pashuplan import report

ORDER = ["today", "week", "later"]


@pytest.fixture(scope="module")
def rep(farm):
    return report.build_report(farm)


def test_milk_headline(rep):
    assert 700 < rep["milk"]["per_day"] < 760
    assert rep["milk"]["change_pct"] < -8
    assert rep["milk"]["litres_less_per_day"] > 50
    assert rep["milk"]["prior_avg"] > rep["milk"]["this_avg"]


def test_causes_come_in_a_fixed_order(rep):
    assert [c["key"] for c in rep["causes"]] == ["sick_cows", "feed", "heat"]


def test_sick_cows_match_the_planted_mastitis_cows(farm, rep):
    sick = next(c for c in rep["causes"] if c["key"] == "sick_cows")
    assert set(sick["cows"]) == set(farm.ground_truth["mastitis_cows"])


def test_feed_and_heat_causes_carry_plain_numbers(rep):
    feed = next(c for c in rep["causes"] if c["key"] == "feed")
    heat = next(c for c in rep["causes"] if c["key"] == "heat")
    assert feed["cp_before"] - feed["cp_after"] >= 2
    assert heat["hot_days"] >= 6 and heat["avg_temp_c"] > 30


def test_fine_cows_are_late_lactation_and_not_sick(farm, rep):
    gt = farm.ground_truth
    expected = set(gt["late_lactation_cows"]) - set(gt["mastitis_cows"])
    assert set(rep["fine_cows"]) == expected
    assert not set(rep["fine_cows"]) & set(gt["mastitis_cows"])


def test_actions_are_ordered_by_urgency(rep):
    whens = [a["when"] for a in rep["actions"]]
    assert whens == sorted(whens, key=ORDER.index)


def test_treat_action_lists_every_sick_cow_with_discard_date(farm, rep):
    treat = next(a for a in rep["actions"] if a["key"] == "treat_cows")
    assert treat["when"] == "today"
    assert {c["cow_id"] for c in treat["cows"]} == set(farm.ground_truth["mastitis_cows"])
    assert all(c["discard_until"] for c in treat["cows"])


def test_tubes_action_says_how_many_days_are_left(rep):
    tubes = next(a for a in rep["actions"] if a["key"] == "buy_tubes")
    assert tubes["when"] == "today"
    assert tubes["days_left"] == 4
    assert tubes["qty"] == 12 and tubes["daily_use"] == 3


def test_feed_action_compares_saving_with_milk_lost(rep):
    feed = next(a for a in rep["actions"] if a["key"] == "fix_feed")
    assert feed["saving_per_day_inr"] == pytest.approx(1760, rel=0.01)
    assert feed["milk_loss_per_day_inr"] > feed["saving_per_day_inr"]


def test_breeding_action_is_last_and_lists_overdue_cows(farm, rep):
    breed = rep["actions"][-1]
    assert breed["key"] == "check_breeding" and breed["when"] == "later"
    assert set(breed["cows"]) == set(farm.ground_truth["overdue_breeding_cows"])


def test_money(rep):
    m = rep["money"]
    assert m["revenue_change_inr"] < 0
    assert m["discarded_litres"] > 0
    assert m["discarded_value_inr"] > 0


def test_bars_cover_two_weeks_in_date_order(rep):
    bars = rep["bars"]
    assert len(bars) == 14
    assert [b["period"] for b in bars] == ["prior"] * 7 + ["this"] * 7
    assert [b["date"] for b in bars] == sorted(b["date"] for b in bars)


def test_without_models_there_are_no_model_numbers(rep):
    assert all("loss_l" not in c for c in rep["causes"])
    assert rep["watch_list"] == []


def test_with_models_each_cause_shows_litres_lost_per_day(farm, models):
    r = report.build_report(farm, models=models)
    loss = {c["key"]: c["loss_l"] for c in r["causes"]}
    assert all(isinstance(v, int) and v > 5 for v in loss.values())
    assert loss["feed"] > loss["heat"] > loss["sick_cows"]


def test_with_models_the_watch_list_names_the_cow_about_to_get_sick(farm, models):
    r = report.build_report(farm, models=models)
    assert r["watch_list"] == farm.ground_truth["at_risk_cows"]
    assert not set(r["watch_list"]) & set(farm.ground_truth["mastitis_cows"])


def test_report_with_models_is_json_safe(farm, models):
    json.dumps(report.build_report(farm, models=models))


TOOL_BEHIND = {"sick_cows": "get_mastitis_suspects", "feed": "get_feed_quality", "heat": "get_heat_stress",
               "treat_cows": "get_mastitis_suspects", "buy_tubes": "get_inventory_status",
               "fix_feed": "get_feed_quality", "protect_from_heat": "get_heat_stress",
               "check_breeding": "get_breeding_status"}


def test_every_cause_and_action_names_the_agent_whose_data_it_comes_from(rep):
    from pashuplan import tools
    for item in [*rep["causes"], *rep["actions"]]:
        agent = item["agent"]
        assert agent in tools.AGENT_TOOLS
        assert TOOL_BEHIND[item["key"]] in tools.AGENT_TOOLS[agent], f"{item['key']} is not {agent}'s data"


def test_the_other_blocks_name_their_agents_too(rep):
    from pashuplan import tools
    sources = rep["sources"]
    assert set(sources) == {"milk", "money", "chart", "fine", "watch"}
    needs = {"milk": "get_herd_milk_trend", "chart": "get_herd_milk_trend", "money": "get_finance_summary",
             "fine": "get_mastitis_suspects", "watch": "get_mastitis_risk"}
    for block, agent in sources.items():
        assert needs[block] in tools.AGENT_TOOLS[agent], f"{block} is not {agent}'s data"


def test_report_is_json_safe(rep):
    json.dumps(rep)


def test_no_tubes_action_when_stock_is_fine(farm):
    inv = farm.inventory.copy()
    inv.loc[inv["item"] == "mastitis_tubes", "qty"] = 500.0
    calm = dataclasses.replace(farm, inventory=inv)
    keys = [a["key"] for a in report.build_report(calm)["actions"]]
    assert "buy_tubes" not in keys
    assert "treat_cows" in keys
