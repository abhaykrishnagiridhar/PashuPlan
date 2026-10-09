import json

import pytest

from pashuplan import tools


def test_every_schema_is_well_formed():
    assert tools.TOOL_SCHEMAS
    for s in tools.TOOL_SCHEMAS.values():
        assert {"name", "description", "input_schema"} <= set(s)
        assert s["input_schema"]["type"] == "object"
        assert len(s["description"]) > 20


def test_allowlists_only_reference_registered_tools():
    for agent, names in tools.AGENT_TOOLS.items():
        assert set(names) <= set(tools.TOOL_SCHEMAS), agent
    assert {"health", "nutrition", "breeding", "inventory", "finance", "manager"} == set(tools.AGENT_TOOLS)


def test_every_tool_is_used_by_at_least_one_agent():
    used = {n for names in tools.AGENT_TOOLS.values() for n in names}
    assert used == set(tools.TOOL_SCHEMAS)


def test_schemas_for_agent_respects_allowlist():
    names = {s["name"] for s in tools.schemas_for("finance")}
    assert names == set(tools.AGENT_TOOLS["finance"])
    assert "get_cow_detail" not in names


def test_run_tool_happy_path_is_json_safe(farm):
    out = tools.run_tool(farm, "health", "get_cow_detail", {"cow_id": farm.ground_truth["mastitis_cows"][0]})
    assert "error" not in out
    json.dumps(out)


def test_run_tool_without_arguments(farm):
    out = tools.run_tool(farm, "health", "get_mastitis_suspects", {})
    assert len(out["suspects"]) == 3


def test_run_tool_blocks_tools_outside_allowlist(farm):
    out = tools.run_tool(farm, "finance", "get_cow_detail", {"cow_id": "C01"})
    assert "not allowed" in out["error"]


def test_run_tool_unknown_tool_and_agent(farm):
    assert "unknown tool" in tools.run_tool(farm, "health", "drop_tables", {})["error"]
    assert "unknown agent" in tools.run_tool(farm, "ghost", "get_feed_quality", {})["error"]


def test_run_tool_feeds_errors_back_instead_of_raising(farm):
    assert "C99" in tools.run_tool(farm, "health", "get_cow_detail", {"cow_id": "C99"})["error"]
    assert "cow_id" in tools.run_tool(farm, "health", "get_cow_detail", {})["error"]
    assert "error" in tools.run_tool(farm, "health", "get_cow_detail", {"cow_id": "C01", "bogus": 1})


def test_model_tools_use_the_trained_models(farm, models):
    risk = tools.run_tool(farm, "health", "get_mastitis_risk", {}, models=models)
    assert [w["cow_id"] for w in risk["watch_list"]] == farm.ground_truth["at_risk_cows"]
    loss = tools.run_tool(farm, "finance", "get_loss_attribution", {}, models=models)
    assert set(loss["causes"]) == {"heat", "feed", "sick"}
    json.dumps([risk, loss])


def test_model_tools_say_so_when_no_models_are_loaded(farm):
    assert "not trained" in tools.run_tool(farm, "health", "get_mastitis_risk", {})["error"]
    assert "not trained" in tools.run_tool(farm, "finance", "get_loss_attribution", {})["error"]


def test_model_tools_respect_allowlists(farm, models):
    assert "not allowed" in tools.run_tool(farm, "breeding", "get_mastitis_risk", {}, models=models)["error"]


def test_top_decliners_shows_each_cow_against_the_herd_so_ordinary_decline_is_not_mistaken_for_illness(farm):
    out = tools.run_tool(farm, "health", "get_top_decliners", {"n": 8})
    assert out["herd_change_pct"] == pytest.approx(-14.3, abs=0.2)
    rows = {r["cow_id"]: r for r in out["decliners"]}
    sick = farm.ground_truth["mastitis_cows"]
    assert all(rows[c]["vs_herd_pct"] <= -10 for c in sick)  # clearly worse than the herd
    ordinary = [r for c, r in rows.items() if c not in sick]
    assert ordinary and all(abs(r["vs_herd_pct"]) < 3 for r in ordinary)  # about the same as every cow
    assert "herd" in out["note"] and "not individually sick" in out["note"].lower()
    json.dumps(out)


def test_run_tool_clamps_unreasonable_limits(farm):
    out = tools.run_tool(farm, "health", "get_top_decliners", {"n": 5000})
    assert len(out["decliners"]) <= tools.MAX_ROWS
