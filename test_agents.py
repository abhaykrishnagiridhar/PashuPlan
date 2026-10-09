import json

import pytest

from pashuplan import agents, tools
from tests.fakes import ScriptedClient, reply, text, tool_use


def _run(client, farm, agent="health", **kw):
    return agents.run_agent(client, farm, agent, "Why is my milk down?", kw.pop("lang", "en"), **kw)


def test_answers_directly_when_no_tool_is_needed(farm):
    client = ScriptedClient([reply(text("All is well."))])
    res = _run(client, farm)
    assert res.answer == "All is well."
    assert res.turns == 1 and res.tool_calls == () and not res.hit_turn_limit


def test_runs_the_tool_the_model_asks_for_and_feeds_the_result_back(farm):
    client = ScriptedClient([reply(text("Checking."), tool_use("get_mastitis_suspects", id="tu_9")),
                             reply(text("C16, C32 and C40 are sick."))])
    res = _run(client, farm)
    assert res.answer == "C16, C32 and C40 are sick."
    assert res.turns == 2
    assert [c["name"] for c in res.tool_calls] == ["get_mastitis_suspects"]
    assert res.tool_calls[0]["error"] is False
    sent_back = client.calls[1]["messages"][-1]
    assert sent_back["role"] == "user"
    block = sent_back["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "tu_9"
    assert "C16" in block["content"] and not block.get("is_error")
    assistant = client.calls[1]["messages"][-2]
    assert assistant["role"] == "assistant"
    assert assistant["content"][1] == {"type": "tool_use", "id": "tu_9", "name": "get_mastitis_suspects", "input": {}}


def test_tool_errors_are_fed_back_so_the_model_can_correct_itself(farm):
    client = ScriptedClient([reply(tool_use("get_cow_detail", {"cow_id": "C99"})),
                             reply(tool_use("get_cow_detail", {"cow_id": "C16"}, id="tu_2")),
                             reply(text("C16 is sick."))])
    res = _run(client, farm)
    assert res.answer == "C16 is sick."
    assert [c["error"] for c in res.tool_calls] == [True, False]
    first_result = client.calls[1]["messages"][-1]["content"][0]
    assert first_result["is_error"] is True and "C99" in first_result["content"]


def test_a_tool_outside_the_allowlist_is_refused_not_run(farm):
    client = ScriptedClient([reply(tool_use("get_finance_summary")), reply(text("ok"))])
    res = _run(client, farm, agent="health")
    assert res.tool_calls[0]["error"] is True
    assert "not allowed" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_only_the_agents_own_tools_are_offered_to_the_model(farm):
    client = ScriptedClient([reply(text("ok"))])
    _run(client, farm, agent="finance")
    offered = {t["name"] for t in client.calls[0]["tools"]}
    assert offered == set(tools.AGENT_TOOLS["finance"])


def test_several_tool_calls_in_one_turn_are_all_answered(farm):
    client = ScriptedClient([reply(tool_use("get_mastitis_suspects", id="a"), tool_use("get_top_decliners", id="b")),
                             reply(text("done"))])
    res = _run(client, farm)
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert len(res.tool_calls) == 2


def test_the_turn_cap_stops_a_looping_agent_and_forces_an_answer(farm):
    looping = [reply(tool_use("get_mastitis_suspects", id=f"t{i}")) for i in range(agents.MAX_TURNS)]
    client = ScriptedClient(looping + [reply(text("Best answer so far."))])
    res = _run(client, farm)
    assert res.hit_turn_limit and res.answer == "Best answer so far."
    assert len(client.calls) == agents.MAX_TURNS + 1
    assert "tools" not in client.calls[-1]  # the forced final call cannot use tools
    assert client.calls[-1]["messages"][-1]["role"] == "user"


def test_the_model_models_and_token_limit_are_passed_through(farm):
    client = ScriptedClient([reply(text("ok"))])
    _run(client, farm, model="claude-test-1")
    assert client.calls[0]["model"] == "claude-test-1"
    assert client.calls[0]["max_tokens"] == agents.MAX_TOKENS


def test_model_tools_get_the_trained_models(farm, models):
    client = ScriptedClient([reply(tool_use("get_mastitis_risk")), reply(text("Watch C02."))])
    _run(client, farm, models=models)
    result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    assert [w["cow_id"] for w in result["watch_list"]] == farm.ground_truth["at_risk_cows"]


@pytest.mark.parametrize("lang,word", [("kn", "Kannada"), ("hi", "Hindi"), ("en", "English")])
def test_system_prompt_pins_the_language_and_the_no_invention_rule(lang, word):
    prompt = agents.system_prompt("health", lang)
    assert word in prompt
    assert "Only use numbers" in prompt
    assert "Health" in prompt


def test_prompts_ban_jargon_and_unit_slang_a_farmer_would_not_know():
    prompt = agents.system_prompt("nutrition", "kn")
    for term in ("THI", "SCC", "mastitis"):
        assert term in prompt  # named in the rule so the model knows what to avoid
    assert "plain" in prompt and "₹" in prompt  # rupee sign, not "Rs"


def test_the_health_advisor_may_only_name_cows_that_a_health_tool_flagged():
    prompt = agents.system_prompt("health", "en")
    assert "get_mastitis_risk" in prompt and "get_mastitis_suspects" in prompt
    assert "declin" in prompt and "herd" in prompt  # a drop equal to the herd's is not illness


def test_the_manager_must_report_urgent_stock_money_and_missing_advisors():
    prompt = agents.system_prompt("manager", "en")
    assert "unavailable" in prompt and "say so" in prompt
    assert "stock" in prompt and "money" in prompt
    assert agents.MANAGER_WORDS > agents.SPECIALIST_WORDS
    assert f"{agents.MANAGER_WORDS} words" in prompt and f"{agents.SPECIALIST_WORDS} words" in agents.system_prompt("health", "en")


def test_advisors_are_told_never_to_ask_the_farmer_anything():
    for agent in ("health", "breeding", "manager"):
        prompt = agents.system_prompt(agent, "en")
        assert "Never ask the farmer" in prompt and "cannot receive replies" in prompt
    assert "Ignore any question" in agents.system_prompt("manager", "en")


def test_the_finance_advisor_knows_discarded_milk_is_not_spoiled_milk():
    prompt = agents.system_prompt("finance", "en")
    assert "medicine waiting" in prompt and "spoiled" in prompt


def _compact(client, farm, agent="health", **kw):
    return agents.run_agent_compact(client, farm, agent, "Why is my milk down?", kw.pop("lang", "en"), **kw)


def test_compact_makes_one_request_without_tools_and_puts_the_facts_in_it(farm, models):
    client = ScriptedClient([reply(text("Plan."))])
    res = _compact(client, farm, models=models)
    assert len(client.calls) == 1 and "tools" not in client.calls[0]
    sent = client.calls[0]["messages"][0]["content"]
    for marker in ("[get_mastitis_suspects]", "C16", "[get_mastitis_risk]", "C02", "[get_top_decliners]", "vs_herd_pct"):
        assert marker in sent
    assert res.answer == "Plan." and res.turns == 1 and not res.hit_turn_limit
    assert [c["name"] for c in res.tool_calls] == ["get_mastitis_suspects", "get_mastitis_risk", "get_top_decliners"]
    assert all(not c["error"] for c in res.tool_calls)


def test_compact_skips_tools_that_need_an_argument(farm, models):
    client = ScriptedClient([reply(text("ok"))])
    _compact(client, farm, models=models)
    assert "get_cow_detail" not in client.calls[0]["messages"][0]["content"]


def test_compact_hands_data_problems_to_the_model_as_data(farm):
    client = ScriptedClient([reply(text("ok"))])
    res = _compact(client, farm, models=None)  # no trained models: the model tool reports an error
    assert "models not trained" in client.calls[0]["messages"][0]["content"]
    assert {c["name"]: c["error"] for c in res.tool_calls}["get_mastitis_risk"] is True


def test_compact_gives_the_manager_the_herd_trend(farm):
    client = ScriptedClient([reply(text("ok"))])
    _compact(client, farm, agent="manager")
    assert "[get_herd_milk_trend]" in client.calls[0]["messages"][0]["content"]


def test_the_compact_prompt_mentions_no_tools_but_keeps_every_other_rule():
    compact, agentic = agents.system_prompt("health", "kn", compact=True), agents.system_prompt("health", "kn")
    assert "Use your tools" in agentic and "get_cow_detail" in agentic
    assert "Use your tools" not in compact and "get_cow_detail" not in compact
    for rule in ("Only use numbers", "Never ask the farmer", "THI", "₹", "Kannada"):
        assert rule in compact
    assert "given in the message" in compact


def test_every_agent_has_a_role_prompt():
    for agent in tools.AGENT_TOOLS:
        assert agents.system_prompt(agent, "en")


def test_unknown_agent_is_rejected(farm):
    with pytest.raises(KeyError):
        agents.run_agent(ScriptedClient([]), farm, "ghost", "hi", "en")
