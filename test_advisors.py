import json
import threading

import pytest

from pashuplan import ask, orchestrator, providers, replay
from tests.fakes import FunctionClient, reply, text, tool_use


@pytest.fixture(autouse=True)
def no_provider_settings(monkeypatch):
    """Whatever keys are set on this machine, these tests start with none."""
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "PASHUPLAN_PROVIDER", "PASHUPLAN_MODEL",
                 "PASHUPLAN_BASE_URL", "PASHUPLAN_API_KEY", "PASHUPLAN_MAX_PARALLEL"):
        monkeypatch.delenv(name, raising=False)

QUESTION = "Milk production has decreased this week"
KN_QUESTION = "ಈ ವಾರ ಹಾಲಿನ ಉತ್ಪಾದನೆ ಕಡಿಮೆಯಾಗಿದೆ"


def _role(kwargs) -> str:
    return next(line.split(":", 1)[1].strip() for line in kwargs["system"].splitlines() if line.startswith("ROLE:"))


def _echo_handler(kwargs):
    role = _role(kwargs)
    if role == "manager":
        return reply(text("PLAN built from:\n" + kwargs["messages"][0]["content"]))
    return reply(text(f"ANSWER from {role}"))


def test_runs_every_specialist_then_the_manager(farm):
    client = FunctionClient(_echo_handler)
    out = orchestrator.run_advisors(client, farm, QUESTION, model="claude-test-1")
    assert [s["agent"] for s in out["specialists"]] == ["health", "nutrition", "breeding", "inventory", "finance"]
    assert all(s["answer"] == f"ANSWER from {s['agent']}" for s in out["specialists"])
    assert out["model"] == "claude-test-1" and out["question"] == QUESTION and out["language"] == "en"
    manager_input = out["plan"]
    assert QUESTION in manager_input
    for agent in orchestrator.SPECIALISTS:
        assert f"ANSWER from {agent}" in manager_input


def test_language_follows_the_question_unless_overridden(farm):
    client = FunctionClient(_echo_handler)
    out = orchestrator.run_advisors(client, farm, KN_QUESTION)
    assert out["language"] == "kn"
    assert all("Kannada" in c["system"] for c in client.calls)
    other = FunctionClient(_echo_handler)
    assert orchestrator.run_advisors(other, farm, KN_QUESTION, lang="hi")["language"] == "hi"
    assert all("Hindi" in c["system"] for c in other.calls)


def test_specialists_really_run_in_parallel(farm):
    barrier = threading.Barrier(len(orchestrator.SPECIALISTS), timeout=5)

    def handler(kwargs):
        if _role(kwargs) != "manager":
            barrier.wait()  # only passes if all five are in flight at once
        return _echo_handler(kwargs)

    out = orchestrator.run_advisors(FunctionClient(handler), farm, QUESTION)
    assert all(not s["error"] for s in out["specialists"])


def test_a_failing_specialist_is_reported_and_the_others_still_run(farm):
    def handler(kwargs):
        if _role(kwargs) == "nutrition":
            raise RuntimeError("boom")
        return _echo_handler(kwargs)

    out = orchestrator.run_advisors(FunctionClient(handler), farm, QUESTION)
    nutrition = next(s for s in out["specialists"] if s["agent"] == "nutrition")
    assert nutrition["error"] == "boom" and nutrition["answer"] == ""
    assert "ANSWER from health" in out["plan"]
    assert "nutrition" in out["plan"] and "unavailable" in out["plan"]


def test_a_failing_manager_keeps_the_specialist_answers(farm):
    def handler(kwargs):
        if _role(kwargs) == "manager":
            raise RuntimeError("manager down")
        return _echo_handler(kwargs)

    out = orchestrator.run_advisors(FunctionClient(handler), farm, QUESTION)
    assert out["plan"] == "" and out["error"] == "manager down"
    assert all(s["answer"] for s in out["specialists"])


def test_indian_script_digits_in_answers_are_converted_to_latin_digits(farm):
    def handler(kwargs):
        role = _role(kwargs)
        return reply(text("೧. ಮೇವು ೪೭.೩ ಲೀಟರ್" if role != "manager" else "ಯೋಜನೆ: ೧೨೧.೬ ಲೀಟರ್, ₹೪೦,೭೦೨"))

    out = orchestrator.run_advisors(FunctionClient(handler), farm, KN_QUESTION)
    assert out["plan"] == "ಯೋಜನೆ: 121.6 ಲೀಟರ್, ₹40,702"
    assert all(s["answer"] == "1. ಮೇವು 47.3 ಲೀಟರ್" for s in out["specialists"])


def _chars(client) -> int:
    return sum(len(json.dumps({"system": c["system"], "tools": c.get("tools", []), "messages": c["messages"]},
                              ensure_ascii=False)) for c in client.calls)


def _agentic_handler(kwargs):
    """Like a real model: ask for every tool it may use in one turn, then answer."""
    role = _role(kwargs)
    if role != "manager" and len(kwargs["messages"]) == 1:
        from pashuplan import tools
        return reply(*[tool_use(name, {"cow_id": "C16"} if name == "get_cow_detail" else {}, id=f"{role}-{i}")
                       for i, name in enumerate(tools.AGENT_TOOLS[role])])
    return reply(text("ಕ" * 600))


def test_compact_is_the_default_and_makes_six_requests_without_tools(farm):
    client = FunctionClient(_echo_handler)
    out = orchestrator.run_advisors(client, farm, QUESTION)
    assert out["mode"] == "compact"
    assert len(client.calls) == 6 and all("tools" not in c for c in client.calls)
    assert all(s["turns"] == 1 for s in out["specialists"])


def test_compact_still_records_the_lookups_made_for_each_advisor(farm, models):
    out = orchestrator.run_advisors(FunctionClient(_echo_handler), farm, QUESTION, models=models)
    health = out["specialists"][0]
    assert [c["name"] for c in health["tool_calls"]] == ["get_mastitis_suspects", "get_mastitis_risk", "get_top_decliners"]


def test_agentic_mode_still_lets_advisors_choose_and_call_tools(farm):
    client = FunctionClient(_agentic_handler)
    out = orchestrator.run_advisors(client, farm, QUESTION, mode="agentic")
    assert out["mode"] == "agentic" and len(client.calls) > 6
    assert any("tools" in c for c in client.calls)
    assert out["specialists"][0]["tool_calls"]


def test_an_unknown_mode_is_rejected(farm):
    with pytest.raises(ValueError, match="mode"):
        orchestrator.run_advisors(FunctionClient(_echo_handler), farm, QUESTION, mode="turbo")


def test_compact_sends_far_less_text_than_the_agentic_loop(farm, models):
    compact, agentic = FunctionClient(_echo_handler), FunctionClient(_agentic_handler)
    orchestrator.run_advisors(compact, farm, QUESTION, models=models)
    orchestrator.run_advisors(agentic, farm, QUESTION, models=models, mode="agentic")
    assert len(compact.calls) * 1.5 < len(agentic.calls)
    assert _chars(compact) < 0.65 * _chars(agentic)


def test_command_line_defaults_to_compact_and_accepts_agentic(farm, tmp_path):
    compact, agentic = FunctionClient(_echo_handler), FunctionClient(_agentic_handler)
    ask.main(["--replay-file", str(tmp_path / "a.json")], client=compact, farm=farm, models=None)
    ask.main(["--mode", "agentic", "--replay-file", str(tmp_path / "b.json")], client=agentic, farm=farm, models=None)
    assert all("tools" not in c for c in compact.calls)
    assert any("tools" in c for c in agentic.calls)


def test_tool_calls_are_logged_and_the_result_is_json_safe(farm):
    def handler(kwargs):
        if _role(kwargs) == "health" and len(kwargs["messages"]) == 1:
            return reply(tool_use("get_mastitis_suspects"))
        return _echo_handler(kwargs)

    out = orchestrator.run_advisors(FunctionClient(handler), farm, QUESTION, mode="agentic")
    health = out["specialists"][0]
    assert [c["name"] for c in health["tool_calls"]] == ["get_mastitis_suspects"]
    json.dumps(out)


def test_replay_roundtrip(tmp_path):
    result = {"plan": "Do this.", "specialists": []}
    path = tmp_path / "advisors.json"
    replay.save(result, path, saved_on="2026-10-09")
    assert replay.load(path) == {**result, "saved_on": "2026-10-09"}


def test_replay_missing_file_is_none_and_corrupt_file_raises(tmp_path):
    assert replay.load(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        replay.load(bad)


def test_save_if_successful_keeps_the_old_answer_when_a_run_failed(tmp_path):
    path = tmp_path / "advisors.json"
    assert replay.save_if_successful({"plan": "Good plan.", "specialists": []}, path) is True
    assert replay.save_if_successful({"plan": "", "error": "401", "specialists": []}, path) is False
    assert replay.load(path)["plan"] == "Good plan."


def test_each_language_keeps_its_own_saved_advice(tmp_path):
    kn = {"plan": "ಕನ್ನಡ ಸಲಹೆ", "language": "kn", "specialists": []}
    hi = {"plan": "हिन्दी सलाह", "language": "hi", "specialists": []}
    assert replay.save_if_successful(kn, directory=tmp_path) and replay.save_if_successful(hi, directory=tmp_path)
    assert replay.load_for("kn", tmp_path)["plan"] == "ಕನ್ನಡ ಸಲಹೆ"
    assert replay.load_for("hi", tmp_path)["plan"] == "हिन्दी सलाह"
    assert replay.load_for("en", tmp_path) is None
    assert replay.available_languages(tmp_path) == ["kn", "hi"]


def test_the_older_single_file_is_only_used_for_its_own_language(tmp_path):
    replay.save({"plan": "Old Kannada advice", "language": "kn", "specialists": []}, tmp_path / "advisors.json")
    assert replay.load_for("kn", tmp_path)["plan"] == "Old Kannada advice"
    assert replay.load_for("hi", tmp_path) is None
    assert replay.available_languages(tmp_path) == ["kn"]


def test_a_new_run_in_a_language_replaces_only_that_language(tmp_path):
    for lang, plan in (("kn", "kn one"), ("hi", "hi one"), ("kn", "kn two")):
        replay.save_if_successful({"plan": plan, "language": lang, "specialists": []}, directory=tmp_path)
    assert replay.load_for("kn", tmp_path)["plan"] == "kn two"
    assert replay.load_for("hi", tmp_path)["plan"] == "hi one"


def test_command_line_saves_into_the_questions_own_language_by_default(farm, tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "REPLAY_DIR", tmp_path)
    ask.main(["--lang", "hi"], client=FunctionClient(_echo_handler), farm=farm, models=None)
    assert replay.load_for("hi", tmp_path)["language"] == "hi"
    assert replay.load_for("kn", tmp_path) is None


def test_a_partial_run_is_never_saved_as_the_demo_answer(tmp_path):
    path = tmp_path / "advisors.json"
    replay.save_if_successful({"plan": "Complete plan.", "specialists": [{"agent": "health", "error": None}]}, path)
    partial = {"plan": "Plan without stock check.",
               "specialists": [{"agent": "health", "error": None}, {"agent": "inventory", "error": "429 quota"}]}
    assert replay.save_if_successful(partial, path) is False
    assert replay.load(path)["plan"] == "Complete plan."


def test_command_line_shows_a_partial_plan_but_warns_and_does_not_save_it(farm, tmp_path, capsys):
    path = tmp_path / "advisors.json"

    def handler(kwargs):
        if _role(kwargs) == "inventory":
            raise RuntimeError("429 quota")
        return _echo_handler(kwargs)

    out = ask.main(["--replay-file", str(path)], client=FunctionClient(handler), farm=farm, models=None)
    captured = capsys.readouterr()
    assert "PLAN built from" in captured.out and out["plan"]
    assert "inventory" in captured.err and "429 quota" in captured.err and "not saved" in captured.err
    assert not path.exists()


def test_command_line_without_a_key_exits_with_a_clear_message_not_a_traceback(farm, tmp_path):
    with pytest.raises(SystemExit) as stop:
        ask.main(["--replay-file", str(tmp_path / "a.json")], farm=farm, models=None)
    assert "GEMINI_API_KEY" in str(stop.value.code) and "ANTHROPIC_API_KEY" in str(stop.value.code)


def test_a_failed_run_exits_nonzero_and_does_not_touch_the_saved_answer(farm, tmp_path):
    path = tmp_path / "advisors.json"
    replay.save({"plan": "Earlier good plan.", "specialists": []}, path)

    def always_down(kwargs):
        raise RuntimeError("401 invalid x-api-key")

    with pytest.raises(SystemExit) as stop:
        ask.main(["--replay-file", str(path)], client=FunctionClient(always_down), farm=farm, models=None)
    assert stop.value.code not in (0, None)
    assert str(stop.value.code).count("401 invalid x-api-key") == 1  # the same error is listed once, not six times
    assert replay.load(path)["plan"] == "Earlier good plan."


def test_at_most_max_parallel_advisors_run_at_the_same_time(farm):
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def handler(kwargs):
        if _role(kwargs) == "manager":
            return _echo_handler(kwargs)
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        threading.Event().wait(0.05)
        with lock:
            state["now"] -= 1
        return _echo_handler(kwargs)

    out = orchestrator.run_advisors(FunctionClient(handler), farm, QUESTION, max_parallel=2)
    assert state["peak"] == 2
    assert all(s["answer"] for s in out["specialists"])


def test_command_line_uses_the_configured_providers_model_and_parallelism(farm, tmp_path, monkeypatch):
    client = FunctionClient(_echo_handler)
    provider = providers.Provider(name="fake", model="prov-model", max_parallel=1, make_client=lambda: client)
    monkeypatch.setattr(providers, "resolve", lambda *a, **k: provider)
    ask.main(["--replay-file", str(tmp_path / "a.json")], farm=farm, models=None)
    assert {c["model"] for c in client.calls} == {"prov-model"}


def test_command_line_reports_a_misconfigured_provider_plainly(farm, tmp_path, monkeypatch):
    def broken(*a, **k):
        raise ValueError("PASHUPLAN_MODEL is required")

    monkeypatch.setattr(providers, "resolve", broken)
    with pytest.raises(SystemExit) as stop:
        ask.main(["--replay-file", str(tmp_path / "a.json")], farm=farm, models=None)
    assert "PASHUPLAN_MODEL" in str(stop.value.code)


def test_command_line_prints_the_plan_and_saves_a_replay(farm, tmp_path, capsys):
    path = tmp_path / "advisors.json"
    out = ask.main(["--question", QUESTION, "--replay-file", str(path)], client=FunctionClient(_echo_handler),
                   farm=farm, models=None)
    printed = capsys.readouterr().out
    assert "PLAN built from" in printed
    assert replay.load(path)["plan"] == out["plan"]


def test_command_line_defaults_to_the_kannada_question(farm, tmp_path):
    client = FunctionClient(_echo_handler)
    out = ask.main(["--replay-file", str(tmp_path / "a.json")], client=client, farm=farm, models=None)
    assert out["language"] == "kn"
