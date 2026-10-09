"""Runs the five specialist advisors, then the Farm Manager over their reports.

mode "compact" (default): one model request per advisor, 6 in all. mode "agentic": each advisor runs its own
tool-use loop. See agents.py.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pashuplan import agents, i18n
from pashuplan.data_gen import FarmData
from pashuplan.models import Models

SPECIALISTS = ("health", "nutrition", "breeding", "inventory", "finance")
MODES = {"compact": agents.run_agent_compact, "agentic": agents.run_agent}
DEFAULT_MODE = "compact"


def _run_specialist(run, client, farm, agent, question, lang, models, model) -> dict:
    try:
        res = run(client, farm, agent, question, lang, models=models, model=model)
    except Exception as exc:  # one advisor failing must not stop the others; the error is shown, not hidden
        return {"agent": agent, "answer": "", "tool_calls": [], "turns": 0, "hit_turn_limit": False,
                "error": str(exc)}
    return {"agent": agent, "answer": i18n.latin_digits(res.answer), "tool_calls": list(res.tool_calls),
            "turns": res.turns, "hit_turn_limit": res.hit_turn_limit, "error": None}


def _manager_message(question: str, specialists: list[dict]) -> str:
    reports = [f"[{s['agent']}] " + (s["answer"] if not s["error"] else f"unavailable (error: {s['error']})")
               for s in specialists]
    return f"Farmer's question: {question}\n\nSpecialist reports:\n" + "\n\n".join(reports)


def run_advisors(client, farm: FarmData, question: str, models: Models | None = None, lang: str | None = None,
                 model: str = agents.DEFAULT_MODEL, max_parallel: int | None = None, mode: str = DEFAULT_MODE) -> dict:
    """max_parallel caps how many advisors call the model at once (free tiers allow only a few requests a minute)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)} (got {mode!r})")
    run = MODES[mode]
    lang = lang or i18n.detect_language(question)
    workers = max(1, min(max_parallel or len(SPECIALISTS), len(SPECIALISTS)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_specialist, run, client, farm, a, question, lang, models, model)
                   for a in SPECIALISTS]
        specialists = [f.result() for f in futures]
    plan, error = "", None
    try:
        plan = i18n.latin_digits(run(client, farm, "manager", _manager_message(question, specialists), lang,
                                     models=models, model=model).answer)
    except Exception as exc:
        error = str(exc)
    return {"question": question, "language": lang, "model": model, "mode": mode, "specialists": specialists,
            "plan": plan, "error": error}
