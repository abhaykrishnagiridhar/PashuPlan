"""Tool registry for the agent tool-use loop.

Schemas are in Anthropic tool format. `run_tool` never raises: any failure is returned as
{"error": "..."} so the model can read it and correct its next call.
"""
from __future__ import annotations

from typing import Any, Callable

from pashuplan import analytics
from pashuplan.data_gen import FarmData
from pashuplan.models import Models

MAX_ROWS = 10
MAX_DETAIL_DAYS = 30
MODELS_MISSING = "models not trained yet: run `python -m pashuplan.train`"

_COW_ID = {"type": "string", "description": "Cow identifier such as C16."}
_WINDOW = {"type": "integer", "description": "Days per comparison period (default 7).", "minimum": 1, "maximum": 30}


def _schema(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties or {}, "required": required or []}}


TOOL_SCHEMAS: dict[str, dict] = {s["name"]: s for s in [
    _schema("get_herd_milk_trend",
            "Herd-wide average daily litres this period versus the prior period, with absolute and percent change.",
            {"window": _WINDOW}),
    _schema("get_top_decliners",
            "Cows whose average daily milk fell the most versus the prior period, most negative first, each compared "
            "with the whole herd's fall. Cows that fell about as much as the herd are not individually sick.",
            {"window": _WINDOW, "n": {"type": "integer", "description": f"How many cows (max {MAX_ROWS}).", "minimum": 1}}),
    _schema("get_mastitis_suspects",
            "Cows with high somatic cell count AND fever in the last 3 days, with treatment and milk discard status.",
            {}),
    _schema("get_mastitis_risk",
            "Early-warning model: cows that look healthy today but show the quiet early signs of mastitis "
            "(risk from 0 to 1, highest first). These cows are not yet sick by the plain SCC and fever rule.",
            {}),
    _schema("get_cow_detail",
            "Daily litres, SCC, temperature and rumination for one cow, plus lactation stage, expected decline "
            "and treatment history. Use to check whether a drop is disease or normal late lactation.",
            {"cow_id": _COW_ID, "days": {"type": "integer", "minimum": 1, "maximum": MAX_DETAIL_DAYS}}, ["cow_id"]),
    _schema("get_feed_quality",
            "Feed batch change detection: crude protein before and after, versus the 16% target, and cost per kg.",
            {}),
    _schema("get_heat_stress",
            "Recent temperature-humidity index (THI): hot days, peak THI and heat stress level.",
            {"window": _WINDOW}),
    _schema("get_loss_attribution",
            "Trained regression model: litres per day the herd lost to heat, weaker feed and sick cows this period "
            "compared with the prior period, plus the part the model cannot explain.",
            {"window": _WINDOW}),
    _schema("get_breeding_status",
            "Pregnancy and breeding overview: overdue open cows, pregnancy checks due, upcoming calvings.",
            {}),
    _schema("get_inventory_status",
            "Stock levels, days of supply remaining and items below reorder level (feed, minerals, mastitis tubes).",
            {}),
    _schema("get_finance_summary",
            "Milk revenue, discarded milk value, feed cost and margin this period versus the prior period, in INR.",
            {"window": _WINDOW}),
]}

AGENT_TOOLS: dict[str, list[str]] = {
    "health": ["get_mastitis_suspects", "get_mastitis_risk", "get_cow_detail", "get_top_decliners"],
    "nutrition": ["get_feed_quality", "get_heat_stress", "get_herd_milk_trend", "get_loss_attribution"],
    "breeding": ["get_breeding_status", "get_cow_detail"],
    "inventory": ["get_inventory_status", "get_mastitis_suspects"],
    "finance": ["get_finance_summary", "get_herd_milk_trend", "get_loss_attribution"],
    "manager": ["get_herd_milk_trend"],
}


DECLINERS_NOTE = ("herd_change_pct is how much the whole herd fell. A cow whose change is about the same is NOT "
                  "individually sick: the herd fell for herd-wide reasons. Only cows with a vs_herd_pct far below zero "
                  "fell more than the herd and deserve a health check.")


def _decliners(farm: FarmData, models: Models | None, **kw: Any) -> dict:
    window = kw.get("window", 7)
    herd = analytics.milk_trend(farm, window=window)["change_pct"]
    rows = analytics.top_decliners(farm, window=window, n=min(int(kw.get("n", 5)), MAX_ROWS))
    return {"herd_change_pct": herd, "note": DECLINERS_NOTE,
            "decliners": [{**r, "vs_herd_pct": round(r["change_pct"] - herd, 1)} for r in rows]}


def _risk(farm: FarmData, models: Models | None, **kw: Any) -> dict:
    if models is None:
        return {"error": MODELS_MISSING}
    return {"watch_list": models.risk.watch_list(farm)[:MAX_ROWS], "alert_threshold": round(models.risk.threshold, 2)}


def _loss(farm: FarmData, models: Models | None, **kw: Any) -> dict:
    return {"error": MODELS_MISSING} if models is None else models.attribution.explain(farm, **kw)


_HANDLERS: dict[str, Callable[..., Any]] = {
    "get_herd_milk_trend": lambda farm, models, **kw: analytics.milk_trend(farm, **kw),
    "get_top_decliners": _decliners,
    "get_mastitis_suspects": lambda farm, models, **kw: {"suspects": analytics.mastitis_suspects(farm)[:MAX_ROWS]},
    "get_mastitis_risk": _risk,
    "get_cow_detail": lambda farm, models, **kw: analytics.cow_detail(farm, **kw),
    "get_feed_quality": lambda farm, models, **kw: analytics.feed_quality(farm),
    "get_heat_stress": lambda farm, models, **kw: analytics.heat_stress(farm, **kw),
    "get_loss_attribution": _loss,
    "get_breeding_status": lambda farm, models, **kw: analytics.breeding_status(farm),
    "get_inventory_status": lambda farm, models, **kw: analytics.inventory_status(farm),
    "get_finance_summary": lambda farm, models, **kw: analytics.finance_summary(farm, **kw),
}


def schemas_for(agent: str) -> list[dict]:
    return [TOOL_SCHEMAS[name] for name in AGENT_TOOLS[agent]]


def run_tool(farm: FarmData, agent: str, name: str, args: dict | None, models: Models | None = None) -> dict:
    if agent not in AGENT_TOOLS:
        return {"error": f"unknown agent {agent!r}"}
    if name not in TOOL_SCHEMAS:
        return {"error": f"unknown tool {name!r}"}
    if name not in AGENT_TOOLS[agent]:
        return {"error": f"tool {name!r} is not allowed for the {agent} agent; allowed: {AGENT_TOOLS[agent]}"}
    args = dict(args or {})
    schema = TOOL_SCHEMAS[name]["input_schema"]
    missing = [k for k in schema["required"] if k not in args]
    if missing:
        return {"error": f"missing required argument(s): {', '.join(missing)}"}
    unexpected = sorted(set(args) - set(schema["properties"]))
    if unexpected:
        return {"error": f"unexpected argument(s): {', '.join(unexpected)}; accepted: {sorted(schema['properties'])}"}
    try:
        return _HANDLERS[name](farm, models, **args)
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}
