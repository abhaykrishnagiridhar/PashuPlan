"""Tool registry for the agent tool-use loop.

Schemas are in Anthropic tool format. `run_tool` never raises: any failure is returned as
{"error": "..."} so the model can read it and correct its next call.
"""
from __future__ import annotations

from typing import Any, Callable

from pashuplan import analytics
from pashuplan.data_gen import FarmData

MAX_ROWS = 10
MAX_DETAIL_DAYS = 30

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
            "Cows whose average daily milk fell the most versus the prior period, most negative first.",
            {"window": _WINDOW, "n": {"type": "integer", "description": f"How many cows (max {MAX_ROWS}).", "minimum": 1}}),
    _schema("get_mastitis_suspects",
            "Cows with high somatic cell count AND fever in the last 3 days, with treatment and milk discard status.",
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
    "health": ["get_mastitis_suspects", "get_cow_detail", "get_top_decliners"],
    "nutrition": ["get_feed_quality", "get_heat_stress", "get_herd_milk_trend"],
    "breeding": ["get_breeding_status", "get_cow_detail"],
    "inventory": ["get_inventory_status", "get_mastitis_suspects"],
    "finance": ["get_finance_summary", "get_herd_milk_trend"],
    "manager": ["get_herd_milk_trend"],
}

_HANDLERS: dict[str, Callable[..., Any]] = {
    "get_herd_milk_trend": lambda farm, **kw: analytics.milk_trend(farm, **kw),
    "get_top_decliners": lambda farm, **kw: {"decliners": analytics.top_decliners(
        farm, **{**kw, "n": min(int(kw.get("n", 5)), MAX_ROWS)})},
    "get_mastitis_suspects": lambda farm, **kw: {"suspects": analytics.mastitis_suspects(farm)[:MAX_ROWS]},
    "get_cow_detail": lambda farm, **kw: analytics.cow_detail(farm, **kw),
    "get_feed_quality": lambda farm, **kw: analytics.feed_quality(farm),
    "get_heat_stress": lambda farm, **kw: analytics.heat_stress(farm, **kw),
    "get_breeding_status": lambda farm, **kw: analytics.breeding_status(farm),
    "get_inventory_status": lambda farm, **kw: analytics.inventory_status(farm),
    "get_finance_summary": lambda farm, **kw: analytics.finance_summary(farm, **kw),
}


def schemas_for(agent: str) -> list[dict]:
    return [TOOL_SCHEMAS[name] for name in AGENT_TOOLS[agent]]


def run_tool(farm: FarmData, agent: str, name: str, args: dict | None) -> dict:
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
        return _HANDLERS[name](farm, **args)
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}
