"""Turns analytics facts into a plain-language report structure for the farmer-facing screen.

The result is a JSON-safe dict: no text, only facts and keys. `render` turns it into HTML in
the farmer's language, and later the agents can fill the same structure.
"""
from __future__ import annotations

from pashuplan import analytics
from pashuplan.data_gen import FarmData
from pashuplan.models import Models

CAUSE_FACTOR = {"sick_cows": "sick", "feed": "feed", "heat": "heat"}
MIN_HOT_DAYS = 3
MIN_PROTEIN_DROP = 1.0
BAR_DAYS = 14
URGENCY_ORDER = {"today": 0, "week": 1, "later": 2}

# Which advisor's data each piece of the screen comes from (each advisor owns the tools behind it, see tools.AGENT_TOOLS)
AGENT_FOR = {"sick_cows": "health", "feed": "nutrition", "heat": "nutrition",
             "treat_cows": "health", "buy_tubes": "inventory", "fix_feed": "nutrition",
             "protect_from_heat": "nutrition", "check_breeding": "breeding"}
SOURCES = {"milk": "manager", "chart": "manager", "money": "finance", "fine": "health", "watch": "health"}


def _daily_bars(farm: FarmData) -> list[dict]:
    daily = farm.milk.groupby("date")["litres"].sum().sort_index().tail(BAR_DAYS)
    half = len(daily) // 2
    return [{"date": d.isoformat(), "litres": round(float(v)), "period": "prior" if i < half else "this"}
            for i, (d, v) in enumerate(daily.items())]


def _causes(suspects: list[dict], feed: dict, heat: dict) -> list[dict]:
    causes = []
    if suspects:
        causes.append({"key": "sick_cows", "cows": sorted(s["cow_id"] for s in suspects)})
    if feed["change_detected"] and feed["below_target"] and feed["cp_delta"] <= -MIN_PROTEIN_DROP:
        causes.append({"key": "feed", "cp_before": feed["cp_before"], "cp_after": feed["cp_after"],
                       "date": feed["change_date"]})
    if heat["days_at_or_above_mild"] >= MIN_HOT_DAYS:
        causes.append({"key": "heat", "hot_days": heat["days_at_or_above_mild"], "avg_temp_c": heat["avg_temp_c"]})
    return [{**c, "agent": AGENT_FOR[c["key"]]} for c in causes]


def _feed_saving_per_day(farm: FarmData, feed: dict) -> float:
    kg_per_cow = float(farm.feed["kg_per_cow"].iloc[-1])
    return (feed["cost_per_kg_before"] - feed["cost_per_kg_after"]) * kg_per_cow * len(farm.cows)


def _actions(farm: FarmData, causes: list[dict], suspects: list[dict], feed: dict, loss_per_day: float) -> list[dict]:
    keys = {c["key"] for c in causes}
    actions = []
    if suspects:
        actions.append({"key": "treat_cows", "when": "today",
                        "cows": [{"cow_id": s["cow_id"], "discard_until": s["discard_until"]}
                                 for s in sorted(suspects, key=lambda s: s["cow_id"])]})
    tubes = next((i for i in analytics.inventory_status(farm)["items"]
                  if i["item"] == "mastitis_tubes" and i["below_reorder_level"]), None)
    if tubes:
        actions.append({"key": "buy_tubes", "when": "today", "days_left": int(tubes["days_remaining"]),
                        "qty": int(tubes["qty"]), "daily_use": int(tubes["daily_use"])})
    if "feed" in keys:
        saving = _feed_saving_per_day(farm, feed)
        if saving > 0:
            actions.append({"key": "fix_feed", "when": "week", "cp_before": feed["cp_before"],
                            "saving_per_day_inr": round(saving), "milk_loss_per_day_inr": round(loss_per_day)})
    if "heat" in keys:
        actions.append({"key": "protect_from_heat", "when": "week"})
    overdue = analytics.breeding_status(farm)["overdue_open_cows"]
    if overdue:
        actions.append({"key": "check_breeding", "when": "later", "cows": overdue})
    return sorted(({**a, "agent": AGENT_FOR[a["key"]]} for a in actions), key=lambda a: URGENCY_ORDER[a["when"]])


def _add_litres_lost(causes: list[dict], farm: FarmData, models: Models) -> None:
    lost = models.attribution.explain(farm)["causes"]
    for cause in causes:
        cause["loss_l"] = round(lost[CAUSE_FACTOR[cause["key"]]])


def build_report(farm: FarmData, models: Models | None = None) -> dict:
    """models is optional: with it, each cause gets litres lost per day and the watch list is filled."""
    trend = analytics.milk_trend(farm)
    fin = analytics.finance_summary(farm)
    suspects = analytics.mastitis_suspects(farm)
    feed = analytics.feed_quality(farm)
    heat = analytics.heat_stress(farm)
    causes = _causes(suspects, feed, heat)
    if models is not None:
        _add_litres_lost(causes, farm, models)
    loss_per_day = -trend["change_l_per_day"] * farm.config["milk_price_per_l"]
    sick_ids = {s["cow_id"] for s in suspects}
    late = farm.cows.loc[farm.cows["dim"] > analytics.LATE_LACTATION_DIM, "cow_id"]
    return {
        "date": farm.today.isoformat(),
        "milk": {
            "per_day": round(trend["this_period_avg_l_per_day"]),
            "prior_avg": round(trend["prior_period_avg_l_per_day"]),
            "this_avg": round(trend["this_period_avg_l_per_day"]),
            "change_pct": trend["change_pct"],
            "litres_less_per_day": round(-trend["change_l_per_day"]),
        },
        "money": {
            "revenue_change_inr": round(fin["revenue_change_inr"]),
            "discarded_litres": round(fin["discarded_milk_litres"]),
            "discarded_value_inr": round(fin["discarded_milk_value_inr"]),
        },
        "causes": causes,
        "watch_list": [w["cow_id"] for w in models.risk.watch_list(farm)] if models is not None else [],
        "fine_cows": sorted(c for c in late if c not in sick_ids),
        "actions": _actions(farm, causes, suspects, feed, loss_per_day),
        "bars": _daily_bars(farm),
        "sources": dict(SOURCES),
    }
