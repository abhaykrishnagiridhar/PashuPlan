"""Renders the farmer report as plain HTML in English or Hindi.

Everything printed goes through `escape`. Numbers stay in Latin digits, rupee amounts use the
Indian grouping (1,00,000), and cows are drawn as ear tags.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

from pashuplan import i18n

CSS_PATH = Path(__file__).with_name("ui.css")
MAX_BAR_PERCENT = 100
CAUSE_PREFIX = {"sick_cows": "sick", "feed": "feed", "heat": "heat"}


def inr(amount: float) -> str:
    """Indian digit grouping: 1234567 -> 12,34,567."""
    digits = str(int(round(abs(amount))))
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def _t(key: str, lang: str, **fmt: object) -> str:
    return escape(i18n.t(key, lang, **fmt))


def _chip(agent: str | None, lang: str) -> str:
    """A small label naming the advisor whose data a block comes from. Blank when the report does not say."""
    return f'<span class="agent">{escape(i18n.agent_label(agent, lang))}</span>' if agent else ""


def _tag(cow_id: str, sub: str = "") -> str:
    small = f"<small>{escape(sub)}</small>" if sub else ""
    return f'<span class="tag"><span class="eartag-text"><b>{escape(str(cow_id))}</b>{small}</span></span>'


def _tags(cow_ids: list[str]) -> str:
    return f'<div class="tags">{"".join(_tag(c) for c in cow_ids)}</div>' if cow_ids else ""


def _cause(cause: dict, lang: str) -> str:
    key = cause["key"]
    params = {
        "sick_cows": {},
        "feed": {"d": i18n.format_date(cause["date"], lang) if cause.get("date") else "",
                 "b": cause.get("cp_before"), "a": cause.get("cp_after")},
        "heat": {"n": cause.get("hot_days"), "t": round(cause.get("avg_temp_c", 0))},
    }[key]
    prefix = CAUSE_PREFIX[key]
    cost = f'<p class="cost">{_t("cause_cost", lang, l=cause["loss_l"])}</p>' if "loss_l" in cause else ""
    return (f'<article class="cause">{_chip(cause.get("agent"), lang)}<h3>{_t(f"cause_{prefix}_title", lang)}</h3>'
            f'<p>{_t(f"cause_{prefix}_body", lang, **params)}</p>{cost}{_tags(cause.get("cows", []))}</article>')


def _action_parts(action: dict, lang: str) -> tuple[str, str, str]:
    """Returns (string-key prefix, body text, extra HTML such as ear tags)."""
    key = action["key"]
    if key == "treat_cows":
        tags = "".join(
            _tag(c["cow_id"], i18n.t("until_date", lang, d=i18n.format_date(c["discard_until"], lang))
                 if c.get("discard_until") else "")
            for c in action["cows"])
        return "act_treat", _t("act_treat_body", lang), f'<div class="tags">{tags}</div>'
    if key == "buy_tubes":
        return "act_tubes", _t("act_tubes_body", lang, n=action["days_left"], q=action["qty"],
                               u=action["daily_use"]), ""
    if key == "fix_feed":
        return "act_feed", _t("act_feed_body", lang, b=action["cp_before"], s=inr(action["saving_per_day_inr"]),
                              l=inr(action["milk_loss_per_day_inr"])), ""
    if key == "protect_from_heat":
        return "act_heat", _t("act_heat_body", lang), ""
    return "act_breed", _t("act_breed_body", lang), _tags(action["cows"])


def _action(action: dict, lang: str) -> str:
    prefix, text, extra = _action_parts(action, lang)
    when = action["when"]
    return (f'<li class="act"><div class="meta">{_chip(action.get("agent"), lang)}'
            f'<span class="when {when}">{_t(f"when_{when}", lang)}</span></div>'
            f'<h3>{_t(f"{prefix}_title", lang)}</h3><p>{text}</p>{extra}</li>')


def _bars(rep: dict, lang: str) -> str:
    bars = rep["bars"]
    top = max(b["litres"] for b in bars) or 1
    cells = "".join(f'<i class="bar {b["period"]}" style="height:{b["litres"] / top * MAX_BAR_PERCENT:.0f}%;--i:{i}"></i>'
                    for i, b in enumerate(bars))
    milk = rep["milk"]
    return (f'<div class="chart"><h2>{_t("chart_heading", lang)}</h2>{_chip(rep.get("sources", {}).get("chart"), lang)}'
            f'<div class="bars" role="img" aria-label="{_t("chart_heading", lang)}">{cells}</div>'
            f'<div class="legend"><div><span class="sw prior"></span>{_t("chart_last_week", lang)}'
            f'<b>{_t("chart_avg", lang, l=milk["prior_avg"])}</b></div>'
            f'<div><span class="sw this"></span>{_t("chart_this_week", lang)}'
            f'<b>{_t("chart_avg", lang, l=milk["this_avg"])}</b></div></div></div>')


def _money(rep: dict, lang: str) -> str:
    m = rep["money"]
    chip = _chip(rep.get("sources", {}).get("money"), lang)
    return (f'<div class="money"><h2>{_t("money_heading", lang)}</h2>'
            f'<p class="fig">{chip}<span class="amt">₹{inr(m["revenue_change_inr"])}</span>'
            f'<span>{_t("money_lost", lang)}</span></p>'
            f'<p class="fig">{chip}<span class="amt">₹{inr(m["discarded_value_inr"])}</span>'
            f'<span>{_t("money_discard", lang, l=m["discarded_litres"])}</span></p></div>')


def _specialist(s: dict, lang: str) -> str:
    """One advisor's line: their answer opens under their own name; one that failed shows only a plain status."""
    name = f'<b>{escape(i18n.agent_label(s["agent"], lang))}</b>'
    if s.get("error") or not s.get("answer"):
        return f'<li>{name} {_t("advisor_unavailable", lang)}</li>'
    looked = _t("tool_calls_count", lang, n=len(s["tool_calls"]))
    return (f'<li><details class="said"><summary>{name} {looked}</summary>'
            f'<p class="said-text">{escape(s["answer"])}</p></details></li>')


def _advisors(advisor: dict, lang: str, saved_on: str | None) -> str:
    """The advisor team's plan (the Farm Manager's answer) plus one line per specialist."""
    saved = (f'<p class="note">{_t("replay_saved", lang, d=i18n.format_date(saved_on, lang))}</p>'
             if saved_on else "")
    if advisor.get("plan"):
        paragraphs = [p.strip() for p in advisor["plan"].split("\n\n") if p.strip()]
        plan = _chip("manager", lang) + "".join(f'<p class="plan">{escape(p)}</p>' for p in paragraphs)
    else:
        plan = f'<p class="plan">{_t("advisor_error", lang)}</p>'
    lines = "".join(_specialist(s, lang) for s in advisor["specialists"])
    problems = dict.fromkeys(e for e in [advisor.get("error"), *(s.get("error") for s in advisor["specialists"])] if e)
    details = (f'<details class="tech"><summary>{_t("advisor_details", lang)}</summary>'
               + "".join(f"<p>{escape(p)}</p>" for p in problems) + "</details>") if problems else ""
    return (f'<section class="card paper advisors"><h2>{_t("advisor_heading", lang)}</h2>{saved}{plan}'
            f'<ul class="looked">{lines}</ul>{details}</section>')


def _other_language_note(other_languages: list[str], lang: str) -> str:
    """Shown when advice exists, but only in other languages: say so instead of showing the wrong language."""
    names = ", ".join(i18n.LANGUAGE_NAMES[code] for code in other_languages)
    return (f'<section class="card paper advisors"><h2>{_t("advisor_heading", lang)}</h2>'
            f'<p class="plan">{_t("advisor_other_language", lang, l=names)}</p></section>')


def body(rep: dict, lang: str = "en", advisor: dict | None = None, saved_on: str | None = None,
         other_languages: list[str] | tuple = ()) -> str:
    milk = rep["milk"]
    sources = rep.get("sources", {})
    causes = "".join(_cause(c, lang) for c in rep["causes"])
    fine = ""
    if rep["fine_cows"]:
        fine = (f'<article class="cause fine">{_chip(sources.get("fine"), lang)}<h3>{_t("fine_title", lang)}</h3>'
                f'<p>{_t("fine_body", lang)}</p>{_tags(rep["fine_cows"])}</article>')
    if rep.get("watch_list"):
        fine = (f'<article class="cause watch">{_chip(sources.get("watch"), lang)}<h3>{_t("watch_title", lang)}</h3>'
                f'<p>{_t("watch_body", lang)}</p>'
                f'{_tags(rep["watch_list"])}</article>') + fine
    actions = "".join(_action(a, lang) for a in rep["actions"])
    delta = _t("delta_text", lang, p=round(abs(milk["change_pct"])), l=milk["litres_less_per_day"])
    causes_html = (f'<section class="card linen causes"><h2>{_t("causes_heading", lang)}</h2>{causes}{fine}</section>'
                   if causes or fine else "")
    actions_html = (f'<section class="card paper steps"><h2>{_t("actions_heading", lang)}</h2>'
                    f'<ol class="actions">{actions}</ol></section>' if actions else "")
    advice_html = (_advisors(advisor, lang, saved_on) if advisor
                   else (_other_language_note(list(other_languages), lang) if other_languages else ""))
    return (
        f'<main class="pp" lang="{escape(lang)}">'
        f'<header class="top"><div class="mark">PashuPlan</div>'
        f'<div class="pill">{escape(i18n.format_date(rep["date"], lang))}</div></header>'
        f'<section class="hero"><p class="pill">{_t("question_label", lang)}</p>'
        f'<h1>{_t("question_text", lang)}</h1>{_chip(sources.get("milk"), lang)}'
        f'<div class="big"><span class="num">{milk["per_day"]}</span><span class="unit">{_t("milk_unit", lang)}</span></div>'
        f'<p class="delta pill">{delta}</p></section>'
        f'<div class="main-col">{advice_html}{causes_html}{actions_html}</div>'
        f'<section class="card dark">{_money(rep, lang)}{_bars(rep, lang)}</section>'
        f'<p class="note">{_t("source_note", lang)}</p><p class="note">{_t("sample_note", lang)}</p></main>'
    )


def page(rep: dict, lang: str = "en", advisor: dict | None = None, saved_on: str | None = None,
         other_languages: list[str] | tuple = ()) -> str:
    return f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>{body(rep, lang, advisor, saved_on, other_languages)}"
