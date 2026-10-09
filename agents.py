"""One advisor agent, in two modes.

compact (the default): our code runs the advisor's no-argument data tools and hands the results to the
model in a single message, so each advisor costs one model request.
agentic: a tool-use loop. The model chooses its own tools for up to MAX_TURNS turns, tool errors go back
to it as results so it can correct itself, and at the cap one last request without tools forces an answer.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from pashuplan import i18n, tools
from pashuplan.data_gen import FarmData
from pashuplan.models import Models

MAX_TURNS = 6
MAX_TOKENS = 1024
DEFAULT_MODEL = os.environ.get("PASHUPLAN_MODEL", "claude-sonnet-5-5")

ROLES: dict[str, tuple[str, str]] = {
    "health": ("Health", "You judge whether a cow's low milk is disease or normal late lactation. "
                         "Call a cow sick only if get_mastitis_suspects lists it, and at risk only if get_mastitis_risk "
                         "lists it. A cow whose milk fell about as much as the herd average is not ill: the whole herd "
                         "is declining for herd-wide reasons such as feed and heat."),
    "nutrition": ("Nutrition", "You judge feed quality and heat stress, and how much milk each cost the herd."),
    "breeding": ("Breeding", "You check pregnancy status, overdue open cows and upcoming calvings."),
    "inventory": ("Inventory", "You check whether feed, minerals and mastitis medicine will run out."),
    "finance": ("Finance", "You work out the money lost, the milk that cannot be sold and the effect on margin. "
                           "Discarded milk is milk held back during the medicine waiting period after treatment; it is "
                           "not spoiled milk, so never say it spoiled."),
    "manager": ("Farm Manager", "You read the specialist reports and write one prioritised plan: what to do today, "
                                "this week and later. Do not add facts that are not in the reports."),
}
# Only the tool-using mode has a tool that can look at one cow
AGENTIC_EXTRA = {"health": " Use get_cow_detail to check any cow that fell far more than the herd."}
SPECIALIST_WORDS = 150
MANAGER_WORDS = 220
TOOL_RULES = ("Use your tools to look at the farm's data before you answer. "
              "Only use numbers that appear in tool results; never estimate or invent numbers. "
              "If a tool returns an error, read it and correct your call. ")
DATA_RULES = ("The farm data you need is given in the message, fetched for you by our tools. "
              "Only use numbers that appear in that data; never estimate or invent numbers. "
              "If a section of the data shows an error, say that check is not available. ")
PLAIN_RULES = ("Write in plain words a farmer with little schooling understands. Never use technical terms or "
               "abbreviations such as THI, SCC or mastitis: say 'very hot weather' and 'infection in the udder' "
               "instead. Write money with the ₹ sign. No tables, no emoji and no dashes between clauses. "
               "Never ask the farmer a question or ask for more information: you cannot receive replies, so ")
ASK_RULE_AGENTIC = "use your tools, and if something is not available, state that as a fact."
ASK_RULE_COMPACT = "work from the data given, and if something is not available, state that as a fact."
SPECIALIST_RULES = f"Answer in at most {SPECIALIST_WORDS} words."
MANAGER_RULES = (f"Write at most {MANAGER_WORDS} words. Put what to do today first, then this week, then later. "
                 "Include every urgent problem the specialists found, especially medicine or feed stock running out "
                 "and the money lost. If a specialist is unavailable, say so in one short sentence so the farmer "
                 "knows that check did not happen. Ignore any question a specialist asks; do not repeat it.")
FINAL_NUDGE = "You have used all your tool turns. Give your final answer now from what you already know."


@dataclass(frozen=True)
class AgentResult:
    agent: str
    answer: str
    tool_calls: tuple
    turns: int
    hit_turn_limit: bool


def system_prompt(agent: str, lang: str, compact: bool = False) -> str:
    label, focus = ROLES[agent]
    focus += "" if compact else AGENTIC_EXTRA.get(agent, "")
    common = (DATA_RULES + PLAIN_RULES + ASK_RULE_COMPACT) if compact else (TOOL_RULES + PLAIN_RULES + ASK_RULE_AGENTIC)
    rules = MANAGER_RULES if agent == "manager" else SPECIALIST_RULES
    return (f"ROLE: {agent}\nYou are the {label} advisor on a dairy farm. {focus}\n{common} {rules}\n"
            f"{i18n.language_instruction(lang)}")


def _block_dicts(blocks) -> list[dict]:
    out = []
    for b in blocks:
        if b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
        else:
            out.append({"type": "text", "text": b.text})
    return out


def _text_of(blocks) -> str:
    return "\n".join(b.text for b in blocks if b.type == "text").strip()


def compact_tools(agent: str) -> list[str]:
    """The agent's tools that need no argument, so our code can run them without asking the model."""
    return [name for name in tools.AGENT_TOOLS[agent] if not tools.TOOL_SCHEMAS[name]["input_schema"]["required"]]


def gather_facts(farm: FarmData, agent: str, models: Models | None) -> tuple[str, list[dict]]:
    """Runs the agent's no-argument tools. Returns the facts as text for the model and a log of the lookups."""
    sections, calls = [], []
    for name in compact_tools(agent):
        out = tools.run_tool(farm, agent, name, {}, models=models)
        calls.append({"name": name, "args": {}, "error": "error" in out})
        sections.append(f"[{name}]\n{json.dumps(out, ensure_ascii=False)}")
    return "\n\n".join(sections), calls


def run_agent_compact(client, farm: FarmData, agent: str, question: str, lang: str, models: Models | None = None,
                      model: str = DEFAULT_MODEL) -> AgentResult:
    """One model request: the farm data is fetched by our code and included in the message."""
    facts, calls = gather_facts(farm, agent, models)
    reply = client.messages.create(
        model=model, max_tokens=MAX_TOKENS, system=system_prompt(agent, lang, compact=True),
        messages=[{"role": "user", "content": f"{question}\n\nFarm data from our tools:\n\n{facts}"}])
    return AgentResult(agent, _text_of(reply.content), tuple(calls), 1, False)


def run_agent(client, farm: FarmData, agent: str, question: str, lang: str, models: Models | None = None,
              model: str = DEFAULT_MODEL, max_turns: int = MAX_TURNS) -> AgentResult:
    system = system_prompt(agent, lang)
    schemas = tools.schemas_for(agent)
    messages: list[dict] = [{"role": "user", "content": question}]
    calls: list[dict] = []
    for turn in range(1, max_turns + 1):
        reply = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=system, tools=schemas,
                                       messages=messages)
        uses = [b for b in reply.content if b.type == "tool_use"]
        if not uses:
            return AgentResult(agent, _text_of(reply.content), tuple(calls), turn, False)
        messages.append({"role": "assistant", "content": _block_dicts(reply.content)})
        results = []
        for use in uses:
            out = tools.run_tool(farm, agent, use.name, use.input, models=models)
            failed = "error" in out
            calls.append({"name": use.name, "args": dict(use.input), "error": failed})
            block = {"type": "tool_result", "tool_use_id": use.id, "content": json.dumps(out)}
            if failed:
                block["is_error"] = True
            results.append(block)
        messages.append({"role": "user", "content": results})
    messages[-1] = {"role": "user", "content": [*messages[-1]["content"], {"type": "text", "text": FINAL_NUDGE}]}
    final = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=system, messages=messages)
    return AgentResult(agent, _text_of(final.content), tuple(calls), max_turns, True)
