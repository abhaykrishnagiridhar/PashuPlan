"""Chooses the AI provider from environment variables. Keys are only ever read from the environment.

  GEMINI_API_KEY (or GOOGLE_API_KEY)   Google Gemini, which has a free tier
  ANTHROPIC_API_KEY                    Anthropic Claude
  PASHUPLAN_BASE_URL + PASHUPLAN_MODEL any OpenAI-compatible server (Ollama, Groq, OpenRouter);
                                       PASHUPLAN_API_KEY is optional
  PASHUPLAN_PROVIDER                   force one of: anthropic, gemini, openai
  PASHUPLAN_MODEL, PASHUPLAN_MAX_PARALLEL   override the model and how many advisors run at once
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from pashuplan.llm import OpenAICompatClient

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "gemini": "gemini-3.8-flash"}
DEFAULT_PARALLEL = {"anthropic": 5, "gemini": 1, "openai-compatible": 1}  # free tiers allow only a few requests a minute
GEMINI_MAX_PARALLEL = DEFAULT_PARALLEL["gemini"]
GEMINI_MIN_OUTPUT_TOKENS = 4096  # thinking tokens share the output budget
ORDER = ("anthropic", "gemini", "openai-compatible")
ALIASES = {"openai": "openai-compatible", "openai-compatible": "openai-compatible",
           "anthropic": "anthropic", "gemini": "gemini", "google": "gemini"}
NOT_CONFIGURED = (
    "No AI provider is set up. For a free key, create a Gemini API key at aistudio.google.com and set "
    "GEMINI_API_KEY. Or set ANTHROPIC_API_KEY. Or set PASHUPLAN_BASE_URL and PASHUPLAN_MODEL to use an "
    "OpenAI-compatible server such as Ollama.")


MODES = ("compact", "agentic")


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    max_parallel: int
    make_client: Callable[[], object]
    mode: str = "compact"  # compact: one request per advisor. agentic: the model chooses and calls tools.


def _max_parallel(env, name: str) -> int:
    raw = env.get("PASHUPLAN_MAX_PARALLEL")
    if raw is None:
        return DEFAULT_PARALLEL[name]
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError("PASHUPLAN_MAX_PARALLEL must be a whole number of 1 or more")
    return value


def resolve(env=None) -> Provider | None:
    """The configured provider, or None if no key or server is set. Bad settings raise ValueError."""
    env = os.environ if env is None else env
    explicit = (env.get("PASHUPLAN_PROVIDER") or "").strip().lower()
    if explicit and explicit not in ALIASES:
        raise ValueError(f"PASHUPLAN_PROVIDER must be one of anthropic, gemini, openai (got {explicit!r})")
    gemini_key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
    available = {"anthropic": bool(env.get("ANTHROPIC_API_KEY")), "gemini": bool(gemini_key),
                 "openai-compatible": bool(env.get("PASHUPLAN_BASE_URL"))}
    name = ALIASES[explicit] if explicit else next((n for n in ORDER if available[n]), None)
    if name is None or not available[name]:
        return None
    model = env.get("PASHUPLAN_MODEL") or DEFAULT_MODELS.get(name)
    if not model:
        raise ValueError("PASHUPLAN_MODEL is required for an OpenAI-compatible server")
    parallel = _max_parallel(env, name)
    mode = (env.get("PASHUPLAN_MODE") or "compact").strip().lower()
    if mode not in MODES:
        raise ValueError(f"PASHUPLAN_MODE must be one of {', '.join(MODES)} (got {mode!r})")
    if name == "anthropic":
        key = env["ANTHROPIC_API_KEY"]

        def make_client():
            import anthropic
            return anthropic.Anthropic(api_key=key)
    elif name == "gemini":
        def make_client():
            return OpenAICompatClient(GEMINI_BASE_URL, gemini_key, min_max_tokens=GEMINI_MIN_OUTPUT_TOKENS,
                                      extra_body={"reasoning_effort": "low"})
    else:
        base_url, key = env["PASHUPLAN_BASE_URL"], env.get("PASHUPLAN_API_KEY", "")

        def make_client():
            return OpenAICompatClient(base_url, key)
    return Provider(name=name, model=model, max_parallel=parallel, make_client=make_client, mode=mode)
