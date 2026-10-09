"""Command line: `python -m pashuplan.ask [--question TEXT] [--lang kn|hi|en]` asks the advisor team.

The provider comes from environment variables (see providers.py): GEMINI_API_KEY for Google's free
tier, ANTHROPIC_API_KEY, or an OpenAI-compatible server. The answer is saved for offline replay.
"""
from __future__ import annotations

import argparse
import sys

from pashuplan import agents, i18n, models as model_store, orchestrator, providers, replay
from pashuplan.data_gen import FarmData, generate_farm

_LOAD = object()


def main(argv: list[str] | None = None, client=None, farm: FarmData | None = None, models=_LOAD) -> dict:
    parser = argparse.ArgumentParser(description="Ask the PashuPlan advisor team a question about the farm.")
    parser.add_argument("--question", default=None, help="the farmer's question (default: why is my milk down)")
    parser.add_argument("--lang", choices=i18n.LANGUAGES, default=None, help="answer language (default: from the question)")
    parser.add_argument("--replay-file", default=None,
                        help="where to save the answer (default: replay/advisors_<language>.json)")
    parser.add_argument("--model", default=None, help="override the provider's default model")
    parser.add_argument("--mode", choices=sorted(orchestrator.MODES), default=None,
                        help="compact: one request per advisor (default). agentic: advisors choose and call tools")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    question = args.question or i18n.t("question_text", args.lang or i18n.DEFAULT_LANGUAGE)
    provider = None
    if client is None:
        try:
            provider = providers.resolve()
        except ValueError as exc:
            sys.exit(str(exc))
        if provider is None:
            sys.exit(providers.NOT_CONFIGURED)
        client = provider.make_client()
    result = orchestrator.run_advisors(
        client, farm or generate_farm(seed=42), question,
        models=model_store.load() if models is _LOAD else models, lang=args.lang,
        model=args.model or (provider.model if provider else agents.DEFAULT_MODEL),
        max_parallel=provider.max_parallel if provider else None,
        mode=args.mode or (provider.mode if provider else orchestrator.DEFAULT_MODE))
    if not result["plan"]:
        problems = dict.fromkeys(e for e in [result["error"], *(s["error"] for s in result["specialists"])] if e)
        sys.exit("The advisors could not make a plan.\n" + "\n".join(problems))
    print(result["plan"])
    if not replay.save_if_successful(result, args.replay_file):
        missing = "; ".join(f"{s['agent']} ({s['error']})" for s in result["specialists"] if s["error"])
        print(f"Warning: some advisors could not answer: {missing}. This answer was not saved for offline replay.",
              file=sys.stderr)
    return result


if __name__ == "__main__":
    main()
