"""Saves advisor answers, one per language, so the app can be demoed offline.

Advice is written in one language, so each language keeps its own file (replay/advisors_kn.json and so on).
The older single file replay/advisors.json is still read, but only for the language written inside it.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pashuplan import i18n

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replay"
LEGACY_NAME = "advisors.json"


def path_for(lang: str, directory: Path | str | None = None) -> Path:
    return Path(directory or REPLAY_DIR) / f"advisors_{lang}.json"


def save(result: dict, path: Path | str, saved_on: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**result, "saved_on": saved_on or date.today().isoformat()}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def save_if_successful(result: dict, path: Path | str | None = None, directory: Path | str | None = None) -> bool:
    """Saves only complete runs: a plan and every specialist answered. A failed or partial run (for example
    one advisor hit a rate limit) must never replace a good saved answer or be shown later as the demo.
    Without an explicit path the answer goes to the file for its own language."""
    if not result.get("plan") or any(s.get("error") for s in result.get("specialists", [])):
        return False
    save(result, path or path_for(result.get("language", i18n.DEFAULT_LANGUAGE), directory))
    return True


def load(path: Path | str) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"replay file {path} is corrupt: {exc}") from exc


def load_for(lang: str, directory: Path | str | None = None) -> dict | None:
    """The saved advice written in this language, or None. Never returns advice in another language."""
    own = load(path_for(lang, directory))
    if own:
        return own
    legacy = load(Path(directory or REPLAY_DIR) / LEGACY_NAME)
    return legacy if legacy and legacy.get("language") == lang else None


def available_languages(directory: Path | str | None = None) -> list[str]:
    return [lang for lang in i18n.LANGUAGES if load_for(lang, directory)]
