"""Drives the real Streamlit app headlessly, the way a farmer would use it."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pashuplan import i18n, providers, replay
from tests.fakes import FunctionClient, reply, text

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _html(at: AppTest) -> str:
    return "".join(str(h.value) for h in at.get("html"))


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """No real key, no saved replay, and nothing written to the repo's replay folder."""
    monkeypatch.setattr(providers, "resolve", lambda *a, **k: None)
    monkeypatch.setattr(replay, "load_for", lambda *a, **k: None)
    saved = []
    monkeypatch.setattr(replay, "save_if_successful",
                        lambda result, *a, **k: saved.append(result) or bool(result.get("plan")))
    return saved


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def test_opens_in_kannada_without_errors():
    at = _run()
    assert not at.exception
    assert 'lang="kn"' in _html(at)
    assert at.radio[0].value == "kn"


def test_the_language_choice_is_a_native_radio_group_in_a_fixed_order():
    """A real radio group is reachable with Tab and the arrow keys and can never be left empty."""
    radio = _run().radio[0]
    assert radio.options == ["ಕನ್ನಡ", "हिन्दी", "English"]
    assert radio.horizontal


def test_the_page_shows_todays_real_date_in_each_language():
    from datetime import date
    at = _run()
    for lang in ("kn", "hi", "en"):
        at.radio[0].set_value(lang).run()
        assert f'<div class="pill">{i18n.format_date(date.today(), lang)}</div>' in _html(at)


def test_every_language_can_be_chosen():
    at = _run()
    for lang in ("hi", "en", "kn"):
        at.radio[0].set_value(lang).run()
        assert not at.exception
        assert f'lang="{lang}"' in _html(at)


def test_without_a_provider_the_advisor_button_is_there_but_disabled_with_the_reason_on_hover():
    at = _run()
    assert len(at.button) == 1
    button = at.button[0]
    assert button.disabled is True
    assert "GEMINI_API_KEY" in button.help  # the hover text says what to set
    assert "ಸಲಹೆಗಾರ" in button.label       # the label is in the chosen language (Kannada by default)


def test_the_disabled_button_explains_itself_in_every_language():
    at = _run()
    helps = {}
    for lang in ("kn", "hi", "en"):
        at.radio[0].set_value(lang).run()
        helps[lang] = at.button[0].help
        assert at.button[0].disabled and "GEMINI_API_KEY" in helps[lang]
    assert len(set(helps.values())) == 3  # a different sentence for each language


def test_a_setup_mistake_is_named_in_the_hover_text_instead_of_a_missing_key(monkeypatch):
    def broken(*a, **k):
        raise ValueError("PASHUPLAN_MODE must be one of compact, agentic (got 'turbo')")

    monkeypatch.setattr(providers, "resolve", broken)
    button = _run().button[0]
    assert button.disabled and "PASHUPLAN_MODE" in button.help and "turbo" in button.help


def test_the_button_is_enabled_and_has_no_warning_once_a_provider_is_set(monkeypatch):
    _with_provider(monkeypatch)
    button = _run().button[0]
    assert button.disabled is False and not button.help


def _with_provider(monkeypatch, client=None, name="fake"):
    provider = providers.Provider(name=name, model="fake-model", max_parallel=2, make_client=lambda: client)
    monkeypatch.setattr(providers, "resolve", lambda *a, **k: provider)
    return provider


def test_advisor_button_appears_when_a_provider_is_configured(monkeypatch):
    _with_provider(monkeypatch)
    assert len(_run().button) == 1


def test_a_misconfigured_provider_shows_a_plain_message_and_the_page_still_works(monkeypatch):
    def broken(*a, **k):
        raise ValueError("PASHUPLAN_MODEL is required")

    monkeypatch.setattr(providers, "resolve", broken)
    at = _run()
    assert not at.exception
    assert "727" in _html(at)


def test_advisor_plan_is_shown_and_saved(monkeypatch, isolated):
    def handler(kwargs):
        is_manager = kwargs["system"].startswith("ROLE: manager")
        return reply(text("PLAN: treat the sick cows first." if is_manager else "report"))

    client = FunctionClient(handler)
    _with_provider(monkeypatch, client)
    at = _run()
    at.button[0].click().run()
    assert not at.exception
    assert "PLAN: treat the sick cows first." in _html(at)
    assert [r["plan"] for r in isolated] == ["PLAN: treat the sick cows first."]
    assert {c["model"] for c in client.calls} == {"fake-model"}  # the provider's model is what gets used


def test_a_failing_advisor_call_shows_the_error_and_keeps_the_page_working(monkeypatch, isolated):
    def down(kwargs):
        raise RuntimeError("401 invalid x-api-key")

    _with_provider(monkeypatch, FunctionClient(down))
    at = _run()
    at.button[0].click().run()
    assert not at.exception
    html = _html(at)
    assert "401 invalid x-api-key" in html
    assert "727" in html  # the farm report is still there
    assert all(not r.get("plan") for r in isolated)  # a failed run is never saved as the replay


def _saved(monkeypatch, **by_language):
    saved = {lang: {"plan": plan, "language": lang, "specialists": [], "error": None, "saved_on": "2026-10-09"}
             for lang, plan in by_language.items()}
    monkeypatch.setattr(replay, "load_for", lambda lang, *a, **k: saved.get(lang))


def test_a_saved_answer_is_shown_offline_with_its_date(monkeypatch):
    _saved(monkeypatch, kn="Saved plan text.")
    html = _html(_run())
    assert "Saved plan text." in html
    assert "9 ಅಕ್ಟೋಬರ್" in html


def test_changing_the_language_never_shows_advice_written_in_another_language(monkeypatch):
    _saved(monkeypatch, kn="KANNADA ADVICE")
    at = _run()
    assert "KANNADA ADVICE" in _html(at)
    at.radio[0].set_value("hi").run()
    html = _html(at)
    assert "KANNADA ADVICE" not in html
    assert "ಕನ್ನಡ" in html  # the note names the language that does have advice
    at.radio[0].set_value("kn").run()
    assert "KANNADA ADVICE" in _html(at)


def test_each_language_shows_its_own_saved_advice(monkeypatch):
    _saved(monkeypatch, kn="KANNADA ADVICE", hi="HINDI ADVICE")
    at = _run()
    at.radio[0].set_value("hi").run()
    assert "HINDI ADVICE" in _html(at) and "KANNADA ADVICE" not in _html(at)


def test_live_advice_is_kept_per_language_in_the_session(monkeypatch, isolated):
    def handler(kwargs):
        is_manager = kwargs["system"].startswith("ROLE: manager")
        language = "KANNADA" if "Respond in Kannada" in kwargs["system"] else "ENGLISH"
        return reply(text(f"{language} PLAN" if is_manager else "report"))

    _with_provider(monkeypatch, FunctionClient(handler))
    at = _run()
    at.button[0].click().run()
    assert "KANNADA PLAN" in _html(at)
    at.radio[0].set_value("en").run()
    assert "KANNADA PLAN" not in _html(at)  # not carried into English
    at.button[0].click().run()
    assert "ENGLISH PLAN" in _html(at) and "KANNADA PLAN" not in _html(at)
    at.radio[0].set_value("kn").run()
    assert "KANNADA PLAN" in _html(at) and "ENGLISH PLAN" not in _html(at)
