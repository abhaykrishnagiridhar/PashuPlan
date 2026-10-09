"""PashuPlan Streamlit app: a plain, phone-first report for the farmer, in Kannada, Hindi or English.

The report comes from deterministic analytics plus two trained models. The advisor team (five
specialist agents and a Farm Manager) runs when an AI provider is configured through environment
variables (see pashuplan/providers.py; GEMINI_API_KEY has a free tier). The last answer is saved
so the app can still be shown offline.
"""
from datetime import date

import streamlit as st

from pashuplan import i18n, orchestrator, providers, render, replay, report
from pashuplan import models as model_store
from pashuplan.data_gen import generate_farm

st.set_page_config(page_title="PashuPlan", layout="centered", initial_sidebar_state="collapsed")


@st.cache_resource
def load_farm(today: date):
    """The demo farm always ends on the real current date. Keyed by the day, so it rolls over at midnight."""
    return generate_farm(seed=42, today=today)


@st.cache_resource
def load_models():
    return model_store.load()


@st.cache_resource
def load_report(today: date) -> dict:
    return report.build_report(load_farm(today), models=load_models())


today = date.today()

# A native radio group: reachable with Tab and the arrow keys, read out by screen readers, and it can
# never be left with nothing selected (the first option, Kannada, is the default).
lang = st.radio("Language", i18n.LANGUAGES, format_func=i18n.LANGUAGE_NAMES.get, horizontal=True,
                label_visibility="collapsed", key="language")

try:
    provider, setup_problem = providers.resolve(), None
except ValueError as exc:  # a mistyped setting is shown, not hidden
    provider, setup_problem = None, str(exc)

# Advice is written in one language, so it is kept per language: switching language never shows advice
# that was written in another one.
live = st.session_state.setdefault("advisors", {})
advisor, saved_on = live.get(lang), None
if advisor is None:
    advisor = replay.load_for(lang)
    saved_on = advisor.get("saved_on") if advisor else None
if advisor is None and setup_problem:
    advisor = {"plan": "", "error": setup_problem, "specialists": []}

# The button is always shown. Without a working provider it is disabled, and hovering it says why.
if provider:
    ask_clicked = st.button(i18n.t("run_button", lang), type="primary")
else:
    reason = (i18n.t("advisor_setup_mistake", lang, e=setup_problem) if setup_problem
              else i18n.t("advisor_off_hint", lang))
    ask_clicked = st.button(i18n.t("run_button", lang), type="primary", disabled=True, help=reason)

if provider and ask_clicked:
    with st.spinner(i18n.t("thinking", lang)):
        try:
            advisor = orchestrator.run_advisors(
                provider.make_client(), load_farm(today), i18n.t("question_text", lang), models=load_models(),
                model=provider.model, max_parallel=provider.max_parallel, lang=lang, mode=provider.mode)
            replay.save_if_successful(advisor)
        except Exception as exc:  # shown to the farmer in their language instead of a stack trace
            advisor = {"plan": "", "error": str(exc), "specialists": []}
    live[lang], saved_on = advisor, None

other_languages = [] if advisor else [code for code in i18n.LANGUAGES
                                      if code != lang and (live.get(code) or replay.load_for(code))]
st.html(render.page(load_report(today), lang, advisor=advisor, saved_on=saved_on, other_languages=other_languages))
