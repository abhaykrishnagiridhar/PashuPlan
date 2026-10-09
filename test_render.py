import copy
import re

import pytest

from pashuplan import i18n, render, report


@pytest.fixture(scope="module")
def rep(farm):
    return report.build_report(farm)


def _text(html: str) -> str:
    return re.sub(r"<style.*?</style>|<[^>]+>", " ", html, flags=re.S)


@pytest.mark.parametrize("lang", ["kn", "hi", "en"])
def test_every_cow_tag_is_shown(farm, rep, lang):
    html = render.body(rep, lang)
    gt = farm.ground_truth
    for cow in gt["mastitis_cows"] + gt["overdue_breeding_cows"] + rep["fine_cows"]:
        assert cow in html


@pytest.mark.parametrize("lang", ["kn", "hi", "en"])
def test_copy_has_no_dashes_emoji_or_unresolved_placeholders(rep, lang):
    text = _text(render.page(rep, lang))
    assert "—" not in text and "–" not in text
    assert not any(ord(ch) >= 0x1F000 for ch in text)
    assert "{" not in text and "}" not in text


def test_hindi_page_is_hindi_with_latin_digits(rep):
    text = _text(render.body(rep, "hi"))
    assert i18n.detect_language(text) == "hi"
    assert not any("०" <= ch <= "९" for ch in text)


def test_kannada_page_is_kannada_with_latin_digits(rep):
    text = _text(render.body(rep, "kn"))
    assert i18n.detect_language(text) == "kn"
    assert not any("೦" <= ch <= "೯" for ch in text)
    assert not any("ऀ" <= ch <= "ॿ" for ch in text)


def test_english_page_has_no_indian_script(rep):
    text = _text(render.body(rep, "en"))
    assert not any("ऀ" <= ch <= "ॿ" or "ಀ" <= ch <= "೿" for ch in text)


def test_page_declares_its_language_and_loads_a_kannada_font(rep):
    html = render.page(rep, "kn")
    assert 'lang="kn"' in html
    assert "Noto Sans Kannada" in html


def test_kannada_headings_use_a_face_that_has_kannada_glyphs(rep):
    css = render.CSS_PATH.read_text(encoding="utf-8")
    assert '[lang="kn"]' in css


def test_urgency_labels_are_words(rep):
    html = render.body(rep, "en")
    assert "TODAY" in html and "THIS WEEK" in html and "LATER" in html


def test_headline_numbers_are_present(rep):
    html = render.body(rep, "en")
    assert str(rep["milk"]["per_day"]) in html
    assert render.inr(abs(rep["money"]["revenue_change_inr"])) in html


@pytest.fixture(scope="module")
def rep_models(farm, models):
    return report.build_report(farm, models=models)


@pytest.mark.parametrize("lang", ["kn", "hi", "en"])
def test_model_numbers_and_watch_list_are_shown(farm, rep_models, lang):
    html = render.body(rep_models, lang)
    text = _text(html)
    cow = farm.ground_truth["at_risk_cows"][0]
    assert cow in html
    assert i18n.t("watch_title", lang) in text
    for cause in rep_models["causes"]:
        assert i18n.t("cause_cost", lang, l=cause["loss_l"]) in text
    assert "{" not in text


def test_no_watch_card_or_cost_line_without_models(rep):
    html = render.body(rep, "en")
    assert "Check these cows today" not in html
    assert "litres less milk every day" not in html


ADVISOR = {
    "question": "q", "language": "en", "model": "m", "error": None,
    "plan": "First step.\n\nSecond <b>step</b>.",
    "specialists": [
        {"agent": "health", "answer": "x", "turns": 2, "hit_turn_limit": False, "error": None,
         "tool_calls": [{"name": "a", "args": {}, "error": False}, {"name": "b", "args": {}, "error": False}]},
        {"agent": "nutrition", "answer": "", "turns": 0, "hit_turn_limit": False, "error": "boom <i>", "tool_calls": []},
    ],
}


@pytest.mark.parametrize("lang", ["kn", "hi", "en"])
def test_advisor_section_shows_the_plan_in_paragraphs_and_names_each_agent(rep, lang):
    html = render.body(rep, lang, advisor=ADVISOR)
    text = _text(html)
    assert i18n.t("advisor_heading", lang) in text
    assert html.count('<p class="plan">') == 2
    assert "First step." in text
    assert i18n.agent_label("health", lang) in text
    assert i18n.t("tool_calls_count", lang, n=2) in text


def test_advisor_text_is_escaped(rep):
    html = render.body(rep, "en", advisor=ADVISOR)
    assert "<b>step</b>" not in html and "&lt;b&gt;step&lt;/b&gt;" in html
    assert "<i>" not in html.replace('<i class="bar', "")


@pytest.mark.parametrize("lang", ["kn", "hi", "en"])
def test_failures_are_plain_language_with_the_technical_error_tucked_into_details(rep, lang):
    html = render.body(rep, lang, advisor=ADVISOR)
    visible = re.sub(r"<details.*?</details>", "", html, flags=re.S)
    assert "boom" not in visible
    assert "boom &lt;i&gt;" in html and i18n.t("advisor_details", lang) in _text(html)
    assert i18n.t("advisor_unavailable", lang) in _text(visible)


def test_a_run_with_no_plan_says_so_plainly_and_hides_the_raw_error(rep):
    failed = {"plan": "", "error": "Error code: 401 - invalid x-api-key", "specialists": []}
    html = render.body(rep, "en", advisor=failed)
    visible = re.sub(r"<details.*?</details>", "", html, flags=re.S)
    assert i18n.t("advisor_error", "en") in _text(visible)
    assert "401" not in visible and "401" in html


def test_saved_answers_say_when_they_were_saved(rep):
    text = _text(render.body(rep, "en", advisor=ADVISOR, saved_on="2026-10-09"))
    assert i18n.t("replay_saved", "en", d="9 Oct") in text


@pytest.mark.parametrize("lang", ["kn", "hi", "en"])
def test_a_note_points_to_the_languages_that_do_have_advice(rep, lang):
    html = render.body(rep, lang, other_languages=["kn", "hi"])
    text = _text(html)
    assert i18n.t("advisor_other_language", lang, l="ಕನ್ನಡ, हिन्दी") in text
    assert "advisors" in html and 'class="plan"' in html
    assert "{" not in text


def test_no_note_when_there_is_no_advice_anywhere(rep):
    assert "advisors" not in render.body(rep, "en", other_languages=[])


def test_no_advisor_section_without_a_result(rep):
    assert "What the advisors say" not in render.body(rep, "en")


def _css_rule(selector: str) -> str:
    css = render.CSS_PATH.read_text(encoding="utf-8")
    return css[css.index(selector + " {"):].split("}")[0]


def _line_height(selector: str) -> float:
    return float(re.search(r"line-height:\s*([\d.]+)", _css_rule(selector)).group(1))


@pytest.mark.parametrize("selector", ['.pp[lang="kn"] .eartag-text', '.pp[lang="kn"] .legend b',
                                      '.pp[lang="kn"] .looked li', '.pp[lang="kn"] .fig span'])
def test_kannada_text_gets_roomy_lines_where_the_latin_layout_is_tight(selector):
    """Kannada glyphs are tall; the ear tags, money lines, chart legend and advisor list must not touch."""
    assert _line_height(selector) >= 1.5


def test_kannada_money_lines_have_a_gap_between_the_amount_and_its_description():
    assert "margin-bottom" in _css_rule('.pp[lang="kn"] .fig .amt')
    assert "margin-top" in _css_rule('.pp[lang="kn"] .unit')


def test_plan_line_breaks_are_kept_so_headings_stay_on_their_own_line():
    css = render.CSS_PATH.read_text(encoding="utf-8")
    plan_rule = css[css.index(".plan {"):].split("}")[0]
    assert "white-space: pre-line" in plan_rule


def _chip(lang, agent):
    return f'<span class="agent">{i18n.agent_label(agent, lang)}</span>'


@pytest.mark.parametrize("lang", ["kn", "hi", "en"])
def test_every_block_of_information_has_its_agent_named_above_it(rep_models, lang):
    html = render.body(rep_models, lang)
    src = rep_models["sources"]
    for cause in rep_models["causes"]:
        assert f'<article class="cause">{_chip(lang, cause["agent"])}' in html
    assert f'<article class="cause watch">{_chip(lang, src["watch"])}' in html
    assert f'<article class="cause fine">{_chip(lang, src["fine"])}' in html
    for action in rep_models["actions"]:
        assert f'<li class="act"><div class="meta">{_chip(lang, action["agent"])}' in html
    assert html.count(f'<p class="fig">{_chip(lang, src["money"])}') == 2
    assert f'{_chip(lang, src["milk"])}<div class="big">' in html
    assert f'<h2>{i18n.t("chart_heading", lang)}</h2>{_chip(lang, src["chart"])}' in html


def test_the_number_of_labels_matches_the_number_of_blocks(rep_models):
    html = render.body(rep_models, "en")
    watch, fine, money, chart, milk = 1, 1, 2, 1, 1
    expected = len(rep_models["causes"]) + len(rep_models["actions"]) + watch + fine + money + chart + milk
    assert html.count('<span class="agent">') == expected


@pytest.mark.parametrize("lang", ["kn", "hi", "en"])
def test_the_advisor_plan_is_signed_by_the_farm_manager_and_each_specialist_answers_under_their_name(rep, lang):
    html = render.body(rep, lang, advisor=ADVISOR)
    assert f'{_chip(lang, "manager")}<p class="plan">' in html
    assert re.search(r'<details class="said"><summary><b>' + re.escape(i18n.agent_label("health", lang)) + "</b>", html)
    assert '<p class="said-text">x</p>' in html


def test_a_specialist_that_failed_gets_no_answer_row_only_its_plain_status(rep):
    html = render.body(rep, "en", advisor=ADVISOR)
    assert "<summary><b>Nutrition Agent</b>" not in html
    assert "<li><b>Nutrition Agent</b> could not answer</li>" in html


def test_specialist_answers_are_escaped(rep):
    nasty = {**ADVISOR, "specialists": [{"agent": "health", "answer": "<script>x</script>", "turns": 1,
                                         "hit_turn_limit": False, "error": None, "tool_calls": []}]}
    html = render.body(rep, "en", advisor=nasty)
    assert "<script>" not in html and "&lt;script&gt;" in html


@pytest.mark.parametrize("lang", ["kn", "hi", "en"])
def test_a_note_explains_what_the_labels_mean(rep, lang):
    assert i18n.t("source_note", lang) in _text(render.body(rep, lang))


def test_advisors_causes_and_steps_share_one_column_between_the_hero_and_the_money_card(rep_models):
    """So a desktop can put them beside the hero and money, while the reading order stays the same."""
    html = render.body(rep_models, "en", advisor=ADVISOR)
    column = html.split('<div class="main-col">')[1].split('<section class="card dark">')[0]
    for name in ("advisors", "causes", "steps"):
        assert f'<section class="card' in column and f" {name}" in column
    assert '<section class="hero">' not in column
    assert html.index('class="hero"') < html.index('class="main-col"') < html.index('card dark')


def test_user_supplied_ids_are_escaped(rep):
    bad = copy.deepcopy(rep)
    bad["fine_cows"] = ["<script>alert(1)</script>"]
    assert "<script>" not in render.body(bad, "en")


def test_page_embeds_the_stylesheet_with_mobile_basics(rep):
    html = render.page(rep, "en")
    assert html.startswith("<style>")
    assert "min-height:44px" in html.replace(" ", "")


@pytest.mark.parametrize("value,expected", [(0, "0"), (4956, "4,956"), (40702, "40,702"), (100000, "1,00,000"),
                                            (1234567, "12,34,567")])
def test_inr_uses_indian_grouping(value, expected):
    assert render.inr(value) == expected


def test_report_without_causes_still_renders(rep):
    quiet = copy.deepcopy(rep)
    quiet["causes"], quiet["fine_cows"], quiet["actions"] = [], [], []
    assert "<main" in render.body(quiet, "en")
