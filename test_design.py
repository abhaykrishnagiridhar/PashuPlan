"""The look must follow DESIGN.md, stay flat, and keep motion cheap enough for a phone."""
import re
from pathlib import Path

import pytest

from pashuplan import render, report

ROOT = Path(__file__).resolve().parent.parent
CSS = render.CSS_PATH.read_text(encoding="utf-8")
DESIGN = (ROOT / "DESIGN.md").read_text(encoding="utf-8")


def clean(css: str) -> str:
    """The stylesheet without comments and @import lines, so every prelude is just a selector or an at-rule."""
    return re.sub(r'@import url\("[^"]*"\);', "", re.sub(r"/\*.*?\*/", "", css, flags=re.S))


def blocks(css: str) -> list[tuple[str, str]]:
    """Top-level (prelude, body) pairs. At-rules keep their nested body as text."""
    css = clean(css)
    out, depth, start, prelude_start, prelude = [], 0, 0, 0, ""
    for i, ch in enumerate(css):
        if ch == "{":
            if depth == 0:
                start, prelude = i + 1, css[prelude_start:i].strip()
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append((prelude, css[start:i]))
                prelude_start = i + 1
    return out


def style_rules() -> list[tuple[str, str]]:
    return [(p, b) for p, b in blocks(CSS) if not p.startswith("@")]


def strip_blocks(css: str, starts: tuple[str, ...]) -> str:
    keep = css = clean(css)
    for prelude, body in blocks(css):
        if prelude.startswith(starts):
            keep = keep.replace(prelude + " {" + body + "}", "").replace(prelude + "{" + body + "}", "")
    return keep


def rule_in(css_text: str, selector: str) -> str:
    """The declarations of the first rule inside a block of CSS whose selector is exactly `selector`."""
    return next(b for p, b in blocks(css_text) if p == selector)


def rule_for(selector: str) -> str:
    return next(b for p, b in style_rules() if p == selector)


def design_colors() -> dict[str, str]:
    first_block = DESIGN.split("```css")[1].split("```")[0]
    return dict(re.findall(r"(--color-[a-z-]+):\s*(#[0-9a-fA-F]{6});", first_block))


def test_every_colour_the_stylesheet_defines_matches_design_md():
    ours = dict(re.findall(r"(--color-[a-z-]+):\s*(#[0-9a-fA-F]{6})", rule_for(":root")))
    assert ours, "the stylesheet should define its palette as --color-* tokens"
    truth = design_colors()
    for name, value in ours.items():
        assert truth.get(name, "").lower() == value.lower(), f"{name} should be {truth.get(name)}"
    for needed in ("forest-ink", "lime-voltage", "linen-mist", "fog", "paper", "charcoal", "alarm-red"):
        assert f"--color-{needed}" in ours


def test_the_look_is_flat_with_no_gradients_shadows_or_blur():
    for forbidden in ("gradient(", "box-shadow", "backdrop-filter", "blur(", "text-shadow", "drop-shadow"):
        assert forbidden not in CSS, forbidden


@pytest.mark.parametrize("selector", [".tag", ".when", ".stButton button", '[data-testid="stRadioGroup"]',
                                      '[data-testid="stRadioOption"]', ".pill"])
def test_buttons_tags_and_the_language_switch_are_pills(selector):
    assert "border-radius: 9999px" in rule_for(selector)


def test_a_disabled_button_looks_off_not_lime_and_shows_a_not_allowed_cursor():
    body = rule_for(".stButton button:disabled")
    assert "cursor: not-allowed" in body
    assert "lime" not in body  # lime means "active" in this design
    assert "var(--color-pebble)" in body


def test_a_button_with_hover_help_is_never_hidden_by_streamlits_tooltip_wrapper_on_a_phone():
    """Streamlit sets the wrapper around a button that has `help=` to display:none on narrow screens."""
    body = rule_for(".stButton :has(> .stTooltipIcon)")
    assert "display: block !important" in body


def test_the_agent_label_is_a_small_pill_that_stays_readable_on_dark_and_light_cards():
    assert "border-radius: 9999px" in rule_for(".agent")
    assert "lime" in rule_for(".dark .agent")  # lime text only on the dark card
    assert float(re.search(r"font-size:\s*(\d+)px", rule_for(".agent")).group(1)) >= 13


def test_sections_are_large_rounded_cards_as_the_design_says():
    assert "border-radius: 28px" in rule_for(".card")


def test_lime_is_only_text_on_dark_surfaces_never_on_light_ones():
    allowed = (".dark", ".tag", ".when.today", ".pill.dark")
    for selector, body in style_rules():
        if re.search(r"(?<![-\w])color:\s*var\(--color-lime-voltage\)", body):
            assert any(a in selector for a in allowed), f"lime text on a light surface: {selector}"


def test_negative_letter_spacing_is_only_on_latin_text_because_it_breaks_indian_scripts():
    for selector, body in style_rules():
        if re.search(r"letter-spacing:\s*-", body):
            assert any(k in selector for k in ('[lang="en"]', ".num", ".mark", ".amt")), selector


def test_heading_fonts_are_loaded_efficiently():
    imports = re.findall(r'@import url\("([^"]+)"\)', CSS)
    assert len(imports) == 2 and all("display=swap" in u for u in imports) and sum(map(len, imports)) < 400
    inter = next(u for u in imports if "family=Inter" in u)
    assert "wght@900" in inter and "text=" in inter and "family=Mukta" not in inter  # text= subsets every family
    indic = next(u for u in imports if "Noto+Sans+Kannada" in u)
    assert "Mukta" in indic and "text=" not in indic and "Inter" not in indic


def test_on_a_desktop_the_page_uses_two_columns_and_the_wide_width_instead_of_a_phone_strip():
    desktop = next(b for p, b in blocks(CSS) if p.startswith("@media") and "min-width: 960px" in p)
    assert "grid-template-columns" in desktop and ".main-col" in desktop
    assert int(re.search(r"max-width:\s*(\d+)px", desktop).group(1)) >= 1100  # the Streamlit block container
    assert "max-width: 420px" in desktop  # the language switch and button stay thumb sized, not 1100px bars


def test_a_wide_desktop_uses_three_columns_so_the_left_side_is_not_left_empty():
    wide = next(b for p, b in blocks(CSS) if p.startswith("@media") and "min-width: 1200px" in p)
    assert "repeat(3" in wide and ".main-col" in wide and "repeat(2" in wide  # causes and steps side by side
    assert int(re.search(r"max-width:\s*(\d+)px", wide).group(1)) >= 1280


def test_the_ask_button_keeps_its_full_width_inside_its_streamlit_wrappers():
    assert "width: 100%" in rule_for('[data-testid="stElementContainer"]:has(.stButton), .stButton')
    # the hover-help wrapper right-aligns its child by default, which pushes the button to the far edge
    assert "justify-content: flex-start" in rule_for(".stButton .stTooltipHoverTarget")


def test_a_tablet_gets_a_wider_single_column():
    tablet = next(b for p, b in blocks(CSS) if p.startswith("@media") and "min-width: 700px" in p)
    assert int(re.search(r"max-width:\s*(\d+)px", tablet).group(1)) > 640


# Motion ----------------------------------------------------------------------------------------

def keyframes() -> dict[str, str]:
    return {p.split()[1]: b for p, b in blocks(CSS) if p.startswith("@keyframes")}


def test_there_are_only_a_few_small_animations():
    frames = keyframes()
    assert 1 <= len(frames) <= 4


def test_animations_only_touch_transform_and_opacity_so_they_run_on_the_gpu():
    for name, body in keyframes().items():
        properties = set(re.findall(r"([a-z-]+)\s*:", body))
        assert properties <= {"transform", "opacity"}, f"{name} animates {properties}"


def test_every_animation_is_quick():
    durations = [float(m) for m in re.findall(r"animation:[^;]*?\s(\d*\.?\d+)s", CSS)]
    assert durations and max(durations) <= 0.8


def test_scroll_driven_effects_are_guarded_so_old_browsers_just_show_the_content():
    outside = strip_blocks(CSS, ("@supports",))
    assert "animation-timeline" not in outside
    assert "@supports (animation-timeline: view())" in CSS


def test_nothing_is_hidden_unless_an_animation_that_reveals_it_is_running():
    outside = strip_blocks(strip_blocks(CSS, ("@supports",)), ("@keyframes",))
    assert "opacity: 0" not in outside


def test_motion_is_switched_off_for_people_who_ask_for_reduced_motion():
    reduced = next(b for p, b in blocks(CSS) if "prefers-reduced-motion" in p)
    assert "animation: none" in reduced and "transition: none" in reduced


def test_the_page_ships_no_javascript():
    farm_report = {"date": "2026-10-08", "milk": {"per_day": 700, "prior_avg": 800, "this_avg": 700,
                                                  "change_pct": -12.0, "litres_less_per_day": 100},
                   "money": {"revenue_change_inr": -1000, "discarded_litres": 0, "discarded_value_inr": 0},
                   "causes": [], "fine_cows": [], "actions": [], "watch_list": [],
                   "bars": [{"date": f"2026-10-{d:02d}", "litres": 800 - d, "period": "prior" if d < 8 else "this"}
                            for d in range(1, 15)]}
    for lang in ("kn", "hi", "en"):
        assert "<script" not in render.page(farm_report, lang).lower()


# Structure -------------------------------------------------------------------------------------

def test_sections_follow_the_design_rhythm_light_then_pale_green_then_white_then_dark(farm, models):
    rep = report.build_report(farm, models=models)
    html = render.body(rep, "en", advisor={"plan": "A plan.", "specialists": [], "error": None})
    classes = re.findall(r'<section class="([^"]+)"', html)
    order = [next(k for k in ("hero", "advisors", "causes", "steps", "dark") if k in c.split()) for c in classes]
    assert order == ["hero", "advisors", "causes", "steps", "dark"]
    assert "card dark" in html and "card linen" in html and "card paper" in html


def test_bars_carry_their_position_so_css_can_stagger_them():
    rep = {"date": "2026-10-08", "milk": {"per_day": 700, "prior_avg": 800, "this_avg": 700, "change_pct": -12.0,
                                          "litres_less_per_day": 100},
           "money": {"revenue_change_inr": -1, "discarded_litres": 0, "discarded_value_inr": 0}, "causes": [],
           "fine_cows": [], "actions": [], "watch_list": [],
           "bars": [{"date": f"2026-10-{d:02d}", "litres": 800, "period": "prior"} for d in range(1, 15)]}
    html = render.body(rep, "en")
    assert [int(n) for n in re.findall(r'class="bar [a-z]+" style="[^"]*--i:(\d+)', html)] == list(range(14))
