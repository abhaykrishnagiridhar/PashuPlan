import pytest

from pashuplan import i18n


@pytest.mark.parametrize("text,lang", [
    ("मेरे दूध उत्पादन में कमी आई है", "hi"),
    ("इस हफ्ते दूध कम क्यों हुआ?", "hi"),
    ("C16 का दूध कम है", "hi"),
    ("Milk production has decreased this week", "en"),
    ("why is C16 milk low", "en"),
    ("milk थोड़ा low है but ok", "en"),
    ("", "en"),
    ("12345 ???", "en"),
])
def test_detect_language(text, lang):
    assert i18n.detect_language(text) == lang


def test_translations_have_identical_keys():
    assert set(i18n.STRINGS["en"]) == set(i18n.STRINGS["hi"])


def test_hindi_strings_are_actually_hindi():
    for key, value in i18n.STRINGS["hi"].items():
        assert i18n.detect_language(value) == "hi" or key in i18n.LATIN_OK_KEYS, key


def test_t_returns_language_specific_text():
    assert i18n.t("app_title", "hi") != i18n.t("app_title", "en")
    assert "PashuPlan" in i18n.t("app_title", "en")


def test_t_unknown_key_and_language():
    with pytest.raises(KeyError):
        i18n.t("nope", "en")
    assert i18n.t("run_button", "fr") == i18n.t("run_button", "en")


def test_t_formats_placeholders():
    assert "3" in i18n.t("tool_calls_count", "en", n=3)
    assert "3" in i18n.t("tool_calls_count", "hi", n=3)


def test_language_instruction_pins_numbers_and_script():
    hi = i18n.language_instruction("hi")
    assert "Hindi" in hi and "Devanagari" in hi
    assert "Latin digits" in hi and "cow tags" in hi
    en = i18n.language_instruction("en")
    assert "English" in en and "Hindi" not in en


def test_agent_label_localised():
    assert i18n.agent_label("health", "en") == "Health Agent"
    assert i18n.detect_language(i18n.agent_label("health", "hi")) == "hi"
    with pytest.raises(KeyError):
        i18n.agent_label("ghost", "en")
