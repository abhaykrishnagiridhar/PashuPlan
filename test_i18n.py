import pytest

from pashuplan import i18n


@pytest.mark.parametrize("text,lang", [
    ("मेरे दूध उत्पादन में कमी आई है", "hi"),
    ("इस हफ्ते दूध कम क्यों हुआ?", "hi"),
    ("C16 का दूध कम है", "hi"),
    ("ನನ್ನ ಹಾಲಿನ ಉತ್ಪಾದನೆ ಕಡಿಮೆಯಾಗಿದೆ", "kn"),
    ("ಈ ವಾರ ಹಾಲು ಏಕೆ ಕಡಿಮೆ?", "kn"),
    ("C16 ಹಸುವಿನ ಹಾಲು ಕಡಿಮೆ", "kn"),
    ("milk ಕಡಿಮೆ low ಆಗಿದೆ but ok", "en"),
    ("Milk production has decreased this week", "en"),
    ("why is C16 milk low", "en"),
    ("milk थोड़ा low है but ok", "en"),
    ("", "en"),
    ("12345 ???", "en"),
])
def test_detect_language(text, lang):
    assert i18n.detect_language(text) == lang


def test_kannada_is_the_default_and_listed_first():
    assert i18n.DEFAULT_LANGUAGE == "kn"
    assert i18n.LANGUAGES == ("kn", "hi", "en")
    assert i18n.LANGUAGE_NAMES == {"kn": "ಕನ್ನಡ", "hi": "हिन्दी", "en": "English"}


def test_translations_have_identical_keys():
    assert set(i18n.STRINGS["en"]) == set(i18n.STRINGS["hi"]) == set(i18n.STRINGS["kn"])


def test_kannada_strings_are_actually_kannada():
    for key, value in i18n.STRINGS["kn"].items():
        assert i18n.detect_language(value) == "kn" or key in i18n.LATIN_OK_KEYS, key


def test_kannada_uses_latin_digits_only():
    for value in i18n.STRINGS["kn"].values():
        assert not any("೦" <= ch <= "೯" for ch in value)


def test_unknown_language_falls_back_to_the_default_language():
    assert i18n.t("run_button", "fr") == i18n.t("run_button", i18n.DEFAULT_LANGUAGE)


def test_hindi_strings_are_actually_hindi():
    for key, value in i18n.STRINGS["hi"].items():
        assert i18n.detect_language(value) == "hi" or key in i18n.LATIN_OK_KEYS, key


def test_t_returns_language_specific_text():
    assert i18n.t("app_title", "hi") != i18n.t("app_title", "en")
    assert "PashuPlan" in i18n.t("app_title", "en")


def test_t_unknown_key_raises():
    with pytest.raises(KeyError):
        i18n.t("nope", "en")


def test_t_formats_placeholders():
    assert "3" in i18n.t("tool_calls_count", "en", n=3)
    assert "3" in i18n.t("tool_calls_count", "hi", n=3)


def test_language_instruction_pins_numbers_and_script():
    hi = i18n.language_instruction("hi")
    assert "Hindi" in hi and "Devanagari" in hi
    assert "Latin digits" in hi and "cow tags" in hi
    en = i18n.language_instruction("en")
    assert "English" in en and "Hindi" not in en


def test_format_date_in_both_languages():
    from datetime import date
    assert i18n.format_date(date(2026, 10, 9), "en") == "9 Oct"
    assert i18n.format_date("2026-10-09", "hi") == "9 अक्टूबर"


def test_format_date_covers_every_month_in_every_language():
    from datetime import date
    for lang in i18n.LANGUAGES:
        for m in range(1, 13):
            assert i18n.format_date(date(2026, m, 1), lang).startswith("1 ")


def test_format_date_in_kannada():
    assert i18n.format_date("2026-10-09", "kn") == "9 ಅಕ್ಟೋಬರ್"


def test_kannada_instruction_pins_script_and_numbers():
    kn = i18n.language_instruction("kn")
    assert "Kannada" in kn and "Kannada script" in kn
    assert "Latin digits" in kn and "cow tags" in kn
    assert "Hindi" not in kn


def test_the_language_instruction_uses_the_rupee_sign_and_never_rs():
    for lang in i18n.LANGUAGES:
        text = i18n.language_instruction(lang)
        assert "₹" in text and "Rs " not in text


def test_the_kannada_and_hindi_instructions_teach_the_right_farm_words():
    kn, hi = i18n.language_instruction("kn"), i18n.language_instruction("hi")
    assert "ಕೆಚ್ಚಲು" in kn and "ಕೆಚ್ಚಲುಬಾವು" in kn  # udder, mastitis (not "ಮಡಿಲು", which means lap)
    assert "थन" in hi and "थनैला" in hi


def test_the_instruction_forbids_indian_script_digits_even_in_lists():
    for lang in i18n.LANGUAGES:
        text = i18n.language_instruction(lang)
        assert "0 to 9" in text and "numbered lists" in text


def test_latin_digits_converts_kannada_and_devanagari_digits_only():
    assert i18n.latin_digits("೧. ಮೇವು ೪೭.೩ ಲೀಟರ್") == "1. ಮೇವು 47.3 ಲೀಟರ್"
    assert i18n.latin_digits("१२३ दिन") == "123 दिन"
    assert i18n.latin_digits("C16 has 14.3% ₹40,702") == "C16 has 14.3% ₹40,702"


def test_agent_label_localised():
    assert i18n.agent_label("health", "en") == "Health Agent"
    assert i18n.detect_language(i18n.agent_label("health", "hi")) == "hi"
    assert i18n.detect_language(i18n.agent_label("health", "kn")) == "kn"
    with pytest.raises(KeyError):
        i18n.agent_label("ghost", "en")
