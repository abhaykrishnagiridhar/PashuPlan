"""English/Hindi strings, language detection and the LLM language instruction."""
from __future__ import annotations

DEVANAGARI_RANGE = (0x0900, 0x097F)
HINDI_SHARE_THRESHOLD = 0.5  # share of letters that must be Devanagari to treat the text as Hindi

# Keys whose Hindi value is intentionally Latin (product names)
LATIN_OK_KEYS = frozenset({"app_title"})

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "app_title": "PashuPlan: AI farm advisors",
        "ask_label": "Ask your farm a question",
        "example_query": "Milk production has decreased this week",
        "run_button": "Ask the agents",
        "language_label": "Language",
        "thinking": "The agents are checking your farm data...",
        "final_answer": "Recommendation",
        "tool_calls_count": "{n} data lookups",
        "replay_mode": "Replay mode (offline demo)",
        "agent_health": "Health Agent",
        "agent_nutrition": "Nutrition Agent",
        "agent_breeding": "Breeding Agent",
        "agent_inventory": "Inventory Agent",
        "agent_finance": "Finance Agent",
        "agent_manager": "Farm Manager",
    },
    "hi": {
        "app_title": "PashuPlan: एआई फार्म सलाहकार",
        "ask_label": "अपने फार्म के बारे में सवाल पूछें",
        "example_query": "इस हफ्ते दूध का उत्पादन कम हो गया है",
        "run_button": "एजेंट्स से पूछें",
        "language_label": "भाषा",
        "thinking": "एजेंट आपके फार्म का डेटा जाँच रहे हैं...",
        "final_answer": "सिफारिश",
        "tool_calls_count": "{n} डेटा जाँच",
        "replay_mode": "रीप्ले मोड (ऑफलाइन डेमो)",
        "agent_health": "स्वास्थ्य एजेंट",
        "agent_nutrition": "पोषण एजेंट",
        "agent_breeding": "प्रजनन एजेंट",
        "agent_inventory": "भंडार एजेंट",
        "agent_finance": "वित्त एजेंट",
        "agent_manager": "फार्म मैनेजर",
    },
}


def detect_language(text: str) -> str:
    """'hi' when most letters are Devanagari, otherwise 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    lo, hi = DEVANAGARI_RANGE
    share = sum(lo <= ord(ch) <= hi for ch in letters) / len(letters)
    return "hi" if share >= HINDI_SHARE_THRESHOLD else "en"


def t(key: str, lang: str = "en", **fmt: object) -> str:
    if key not in STRINGS["en"]:
        raise KeyError(key)
    text = STRINGS.get(lang, STRINGS["en"])[key]
    return text.format(**fmt) if fmt else text


def agent_label(agent: str, lang: str = "en") -> str:
    return t(f"agent_{agent}", lang)


def language_instruction(lang: str) -> str:
    if lang == "hi":
        return ("Respond in Hindi written in Devanagari script, in simple words a dairy farmer would use. "
                "Keep numbers, rupee amounts and units in Latin digits (for example 14.3% or Rs 31,000), "
                "and keep cow tags such as C16 exactly as written. Do not translate tool names.")
    return "Respond in English, in simple words a dairy farmer would use."
