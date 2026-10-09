"""English/Hindi strings, language detection and the LLM language instruction."""
from __future__ import annotations

from datetime import date

DEFAULT_LANGUAGE = "kn"
LANGUAGES = ("kn", "hi", "en")
LANGUAGE_NAMES = {"kn": "ಕನ್ನಡ", "hi": "हिन्दी", "en": "English"}
SCRIPT_RANGES = {"hi": (0x0900, 0x097F), "kn": (0x0C80, 0x0CFF)}  # Devanagari, Kannada
SCRIPT_SHARE_THRESHOLD = 0.5  # share of letters that must be in one script to call the text that language

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
        "question_label": "Your question",
        "question_text": "Why is my milk going down?",
        "milk_unit": "litres of milk a day",
        "delta_text": "{p}% less than last week. That is {l} litres less every day.",
        "causes_heading": "What is going wrong",
        "cause_sick_title": "Sick cows",
        "cause_sick_body": "Infection in the udder (mastitis). They have fever and give less milk.",
        "cause_feed_title": "Weaker feed",
        "cause_feed_body": "From {d}, the protein in the feed fell from {b}% to {a}%.",
        "cause_heat_title": "Hot weather",
        "cause_heat_body": "{n} hot days this week, around {t}°C. Cows eat less and give less milk.",
        "fine_title": "These cows are fine",
        "fine_body": "They are near the end of their milking period, so less milk is normal.",
        "actions_heading": "What to do",
        "when_today": "TODAY",
        "when_week": "THIS WEEK",
        "when_later": "LATER",
        "act_treat_title": "Treat these cows",
        "act_treat_body": "Call the vet. Keep their milk out of the tank until the date shown.",
        "until_date": "until {d}",
        "act_tubes_title": "Buy mastitis medicine",
        "act_tubes_body": "Only {n} days left. You have {q} tubes and use {u} every day.",
        "act_feed_title": "Fix the feed",
        "act_feed_body": "Ask your supplier for the earlier feed with {b}% protein. The cheaper feed saves ₹{s} a day, "
                         "but the milk you lost is worth about ₹{l} a day.",
        "act_heat_title": "Keep them cool",
        "act_heat_body": "Give shade and cool water. Feed early in the morning and in the evening.",
        "act_breed_title": "Check breeding",
        "act_breed_body": "These cows are not pregnant and it is over 120 days since they calved.",
        "money_heading": "Money this week",
        "money_lost": "less milk income than last week",
        "money_discard": "of milk cannot be sold ({l} litres, medicine waiting time)",
        "chart_heading": "Milk each day",
        "chart_last_week": "Last week",
        "chart_this_week": "This week",
        "chart_avg": "{l} litres a day",
        "sample_note": "This is sample farm data, not a real farm.",
        "source_note": "The name above each block is the advisor whose data it comes from.",
        "cause_cost": "About {l} litres less milk every day.",
        "watch_title": "Check these cows today",
        "watch_body": "They look healthy, but the early warning model sees the first signs of udder infection. "
                      "Feel the udder and check the milk for clots.",
        "advisor_heading": "What the advisors say",
        "advisor_error": "The advisors could not answer right now. Check the internet connection and try again.",
        "advisor_unavailable": "could not answer",
        "advisor_details": "Details",
        "replay_saved": "Saved answer from {d}",
        "advisor_other_language": "There is no advice in this language yet. Switch to {l} to read the saved advice.",
        "advisor_off_hint": "The advisors are switched off because no AI service is set up. "
                            "Set GEMINI_API_KEY (a free key) and restart the app.",
        "advisor_setup_mistake": "The advisors are switched off because their setup has a mistake: {e}",
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
        "question_label": "आपका सवाल",
        "question_text": "मेरा दूध कम क्यों हो रहा है?",
        "milk_unit": "लीटर दूध रोज़",
        "delta_text": "पिछले हफ्ते से {p}% कम। यानी रोज़ {l} लीटर कम।",
        "causes_heading": "क्या गड़बड़ है",
        "cause_sick_title": "बीमार गायें",
        "cause_sick_body": "थन में संक्रमण (थनैला)। इन्हें बुखार है और ये कम दूध दे रही हैं।",
        "cause_feed_title": "कमज़ोर चारा",
        "cause_feed_body": "{d} से चारे में प्रोटीन {b}% से घटकर {a}% रह गया।",
        "cause_heat_title": "तेज़ गर्मी",
        "cause_heat_body": "इस हफ्ते {n} दिन बहुत गर्मी रही, लगभग {t}°C। गायें कम खाती हैं और कम दूध देती हैं।",
        "fine_title": "इन गायों में कोई दिक्कत नहीं",
        "fine_body": "ये दूध के आखिरी दौर में हैं, इसलिए कम दूध देना सामान्य है।",
        "actions_heading": "अब क्या करें",
        "when_today": "आज ही",
        "when_week": "इसी हफ्ते",
        "when_later": "बाद में",
        "act_treat_title": "इन गायों का इलाज कराएँ",
        "act_treat_body": "पशु डॉक्टर को बुलाएँ। दिखाई गई तारीख तक इनका दूध टंकी में न मिलाएँ।",
        "until_date": "{d} तक",
        "act_tubes_title": "थनैला की दवा मँगवाएँ",
        "act_tubes_body": "सिर्फ़ {n} दिन की दवा बची है। {q} ट्यूब हैं और रोज़ {u} लगती हैं।",
        "act_feed_title": "चारा ठीक करें",
        "act_feed_body": "सप्लायर से पहले वाला {b}% प्रोटीन का चारा माँगें। सस्ता चारा रोज़ ₹{s} बचाता है, "
                         "पर खोया हुआ दूध लगभग ₹{l} रोज़ का है।",
        "act_heat_title": "गायों को ठंडा रखें",
        "act_heat_body": "छाया और ठंडा पानी दें। सुबह जल्दी और शाम को चारा दें।",
        "act_breed_title": "गर्भाधान की जाँच कराएँ",
        "act_breed_body": "ये गायें गाभिन नहीं हैं और ब्याए हुए 120 दिन से ज़्यादा हो गए हैं।",
        "money_heading": "इस हफ्ते का पैसा",
        "money_lost": "पिछले हफ्ते से कम दूध की कमाई",
        "money_discard": "का दूध बिक नहीं सकता ({l} लीटर, दवा का इंतज़ार)",
        "chart_heading": "रोज़ का दूध",
        "chart_last_week": "पिछला हफ्ता",
        "chart_this_week": "यह हफ्ता",
        "chart_avg": "रोज़ {l} लीटर",
        "sample_note": "यह नमूना डेटा है, किसी असली फार्म का नहीं।",
        "source_note": "हर हिस्से के ऊपर लिखा नाम उस सलाहकार का है जिसके आँकड़ों से यह लिया गया है।",
        "cause_cost": "लगभग {l} लीटर दूध रोज़ कम।",
        "watch_title": "आज इन गायों की जाँच करें",
        "watch_body": "ये ठीक दिखती हैं, पर चेतावनी मॉडल को थन के संक्रमण के पहले संकेत दिख रहे हैं। "
                      "थन छूकर देखें और दूध में थक्के जाँचें।",
        "advisor_heading": "सलाहकार क्या कहते हैं",
        "advisor_error": "सलाहकार अभी जवाब नहीं दे सके। इंटरनेट कनेक्शन जाँचें और दोबारा कोशिश करें।",
        "advisor_unavailable": "जवाब नहीं दे सका",
        "advisor_details": "विवरण",
        "replay_saved": "{d} का सहेजा गया जवाब",
        "advisor_other_language": "इस भाषा में अभी कोई सलाह नहीं है। सहेजी गई सलाह पढ़ने के लिए {l} चुनें।",
        "advisor_off_hint": "सलाहकार बंद हैं क्योंकि कोई एआई सेवा चालू नहीं है। "
                            "GEMINI_API_KEY (मुफ़्त कुंजी) सेट करें और ऐप दोबारा शुरू करें।",
        "advisor_setup_mistake": "सलाहकार बंद हैं क्योंकि उनकी सेटिंग में गलती है: {e}",
    },
    "kn": {
        "app_title": "PashuPlan: ಎಐ ಹೈನುಗಾರಿಕೆ ಸಲಹೆಗಾರ",
        "ask_label": "ನಿಮ್ಮ ಹೈನುಗಾರಿಕೆ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ",
        "example_query": "ಈ ವಾರ ಹಾಲಿನ ಉತ್ಪಾದನೆ ಕಡಿಮೆಯಾಗಿದೆ",
        "run_button": "ಸಲಹೆಗಾರರನ್ನು ಕೇಳಿ",
        "language_label": "ಭಾಷೆ",
        "thinking": "ಸಲಹೆಗಾರರು ನಿಮ್ಮ ಫಾರ್ಮ್ ಮಾಹಿತಿಯನ್ನು ಪರಿಶೀಲಿಸುತ್ತಿದ್ದಾರೆ...",
        "final_answer": "ಶಿಫಾರಸು",
        "tool_calls_count": "{n} ಮಾಹಿತಿ ಪರಿಶೀಲನೆಗಳು",
        "replay_mode": "ರೀಪ್ಲೇ ಮೋಡ್ (ಆಫ್‌ಲೈನ್ ಡೆಮೊ)",
        "agent_health": "ಆರೋಗ್ಯ ಸಲಹೆಗಾರ",
        "agent_nutrition": "ಪೋಷಣೆ ಸಲಹೆಗಾರ",
        "agent_breeding": "ಸಂತಾನೋತ್ಪತ್ತಿ ಸಲಹೆಗಾರ",
        "agent_inventory": "ದಾಸ್ತಾನು ಸಲಹೆಗಾರ",
        "agent_finance": "ಹಣಕಾಸು ಸಲಹೆಗಾರ",
        "agent_manager": "ಫಾರ್ಮ್ ಮ್ಯಾನೇಜರ್",
        "question_label": "ನಿಮ್ಮ ಪ್ರಶ್ನೆ",
        "question_text": "ನನ್ನ ಹಾಲು ಏಕೆ ಕಡಿಮೆಯಾಗುತ್ತಿದೆ?",
        "milk_unit": "ಲೀಟರ್ ಹಾಲು ಪ್ರತಿದಿನ",
        "delta_text": "ಕಳೆದ ವಾರಕ್ಕಿಂತ {p}% ಕಡಿಮೆ. ಅಂದರೆ ಪ್ರತಿದಿನ {l} ಲೀಟರ್ ಕಡಿಮೆ.",
        "causes_heading": "ಏನು ತೊಂದರೆ ಇದೆ",
        "cause_sick_title": "ಅನಾರೋಗ್ಯದ ಹಸುಗಳು",
        "cause_sick_body": "ಕೆಚ್ಚಲಿನಲ್ಲಿ ಸೋಂಕು (ಕೆಚ್ಚಲುಬಾವು). ಜ್ವರ ಇದೆ ಮತ್ತು ಹಾಲು ಕಡಿಮೆ ಕೊಡುತ್ತಿವೆ.",
        "cause_feed_title": "ದುರ್ಬಲ ಮೇವು",
        "cause_feed_body": "{d} ರಿಂದ ಮೇವಿನಲ್ಲಿ ಪ್ರೋಟೀನ್ {b}% ರಿಂದ {a}% ಕ್ಕೆ ಇಳಿದಿದೆ.",
        "cause_heat_title": "ಬಿಸಿಲಿನ ತಾಪ",
        "cause_heat_body": "ಈ ವಾರ {n} ದಿನ ತುಂಬಾ ಬಿಸಿ ಇತ್ತು, ಸುಮಾರು {t}°C. ಹಸುಗಳು ಕಡಿಮೆ ತಿನ್ನುತ್ತವೆ ಮತ್ತು ಕಡಿಮೆ ಹಾಲು ಕೊಡುತ್ತವೆ.",
        "fine_title": "ಈ ಹಸುಗಳು ಸರಿಯಾಗಿವೆ",
        "fine_body": "ಇವು ಹಾಲು ಕೊಡುವ ಅವಧಿಯ ಕೊನೆಯಲ್ಲಿವೆ, ಆದ್ದರಿಂದ ಕಡಿಮೆ ಹಾಲು ಸಹಜ.",
        "actions_heading": "ಏನು ಮಾಡಬೇಕು",
        "when_today": "ಇಂದೇ",
        "when_week": "ಈ ವಾರ",
        "when_later": "ನಂತರ",
        "act_treat_title": "ಈ ಹಸುಗಳಿಗೆ ಚಿಕಿತ್ಸೆ ಕೊಡಿಸಿ",
        "act_treat_body": "ಪಶುವೈದ್ಯರನ್ನು ಕರೆಯಿರಿ. ತೋರಿಸಿದ ದಿನಾಂಕದವರೆಗೆ ಇವುಗಳ ಹಾಲನ್ನು ಟ್ಯಾಂಕ್‌ಗೆ ಸೇರಿಸಬೇಡಿ.",
        "until_date": "{d} ವರೆಗೆ",
        "act_tubes_title": "ಕೆಚ್ಚಲುಬಾವು ಔಷಧಿ ತರಿಸಿ",
        "act_tubes_body": "ಕೇವಲ {n} ದಿನಕ್ಕೆ ಸಾಕಾಗುವಷ್ಟು ಉಳಿದಿದೆ. {q} ಟ್ಯೂಬ್ ಇವೆ, ಪ್ರತಿದಿನ {u} ಬೇಕಾಗುತ್ತದೆ.",
        "act_feed_title": "ಮೇವನ್ನು ಸರಿಪಡಿಸಿ",
        "act_feed_body": "ಪೂರೈಕೆದಾರರಿಂದ ಮೊದಲಿನ {b}% ಪ್ರೋಟೀನ್ ಇರುವ ಮೇವನ್ನು ಕೇಳಿ. ಅಗ್ಗದ ಮೇವು ದಿನಕ್ಕೆ ₹{s} ಉಳಿಸುತ್ತದೆ, "
                         "ಆದರೆ ಕಳೆದುಕೊಂಡ ಹಾಲಿನ ಮೌಲ್ಯ ದಿನಕ್ಕೆ ಸುಮಾರು ₹{l}.",
        "act_heat_title": "ಹಸುಗಳನ್ನು ತಂಪಾಗಿಡಿ",
        "act_heat_body": "ನೆರಳು ಮತ್ತು ತಣ್ಣನೆಯ ನೀರು ಕೊಡಿ. ಬೆಳಿಗ್ಗೆ ಬೇಗ ಮತ್ತು ಸಂಜೆ ಮೇವು ಕೊಡಿ.",
        "act_breed_title": "ಗರ್ಭಧಾರಣೆ ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ",
        "act_breed_body": "ಈ ಹಸುಗಳು ಗರ್ಭ ಧರಿಸಿಲ್ಲ ಮತ್ತು ಕರು ಹಾಕಿ 120 ದಿನಕ್ಕಿಂತ ಹೆಚ್ಚಾಗಿದೆ.",
        "money_heading": "ಈ ವಾರದ ಹಣ",
        "money_lost": "ಕಳೆದ ವಾರಕ್ಕಿಂತ ಕಡಿಮೆ ಹಾಲಿನ ಆದಾಯ",
        "money_discard": "ಹಾಲು ಮಾರಲು ಆಗುವುದಿಲ್ಲ ({l} ಲೀಟರ್, ಔಷಧಿ ಕಾಯುವ ಅವಧಿ)",
        "chart_heading": "ಪ್ರತಿದಿನದ ಹಾಲು",
        "chart_last_week": "ಕಳೆದ ವಾರ",
        "chart_this_week": "ಈ ವಾರ",
        "chart_avg": "ಪ್ರತಿದಿನ {l} ಲೀಟರ್",
        "sample_note": "ಇದು ಮಾದರಿ ಮಾಹಿತಿ, ನಿಜವಾದ ಫಾರ್ಮ್‌ನದ್ದಲ್ಲ.",
        "source_note": "ಪ್ರತಿ ಭಾಗದ ಮೇಲಿನ ಹೆಸರು, ಆ ಮಾಹಿತಿ ಯಾರ ಅಂಕಿಅಂಶಗಳಿಂದ ಬಂದಿದೆಯೋ ಆ ಸಲಹೆಗಾರರದ್ದು.",
        "cause_cost": "ಪ್ರತಿದಿನ ಸುಮಾರು {l} ಲೀಟರ್ ಹಾಲು ಕಡಿಮೆ.",
        "watch_title": "ಇಂದು ಈ ಹಸುಗಳನ್ನು ಪರೀಕ್ಷಿಸಿ",
        "watch_body": "ಇವು ಆರೋಗ್ಯವಾಗಿ ಕಾಣುತ್ತವೆ, ಆದರೆ ಮುನ್ನೆಚ್ಚರಿಕೆ ಮಾದರಿಗೆ ಕೆಚ್ಚಲು ಸೋಂಕಿನ ಆರಂಭಿಕ ಲಕ್ಷಣಗಳು ಕಾಣುತ್ತಿವೆ. "
                      "ಕೆಚ್ಚಲನ್ನು ಮುಟ್ಟಿ ನೋಡಿ ಮತ್ತು ಹಾಲಿನಲ್ಲಿ ಗಂಟುಗಳಿವೆಯೇ ನೋಡಿ.",
        "advisor_heading": "ಸಲಹೆಗಾರರು ಏನು ಹೇಳುತ್ತಾರೆ",
        "advisor_error": "ಸಲಹೆಗಾರರು ಈಗ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಇಂಟರ್ನೆಟ್ ಸಂಪರ್ಕ ಪರಿಶೀಲಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "advisor_unavailable": "ಉತ್ತರಿಸಲಾಗಲಿಲ್ಲ",
        "advisor_details": "ವಿವರಗಳು",
        "replay_saved": "{d} ರ ಉಳಿಸಿದ ಉತ್ತರ",
        "advisor_other_language": "ಈ ಭಾಷೆಯಲ್ಲಿ ಇನ್ನೂ ಸಲಹೆ ಇಲ್ಲ. ಉಳಿಸಿದ ಸಲಹೆ ಓದಲು {l} ಆಯ್ಕೆಮಾಡಿ.",
        "advisor_off_hint": "ಸಲಹೆಗಾರರು ಆಫ್ ಆಗಿದ್ದಾರೆ, ಏಕೆಂದರೆ ಯಾವುದೇ ಎಐ ಸೇವೆ ಸಿದ್ಧವಾಗಿಲ್ಲ. "
                            "GEMINI_API_KEY (ಉಚಿತ ಕೀ) ಹೊಂದಿಸಿ ಮತ್ತು ಆ್ಯಪ್ ಅನ್ನು ಮತ್ತೆ ಆರಂಭಿಸಿ.",
        "advisor_setup_mistake": "ಸಲಹೆಗಾರರು ಆಫ್ ಆಗಿದ್ದಾರೆ, ಏಕೆಂದರೆ ಅವರ ಸೆಟ್ಟಿಂಗ್‌ನಲ್ಲಿ ತಪ್ಪಿದೆ: {e}",
    },
}


def detect_language(text: str) -> str:
    """'kn' or 'hi' when most letters are in that script, otherwise 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    for lang, (lo, hi) in SCRIPT_RANGES.items():
        if sum(lo <= ord(ch) <= hi for ch in letters) / len(letters) >= SCRIPT_SHARE_THRESHOLD:
            return lang
    return "en"


def t(key: str, lang: str = DEFAULT_LANGUAGE, **fmt: object) -> str:
    if key not in STRINGS["en"]:
        raise KeyError(key)
    text = STRINGS.get(lang, STRINGS[DEFAULT_LANGUAGE])[key]
    return text.format(**fmt) if fmt else text


MONTHS = {
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "hi": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    "kn": ["ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್", "ಜುಲೈ", "ಆಗಸ್ಟ್", "ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್", "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್"],
}


def format_date(value: date | str, lang: str = "en") -> str:
    """'9 Oct' or '9 अक्टूबर': day in Latin digits, month name in the chosen language."""
    d = date.fromisoformat(value) if isinstance(value, str) else value
    return f"{d.day} {MONTHS.get(lang, MONTHS[DEFAULT_LANGUAGE])[d.month - 1]}"


def agent_label(agent: str, lang: str = "en") -> str:
    return t(f"agent_{agent}", lang)


DIGIT_RULE = ("Write every number with the digits 0 to 9 (Latin digits), never with Kannada or Devanagari digits, "
              "also in numbered lists. Write rupee amounts with the ₹ sign (for example ₹31,000) and keep "
              "cow tags such as C16 exactly as written. Do not translate tool names.")
LANGUAGE_RULES = {
    "kn": ("Respond in Kannada written in Kannada script, in simple words a dairy farmer would use. Use these words: "
           "ಕೆಚ್ಚಲು for udder, ಕೆಚ್ಚಲುಬಾವು for mastitis, ಹಸು for cow, ಮೇವು for feed, ಹಾಲು for milk. "),
    "hi": ("Respond in Hindi written in Devanagari script, in simple words a dairy farmer would use. Use these words: "
           "थन for udder, थनैला for mastitis, गाय for cow, चारा for feed, दूध for milk. "),
    "en": "Respond in English, in simple words a dairy farmer would use. ",
}
_INDIAN_DIGITS = {**{0x0CE6 + i: str(i) for i in range(10)}, **{0x0966 + i: str(i) for i in range(10)}}


def latin_digits(text: str) -> str:
    """Kannada and Devanagari digits to 0-9, so numbers are always written the same way."""
    return text.translate(_INDIAN_DIGITS)


def language_instruction(lang: str) -> str:
    return LANGUAGE_RULES.get(lang, LANGUAGE_RULES["en"]) + DIGIT_RULE
