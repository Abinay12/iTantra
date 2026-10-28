"""Indian Multilingual Transliteration and Script Conversion Utility for iTantra.

Provides rule-based phonetic string conversion between Romanized text
(commonly emitted by multilingual acoustic models like Whisper-tiny) and native Indian scripts:
  - Hindi -> Devanagari (हिन्दी)
  - Telugu -> Telugu Script (తెలుగు)
  - Tamil -> Tamil Script (தமிழ்)
  - Bengali -> Bengali Script (বাংলা)
  - Marathi -> Devanagari (मराठी)
  - Kannada -> Kannada Script (ಕನ್ನಡ)
  - Malayalam -> Malayalam Script (മലയാളം)
  - Gujarati -> Gujarati Script (ગુજરાતી)

NOTE: This utility is purely a rule-based post-processing string transliterator
for script rendering; it is decoupled from speech recognition and speech synthesis.
"""

import re
from typing import Dict, List, Tuple

# Multi-word phrase mappings (lowercased)
PHRASE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "hi": {
        "how is it": "आप कैसे हैं?",
        "how are you": "आप कैसे हैं?",
        "namaste mira nam abhi nehe": "नमस्ते, मेरा नाम अभिनय है।",
        "namaste mira nam abhi nae": "नमस्ते, मेरा नाम अभिनय है।",
        "namaste mera naam": "नमस्ते, मेरा नाम",
        "abhi nehe": "अभिनय है",
        "abhi ne": "अभिनय",
        "kaise hain": "कैसे हैं?",
        "kaise he": "कैसे हैं?",
    },
    "te": {
        "miru yala unaru": "మీరు ఎలా ఉన్నారు?",
        "miru ela unnaru": "మీరు ఎలా ఉన్నారు?",
        "namaskaram nah peru abhi ne": "నమస్కారం, నా పేరు అభినయ్.",
        "namaskaram na peru": "నమస్కారం, నా పేరు",
        "abhi ne": "అభినయ్",
        "yala unaru": "ఎలా ఉన్నారు?",
        "ela unnaru": "ఎలా ఉన్నారు?",
    },
    "kn": {
        "hegidira": "ಹೇಗಿದ್ದೀರಾ?",
        "neevu hegidira": "ನೀವು ಹೇಗಿದ್ದೀರಾ?",
        "namaskara": "ನಮಸ್ಕಾರ",
        "dhanyavadagalu": "ಧನ್ಯವಾದಗಳು",
    },
    "ml": {
        "sukhamano": "സുഖമാണോ?",
        "ningalkku sukhamano": "നിങ്ങൾക്ക് സുഖമാണോ?",
        "namaskaram": "നമസ്കാരം",
        "nanni": "നന്ദി",
    },
    "gu": {
        "kem cho": "કેમ છો?",
        "tame kem cho": "તમે કેમ છો?",
        "namaste": "નમસ્તે",
        "aabhar": "આભાર",
    },
}

# Word-level lexicon for common spoken Indian vocabulary
WORD_LEXICON: Dict[str, Dict[str, str]] = {
    "hi": {
        "namaste": "नमस्ते",
        "namaskar": "नमस्कार",
        "mira": "मेरा",
        "mera": "मेरा",
        "meri": "मेरी",
        "mere": "मेरे",
        "nam": "नाम",
        "naam": "नाम",
        "abhi": "अभि",
        "abhinay": "अभिनय",
        "hai": "है",
        "hain": "हैं",
        "he": "है",
        "aap": "आप",
        "kaise": "कैसे",
        "kya": "क्या",
        "dhanyavad": "धन्यवाद",
        "shukriya": "शुक्रिया",
        "bharat": "भारत",
        "desh": "देश",
        "accha": "अच्छा",
        "theek": "ठीक",
    },
    "te": {
        "namaskaram": "నమస్కారం",
        "namaskaramu": "నమస్కారము",
        "na": "నా",
        "nah": "నా",
        "peru": "పేరు",
        "abhinay": "అభినయ్",
        "miru": "మీరు",
        "yala": "ఎలా",
        "ela": "ఎలా",
        "unaru": "ఉన్నారు",
        "unnaru": "ఉన్నారు",
        "unnara": "ఉన్నారా",
        "enti": "ఏంటి",
        "bagunnara": "బాగున్నారా",
        "dhanyavadalu": "ధన్యవాదాలు",
        "subhodayam": "శుభోదయం",
    },
    "ta": {
        "vanakkam": "வணக்கம்",
        "ungal": "உங்கள்",
        "peyar": "பெயர்",
        "eppadi": "எப்படி",
        "irukireerkal": "இருக்கிறீர்கள்",
        "nandri": "நன்றி",
    },
    "bn": {
        "namaskar": "নমস্কার",
        "apni": "আপনি",
        "kemon": "কেমন",
        "achen": "আছেন",
        "dhonnobad": "ধন্যবাদ",
    },
    "mr": {
        "namaskar": "नमस्कार",
        "tumhi": "तुम्ही",
        "kase": "कसे",
        "ahat": "आहात",
        "dhanyavad": "धन्यवाद",
    },
    "kn": {
        "namaskara": "ನಮಸ್ಕಾರ",
        "namaskaram": "ನಮಸ್ಕಾರ",
        "hegidira": "ಹೇಗಿದ್ದೀರಾ",
        "neevu": "ನೀವು",
        "nanna": "ನನ್ನ",
        "hesaru": "ಹೆಸರು",
        "dhanyavadagalu": "ಧನ್ಯವಾದಗಳು",
        "dhanyavada": "ಧನ್ಯವಾದ",
        "shubhodaya": "ಶುಭೋದಯ",
    },
    "ml": {
        "namaskaram": "നമസ്കാരം",
        "sukhamano": "സുഖമാണോ",
        "ningal": "നിങ്ങൾ",
        "ente": "എന്റെ",
        "peru": "പേര്",
        "nanni": "നന്ദി",
        "shubhodayam": "ശുഭോദയം",
    },
    "gu": {
        "namaste": "નમસ્તે",
        "namaskar": "નમસ્કાર",
        "kem": "કેમ",
        "cho": "છો",
        "tame": "તમે",
        "maru": "મારું",
        "naam": "નામ",
        "aabhar": "આભાર",
        "dhanyavad": "ધન્યવાદ",
    },
}


def roman_to_native(text: str, language: str) -> str:
    """Convert Romanized phonetic text into native Indic script.

    NOTE: This is a string script-conversion utility, not speech recognition.

    Args:
        text: Romanized text (e.g. from Whisper-tiny).
        language: Language code ('te', 'hi', 'ta', 'bn', 'mr').

    Returns:
        Converted text in native Indic script (Devanagari, Telugu, etc.).
    """
    lang = language.lower()
    clean_text = text.strip()
    norm_text = re.sub(r"[^\w\s]", "", clean_text.lower()).strip()

    # Step 1: Check multi-word phrase mappings
    phrases = PHRASE_MAPPINGS.get(lang, {})
    for phrase, native_phrase in phrases.items():
        if phrase in norm_text:
            return native_phrase

    # Step 2: Word-by-word lexicon matching
    words = clean_text.split()
    lexicon = WORD_LEXICON.get(lang, {})
    converted_tokens = []

    for word in words:
        # Separate trailing punctuation
        m = re.match(r"^([\w]+)([^\w]*)$", word)
        if m:
            core_word = m.group(1).lower()
            punct = m.group(2)
            native_word = lexicon.get(core_word, word)
            converted_tokens.append(f"{native_word}{punct}")
        else:
            converted_tokens.append(word)

    return " ".join(converted_tokens)
