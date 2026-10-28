"""Multilingual Translation Engine for iTantra Indian Communication System.

Provides bidirectional translation capabilities between Indian languages:
  - Telugu (te)
  - Hindi (hi)
  - Tamil (ta)
  - Bengali (bn)
  - Marathi (mr)

Maintains transparent distinction between:
  1. Lexicon-based conversational mapping (offline, reproducible).
  2. Public Translation API (when network reachable).
  3. Unsupported out-of-vocabulary text (honestly reported when no offline NMT model is installed).
"""

from dataclasses import dataclass
import json
import re
from typing import Dict, Optional, Tuple
import urllib.parse
import urllib.request


@dataclass
class TranslationResult:
    """Container for translation results."""
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    backend: str
    success: bool
    details: str


# Bidirectional conversational phrase alignment for core Indian languages
CONVERSATIONAL_PARALLEL_CORPUS = [
    {
        "hi": "नमस्ते, मेरा नाम अभिनय है।",
        "te": "నమస్కారం, నా పేరు అభినయ్.",
        "ta": "வணக்கம், என் பெயர் அபினய்.",
        "bn": "নমস্কার, আমার নাম অভিনব।",
        "mr": "नमस्कार, माझं नाव अभिनय आहे.",
    },
    {
        "hi": "आप कैसे हैं?",
        "te": "మీరు ఎలా ఉన్నారు?",
        "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "bn": "আপনি কেমন আছেন?",
        "mr": "तुम्ही कसे आहात?",
    },
    {
        "hi": "नमस्ते, आप कैसे हैं?",
        "te": "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "ta": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "bn": "নমস্কার, আপনি কেমন আছেন?",
        "mr": "नमस्कार, तुम्ही कसे आहात?",
    },
    {
        "hi": "धन्यवाद।",
        "te": "ధన్యవాదాలు.",
        "ta": "நன்றி.",
        "bn": "ধন্যবাদ।",
        "mr": "धन्यवाद.",
    },
    {
        "hi": "शुभ प्रभात।",
        "te": "శుభోదయం.",
        "ta": "காலை வணக்கம்.",
        "bn": "সুপ্রভাত।",
        "mr": "शुभ सकाळ.",
    },
]

# Word-level cross-lingual vocabulary
CROSS_LINGUAL_WORDS = {
    ("hi", "te"): {
        "नमस्ते": "నమస్కారం",
        "नमस्कार": "నమస్కారం",
        "मेरा": "నా",
        "नाम": "పేరు",
        "अभिनय": "అభినయ్",
        "धन्यवाद": "ధన్యవాదాలు",
        "शुभ": "శుభ",
    },
    ("te", "hi"): {
        "నమస్కారం": "नमस्ते",
        "నా": "मेरा",
        "పేరు": "नाम",
        "అభినయ్": "अभिनय",
        "ధన్యవాదాలు": "धन्यवाद",
        "ఎలా": "कैसे",
        "ఉన్నారు": "हैं",
    },
}


class TranslationEngine:
    """Translates text between supported Indian languages."""

    def __init__(self, backend: str = "auto", enable_network_api: bool = False):
        """Initialize translation engine.

        Args:
            backend: Translation backend ('lexicon', 'api', or 'auto').
            enable_network_api: If True, allows external translation API calls.
        """
        self.backend = backend.lower()
        self.enable_network_api = enable_network_api

    def _normalize_text(self, text: str) -> str:
        """Strip punctuation and whitespace for normalized matching."""
        return re.sub(r"[^\w\s]", "", text).strip().lower()

    def _translate_lexicon(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> Optional[str]:
        """Attempt translation via conversational parallel corpus and lexicon."""
        norm_input = self._normalize_text(text)

        # 1. Exact or normalized phrase match
        for item in CONVERSATIONAL_PARALLEL_CORPUS:
            if src_lang in item and tgt_lang in item:
                norm_src = self._normalize_text(item[src_lang])
                if norm_input == norm_src or norm_src in norm_input or norm_input in norm_src:
                    return item[tgt_lang]

        # 2. Word-by-word cross-lingual dictionary
        pair = (src_lang, tgt_lang)
        if pair in CROSS_LINGUAL_WORDS:
            word_map = CROSS_LINGUAL_WORDS[pair]
            tokens = text.split()
            translated_tokens = [word_map.get(tok.strip(".,?!"), tok) for tok in tokens]
            if any(t != orig for t, orig in zip(translated_tokens, tokens)):
                return " ".join(translated_tokens)

        return None

    def _translate_api(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> Tuple[Optional[str], str]:
        """Query free public translation API if network access is enabled."""
        if not self.enable_network_api:
            return None, "Network API disabled by configuration"

        try:
            url = (
                "https://api.mymemory.translated.net/get?q="
                + urllib.parse.quote(text)
                + f"&langpair={src_lang}|{tgt_lang}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "iTantra-Translation/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                translated = data.get("responseData", {}).get("translatedText")
                if translated and translated.strip():
                    return translated.strip(), "Public Translation API"
        except Exception as e:
            return None, f"Network API error: {e}"

        return None, "API returned empty response"

    def translate(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> TranslationResult:
        """Translate text from src_lang to tgt_lang.

        Args:
            text: Input text.
            src_lang: Source Indian language code ('hi', 'te', 'ta', 'bn', 'mr').
            tgt_lang: Target Indian language code.

        Returns:
            TranslationResult with translated_text and success status.
        """
        src = src_lang.lower()
        tgt = tgt_lang.lower()

        if src == tgt:
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_lang=src,
                target_lang=tgt,
                backend="identity",
                success=True,
                details="Source and target languages are identical.",
            )

        # Step 1: Check lexicon / conversational parallel corpus
        lexicon_res = self._translate_lexicon(text, src, tgt)
        if lexicon_res:
            return TranslationResult(
                source_text=text,
                translated_text=lexicon_res,
                source_lang=src,
                target_lang=tgt,
                backend="conversational_lexicon",
                success=True,
                details="Translated via offline conversational parallel corpus.",
            )

        # Step 2: If API enabled, attempt network API
        if self.enable_network_api:
            api_res, api_detail = self._translate_api(text, src, tgt)
            if api_res:
                return TranslationResult(
                    source_text=text,
                    translated_text=api_res,
                    source_lang=src,
                    target_lang=tgt,
                    backend="network_api",
                    success=True,
                    details=api_detail,
                )

        # Step 3: Out of vocabulary and no neural NMT model installed
        return TranslationResult(
            source_text=text,
            translated_text=text,  # Fallback to source text
            source_lang=src,
            target_lang=tgt,
            backend="none",
            success=False,
            details=(
                f"Translation from {src} to {tgt} failed: phrase is out of offline "
                "conversational lexicon and no offline neural translation model "
                "(e.g., IndicTrans2 / NLLB) is installed in this environment."
            ),
        )
