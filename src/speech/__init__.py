"""Speech processing package for iTantra.

Contains Indian Multilingual Text-to-Speech (TTS) and Speech-to-Text (STT)
engines with planned support for Telugu, Hindi, Tamil, Bengali, and Marathi.
"""

from src.speech.tts_engine import (
    TTSEngine,
    TTSOutput,
    INDIAN_LANGUAGES,
    LanguageInfo,
)
from src.speech.stt_engine import (
    STTEngine,
    STTResult,
)
from src.speech.transliterate import roman_to_native

__all__ = [
    "TTSEngine",
    "TTSOutput",
    "STTEngine",
    "STTResult",
    "INDIAN_LANGUAGES",
    "LanguageInfo",
    "roman_to_native",
]
