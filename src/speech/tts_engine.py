"""Text-to-Speech (TTS) Engine for iTantra Indian Multilingual System.

Supports target Indian languages:
  - Telugu (te)
  - Hindi (hi)
  - Tamil (ta)
  - Bengali (bn)
  - Marathi (mr)
  - Kannada (kn)
  - Malayalam (ml)

Backends:
  - 'neural': High-fidelity neural voice synthesis (Edge-TTS Indian neural voices).
  - 'mms': Meta Massively Multilingual Speech (MMS-TTS) VITS models (facebook/mms-tts-*).
  - 'lightweight': Fast rule-based acoustic generator for offline CPU unit tests.
  - 'auto': Attempts neural synthesis, with graceful fallback to lightweight if offline.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import sys
import tempfile
import numpy as np

# Suppress harmless symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@dataclass
class LanguageInfo:
    """Metadata describing an Indian language supported by iTantra."""
    code: str
    name: str
    script: str
    locale: str
    unicode_range: tuple
    mms_model_id: str
    neural_voice: str
    tts_available: bool = False
    stt_available: bool = False
    in_scope: bool = True
    tts_verified: bool = False

    @property
    def is_fully_supported(self) -> bool:
        """Returns True only when both STT and TTS are operational and in scope."""
        return self.tts_available and self.stt_available and self.in_scope

    @property
    def status(self) -> str:
        """Status string representing availability level."""
        if not self.in_scope:
            return "out_of_scope"
        if self.is_fully_supported:
            return "fully_supported"
        elif self.tts_available or self.stt_available:
            return "partially_supported"
        return "configured"

    @property
    def verification_status(self) -> str:
        """Detailed status distinguishing configured, installed, and verified models."""
        if not self.in_scope:
            return "out_of_scope"
        if self.tts_verified and self.tts_available and self.stt_available:
            return "verified"
        if self.tts_available:
            return "installed"
        return "configured"


INDIAN_LANGUAGES: Dict[str, LanguageInfo] = {
    "te": LanguageInfo(
        code="te",
        name="Telugu",
        script="Telugu",
        locale="te-IN",
        unicode_range=(0x0C00, 0x0C7F),
        mms_model_id="facebook/mms-tts-tel",
        neural_voice="te-IN-MohanNeural",
        tts_available=True,
        stt_available=True,
        in_scope=True,
        tts_verified=True,
    ),
    "hi": LanguageInfo(
        code="hi",
        name="Hindi",
        script="Devanagari",
        locale="hi-IN",
        unicode_range=(0x0900, 0x097F),
        mms_model_id="facebook/mms-tts-hin",
        neural_voice="hi-IN-MadhurNeural",
        tts_available=True,
        stt_available=True,
        in_scope=True,
        tts_verified=True,
    ),
    "ta": LanguageInfo(
        code="ta",
        name="Tamil",
        script="Tamil",
        locale="ta-IN",
        unicode_range=(0x0B80, 0x0BFF),
        mms_model_id="facebook/mms-tts-tam",
        neural_voice="ta-IN-ValluvarNeural",
        tts_available=True,
        stt_available=True,
        in_scope=True,
        tts_verified=True,
    ),
    "bn": LanguageInfo(
        code="bn",
        name="Bengali",
        script="Bengali",
        locale="bn-IN",
        unicode_range=(0x0980, 0x09FF),
        mms_model_id="facebook/mms-tts-ben",
        neural_voice="bn-IN-BashkarNeural",
        tts_available=True,
        stt_available=True,
        in_scope=True,
        tts_verified=True,
    ),
    "mr": LanguageInfo(
        code="mr",
        name="Marathi",
        script="Devanagari",
        locale="mr-IN",
        unicode_range=(0x0900, 0x097F),
        mms_model_id="facebook/mms-tts-mar",
        neural_voice="mr-IN-AarohiNeural",
        tts_available=True,
        stt_available=True,
        in_scope=True,
        tts_verified=True,
    ),
    "kn": LanguageInfo(
        code="kn",
        name="Kannada",
        script="Kannada",
        locale="kn-IN",
        unicode_range=(0x0C80, 0x0CFF),
        mms_model_id="facebook/mms-tts-kan",
        neural_voice="kn-IN-GaganNeural",
        tts_available=True,
        stt_available=True,
        in_scope=True,
        tts_verified=True,
    ),
    "ml": LanguageInfo(
        code="ml",
        name="Malayalam",
        script="Malayalam",
        locale="ml-IN",
        unicode_range=(0x0D00, 0x0D7F),
        mms_model_id="facebook/mms-tts-mal",
        neural_voice="ml-IN-MidhunNeural",
        tts_available=True,
        stt_available=True,
        in_scope=True,
        tts_verified=True,
    ),
    "gu": LanguageInfo(
        code="gu",
        name="Gujarati",
        script="Gujarati",
        locale="gu-IN",
        unicode_range=(0x0A80, 0x0AFF),
        mms_model_id="facebook/mms-tts-guj",
        neural_voice="gu-IN-NiranjanNeural",
        tts_available=False,
        stt_available=True,
        in_scope=False,
        tts_verified=False,
    ),
}


@dataclass
class TTSOutput:
    """Output container for synthesized speech audio."""
    audio: np.ndarray
    sample_rate: int
    language: str
    text: str
    duration_sec: float


class TTSEngine:
    """Multilingual Indian Text-to-Speech Engine."""

    def __init__(
        self,
        sample_rate: int = 16000,
        default_language: str = "hi",
        backend: str = "neural",
    ):
        """Initialize TTS engine.

        Args:
            sample_rate: Target audio sampling rate in Hz (default: 16000).
            default_language: Default ISO language code.
            backend: Synthesis backend ('neural', 'mms', 'lightweight', or 'auto').
        """
        self.sample_rate = sample_rate
        self.default_language = default_language
        self.backend = backend.lower()
        self.supported_languages = INDIAN_LANGUAGES
        self._mms_models = {}
        self._mms_tokenizers = {}

    def is_language_supported(self, lang_code: str) -> bool:
        """Check if an Indian language code is in the registry."""
        return lang_code.lower() in self.supported_languages

    def is_in_scope(self, lang_code: str) -> bool:
        """Check if language is within the active project scope."""
        lang = lang_code.lower()
        return lang in self.supported_languages and self.supported_languages[lang].in_scope

    def is_tts_supported(self, lang_code: str) -> bool:
        """Check if TTS model is integrated and operational for this language."""
        lang = lang_code.lower()
        return (
            lang in self.supported_languages
            and self.supported_languages[lang].tts_available
            and self.supported_languages[lang].in_scope
        )

    def is_stt_supported(self, lang_code: str) -> bool:
        """Check if STT model is integrated and operational for this language."""
        lang = lang_code.lower()
        return lang in self.supported_languages and self.supported_languages[lang].stt_available

    def is_fully_supported(self, lang_code: str) -> bool:
        """Check if both STT and TTS models are integrated for this language."""
        lang = lang_code.lower()
        return lang in self.supported_languages and self.supported_languages[lang].is_fully_supported

    def is_mms_model_cached(self, language: str) -> bool:
        """Check if Meta MMS-TTS weights for this language are installed locally in Hugging Face cache."""
        lang = language.lower()
        if lang not in self.supported_languages:
            return False
        model_id = self.supported_languages[lang].mms_model_id
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        hub_name = f"models--{model_id.replace('/', '--')}"
        model_cache_path = os.path.join(cache_dir, hub_name)
        if not os.path.exists(model_cache_path):
            return False
        snapshots_dir = os.path.join(model_cache_path, "snapshots")
        if not os.path.exists(snapshots_dir):
            return False
        for snapshot in os.listdir(snapshots_dir):
            snap_path = os.path.join(snapshots_dir, snapshot)
            if os.path.isdir(snap_path):
                files = os.listdir(snap_path)
                if any(f in ("model.safetensors", "pytorch_model.bin") for f in files):
                    return True
        return False

    def get_supported_languages(self, in_scope_only: bool = False) -> List[Dict[str, Any]]:
        """Return a list of metadata for all planned and supported languages."""
        languages = self.supported_languages.values()
        if in_scope_only:
            languages = [info for info in languages if info.in_scope]

        return [
            {
                "code": info.code,
                "name": info.name,
                "script": info.script,
                "locale": info.locale,
                "mms_model": info.mms_model_id,
                "neural_voice": info.neural_voice,
                "tts_available": info.tts_available,
                "stt_available": info.stt_available,
                "is_fully_supported": info.is_fully_supported,
                "in_scope": info.in_scope,
                "tts_verified": info.tts_verified,
                "status": info.status,
                "verification_status": info.verification_status,
                "installed": self.is_mms_model_cached(info.code),
            }
            for info in languages
        ]

    def get_language_status_matrix(self) -> List[Dict[str, Any]]:
        """Return comprehensive status matrix distinguishing configured, installed, and verified models."""
        matrix = []
        for info in self.supported_languages.values():
            is_installed = self.is_mms_model_cached(info.code)
            matrix.append({
                "code": info.code,
                "name": info.name,
                "script": info.script,
                "locale": info.locale,
                "in_scope": info.in_scope,
                "configured": True,
                "installed": is_installed,
                "tts_available": info.tts_available,
                "stt_available": info.stt_available,
                "tts_verified": info.tts_verified,
                "status": info.status,
                "verification_status": info.verification_status,
                "mms_model": info.mms_model_id,
                "neural_voice": info.neural_voice,
            })
        return matrix

    def _validate_language(self, language: str) -> str:
        lang = language.lower()
        if not self.is_language_supported(lang):
            supported = ", ".join([c for c, info in self.supported_languages.items() if info.in_scope])
            raise ValueError(
                f"Unsupported language code '{language}'. Supported Indian languages: {supported}"
            )
        info = self.supported_languages[lang]
        if not info.in_scope:
            raise ValueError(
                f"Language '{language}' ({info.name}) is out of project scope."
            )
        return lang

    def _synthesize_neural(self, text: str, language: str) -> TTSOutput:
        """Generate human-quality Indian speech using neural voices."""
        import asyncio
        import edge_tts
        import soundfile as sf
        from scipy.signal import resample_poly
        import math

        lang_info = self.supported_languages[language]
        voice = lang_info.neural_voice

        # Use temporary file for clean, cross-platform audio decoding
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            async def _synthesize():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(tmp_path)

            asyncio.run(_synthesize())

            # Read decoded audio using soundfile (bundled libsndfile decodes MP3)
            audio_data, source_sr = sf.read(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        # Convert stereo to mono if necessary
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to target sample rate (e.g., 24000 -> 16000)
        if source_sr != self.sample_rate:
            gcd = math.gcd(self.sample_rate, source_sr)
            up = self.sample_rate // gcd
            down = source_sr // gcd
            audio_resampled = resample_poly(audio_data, up, down).astype(np.float32)
        else:
            audio_resampled = audio_data.astype(np.float32)

        # Normalize peak amplitude
        max_val = np.max(np.abs(audio_resampled))
        if max_val > 1e-6:
            audio_resampled = (audio_resampled / max_val) * 0.95

        duration_sec = float(len(audio_resampled)) / self.sample_rate

        return TTSOutput(
            audio=audio_resampled,
            sample_rate=self.sample_rate,
            language=language,
            text=text,
            duration_sec=duration_sec,
        )

    def _synthesize_mms(self, text: str, language: str) -> TTSOutput:
        """Generate speech using Meta MMS-TTS (VitsModel) from Hugging Face."""
        import torch
        from transformers import VitsModel, AutoTokenizer
        from scipy.signal import resample_poly
        import math

        lang_info = self.supported_languages[language]
        model_id = lang_info.mms_model_id

        if not self.is_mms_model_cached(language):
            raise FileNotFoundError(
                f"Offline Meta MMS-TTS model checkpoint for {lang_info.name} ('{model_id}') is not installed locally in Hugging Face cache. "
                f"Model weights (~140MB) must be downloaded and approved before offline neural synthesis."
            )

        if language not in self._mms_models:
            print(f"Loading Meta MMS-TTS model for {lang_info.name} ({model_id})...")
            self._mms_tokenizers[language] = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            self._mms_models[language] = VitsModel.from_pretrained(model_id, local_files_only=True)

        tokenizer = self._mms_tokenizers[language]
        model = self._mms_models[language]

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform

        audio_raw = output.squeeze().cpu().numpy().astype(np.float32)
        source_sr = getattr(model.config, "sampling_rate", 16000)

        if source_sr != self.sample_rate:
            gcd = math.gcd(self.sample_rate, source_sr)
            up = self.sample_rate // gcd
            down = source_sr // gcd
            audio_resampled = resample_poly(audio_raw, up, down).astype(np.float32)
        else:
            audio_resampled = audio_raw

        max_val = np.max(np.abs(audio_resampled))
        if max_val > 1e-6:
            audio_resampled = (audio_resampled / max_val) * 0.95

        duration_sec = float(len(audio_resampled)) / self.sample_rate

        return TTSOutput(
            audio=audio_resampled,
            sample_rate=self.sample_rate,
            language=language,
            text=text,
            duration_sec=duration_sec,
        )

    def _synthesize_lightweight(
        self, text: str, language: str, speed: float = 1.0
    ) -> TTSOutput:
        """Lightweight rule-based acoustic speech generator for offline unit tests.

        Generates voiced harmonic waveforms shaped by vowel formants and word
        rhythms, producing real audio signals suitable for end-to-end testing.
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace.")

        effective_chars = max(len(text.strip()), 5)
        duration_sec = max(0.4, (effective_chars * 0.07) / max(0.1, speed))
        num_samples = int(self.sample_rate * duration_sec)

        t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)

        pitch_map = {
            "te": 150.0,
            "hi": 140.0,
            "ta": 155.0,
            "bn": 145.0,
            "mr": 140.0,
            "kn": 145.0,
            "ml": 150.0,
            "gu": 140.0,
        }
        f0 = pitch_map.get(language, 140.0)

        formants = [500.0, 1500.0, 2500.0]
        waveform = np.zeros(num_samples, dtype=np.float32)

        for harmonic in range(1, 10):
            freq = f0 * harmonic
            amp = 1.0 / (harmonic ** 0.8)
            for f in formants:
                if abs(freq - f) < 300:
                    amp *= 2.0
            phase = (hash(text) % 100) * 0.01 * harmonic
            waveform += amp * np.sin(2.0 * np.pi * freq * t + phase)

        envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t - np.pi / 2))
        fade_len = int(0.05 * self.sample_rate)
        if fade_len > 0 and num_samples > 2 * fade_len:
            attack = np.linspace(0, 1, fade_len, dtype=np.float32)
            decay = np.linspace(1, 0, fade_len, dtype=np.float32)
            envelope[:fade_len] *= attack
            envelope[-fade_len:] *= decay

        waveform *= envelope

        max_val = np.max(np.abs(waveform))
        if max_val > 1e-6:
            waveform = (waveform / max_val) * 0.9

        return TTSOutput(
            audio=waveform,
            sample_rate=self.sample_rate,
            language=language,
            text=text,
            duration_sec=duration_sec,
        )

    def synthesize(
        self, text: str, language: Optional[str] = None, speed: float = 1.0
    ) -> TTSOutput:
        """Synthesize text into speech audio.

        Args:
            text: Text to synthesize in supported Indian languages.
            language: Indian language code ('te', 'hi', 'ta', 'bn', 'mr').
            speed: Speech rate multiplier (1.0 = normal).

        Returns:
            TTSOutput containing audio waveform (16 kHz mono float32) and metadata.
        """
        lang = self._validate_language(language or self.default_language)

        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace.")

        if self.backend == "lightweight":
            return self._synthesize_lightweight(text, lang, speed=speed)

        elif self.backend == "neural":
            try:
                return self._synthesize_neural(text, lang)
            except Exception as e:
                print(f"[Warning] Neural synthesis failed ({e}). Falling back to lightweight acoustic mode.")
                return self._synthesize_lightweight(text, lang, speed=speed)

        elif self.backend == "mms":
            try:
                return self._synthesize_mms(text, lang)
            except Exception as e:
                print(f"[Warning] MMS synthesis failed ({e}). Falling back to neural voice mode.")
                return self._synthesize_neural(text, lang)

        elif self.backend == "auto":
            if self.is_mms_model_cached(lang):
                try:
                    return self._synthesize_mms(text, lang)
                except Exception as e:
                    print(f"[Warning] Offline MMS synthesis failed ({e}). Trying neural voice...")
            try:
                return self._synthesize_neural(text, lang)
            except Exception:
                return self._synthesize_lightweight(text, lang, speed=speed)

        else:
            raise ValueError(
                f"Unknown TTS backend '{self.backend}'. Choose 'neural', 'mms', 'auto', or 'lightweight'."
            )
