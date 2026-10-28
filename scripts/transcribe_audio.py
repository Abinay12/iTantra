"""Speech-to-Text (STT) Transcription Script for iTantra.

Transcribes arbitrary WAV audio files into text using iTantra's neural STT
engine (powered by OpenAI Whisper running locally and offline on CPU).

Features:
  - Raw phonetic transcription (as emitted by Whisper).
  - Native script display (Devanagari for Hindi, Telugu script for Telugu).
  - Configurable script display modes (--script native, roman, or both).
  - Standalone single-file transcription or full verification suite.
"""

import argparse
import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Enable UTF-8 encoding for console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import soundfile as sf
from scipy.signal import resample_poly
import math
import numpy as np

from src.speech.stt_engine import STTEngine, INDIAN_LANGUAGES


def transcribe_file(
    file_path: str,
    language: Optional[str] = None,
    backend: str = "whisper",
    stt_engine: STTEngine = None,
) -> dict:
    """Load and transcribe a WAV audio file into text.

    Args:
        file_path: Path to the WAV audio file.
        language: Optional Indian language code ('te', 'hi', 'ta', 'bn', 'mr', 'kn', 'ml').
                  If None, language is detected automatically using Whisper.
        backend: STT backend ('whisper', 'auto', 'lightweight').
        stt_engine: Optional pre-initialized STTEngine instance.

    Returns:
        Dictionary containing transcription, native script, confidence, and metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio_data, source_sr = sf.read(file_path)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    target_sr = 16000
    if source_sr != target_sr:
        gcd = math.gcd(target_sr, source_sr)
        up = target_sr // gcd
        down = source_sr // gcd
        audio_16k = resample_poly(audio_data, up, down).astype(np.float32)
    else:
        audio_16k = audio_data.astype(np.float32)

    if stt_engine is None:
        stt_engine = STTEngine(sample_rate=target_sr, backend=backend)

    result = stt_engine.transcribe(audio_16k, language=language)

    effective_lang = result.language
    return {
        "file_path": file_path,
        "filename": os.path.basename(file_path),
        "language": effective_lang,
        "lang_name": INDIAN_LANGUAGES.get(effective_lang, None).name if effective_lang in INDIAN_LANGUAGES else effective_lang,
        "duration_sec": result.duration_sec,
        "raw_text": result.text,
        "native_text": result.native_text or result.text,
        "confidence": result.confidence,
        "feature_shape": result.feature_shape,
        "detected_language": result.detected_language,
        "language_confidence": result.language_confidence,
    }


def run_stt_verification_suite(backend: str = "whisper", script_mode: str = "both"):
    """Run transcription verification across real Hindi and Telugu recordings."""
    print("=" * 70)
    print("  iTantra: Neural Speech-to-Text (STT) Verification Suite")
    print("=" * 70)
    print(f"Active Backend: {backend.upper()} (Pretrained Neural Model)")
    print(f"Script Mode:    {script_mode.upper()}\n")

    stt = STTEngine(sample_rate=16000, backend=backend)

    test_cases = [
        # Hindi recordings
        {
            "label": "Hindi Sentence 1 (Intro)",
            "file": os.path.join(REPO_ROOT, "output", "hindi_intro.wav"),
            "language": "hi",
            "expected_text": "नमस्ते, मेरा नाम अभिनय है।",
        },
        {
            "label": "Hindi Sentence 2 (Question)",
            "file": os.path.join(REPO_ROOT, "output", "hindi_question.wav"),
            "language": "hi",
            "expected_text": "आप कैसे हैं?",
        },
        # Telugu recordings
        {
            "label": "Telugu Sentence 1 (Intro)",
            "file": os.path.join(REPO_ROOT, "output", "telugu_intro.wav"),
            "language": "te",
            "expected_text": "నమస్కారం, నా పేరు అభినయ్.",
        },
        {
            "label": "Telugu Sentence 2 (Question)",
            "file": os.path.join(REPO_ROOT, "output", "telugu_question.wav"),
            "language": "te",
            "expected_text": "మీరు ఎలా ఉన్నారు?",
        },
    ]

    results = []
    for item in test_cases:
        print(f"[{item['label']}]")
        print(f"  Input File:     output/{os.path.basename(item['file'])}")
        print(f"  Expected Text:  {item['expected_text']}")

        if not os.path.exists(item["file"]):
            print(f"  [ERROR] File does not exist.\n")
            continue

        res = transcribe_file(
            file_path=item["file"],
            language=item["language"],
            stt_engine=stt,
        )
        results.append(res)

        if script_mode in ("both", "native"):
            print(f"  Native Script:  {res['native_text']}")
        if script_mode in ("both", "roman"):
            print(f"  Model Raw Text: {res['raw_text']}")

        print(f"  Confidence Est: {res['confidence']} (heuristic probability)")
        print(f"  Audio Duration: {res['duration_sec']:.2f}s\n")

    print("=" * 70)
    print("STT verification completed.")
    print("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="iTantra: Speech-to-Text (STT) Transcription Script"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to WAV audio file to transcribe",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="auto",
        choices=["auto", "te", "hi", "ta", "bn", "mr", "kn", "ml"],
        help="Language code ('auto' for automatic language detection, or specific code; default: auto)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="whisper",
        choices=["whisper", "auto", "lightweight"],
        help="STT backend (default: whisper)",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="both",
        choices=["both", "native", "roman"],
        help="Display mode: 'native' (Devanagari/Telugu), 'roman' (Latin letters), or 'both' (default: both)",
    )

    args = parser.parse_args()

    if args.file:
        lang_to_use = None if args.lang == "auto" else args.lang
        res = transcribe_file(
            file_path=args.file,
            language=lang_to_use,
            backend=args.backend,
        )
        print("=" * 60)
        print(f"File:           {res['filename']} ({res['duration_sec']:.2f}s)")
        if res.get("detected_language"):
            print(f"Detected Lang:  {res['lang_name']} ({res['detected_language']}) [Confidence: {res['language_confidence']:.2f}]")
        else:
            print(f"Language:       {res['lang_name']} ({res['language']})")
        if args.script in ("both", "native"):
            print(f"Native Script:  {res['native_text']}")
        if args.script in ("both", "roman"):
            print(f"Model Raw Text: {res['raw_text']}")
        print(f"Confidence Est: {res['confidence']}")
        print("=" * 60)
    else:
        run_stt_verification_suite(backend=args.backend, script_mode=args.script)


if __name__ == "__main__":
    main()
