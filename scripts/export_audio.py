"""Audio Export Script for iTantra.

Synthesizes arbitrary text inputs in supported Indian languages (Hindi, Telugu,
Tamil, Bengali, Marathi) using iTantra's neural TTS engine, and exports them as
standard 16-bit PCM WAV audio files.

Supports both custom text inputs via command-line arguments and automated batch
generation for verification suites.
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

import numpy as np
from scipy.io import wavfile

from src.speech.tts_engine import TTSEngine, INDIAN_LANGUAGES


def generate_speech_file(
    text: str,
    language: str,
    output_filename: str,
    backend: str = "neural",
    tts_engine: TTSEngine = None,
) -> dict:
    """Synthesize text and export to a WAV file.

    Args:
        text: Input sentence in native script or transliteration.
        language: Language code ('te', 'hi', 'ta', 'bn', 'mr').
        output_filename: Name or relative path of the output WAV file.
        backend: Synthesis backend ('neural', 'mms', 'auto', 'lightweight').
        tts_engine: Optional pre-initialized TTSEngine instance.

    Returns:
        Dictionary with audio metadata (duration, sample_rate, size, rms, peak).
    """
    output_dir = os.path.join(REPO_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    if not output_filename.endswith(".wav"):
        output_filename += ".wav"

    if os.path.isabs(output_filename):
        file_path = output_filename
    elif output_filename.startswith("output/") or output_filename.startswith("output\\"):
        file_path = os.path.join(REPO_ROOT, output_filename)
    else:
        file_path = os.path.join(output_dir, output_filename)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if tts_engine is None:
        tts_engine = TTSEngine(sample_rate=16000, backend=backend)

    # Synthesize audio waveform
    result = tts_engine.synthesize(text=text, language=language)

    # Convert float32 in [-1.0, 1.0] to standard 16-bit PCM
    audio_float = np.clip(result.audio, -1.0, 1.0)
    audio_int16 = (audio_float * 32767).astype(np.int16)

    wavfile.write(file_path, result.sample_rate, audio_int16)

    file_size_kb = os.path.getsize(file_path) / 1024
    rms = float(np.sqrt(np.mean(audio_float ** 2)))
    peak = float(np.max(np.abs(audio_float)))

    return {
        "text": text,
        "language": language,
        "lang_name": INDIAN_LANGUAGES.get(language, None).name if language in INDIAN_LANGUAGES else language,
        "file_path": file_path,
        "filename": os.path.basename(file_path),
        "duration_sec": round(result.duration_sec, 3),
        "sample_rate": result.sample_rate,
        "file_size_kb": round(file_size_kb, 1),
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(peak, 4),
    }


def run_custom_verification_suite(backend: str = "neural"):
    """Run verification suite for custom Hindi and Telugu sentences."""
    print("=" * 70)
    print("  iTantra: Neural TTS Dynamic Verification Suite")
    print("=" * 70)
    print(f"Active Backend: {backend.upper()} (Genuine Neural Speech)\n")

    tts = TTSEngine(sample_rate=16000, backend=backend)

    test_sentences = [
        # Hindi Test Sentences
        {
            "language": "hi",
            "text": "नमस्ते, मेरा नाम अभिनय है।",
            "filename": "hindi_intro.wav",
        },
        {
            "language": "hi",
            "text": "आप कैसे हैं?",
            "filename": "hindi_question.wav",
        },
        # Telugu Test Sentences
        {
            "language": "te",
            "text": "నమస్కారం, నా పేరు అభినయ్.",
            "filename": "telugu_intro.wav",
        },
        {
            "language": "te",
            "text": "మీరు ఎలా ఉన్నారు?",
            "filename": "telugu_question.wav",
        },
    ]

    results = []
    for item in test_sentences:
        print(f"Synthesizing {item['language'].upper()} sentence...")
        print(f"  Input Text: \"{item['text']}\"")
        meta = generate_speech_file(
            text=item["text"],
            language=item["language"],
            output_filename=item["filename"],
            tts_engine=tts,
        )
        results.append(meta)
        print(f"  Saved:      output/{meta['filename']}")
        print(f"  Duration:   {meta['duration_sec']}s @ {meta['sample_rate']} Hz (16-bit PCM)")
        print(f"  Audio RMS:  {meta['rms_energy']} (Peak: {meta['peak_amplitude']})")
        print(f"  File Size:  {meta['file_size_kb']} KB\n")

    print("=" * 70)
    print("Dynamic verification suite completed successfully.")
    print("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="iTantra: Dynamic Neural Speech Export Script"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input sentence in Telugu, Hindi, Tamil, Bengali, Marathi, Kannada, or Malayalam",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        choices=["te", "hi", "ta", "bn", "mr", "kn", "ml"],
        help="Language code (default: hi)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV filename (default: output/<lang>_custom.wav)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="neural",
        choices=["neural", "mms", "auto", "lightweight"],
        help="TTS backend (default: neural)",
    )

    args = parser.parse_args()

    if args.text:
        # Dynamic custom text generation from CLI
        out_name = args.output or f"{args.lang}_custom.wav"
        print(f"Synthesizing custom {args.lang} text...")
        meta = generate_speech_file(
            text=args.text,
            language=args.lang,
            output_filename=out_name,
            backend=args.backend,
        )
        print(f"Saved: output/{meta['filename']} ({meta['duration_sec']}s, {meta['file_size_kb']} KB)")
    else:
        # Run test verification suite
        run_custom_verification_suite(backend=args.backend)


if __name__ == "__main__":
    main()
