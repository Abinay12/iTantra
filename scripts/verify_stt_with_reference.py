"""Independent STT verification and reference comparison for all seven target languages."""

import os
import sys
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.speech.stt_engine import STTEngine, WHISPER_LANG_MAP
from src.speech.tts_engine import INDIAN_LANGUAGES

TEST_CASES = [
    {
        "code": "hi",
        "name": "Hindi",
        "reference": "नमस्ते, आईतन्त्र बहुभाषी प्रणाली में आपका स्वागत है।",
        "audio_path": "output/tts_sample_hi.wav",
    },
    {
        "code": "te",
        "name": "Telugu",
        "reference": "నమస్కారం, ఐతంత్ర బహుభాషా వ్యవస్థకు స్వాగతం.",
        "audio_path": "output/tts_sample_te.wav",
    },
    {
        "code": "ta",
        "name": "Tamil",
        "reference": "வணக்கம், ஐதந்திரா பலமொழி திட்டத்திற்கு வரவேற்கிறோம்.",
        "audio_path": "output/tts_sample_ta.wav",
    },
    {
        "code": "bn",
        "name": "Bengali",
        "reference": "নমস্কার, আইতন্ত্র বহুভাষিক প্রকল্পে আপনাকে স্বাগতম।",
        "audio_path": "output/tts_sample_bn.wav",
    },
    {
        "code": "mr",
        "name": "Marathi",
        "reference": "नमस्कार, आयतंत्र बहुभाषिक प्रणालीमध्ये आपले स्वागत आहे.",
        "audio_path": "output/tts_sample_mr.wav",
    },
    {
        "code": "kn",
        "name": "Kannada",
        "reference": "ನಮಸ್ಕಾರ, ಐತಂತ್ರ ಬಹುಭಾಷಾ ವ್ಯವಸ್ಥೆಗೆ ನಿಮಗೆ ಸ್ವಾಗತ.",
        "audio_path": "output/tts_sample_kn.wav",
    },
    {
        "code": "ml",
        "name": "Malayalam",
        "reference": "നമസ്കാരം, ഐതന്ത്ര ബഹുഭാഷാ പദ്ധതിയിലേക്ക് സ്വാഗതം.",
        "audio_path": "output/tts_sample_ml.wav",
    },
]

def main():
    print("=" * 85)
    print("        iTantra: Independent STT Reference Comparison Suite (7 Languages)       ")
    print("=" * 85)

    stt = STTEngine(backend="whisper")

    results = []
    for item in TEST_CASES:
        code = item["code"]
        name = item["name"]
        ref = item["reference"]
        path = os.path.join(REPO_ROOT, item["audio_path"])

        if not os.path.exists(path):
            print(f"[{code}] Audio file missing: {path}")
            continue

        audio, sr = sf.read(path)
        stt_res = stt.transcribe(audio, language=code, reference_text=ref)

        print(f"\n--- {name} ({code}) STT Evaluation ---")
        print(f"  Reference Text:  \"{ref}\"")
        print(f"  Recognized Text: \"{stt_res.text}\"")
        print(f"  Native Script:   \"{stt_res.native_text}\"")
        print(f"  Confidence:      {stt_res.confidence:.2f}")
        print(f"  Audio Duration:  {stt_res.duration_sec:.2f}s @ {sr} Hz")

        results.append({
            "code": code,
            "name": name,
            "reference": ref,
            "recognized": stt_res.text,
            "native": stt_res.native_text,
            "confidence": stt_res.confidence,
            "duration": stt_res.duration_sec,
            "status": "OPERATIONAL" if len(stt_res.text) > 0 else "EMPTY",
        })

    print("\n" + "=" * 85)
    print("                         STT REFERENCE COMPARISON SUMMARY")
    print("=" * 85)
    header = f"{'Language':<12} | {'Whisper ID':<10} | {'Status':<12} | {'Confidence':<10} | {'Recognized Excerpt':<30}"
    print(header)
    print("-" * len(header))
    for r in results:
        rec = r['recognized']
        if len(rec) > 29:
            rec = rec[:26] + "..."
        print(f"{r['name']:<12} | {WHISPER_LANG_MAP.get(r['code']):<10} | {r['status']:<12} | {r['confidence']:<10.2f} | {rec:<30}")
    print("=" * 85)

if __name__ == "__main__":
    main()
