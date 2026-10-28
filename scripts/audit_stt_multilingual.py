"""Independent STT Audit for all seven target Indian languages in iTantra."""

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

LANGUAGES = [
    ("hi", "Hindi", "output/tts_sample_hi.wav", "Devanagari"),
    ("te", "Telugu", "output/tts_sample_te.wav", "Telugu"),
    ("ta", "Tamil", "output/tts_sample_ta.wav", "Tamil"),
    ("bn", "Bengali", "output/tts_sample_bn.wav", "Bengali"),
    ("mr", "Marathi", "output/tts_sample_mr.wav", "Devanagari"),
    ("kn", "Kannada", "output/tts_sample_kn.wav", "Kannada"),
    ("ml", "Malayalam", "output/tts_sample_ml.wav", "Malayalam"),
]

print("================================================================================")
print("             iTantra Independent STT Multilingual Audit (7 Languages)           ")
print("================================================================================")

stt = STTEngine(backend="whisper")

results = []
for code, name, path, script in LANGUAGES:
    full_path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full_path):
        print(f"[{code}] WAV file not found: {full_path}")
        results.append({"code": code, "name": name, "status": "MISSING_AUDIO"})
        continue

    audio, sr = sf.read(full_path)
    try:
        res = stt.transcribe(audio, language=code)
        recognized_text = res.text.strip()
        confidence = res.confidence
        has_text = len(recognized_text) > 0
        status = "OPERATIONAL" if has_text else "EMPTY_RECOGNITION"
        print(f"\n[{code}] {name} ({script}):")
        print(f"  Input Audio:     {path} ({len(audio)/sr:.2f}s @ {sr} Hz)")
        print(f"  Whisper Lang:    {WHISPER_LANG_MAP.get(code)}")
        print(f"  Recognized Text: {recognized_text}")
        print(f"  Native Script:   {res.native_text}")
        print(f"  Confidence:      {confidence:.2f}")
        print(f"  Audit Status:    {status}")
        results.append({
            "code": code,
            "name": name,
            "status": status,
            "recognized_text": recognized_text,
            "confidence": confidence,
            "duration": round(len(audio)/sr, 2)
        })
    except Exception as e:
        print(f"[{code}] ERROR during STT transcription: {e}")
        results.append({"code": code, "name": name, "status": f"ERROR: {e}"})

print("\n" + "=" * 80)
print("                         STT AUDIT SUMMARY TABLE")
print("=" * 80)
print(f"{'Language':<12} | {'Whisper Map':<12} | {'Audio Dur':<10} | {'Status':<15} | {'Recognized Output (Excerpt)':<25}")
print("-" * 80)
for r in results:
    rec = r.get("recognized_text", "N/A")
    if len(rec) > 24:
        rec = rec[:21] + "..."
    dur = f"{r.get('duration', 0)}s"
    print(f"{r['name']:<12} | {WHISPER_LANG_MAP.get(r['code']):<12} | {dur:<10} | {r['status']:<15} | {rec:<25}")
print("=" * 80)
