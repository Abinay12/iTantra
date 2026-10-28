"""Generate and rigorously verify genuine neural TTS samples for all seven target Indian languages.

Languages:
1. Hindi (hi)     - facebook/mms-tts-hin
2. Telugu (te)    - facebook/mms-tts-tel
3. Tamil (ta)     - facebook/mms-tts-tam
4. Bengali (bn)   - facebook/mms-tts-ben
5. Marathi (mr)   - facebook/mms-tts-mar
6. Kannada (kn)   - facebook/mms-tts-kan
7. Malayalam (ml) - facebook/mms-tts-mal

Outputs are saved in the project output directory following project naming convention:
output/tts_sample_<lang>.wav
"""

import os
import sys
import wave
import numpy as np
import soundfile as sf

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.speech.tts_engine import TTSEngine, INDIAN_LANGUAGES

# Target Indian languages with authentic native-script test phrases
TARGET_LANGUAGES = [
    {
        "code": "hi",
        "name": "Hindi",
        "model_id": "facebook/mms-tts-hin",
        "text": "नमस्ते, आईतन्त्र बहुभाषी प्रणाली में आपका स्वागत है।",
        "filename": "tts_sample_hi.wav",
    },
    {
        "code": "te",
        "name": "Telugu",
        "model_id": "facebook/mms-tts-tel",
        "text": "నమస్కారం, ఐతంత్ర బహుభాషా వ్యవస్థకు స్వాగతం.",
        "filename": "tts_sample_te.wav",
    },
    {
        "code": "ta",
        "name": "Tamil",
        "model_id": "facebook/mms-tts-tam",
        "text": "வணக்கம், ஐதந்திரா பலமொழி திட்டத்திற்கு வரவேற்கிறோம்.",
        "filename": "tts_sample_ta.wav",
    },
    {
        "code": "bn",
        "name": "Bengali",
        "model_id": "facebook/mms-tts-ben",
        "text": "নমস্কার, আইতন্ত্র বহুভাষিক প্রকল্পে আপনাকে স্বাগতম।",
        "filename": "tts_sample_bn.wav",
    },
    {
        "code": "mr",
        "name": "Marathi",
        "model_id": "facebook/mms-tts-mar",
        "text": "नमस्कार, आयतंत्र बहुभाषिक प्रणालीमध्ये आपले स्वागत आहे.",
        "filename": "tts_sample_mr.wav",
    },
    {
        "code": "kn",
        "name": "Kannada",
        "model_id": "facebook/mms-tts-kan",
        "text": "ನಮಸ್ಕಾರ, ಐತಂತ್ರ ಬಹುಭಾಷಾ ವ್ಯವಸ್ಥೆಗೆ ನಿಮಗೆ ಸ್ವಾಗತ.",
        "filename": "tts_sample_kn.wav",
    },
    {
        "code": "ml",
        "name": "Malayalam",
        "model_id": "facebook/mms-tts-mal",
        "text": "നമസ്കാരം, ഐതന്ത്ര ബഹുഭാഷാ പദ്ധതിയിലേക്ക് സ്വാഗതം.",
        "filename": "tts_sample_ml.wav",
    },
]


def verify_wav_file(file_path: str, expected_sr: int = 16000) -> dict:
    """Verify that a generated file exists, is non-empty, and has a valid WAV header and acoustic content."""
    if not os.path.exists(file_path):
        return {"valid": False, "error": "File does not exist"}

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return {"valid": False, "error": "File is empty (0 bytes)"}

    # Verify standard RIFF/WAVE header via wave module
    try:
        with wave.open(file_path, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
    except Exception as e:
        return {"valid": False, "error": f"Invalid WAV header: {e}"}

    # Read audio samples with soundfile
    try:
        data, sr = sf.read(file_path)
    except Exception as e:
        return {"valid": False, "error": f"Failed reading audio data: {e}"}

    duration = float(len(data)) / sr
    rms = float(np.sqrt(np.mean(data ** 2)))
    peak = float(np.max(np.abs(data)))

    # Acoustic checks
    is_valid_audio = (
        channels == 1
        and sr == expected_sr
        and duration > 0.5
        and rms > 0.02
        and peak <= 1.05
    )

    return {
        "valid": is_valid_audio,
        "file_size_bytes": file_size,
        "channels": channels,
        "sample_rate": sr,
        "duration_sec": round(duration, 3),
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(peak, 4),
        "error": None if is_valid_audio else "Acoustic parameters out of bounds",
    }


def main():
    print("=" * 80)
    print("    iTantra: Genuine Neural TTS Generation & Verification Suite (7 Languages)")
    print("=" * 80)

    output_dir = os.path.join(REPO_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TTS Engine with MMS backend
    print("\n[Step 1] Initializing TTSEngine with Meta MMS offline backend...")
    tts = TTSEngine(sample_rate=16000, backend="mms")

    # Confirm all 7 models are cached locally
    print("\n[Step 2] Checking local cache status for all seven models...")
    for item in TARGET_LANGUAGES:
        code = item["code"]
        cached = tts.is_mms_model_cached(code)
        print(f"  [{code}] {item['name']:<10} ({item['model_id']}): cached = {cached}")
        if not cached:
            print(f"  ERROR: Model for {item['name']} is NOT cached! Exiting.")
            sys.exit(1)

    print("\n[Step 3] Generating speech samples using native script and locally cached models...")
    results = []
    for item in TARGET_LANGUAGES:
        code = item["code"]
        name = item["name"]
        text = item["text"]
        filename = item["filename"]
        out_path = os.path.join(output_dir, filename)

        print(f"\n--- Synthesizing {name} ({code}) ---")
        print(f"  Model ID:   {item['model_id']}")
        print(f"  Input Text: \"{text}\"")
        print(f"  Target WAV: {out_path}")

        try:
            output = tts.synthesize(text, language=code)
            # Save audio to 16-bit PCM WAV
            audio_clipped = np.clip(output.audio, -1.0, 1.0)
            sf.write(out_path, audio_clipped, output.sample_rate, subtype="PCM_16")

            # Rigorously verify file
            v_res = verify_wav_file(out_path, expected_sr=16000)

            status = "SUCCESS" if v_res["valid"] else f"FAILED: {v_res['error']}"
            print(f"  Generation: {status}")
            print(f"  WAV Size:   {v_res.get('file_size_bytes', 0):,} bytes")
            print(f"  Duration:   {v_res.get('duration_sec', 0.0)}s @ {v_res.get('sample_rate', 0)} Hz")
            print(f"  Acoustics:  RMS={v_res.get('rms_energy', 0.0)}, Peak={v_res.get('peak_amplitude', 0.0)}")

            results.append({
                "code": code,
                "name": name,
                "model_id": item["model_id"],
                "text": text,
                "output_path": out_path,
                "filename": filename,
                "status": status,
                "valid": v_res["valid"],
                "size_bytes": v_res.get("file_size_bytes", 0),
                "duration_sec": v_res.get("duration_sec", 0.0),
                "rms_energy": v_res.get("rms_energy", 0.0),
                "peak_amplitude": v_res.get("peak_amplitude", 0.0),
            })
        except Exception as e:
            print(f"  ERROR during synthesis: {e}")
            results.append({
                "code": code,
                "name": name,
                "model_id": item["model_id"],
                "text": text,
                "output_path": out_path,
                "filename": filename,
                "status": f"ERROR: {str(e)}",
                "valid": False,
                "size_bytes": 0,
                "duration_sec": 0.0,
                "rms_energy": 0.0,
                "peak_amplitude": 0.0,
            })

    print("\n" + "=" * 80)
    print("                     GENERATION & VERIFICATION SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Language':<12} | {'Model Identifier':<24} | {'Output File':<20} | {'Duration':<10} | {'Size':<10} | {'Status':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        dur_str = f"{r['duration_sec']}s" if r['valid'] else "N/A"
        size_str = f"{r['size_bytes']:,} B" if r['valid'] else "0 B"
        status_str = "VERIFIED" if r['valid'] else "FAILED"
        print(f"{r['name']:<12} | {r['model_id']:<24} | {r['filename']:<20} | {dur_str:<10} | {size_str:<10} | {status_str:<10}")
    print("=" * 80)

    all_passed = all(r["valid"] for r in results)
    if all_passed:
        print("\nAll 7 languages successfully generated genuine neural audio samples and passed verification!")
    else:
        print("\nSome languages failed verification. Please review the table above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
