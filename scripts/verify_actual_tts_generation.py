"""Verification script for actual TTS generation across all seven target Indian languages.

Languages verified:
1. Hindi (hi)     - facebook/mms-tts-hin
2. Telugu (te)    - facebook/mms-tts-tel
3. Tamil (ta)     - facebook/mms-tts-tam
4. Bengali (bn)   - facebook/mms-tts-ben
5. Marathi (mr)   - facebook/mms-tts-mar
6. Kannada (kn)   - facebook/mms-tts-kan
7. Malayalam (ml) - facebook/mms-tts-mal
"""

import os
import sys
import wave
import json
import numpy as np
import soundfile as sf

# UTF-8 stdout configuration for Windows
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

VERIFICATION_DATA = [
    {
        "code": "hi",
        "name": "Hindi",
        "model_id": "facebook/mms-tts-hin",
        "script": "Devanagari",
        "sample_text": "नमस्ते, आईतन्त्र बहुभाषी प्रणाली में आपका स्वागत है।",
        "filename": "tts_sample_hi.wav",
    },
    {
        "code": "te",
        "name": "Telugu",
        "model_id": "facebook/mms-tts-tel",
        "script": "Telugu",
        "sample_text": "నమస్కారం, ఐతంత్ర బహుభాషా వ్యవస్థకు స్వాగతం.",
        "filename": "tts_sample_te.wav",
    },
    {
        "code": "ta",
        "name": "Tamil",
        "model_id": "facebook/mms-tts-tam",
        "script": "Tamil",
        "sample_text": "வணக்கம், ஐதந்திரா பலமொழி திட்டத்திற்கு வரவேற்கிறோம்.",
        "filename": "tts_sample_ta.wav",
    },
    {
        "code": "bn",
        "name": "Bengali",
        "model_id": "facebook/mms-tts-ben",
        "script": "Bengali",
        "sample_text": "নমস্কার, আইতন্ত্র বহুভাষিক প্রকল্পে আপনাকে স্বাগতম।",
        "filename": "tts_sample_bn.wav",
    },
    {
        "code": "mr",
        "name": "Marathi",
        "model_id": "facebook/mms-tts-mar",
        "script": "Devanagari",
        "sample_text": "नमस्कार, आयतंत्र बहुभाषिक प्रणालीमध्ये आपले स्वागत आहे.",
        "filename": "tts_sample_mr.wav",
    },
    {
        "code": "kn",
        "name": "Kannada",
        "model_id": "facebook/mms-tts-kan",
        "script": "Kannada",
        "sample_text": "ನಮಸ್ಕಾರ, ಐತಂತ್ರ ಬಹುಭಾಷಾ ವ್ಯವಸ್ಥೆಗೆ ನಿಮಗೆ ಸ್ವಾಗತ.",
        "filename": "tts_sample_kn.wav",
    },
    {
        "code": "ml",
        "name": "Malayalam",
        "model_id": "facebook/mms-tts-mal",
        "script": "Malayalam",
        "sample_text": "നമസ്കാരം, ഐതന്ത്ര ബഹുഭാഷാ പദ്ധതിയിലേക്ക് സ്വാഗതം.",
        "filename": "tts_sample_ml.wav",
    },
]


def test_open_wav_audio(file_path: str):
    """Test opening and parsing the audio file with both standard 'wave' and 'soundfile' libraries."""
    if not os.path.exists(file_path):
        return False, "File does not exist", {}

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "File is empty (0 bytes)", {}

    # 1. Standard library wave module test
    try:
        with wave.open(file_path, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            wave_duration = nframes / float(framerate)
    except Exception as e:
        return False, f"wave module failed to open: {e}", {}

    # 2. soundfile library decoding test
    try:
        audio_data, sf_sr = sf.read(file_path)
        sf_duration = float(len(audio_data)) / sf_sr
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        peak = float(np.max(np.abs(audio_data)))
    except Exception as e:
        return False, f"soundfile failed to decode audio: {e}", {}

    # 3. Acoustic parameters validation
    if channels != 1:
        return False, f"Expected 1 channel (mono), got {channels}", {}
    if framerate != 16000 or sf_sr != 16000:
        return False, f"Expected 16000 Hz sample rate, got {framerate}", {}
    if sampwidth != 2:
        return False, f"Expected 16-bit PCM (2 bytes/sample), got {sampwidth}", {}
    if wave_duration <= 0.5:
        return False, f"Audio duration too short: {wave_duration:.2f}s", {}
    if rms < 0.02:
        return False, f"Audio silent or near-zero energy: RMS={rms:.4f}", {}
    if peak > 1.05:
        return False, f"Audio waveform clipped or invalid: Peak={peak:.4f}", {}

    metrics = {
        "file_size": file_size,
        "channels": channels,
        "sample_rate": framerate,
        "sampwidth_bytes": sampwidth,
        "duration_sec": round(wave_duration, 3),
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(peak, 4),
    }
    return True, "Valid WAV Audio", metrics


def main():
    print("=" * 80)
    print("  iTantra: Actual TTS Generation & WAV Validation Suite (7 Languages)")
    print("=" * 80)

    output_dir = os.path.join(REPO_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("\n[Step 1] Initializing TTSEngine with Meta MMS offline backend...")
    tts = TTSEngine(sample_rate=16000, backend="mms")

    print("\n[Step 2] Synthesizing genuine neural speech and verifying WAV outputs...")
    results = []

    for item in VERIFICATION_DATA:
        code = item["code"]
        name = item["name"]
        text = item["sample_text"]
        filename = item["filename"]
        out_path = os.path.join(output_dir, filename)

        print(f"\nProcessing {name} ({code}):")
        print(f"  Model ID:     {item['model_id']}")
        print(f"  Sample Text:  \"{text}\"")
        print(f"  Output Path:  {out_path}")

        try:
            # Generate neural audio using locally cached MMS model
            tts_res = tts.synthesize(text, language=code)

            # Save clean 16-bit PCM WAV
            audio_clipped = np.clip(tts_res.audio, -1.0, 1.0)
            sf.write(out_path, audio_clipped, tts_res.sample_rate, subtype="PCM_16")

            # Open and rigorously validate WAV audio
            is_valid, msg, metrics = test_open_wav_audio(out_path)

            if is_valid:
                print(f"  Result:       SUCCESS ({msg})")
                print(f"  File Size:    {metrics['file_size']:,} bytes")
                print(f"  Duration:     {metrics['duration_sec']}s @ {metrics['sample_rate']} Hz (16-bit mono)")
                print(f"  Acoustics:    RMS={metrics['rms_energy']}, Peak={metrics['peak_amplitude']}")
                results.append({
                    "language": name,
                    "code": code,
                    "model": item["model_id"],
                    "output_file": filename,
                    "output_path": out_path,
                    "status": "SUCCESS",
                    "valid": True,
                    "size_bytes": metrics["file_size"],
                    "duration_sec": metrics["duration_sec"],
                    "rms_energy": metrics["rms_energy"],
                    "peak_amplitude": metrics["peak_amplitude"],
                    "notes": "Verified genuine neural speech"
                })
            else:
                print(f"  Result:       FAILED validation: {msg}")
                results.append({
                    "language": name,
                    "code": code,
                    "model": item["model_id"],
                    "output_file": filename,
                    "output_path": out_path,
                    "status": f"FAILED: {msg}",
                    "valid": False,
                    "size_bytes": 0,
                    "duration_sec": 0.0,
                    "rms_energy": 0.0,
                    "peak_amplitude": 0.0,
                    "notes": msg
                })

        except Exception as e:
            print(f"  Result:       ERROR in generation: {e}")
            results.append({
                "language": name,
                "code": code,
                "model": item["model_id"],
                "output_file": filename,
                "output_path": out_path,
                "status": f"ERROR: {str(e)}",
                "valid": False,
                "size_bytes": 0,
                "duration_sec": 0.0,
                "rms_energy": 0.0,
                "peak_amplitude": 0.0,
                "notes": str(e)
            })

    print("\n" + "=" * 80)
    print("                     ACTUAL TTS VERIFICATION RESULTS TABLE")
    print("=" * 80)
    fmt = "{:<12} | {:<22} | {:<20} | {:<10} | {:<10} | {:<10}"
    print(fmt.format("Language", "Model Identifier", "Generated File", "Duration", "File Size", "Status"))
    print("-" * 80)
    for r in results:
        dur = f"{r['duration_sec']}s" if r['valid'] else "N/A"
        sz = f"{r['size_bytes']:,} B" if r['valid'] else "0 B"
        print(fmt.format(r["language"], r["model"], r["output_file"], dur, sz, r["status"]))
    print("=" * 80)

    # Step 3: Verify Hindi and Telugu preservation
    print("\n[Step 3] Confirming existing Hindi and Telugu Edge-TTS neural voice registry...")
    hi_voice = INDIAN_LANGUAGES["hi"].neural_voice
    te_voice = INDIAN_LANGUAGES["te"].neural_voice
    print(f"  Hindi neural voice:  {hi_voice} (Expected: hi-IN-MadhurNeural)")
    print(f"  Telugu neural voice: {te_voice} (Expected: te-IN-MohanNeural)")
    assert hi_voice == "hi-IN-MadhurNeural", "Hindi neural voice modified!"
    assert te_voice == "te-IN-MohanNeural", "Telugu neural voice modified!"
    print("  Hindi and Telugu voice configurations are verified and preserved.")

    # All passed assertion
    assert all(r["valid"] for r in results), "One or more languages failed TTS verification!"
    print("\nALL SEVEN TARGET LANGUAGES HAVE BEEN RIGOROUSLY GENERATED AND VERIFIED AS VALID WAV AUDIO!")


if __name__ == "__main__":
    main()
