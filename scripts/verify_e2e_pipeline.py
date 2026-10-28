"""Verification script for actual end-to-end communication in iTantra."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.pipeline import ITantraPipeline

pipeline = ITantraPipeline()

test_cases = [
    ("Hindi", "hi", "नमस्ते, आप कैसे हैं?"),
    ("Telugu", "te", "నమస్కారం, మీరు ఎలా ఉన్నారు?"),
]

channels = [
    ("AWGN", "awgn", 15.0),
    ("Rayleigh", "rayleigh", 15.0),
]

for name, lang, text in test_cases:
    print("=" * 75)
    print(f"LANGUAGE: {name} ({lang})")
    print(f"ORIGINAL MESSAGE: '{text}'")
    print("=" * 75)
    for ch_name, ch_type, snr in channels:
        res = pipeline.run_transmission(text=text, language=lang, channel_type=ch_type, snr_db=snr)
        exact_match = (res["rx_stt_native"] == text)
        print(f"\n--- Channel: {ch_name} @ {snr:.1f} dB ---")
        print(f"Audio Duration:          {res['audio_duration_sec']}s")
        print(f"Feature Reconstruct MSE: {res['feature_mse']}")
        print(f"Radio Packets:           {res['packets_transmitted']} ({res['total_bytes']} bytes)")
        print(f"CRC-16 Passed:           {res['crc_integrity_passed']}")
        print(f"Channel Recovered STT:   '{res['rx_stt_transcription']}'")
        print(f"Channel Recovered Native:'{res['rx_stt_native']}'")
        print(f"Tx Clean Baseline STT:   '{res['tx_stt_transcription']}'")
        print(f"Tx Clean Baseline Native:'{res['tx_stt_native']}'")
        print(f"Exact Match to Input:    {exact_match}")
    print()
