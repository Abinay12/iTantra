"""Test full speech-processing pipeline with AWGN and Rayleigh channel simulations across target languages."""

import os
import sys
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline import ITantraPipeline

TEST_CASES = [
    # (Language, Channel, SNR, Input Text)
    ("hi", "awgn", 20.0, "नमस्ते, यह एडब्ल्यूजीएन चैनल परीक्षण है।"),
    ("hi", "rayleigh", 15.0, "नमस्ते, यह रेली चैनल परीक्षण है।"),
    ("te", "awgn", 20.0, "నమస్కారం, ఇది ఛానల్ పరీక్ష."),
    ("te", "rayleigh", 15.0, "నమస్కారం, ఇది రేలీ ఫేడింగ్ పరీక్ష."),
    ("ta", "awgn", 20.0, "வணக்கம், இது ஏடபிள்யூஜிஎன் சோதனை."),
    ("bn", "rayleigh", 15.0, "নমস্কার, এটি চ্যানেল পরীক্ষা।"),
    ("mr", "awgn", 20.0, "नमस्कार, ही चॅनल चाचणी आहे."),
    ("kn", "rayleigh", 15.0, "ನಮಸ್ಕಾರ, ಇದು ಚಾನೆಲ್ ಪರೀಕ್ಷೆ."),
    ("ml", "awgn", 20.0, "നമസ്കാരം, ഇത് ചാനൽ പരീക്ഷണമാണ്."),
]

def main():
    print("=" * 85)
    print("       iTantra: Speech Pipeline & Wireless Channel Simulation Verification        ")
    print("=" * 85)

    pipeline = ITantraPipeline()

    results = []
    for lang, channel, snr, text in TEST_CASES:
        print(f"\n--- Running Pipeline: {lang.upper()} over {channel.upper()} @ {snr} dB ---")
        print(f"  Input Text: \"{text}\"")

        res = pipeline.run_transmission(
            text=text,
            language=lang,
            channel_type=channel,
            snr_db=snr,
        )

        mse = res["feature_mse"]
        crc_ok = res["crc_integrity_passed"]
        pkts = res["packets_transmitted"]
        rx_txt = res["rx_stt_transcription"]

        print(f"  Duration:          {res['audio_duration_sec']}s")
        print(f"  Packets Sent:      {pkts} (CRC Integrity: {'PASSED' if crc_ok else 'FAILED'})")
        print(f"  Feature MSE Loss:  {mse:.5f}")
        print(f"  Rx Transcribed:    \"{rx_txt}\"")

        results.append({
            "language": lang,
            "channel": channel,
            "snr_db": snr,
            "packets": pkts,
            "crc_passed": crc_ok,
            "mse": mse,
            "rx_transcript": rx_txt,
            "status": "PASSED" if crc_ok and np.isfinite(mse) and pkts > 0 else "FAILED",
        })

    print("\n" + "=" * 85)
    print("                    PIPELINE & CHANNEL SIMULATION SUMMARY")
    print("=" * 85)
    fmt = "{:<8} | {:<10} | {:<8} | {:<10} | {:<12} | {:<12} | {:<8}"
    print(fmt.format("Lang", "Channel", "SNR(dB)", "Packets", "CRC Status", "Feature MSE", "Result"))
    print("-" * 85)
    for r in results:
        crc_str = "VALID" if r['crc_passed'] else "CORRUPT"
        print(fmt.format(r['language'], r['channel'].upper(), f"{r['snr_db']:.1f}", r['packets'], crc_str, f"{r['mse']:.5f}", r['status']))
    print("=" * 85)

    assert all(r['status'] == "PASSED" for r in results), "Pipeline test failed!"
    print("\nALL SPEECH-PROCESSING PIPELINE AND CHANNEL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
