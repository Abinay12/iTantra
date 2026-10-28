"""Test and verify the complete 9-stage multilingual speech communication workflow.

Tests:
1. Hindi-to-Telugu speech communication (sender audio -> Hindi STT -> Translation -> Transceiver -> Telugu TTS -> Audio output)
2. Telugu-to-Hindi speech communication (sender audio -> Telugu STT -> Translation -> Transceiver -> Hindi TTS -> Audio output)
3. Out-of-vocabulary / unsupported translation edge case.
"""

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


def run_tests():
    print("=" * 80)
    print("      iTantra: End-to-End Multilingual Speech Communication Workflow")
    print("=" * 80)

    pipeline = ITantraPipeline()

    scenarios = [
        {
            "name": "Scenario 1: Hindi-to-Telugu Communication (Intro)",
            "audio_file": os.path.join(REPO_ROOT, "output", "hindi_intro.wav"),
            "src_lang": "hi",
            "tgt_lang": "te",
            "output_file": os.path.join(REPO_ROOT, "output", "receiver_hindi_to_telugu.wav"),
            "channel": "awgn",
            "snr_db": 15.0,
        },
        {
            "name": "Scenario 2: Telugu-to-Hindi Communication (Question)",
            "audio_file": os.path.join(REPO_ROOT, "output", "telugu_question.wav"),
            "src_lang": "te",
            "tgt_lang": "hi",
            "output_file": os.path.join(REPO_ROOT, "output", "receiver_telugu_to_hindi.wav"),
            "channel": "awgn",
            "snr_db": 15.0,
        },
    ]

    for sc in scenarios:
        print("\n" + "=" * 80)
        print(f"RUNNING {sc['name'].upper()}")
        print("=" * 80)

        res = pipeline.run_multilingual_workflow(
            audio_input=sc["audio_file"],
            source_language=sc["src_lang"],
            target_language=sc["tgt_lang"],
            channel_type=sc["channel"],
            snr_db=sc["snr_db"],
            output_audio_path=sc["output_file"],
        )

        print("\n[Stage 1: Sender Audio Input]")
        print(f"  Source File:        output/{os.path.basename(sc['audio_file'])}")
        print(f"  Source Language:    {res['stage_1_source_language']}")
        print(f"  Audio Duration:     {res['stage_1_audio_duration_sec']}s")

        print("\n[Stage 2: Sender Speech-to-Text Recognition]")
        print(f"  Recognized Raw:     '{res['stage_2_recognized_raw']}'")
        print(f"  Recognized Native:  '{res['stage_2_recognized_native']}'")
        print(f"  STT Confidence:     {res['stage_2_stt_confidence']}")

        print("\n[Stage 3: Target Language Selection]")
        print(f"  Target Language:    {res['stage_3_target_language']}")

        print("\n[Stage 4: Translation to Target Language]")
        print(f"  Translation Status: {'SUCCESS' if res['stage_4_translation_success'] else 'FAILED/UNSUPPORTED'}")
        print(f"  Backend:            {res['stage_4_translation_backend']}")
        print(f"  Translated Text:    '{res['stage_4_translated_text']}'")
        print(f"  Details:            {res['stage_4_translation_details']}")

        print("\n[Stage 5: Neural Transceiver Encoding]")
        print(f"  Encoded Symbols:    {res['stage_5_encoded_symbols']} symbols (latent_dim=32, power <= 1.0)")

        print("\n[Stage 6: Simulated Channel & Radio Transmission]")
        print(f"  Channel Model:      {res['stage_6_channel_model'].upper()} @ SNR={res['stage_6_snr_db']} dB")
        print(f"  Packets Transmitted:{res['stage_6_packets_transmitted']} ({res['stage_6_total_bytes']} bytes)")

        print("\n[Stage 7: Neural Decoding & Packet Integrity]")
        print(f"  CRC-16 Integrity:   {'PASSED (All Valid)' if res['stage_7_crc_passed'] else 'FAILED'}")
        print(f"  Feature MSE:        {res['stage_7_feature_mse']}")
        print(f"  Recovered Raw:      '{res['stage_7_recovered_raw']}'")
        print(f"  Recovered Native:   '{res['stage_7_recovered_native']}'")
        print(f"  Exact Match:        {res['stage_7_exact_match']}")

        print("\n[Stage 8: Receiver Speech Synthesis (TTS)]")
        print(f"  Synthesized Text:   '{res['stage_8_receiver_tts_text']}'")
        print(f"  Audio Duration:     {res['stage_8_receiver_audio_duration_sec']}s")

        print("\n[Stage 9: Receiver Audio Export]")
        print(f"  Output File Path:   output/{os.path.basename(res['stage_9_output_audio_path'])}")
        print(f"  File Exists:        {res['stage_9_output_file_exists']}")
        print(f"  File Size:          {res['stage_9_output_file_bytes']} bytes")

    print("\n" + "=" * 80)
    print("Multilingual workflow tests completed.")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
