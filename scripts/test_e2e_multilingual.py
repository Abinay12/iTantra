"""Comprehensive End-to-End Integration Test of the Multilingual Speech Workflow.

Workflow Stages:
  1. Load real Hindi / Telugu WAV recording.
  2. Recognize spoken speech using STTEngine (Whisper-tiny + native script transliteration).
  3. Translate recognized text into target language using TranslationEngine.
  4. Encode resulting message using NeuralEncoder.
  5. Transmit through simulated wireless channel (AWGN & Rayleigh tested separately).
  6. Decode received message with NeuralDecoder and verify packet CRC integrity.
  7. Generate target speech from recovered text using TTSEngine.
  8. Save receiver audio in output/ directory.
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


def run_e2e_integration_tests():
    print("=" * 80)
    print("      iTantra: Real End-to-End Multilingual Speech Workflow Test")
    print("=" * 80)

    pipeline = ITantraPipeline()

    test_runs = [
        {
            "title": "Hindi to Telugu via AWGN Channel",
            "audio_file": os.path.join(REPO_ROOT, "output", "hindi_question.wav"),
            "src_lang": "hi",
            "tgt_lang": "te",
            "channel": "awgn",
            "snr_db": 15.0,
            "output_wav": os.path.join(REPO_ROOT, "output", "receiver_hindi_to_telugu_awgn.wav"),
        },
        {
            "title": "Hindi to Telugu via Rayleigh Channel",
            "audio_file": os.path.join(REPO_ROOT, "output", "hindi_question.wav"),
            "src_lang": "hi",
            "tgt_lang": "te",
            "channel": "rayleigh",
            "snr_db": 15.0,
            "output_wav": os.path.join(REPO_ROOT, "output", "receiver_hindi_to_telugu_rayleigh.wav"),
        },
        {
            "title": "Telugu to Hindi via AWGN Channel",
            "audio_file": os.path.join(REPO_ROOT, "output", "telugu_question.wav"),
            "src_lang": "te",
            "tgt_lang": "hi",
            "channel": "awgn",
            "snr_db": 15.0,
            "output_wav": os.path.join(REPO_ROOT, "output", "receiver_telugu_to_hindi_awgn.wav"),
        },
        {
            "title": "Telugu to Hindi via Rayleigh Channel",
            "audio_file": os.path.join(REPO_ROOT, "output", "telugu_question.wav"),
            "src_lang": "te",
            "tgt_lang": "hi",
            "channel": "rayleigh",
            "snr_db": 15.0,
            "output_wav": os.path.join(REPO_ROOT, "output", "receiver_telugu_to_hindi_rayleigh.wav"),
        },
    ]

    results = []

    for run in test_runs:
        print("\n" + "=" * 80)
        print(f"TEST: {run['title'].upper()}")
        print("=" * 80)

        res = pipeline.run_multilingual_workflow(
            audio_input=run["audio_file"],
            source_language=run["src_lang"],
            target_language=run["tgt_lang"],
            channel_type=run["channel"],
            snr_db=run["snr_db"],
            output_audio_path=run["output_wav"],
        )
        results.append((run, res))

        print(f"1. Sender Audio Input:     output/{os.path.basename(run['audio_file'])} ({res['stage_1_audio_duration_sec']}s)")
        print(f"2. Recognized Spoken Text: '{res['stage_2_recognized_native']}' (raw: '{res['stage_2_recognized_raw']}')")
        print(f"3. Target Language:        {res['stage_3_target_language']}")
        print(f"4. Translated Text:        '{res['stage_4_translated_text']}'")
        print(f"   Translation Backend:    {res['stage_4_translation_backend']} (Success: {res['stage_4_translation_success']})")
        print(f"5. Transceiver Encoding:   {res['stage_5_encoded_symbols']} latent symbols (power <= 1.0)")
        print(f"6. Channel Transmission:   {res['stage_6_channel_model'].upper()} @ SNR={res['stage_6_snr_db']} dB")
        print(f"   Packets over SDR:       {res['stage_6_packets_transmitted']} packets ({res['stage_6_total_bytes']} bytes)")
        print(f"7. Decoded & Integrity:    CRC-16 Passed: {res['stage_7_crc_passed']} | Feature MSE: {res['stage_7_feature_mse']}")
        print(f"   Recovered Text:         '{res['stage_7_recovered_native']}' (raw: '{res['stage_7_recovered_raw']}')")
        print(f"   Exact Match:            {res['stage_7_exact_match']}")
        print(f"8. Receiver TTS Generated: '{res['stage_8_receiver_tts_text']}' ({res['stage_8_receiver_audio_duration_sec']}s)")
        print(f"9. Receiver Audio File:    output/{os.path.basename(res['stage_9_output_audio_path'])}")
        print(f"   File Exists & Size:     {res['stage_9_output_file_exists']} ({res['stage_9_output_file_bytes']} bytes)")

    print("\n" + "=" * 80)
    print("All end-to-end integration tests completed.")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_e2e_integration_tests()
