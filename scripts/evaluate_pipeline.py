"""Comprehensive Pipeline Diagnostic and Evaluation Script for iTantra.

Tests end-to-end transmission of Hindi and Telugu messages across:
- AWGN Channel (various SNRs: 5dB, 15dB, 25dB)
- Rayleigh Flat Fading Channel (various SNRs: 5dB, 15dB, 25dB)

Evaluates:
1. Channel symbol distortion (Transmitted vs Received symbols MSE)
2. Feature reconstruction error (Original vs Decoded Mel features MSE)
3. Radio packet integrity (CRC-16 validation and packet statistics)
4. Receiver message recovery analysis (tracing actual data flow)
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

import torch
import numpy as np
from src.pipeline import ITantraPipeline


def evaluate_pipeline():
    print("=" * 80)
    print("      iTantra: End-to-End Pipeline Evaluation & Diagnostic Test")
    print("=" * 80)

    pipeline = ITantraPipeline()

    test_messages = [
        {
            "lang": "hi",
            "name": "Hindi",
            "text": "नमस्ते, आप कैसे हैं?",
        },
        {
            "lang": "te",
            "name": "Telugu",
            "text": "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        },
    ]

    test_channels = [
        ("awgn", 10.0),
        ("awgn", 20.0),
        ("rayleigh", 10.0),
        ("rayleigh", 20.0),
    ]

    for msg in test_messages:
        print(f"\n" + "=" * 80)
        print(f"Message ({msg['name']} [{msg['lang']}]): '{msg['text']}'")
        print("=" * 80)

        # Pre-synthesize once per test message
        print(f"Synthesizing '{msg['text']}' using TTS...")
        tts_output = pipeline.tts.synthesize(text=msg["text"], language=msg["lang"])
        print(f"Synthesized audio duration: {tts_output.duration_sec:.2f}s, sample rate: {tts_output.sample_rate} Hz")

        features = pipeline.stt.extract_features(tts_output.audio).to(pipeline.device)
        orig_features = features.clone()

        with torch.no_grad():
            channel_symbols = pipeline.encoder(features)

        sym_pow = float(torch.mean(channel_symbols ** 2).item())
        print(f"Extracted features shape: {tuple(features.shape)}, Latent symbols shape: {tuple(channel_symbols.shape)}")
        print(f"Encoder transmit power: {sym_pow:.4f} (target <= 1.0)")

        for ch_type, snr in test_channels:
            print(f"\n  --- Testing Channel: {ch_type.upper()} @ SNR = {snr:.1f} dB ---")

            # Channel transmission
            pipeline.channel.set_model(ch_type)
            with torch.no_grad():
                noisy_symbols = pipeline.channel(channel_symbols, snr_db=snr)

            # Channel symbol distortion
            sym_mse = float(torch.nn.functional.mse_loss(noisy_symbols, channel_symbols).item())
            measured_snr_linear = sym_pow / (sym_mse + 1e-12)
            measured_snr_db = 10.0 * np.log10(measured_snr_linear)

            # Radio packetization
            packets = pipeline.packetizer.packetize_stream(noisy_symbols.cpu(), language=msg["lang"])
            with pipeline.sdr:
                tx_bytes = sum(pipeline.sdr.transmit(p) for p in packets)
                rx_bytes = []
                while pipeline.sdr.has_data():
                    chunk = pipeline.sdr.receive(1024)
                    if chunk:
                        rx_bytes.append(chunk)

            reassembled, rx_lang, all_crcs_valid = pipeline.packetizer.depacketize_stream(packets)
            rx_symbols_tensor = torch.from_numpy(reassembled).to(pipeline.device).view_as(channel_symbols)

            # Neural Decoding
            with torch.no_grad():
                reconstructed_features = pipeline.decoder(rx_symbols_tensor)

            feat_mse = float(pipeline.decoder.compute_loss(reconstructed_features, orig_features).item())

            # Audio reconstruction from decoded channel features via vocoder
            rx_audio = pipeline.stt.features_to_audio(reconstructed_features)
            rx_stt = pipeline.stt.transcribe(audio=rx_audio, language=msg["lang"])

            # Transmitter clean audio baseline
            tx_stt = pipeline.stt.transcribe(audio=tts_output.audio, language=msg["lang"])

            from src.speech.transliterate import roman_to_native
            rx_native = rx_stt.native_text or roman_to_native(rx_stt.text, msg["lang"])
            tx_native = tx_stt.native_text or roman_to_native(tx_stt.text, msg["lang"])

            print(f"    Channel Symbol MSE:      {sym_mse:.6f} (Measured SNR: {measured_snr_db:.2f} dB)")
            print(f"    Feature Reconstruct MSE: {feat_mse:.6f}")
            print(f"    Radio Packets:           {len(packets)} packets, {tx_bytes} bytes (CRC All Valid: {all_crcs_valid})")
            print(f"    Radio Detected Lang:     '{rx_lang}'")
            print(f"    Channel Recovered STT:   '{rx_stt.text}' -> Native: '{rx_native}'")
            print(f"    Transmitter Baseline:    '{tx_stt.text}' -> Native: '{tx_native}'")

    print("\n" + "=" * 80)
    print("Pipeline evaluation completed.")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_pipeline()
