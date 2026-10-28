"""Train lightweight Neural Transceiver (JSCC) autoencoder on speech audio.

Trains NeuralEncoder and NeuralDecoder to reconstruct 80-bin log-Mel
spectrogram frames under power normalization and wireless channel noise.
Saves checkpoint to configs/transceiver_weights.pt.
"""

import glob
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

from src.transceiver.channel import AWGNChannel
from src.transceiver.decoder import NeuralDecoder
from src.transceiver.encoder import NeuralEncoder


def train_transceiver(
    epochs: int = 150,
    lr: float = 1e-3,
    save_path: str = "configs/transceiver_weights.pt",
):
    print("=" * 70)
    print("  iTantra: Training Neural Transceiver (JSCC) Autoencoder")
    print("=" * 70)

    # Gather audio files from output/
    audio_files = sorted(glob.glob(os.path.join(REPO_ROOT, "output", "*.wav")))
    if not audio_files:
        print("No WAV files found in output/. Cannot train.")
        return

    print(f"Found {len(audio_files)} speech audio files for training:")
    for f in audio_files:
        print(f"  - output/{os.path.basename(f)}")

    # Extract log-Mel spectrogram features
    mel_tf = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        power=2.0,
    )

    feature_list = []
    for f in audio_files:
        data, sr = sf.read(f)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        mel = mel_tf(t)
        log_mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2)
        feature_list.append(log_mel)

    X = torch.cat(feature_list, dim=1)  # (1, total_frames, 80)
    print(f"\nAggregated Training Tensor Shape: {tuple(X.shape)}")

    # Initialize encoder, decoder, channel
    encoder = NeuralEncoder(input_dim=80, hidden_dim=128, latent_dim=32, normalize_power=True)
    decoder = NeuralDecoder(latent_dim=32, hidden_dim=128, output_dim=80)
    channel = AWGNChannel(default_snr_db=15.0)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )

    print("\nTraining Neural Transceiver over AWGN channel...")
    start_time = time.time()
    initial_loss = None

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        symbols = encoder(X)
        noisy_symbols = channel(symbols, snr_db=20.0)
        reconstructed = decoder(noisy_symbols)
        loss = torch.nn.functional.mse_loss(reconstructed, X)
        loss.backward()
        optimizer.step()

        if epoch == 1:
            initial_loss = loss.item()
        if epoch % 25 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs:3d} | Reconstruction MSE: {loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s.")
    print(f"Initial MSE: {initial_loss:.4f} -> Final MSE: {loss.item():.4f}")

    # Save checkpoint
    checkpoint = {
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "input_dim": 80,
        "hidden_dim": 128,
        "latent_dim": 32,
        "final_loss": loss.item(),
        "trained_epochs": epochs,
    }
    full_save_path = os.path.join(REPO_ROOT, save_path)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    torch.save(checkpoint, full_save_path)
    print(f"Saved trained transceiver checkpoint to: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_transceiver()
