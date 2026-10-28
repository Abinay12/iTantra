"""Unit tests for wireless channel simulations and neural transceiver components."""

import unittest
import torch
import numpy as np

from src.transceiver.channel import (
    PowerNormalization,
    AWGNChannel,
    RayleighChannel,
    WirelessChannel,
)
from src.transceiver.encoder import NeuralEncoder
from src.transceiver.decoder import NeuralDecoder


class TestPowerNormalization(unittest.TestCase):
    """Test average power normalization layer."""

    def test_normalization_constraint(self):
        norm_layer = PowerNormalization(target_power=1.0)
        # Random input with high arbitrary power
        x = torch.randn(10, 50, 32) * 5.0
        normalized = norm_layer(x)

        # Average power along last dimension should be close to 1.0
        avg_power = torch.mean(normalized ** 2, dim=-1)
        self.assertTrue(torch.allclose(avg_power, torch.ones_like(avg_power), atol=1e-3))


class TestAWGNChannel(unittest.TestCase):
    """Test AWGN channel simulation."""

    def test_output_shape(self):
        channel = AWGNChannel(default_snr_db=10.0)
        x = torch.randn(4, 20, 32)
        y = channel(x)
        self.assertEqual(y.shape, x.shape)

    def test_noise_scaling_with_snr(self):
        channel = AWGNChannel()
        x = torch.ones(100, 100)

        # High SNR -> small noise
        y_high_snr = channel(x, snr_db=30.0)
        err_high = torch.mean((y_high_snr - x) ** 2).item()

        # Low SNR -> large noise
        y_low_snr = channel(x, snr_db=0.0)
        err_low = torch.mean((y_low_snr - x) ** 2).item()

        self.assertLess(err_high, err_low)
        self.assertLess(err_high, 0.01)
        self.assertGreater(err_low, 0.5)


class TestRayleighChannel(unittest.TestCase):
    """Test Rayleigh flat fading channel simulation."""

    def test_output_shape_and_csi(self):
        channel = RayleighChannel(default_snr_db=15.0, equalize=True)
        x = torch.randn(8, 16, 32)
        y, csi = channel(x, return_csi=True)

        self.assertEqual(y.shape, x.shape)
        # CSI contains (h_r, h_i) components
        self.assertEqual(csi.shape[-1], 2)

    def test_fading_with_equalization(self):
        channel = RayleighChannel(default_snr_db=25.0, equalize=True)
        x = torch.ones(32, 32)
        y = channel(x, snr_db=25.0)

        # With high SNR and zero-forcing equalization, signal should be approximately recovered
        diff = torch.mean(torch.abs(y - x)).item()
        self.assertLess(diff, 0.5)


class TestWirelessChannelDispatcher(unittest.TestCase):
    """Test unified channel dispatcher."""

    def test_model_switching(self):
        wc = WirelessChannel(model_type="awgn")
        x = torch.randn(2, 10)
        y_awgn = wc(x)
        self.assertEqual(y_awgn.shape, x.shape)

        wc.set_model("rayleigh")
        y_ray = wc(x)
        self.assertEqual(y_ray.shape, x.shape)

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            WirelessChannel(model_type="invalid_channel_type")


class TestNeuralTransceiver(unittest.TestCase):
    """Test Neural Encoder and Decoder."""

    def setUp(self):
        self.input_dim = 80
        self.latent_dim = 32
        self.hidden_dim = 64
        self.encoder = NeuralEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            normalize_power=True,
        )
        self.decoder = NeuralDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.input_dim,
        )

    def test_shapes(self):
        batch_size = 4
        seq_len = 25
        x = torch.randn(batch_size, seq_len, self.input_dim)

        symbols = self.encoder(x)
        self.assertEqual(symbols.shape, (batch_size, seq_len, self.latent_dim))

        reconstructed = self.decoder(symbols)
        self.assertEqual(reconstructed.shape, x.shape)

    def test_input_dimension_validation(self):
        x_invalid = torch.randn(2, 40)
        with self.assertRaises(ValueError):
            self.encoder(x_invalid)

    def test_transceiver_over_channel(self):
        channel = AWGNChannel(default_snr_db=20.0)
        x = torch.randn(2, 10, self.input_dim)

        symbols = self.encoder(x)
        noisy_symbols = channel(symbols)
        reconstructed = self.decoder(noisy_symbols)

        loss = self.decoder.compute_loss(reconstructed, x)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
