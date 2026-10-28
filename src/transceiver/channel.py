"""Wireless communication channel simulations for iTantra.

Provides PyTorch-based implementations of AWGN (Additive White Gaussian Noise)
and Rayleigh flat fading channels with average power normalization.
Compatible with CPU-based and GPU-based execution.
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn


class PowerNormalization(nn.Module):
    """Enforces average transmission power constraint E[|x|^2] <= target_power.

    For a batch of channel symbols x, computes the average power across channel
    dimensions and normalizes the signal.
    """

    def __init__(self, target_power: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.target_power = target_power
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize signal tensor x to have average power target_power.

        Args:
            x: Input tensor of shape (..., channel_dim).

        Returns:
            Normalized tensor with matching shape.
        """
        # Average power per element across the last dimension
        current_power = torch.mean(x ** 2, dim=-1, keepdim=True)
        scale = torch.sqrt(self.target_power / (current_power + self.eps))
        return x * scale


class AWGNChannel(nn.Module):
    """Additive White Gaussian Noise (AWGN) channel simulator.

    Given a transmitted signal x and Signal-to-Noise Ratio (SNR) in dB,
    injects Gaussian noise:
        y = x + n, where n ~ N(0, sigma^2)
        sigma^2 = P_signal / (10^(SNR_dB / 10))
    """

    def __init__(self, default_snr_db: float = 10.0):
        super().__init__()
        self.default_snr_db = default_snr_db

    def forward(
        self, x: torch.Tensor, snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """Transmit signal x through AWGN channel.

        Args:
            x: Input signal tensor.
            snr_db: Optional SNR in decibels. If None, uses default_snr_db.

        Returns:
            Noisy received signal tensor y.
        """
        snr = self.default_snr_db if snr_db is None else snr_db
        snr_linear = 10.0 ** (snr / 10.0)

        # Measure instantaneous signal power for accurate noise addition
        signal_power = torch.mean(x ** 2)
        noise_variance = signal_power / snr_linear
        noise_std = torch.sqrt(noise_variance + 1e-12)

        noise = torch.randn_like(x) * noise_std
        return x + noise


class RayleighChannel(nn.Module):
    """Rayleigh flat fading channel simulator with AWGN noise.

    Simulates multipath fading where channel tap magnitude |h| follows a
    Rayleigh distribution. Can operate on real-valued signals or paired I/Q
    representations.

    For I/Q pairs (real/imaginary):
        y_I = h_r * x_I - h_i * x_Q + n_I
        y_Q = h_i * x_I + h_r * x_Q + n_Q
    where h_r, h_i ~ N(0, 1/2), ensuring E[|h|^2] = 1.
    """

    def __init__(
        self,
        default_snr_db: float = 10.0,
        coherence_block: int = 64,
        equalize: bool = True,
    ):
        """Initialize Rayleigh channel.

        Args:
            default_snr_db: Default SNR in dB.
            coherence_block: Number of channel uses over which h stays constant.
            equalize: If True, applies Zero-Forcing (ZF) equalization using CSI.
        """
        super().__init__()
        self.default_snr_db = default_snr_db
        self.coherence_block = coherence_block
        self.equalize = equalize

    def forward(
        self,
        x: torch.Tensor,
        snr_db: Optional[float] = None,
        return_csi: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Transmit signal x through Rayleigh fading channel.

        Args:
            x: Input signal tensor of shape (batch, ..., dim).
               If dim is even, treats adjacent pairs as I/Q symbols.
            snr_db: Optional SNR in decibels.
            return_csi: If True, returns (received_signal, channel_state_info).

        Returns:
            Received signal (or equalized signal), optionally with CSI tensor.
        """
        snr = self.default_snr_db if snr_db is None else snr_db
        snr_linear = 10.0 ** (snr / 10.0)

        orig_shape = x.shape
        # Flatten spatial/sequence dimensions for block fading
        dim = orig_shape[-1]
        x_flat = x.view(-1, dim)
        num_vectors = x_flat.shape[0]

        # Generate Rayleigh channel coefficients: h = h_r + j*h_i with E[|h|^2] = 1
        # Each coherence block shares the same fading coefficient
        h_r = torch.randn(num_vectors, 1, device=x.device) * (1.0 / math.sqrt(2.0))
        h_i = torch.randn(num_vectors, 1, device=x.device) * (1.0 / math.sqrt(2.0))

        if dim % 2 == 0:
            # Pairwise I/Q treatment
            x_i = x_flat[:, 0::2]
            x_q = x_flat[:, 1::2]

            # Fading: complex multiplication (h_r + j*h_i) * (x_i + j*x_q)
            faded_i = h_r * x_i - h_i * x_q
            faded_q = h_i * x_i + h_r * x_q

            # Interleave back to original layout
            faded = torch.empty_like(x_flat)
            faded[:, 0::2] = faded_i
            faded[:, 1::2] = faded_q
        else:
            # Scalar Rayleigh envelope |h|
            h_mag = torch.sqrt(h_r ** 2 + h_i ** 2 + 1e-12)
            faded = x_flat * h_mag

        # Add Gaussian noise
        signal_power = torch.mean(faded ** 2)
        noise_variance = signal_power / snr_linear
        noise_std = torch.sqrt(noise_variance + 1e-12)
        noise = torch.randn_like(faded) * noise_std
        y_flat = faded + noise

        # Equalization (Zero-Forcing with small regularization to avoid inversion blowing up)
        if self.equalize:
            h_power = h_r ** 2 + h_i ** 2 + 1e-6
            if dim % 2 == 0:
                y_i = y_flat[:, 0::2]
                y_q = y_flat[:, 1::2]
                # ZF: y / h = y * h* / |h|^2
                eq_i = (y_i * h_r + y_q * h_i) / h_power
                eq_q = (y_q * h_r - y_i * h_i) / h_power
                y_flat = torch.empty_like(y_flat)
                y_flat[:, 0::2] = eq_i
                y_flat[:, 1::2] = eq_q
            else:
                h_mag = torch.sqrt(h_power)
                y_flat = y_flat / h_mag

        y = y_flat.view(orig_shape)
        h = torch.cat([h_r, h_i], dim=-1)

        if return_csi:
            return y, h
        return y


class WirelessChannel(nn.Module):
    """Unified wireless channel simulation interface.

    Allows runtime switching between 'awgn' and 'rayleigh' models.
    """

    def __init__(self, model_type: str = "awgn", snr_db: float = 10.0):
        super().__init__()
        self.set_model(model_type)
        self.awgn = AWGNChannel(default_snr_db=snr_db)
        self.rayleigh = RayleighChannel(default_snr_db=snr_db)

    def set_model(self, model_type: str):
        """Set active channel model ('awgn' or 'rayleigh')."""
        if model_type.lower() not in ("awgn", "rayleigh"):
            raise ValueError(f"Unsupported channel model: {model_type}. Choose 'awgn' or 'rayleigh'.")
        self.model_type = model_type.lower()

    def forward(
        self, x: torch.Tensor, snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """Transmit signal through selected channel model."""
        if self.model_type == "awgn":
            return self.awgn(x, snr_db=snr_db)
        elif self.model_type == "rayleigh":
            return self.rayleigh(x, snr_db=snr_db)
        else:
            raise ValueError(f"Unknown channel model: {self.model_type}")
