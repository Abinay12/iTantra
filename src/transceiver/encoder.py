"""Neural Transceiver Encoder for iTantra.

Provides a lightweight, CPU-efficient neural encoder that maps source speech
features (e.g., mel filterbank frames or semantic embeddings) into channel
symbols under an average transmit power constraint.
"""

from typing import Optional
import torch
import torch.nn as nn
from src.transceiver.channel import PowerNormalization


class NeuralEncoder(nn.Module):
    """Lightweight neural encoder for Joint Source-Channel Coding (JSCC).

    Maps high-dimensional speech features into compact channel symbols suitable
    for physical or simulated wireless transmission.

    Architecture:
        x -> Linear(input_dim, hidden_dim) -> LayerNorm -> GELU
          -> Linear(hidden_dim, hidden_dim) -> LayerNorm -> GELU
          -> Linear(hidden_dim, latent_dim) -> PowerNormalization
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        normalize_power: bool = True,
        target_power: float = 1.0,
    ):
        """Initialize neural encoder.

        Args:
            input_dim: Dimension of source features (e.g., mel bins).
            hidden_dim: Dimension of intermediate hidden representations.
            latent_dim: Number of channel symbols produced per source vector.
            normalize_power: If True, applies power normalization.
            target_power: Maximum average symbol power constraint (default 1.0).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.normalize_power = normalize_power

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.power_norm = PowerNormalization(target_power=target_power) if normalize_power else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode source features into channel symbols.

        Args:
            x: Source tensor of shape (batch_size, sequence_length, input_dim)
               or (batch_size, input_dim).

        Returns:
            Channel symbols tensor with last dimension equal to latent_dim.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}, got {x.shape[-1]}."
            )
        latent = self.net(x)
        return self.power_norm(latent)
