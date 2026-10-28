"""Neural Transceiver Decoder for iTantra.

Provides a lightweight, CPU-efficient neural decoder that reconstructs source
speech features from channel symbols corrupted by noise and fading.
"""

from typing import Optional
import torch
import torch.nn as nn


class NeuralDecoder(nn.Module):
    """Lightweight neural decoder for Joint Source-Channel Coding (JSCC).

    Recovers reconstructed speech features from received channel symbols.

    Architecture:
        y -> Linear(latent_dim, hidden_dim) -> LayerNorm -> GELU
          -> Linear(hidden_dim, hidden_dim) -> LayerNorm -> GELU
          -> Linear(hidden_dim, output_dim)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 80,
    ):
        """Initialize neural decoder.

        Args:
            latent_dim: Dimension of received channel symbols.
            hidden_dim: Dimension of intermediate hidden representations.
            output_dim: Dimension of reconstructed speech features.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Decode received channel symbols into reconstructed features.

        Args:
            y: Received channel symbols of shape (..., latent_dim).

        Returns:
            Reconstructed feature tensor of shape (..., output_dim).
        """
        if y.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent symbol dimension {self.latent_dim}, got {y.shape[-1]}."
            )
        return self.net(y)

    def compute_loss(
        self, reconstructed: torch.Tensor, original: torch.Tensor
    ) -> torch.Tensor:
        """Compute Mean Squared Error (MSE) reconstruction loss."""
        return nn.functional.mse_loss(reconstructed, original)
