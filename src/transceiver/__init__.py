"""Transceiver package for iTantra.

Contains neural joint source-channel coding components (encoder, decoder)
and wireless channel simulations (AWGN, Rayleigh fading).
"""

from src.transceiver.channel import (
    PowerNormalization,
    AWGNChannel,
    RayleighChannel,
    WirelessChannel,
)
from src.transceiver.encoder import NeuralEncoder
from src.transceiver.decoder import NeuralDecoder

__all__ = [
    "PowerNormalization",
    "AWGNChannel",
    "RayleighChannel",
    "WirelessChannel",
    "NeuralEncoder",
    "NeuralDecoder",
]
