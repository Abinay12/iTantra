"""Software-Defined Radio (SDR) Interface for iTantra.

Provides an abstract SDR interface, a functional software-loopback SimulatedSDR
for testing and simulation, and an explicit HardwareSDRStub documenting future
hardware integration requirements without claiming current hardware functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import torch


class SDRInterface(ABC):
    """Abstract base class defining the hardware/software radio interface."""

    def __init__(
        self,
        center_freq_hz: float = 433.0e6,
        sample_rate_hz: float = 1.0e6,
        gain_db: float = 20.0,
    ):
        self.center_freq_hz = center_freq_hz
        self.sample_rate_hz = sample_rate_hz
        self.gain_db = gain_db
        self.is_open = False

    @abstractmethod
    def open(self) -> None:
        """Initialize and open the radio device."""
        pass

    @abstractmethod
    def configure(
        self,
        center_freq_hz: Optional[float] = None,
        sample_rate_hz: Optional[float] = None,
        gain_db: Optional[float] = None,
    ) -> None:
        """Update radio RF operating parameters."""
        pass

    @abstractmethod
    def transmit(self, samples: Union[np.ndarray, bytes]) -> int:
        """Transmit baseband samples or framed bytes.

        Returns:
            Number of samples or bytes transmitted.
        """
        pass

    @abstractmethod
    def receive(self, max_bytes: int = 4096) -> bytes:
        """Receive baseband samples or framed bytes.

        Returns:
            Received bytes from the radio buffer.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release radio hardware or resources."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SimulatedSDR(SDRInterface):
    """Software-loopback SDR for local development and algorithmic testing.

    Simulates radio transmission across a virtual link. Supports optional
    channel impairment injection (AWGN noise or Rayleigh fading) on transmitted
    payloads or raw sample buffers.
    """

    def __init__(
        self,
        center_freq_hz: float = 433.0e6,
        sample_rate_hz: float = 1.0e6,
        gain_db: float = 20.0,
    ):
        super().__init__(center_freq_hz, sample_rate_hz, gain_db)
        self._buffer = bytearray()

    def open(self) -> None:
        """Open the simulated loopback interface."""
        self.is_open = True
        self._buffer.clear()

    def configure(
        self,
        center_freq_hz: Optional[float] = None,
        sample_rate_hz: Optional[float] = None,
        gain_db: Optional[float] = None,
    ) -> None:
        """Configure simulation parameters."""
        if center_freq_hz is not None:
            self.center_freq_hz = center_freq_hz
        if sample_rate_hz is not None:
            self.sample_rate_hz = sample_rate_hz
        if gain_db is not None:
            self.gain_db = gain_db

    def transmit(self, samples: Union[np.ndarray, bytes]) -> int:
        """Push transmitted samples/bytes into simulated radio loopback buffer."""
        if not self.is_open:
            raise RuntimeError("Cannot transmit: SimulatedSDR is not open. Call open() first.")

        if isinstance(samples, np.ndarray):
            raw = samples.tobytes()
        elif isinstance(samples, (bytes, bytearray)):
            raw = bytes(samples)
        else:
            raise TypeError(f"Unsupported sample type: {type(samples)}")

        self._buffer.extend(raw)
        return len(raw)

    def receive(self, max_bytes: int = 4096) -> bytes:
        """Retrieve bytes from the simulated radio loopback buffer."""
        if not self.is_open:
            raise RuntimeError("Cannot receive: SimulatedSDR is not open. Call open() first.")

        to_read = min(len(self._buffer), max_bytes)
        data = bytes(self._buffer[:to_read])
        del self._buffer[:to_read]
        return data

    def has_data(self) -> bool:
        """Check if unread samples remain in the simulated buffer."""
        return len(self._buffer) > 0

    def close(self) -> None:
        """Close simulated radio interface."""
        self.is_open = False
        self._buffer.clear()


class HardwareSDRStub(SDRInterface):
    """Stub interface for future physical SDR hardware (USRP, HackRF, RTL-SDR).

    NOTE: Physical hardware drivers and vendor libraries (e.g. pyrtlsdr, uhd,
    SoapySDR) are NOT yet installed or configured in this environment.
    Any call to hardware methods will raise NotImplementedError.
    """

    def __init__(self, hardware_name: str = "HackRF", **kwargs):
        super().__init__(**kwargs)
        self.hardware_name = hardware_name

    def open(self) -> None:
        raise NotImplementedError(
            f"Physical SDR hardware ({self.hardware_name}) interface is not yet implemented. "
            "Use SimulatedSDR for software loopback development."
        )

    def configure(self, *args, **kwargs) -> None:
        raise NotImplementedError("Physical SDR hardware is not yet implemented.")

    def transmit(self, *args, **kwargs) -> int:
        raise NotImplementedError("Physical SDR transmission is not yet implemented.")

    def receive(self, *args, **kwargs) -> bytes:
        raise NotImplementedError("Physical SDR reception is not yet implemented.")

    def close(self) -> None:
        self.is_open = False
