"""Radio packet framing and serialization for iTantra.

Provides packetization of neural channel symbols with preambles, headers,
language tags, sequence numbers, and CRC-16 integrity verification.
Decoupled from physical radio hardware and channel simulations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import struct
import numpy as np
import torch

# Standard preamble sync word (4 bytes)
PREAMBLE_SYNC = b"\xAA\x55\xAA\x55"
PROTOCOL_VERSION = 1

# Language ID mapping for packet headers (1-byte unsigned char field)
LANG_TO_ID = {
    "te": 1,
    "hi": 2,
    "ta": 3,
    "bn": 4,
    "mr": 5,
    "kn": 6,
    "ml": 7,
    "gu": 8,
}
ID_TO_LANG = {v: k for k, v in LANG_TO_ID.items()}


@dataclass
class RadioPacket:
    """Decoded radio packet structure."""
    version: int
    language: str
    packet_id: int
    total_packets: int
    payload_symbols: np.ndarray
    crc_valid: bool


class Packetizer:
    """Packs and unpacks neural channel symbol streams into radio frames.

    Packet Binary Layout:
      [0:4]   Preamble Sync Word (0xAA55AA55)
      [4:5]   Version (1 byte)
      [5:6]   Language ID (1 byte: 1=te, 2=hi, 3=ta, 4=bn, 5=mr, 6=kn, 7=ml, 8=gu)
      [6:8]   Packet ID (2 bytes, unsigned short, big-endian)
      [8:10]  Total Packets (2 bytes, unsigned short, big-endian)
      [10:12] Symbol Count (2 bytes, unsigned short, big-endian)
      [12:12+4*N] Payload (N float32 channel symbols)
      [-2:]   CRC-16 Checksum (2 bytes, big-endian)
    """

    def __init__(self, max_payload_symbols: int = 256):
        """Initialize packetizer.

        Args:
            max_payload_symbols: Maximum number of float32 symbols per packet.
        """
        self.max_payload_symbols = max_payload_symbols

    @staticmethod
    def compute_crc16(data: bytes) -> int:
        """Compute CRC-16-CCITT checksum over arbitrary bytes."""
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    def packetize(
        self,
        symbols: Union[np.ndarray, torch.Tensor],
        language: str = "hi",
        packet_id: int = 0,
        total_packets: int = 1,
    ) -> bytes:
        """Convert channel symbols into a framed radio byte packet.

        Args:
            symbols: 1D array or tensor of float32 symbols.
            language: Language code ('te', 'hi', 'ta', 'bn', 'mr').
            packet_id: Index of this packet in the message sequence.
            total_packets: Total number of packets for the message.

        Returns:
            Framed bytes ready for transmission.
        """
        if isinstance(symbols, torch.Tensor):
            arr = symbols.detach().cpu().numpy().flatten().astype(np.float32)
        else:
            arr = np.asarray(symbols, dtype=np.float32).flatten()

        lang_id = LANG_TO_ID.get(language.lower(), 0)
        num_symbols = len(arr)

        # Header: Version (B), LangID (B), PacketID (H), TotalPackets (H), SymbolCount (H)
        header = struct.pack(
            ">BBHHH",
            PROTOCOL_VERSION,
            lang_id,
            packet_id,
            total_packets,
            num_symbols,
        )

        payload_bytes = arr.tobytes()
        body = header + payload_bytes
        crc = self.compute_crc16(body)
        crc_bytes = struct.pack(">H", crc)

        return PREAMBLE_SYNC + body + crc_bytes

    def depacketize(self, raw_bytes: bytes) -> RadioPacket:
        """Parse raw radio bytes into a RadioPacket with integrity verification.

        Args:
            raw_bytes: Byte buffer containing at least one framed packet.

        Returns:
            RadioPacket with payload symbols and crc_valid status.
        """
        if len(raw_bytes) < 4 + 8 + 2:
            raise ValueError(f"Packet too short ({len(raw_bytes)} bytes).")

        sync_idx = raw_bytes.find(PREAMBLE_SYNC)
        if sync_idx == -1:
            raise ValueError("Preamble synchronization word not found in byte stream.")

        packet_data = raw_bytes[sync_idx:]
        body_start = 4
        # Minimum body size is header (8 bytes) + CRC (2 bytes) = 10 bytes
        if len(packet_data) < body_start + 10:
            raise ValueError("Incomplete packet following sync word.")

        version, lang_id, packet_id, total_packets, num_symbols = struct.unpack(
            ">BBHHH", packet_data[body_start : body_start + 8]
        )

        expected_payload_bytes = num_symbols * 4
        expected_total_len = body_start + 8 + expected_payload_bytes + 2

        if len(packet_data) < expected_total_len:
            raise ValueError(
                f"Packet truncated: expected {expected_total_len} bytes, got {len(packet_data)}."
            )

        payload_start = body_start + 8
        payload_end = payload_start + expected_payload_bytes
        payload_raw = packet_data[payload_start:payload_end]

        body = packet_data[body_start:payload_end]
        (received_crc,) = struct.unpack(">H", packet_data[payload_end : payload_end + 2])
        computed_crc = self.compute_crc16(body)

        crc_valid = (received_crc == computed_crc)
        symbols = np.frombuffer(payload_raw, dtype=np.float32).copy()
        lang_str = ID_TO_LANG.get(lang_id, "unknown")

        return RadioPacket(
            version=version,
            language=lang_str,
            packet_id=packet_id,
            total_packets=total_packets,
            payload_symbols=symbols,
            crc_valid=crc_valid,
        )

    def packetize_stream(
        self,
        symbols: Union[np.ndarray, torch.Tensor],
        language: str = "hi",
    ) -> List[bytes]:
        """Split a long symbol stream into multiple MTU-compliant packets."""
        if isinstance(symbols, torch.Tensor):
            arr = symbols.detach().cpu().numpy().flatten().astype(np.float32)
        else:
            arr = np.asarray(symbols, dtype=np.float32).flatten()

        total_symbols = len(arr)
        chunk_size = self.max_payload_symbols
        num_packets = max(1, int(np.ceil(total_symbols / chunk_size)))

        packets = []
        for i in range(num_packets):
            chunk = arr[i * chunk_size : (i + 1) * chunk_size]
            pkt = self.packetize(
                chunk, language=language, packet_id=i, total_packets=num_packets
            )
            packets.append(pkt)
        return packets

    def depacketize_stream(
        self, packets: Union[List[bytes], bytes, bytearray]
    ) -> Tuple[np.ndarray, str, bool]:
        """Reassemble a sequence of packets or contiguous byte stream into a single symbol array.

        Returns:
            Tuple of (reassembled_symbols, language_code, all_crcs_valid).
        """
        if not packets:
            return np.array([], dtype=np.float32), "hi", False

        if isinstance(packets, (bytes, bytearray)):
            stream = bytes(packets)
            packet_list = []
            idx = 0
            while idx < len(stream):
                sync_idx = stream.find(PREAMBLE_SYNC, idx)
                if sync_idx == -1:
                    break
                if len(stream) < sync_idx + 14:  # 4 sync + 8 header + 2 crc
                    break
                num_symbols = struct.unpack(">H", stream[sync_idx + 10 : sync_idx + 12])[0]
                pkt_len = 4 + 8 + num_symbols * 4 + 2
                if len(stream) < sync_idx + pkt_len:
                    break
                packet_list.append(stream[sync_idx : sync_idx + pkt_len])
                idx = sync_idx + pkt_len
            packets = packet_list

        if not packets:
            return np.array([], dtype=np.float32), "hi", False

        parsed = [self.depacketize(p) for p in packets]
        # Sort by packet_id
        parsed.sort(key=lambda x: x.packet_id)

        all_valid = all(p.crc_valid for p in parsed)
        language = parsed[0].language if parsed else "hi"

        symbols_list = [p.payload_symbols for p in parsed]
        reassembled = np.concatenate(symbols_list) if symbols_list else np.array([], dtype=np.float32)

        return reassembled, language, all_valid
