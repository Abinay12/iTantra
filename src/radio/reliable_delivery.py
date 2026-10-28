"""Reliable Message Delivery and Link Layer Protocol for iTantra.

Provides:
  1. ACK-based message delivery confirmation.
  2. Automatic Repeat reQuest (ARQ) retransmission with configurable retry limit.
  3. Store-and-forward message queuing when the receiver is temporarily offline/unavailable.
  4. Message IDs (UUID or custom string/int) for duplicate message suppression.
  5. Clear delivery statuses: PENDING, DELIVERED, FAILED, and QUEUED.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import itertools
import struct
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch

from src.radio.packetizer import Packetizer, RadioPacket, PREAMBLE_SYNC, PROTOCOL_VERSION
from src.radio.sdr_interface import SimulatedSDR

_MSG_COUNTER = itertools.count(1)


class MessagePriority(IntEnum):
    """Emergency-aware message priority levels."""
    LOW = 10
    NORMAL = 20
    HIGH = 30
    EMERGENCY = 40

    @classmethod
    def from_value(cls, val: Any) -> "MessagePriority":
        """Parse priority from int, str ('EMERGENCY', 'HIGH', etc.), or MessagePriority."""
        if isinstance(val, cls):
            return val
        if isinstance(val, int):
            for member in cls:
                if member.value == val:
                    return member
            # Allow raw 1, 2, 3, 4 mappings as well
            mapping = {1: cls.LOW, 2: cls.NORMAL, 3: cls.HIGH, 4: cls.EMERGENCY}
            if val in mapping:
                return mapping[val]
            return cls.NORMAL
        if isinstance(val, str):
            clean = val.strip().upper()
            if hasattr(cls, clean):
                return getattr(cls, clean)
            try:
                int_val = int(clean)
                return cls.from_value(int_val)
            except ValueError:
                pass
        return cls.NORMAL


class DeliveryStatus(str, Enum):
    """Message delivery lifecycle states."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    QUEUED = "queued"
    EXPIRED = "expired"


# Packet Types for Link-Layer Protocol
PACKET_TYPE_DATA = 0x01
PACKET_TYPE_ACK = 0x06
PACKET_TYPE_NACK = 0x15


@dataclass
class Message:
    """Logical message container with reliable delivery metadata, priority, and TTL."""
    msg_id: str
    payload: Any  # Neural symbols (np.ndarray / torch.Tensor) or text
    language: str = "hi"
    status: DeliveryStatus = DeliveryStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    delivered_at: Optional[float] = None
    error: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: Optional[float] = None  # Time-To-Live in seconds; None means no expiration
    seq: int = field(default_factory=lambda: next(_MSG_COUNTER))
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if message has exceeded its Time-To-Live (TTL)."""
        if self.status in (DeliveryStatus.DELIVERED, DeliveryStatus.EXPIRED):
            return self.status == DeliveryStatus.EXPIRED
        if self.ttl is not None:
            if self.ttl <= 0:
                return True
            now = current_time if current_time is not None else time.time()
            return now >= (self.created_at + self.ttl)
        return False

    def mark_expired(self, reason: Optional[str] = None):
        """Mark message as expired due to TTL."""
        self.status = DeliveryStatus.EXPIRED
        self.error = reason or (
            f"Message expired: TTL of {self.ttl}s exceeded before delivery"
            if self.ttl is not None else "Message expired: TTL exceeded"
        )

    def mark_delivered(self):
        self.status = DeliveryStatus.DELIVERED
        self.delivered_at = time.time()
        self.error = None

    def mark_failed(self, error: str):
        self.status = DeliveryStatus.FAILED
        self.error = error

    def mark_queued(self, error: Optional[str] = None):
        self.status = DeliveryStatus.QUEUED
        self.error = error or "Receiver unavailable: queued for store-and-forward delivery"


def hash_msg_id_16(msg_id: Any) -> int:
    """Compute deterministic 16-bit hash of string or identifier message ID."""
    digest = hashlib.md5(str(msg_id).encode("utf-8")).digest()
    return struct.unpack(">H", digest[:2])[0]


class ReliablePacketizer(Packetizer):
    """Extended packetizer supporting ACK/NACK control frames and Message IDs."""

    def packetize_ack(self, msg_id: str, status_code: int = 0) -> bytes:
        """Create a binary framed ACK packet."""
        msg_hash = hash_msg_id_16(msg_id)
        # Header: Version (B), PacketType (B), MsgHash (H), StatusCode (H), 0 (H)
        header = struct.pack(">BBHHH", PROTOCOL_VERSION, PACKET_TYPE_ACK, msg_hash, status_code, 0)
        crc = self.compute_crc16(header)
        return PREAMBLE_SYNC + header + struct.pack(">H", crc)

    def packetize_nack(self, msg_id: str, reason_code: int = 1) -> bytes:
        """Create a binary framed NACK packet."""
        msg_hash = hash_msg_id_16(msg_id)
        header = struct.pack(">BBHHH", PROTOCOL_VERSION, PACKET_TYPE_NACK, msg_hash, reason_code, 0)
        crc = self.compute_crc16(header)
        return PREAMBLE_SYNC + header + struct.pack(">H", crc)

    def parse_control_frame(self, raw_bytes: bytes) -> Optional[Tuple[int, int, int, bool]]:
        """Parse control frame.

        Returns:
            Tuple of (packet_type, msg_hash, code, crc_valid) or None if not control frame.
        """
        if len(raw_bytes) < 4 + 8 + 2:
            return None
        sync_idx = raw_bytes.find(PREAMBLE_SYNC)
        if sync_idx == -1:
            return None

        body_start = sync_idx + 4
        if len(raw_bytes) < body_start + 10:
            return None
        body = raw_bytes[body_start : body_start + 8]
        (received_crc,) = struct.unpack(">H", raw_bytes[body_start + 8 : body_start + 10])
        computed_crc = self.compute_crc16(body)
        crc_valid = (received_crc == computed_crc)

        version, pkt_type, msg_hash, code, _ = struct.unpack(">BBHHH", body)
        if pkt_type in (PACKET_TYPE_ACK, PACKET_TYPE_NACK):
            return pkt_type, msg_hash, code, crc_valid
        return None


class ReliableReceiver:
    """Receiver node with duplicate detection and ACK generation."""

    def __init__(
        self,
        node_id: str = "rx_node_01",
        db_path: Optional[str] = None,
        store: Optional[Any] = None,
    ):
        self.node_id = node_id
        self.is_available = True
        self.packetizer = ReliablePacketizer(max_payload_symbols=128)
        self.db_path = db_path
        if store is not None:
            self.store = store
        elif db_path is not None:
            from src.radio.message_store import MessageStore
            self.store = MessageStore(db_path=db_path)
        else:
            self.store = None

        self.seen_message_ids: Set[str] = set()
        if self.store is not None:
            self.seen_message_ids.update(self.store.get_seen_message_ids(self.node_id))
        self.received_messages: List[Message] = []
        self.sdr = SimulatedSDR()

    def reset(self):
        """Reset received messages and seen message IDs."""
        self.seen_message_ids.clear()
        self.received_messages.clear()
        if self.store is not None:
            self.store.clear_received(self.node_id)

    def set_availability(self, available: bool):
        """Set receiver online/offline status."""
        self.is_available = available

    def close(self):
        """Close persistent SQLite storage connection if opened."""
        if self.store is not None:
            self.store.close()

    def process_incoming_stream(
        self,
        raw_stream: bytes,
        msg_id: str,
        language: str = "hi",
    ) -> Tuple[Optional[bytes], bool]:
        """Process incoming raw byte stream from radio channel.

        Returns:
            Tuple of (ack_packet_bytes, is_duplicate).
        """
        if not self.is_available:
            return None, False

        # Attempt depacketization and CRC verification
        try:
            symbols, rx_lang, all_crcs_valid = self.packetizer.depacketize_stream(raw_stream)
        except Exception:
            # Framing corruption
            nack = self.packetizer.packetize_nack(msg_id, reason_code=0xEE)
            return nack, False

        if not all_crcs_valid or len(symbols) == 0:
            # CRC corruption detected
            nack = self.packetizer.packetize_nack(msg_id, reason_code=0x01)
            return nack, False

        # Check for duplicate message delivery
        if msg_id in self.seen_message_ids:
            # Resend ACK so sender can clear state, but suppress duplicate delivery
            ack = self.packetizer.packetize_ack(msg_id)
            return ack, True

        # New valid message: record and mark delivered
        self.seen_message_ids.add(msg_id)
        if self.store is not None:
            self.store.record_received_message(
                msg_id=msg_id,
                node_id=self.node_id,
                payload=symbols,
                language=rx_lang or language,
            )
        delivered_msg = Message(
            msg_id=msg_id,
            payload=symbols,
            language=rx_lang or language,
            status=DeliveryStatus.DELIVERED,
            delivered_at=time.time(),
        )
        self.received_messages.append(delivered_msg)

        ack = self.packetizer.packetize_ack(msg_id)
        return ack, False


class ReliableSender:
    """Sender node with ARQ retransmission and persistent SQLite store-and-forward queue."""

    def __init__(
        self,
        node_id: str = "tx_node_01",
        default_max_retries: int = 3,
        default_ttl: Optional[float] = None,
        default_priority: Union[MessagePriority, str, int] = MessagePriority.NORMAL,
        packetizer: Optional[ReliablePacketizer] = None,
        db_path: Optional[str] = None,
        store: Optional[Any] = None,
        remove_on_delivered: bool = False,
    ):
        self.node_id = node_id
        self.default_max_retries = default_max_retries
        self.default_ttl = default_ttl
        self.default_priority = MessagePriority.from_value(default_priority)
        self.packetizer = packetizer or ReliablePacketizer(max_payload_symbols=128)
        self.db_path = db_path
        self.remove_on_delivered = remove_on_delivered
        if store is not None:
            self.store = store
        else:
            from src.radio.message_store import MessageStore
            self.store = MessageStore(db_path=db_path or ":memory:")
        self.queue: List[Message] = []
        self.history: Dict[str, Message] = {}
        self.sdr = SimulatedSDR()

        # Restore pending/queued messages from persistent SQLite storage on startup
        self._restore_from_store()

    @staticmethod
    def _msg_sort_key(m: Message) -> Tuple[int, float, int]:
        prio_val = int(m.priority.value if hasattr(m.priority, "value") else m.priority)
        return (-prio_val, m.created_at, getattr(m, "seq", 0))

    def _enqueue(self, msg: Message):
        """Insert message into queue maintaining priority (highest first) and FIFO order (earliest first)."""
        self.queue = [m for m in self.queue if m.msg_id != msg.msg_id]
        self.queue.append(msg)
        self.queue.sort(key=self._msg_sort_key)

    def _restore_from_store(self):
        """Restore pending and queued messages from persistent SQLite storage on startup in priority/FIFO order."""
        stored = self.store.get_pending_and_queued_messages()
        now = time.time()
        for msg in stored:
            # Check if message expired while application was terminated/offline
            if msg.is_expired(now):
                msg.mark_expired("TTL expired while application was offline")
                self.store.mark_expired(msg.msg_id, error=msg.error)
                self.history[msg.msg_id] = msg
                continue

            if msg.status == DeliveryStatus.PENDING:
                # Interrupted in-flight message from previous crashed/terminated session
                restore_err = "Restored from pending: queued for store-and-forward delivery"
                msg.mark_queued(error=restore_err)
                self.store.mark_queued(
                    msg.msg_id,
                    error=restore_err,
                )
            self._enqueue(msg)
            self.history[msg.msg_id] = msg

    def send_message(
        self,
        msg_id: str,
        payload: Union[np.ndarray, torch.Tensor],
        language: str = "hi",
        receiver: Optional[ReliableReceiver] = None,
        channel: Any = None,
        snr_db: float = 15.0,
        packet_loss_rate: float = 0.0,
        packet_corruption_rate: float = 0.0,
        max_retries: Optional[int] = None,
        priority: Optional[Union[MessagePriority, str, int]] = None,
        ttl: Optional[float] = None,
        created_at: Optional[float] = None,
    ) -> Message:
        """Send message reliably using ARQ with duplicate suppression, emergency prioritization, and TTL.

        Args:
            msg_id: Unique message identifier.
            payload: Channel symbols tensor or array to transmit.
            language: Language code ('te', 'hi', 'ta', 'bn', 'mr').
            receiver: Target ReliableReceiver instance.
            channel: Optional wireless channel simulator (AWGNChannel / RayleighChannel).
            snr_db: SNR in dB for channel simulation.
            packet_loss_rate: Simulated link packet loss probability [0.0, 1.0].
            packet_corruption_rate: Simulated packet/CRC corruption probability [0.0, 1.0].
            max_retries: Maximum retransmission attempts on failure.
            priority: Priority level (EMERGENCY, HIGH, NORMAL, LOW).
            ttl: Time-To-Live in seconds; expired messages are never transmitted.
            created_at: Original timestamp for queued or restored messages.

        Returns:
            Message object with updated delivery status, telemetry, and metadata.
        """
        start_time = time.time()
        packets_sent_total = 0
        packets_lost_total = 0
        crc_failures_total = 0
        sym_mse_last = 0.0

        def _finalize_msg(m: Message, retries_taken: int) -> Message:
            elapsed_sec = time.time() - start_time
            m.telemetry = {
                "latency_sec": elapsed_sec,
                "latency_ms": elapsed_sec * 1000.0,
                "packets_sent": packets_sent_total,
                "packets_lost": packets_lost_total,
                "crc_failures": crc_failures_total,
                "sym_mse": sym_mse_last,
                "attempts": retries_taken,
                "retry_count": m.retry_count,
            }
            return m

        retries_limit = self.default_max_retries if max_retries is None else max_retries
        msg_prio = self.default_priority if priority is None else MessagePriority.from_value(priority)
        msg_ttl = self.default_ttl if ttl is None else ttl

        # Check if already delivered or expired in persistent store to avoid duplicate reprocessing
        existing = self.store.get_message(msg_id)
        if existing is not None:
            if existing.status in (DeliveryStatus.DELIVERED, DeliveryStatus.EXPIRED):
                self.history[msg_id] = existing
                return _finalize_msg(existing, existing.retry_count)

        # Ensure symbols are detached numpy array
        if isinstance(payload, torch.Tensor):
            symbols = payload.detach().cpu().numpy().flatten().astype(np.float32)
        elif isinstance(payload, (str, bytes)):
            raw = payload.encode("utf-8") if isinstance(payload, str) else payload
            symbols = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        elif payload is None:
            symbols = np.array([], dtype=np.float32)
        else:
            symbols = np.asarray(payload, dtype=np.float32).flatten()

        msg_created_at = created_at if created_at is not None else (
            existing.created_at if existing is not None else time.time()
        )
        msg_seq = getattr(existing, "seq", None) if existing is not None else None

        msg_kwargs: Dict[str, Any] = {
            "msg_id": msg_id,
            "payload": symbols,
            "language": language,
            "status": DeliveryStatus.PENDING,
            "max_retries": retries_limit,
            "priority": msg_prio,
            "ttl": msg_ttl,
            "created_at": msg_created_at,
        }
        if msg_seq is not None:
            msg_kwargs["seq"] = msg_seq
        msg = Message(**msg_kwargs)
        self.history[msg_id] = msg

        # Stage 0: Check if message has already expired before initial transmission attempt
        if msg.is_expired():
            msg.mark_expired("TTL expired before transmission attempt")
            self.store.save_message(msg)
            self.queue = [m for m in self.queue if m.msg_id != msg_id]
            return _finalize_msg(msg, 0)

        # Stage 1: Check receiver availability for store-and-forward queuing
        if receiver is None or not receiver.is_available:
            msg.mark_queued()
            self._enqueue(msg)
            self.store.save_message(msg)
            return _finalize_msg(msg, 0)

        # Receiver is online; persist initial pending message record
        self.store.save_message(msg)

        # Stage 2: Packetize and transmit with ARQ retransmissions
        attempt = 0
        expected_hash = hash_msg_id_16(msg_id)

        while attempt <= retries_limit:
            # Check for TTL expiration between retransmission attempts
            if msg.is_expired():
                msg.mark_expired(f"TTL expired during retransmission attempts (attempt {attempt})")
                self.store.mark_expired(msg_id, error=msg.error)
                self.queue = [m for m in self.queue if m.msg_id != msg_id]
                return _finalize_msg(msg, attempt)

            msg.retry_count = attempt
            self.store.update_retry_count(msg_id, attempt)

            # Apply channel simulation to symbols
            if channel is not None:
                ch_device = getattr(channel, "device", None)
                if ch_device is None:
                    try:
                        ch_device = next(channel.parameters()).device
                    except (StopIteration, AttributeError):
                        ch_device = "cpu"
                tensor_sym = torch.from_numpy(symbols).to(ch_device)
                with torch.no_grad():
                    noisy_sym = channel(tensor_sym, snr_db=snr_db)
                tx_symbols = noisy_sym.cpu().numpy()
                try:
                    sym_mse_last = float(torch.nn.functional.mse_loss(noisy_sym, tensor_sym).item())
                except Exception:
                    sym_mse_last = 0.0
            else:
                tx_symbols = symbols
                sym_mse_last = 0.0

            # Packetize into radio frames
            packets = self.packetizer.packetize_stream(tx_symbols, language=language)
            packets_sent_total += len(packets)

            # Simulated SDR loopback transmission
            with self.sdr:
                for p in packets:
                    self.sdr.transmit(p)
                raw_bytes = bytearray()
                while self.sdr.has_data():
                    chunk = self.sdr.receive(1024)
                    if chunk:
                        raw_bytes.extend(chunk)

            # Simulated packet loss / channel erasure
            if packet_loss_rate > 0.0 and np.random.rand() < packet_loss_rate:
                # Packet lost in transit: receiver sees nothing
                packets_lost_total += len(packets)
                attempt += 1
                continue

            # Simulated packet bit corruption / CRC integrity failure
            if packet_corruption_rate > 0.0 and np.random.rand() < packet_corruption_rate:
                if len(raw_bytes) > 10:
                    corrupt_idx = int(np.random.randint(4, len(raw_bytes)))
                    raw_bytes[corrupt_idx] ^= 0xFF

            # Pass stream to receiver
            ack_response, is_duplicate = receiver.process_incoming_stream(
                bytes(raw_bytes), msg_id=msg_id, language=language
            )

            # Verify ACK response
            if ack_response is not None:
                ctrl = self.packetizer.parse_control_frame(ack_response)
                if ctrl is not None:
                    pkt_type, rx_hash, code, crc_valid = ctrl
                    if pkt_type == PACKET_TYPE_NACK:
                        crc_failures_total += len(packets)
                    elif crc_valid and pkt_type == PACKET_TYPE_ACK and rx_hash == expected_hash:
                        # Delivery confirmed!
                        msg.mark_delivered()
                        if self.remove_on_delivered:
                            self.store.delete_message(msg_id)
                        else:
                            self.store.mark_delivered(msg_id, msg.delivered_at)
                        self.queue = [m for m in self.queue if m.msg_id != msg_id]
                        return _finalize_msg(msg, attempt)
            else:
                # No ACK response received
                pass

            # If ACK failed or was NACK, retry
            attempt += 1

        # Exhausted maximum retries
        msg.mark_failed(f"Failed to deliver after {retries_limit} retries (no valid ACK received)")
        self.store.mark_failed(msg_id, msg.error, retry_count=msg.retry_count)
        self.queue = [m for m in self.queue if m.msg_id != msg_id]
        return _finalize_msg(msg, attempt)

    def flush_queue(
        self,
        receiver: ReliableReceiver,
        channel: Any = None,
        snr_db: float = 15.0,
        packet_loss_rate: float = 0.0,
        packet_corruption_rate: float = 0.0,
    ) -> List[Message]:
        """Flush store-and-forward queue to receiver in priority order, expiring TTL-exceeded messages."""
        if not receiver.is_available:
            return self.queue

        # Prioritize queue: EMERGENCY (40) -> HIGH (30) -> NORMAL (20) -> LOW (10), FIFO within same priority
        self.queue.sort(key=self._msg_sort_key)

        flushed = []
        pending_queue = list(self.queue)
        self.queue.clear()

        now = time.time()
        for msg in pending_queue:
            # Check for TTL expiration while awaiting receiver
            if msg.is_expired(now):
                msg.mark_expired("TTL expired while queued awaiting receiver")
                self.store.mark_expired(msg.msg_id, error=msg.error)
                self.history[msg.msg_id] = msg
                flushed.append(msg)
                continue

            delivered_msg = self.send_message(
                msg_id=msg.msg_id,
                payload=msg.payload,
                language=msg.language,
                receiver=receiver,
                channel=channel,
                snr_db=snr_db,
                packet_loss_rate=packet_loss_rate,
                packet_corruption_rate=packet_corruption_rate,
                max_retries=msg.max_retries,
                priority=msg.priority,
                ttl=msg.ttl,
                created_at=msg.created_at,
            )
            flushed.append(delivered_msg)

        return flushed

    def expire_stale_messages(self, current_time: Optional[float] = None) -> List[Message]:
        """Scan queued messages, mark expired any whose TTL has elapsed, and update SQLite."""
        now = current_time if current_time is not None else time.time()
        expired_msgs = []
        remaining_queue = []
        for msg in self.queue:
            if msg.is_expired(now):
                msg.mark_expired("TTL expired while queued")
                self.store.mark_expired(msg.msg_id, error=msg.error)
                self.history[msg.msg_id] = msg
                expired_msgs.append(msg)
            else:
                remaining_queue.append(msg)
        self.queue = remaining_queue
        return expired_msgs

    def clear_queue(self):
        """Clear store-and-forward in-memory queue and SQLite queued records."""
        self.queue.clear()
        self.store.clear_queued()

    def clear_all_store(self):
        """Clear all messages from in-memory and SQLite storage."""
        self.queue.clear()
        self.history.clear()
        self.store.clear_all()

    def get_stored_message(self, msg_id: str) -> Optional[Message]:
        """Retrieve message record directly from persistent SQLite storage."""
        return self.store.get_message(msg_id)

    def get_all_stored_messages(self) -> List[Message]:
        """Retrieve all message records directly from persistent SQLite storage."""
        return self.store.get_all_messages()

    def remove_delivered(self, msg_id: Optional[str] = None) -> int:
        """Remove delivered message(s) from persistent SQLite storage."""
        if msg_id is not None:
            deleted = self.store.delete_message(msg_id)
            return 1 if deleted else 0
        return self.store.delete_delivered()

    def prune_delivered(self) -> int:
        """Prune all delivered messages from SQLite storage."""
        return self.store.delete_delivered()

    def prune_expired(self) -> int:
        """Prune all expired messages from SQLite storage."""
        return self.store.delete_expired()

    def delete_stored_message(self, msg_id: str) -> bool:
        """Delete specific message from SQLite store."""
        return self.store.delete_message(msg_id)

    @property
    def queued_count(self) -> int:
        """Number of messages currently queued."""
        return len(self.queue)

    def close(self):
        """Close persistent SQLite storage connection if opened."""
        if hasattr(self, "store") and self.store is not None:
            self.store.close()
