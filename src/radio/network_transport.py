"""TCP-based Real Network Transport for iTantra.

Enables real-world laptop-to-laptop neural speech and radio packet
transmission over local Wi-Fi networks using Python's built-in socket library.

Reuses existing radio packet framing (PREAMBLE_SYNC, headers, CRC-16),
SQLite message persistence, ACK/retry handling, duplicate suppression,
emergency priority queuing, and TTL expiration.

Maintains clear separation between real Wi-Fi network transport (unimpaired TCP)
and simulated RF channel models (AWGN / Rayleigh), which can be optionally enabled
for hybrid experimentation.
"""

from dataclasses import dataclass, field
import json
import os
import select
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.radio.packetizer import (
    PREAMBLE_SYNC,
    PROTOCOL_VERSION,
    Packetizer,
)
from src.radio.reliable_delivery import (
    DeliveryStatus,
    Message,
    MessagePriority,
    PACKET_TYPE_ACK,
    PACKET_TYPE_NACK,
    ReliablePacketizer,
    ReliableReceiver,
    ReliableSender,
    hash_msg_id_16,
)
from src.radio.message_store import MessageStore


# Protocol Constants for TCP Stream Framing
NET_MAGIC = b"ITNT"  # 4 bytes magic word
NET_VERSION = 1       # Network transport protocol version

# Message Types
MSG_TYPE_DATA = 1       # Neural radio stream message frame
MSG_TYPE_ACK = 2        # Acknowledgement control frame
MSG_TYPE_NACK = 3       # Negative acknowledgement control frame
MSG_TYPE_HEARTBEAT = 4  # Keepalive ping frame


def get_local_ip_addresses() -> List[str]:
    """Discover local IP addresses, prioritizing active Wi-Fi / LAN IPv4 interfaces."""
    ip_list = []
    # Method 1: UDP routing probe (does not send real data over the internet)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        s.close()
        if primary_ip and primary_ip not in ip_list:
            ip_list.append(primary_ip)
    except Exception:
        pass

    # Method 2: Hostname resolution fallback
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ip_list and not ip.startswith("127."):
                ip_list.append(ip)
    except Exception:
        pass

    if not ip_list:
        ip_list.append("127.0.0.1")
    return ip_list


def send_net_frame(
    sock: socket.socket,
    msg_type: int,
    metadata: Dict[str, Any],
    payload: bytes = b"",
) -> None:
    """Send a length-prefixed framed message over a TCP socket.

    Binary Frame Layout:
      [0:4]   Magic Word (b"ITNT")
      [4:5]   Net Version (1 byte)
      [5:6]   Msg Type (1 byte: DATA=1, ACK=2, NACK=3, HEARTBEAT=4)
      [6:8]   Metadata JSON Length (2 bytes, unsigned short, big-endian)
      [8:12]  Payload Length (4 bytes, unsigned int, big-endian)
      [12:12+M] Metadata JSON UTF-8 bytes
      [12+M:12+M+N] Payload binary bytes (e.g. radio packets or control frames)
    """
    meta_bytes = json.dumps(metadata).encode("utf-8")
    meta_len = len(meta_bytes)
    payload_len = len(payload)

    header = struct.pack(
        ">4sBBHI",
        NET_MAGIC,
        NET_VERSION,
        msg_type,
        meta_len,
        payload_len,
    )
    sock.sendall(header + meta_bytes + payload)


def recv_exact(sock: socket.socket, num_bytes: int, timeout: Optional[float] = None) -> bytes:
    """Receive exactly num_bytes from a socket stream, handling TCP fragmentation."""
    if timeout is not None:
        sock.settimeout(timeout)

    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = sock.recv(num_bytes - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Connection closed by peer before {num_bytes} bytes could be read (got {len(buf)} bytes)."
            )
        buf.extend(chunk)
    return bytes(buf)


def recv_net_frame(
    sock: socket.socket,
    timeout: Optional[float] = None,
) -> Tuple[int, Dict[str, Any], bytes]:
    """Receive a complete length-prefixed frame from a TCP socket.

    Returns:
        Tuple of (msg_type, metadata_dict, payload_bytes).
    """
    # Header is 12 bytes
    header_bytes = recv_exact(sock, 12, timeout=timeout)
    magic, version, msg_type, meta_len, payload_len = struct.unpack(">4sBBHI", header_bytes)

    if magic != NET_MAGIC:
        raise ValueError(f"Invalid network frame magic: {magic!r} (expected {NET_MAGIC!r})")

    meta_raw = recv_exact(sock, meta_len, timeout=timeout)
    metadata = json.loads(meta_raw.decode("utf-8")) if meta_len > 0 else {}

    payload = recv_exact(sock, payload_len, timeout=timeout) if payload_len > 0 else b""
    return msg_type, metadata, payload


@dataclass
class NetworkDeliveryResult:
    """Outcome and telemetry of a network message transmission."""
    msg_id: str
    status: DeliveryStatus
    is_delivered: bool
    is_expired: bool
    is_queued: bool
    is_failed: bool
    retry_count: int
    max_retries: int
    latency_ms: float
    packets_sent: int
    bytes_sent: int
    priority: str
    error: Optional[str] = None
    server_address: Optional[Tuple[str, int]] = None
    is_duplicate: bool = False


class TcpRadioReceiverServer:
    """TCP-based Radio Receiver Server for iTantra.

    Listens on a local network interface / Wi-Fi IP and receives framed radio packets
    sent from another laptop. Verifies preamble, header, and CRC-16 integrity,
    suppresses duplicate messages via persistent SQLite storage, enforces TTL expiration,
    and returns binary ACK/NACK control frames over the TCP connection.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        node_id: str = "rx_laptop",
        db_path: Optional[str] = "data/receiver_messages.db",
        receiver: Optional[ReliableReceiver] = None,
        on_message_delivered: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_duplicate_received: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize TCP Radio Receiver Server.

        Args:
            host: Bind host IP ('0.0.0.0' binds to all local interfaces, including Wi-Fi).
            port: Port to listen on (default: 8765).
            node_id: Receiver station identifier.
            db_path: Path to persistent SQLite store for received messages.
            receiver: Optional existing ReliableReceiver instance.
            on_message_delivered: Optional callback invoked when a valid, new message is delivered.
            on_duplicate_received: Optional callback invoked when a duplicate message is received and suppressed.
        """
        self.host = host
        self.port = port
        self.node_id = node_id
        self.db_path = db_path
        self.on_message_delivered = on_message_delivered
        self.on_duplicate_received = on_duplicate_received

        if receiver is not None:
            self.receiver = receiver
        else:
            self.receiver = ReliableReceiver(node_id=node_id, db_path=db_path)

        self.packetizer = ReliablePacketizer(max_payload_symbols=128)
        self._server_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_running = False
        self.bound_port = port

    def start(self, daemon: bool = True):
        """Start receiver server in a background thread."""
        if self.is_running:
            return

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self.bound_port = self._server_sock.getsockname()[1]
        self._server_sock.listen(16)
        self._server_sock.settimeout(0.5)

        self._stop_event.clear()
        self.is_running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=daemon, name="TcpRadioReceiver")
        self._thread.start()

    def _listen_loop(self):
        """Continuous connection accept loop."""
        while not self._stop_event.is_set():
            try:
                client_sock, client_addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except (OSError, socket.error):
                break

            try:
                self._handle_client(client_sock, client_addr)
            except Exception as e:
                pass
            finally:
                try:
                    client_sock.close()
                except Exception:
                    pass

    def _handle_client(self, client_sock: socket.socket, client_addr: Tuple[str, int]):
        """Handle incoming transmission from a sender laptop."""
        client_sock.settimeout(10.0)
        try:
            msg_type, metadata, payload = recv_net_frame(client_sock, timeout=10.0)
        except Exception as e:
            return

        if msg_type == MSG_TYPE_HEARTBEAT:
            # Simple keepalive ping
            send_net_frame(client_sock, MSG_TYPE_ACK, {"status": "ok", "role": "receiver"})
            return

        if msg_type != MSG_TYPE_DATA:
            return

        msg_id = metadata.get("msg_id", "UNKNOWN_MSG")
        language = metadata.get("language", "hi")
        priority_val = metadata.get("priority", "normal")
        ttl = metadata.get("ttl")
        created_at = metadata.get("created_at")
        text = metadata.get("text", "")

        # 1. TTL Expiration Check on Arrival
        if ttl is not None:
            c_time = created_at if created_at is not None else time.time()
            if ttl <= 0 or (time.time() >= c_time + ttl):
                # Message expired before or during transit: generate NACK with expiration reason
                nack_packet = self.packetizer.packetize_nack(msg_id, reason_code=0xEE)
                send_net_frame(
                    client_sock,
                    MSG_TYPE_NACK,
                    {
                        "msg_id": msg_id,
                        "status": DeliveryStatus.EXPIRED.value,
                        "error": f"Message expired: TTL of {ttl}s exceeded before receiver processing",
                    },
                    payload=nack_packet,
                )
                return

        # 2. Process incoming radio stream through ReliableReceiver
        # Reuses exact packet depacketization, CRC-16 checks, and duplicate detection
        ack_packet, is_duplicate = self.receiver.process_incoming_stream(
            payload, msg_id=msg_id, language=language
        )

        # 3. Determine control response
        if ack_packet is None:
            # Receiver is marked unavailable
            nack_packet = self.packetizer.packetize_nack(msg_id, reason_code=0x02)
            send_net_frame(
                client_sock,
                MSG_TYPE_NACK,
                {"msg_id": msg_id, "status": "unavailable", "error": "Receiver offline"},
                payload=nack_packet,
            )
            return

        ctrl = self.packetizer.parse_control_frame(ack_packet)
        if ctrl is not None:
            pkt_type, rx_hash, code, crc_valid = ctrl
            if pkt_type == PACKET_TYPE_NACK:
                # CRC corruption or framing error detected by receiver
                send_net_frame(
                    client_sock,
                    MSG_TYPE_NACK,
                    {
                        "msg_id": msg_id,
                        "status": "corrupted",
                        "error": "CRC integrity failure on one or more radio packets",
                    },
                    payload=ack_packet,
                )
                return

        # 4. Invoke delivery callback for application pipeline processing
        if not is_duplicate and self.on_message_delivered is not None:
            try:
                # Extract reconstructed neural symbols from last delivered message
                last_msg = self.receiver.received_messages[-1] if self.receiver.received_messages else None
                symbols_arr = last_msg.payload if last_msg is not None else np.array([], dtype=np.float32)
                event_data = {
                    "msg_id": msg_id,
                    "language": language,
                    "text": text,
                    "priority": priority_val,
                    "symbols": symbols_arr,
                    "sender_addr": client_addr,
                    "received_at": time.time(),
                    "is_duplicate": False,
                }
                self.on_message_delivered(event_data)
            except Exception as e:
                import traceback
                traceback.print_exc()
        elif is_duplicate and hasattr(self, "on_duplicate_received") and self.on_duplicate_received is not None:
            try:
                self.on_duplicate_received({
                    "msg_id": msg_id,
                    "language": language,
                    "sender_addr": client_addr,
                    "received_at": time.time(),
                    "is_duplicate": True,
                })
            except Exception:
                pass

        # Valid delivery (or duplicate re-acknowledgement): send ACK to client
        send_net_frame(
            client_sock,
            MSG_TYPE_ACK,
            {
                "msg_id": msg_id,
                "status": DeliveryStatus.DELIVERED.value,
                "is_duplicate": is_duplicate,
                "receiver_node": self.node_id,
            },
            payload=ack_packet,
        )

    def stop(self):
        """Stop receiver server and release socket."""
        self._stop_event.set()
        self.is_running = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self):
        """Clean up server and persistent store."""
        self.stop()
        if hasattr(self.receiver, "close"):
            self.receiver.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TcpRadioSenderClient:
    """TCP-based Radio Sender Client for iTantra.

    Connects to a receiver laptop over Wi-Fi / local TCP socket.
    Encodes neural channel symbols into framed radio packets with CRC-16 checksums,
    transmits them over the network, waits for ACK confirmation, handles ARQ retries,
    enforces emergency message priority order and TTL expiration, and persists
    outgoing message status in a local SQLite database.
    """

    def __init__(
        self,
        node_id: str = "tx_laptop",
        default_max_retries: int = 3,
        socket_timeout: float = 4.0,
        db_path: Optional[str] = "data/sender_messages.db",
        store: Optional[MessageStore] = None,
    ):
        """Initialize TCP Radio Sender Client.

        Args:
            node_id: Sender station identifier.
            default_max_retries: Max ARQ retransmission attempts.
            socket_timeout: TCP socket timeout in seconds.
            db_path: Path to persistent SQLite store for sent/queued messages.
            store: Optional existing MessageStore instance.
        """
        self.node_id = node_id
        self.default_max_retries = default_max_retries
        self.socket_timeout = socket_timeout
        self.db_path = db_path
        self.packetizer = ReliablePacketizer(max_payload_symbols=128)

        if store is not None:
            self.store = store
        else:
            self.store = MessageStore(db_path=db_path or ":memory:")

        self.queue: List[Message] = []
        self._restore_from_store()

    def _msg_sort_key(self, m: Message) -> Tuple[int, float]:
        prio_val = int(m.priority.value if hasattr(m.priority, "value") else m.priority)
        return (-prio_val, m.created_at)

    def _enqueue(self, msg: Message):
        self.queue = [m for m in self.queue if m.msg_id != msg.msg_id]
        self.queue.append(msg)
        self.queue.sort(key=self._msg_sort_key)

    def _restore_from_store(self):
        stored = self.store.get_pending_and_queued_messages()
        now = time.time()
        for msg in stored:
            if msg.is_expired(now):
                msg.mark_expired("TTL expired while client was offline")
                self.store.mark_expired(msg.msg_id, error=msg.error)
                continue
            if msg.status == DeliveryStatus.PENDING:
                msg.mark_queued("Restored from pending: queued for network transmission")
                self.store.mark_queued(msg.msg_id, error=msg.error)
            self._enqueue(msg)

    def send_message(
        self,
        receiver_host: str,
        receiver_port: int,
        msg_id: str,
        payload: Union[np.ndarray, torch.Tensor],
        language: str = "hi",
        text: str = "",
        priority: Any = "normal",
        ttl: Optional[float] = None,
        max_retries: Optional[int] = None,
        channel: Optional[Any] = None,
        snr_db: Optional[float] = None,
        created_at: Optional[float] = None,
    ) -> NetworkDeliveryResult:
        """Send a message to a receiver laptop over TCP.

        Args:
            receiver_host: IP address or hostname of receiver laptop (e.g. '192.168.1.45').
            receiver_port: Port receiver server is listening on (e.g. 8765).
            msg_id: Unique message identifier.
            payload: Neural channel symbols (numpy array or torch Tensor).
            language: Language code ('hi', 'te', 'ta', 'bn', 'mr').
            text: Optional message text for metadata reference.
            priority: Priority level ('emergency', 'high', 'normal', 'low' or MessagePriority).
            ttl: Time-To-Live in seconds; expired messages are never transmitted.
            max_retries: Maximum ARQ retransmissions.
            channel: Optional simulated wireless channel (AWGN/Rayleigh) for hybrid testing.
                     Set to None for pure unimpaired Wi-Fi network transmission.
            snr_db: Optional SNR for channel simulation if channel is provided.
            created_at: Message creation timestamp.

        Returns:
            NetworkDeliveryResult containing status, retry count, latency, and telemetry.
        """
        start_time = time.time()
        retries_limit = self.default_max_retries if max_retries is None else max_retries
        msg_prio = MessagePriority.from_value(priority)
        msg_created_at = created_at if created_at is not None else time.time()

        # Detach symbols to float32 numpy array
        if isinstance(payload, torch.Tensor):
            symbols = payload.detach().cpu().numpy().flatten().astype(np.float32)
        elif isinstance(payload, (bytes, bytearray)):
            symbols = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
        else:
            symbols = np.asarray(payload, dtype=np.float32).flatten()

        msg = Message(
            msg_id=msg_id,
            payload=symbols,
            language=language,
            status=DeliveryStatus.PENDING,
            max_retries=retries_limit,
            priority=msg_prio,
            ttl=ttl,
            created_at=msg_created_at,
        )

        # 1. Pre-check TTL Expiration before connecting
        if msg.is_expired():
            msg.mark_expired("TTL expired before network transmission attempt")
            self.store.save_message(msg)
            self.queue = [m for m in self.queue if m.msg_id != msg_id]
            elapsed_ms = (time.time() - start_time) * 1000.0
            return NetworkDeliveryResult(
                msg_id=msg_id,
                status=DeliveryStatus.EXPIRED,
                is_delivered=False,
                is_expired=True,
                is_queued=False,
                is_failed=False,
                retry_count=0,
                max_retries=retries_limit,
                latency_ms=elapsed_ms,
                packets_sent=0,
                bytes_sent=0,
                priority=msg_prio.name.lower(),
                error=msg.error,
                server_address=(receiver_host, receiver_port),
            )

        # 2. Persist message in local SQLite store
        self.store.save_message(msg)

        # 3. Optional Channel Simulation (AWGN/Rayleigh) for experimental evaluation
        # By default channel=None, representing clean Wi-Fi transmission
        if channel is not None:
            ch_device = getattr(channel, "device", "cpu")
            tensor_sym = torch.from_numpy(symbols).to(ch_device)
            with torch.no_grad():
                noisy_sym = channel(tensor_sym, snr_db=snr_db or 15.0)
            tx_symbols = noisy_sym.cpu().numpy().flatten().astype(np.float32)
        else:
            tx_symbols = symbols

        # 4. Radio packetization (standard iTantra radio framing with preamble & CRC-16)
        radio_packets = self.packetizer.packetize_stream(tx_symbols, language=language)
        radio_stream = b"".join(radio_packets)
        expected_hash = hash_msg_id_16(msg_id)

        metadata = {
            "msg_id": msg_id,
            "language": language,
            "text": text,
            "priority": msg_prio.name.lower(),
            "priority_level": int(msg_prio.value),
            "ttl": ttl,
            "created_at": msg_created_at,
            "num_packets": len(radio_packets),
            "num_symbols": len(tx_symbols),
            "sender_node": self.node_id,
        }

        attempt = 0
        packets_sent_total = 0
        bytes_sent_total = 0
        last_error: Optional[str] = None
        server_unavailable = False

        while attempt <= retries_limit:
            # Check TTL between retries
            if msg.is_expired():
                msg.mark_expired(f"TTL expired during network retries (attempt {attempt})")
                self.store.mark_expired(msg_id, error=msg.error)
                self.queue = [m for m in self.queue if m.msg_id != msg_id]
                elapsed_ms = (time.time() - start_time) * 1000.0
                return NetworkDeliveryResult(
                    msg_id=msg_id,
                    status=DeliveryStatus.EXPIRED,
                    is_delivered=False,
                    is_expired=True,
                    is_queued=False,
                    is_failed=False,
                    retry_count=attempt,
                    max_retries=retries_limit,
                    latency_ms=elapsed_ms,
                    packets_sent=packets_sent_total,
                    bytes_sent=bytes_sent_total,
                    priority=msg_prio.name.lower(),
                    error=msg.error,
                    server_address=(receiver_host, receiver_port),
                )

            msg.retry_count = attempt
            self.store.update_retry_count(msg_id, attempt)

            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_sock.settimeout(self.socket_timeout)

            try:
                # Connect to receiver laptop
                client_sock.connect((receiver_host, receiver_port))
                server_unavailable = False

                # Send network frame containing radio stream
                send_net_frame(client_sock, MSG_TYPE_DATA, metadata, payload=radio_stream)
                packets_sent_total += len(radio_packets)
                bytes_sent_total += len(radio_stream)

                # Wait for ACK response from receiver
                resp_type, resp_meta, ack_payload = recv_net_frame(client_sock, timeout=self.socket_timeout)

                if resp_type == MSG_TYPE_ACK:
                    ctrl = self.packetizer.parse_control_frame(ack_payload)
                    if ctrl is not None:
                        pkt_type, rx_hash, code, crc_valid = ctrl
                        if crc_valid and pkt_type == PACKET_TYPE_ACK and rx_hash == expected_hash:
                            # Confirmed delivery over network!
                            msg.mark_delivered()
                            self.store.mark_delivered(msg_id, msg.delivered_at)
                            self.queue = [m for m in self.queue if m.msg_id != msg_id]
                            elapsed_ms = (time.time() - start_time) * 1000.0
                            is_dup = resp_meta.get("is_duplicate", False)
                            return NetworkDeliveryResult(
                                msg_id=msg_id,
                                status=DeliveryStatus.DELIVERED,
                                is_delivered=True,
                                is_expired=False,
                                is_queued=False,
                                is_failed=False,
                                retry_count=attempt,
                                max_retries=retries_limit,
                                latency_ms=elapsed_ms,
                                packets_sent=packets_sent_total,
                                bytes_sent=bytes_sent_total,
                                priority=msg_prio.name.lower(),
                                server_address=(receiver_host, receiver_port),
                                is_duplicate=is_dup,
                            )
                elif resp_type == MSG_TYPE_NACK:
                    # Explicit NACK from server (e.g. CRC corruption or expired on server)
                    nack_reason = resp_meta.get("error", "Received NACK from server")
                    last_error = f"NACK from server: {nack_reason}"
                    if resp_meta.get("status") == DeliveryStatus.EXPIRED.value:
                        msg.mark_expired(nack_reason)
                        self.store.mark_expired(msg_id, error=msg.error)
                        self.queue = [m for m in self.queue if m.msg_id != msg_id]
                        elapsed_ms = (time.time() - start_time) * 1000.0
                        return NetworkDeliveryResult(
                            msg_id=msg_id,
                            status=DeliveryStatus.EXPIRED,
                            is_delivered=False,
                            is_expired=True,
                            is_queued=False,
                            is_failed=False,
                            retry_count=attempt,
                            max_retries=retries_limit,
                            latency_ms=elapsed_ms,
                            packets_sent=packets_sent_total,
                            bytes_sent=bytes_sent_total,
                            priority=msg_prio.name.lower(),
                            error=last_error,
                            server_address=(receiver_host, receiver_port),
                        )

            except (ConnectionRefusedError, socket.timeout, ConnectionResetError, OSError) as e:
                last_error = f"Network error ({type(e).__name__}): {e}"
                server_unavailable = True
            finally:
                try:
                    client_sock.close()
                except Exception:
                    pass

            attempt += 1
            if attempt <= retries_limit:
                time.sleep(0.1 * attempt)

        # Transmission attempts exhausted
        elapsed_ms = (time.time() - start_time) * 1000.0

        if server_unavailable:
            # Server is offline: queue message in store-and-forward queue
            msg.mark_queued(f"Receiver unreachable at {receiver_host}:{receiver_port}. Queued for later delivery.")
            self._enqueue(msg)
            self.store.mark_queued(msg_id, error=msg.error)
            return NetworkDeliveryResult(
                msg_id=msg_id,
                status=DeliveryStatus.QUEUED,
                is_delivered=False,
                is_expired=False,
                is_queued=True,
                is_failed=False,
                retry_count=attempt - 1,
                max_retries=retries_limit,
                latency_ms=elapsed_ms,
                packets_sent=packets_sent_total,
                bytes_sent=bytes_sent_total,
                priority=msg_prio.name.lower(),
                error=msg.error,
                server_address=(receiver_host, receiver_port),
            )

        # Normal failed delivery after retries
        msg.mark_failed(f"Failed to deliver after {retries_limit} retries. Last error: {last_error}")
        self.store.mark_failed(msg_id, msg.error, retry_count=msg.retry_count)
        self.queue = [m for m in self.queue if m.msg_id != msg_id]
        return NetworkDeliveryResult(
            msg_id=msg_id,
            status=DeliveryStatus.FAILED,
            is_delivered=False,
            is_expired=False,
            is_queued=False,
            is_failed=True,
            retry_count=attempt - 1,
            max_retries=retries_limit,
            latency_ms=elapsed_ms,
            packets_sent=packets_sent_total,
            bytes_sent=bytes_sent_total,
            priority=msg_prio.name.lower(),
            error=msg.error,
            server_address=(receiver_host, receiver_port),
        )

    def flush_queue(self, receiver_host: str, receiver_port: int) -> List[NetworkDeliveryResult]:
        """Flush store-and-forward queue in priority order once receiver becomes available."""
        results = []
        pending_msgs = list(self.queue)
        for msg in pending_msgs:
            res = self.send_message(
                receiver_host=receiver_host,
                receiver_port=receiver_port,
                msg_id=msg.msg_id,
                payload=msg.payload,
                language=msg.language,
                priority=msg.priority,
                ttl=msg.ttl,
                max_retries=msg.max_retries,
                created_at=msg.created_at,
            )
            results.append(res)
            if res.is_queued:
                # Server is still unavailable, abort remaining queue
                break
        return results

    def close(self):
        """Close persistent store."""
        if hasattr(self.store, "close"):
            self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
