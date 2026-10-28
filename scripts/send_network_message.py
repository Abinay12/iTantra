"""TCP Network Sender Client for iTantra.

Run this script on the Sender Laptop to transmit neural speech messages
to a Receiver Laptop over a local Wi-Fi network.

Features:
  - Synthesizes speech with Hindi/Telugu TTS engines.
  - Extracts acoustic Mel features and encodes them into neural symbols.
  - Encodes symbols into radio packet frames with preamble and CRC-16 checksums.
  - Transmits over TCP socket connection to receiver laptop's Wi-Fi IP address.
  - Waits for binary ACK control frame verification from receiver.
  - Automatically handles ARQ retransmissions on packet loss or corruption.
  - Enforces emergency priority order and Time-To-Live (TTL) deadlines.
  - Stores offline messages in SQLite store-and-forward queue if receiver is unreachable.
  - Clearly distinguishes pure Wi-Fi transport from optional simulated RF channels.

Usage:
  python scripts/send_network_message.py --host 192.168.1.45 --port 8765 --text "नमस्ते, आप कैसे हैं?" --lang hi
  python scripts/send_network_message.py --host 192.168.1.45 --priority emergency --text "आपातकालीन सहायता चाहिए!"
  python scripts/send_network_message.py --host 192.168.1.45 --flush-queue
"""

import argparse
import os
import sys
import time
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.pipeline import ITantraPipeline
from src.radio.network_transport import TcpRadioSenderClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transmit neural speech message over Wi-Fi TCP from Sender Laptop."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Target IP address of receiver laptop (e.g. '192.168.1.45', default: '127.0.0.1').",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port number receiver server is listening on (default: 8765).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="नमस्ते, आप कैसे हैं?",
        help="Message text to synthesize and send (default: 'नमस्ते, आप कैसे हैं?').",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        choices=["hi", "te", "ta", "bn", "mr", "kn", "ml"],
        help="Language code ('hi', 'te', 'ta', 'bn', 'mr', 'kn', 'ml', default: 'hi').",
    )
    parser.add_argument(
        "--priority",
        type=str,
        default="normal",
        choices=["emergency", "high", "normal", "low"],
        help="Message priority level (default: 'normal').",
    )
    parser.add_argument(
        "--ttl",
        type=float,
        default=None,
        help="Time-To-Live in seconds (e.g. 60.0). Expired messages will not be transmitted.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum ARQ retransmission attempts (default: 3).",
    )
    parser.add_argument(
        "--msg-id",
        type=str,
        default=None,
        help="Custom unique message identifier (auto-generated if omitted).",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        choices=[None, "awgn", "rayleigh"],
        help="Optional simulated channel impairment (default: None for pure unimpaired Wi-Fi).",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=15.0,
        help="SNR in dB if channel simulation is enabled (default: 15.0 dB).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/sender_messages.db",
        help="SQLite database path for sender message tracking (default: 'data/sender_messages.db').",
    )
    parser.add_argument(
        "--flush-queue",
        action="store_true",
        help="Flush stored offline messages to receiver in priority order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("      iTantra: TCP Radio Sender Client (Sender Laptop)")
    print("=" * 80)
    print(f"Target Receiver:  {args.host}:{args.port}")
    print(f"Priority Level:   {args.priority.upper()}")
    print(f"Time-To-Live:     {f'{args.ttl}s' if args.ttl is not None else 'None (Persistent)'}")
    print(f"Max ARQ Retries:  {args.retries}")
    print(f"Database Path:    {os.path.abspath(args.db_path)}")

    if args.channel is None:
        print(f"RF Channel Mode:  DIRECT REAL WI-FI (Clean TCP transport, no simulated RF noise)")
    else:
        print(f"RF Channel Mode:  HYBRID EXPERIMENT (Simulated {args.channel.upper()} @ {args.snr}dB applied before Wi-Fi)")
    print("=" * 80)

    # Initialize sender pipeline
    print("\n[*] Initializing transmitter pipeline (TTS + Neural Encoder)...")
    pipeline = ITantraPipeline(db_path=args.db_path)
    print("[*] Transmitter pipeline ready.")

    sender_client = TcpRadioSenderClient(
        node_id="tx_laptop_01",
        default_max_retries=args.retries,
        db_path=args.db_path,
        store=pipeline.reliable_sender.store,
    )

    if args.flush_queue:
        print(f"\n[*] Flushing pending/queued store-and-forward messages to {args.host}:{args.port}...")
        results = sender_client.flush_queue(receiver_host=args.host, receiver_port=args.port)
        print(f"[*] Processed {len(results)} queued messages.")
        for r in results:
            stat_str = "DELIVERED" if r.is_delivered else ("QUEUED" if r.is_queued else "FAILED")
            print(f"    - Message {r.msg_id} (Prio: {r.priority.upper()}) -> {stat_str} (Latency: {r.latency_ms:.1f}ms, Retries: {r.retry_count})")
        pipeline.close()
        return

    msg_id = args.msg_id or f"WIFI_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    print(f"\n[*] Preparing Message: \"{args.text}\" (Language: {args.lang}, ID: {msg_id})")

    res = pipeline.send_network_message(
        receiver_host=args.host,
        receiver_port=args.port,
        text=args.text,
        language=args.lang,
        msg_id=msg_id,
        priority=args.priority,
        ttl=args.ttl,
        max_retries=args.retries,
        channel_type=args.channel,
        snr_db=args.snr,
    )

    print("\n" + "=" * 80)
    print("                     TRANSMISSION OUTCOME")
    print("=" * 80)
    print(f"Message ID:       {res.msg_id}")
    print(f"Delivery Status:  {res.status.value.upper()}")
    print(f"Delivery Latency: {res.latency_ms:.2f} ms")
    print(f"Retries Taken:    {res.retry_count} / {res.max_retries}")
    print(f"Packets Sent:     {res.packets_sent}")
    print(f"Bytes Sent:       {res.bytes_sent} bytes")
    print(f"Target Server:    {res.server_address[0]}:{res.server_address[1]}")

    if res.is_delivered:
        print("\n[SUCCESS] Message successfully delivered to receiver laptop and acknowledged via CRC-verified ACK frame!")
    elif res.is_queued:
        print(f"\n[QUEUED] Receiver was unreachable ({res.error}). Message is safely stored in local SQLite database.")
        print(f"         Run with '--flush-queue' once receiver is online to transmit.")
    elif res.is_expired:
        print(f"\n[EXPIRED] Message exceeded its configured TTL of {args.ttl}s. Transmission aborted.")
    elif res.is_failed:
        print(f"\n[FAILED] Message delivery failed after {res.retry_count} retries. Error: {res.error}")
    print("=" * 80)

    pipeline.close()


if __name__ == "__main__":
    main()
