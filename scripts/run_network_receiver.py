"""TCP Network Receiver Server Node for iTantra.

Run this script on the Receiver Laptop to accept real-time neural radio
transmissions over a local Wi-Fi network.

Features:
  - Auto-detects and displays local Wi-Fi / LAN IP addresses.
  - Verifies radio packet preambles, headers, and CRC-16 checksums.
  - Re-generates binary ACK control frames and returns them over TCP.
  - Detects duplicate deliveries using persistent SQLite message storage.
  - Enforces message Time-To-Live (TTL) expiration.
  - Optional neural symbol decoding and vocoder audio reconstruction.

Usage:
  python scripts/run_network_receiver.py --port 8765
  python scripts/run_network_receiver.py --port 8765 --decode
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch

from src.radio.network_transport import (
    TcpRadioReceiverServer,
    get_local_ip_addresses,
)
from src.pipeline import ITantraPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run iTantra TCP Radio Receiver Server on Receiver Laptop."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to ('0.0.0.0' binds to all network interfaces, default: '0.0.0.0').",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen for incoming sender connections (default: 8765).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/receiver_messages.db",
        help="SQLite database path to persist received messages (default: 'data/receiver_messages.db').",
    )
    parser.add_argument(
        "--node-id",
        type=str,
        default="rx_laptop_01",
        help="Receiver node identifier (default: 'rx_laptop_01').",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Enable neural decoding and STT transcription on received symbols.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory to save received reconstructed audio (default: 'output').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    local_ips = get_local_ip_addresses()

    print("=" * 80)
    print("      iTantra: TCP Radio Receiver Server (Receiver Laptop)")
    print("=" * 80)
    print(f"Node ID:          {args.node_id}")
    print(f"Listening Port:   {args.port}")
    print(f"Bind Host:        {args.host}")
    print(f"Database:         {os.path.abspath(args.db_path)}")
    print(f"Neural Decoding:  {'ENABLED' if args.decode else 'DISABLED (Header/CRC/ACK mode)'}")
    print("-" * 80)
    print("Detected Local IP Addresses on this Laptop:")
    for idx, ip in enumerate(local_ips, 1):
        is_wifi_hint = " (Likely Wi-Fi / LAN IP - use this on sender laptop)" if not ip.startswith("127.") else ""
        print(f"  [{idx}] {ip}:{args.port}{is_wifi_hint}")
    print("-" * 80)
    print("Instructions:")
    print(f"  1. Ensure this laptop and the sender laptop are connected to the SAME Wi-Fi.")
    print(f"  2. On the sender laptop, run:")
    primary_ip = [ip for ip in local_ips if not ip.startswith("127.")][0] if any(not ip.startswith("127.") for ip in local_ips) else "127.0.0.1"
    print(f"     python scripts/send_network_message.py --host {primary_ip} --port {args.port} --text \"नमस्ते\" --lang hi")
    print(f"  3. Press Ctrl+C in this terminal to stop the receiver server.")
    print("=" * 80)

    pipeline = None
    if args.decode:
        print("\n[*] Initializing neural decoder pipeline for speech reconstruction...")
        pipeline = ITantraPipeline(db_path=args.db_path)
        print("[*] Neural pipeline ready.")

    msg_count = 0

    def on_duplicate(event):
        nonlocal msg_count
        msg_count += 1
        msg_id = event["msg_id"]
        sender_addr = event["sender_addr"]
        rec_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event["received_at"]))
        print("\n" + "=" * 80)
        print(f"[RECEIVER EVENT #{msg_count}] Duplicate Message Suppressed (ID: {msg_id})")
        print("=" * 80)
        print(f"Timestamp:       {rec_time}")
        print(f"Sender IP:       {sender_addr[0]}:{sender_addr[1]}")
        print(f"Message ID:      {msg_id}")
        print(f"Status:          DUPLICATE DETECTED via SQLite seen_message_ids")
        print(f"Action:          Re-sent Idempotent ACK to Sender (Duplicate Synthesis Suppressed)")
        print("=" * 80)

    def on_message(event):
        nonlocal msg_count
        msg_count += 1
        msg_id = event["msg_id"]
        lang = event["language"]
        prio = str(event["priority"]).upper()
        symbols = event["symbols"]
        sender_addr = event["sender_addr"]
        rec_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event["received_at"]))

        print("\n" + "=" * 80)
        print(f"[RECEIVER EVENT #{msg_count}] Incoming Message Delivered over Wi-Fi!")
        print("=" * 80)
        print(f"Timestamp:       {rec_time}")
        print(f"Sender IP:       {sender_addr[0]}:{sender_addr[1]}")
        print(f"Message ID:      {msg_id}")
        print(f"Language:        {lang}")
        print(f"Priority:        {prio}")
        print(f"Neural Symbols:  {len(symbols)} float32 values received")
        print(f"CRC-16 Status:   ALL PACKET CHECKSUMS PASSED")
        print(f"ACK Status:      ACK Generated & Sent to Client")

        if event.get("text"):
            print(f"Reference Text:  \"{event['text']}\"")

        if pipeline is not None and len(symbols) > 0:
            try:
                sym_tensor = torch.from_numpy(symbols).to(pipeline.device)
                if len(symbols) % pipeline.latent_dim == 0:
                    sym_tensor = sym_tensor.view(1, -1, pipeline.latent_dim)
                else:
                    pad = pipeline.latent_dim - (len(symbols) % pipeline.latent_dim)
                    sym_tensor = torch.nn.functional.pad(sym_tensor, (0, pad)).view(1, -1, pipeline.latent_dim)
                with torch.no_grad():
                    rec_features = pipeline.decoder(sym_tensor)
                rx_audio = pipeline.stt.features_to_audio(rec_features)
                transcription = pipeline.stt.transcribe(audio=rx_audio, language=lang)
                print(f"Recovered Speech:\"{transcription.text}\"")

                os.makedirs(args.output_dir, exist_ok=True)
                audio_path = os.path.join(args.output_dir, f"rx_{msg_id}.wav")
                pipeline.tts.save_audio(rx_audio, audio_path)
                print(f"Reconstructed Audio Saved: {os.path.abspath(audio_path)}")
            except Exception as ex:
                print(f"Speech Reconstruction Notice: {ex}")

        print("=" * 80)

    server = TcpRadioReceiverServer(
        host=args.host,
        port=args.port,
        node_id=args.node_id,
        db_path=args.db_path,
        on_message_delivered=on_message,
        on_duplicate_received=on_duplicate,
    )

    server.start()
    print(f"\n[+] Server is ACTIVE and listening on {args.host}:{server.bound_port}...")
    print("[+] Waiting for incoming messages from sender laptop...\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[*] Stopping receiver server...")
    finally:
        server.close()
        if pipeline is not None:
            pipeline.close()
        print("[+] Receiver server stopped cleanly.")


if __name__ == "__main__":
    main()
