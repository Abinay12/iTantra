"""Verification script for Step 9: Offline TCP Communication Test.
Tests live TCP socket transmission, framing, CRC-16 verification, ACK reply,
and SQLite persistence on the local network (loopback).
"""

import os
import sys
import time
import tempfile
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.radio.network_transport import (
    TcpRadioReceiverServer,
    TcpRadioSenderClient,
    get_local_ip_addresses,
)
from src.radio.reliable_delivery import DeliveryStatus
from src.radio.message_store import MessageStore
from src.pipeline import ITantraPipeline


def main():
    print("=" * 80)
    print("  iTantra: Step 9 - Offline TCP Communication Verification")
    print("=" * 80)

    # 1. Local IP discovery
    local_ips = get_local_ip_addresses()
    print("1. Network Interface Inspection:")
    for idx, ip in enumerate(local_ips, 1):
        hint = " (Localhost Loopback)" if ip.startswith("127.") else " (Local Wi-Fi / LAN IP)"
        print(f"   Interface [{idx}]: {ip}{hint}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        rx_db = os.path.join(tmp_dir, "receiver_test.db")
        tx_db = os.path.join(tmp_dir, "sender_test.db")

        # 2. Start Receiver Server on local loopback
        print("\n2. Launching TCP Receiver Server Node (port 8769 on 127.0.0.1)...")
        received_messages = []

        def on_msg_delivered(event):
            received_messages.append(event)
            print(f"   [RECEIVER] Incoming message received: ID={event['msg_id']}, Lang={event['language']}, Priority={event['priority']}")

        def on_msg_duplicate(event):
            print(f"   [RECEIVER] Duplicate message suppressed: ID={event['msg_id']}")

        server = TcpRadioReceiverServer(
            host="127.0.0.1",
            port=8769,
            node_id="rx_node_laptop_b",
            db_path=rx_db,
            on_message_delivered=on_msg_delivered,
            on_duplicate_received=on_msg_duplicate,
        )
        server.start()
        actual_port = server.bound_port
        print(f"   [+] Receiver server listening on 127.0.0.1:{actual_port}")

        try:
            # 3. Transmit messages from Sender Client
            print("\n3. Launching TCP Sender Client Node...")
            client = TcpRadioSenderClient(
                node_id="tx_node_laptop_a",
                db_path=tx_db,
                socket_timeout=5.0,
            )

            # Test 1: Emergency Hindi Message Transmission
            print("\n   [Test 9.1] Sending Emergency Hindi message over TCP...")
            dummy_symbols_hi = np.sin(np.linspace(0, 4 * np.pi, 64), dtype=np.float32)
            res1 = client.send_message(
                receiver_host="127.0.0.1",
                receiver_port=actual_port,
                msg_id="TCP_EMERGENCY_HINDI_001",
                payload=dummy_symbols_hi,
                language="hi",
                text="आपातकालीन सहायता की आवश्यकता है!",
                priority="emergency",
                ttl=60.0,
            )
            print(f"     Status:           {res1.status.value.upper()}")
            print(f"     Is Delivered:     {res1.is_delivered}")
            print(f"     Latency:          {res1.latency_ms:.2f} ms")
            print(f"     Packets Sent:     {res1.packets_sent}")
            print(f"     Bytes Sent:       {res1.bytes_sent}")
            print(f"     Retries:          {res1.retry_count}")
            assert res1.is_delivered, "TCP Test 1 failed delivery!"
            assert res1.status == DeliveryStatus.DELIVERED

            # Test 2: Idempotent Duplicate Transmission (Duplicate Suppression)
            print("\n   [Test 9.2] Retransmitting identical message ID (Duplicate Suppression Check)...")
            res_dup = client.send_message(
                receiver_host="127.0.0.1",
                receiver_port=actual_port,
                msg_id="TCP_EMERGENCY_HINDI_001",
                payload=dummy_symbols_hi,
                language="hi",
                text="आपातकालीन सहायता की आवश्यकता है!",
                priority="emergency",
            )
            print(f"     Status:           {res_dup.status.value.upper()}")
            print(f"     Is Duplicate:     {res_dup.is_duplicate}")
            assert res_dup.is_delivered, "Duplicate should still return delivered ACK"
            assert res_dup.is_duplicate, "Receiver failed to detect duplicate!"

            # Test 3: Normal Priority Telugu Message Transmission
            print("\n   [Test 9.3] Sending Normal Telugu message over TCP...")
            dummy_symbols_te = np.cos(np.linspace(0, 2 * np.pi, 48), dtype=np.float32)
            res2 = client.send_message(
                receiver_host="127.0.0.1",
                receiver_port=actual_port,
                msg_id="TCP_NORMAL_TELUGU_002",
                payload=dummy_symbols_te,
                language="te",
                text="నమస్కారం, ఇది పరీక్ష.",
                priority="normal",
                ttl=None,
            )
            print(f"     Status:           {res2.status.value.upper()}")
            print(f"     Is Delivered:     {res2.is_delivered}")
            print(f"     Latency:          {res2.latency_ms:.2f} ms")
            assert res2.is_delivered, "TCP Test 3 failed delivery!"

            # Verify SQLite databases on both sides
            print("\n4. Verifying Persistent Storage on Both Sender & Receiver Nodes:")
            rx_store = MessageStore(db_path=rx_db)
            seen_ids = rx_store.get_seen_message_ids(server.node_id)
            print(f"   Receiver Node Seen Message IDs in SQLite: {list(seen_ids)}")
            assert "TCP_EMERGENCY_HINDI_001" in seen_ids
            assert "TCP_NORMAL_TELUGU_002" in seen_ids
            rx_store.close()

            tx_store = MessageStore(db_path=tx_db)
            tx_m1 = tx_store.get_message("TCP_EMERGENCY_HINDI_001")
            tx_m2 = tx_store.get_message("TCP_NORMAL_TELUGU_002")
            print(f"   Sender Node Record 1 in SQLite: ID={tx_m1.msg_id}, Status={tx_m1.status.value.upper()}")
            print(f"   Sender Node Record 2 in SQLite: ID={tx_m2.msg_id}, Status={tx_m2.status.value.upper()}")
            assert tx_m1.status == DeliveryStatus.DELIVERED
            assert tx_m2.status == DeliveryStatus.DELIVERED
            tx_store.close()

            client.close()

        finally:
            server.close()
            print("\n[+] Receiver server closed cleanly.")

    print("\n" + "=" * 80)
    print("  OFFLINE TCP COMMUNICATION TEST (LOCAL LOOPBACK) PASSED (100% SUCCESS)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
