"""Comprehensive Verification Script for Steps 6 & 7:
Packet framing, language IDs, CRC-16 integrity, malformed packets, error handling,
ACK/retries, duplicate suppression, SQLite persistence, store-and-forward,
emergency priority, and TTL expiration.
"""

import os
import sys
import struct
import tempfile
import time
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

from src.radio.packetizer import (
    Packetizer,
    PREAMBLE_SYNC,
    PROTOCOL_VERSION,
    LANG_TO_ID,
    ID_TO_LANG,
)
from src.radio.reliable_delivery import (
    DeliveryStatus,
    Message,
    MessagePriority,
    ReliablePacketizer,
    ReliableReceiver,
    ReliableSender,
    PACKET_TYPE_ACK,
    PACKET_TYPE_NACK,
    hash_msg_id_16,
)
from src.radio.message_store import MessageStore


def test_packet_framing_and_language_ids():
    print("\n--- Sub-test 6.1: Packet Framing & Language IDs for All 7 Target Languages ---")
    packetizer = Packetizer(max_payload_symbols=64)
    expected_ids = {
        "te": 1,
        "hi": 2,
        "ta": 3,
        "bn": 4,
        "mr": 5,
        "kn": 6,
        "ml": 7,
    }

    dummy_symbols = np.linspace(-1.0, 1.0, 32, dtype=np.float32)

    for lang, expected_id in expected_ids.items():
        # Check registry
        actual_id = LANG_TO_ID.get(lang)
        assert actual_id == expected_id, f"Language ID mismatch for {lang}: expected {expected_id}, got {actual_id}"

        # Packetize
        raw = packetizer.packetize(dummy_symbols, language=lang, packet_id=1, total_packets=5)
        assert raw.startswith(PREAMBLE_SYNC), f"Preamble missing for {lang}"

        # Check raw header fields
        version, l_id, pkt_id, tot_pkts, sym_cnt = struct.unpack(">BBHHH", raw[4:12])
        assert version == PROTOCOL_VERSION, f"Version mismatch: {version}"
        assert l_id == expected_id, f"Lang ID in packet mismatch: {l_id} != {expected_id}"
        assert pkt_id == 1, f"Packet ID mismatch: {pkt_id}"
        assert tot_pkts == 5, f"Total packets mismatch: {tot_pkts}"
        assert sym_cnt == 32, f"Symbol count mismatch: {sym_cnt}"

        # Depacketize and verify CRC
        pkt = packetizer.depacketize(raw)
        assert pkt.crc_valid is True, f"CRC-16 check failed for {lang}"
        assert pkt.language == lang, f"Decoded language mismatch: {pkt.language} != {lang}"
        assert np.allclose(pkt.payload_symbols, dummy_symbols), f"Payload symbols altered for {lang}"
        print(f"  [OK] Language '{lang}': ID={l_id}, Preamble=0xAA55AA55, Syms={sym_cnt}, CRC=VALID")


def test_crc16_and_malformed_packets():
    print("\n--- Sub-test 6.2: CRC-16 Integrity Checks & Malformed Packet Handling ---")
    packetizer = Packetizer(max_payload_symbols=64)
    dummy_symbols = np.array([0.5, -0.25, 0.75, -0.1], dtype=np.float32)
    valid_raw = packetizer.packetize(dummy_symbols, language="hi", packet_id=0, total_packets=1)

    # 1. Valid packet test
    pkt = packetizer.depacketize(valid_raw)
    assert pkt.crc_valid is True, "Valid packet failed CRC"
    print("  [OK] Valid packet: CRC integrity verified successfully.")

    # 2. Corrupt payload byte
    corrupted_payload = bytearray(valid_raw)
    corrupted_payload[14] ^= 0xFF  # Flip bits in payload
    corrupted_pkt = packetizer.depacketize(bytes(corrupted_payload))
    assert corrupted_pkt.crc_valid is False, "Corrupted payload was not flagged by CRC!"
    print("  [OK] Corrupted payload: CRC check correctly detected bit error (crc_valid=False).")

    # 3. Corrupt CRC bytes
    corrupted_crc = bytearray(valid_raw)
    corrupted_crc[-1] ^= 0x01
    corrupted_crc_pkt = packetizer.depacketize(bytes(corrupted_crc))
    assert corrupted_crc_pkt.crc_valid is False, "Corrupted CRC was not flagged!"
    print("  [OK] Corrupted CRC checksum: Detected mismatch (crc_valid=False).")

    # 4. Truncated packet (< minimum 14 bytes)
    try:
        packetizer.depacketize(b"\xAA\x55\xAA\x55\x01\x02")
        assert False, "Should have raised ValueError on truncated packet"
    except ValueError as e:
        print(f"  [OK] Truncated packet (<14 bytes): Correctly raised ValueError: {e}")

    # 5. Missing / corrupt preamble sync word
    try:
        bad_sync = b"\x00\x00\x00\x00" + valid_raw[4:]
        packetizer.depacketize(bad_sync)
        assert False, "Should have raised ValueError on missing sync preamble"
    except ValueError as e:
        print(f"  [OK] Missing sync preamble: Correctly raised ValueError: {e}")

    # 6. Truncated payload relative to symbol count header
    try:
        truncated_payload = valid_raw[:-10]
        packetizer.depacketize(truncated_payload)
        assert False, "Should have raised ValueError on truncated payload"
    except ValueError as e:
        print(f"  [OK] Truncated payload: Correctly raised ValueError: {e}")


def test_control_frames_ack_nack():
    print("\n--- Sub-test 6.3: Control Frames (ACK/NACK) & Malformed Control Handling ---")
    rel_packetizer = ReliablePacketizer(max_payload_symbols=64)
    msg_id = "MSG_CTRL_TEST_001"

    # 1. ACK frame
    ack_bytes = rel_packetizer.packetize_ack(msg_id=msg_id, status_code=0)
    assert ack_bytes.startswith(PREAMBLE_SYNC)
    parsed_ack = rel_packetizer.parse_control_frame(ack_bytes)
    assert parsed_ack is not None
    pkt_type, msg_hash, code, crc_valid = parsed_ack
    assert pkt_type == PACKET_TYPE_ACK
    assert msg_hash == hash_msg_id_16(msg_id)
    assert code == 0
    assert crc_valid is True
    print(f"  [OK] ACK frame generated and verified: msg_hash=0x{msg_hash:04X}, crc_valid={crc_valid}")

    # 2. NACK frame
    nack_bytes = rel_packetizer.packetize_nack(msg_id=msg_id, reason_code=0x05)
    parsed_nack = rel_packetizer.parse_control_frame(nack_bytes)
    assert parsed_nack is not None
    pkt_type_n, msg_hash_n, code_n, crc_valid_n = parsed_nack
    assert pkt_type_n == PACKET_TYPE_NACK
    assert code_n == 0x05
    assert crc_valid_n is True
    print(f"  [OK] NACK frame generated and verified: reason_code=0x{code_n:02X}, crc_valid={crc_valid_n}")

    # 3. Corrupt ACK CRC
    corrupt_ack = bytearray(ack_bytes)
    corrupt_ack[-1] ^= 0xAA
    parsed_corrupt = rel_packetizer.parse_control_frame(bytes(corrupt_ack))
    assert parsed_corrupt is not None
    assert parsed_corrupt[3] is False
    print("  [OK] Corrupted control frame CRC: Correctly flagged crc_valid=False")

    # 4. Malformed short control frame
    short_parsed = rel_packetizer.parse_control_frame(b"short")
    assert short_parsed is None
    print("  [OK] Malformed short control frame: Correctly returned None")


def test_arq_and_duplicate_protection():
    print("\n--- Sub-test 6.4: ARQ Retries, Error Handling & Duplicate Suppression ---")
    rel_packetizer = ReliablePacketizer(max_payload_symbols=64)
    sender = ReliableSender(node_id="tx_tester", default_max_retries=3, packetizer=rel_packetizer)
    receiver = ReliableReceiver(node_id="rx_tester")
    symbols = np.sin(np.linspace(0, 2 * np.pi, 32), dtype=np.float32)

    # 1. Direct successful transmission
    msg1 = sender.send_message(msg_id="MSG_SUCCESS_01", payload=symbols, language="te", receiver=receiver)
    assert msg1.status == DeliveryStatus.DELIVERED
    assert msg1.retry_count == 0
    assert len(receiver.received_messages) == 1
    print("  [OK] Direct transmission: status=DELIVERED, retries=0")

    # 2. Duplicate transmission protection
    msg1_dup = sender.send_message(msg_id="MSG_SUCCESS_01", payload=symbols, language="te", receiver=receiver)
    assert msg1_dup.status == DeliveryStatus.DELIVERED
    assert len(receiver.received_messages) == 1, "Duplicate message was not suppressed by receiver!"
    assert len(receiver.seen_message_ids) == 1
    print("  [OK] Duplicate transmission: correctly suppressed (receiver message count unchanged at 1)")

    # 3. Exceeded max retries failure
    msg_fail = sender.send_message(
        msg_id="MSG_FAIL_TEST",
        payload=symbols,
        language="hi",
        receiver=receiver,
        packet_loss_rate=1.0,
        max_retries=2,
    )
    assert msg_fail.status == DeliveryStatus.FAILED
    assert msg_fail.retry_count == 2
    assert "retries" in msg_fail.error.lower()
    print(f"  [OK] Exceeded retries: status=FAILED, retries={msg_fail.retry_count}, error='{msg_fail.error}'")


def test_sqlite_persistence_priority_and_expiration():
    print("\n--- Sub-test 7.1: SQLite Persistence, Store-and-Forward across Process Restart ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = os.path.join(tmp_dir, "test_persistence.db")

        # Instance 1: receiver offline, queue message
        store1 = MessageStore(db_path=db_file)
        sender1 = ReliableSender(node_id="tx_persist", store=store1)
        receiver1 = ReliableReceiver(node_id="rx_persist")
        receiver1.set_availability(False)

        symbols = np.ones(16, dtype=np.float32)
        msg_queued = sender1.send_message(
            msg_id="MSG_PERSIST_001",
            payload=symbols,
            language="bn",
            receiver=receiver1,
            priority=MessagePriority.HIGH,
        )
        assert msg_queued.status == DeliveryStatus.QUEUED
        assert sender1.queued_count == 1
        print(f"  [OK] Sender instance 1 queued message to SQLite: msg_id={msg_queued.msg_id}, status={msg_queued.status.value}")

        # Close instance 1
        store1.close()
        del sender1, store1

        # Instance 2: new process start, restore queued messages
        store2 = MessageStore(db_path=db_file)
        sender2 = ReliableSender(node_id="tx_persist", store=store2)
        assert sender2.queued_count == 1, f"Failed to restore queued count: got {sender2.queued_count}"
        restored_msg = sender2.queue[0]
        assert restored_msg.msg_id == "MSG_PERSIST_001"
        assert restored_msg.status == DeliveryStatus.QUEUED
        assert restored_msg.priority == MessagePriority.HIGH
        print(f"  [OK] Process restart: restored queued message '{restored_msg.msg_id}' from SQLite database")

        # Reconnect receiver and flush
        receiver1.set_availability(True)
        flushed = sender2.flush_queue(receiver=receiver1)
        assert len(flushed) == 1
        assert flushed[0].status == DeliveryStatus.DELIVERED
        assert sender2.queued_count == 0

        # Verify SQLite record is DELIVERED
        stored_record = store2.get_message("MSG_PERSIST_001")
        assert stored_record is not None
        assert stored_record.status == DeliveryStatus.DELIVERED
        print("  [OK] Reconnected and flushed: message DELIVERED and updated in SQLite store.")
        store2.close()

    print("\n--- Sub-test 7.2: Emergency Priority Ordering & TTL Expiration ---")
    with tempfile.TemporaryDirectory() as tmp_dir2:
        db_file2 = os.path.join(tmp_dir2, "test_prio_ttl.db")
        store = MessageStore(db_path=db_file2)
        sender = ReliableSender(node_id="tx_prio", store=store)
        receiver = ReliableReceiver(node_id="rx_prio")
        receiver.set_availability(False)

        symbols = np.zeros(8, dtype=np.float32)
        # Queue in arbitrary order: LOW, NORMAL, EMERGENCY, EXPIRED
        sender.send_message(msg_id="MSG_LOW", payload=symbols, language="te", receiver=receiver, priority=MessagePriority.LOW)
        sender.send_message(msg_id="MSG_NORMAL", payload=symbols, language="te", receiver=receiver, priority=MessagePriority.NORMAL)
        sender.send_message(msg_id="MSG_EMERGENCY", payload=symbols, language="te", receiver=receiver, priority=MessagePriority.EMERGENCY)
        sender.send_message(msg_id="MSG_EXPIRED", payload=symbols, language="te", receiver=receiver, priority=MessagePriority.HIGH, ttl=0.08)

        # Verify ordering: EMERGENCY must be first
        assert sender.queue[0].msg_id == "MSG_EMERGENCY", f"Emergency not at front: {sender.queue[0].msg_id}"
        print(f"  [OK] Priority queue order verified: {[m.msg_id for m in sender.queue]}")

        # Wait for TTL to expire
        time.sleep(0.12)

        receiver.set_availability(True)
        flushed_results = sender.flush_queue(receiver=receiver)

        flushed_dict = {m.msg_id: m for m in flushed_results}
        assert flushed_dict["MSG_EMERGENCY"].status == DeliveryStatus.DELIVERED
        assert flushed_dict["MSG_NORMAL"].status == DeliveryStatus.DELIVERED
        assert flushed_dict["MSG_LOW"].status == DeliveryStatus.DELIVERED
        assert flushed_dict["MSG_EXPIRED"].status == DeliveryStatus.EXPIRED, f"Expired status wrong: {flushed_dict['MSG_EXPIRED'].status}"

        # Receiver should only have 3 messages delivered (not expired)
        assert len(receiver.received_messages) == 3
        rx_ids = [m.msg_id for m in receiver.received_messages]
        assert "MSG_EXPIRED" not in rx_ids
        assert rx_ids[0] == "MSG_EMERGENCY"  # Emergency delivered first
        print(f"  [OK] Flush results: 3 delivered (Emergency first: {rx_ids[0]}), 1 expired ({flushed_dict['MSG_EXPIRED'].msg_id})")

        store.close()


def main():
    print("=" * 80)
    print("  iTantra: Targeted Protocol, Framing, Integrity, ARQ & SQLite Test Suite")
    print("=" * 80)

    test_packet_framing_and_language_ids()
    test_crc16_and_malformed_packets()
    test_control_frames_ack_nack()
    test_arq_and_duplicate_protection()
    test_sqlite_persistence_priority_and_expiration()

    print("\n" + "=" * 80)
    print("  ALL PROTOCOL, FRAMING, CRC-16, ARQ & PERSISTENCE TESTS PASSED (100% SUCCESS)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
