"""Unit tests for Reliable Message Delivery, ARQ, and Store-and-Forward Protocol.

Tests verify:
1. ACK-based message delivery confirmation.
2. Automatic Repeat reQuest (ARQ) retransmission on packet loss and CRC corruption with configurable retry limits.
3. Store-and-forward message queuing when the receiver is offline/unavailable.
4. Message IDs (msg_id) and duplicate message suppression at the receiver application layer.
5. Delivery statuses: PENDING, DELIVERED, FAILED, and QUEUED.
6. Integration with ITantraPipeline and wireless channels (AWGN & Rayleigh).
"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch

from src.radio.reliable_delivery import (
    DeliveryStatus,
    Message,
    MessagePriority,
    ReliablePacketizer,
    ReliableSender,
    ReliableReceiver,
    PACKET_TYPE_DATA,
    PACKET_TYPE_ACK,
    PACKET_TYPE_NACK,
    hash_msg_id_16,
)
from src.radio.message_store import MessageStore
from src.radio.packetizer import PREAMBLE_SYNC, PROTOCOL_VERSION
from src.pipeline import ITantraPipeline


class TestDeliveryStatusAndMessage(unittest.TestCase):
    """Test Message dataclass and DeliveryStatus lifecycle."""

    def test_delivery_status_enum_values(self):
        self.assertEqual(DeliveryStatus.PENDING.value, "pending")
        self.assertEqual(DeliveryStatus.DELIVERED.value, "delivered")
        self.assertEqual(DeliveryStatus.FAILED.value, "failed")
        self.assertEqual(DeliveryStatus.QUEUED.value, "queued")
        self.assertEqual(DeliveryStatus.EXPIRED.value, "expired")

        # Verify string comparison equality
        self.assertEqual(DeliveryStatus.PENDING, "pending")
        self.assertEqual(DeliveryStatus.DELIVERED, "delivered")
        self.assertEqual(DeliveryStatus.FAILED, "failed")
        self.assertEqual(DeliveryStatus.QUEUED, "queued")
        self.assertEqual(DeliveryStatus.EXPIRED, "expired")

    def test_message_lifecycle_transitions(self):
        symbols = np.array([0.1, -0.2, 0.5, -0.9], dtype=np.float32)
        msg = Message(
            msg_id="test_msg_001",
            payload=symbols,
            language="hi",
            max_retries=3,
        )

        # Initial state
        self.assertEqual(msg.status, DeliveryStatus.PENDING)
        self.assertEqual(msg.retry_count, 0)
        self.assertIsNone(msg.delivered_at)
        self.assertIsNone(msg.error)

        # Transition to QUEUED
        msg.mark_queued()
        self.assertEqual(msg.status, DeliveryStatus.QUEUED)
        self.assertIsNotNone(msg.error)
        self.assertIn("queued", msg.error.lower())

        # Transition to DELIVERED
        msg.mark_delivered()
        self.assertEqual(msg.status, DeliveryStatus.DELIVERED)
        self.assertIsNotNone(msg.delivered_at)
        self.assertIsNone(msg.error)

        # Transition to FAILED
        msg.mark_failed("Exceeded retry limit")
        self.assertEqual(msg.status, DeliveryStatus.FAILED)
        self.assertEqual(msg.error, "Exceeded retry limit")

        # Transition to EXPIRED
        msg.mark_expired("TTL timed out")
        self.assertEqual(msg.status, DeliveryStatus.EXPIRED)
        self.assertEqual(msg.error, "TTL timed out")


class TestReliablePacketizer(unittest.TestCase):
    """Test control frames (ACK/NACK) and packet integrity."""

    def setUp(self):
        self.packetizer = ReliablePacketizer(max_payload_symbols=64)

    def test_ack_packet_structure_and_parsing(self):
        msg_id = "ack_test_123"
        ack_bytes = self.packetizer.packetize_ack(msg_id=msg_id, status_code=0)

        # Verify preamble
        self.assertTrue(ack_bytes.startswith(PREAMBLE_SYNC))

        # Parse control frame
        parsed = self.packetizer.parse_control_frame(ack_bytes)
        self.assertIsNotNone(parsed)
        pkt_type, msg_hash, code, crc_valid = parsed

        self.assertEqual(pkt_type, PACKET_TYPE_ACK)
        self.assertEqual(msg_hash, hash_msg_id_16(msg_id))
        self.assertEqual(code, 0)
        self.assertTrue(crc_valid)

    def test_nack_packet_structure_and_parsing(self):
        msg_id = "nack_test_456"
        nack_bytes = self.packetizer.packetize_nack(msg_id=msg_id, reason_code=0x02)

        self.assertTrue(nack_bytes.startswith(PREAMBLE_SYNC))

        parsed = self.packetizer.parse_control_frame(nack_bytes)
        self.assertIsNotNone(parsed)
        pkt_type, msg_hash, code, crc_valid = parsed

        self.assertEqual(pkt_type, PACKET_TYPE_NACK)
        self.assertEqual(msg_hash, hash_msg_id_16(msg_id))
        self.assertEqual(code, 0x02)
        self.assertTrue(crc_valid)

    def test_control_frame_crc_corruption_detection(self):
        msg_id = "crc_test_789"
        ack_bytes = bytearray(self.packetizer.packetize_ack(msg_id=msg_id))

        # Corrupt CRC byte
        ack_bytes[-1] ^= 0x55

        parsed = self.packetizer.parse_control_frame(bytes(ack_bytes))
        self.assertIsNotNone(parsed)
        pkt_type, msg_hash, code, crc_valid = parsed
        self.assertFalse(crc_valid)

    def test_malformed_control_frame_returns_none(self):
        self.assertIsNone(self.packetizer.parse_control_frame(b"short"))
        self.assertIsNone(self.packetizer.parse_control_frame(b"no_sync_header_data_here"))


class TestReliableDeliveryProtocol(unittest.TestCase):
    """Test reliable sender and receiver interactions (ACK, ARQ, Dup suppression, Queuing)."""

    def setUp(self):
        self.packetizer = ReliablePacketizer(max_payload_symbols=64)
        self.sender = ReliableSender(node_id="tx_test", default_max_retries=3, packetizer=self.packetizer)
        self.receiver = ReliableReceiver(node_id="rx_test")
        self.dummy_symbols = np.sin(np.linspace(0, 4 * np.pi, 64), dtype=np.float32)

    def test_ack_delivery_success(self):
        """Feature 1: Direct transmission delivers message and receives ACK."""
        msg = self.sender.send_message(
            msg_id="msg_ack_01",
            payload=self.dummy_symbols,
            language="hi",
            receiver=self.receiver,
        )

        self.assertEqual(msg.status, DeliveryStatus.DELIVERED)
        self.assertEqual(msg.retry_count, 0)
        self.assertIsNotNone(msg.delivered_at)
        self.assertIsNone(msg.error)

        # Receiver should have stored exactly one message
        self.assertIn("msg_ack_01", self.receiver.seen_message_ids)
        self.assertEqual(len(self.receiver.received_messages), 1)
        self.assertEqual(self.receiver.received_messages[0].msg_id, "msg_ack_01")
        self.assertEqual(self.receiver.received_messages[0].language, "hi")

    def test_automatic_retransmission_on_packet_loss(self):
        """Feature 2: Packet loss triggers ARQ retransmission until delivery succeeds."""
        # Mock random to drop the first attempt, then succeed on second attempt
        with patch("numpy.random.rand", side_effect=[0.05, 0.95, 0.95]):
            msg = self.sender.send_message(
                msg_id="msg_arq_loss",
                payload=self.dummy_symbols,
                language="te",
                receiver=self.receiver,
                packet_loss_rate=0.5,
                max_retries=3,
            )

            self.assertEqual(msg.status, DeliveryStatus.DELIVERED)
            self.assertEqual(msg.retry_count, 1)  # Retried once and succeeded
            self.assertEqual(len(self.receiver.received_messages), 1)

    def test_retransmission_exceeding_retry_limit_fails(self):
        """Feature 2: Persistent failure marks message as FAILED when max_retries exceeded."""
        # 100% packet loss rate
        msg = self.sender.send_message(
            msg_id="msg_arq_fail",
            payload=self.dummy_symbols,
            language="hi",
            receiver=self.receiver,
            packet_loss_rate=1.0,
            max_retries=2,
        )

        self.assertEqual(msg.status, DeliveryStatus.FAILED)
        self.assertEqual(msg.retry_count, 2)
        self.assertIsNotNone(msg.error)
        self.assertIn("Failed to deliver after 2 retries", msg.error)
        self.assertEqual(len(self.receiver.received_messages), 0)

    def test_retransmission_on_crc_integrity_failure(self):
        """Feature 2: Corrupted packets causing CRC failure trigger retransmission."""
        original_process = self.receiver.process_incoming_stream
        call_count = {"count": 0}

        def mock_process_stream(raw_stream, msg_id, language="hi"):
            call_count["count"] += 1
            if call_count["count"] == 1:
                # First attempt: simulate corrupted CRC by returning NACK
                nack = self.receiver.packetizer.packetize_nack(msg_id, reason_code=0x01)
                return nack, False
            # Second attempt: normal clean delivery
            return original_process(raw_stream, msg_id, language)

        with patch.object(self.receiver, "process_incoming_stream", side_effect=mock_process_stream):
            msg = self.sender.send_message(
                msg_id="msg_crc_retry",
                payload=self.dummy_symbols,
                language="ta",
                receiver=self.receiver,
                max_retries=3,
            )

            self.assertEqual(msg.status, DeliveryStatus.DELIVERED)
            self.assertEqual(msg.retry_count, 1)
            self.assertEqual(call_count["count"], 2)
            self.assertEqual(len(self.receiver.received_messages), 1)

    def test_store_and_forward_queuing_when_receiver_unavailable(self):
        """Feature 3: Sender queues messages when receiver is offline."""
        self.receiver.set_availability(False)

        msg = self.sender.send_message(
            msg_id="msg_queued_01",
            payload=self.dummy_symbols,
            language="bn",
            receiver=self.receiver,
        )

        self.assertEqual(msg.status, DeliveryStatus.QUEUED)
        self.assertEqual(self.sender.queued_count, 1)
        self.assertIn("Receiver unavailable", msg.error)
        self.assertEqual(len(self.receiver.received_messages), 0)

    def test_store_and_forward_flush_queue(self):
        """Feature 3: Queued messages are flushed and delivered when receiver is back online."""
        self.receiver.set_availability(False)

        # Queue two messages while receiver is offline
        msg1 = self.sender.send_message("msg_sf_1", self.dummy_symbols, language="te", receiver=self.receiver)
        msg2 = self.sender.send_message("msg_sf_2", self.dummy_symbols, language="hi", receiver=self.receiver)

        self.assertEqual(self.sender.queued_count, 2)
        self.assertEqual(msg1.status, DeliveryStatus.QUEUED)
        self.assertEqual(msg2.status, DeliveryStatus.QUEUED)

        # Receiver comes back online
        self.receiver.set_availability(True)

        flushed = self.sender.flush_queue(receiver=self.receiver)

        self.assertEqual(len(flushed), 2)
        self.assertEqual(self.sender.queued_count, 0)
        self.assertTrue(all(m.status == DeliveryStatus.DELIVERED for m in flushed))
        self.assertEqual(len(self.receiver.received_messages), 2)
        self.assertEqual(self.receiver.received_messages[0].msg_id, "msg_sf_1")
        self.assertEqual(self.receiver.received_messages[1].msg_id, "msg_sf_2")

    def test_duplicate_message_prevention(self):
        """Feature 4: Resending identical msg_id produces ACK but suppresses duplicate delivery."""
        # First transmission
        msg1 = self.sender.send_message(
            msg_id="dup_msg_99",
            payload=self.dummy_symbols,
            language="mr",
            receiver=self.receiver,
        )
        self.assertEqual(msg1.status, DeliveryStatus.DELIVERED)
        self.assertEqual(len(self.receiver.received_messages), 1)

        # Second transmission with exact same msg_id (e.g. lost ACK retransmission)
        msg2 = self.sender.send_message(
            msg_id="dup_msg_99",
            payload=self.dummy_symbols,
            language="mr",
            receiver=self.receiver,
        )
        self.assertEqual(msg2.status, DeliveryStatus.DELIVERED)

        # Receiver must STILL have only 1 delivered message (duplicate suppressed)
        self.assertEqual(len(self.receiver.received_messages), 1)
        self.assertEqual(self.receiver.received_messages[0].msg_id, "dup_msg_99")


class TestPipelineReliableDeliveryIntegration(unittest.TestCase):
    """Test reliable delivery integrated with ITantraPipeline and wireless channels."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = ITantraPipeline()

    def setUp(self):
        self.pipeline.reliable_receiver.reset()
        self.pipeline.reliable_sender.clear_queue()

    def test_pipeline_send_reliable_hindi_awgn(self):
        """Test sending Hindi message over AWGN channel with reliable delivery confirmation."""
        res = self.pipeline.send_reliable_message(
            msg_id="pipe_hi_01",
            text="नमस्ते, आप कैसे हैं?",
            language="hi",
            channel_type="awgn",
            snr_db=20.0,
        )

        self.assertEqual(res["msg_id"], "pipe_hi_01")
        self.assertEqual(res["status"], "delivered")
        self.assertTrue(res["is_delivered"])
        self.assertFalse(res["is_queued"])
        self.assertFalse(res["is_failed"])
        self.assertEqual(res["retry_count"], 0)
        self.assertEqual(res["receiver_delivered_messages"], 1)

    def test_pipeline_send_reliable_telugu_rayleigh(self):
        """Test sending Telugu message over Rayleigh channel with reliable delivery confirmation."""
        res = self.pipeline.send_reliable_message(
            msg_id="pipe_te_01",
            text="నమస్కారం, మీరు ఎలా ఉన్నారు?",
            language="te",
            channel_type="rayleigh",
            snr_db=25.0,
        )

        self.assertEqual(res["msg_id"], "pipe_te_01")
        self.assertEqual(res["status"], "delivered")
        self.assertTrue(res["is_delivered"])
        self.assertEqual(res["retry_count"], 0)

    def test_pipeline_store_and_forward_flow(self):
        """Test store-and-forward queueing and flushing via ITantraPipeline."""
        # 1. Receiver is offline -> message is queued
        res_queued = self.pipeline.send_reliable_message(
            msg_id="pipe_sf_01",
            text="नमस्ते",
            language="hi",
            receiver_available=False,
        )
        self.assertEqual(res_queued["status"], "queued")
        self.assertTrue(res_queued["is_queued"])
        self.assertFalse(res_queued["is_delivered"])

        # 2. Flush queue -> message is delivered
        flushed = self.pipeline.flush_reliable_queue(channel_type="awgn", snr_db=20.0)
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0]["msg_id"], "pipe_sf_01")
        self.assertEqual(flushed[0]["status"], "delivered")
        self.assertTrue(flushed[0]["is_delivered"])

    def test_pipeline_duplicate_suppression(self):
        """Test duplicate message suppression via pipeline entry point."""
        res1 = self.pipeline.send_reliable_message(
            msg_id="dup_pipe_01",
            text="నమస్కారం",
            language="te",
        )
        self.assertTrue(res1["is_delivered"])
        self.assertEqual(res1["receiver_delivered_messages"], 1)

        # Transmit again with identical msg_id
        res2 = self.pipeline.send_reliable_message(
            msg_id="dup_pipe_01",
            text="నమస్కారం",
            language="te",
        )
        self.assertTrue(res2["is_delivered"])
        # Delivered message count remains 1
        self.assertEqual(res2["receiver_delivered_messages"], 1)


class TestMessageStorePersistence(unittest.TestCase):
    """Test SQLite MessageStore CRUD, status tracking, and duplicate prevention."""

    def setUp(self):
        self.store = MessageStore(db_path=":memory:")

    def tearDown(self):
        self.store.close()

    def test_save_and_retrieve_symbols_message(self):
        symbols = np.array([0.5, -0.25, 0.125, -0.875], dtype=np.float32)
        msg = Message(
            msg_id="store_test_01",
            payload=symbols,
            language="te",
            status=DeliveryStatus.QUEUED,
            retry_count=1,
            max_retries=4,
            error="Receiver offline",
        )
        self.assertTrue(self.store.save_message(msg))

        retrieved = self.store.get_message("store_test_01")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.msg_id, "store_test_01")
        self.assertEqual(retrieved.language, "te")
        self.assertEqual(retrieved.status, DeliveryStatus.QUEUED)
        self.assertEqual(retrieved.retry_count, 1)
        self.assertEqual(retrieved.max_retries, 4)
        self.assertEqual(retrieved.error, "Receiver offline")
        self.assertTrue(np.allclose(retrieved.payload, symbols, atol=1e-6))

    def test_save_and_retrieve_text_payload(self):
        msg = Message(
            msg_id="text_msg_01",
            payload="నమస్కారం ప్రపంచం",
            language="te",
            status=DeliveryStatus.PENDING,
        )
        self.assertTrue(self.store.save_message(msg))

        retrieved = self.store.get_message("text_msg_01")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.payload, "నమస్కారం ప్రపంచం")
        self.assertEqual(retrieved.status, DeliveryStatus.PENDING)

    def test_status_transitions_in_store(self):
        msg = Message(msg_id="status_trans_01", payload=np.zeros(4, dtype=np.float32), language="hi")
        self.store.save_message(msg)

        # Mark queued
        self.assertTrue(self.store.mark_queued("status_trans_01", error="Receiver busy"))
        m = self.store.get_message("status_trans_01")
        self.assertEqual(m.status, DeliveryStatus.QUEUED)
        self.assertEqual(m.error, "Receiver busy")

        # Mark delivered
        deliv_time = 1234567.89
        self.assertTrue(self.store.mark_delivered("status_trans_01", delivered_at=deliv_time))
        m = self.store.get_message("status_trans_01")
        self.assertEqual(m.status, DeliveryStatus.DELIVERED)
        self.assertEqual(m.delivered_at, deliv_time)
        self.assertIsNone(m.error)

        # Mark failed
        self.assertTrue(self.store.mark_failed("status_trans_01", error="Max retries reached", retry_count=3))
        m = self.store.get_message("status_trans_01")
        self.assertEqual(m.status, DeliveryStatus.FAILED)
        self.assertEqual(m.error, "Max retries reached")
        self.assertEqual(m.retry_count, 3)

    def test_duplicate_save_does_not_revert_delivered_status(self):
        msg = Message(
            msg_id="dup_status_01",
            payload=np.ones(4, dtype=np.float32),
            language="hi",
            status=DeliveryStatus.DELIVERED,
        )
        self.store.save_message(msg)

        # Attempt to save same message again with status PENDING (simulating re-processing)
        duplicate_msg = Message(
            msg_id="dup_status_01",
            payload=np.ones(4, dtype=np.float32),
            language="hi",
            status=DeliveryStatus.PENDING,
        )
        saved = self.store.save_message(duplicate_msg)
        self.assertFalse(saved)

        # Status in database must still be DELIVERED
        retrieved = self.store.get_message("dup_status_01")
        self.assertEqual(retrieved.status, DeliveryStatus.DELIVERED)

    def test_get_pending_and_queued_filtering(self):
        self.store.save_message(Message("m_deliv", np.zeros(2, dtype=np.float32), status=DeliveryStatus.DELIVERED))
        self.store.save_message(Message("m_fail", np.zeros(2, dtype=np.float32), status=DeliveryStatus.FAILED))
        self.store.save_message(Message("m_pend", np.zeros(2, dtype=np.float32), status=DeliveryStatus.PENDING))
        self.store.save_message(Message("m_queue", np.zeros(2, dtype=np.float32), status=DeliveryStatus.QUEUED))

        pending_or_queued = self.store.get_pending_and_queued_messages()
        ids = [m.msg_id for m in pending_or_queued]
        self.assertEqual(len(ids), 2)
        self.assertIn("m_pend", ids)
        self.assertIn("m_queue", ids)
        self.assertNotIn("m_deliv", ids)
        self.assertNotIn("m_fail", ids)

    def test_clear_queued_leaves_delivered_intact(self):
        self.store.save_message(Message("m_deliv", np.zeros(2, dtype=np.float32), status=DeliveryStatus.DELIVERED))
        self.store.save_message(Message("m_queue", np.zeros(2, dtype=np.float32), status=DeliveryStatus.QUEUED))

        self.store.clear_queued()
        self.assertIsNone(self.store.get_message("m_queue"))
        self.assertIsNotNone(self.store.get_message("m_deliv"))
        self.assertEqual(self.store.get_message("m_deliv").status, DeliveryStatus.DELIVERED)

    def test_receiver_message_store_persistence(self):
        """Test recording received messages and restoring seen message IDs from SQLite."""
        self.assertEqual(len(self.store.get_seen_message_ids()), 0)
        self.store.record_received_message("rx_msg_01", node_id="rx_1", language="hi", payload=np.array([1.0, 2.0]))
        self.store.record_received_message("rx_msg_02", node_id="rx_1", language="te", payload="text")

        seen = self.store.get_seen_message_ids()
        self.assertEqual(seen, {"rx_msg_01", "rx_msg_02"})

        # Record duplicate should be ignored (INSERT OR IGNORE)
        self.store.record_received_message("rx_msg_01", node_id="rx_1", language="hi")
        self.assertEqual(len(self.store.get_seen_message_ids()), 2)

        self.store.clear_received()
        self.assertEqual(len(self.store.get_seen_message_ids()), 0)

    def test_delete_delivered_and_status_filtering(self):
        """Test delete_delivered, get_delivered_messages, and get_failed_messages."""
        self.store.save_message(Message("m_del_1", np.zeros(2), status=DeliveryStatus.DELIVERED))
        self.store.save_message(Message("m_del_2", np.zeros(2), status=DeliveryStatus.DELIVERED))
        self.store.save_message(Message("m_fail_1", np.zeros(2), status=DeliveryStatus.FAILED, error="Timeout"))
        self.store.save_message(Message("m_queue_1", np.zeros(2), status=DeliveryStatus.QUEUED))

        delivered = self.store.get_delivered_messages()
        failed = self.store.get_failed_messages()
        self.assertEqual(len(delivered), 2)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].msg_id, "m_fail_1")

        deleted_count = self.store.delete_delivered()
        self.assertEqual(deleted_count, 2)
        self.assertEqual(len(self.store.get_delivered_messages()), 0)
        self.assertIsNotNone(self.store.get_message("m_fail_1"))
        self.assertIsNotNone(self.store.get_message("m_queue_1"))


class TestPersistentStoreAndForwardApplicationRestart(unittest.TestCase):
    """Test persistence across application restarts, ARQ retries, and SQLite store-and-forward."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "messages.db")
        self.receiver = ReliableReceiver(node_id="rx_restart_test")
        self.dummy_symbols = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    def tearDown(self):
        if hasattr(self, "receiver") and hasattr(self.receiver, "close"):
            self.receiver.close()
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_application_restart_restores_queued_messages(self):
        """Feature: Messages queued while receiver is offline survive process restart and are delivered when receiver reconnects."""
        self.receiver.set_availability(False)

        # --- Process Instance 1 ---
        sender1 = ReliableSender(node_id="tx_1", db_path=self.db_path)
        msg1 = sender1.send_message("REST_01", self.dummy_symbols, language="hi", receiver=self.receiver)
        msg2 = sender1.send_message("REST_02", self.dummy_symbols, language="te", receiver=self.receiver)
        self.assertEqual(msg1.status, DeliveryStatus.QUEUED)
        self.assertEqual(msg2.status, DeliveryStatus.QUEUED)
        self.assertEqual(sender1.queued_count, 2)
        sender1.store.close()
        del sender1

        # --- Process Instance 2 (Application Restart) ---
        sender2 = ReliableSender(node_id="tx_1", db_path=self.db_path)
        # Verify restored queue on startup
        self.assertEqual(sender2.queued_count, 2)
        self.assertEqual(sender2.queue[0].msg_id, "REST_01")
        self.assertEqual(sender2.queue[1].msg_id, "REST_02")
        self.assertEqual(sender2.queue[0].status, DeliveryStatus.QUEUED)
        self.assertEqual(sender2.queue[1].status, DeliveryStatus.QUEUED)
        self.assertTrue(np.allclose(sender2.queue[0].payload, self.dummy_symbols, atol=1e-6))

        # Receiver comes online -> flush queue
        self.receiver.set_availability(True)
        flushed = sender2.flush_queue(receiver=self.receiver)
        self.assertEqual(len(flushed), 2)
        self.assertTrue(all(m.status == DeliveryStatus.DELIVERED for m in flushed))
        self.assertEqual(sender2.queued_count, 0)
        self.assertEqual(len(self.receiver.received_messages), 2)

        # Verify SQLite database reflects delivered status
        db_m1 = sender2.get_stored_message("REST_01")
        db_m2 = sender2.get_stored_message("REST_02")
        self.assertEqual(db_m1.status, DeliveryStatus.DELIVERED)
        self.assertEqual(db_m2.status, DeliveryStatus.DELIVERED)
        self.assertIsNotNone(db_m1.delivered_at)
        self.assertIsNotNone(db_m2.delivered_at)
        sender2.store.close()
        del sender2

        # --- Process Instance 3 (Post-delivery Restart) ---
        sender3 = ReliableSender(node_id="tx_1", db_path=self.db_path)
        # Delivered messages must NOT be restored to pending queue
        self.assertEqual(sender3.queued_count, 0)
        sender3.store.close()

    def test_remove_on_delivered_option(self):
        """Feature: When remove_on_delivered is True, messages are removed from SQLite after successful delivery."""
        sender_remove = ReliableSender(node_id="tx_rm", db_path=self.db_path, remove_on_delivered=True)
        res = sender_remove.send_message("MSG_RM_01", self.dummy_symbols, language="hi", receiver=self.receiver)
        self.assertEqual(res.status, DeliveryStatus.DELIVERED)

        # In SQLite, message record must have been deleted
        self.assertIsNone(sender_remove.get_stored_message("MSG_RM_01"))

        # In contrast, when remove_on_delivered is False, record persists as DELIVERED
        sender_keep = ReliableSender(node_id="tx_keep", db_path=self.db_path, remove_on_delivered=False)
        res2 = sender_keep.send_message("MSG_KEEP_01", self.dummy_symbols, language="hi", receiver=self.receiver)
        self.assertEqual(res2.status, DeliveryStatus.DELIVERED)
        stored_keep = sender_keep.get_stored_message("MSG_KEEP_01")
        self.assertIsNotNone(stored_keep)
        self.assertEqual(stored_keep.status, DeliveryStatus.DELIVERED)

        sender_remove.store.close()
        sender_keep.store.close()

    def test_prune_and_delete_delivered_messages(self):
        """Feature: prune_delivered and delete_stored_message remove delivered entries from SQLite."""
        sender = ReliableSender(node_id="tx_prune", db_path=self.db_path, remove_on_delivered=False)
        self.receiver.set_availability(False)
        sender.send_message("Q_MSG_01", self.dummy_symbols, language="te", receiver=self.receiver)
        self.receiver.set_availability(True)
        sender.send_message("DEL_MSG_01", self.dummy_symbols, language="te", receiver=self.receiver)

        self.assertIsNotNone(sender.get_stored_message("Q_MSG_01"))
        self.assertIsNotNone(sender.get_stored_message("DEL_MSG_01"))

        # Prune delivered
        count = sender.prune_delivered()
        self.assertEqual(count, 1)
        self.assertIsNone(sender.get_stored_message("DEL_MSG_01"))
        # Queued message remains untouched
        self.assertIsNotNone(sender.get_stored_message("Q_MSG_01"))

        # Delete specific stored message
        self.assertTrue(sender.delete_stored_message("Q_MSG_01"))
        self.assertIsNone(sender.get_stored_message("Q_MSG_01"))
        sender.store.close()

    def test_interrupted_pending_message_restored_as_queued_on_restart(self):
        """Feature: A message left in PENDING status due to an ungraceful shutdown is restored to QUEUED on restart."""
        # Create a store and insert a PENDING message (simulating crash during flight)
        raw_store = MessageStore(self.db_path)
        pending_msg = Message(
            msg_id="CRASH_PENDING_01",
            payload=self.dummy_symbols,
            language="bn",
            status=DeliveryStatus.PENDING,
            retry_count=1,
            max_retries=3,
        )
        raw_store.save_message(pending_msg)
        raw_store.close()

        # Restart sender: should restore the interrupted message as QUEUED
        sender = ReliableSender(node_id="tx_crash", db_path=self.db_path)
        self.assertEqual(sender.queued_count, 1)
        restored = sender.queue[0]
        self.assertEqual(restored.msg_id, "CRASH_PENDING_01")
        self.assertEqual(restored.status, DeliveryStatus.QUEUED)
        self.assertIn("Restored from pending", restored.error)

        # Verify DB also updated to QUEUED
        db_msg = sender.get_stored_message("CRASH_PENDING_01")
        self.assertEqual(db_msg.status, DeliveryStatus.QUEUED)

        # Now deliver it successfully
        flushed = sender.flush_queue(receiver=self.receiver)
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0].status, DeliveryStatus.DELIVERED)
        sender.store.close()

    def test_multiple_messages_fifo_restoration_order(self):
        """Feature: Multiple messages queued offline restore in strictly chronological FIFO order."""
        self.receiver.set_availability(False)
        sender1 = ReliableSender(node_id="tx_fifo", db_path=self.db_path)

        for i in range(5):
            sender1.send_message(f"FIFO_MSG_{i:02d}", self.dummy_symbols * (i + 1), language="hi", receiver=self.receiver)

        self.assertEqual(sender1.queued_count, 5)
        sender1.store.close()
        del sender1

        # Restart sender
        sender2 = ReliableSender(node_id="tx_fifo", db_path=self.db_path)
        self.assertEqual(sender2.queued_count, 5)

        restored_ids = [m.msg_id for m in sender2.queue]
        self.assertEqual(restored_ids, [f"FIFO_MSG_{i:02d}" for i in range(5)])

        # Check payload fidelity for each
        for i, m in enumerate(sender2.queue):
            expected_payload = self.dummy_symbols * (i + 1)
            self.assertTrue(np.allclose(m.payload, expected_payload, atol=1e-6))

        # Deliver all in order
        self.receiver.set_availability(True)
        flushed = sender2.flush_queue(receiver=self.receiver)
        self.assertEqual(len(flushed), 5)
        delivered_ids = [m.msg_id for m in flushed]
        self.assertEqual(delivered_ids, [f"FIFO_MSG_{i:02d}" for i in range(5)])
        sender2.store.close()

    def test_corrupted_packet_retry_limit_persisted_as_failed_not_restored(self):
        """Feature: Corrupted packets exceeding retry limit are marked FAILED in SQLite and NOT restored to the queue on restart."""
        sender = ReliableSender(node_id="tx_fail", db_path=self.db_path)

        # Force receiver to reject with NACK due to CRC mismatch every time
        def mock_process_stream_nack(raw_stream, msg_id, language="hi"):
            nack = self.receiver.packetizer.packetize_nack(msg_id, reason_code=0x01)
            return nack, False

        with patch.object(self.receiver, "process_incoming_stream", side_effect=mock_process_stream_nack):
            res = sender.send_message(
                msg_id="CORRUPT_01",
                payload=self.dummy_symbols,
                language="ta",
                receiver=self.receiver,
                max_retries=2,
            )

        self.assertEqual(res.status, DeliveryStatus.FAILED)
        self.assertEqual(res.retry_count, 2)

        # Verify status in SQLite
        db_res = sender.get_stored_message("CORRUPT_01")
        self.assertIsNotNone(db_res)
        self.assertEqual(db_res.status, DeliveryStatus.FAILED)
        self.assertIn("Failed to deliver after 2 retries", db_res.error)
        sender.store.close()
        del sender

        # Restart sender: FAILED message must NOT be restored to pending queue
        sender_restart = ReliableSender(node_id="tx_fail", db_path=self.db_path)
        self.assertEqual(sender_restart.queued_count, 0)
        sender_restart.store.close()

    def test_receiver_duplicate_suppression_persists_across_restart(self):
        """Feature: ReliableReceiver with persistent SQLite store retains seen_message_ids across restarts."""
        rx_db = os.path.join(self.temp_dir.name, "receiver.db")
        rx1 = ReliableReceiver(node_id="rx_persist", db_path=rx_db)
        sender = ReliableSender(node_id="tx_1", db_path=self.db_path)

        # Send first message
        res1 = sender.send_message("PERSIST_DUP_01", self.dummy_symbols, language="te", receiver=rx1)
        self.assertEqual(res1.status, DeliveryStatus.DELIVERED)
        self.assertEqual(len(rx1.received_messages), 1)
        rx1.store.close()
        del rx1

        # Restart receiver with same db_path
        rx2 = ReliableReceiver(node_id="rx_persist", db_path=rx_db)
        self.assertIn("PERSIST_DUP_01", rx2.seen_message_ids)

        # Attempt to deliver duplicate message to restarted receiver
        res2 = sender.send_message("PERSIST_DUP_01", self.dummy_symbols, language="te", receiver=rx2)
        self.assertEqual(res2.status, DeliveryStatus.DELIVERED)

        # Receiver should suppress duplicate: received_messages on rx2 should have 0 new deliveries
        self.assertEqual(len(rx2.received_messages), 0)
        rx2.store.close()
        sender.store.close()

    def test_duplicate_reprocessing_with_sqlite_persistence(self):
        """Feature: Reprocessing a message already delivered does not recreate or duplicate entries in SQLite."""
        sender = ReliableSender(node_id="tx_dup", db_path=self.db_path)

        # Initial successful delivery
        res1 = sender.send_message("DUP_01", self.dummy_symbols, language="ta", receiver=self.receiver)
        self.assertEqual(res1.status, DeliveryStatus.DELIVERED)

        # Reprocess same msg_id
        res2 = sender.send_message("DUP_01", self.dummy_symbols, language="ta", receiver=self.receiver)
        self.assertEqual(res2.status, DeliveryStatus.DELIVERED)

        # Verify only 1 record exists in SQLite
        all_stored = sender.store.get_all_messages()
        matching = [m for m in all_stored if m.msg_id == "DUP_01"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].status, DeliveryStatus.DELIVERED)
        sender.store.close()

    def test_pipeline_persistent_store_and_restart(self):
        """Feature: ITantraPipeline store-and-forward persistence across pipeline restarts."""
        # 1. Pipeline instance 1: receiver offline, message queued to SQLite
        pipe1 = ITantraPipeline(db_path=self.db_path)
        res_q = pipe1.send_reliable_message(
            msg_id="PIPE_PERSIST_01",
            text="नमस्ते, यह संदेश सहेजा गया है।",
            language="hi",
            receiver_available=False,
        )
        self.assertEqual(res_q["status"], "queued")
        self.assertTrue(res_q["is_queued"])
        self.assertEqual(pipe1.reliable_sender.queued_count, 1)
        pipe1.close()
        del pipe1

        # 2. Pipeline instance 2 (Restart): message restored from SQLite
        pipe2 = ITantraPipeline(db_path=self.db_path)
        self.assertEqual(pipe2.reliable_sender.queued_count, 1)
        self.assertEqual(pipe2.reliable_sender.queue[0].msg_id, "PIPE_PERSIST_01")

        # 3. Flush queue to online receiver
        flushed = pipe2.flush_reliable_queue(channel_type="awgn", snr_db=20.0)
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0]["msg_id"], "PIPE_PERSIST_01")
        self.assertEqual(flushed[0]["status"], "delivered")
        self.assertEqual(pipe2.reliable_sender.queued_count, 0)
        pipe2.close()

    def test_saving_queued_message(self):
        """Requirement 6.1: Saving a queued message in SQLite when receiver is offline."""
        self.receiver.set_availability(False)
        sender = ReliableSender(node_id="tx_save_test", db_path=self.db_path)
        msg = sender.send_message(
            msg_id="SAVE_QUEUED_001",
            payload=self.dummy_symbols,
            language="te",
            receiver=self.receiver,
        )
        self.assertEqual(msg.status, DeliveryStatus.QUEUED)
        self.assertEqual(sender.queued_count, 1)

        # Inspect SQLite directly
        stored = sender.get_stored_message("SAVE_QUEUED_001")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.msg_id, "SAVE_QUEUED_001")
        self.assertEqual(stored.status, DeliveryStatus.QUEUED)
        self.assertEqual(stored.language, "te")
        self.assertEqual(stored.retry_count, 0)
        self.assertIn("Receiver unavailable", stored.error)
        self.assertTrue(np.allclose(stored.payload, self.dummy_symbols, atol=1e-6))
        sender.close()

    def test_restarting_and_restoring_pending_messages(self):
        """Requirement 6.2: Restarting and restoring pending/queued messages from SQLite."""
        self.receiver.set_availability(False)
        # Session 1: queue messages
        sender1 = ReliableSender(node_id="tx_restart", db_path=self.db_path)
        sender1.send_message("RESTORE_P_01", self.dummy_symbols, language="hi", receiver=self.receiver)
        sender1.send_message("RESTORE_P_02", self.dummy_symbols * 2, language="bn", receiver=self.receiver)
        self.assertEqual(sender1.queued_count, 2)
        sender1.close()
        del sender1

        # Session 2: application restart
        sender2 = ReliableSender(node_id="tx_restart", db_path=self.db_path)
        self.assertEqual(sender2.queued_count, 2)
        self.assertEqual(sender2.queue[0].msg_id, "RESTORE_P_01")
        self.assertEqual(sender2.queue[1].msg_id, "RESTORE_P_02")
        self.assertEqual(sender2.queue[0].status, DeliveryStatus.QUEUED)
        self.assertEqual(sender2.queue[1].status, DeliveryStatus.QUEUED)
        self.assertTrue(np.allclose(sender2.queue[0].payload, self.dummy_symbols, atol=1e-6))
        self.assertTrue(np.allclose(sender2.queue[1].payload, self.dummy_symbols * 2, atol=1e-6))
        sender2.close()

    def test_successfully_delivering_queued_message(self):
        """Requirement 6.3: Successfully delivering a queued message once receiver is available."""
        self.receiver.set_availability(False)
        sender = ReliableSender(node_id="tx_deliver_queued", db_path=self.db_path)
        sender.send_message("DELIV_Q_01", self.dummy_symbols, language="mr", receiver=self.receiver)
        self.assertEqual(sender.queued_count, 1)

        # SQLite status is QUEUED
        stored_before = sender.get_stored_message("DELIV_Q_01")
        self.assertEqual(stored_before.status, DeliveryStatus.QUEUED)

        # Receiver comes back online
        self.receiver.set_availability(True)
        flushed = sender.flush_queue(receiver=self.receiver)
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0].status, DeliveryStatus.DELIVERED)
        self.assertEqual(sender.queued_count, 0)

        # SQLite status updated to DELIVERED with delivered_at timestamp
        stored_after = sender.get_stored_message("DELIV_Q_01")
        self.assertEqual(stored_after.status, DeliveryStatus.DELIVERED)
        self.assertIsNotNone(stored_after.delivered_at)
        self.assertIsNone(stored_after.error)
        self.assertEqual(len(self.receiver.received_messages), 1)
        sender.close()

    def test_preventing_duplicate_message_delivery(self):
        """Requirement 6.4: Preventing duplicate message delivery upon re-processing or retransmission."""
        sender = ReliableSender(node_id="tx_dup_prevent", db_path=self.db_path)
        # First send
        res1 = sender.send_message("NO_DUP_01", self.dummy_symbols, language="ta", receiver=self.receiver)
        self.assertEqual(res1.status, DeliveryStatus.DELIVERED)
        self.assertEqual(len(self.receiver.received_messages), 1)

        # Second send with identical msg_id
        res2 = sender.send_message("NO_DUP_01", self.dummy_symbols, language="ta", receiver=self.receiver)
        self.assertEqual(res2.status, DeliveryStatus.DELIVERED)
        # Receiver delivered count must remain 1
        self.assertEqual(len(self.receiver.received_messages), 1)

        # Exactly 1 record in SQLite
        all_msgs = sender.store.get_all_messages()
        matching = [m for m in all_msgs if m.msg_id == "NO_DUP_01"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].status, DeliveryStatus.DELIVERED)
        sender.close()

    def test_handling_retry_limits_and_failed_delivery(self):
        """Requirement 6.5: Handling retry limits and failed delivery under link failure."""
        sender = ReliableSender(node_id="tx_retry_limit", db_path=self.db_path)
        # 100% packet loss simulating link failure
        res = sender.send_message(
            msg_id="FAIL_LIMIT_01",
            payload=self.dummy_symbols,
            language="hi",
            receiver=self.receiver,
            packet_loss_rate=1.0,
            max_retries=2,
        )
        self.assertEqual(res.status, DeliveryStatus.FAILED)
        self.assertEqual(res.retry_count, 2)
        self.assertIn("Failed to deliver after 2 retries", res.error)

        # SQLite status is FAILED
        stored = sender.get_stored_message("FAIL_LIMIT_01")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.status, DeliveryStatus.FAILED)
        self.assertEqual(stored.retry_count, 2)
        self.assertIn("Failed to deliver after 2 retries", stored.error)

        # On restart, FAILED messages must NOT be restored into active send queue
        sender.close()
        del sender
        sender_restart = ReliableSender(node_id="tx_retry_limit", db_path=self.db_path)
        self.assertEqual(sender_restart.queued_count, 0)
        sender_restart.close()


class TestMessagePriorityAndOrdering(unittest.TestCase):
    """Test emergency-aware message priority levels and FIFO preservation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "priority_test.db")
        self.receiver = ReliableReceiver(node_id="rx_prio_test")
        self.dummy_symbols = np.linspace(-1.0, 1.0, 32, dtype=np.float32)

    def tearDown(self):
        if hasattr(self, "receiver") and hasattr(self.receiver, "close"):
            self.receiver.close()
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_priority_enum_values_and_ordering(self):
        """Requirement 1: Verify EMERGENCY, HIGH, NORMAL, and LOW priority values and comparison."""
        self.assertEqual(MessagePriority.LOW.value, 10)
        self.assertEqual(MessagePriority.NORMAL.value, 20)
        self.assertEqual(MessagePriority.HIGH.value, 30)
        self.assertEqual(MessagePriority.EMERGENCY.value, 40)

        # Priority ordering: EMERGENCY > HIGH > NORMAL > LOW
        self.assertTrue(MessagePriority.EMERGENCY > MessagePriority.HIGH)
        self.assertTrue(MessagePriority.HIGH > MessagePriority.NORMAL)
        self.assertTrue(MessagePriority.NORMAL > MessagePriority.LOW)

        # String parsing
        self.assertEqual(MessagePriority.from_value("emergency"), MessagePriority.EMERGENCY)
        self.assertEqual(MessagePriority.from_value("EMERGENCY"), MessagePriority.EMERGENCY)
        self.assertEqual(MessagePriority.from_value("high"), MessagePriority.HIGH)
        self.assertEqual(MessagePriority.from_value("HIGH"), MessagePriority.HIGH)
        self.assertEqual(MessagePriority.from_value("normal"), MessagePriority.NORMAL)
        self.assertEqual(MessagePriority.from_value("NORMAL"), MessagePriority.NORMAL)
        self.assertEqual(MessagePriority.from_value("low"), MessagePriority.LOW)
        self.assertEqual(MessagePriority.from_value("LOW"), MessagePriority.LOW)

        # Integer parsing
        self.assertEqual(MessagePriority.from_value(40), MessagePriority.EMERGENCY)
        self.assertEqual(MessagePriority.from_value(30), MessagePriority.HIGH)
        self.assertEqual(MessagePriority.from_value(20), MessagePriority.NORMAL)
        self.assertEqual(MessagePriority.from_value(10), MessagePriority.LOW)
        self.assertEqual(MessagePriority.from_value(4), MessagePriority.EMERGENCY)
        self.assertEqual(MessagePriority.from_value(3), MessagePriority.HIGH)
        self.assertEqual(MessagePriority.from_value(2), MessagePriority.NORMAL)
        self.assertEqual(MessagePriority.from_value(1), MessagePriority.LOW)
        self.assertEqual(MessagePriority.from_value("40"), MessagePriority.EMERGENCY)
        self.assertEqual(MessagePriority.from_value("10"), MessagePriority.LOW)

        # Unknown value fallback
        self.assertEqual(MessagePriority.from_value("invalid_level"), MessagePriority.NORMAL)

    def test_queue_priority_ordering_highest_first(self):
        """Requirement 1 & 2: Queue messages with different priorities and flush in priority order."""
        self.receiver.set_availability(False)
        sender = ReliableSender(node_id="tx_prio_queue", db_path=self.db_path)

        # Enqueue in reverse priority order (LOW first, then NORMAL, HIGH, EMERGENCY)
        sender.send_message("MSG_LOW", self.dummy_symbols, priority=MessagePriority.LOW, receiver=self.receiver)
        sender.send_message("MSG_NORM", self.dummy_symbols, priority=MessagePriority.NORMAL, receiver=self.receiver)
        sender.send_message("MSG_HIGH", self.dummy_symbols, priority=MessagePriority.HIGH, receiver=self.receiver)
        sender.send_message("MSG_EMERG", self.dummy_symbols, priority=MessagePriority.EMERGENCY, receiver=self.receiver)

        self.assertEqual(sender.queued_count, 4)

        # In-memory queue order must be: EMERGENCY -> HIGH -> NORMAL -> LOW
        queued_ids = [m.msg_id for m in sender.queue]
        self.assertEqual(queued_ids, ["MSG_EMERG", "MSG_HIGH", "MSG_NORM", "MSG_LOW"])

        # Flush to receiver and verify delivery order
        self.receiver.set_availability(True)
        flushed = sender.flush_queue(receiver=self.receiver)
        self.assertEqual(len(flushed), 4)
        flushed_ids = [m.msg_id for m in flushed]
        self.assertEqual(flushed_ids, ["MSG_EMERG", "MSG_HIGH", "MSG_NORM", "MSG_LOW"])
        self.assertTrue(all(m.status == DeliveryStatus.DELIVERED for m in flushed))
        sender.close()

    def test_fifo_order_preserved_for_equal_priority(self):
        """Requirement 2: FIFO order is preserved among messages with identical priority."""
        self.receiver.set_availability(False)
        sender = ReliableSender(node_id="tx_fifo_prio", db_path=self.db_path)

        # Send 5 messages with same priority (EMERGENCY)
        for i in range(5):
            sender.send_message(f"EMERG_{i}", self.dummy_symbols, priority="emergency", receiver=self.receiver)

        queued_ids = [m.msg_id for m in sender.queue]
        self.assertEqual(queued_ids, [f"EMERG_{i}" for i in range(5)])

        # Flush and verify FIFO delivery
        self.receiver.set_availability(True)
        flushed = sender.flush_queue(receiver=self.receiver)
        delivered_ids = [m.msg_id for m in flushed]
        self.assertEqual(delivered_ids, [f"EMERG_{i}" for i in range(5)])
        sender.close()

    def test_mixed_priority_and_fifo_interleaving(self):
        """Requirement 1 & 2: Mixed arrivals maintain EMERGENCY > HIGH > NORMAL > LOW while strictly preserving FIFO within ties."""
        self.receiver.set_availability(False)
        sender = ReliableSender(node_id="tx_mixed", db_path=self.db_path)

        # Interleaved enqueue
        sender.send_message("M_NORM_1", self.dummy_symbols, priority="normal", receiver=self.receiver)
        sender.send_message("M_EMERG_1", self.dummy_symbols, priority="emergency", receiver=self.receiver)
        sender.send_message("M_HIGH_1", self.dummy_symbols, priority="high", receiver=self.receiver)
        sender.send_message("M_NORM_2", self.dummy_symbols, priority="normal", receiver=self.receiver)
        sender.send_message("M_EMERG_2", self.dummy_symbols, priority="emergency", receiver=self.receiver)
        sender.send_message("M_LOW_1", self.dummy_symbols, priority="low", receiver=self.receiver)

        expected_order = [
            "M_EMERG_1",
            "M_EMERG_2",
            "M_HIGH_1",
            "M_NORM_1",
            "M_NORM_2",
            "M_LOW_1",
        ]

        self.assertEqual([m.msg_id for m in sender.queue], expected_order)

        self.receiver.set_availability(True)
        flushed = sender.flush_queue(receiver=self.receiver)
        self.assertEqual([m.msg_id for m in flushed], expected_order)
        sender.close()

    def test_persistence_preserves_priority_and_fifo_across_restart(self):
        """Requirement 5: Priority and FIFO order survive application restart via SQLite."""
        self.receiver.set_availability(False)
        sender1 = ReliableSender(node_id="tx_prio_restart", db_path=self.db_path)

        sender1.send_message("R_NORM_1", self.dummy_symbols, priority="normal", receiver=self.receiver)
        sender1.send_message("R_EMERG_1", self.dummy_symbols, priority="emergency", receiver=self.receiver)
        sender1.send_message("R_HIGH_1", self.dummy_symbols, priority="high", receiver=self.receiver)
        sender1.send_message("R_NORM_2", self.dummy_symbols, priority="normal", receiver=self.receiver)
        sender1.send_message("R_EMERG_2", self.dummy_symbols, priority="emergency", receiver=self.receiver)

        sender1.close()
        del sender1

        # Restart session
        sender2 = ReliableSender(node_id="tx_prio_restart", db_path=self.db_path)
        self.assertEqual(sender2.queued_count, 5)

        expected_order = [
            "R_EMERG_1",
            "R_EMERG_2",
            "R_HIGH_1",
            "R_NORM_1",
            "R_NORM_2",
        ]
        self.assertEqual([m.msg_id for m in sender2.queue], expected_order)

        self.receiver.set_availability(True)
        flushed = sender2.flush_queue(receiver=self.receiver)
        self.assertEqual([m.msg_id for m in flushed], expected_order)
        self.assertTrue(all(m.status == DeliveryStatus.DELIVERED for m in flushed))
        sender2.close()

    def test_ack_and_retry_behavior_with_priority(self):
        """Requirement 9: ACK and ARQ retry behavior remain fully functional with priority and TTL."""
        sender = ReliableSender(node_id="tx_ack_retry_prio", db_path=self.db_path)

        # 1. Test clean ACK delivery with EMERGENCY priority
        res_ack = sender.send_message(
            msg_id="ACK_EMERG_01",
            payload=self.dummy_symbols,
            language="hi",
            receiver=self.receiver,
            priority=MessagePriority.EMERGENCY,
            ttl=30.0,
        )
        self.assertEqual(res_ack.status, DeliveryStatus.DELIVERED)
        self.assertEqual(res_ack.retry_count, 0)
        self.assertIsNotNone(res_ack.delivered_at)

        # 2. Test ARQ retransmission under packet loss with HIGH priority
        with patch("numpy.random.rand", side_effect=[0.05, 0.95, 0.95]):
            res_retry = sender.send_message(
                msg_id="RETRY_HIGH_01",
                payload=self.dummy_symbols,
                language="te",
                receiver=self.receiver,
                priority=MessagePriority.HIGH,
                ttl=30.0,
                packet_loss_rate=0.5,
                max_retries=3,
            )
            self.assertEqual(res_retry.status, DeliveryStatus.DELIVERED)
            self.assertEqual(res_retry.retry_count, 1)

        # 3. Test retry limit failure with LOW priority
        res_fail = sender.send_message(
            msg_id="FAIL_LOW_01",
            payload=self.dummy_symbols,
            language="ta",
            receiver=self.receiver,
            priority=MessagePriority.LOW,
            packet_loss_rate=1.0,
            max_retries=2,
        )
        self.assertEqual(res_fail.status, DeliveryStatus.FAILED)
        self.assertEqual(res_fail.retry_count, 2)
        self.assertIn("Failed to deliver after 2 retries", res_fail.error)

        sender.close()


class TestMessageTTLExpiration(unittest.TestCase):
    """Test configurable Time-To-Live (TTL) expiration and non-transmission of expired messages."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "ttl_test.db")
        self.receiver = ReliableReceiver(node_id="rx_ttl_test")
        self.dummy_symbols = np.linspace(-1.0, 1.0, 32, dtype=np.float32)

    def tearDown(self):
        if hasattr(self, "receiver") and hasattr(self.receiver, "close"):
            self.receiver.close()
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_ttl_dataclass_methods(self):
        """Requirement 3: Verify is_expired logic across timestamps and lifecycle states."""
        now = time.time()
        msg_no_ttl = Message("m_no_ttl", self.dummy_symbols, ttl=None, created_at=now)
        self.assertFalse(msg_no_ttl.is_expired(now + 1000))

        msg_positive = Message("m_pos", self.dummy_symbols, ttl=5.0, created_at=now)
        self.assertFalse(msg_positive.is_expired(now + 4.9))
        self.assertTrue(msg_positive.is_expired(now + 5.0))
        self.assertTrue(msg_positive.is_expired(now + 10.0))

        msg_zero = Message("m_zero", self.dummy_symbols, ttl=0.0, created_at=now)
        self.assertTrue(msg_zero.is_expired(now))

        msg_neg = Message("m_neg", self.dummy_symbols, ttl=-1.0, created_at=now)
        self.assertTrue(msg_neg.is_expired(now))

        # Delivered message never expires retroactively
        msg_delivered = Message("m_deliv", self.dummy_symbols, ttl=5.0, created_at=now, status=DeliveryStatus.DELIVERED)
        self.assertFalse(msg_delivered.is_expired(now + 100))

        # Mark expired updates status and error
        msg_positive.mark_expired("TTL timed out")
        self.assertEqual(msg_positive.status, DeliveryStatus.EXPIRED)
        self.assertEqual(msg_positive.error, "TTL timed out")

    def test_zero_or_negative_ttl_not_transmitted(self):
        """Requirement 4: Messages with zero or negative TTL are never transmitted."""
        sender = ReliableSender(node_id="tx_ttl_zero", db_path=self.db_path)

        res0 = sender.send_message("MSG_TTL_0", self.dummy_symbols, ttl=0.0, receiver=self.receiver)
        self.assertEqual(res0.status, DeliveryStatus.EXPIRED)
        self.assertIn("expired", res0.error.lower())

        res_neg = sender.send_message("MSG_TTL_NEG", self.dummy_symbols, ttl=-5.0, receiver=self.receiver)
        self.assertEqual(res_neg.status, DeliveryStatus.EXPIRED)

        # Receiver should not receive anything
        self.assertEqual(len(self.receiver.received_messages), 0)
        self.assertEqual(sender.queued_count, 0)

        # SQLite reflects EXPIRED
        stored0 = sender.get_stored_message("MSG_TTL_0")
        self.assertEqual(stored0.status, DeliveryStatus.EXPIRED)
        sender.close()

    def test_queued_message_expires_before_receiver_available(self):
        """Requirement 3 & 4: Stored message whose TTL expires while receiver is offline is not transmitted upon flush."""
        self.receiver.set_availability(False)
        sender = ReliableSender(node_id="tx_ttl_queue", db_path=self.db_path)

        # Enqueue one message with very short TTL, and one with long TTL
        sender.send_message("MSG_EXPIRE", self.dummy_symbols, ttl=0.08, priority="high", receiver=self.receiver)
        sender.send_message("MSG_SURVIVE", self.dummy_symbols, ttl=60.0, priority="normal", receiver=self.receiver)

        self.assertEqual(sender.queued_count, 2)

        # Wait for MSG_EXPIRE to expire
        time.sleep(0.12)

        # Receiver comes online -> flush queue
        self.receiver.set_availability(True)
        flushed = sender.flush_queue(receiver=self.receiver)

        self.assertEqual(len(flushed), 2)
        m_exp = [m for m in flushed if m.msg_id == "MSG_EXPIRE"][0]
        m_surv = [m for m in flushed if m.msg_id == "MSG_SURVIVE"][0]

        self.assertEqual(m_exp.status, DeliveryStatus.EXPIRED)
        self.assertEqual(m_surv.status, DeliveryStatus.DELIVERED)

        # Receiver must only receive the unexpired message
        self.assertEqual(len(self.receiver.received_messages), 1)
        self.assertEqual(self.receiver.received_messages[0].msg_id, "MSG_SURVIVE")

        # SQLite reflects accurate statuses
        self.assertEqual(sender.get_stored_message("MSG_EXPIRE").status, DeliveryStatus.EXPIRED)
        self.assertEqual(sender.get_stored_message("MSG_SURVIVE").status, DeliveryStatus.DELIVERED)
        sender.close()

    def test_ttl_expires_during_retransmission_attempts(self):
        """Requirement 4: Message expiring during ARQ retries aborts further transmissions and marks EXPIRED."""
        sender = ReliableSender(node_id="tx_ttl_arq", db_path=self.db_path)

        # Mock SDR transmit to simulate delay
        original_transmit = sender.sdr.transmit
        def slow_transmit(packet):
            time.sleep(0.04)
            return original_transmit(packet)

        with patch.object(sender.sdr, "transmit", side_effect=slow_transmit):
            # 100% loss with 0.06s TTL and 5 retries -> will expire around attempt 1 or 2
            res = sender.send_message(
                msg_id="TTL_ARQ_01",
                payload=self.dummy_symbols,
                receiver=self.receiver,
                packet_loss_rate=1.0,
                max_retries=5,
                ttl=0.06,
            )

        self.assertEqual(res.status, DeliveryStatus.EXPIRED)
        self.assertIn("TTL expired during retransmission attempts", res.error)
        self.assertTrue(res.retry_count < 5)  # Aborted before exhausting all 5 retries
        self.assertEqual(len(self.receiver.received_messages), 0)

        db_msg = sender.get_stored_message("TTL_ARQ_01")
        self.assertEqual(db_msg.status, DeliveryStatus.EXPIRED)
        sender.close()

    def test_ttl_expiration_while_offline_application_restart(self):
        """Requirement 5: Messages expiring while application is terminated/offline are marked EXPIRED on startup and not transmitted."""
        self.receiver.set_availability(False)
        sender1 = ReliableSender(node_id="tx_ttl_restart", db_path=self.db_path)

        # Short TTL
        sender1.send_message("OFFLINE_EXP_01", self.dummy_symbols, ttl=0.08, receiver=self.receiver)
        # Long TTL
        sender1.send_message("OFFLINE_SURV_01", self.dummy_symbols, ttl=60.0, receiver=self.receiver)

        self.assertEqual(sender1.queued_count, 2)
        sender1.close()
        del sender1

        # Wait for first message to expire during application shutdown
        time.sleep(0.12)

        # Restart sender
        sender2 = ReliableSender(node_id="tx_ttl_restart", db_path=self.db_path)

        # Only the surviving message should be restored into the active queue
        self.assertEqual(sender2.queued_count, 1)
        self.assertEqual(sender2.queue[0].msg_id, "OFFLINE_SURV_01")

        # Expired message is marked EXPIRED in SQLite
        db_exp = sender2.get_stored_message("OFFLINE_EXP_01")
        self.assertEqual(db_exp.status, DeliveryStatus.EXPIRED)
        self.assertIn("offline", db_exp.error.lower())

        # Flush transmits only the surviving message
        self.receiver.set_availability(True)
        flushed = sender2.flush_queue(receiver=self.receiver)
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0].msg_id, "OFFLINE_SURV_01")
        self.assertEqual(len(self.receiver.received_messages), 1)
        sender2.close()

    def test_expire_stale_messages_method(self):
        """Feature: expire_stale_messages scans and prunes expired items from queue and updates SQLite."""
        self.receiver.set_availability(False)
        sender = ReliableSender(node_id="tx_stale", db_path=self.db_path)

        sender.send_message("STALE_01", self.dummy_symbols, ttl=0.06, receiver=self.receiver)
        sender.send_message("VALID_01", self.dummy_symbols, ttl=60.0, receiver=self.receiver)
        self.assertEqual(sender.queued_count, 2)

        time.sleep(0.08)
        expired = sender.expire_stale_messages()
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].msg_id, "STALE_01")
        self.assertEqual(expired[0].status, DeliveryStatus.EXPIRED)

        self.assertEqual(sender.queued_count, 1)
        self.assertEqual(sender.queue[0].msg_id, "VALID_01")
        sender.close()

    def test_prune_expired_messages(self):
        """Feature: prune_expired deletes only expired records from SQLite."""
        sender = ReliableSender(node_id="tx_prune_exp", db_path=self.db_path)
        # 1 zero TTL (expired)
        sender.send_message("EXP_P_01", self.dummy_symbols, ttl=0.0, receiver=self.receiver)
        # 1 delivered
        sender.send_message("DEL_P_01", self.dummy_symbols, ttl=60.0, receiver=self.receiver)

        self.assertEqual(sender.store.count_by_status(DeliveryStatus.EXPIRED), 1)
        self.assertEqual(sender.store.count_by_status(DeliveryStatus.DELIVERED), 1)

        pruned = sender.prune_expired()
        self.assertEqual(pruned, 1)
        self.assertIsNone(sender.get_stored_message("EXP_P_01"))
        self.assertIsNotNone(sender.get_stored_message("DEL_P_01"))
        sender.close()


class TestPipelinePriorityAndTTL(unittest.TestCase):
    """Test ITantraPipeline end-to-end integration with priority and TTL."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "pipe_prio_test.db")
        self.pipeline = None

    def tearDown(self):
        if hasattr(self, "pipeline") and self.pipeline is not None:
            self.pipeline.close()
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_pipeline_send_emergency_message(self):
        """Verify pipeline sending EMERGENCY priority message."""
        self.pipeline = ITantraPipeline(db_path=self.db_path)
        res = self.pipeline.send_reliable_message(
            msg_id="PIPE_EMERG_01",
            text="ఆపదలో సహాయం కావాలి",
            language="te",
            priority="emergency",
            ttl=30.0,
        )
        self.assertTrue(res["is_delivered"])
        self.assertEqual(res["priority"], "emergency")
        self.assertEqual(res["ttl"], 30.0)
        self.assertEqual(res["receiver_delivered_messages"], 1)

    def test_pipeline_flush_respects_priority_and_ttl(self):
        """Verify pipeline queue flush respects priority and expires stale messages."""
        self.pipeline = ITantraPipeline(db_path=self.db_path)

        # Enqueue with receiver offline
        res_low = self.pipeline.send_reliable_message(
            msg_id="PIPE_LOW",
            text="सामान्य सूचना",
            language="hi",
            priority="low",
            ttl=60.0,
            receiver_available=False,
        )
        self.assertTrue(res_low["is_queued"])

        res_emerg = self.pipeline.send_reliable_message(
            msg_id="PIPE_EMERG",
            text="आपातकालीन चेतावनी",
            language="hi",
            priority="emergency",
            ttl=60.0,
            receiver_available=False,
        )
        self.assertTrue(res_emerg["is_queued"])

        res_stale = self.pipeline.send_reliable_message(
            msg_id="PIPE_STALE",
            text="पुराना संदेश",
            language="hi",
            priority="high",
            ttl=0.08,
            receiver_available=False,
        )
        self.assertTrue(res_stale["is_queued"])

        self.assertEqual(self.pipeline.reliable_sender.queued_count, 3)

        # Wait for PIPE_STALE to expire
        time.sleep(0.12)

        # Flush queue
        flushed = self.pipeline.flush_reliable_queue()
        self.assertEqual(len(flushed), 3)

        # First in flush order should be EMERGENCY (delivered)
        self.assertEqual(flushed[0]["msg_id"], "PIPE_EMERG")
        self.assertTrue(flushed[0]["is_delivered"])

        # Second in priority was HIGH (PIPE_STALE), which expired
        self.assertEqual(flushed[1]["msg_id"], "PIPE_STALE")
        self.assertTrue(flushed[1]["is_expired"])
        self.assertFalse(flushed[1]["is_delivered"])

        # Third is LOW (delivered)
        self.assertEqual(flushed[2]["msg_id"], "PIPE_LOW")
        self.assertTrue(flushed[2]["is_delivered"])

        # Receiver only received 2 messages (EMERGENCY and LOW, NOT STALE)
        self.assertEqual(len(self.pipeline.reliable_receiver.received_messages), 2)
        rx_ids = [m.msg_id for m in self.pipeline.reliable_receiver.received_messages]
        self.assertIn("PIPE_EMERG", rx_ids)
        self.assertIn("PIPE_LOW", rx_ids)
        self.assertNotIn("PIPE_STALE", rx_ids)


if __name__ == "__main__":
    unittest.main()

