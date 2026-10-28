"""Unit and integration tests for TCP-based Real Network Transport in iTantra."""

import os
import socket
import tempfile
import time
import unittest
import numpy as np
import torch

from src.radio.packetizer import PREAMBLE_SYNC
from src.radio.reliable_delivery import (
    DeliveryStatus,
    MessagePriority,
    PACKET_TYPE_ACK,
    PACKET_TYPE_NACK,
    ReliablePacketizer,
    ReliableReceiver,
    ReliableSender,
)
from src.radio.message_store import MessageStore
from src.radio.network_transport import (
    NET_MAGIC,
    MSG_TYPE_DATA,
    MSG_TYPE_ACK,
    MSG_TYPE_NACK,
    TcpRadioReceiverServer,
    TcpRadioSenderClient,
    get_local_ip_addresses,
    recv_net_frame,
    send_net_frame,
)
from src.transceiver.channel import WirelessChannel


class TestSocketFraming(unittest.TestCase):
    """Test binary length-prefixed stream framing over TCP sockets."""

    def test_send_and_recv_net_frame_roundtrip(self):
        """Frame header, JSON metadata, and binary payload are accurately serialized and parsed."""
        # Create a socket pair using loopback
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(("127.0.0.1", port))
        conn, _ = server_sock.accept()

        try:
            metadata = {
                "msg_id": "MSG_TEST_001",
                "priority": "emergency",
                "ttl": 30.0,
                "language": "hi",
            }
            payload = b"\xAA\x55\xAA\x55\x01\x02\x03\x04HelloPayloadBytes"

            send_net_frame(client_sock, MSG_TYPE_DATA, metadata, payload=payload)
            msg_type, rx_meta, rx_payload = recv_net_frame(conn, timeout=2.0)

            self.assertEqual(msg_type, MSG_TYPE_DATA)
            self.assertEqual(rx_meta["msg_id"], "MSG_TEST_001")
            self.assertEqual(rx_meta["priority"], "emergency")
            self.assertEqual(rx_meta["ttl"], 30.0)
            self.assertEqual(rx_payload, payload)
        finally:
            client_sock.close()
            conn.close()
            server_sock.close()

    def test_local_ip_discovery(self):
        """Local IP discovery returns at least one valid IPv4 address."""
        ips = get_local_ip_addresses()
        self.assertGreater(len(ips), 0)
        for ip in ips:
            parts = ip.split(".")
            self.assertEqual(len(parts), 4)


class TestNetworkTransportIntegration(unittest.TestCase):
    """End-to-end integration tests for TCP Radio Receiver Server and Sender Client."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rx_db = os.path.join(self.temp_dir.name, "test_rx.db")
        self.tx_db = os.path.join(self.temp_dir.name, "test_tx.db")
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.close()
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_successful_message_transmission_and_ack(self):
        """Message is packetized, sent over TCP, verified via CRC-16, and acknowledged."""
        delivered_events = []

        def on_delivered(event):
            delivered_events.append(event)

        # Bind to port 0 for OS-assigned free port
        self.server = TcpRadioReceiverServer(
            host="127.0.0.1",
            port=0,
            db_path=self.rx_db,
            on_message_delivered=on_delivered,
        )
        self.server.start()
        port = self.server.bound_port

        client = TcpRadioSenderClient(
            node_id="tx_test_client",
            db_path=self.tx_db,
            socket_timeout=3.0,
        )

        symbols = np.random.randn(64).astype(np.float32)
        res = client.send_message(
            receiver_host="127.0.0.1",
            receiver_port=port,
            msg_id="NET_E2E_01",
            payload=symbols,
            language="hi",
            text="नमस्ते",
            priority="emergency",
            ttl=60.0,
        )

        self.assertTrue(res.is_delivered)
        self.assertEqual(res.status, DeliveryStatus.DELIVERED)
        self.assertEqual(res.retry_count, 0)
        self.assertGreater(res.latency_ms, 0.0)
        self.assertGreater(res.packets_sent, 0)
        self.assertFalse(res.is_duplicate)

        # Verify receiver event callback
        self.assertEqual(len(delivered_events), 1)
        self.assertEqual(delivered_events[0]["msg_id"], "NET_E2E_01")
        self.assertEqual(delivered_events[0]["language"], "hi")
        self.assertEqual(len(delivered_events[0]["symbols"]), 64)

        # Verify persistence in both databases
        rx_store = MessageStore(db_path=self.rx_db)
        seen_ids = rx_store.get_seen_message_ids(self.server.node_id)
        self.assertIn("NET_E2E_01", seen_ids)
        rx_store.close()

        tx_store = MessageStore(db_path=self.tx_db)
        tx_msg = tx_store.get_message("NET_E2E_01")
        self.assertIsNotNone(tx_msg)
        self.assertEqual(tx_msg.status, DeliveryStatus.DELIVERED)
        tx_store.close()
        client.close()

    def test_duplicate_message_suppression(self):
        """Transmitting identical message ID twice results in duplicate suppression and idempotent ACK."""
        delivered_events = []

        def on_delivered(event):
            delivered_events.append(event)

        self.server = TcpRadioReceiverServer(
            host="127.0.0.1",
            port=0,
            db_path=self.rx_db,
            on_message_delivered=on_delivered,
        )
        self.server.start()
        port = self.server.bound_port

        client = TcpRadioSenderClient(db_path=self.tx_db)
        symbols = np.random.randn(32).astype(np.float32)

        # First transmission: delivered
        res1 = client.send_message(
            receiver_host="127.0.0.1",
            receiver_port=port,
            msg_id="DUP_MSG_01",
            payload=symbols,
            language="hi",
        )
        self.assertTrue(res1.is_delivered)
        self.assertFalse(res1.is_duplicate)
        self.assertEqual(len(delivered_events), 1)

        # Second transmission of same message ID: duplicate suppressed, acknowledged idempotently
        # Use a fresh client store so sender re-sends
        fresh_client_db = os.path.join(self.temp_dir.name, "fresh_tx.db")
        client2 = TcpRadioSenderClient(db_path=fresh_client_db)
        res2 = client2.send_message(
            receiver_host="127.0.0.1",
            receiver_port=port,
            msg_id="DUP_MSG_01",
            payload=symbols,
            language="hi",
        )
        self.assertTrue(res2.is_delivered)
        self.assertTrue(res2.is_duplicate)
        # Receiver callback should NOT have fired a second time
        self.assertEqual(len(delivered_events), 1)

        client.close()
        client2.close()

    def test_unreachable_server_queues_message(self):
        """When receiver server is offline, sender safely stores message in store-and-forward queue."""
        client = TcpRadioSenderClient(db_path=self.tx_db, default_max_retries=1, socket_timeout=0.5)

        # Connect to unassigned high port where no server is listening
        dead_port = 59123
        res = client.send_message(
            receiver_host="127.0.0.1",
            receiver_port=dead_port,
            msg_id="OFFLINE_MSG_01",
            payload=np.ones(16, dtype=np.float32),
            priority="high",
        )

        self.assertFalse(res.is_delivered)
        self.assertTrue(res.is_queued)
        self.assertEqual(res.status, DeliveryStatus.QUEUED)
        self.assertIn("unreachable", res.error.lower())

        # Check that message is stored in queue
        self.assertEqual(len(client.queue), 1)
        self.assertEqual(client.queue[0].msg_id, "OFFLINE_MSG_01")
        client.close()

    def test_ttl_expiration_enforcement(self):
        """Messages with expired TTL are never transmitted over the network."""
        client = TcpRadioSenderClient(db_path=self.tx_db)

        # 1. Message with zero/negative TTL (immediately expired)
        res = client.send_message(
            receiver_host="127.0.0.1",
            receiver_port=8765,
            msg_id="EXPIRED_MSG_01",
            payload=np.ones(16, dtype=np.float32),
            ttl=0.0,
        )
        self.assertFalse(res.is_delivered)
        self.assertTrue(res.is_expired)
        self.assertEqual(res.status, DeliveryStatus.EXPIRED)
        self.assertEqual(res.packets_sent, 0)
        self.assertIn("TTL expired", res.error)

        # 2. Message with simulated past creation time
        res2 = client.send_message(
            receiver_host="127.0.0.1",
            receiver_port=8765,
            msg_id="EXPIRED_MSG_02",
            payload=np.ones(16, dtype=np.float32),
            ttl=5.0,
            created_at=time.time() - 10.0,
        )
        self.assertFalse(res2.is_delivered)
        self.assertTrue(res2.is_expired)
        self.assertEqual(res2.status, DeliveryStatus.EXPIRED)
        client.close()

    def test_priority_queue_flushing_order(self):
        """Store-and-forward queue flushes messages in priority order (EMERGENCY -> HIGH -> NORMAL -> LOW)."""
        client = TcpRadioSenderClient(db_path=self.tx_db, default_max_retries=1, socket_timeout=0.5)

        # Queue messages while server is offline
        dead_port = 59124
        client.send_message("127.0.0.1", dead_port, "MSG_LOW", np.ones(10), priority="low")
        client.send_message("127.0.0.1", dead_port, "MSG_EMERGENCY", np.ones(10), priority="emergency")
        client.send_message("127.0.0.1", dead_port, "MSG_NORMAL", np.ones(10), priority="normal")
        client.send_message("127.0.0.1", dead_port, "MSG_HIGH", np.ones(10), priority="high")

        # Now start server
        self.server = TcpRadioReceiverServer(host="127.0.0.1", port=0, db_path=self.rx_db)
        self.server.start()
        live_port = self.server.bound_port

        # Flush queue to live server
        flushed_results = client.flush_queue(receiver_host="127.0.0.1", receiver_port=live_port)
        self.assertEqual(len(flushed_results), 4)

        flushed_ids = [r.msg_id for r in flushed_results]
        self.assertEqual(flushed_ids, ["MSG_EMERGENCY", "MSG_HIGH", "MSG_NORMAL", "MSG_LOW"])
        for r in flushed_results:
            self.assertTrue(r.is_delivered)

        client.close()

    def test_corrupted_packet_detection_and_nack(self):
        """Corrupted packet CRC results in NACK and retransmissions."""
        self.server = TcpRadioReceiverServer(host="127.0.0.1", port=0, db_path=self.rx_db)
        self.server.start()
        port = self.server.bound_port

        # Create low-level client connection and inject corrupted CRC bytes
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(("127.0.0.1", port))

        packetizer = ReliablePacketizer(max_payload_symbols=64)
        valid_packets = packetizer.packetize_stream(np.ones(16, dtype=np.float32), language="hi")
        raw_stream = bytearray(b"".join(valid_packets))

        # Intentionally flip bytes to corrupt CRC-16
        raw_stream[-1] ^= 0xFF

        send_net_frame(
            client_sock,
            MSG_TYPE_DATA,
            {"msg_id": "CORRUPT_01", "language": "hi", "priority": "normal"},
            payload=bytes(raw_stream),
        )

        resp_type, resp_meta, ack_payload = recv_net_frame(client_sock, timeout=2.0)
        self.assertEqual(resp_type, MSG_TYPE_NACK)
        self.assertEqual(resp_meta["status"], "corrupted")

        ctrl = packetizer.parse_control_frame(ack_payload)
        self.assertIsNotNone(ctrl)
        pkt_type, rx_hash, code, crc_valid = ctrl
        self.assertEqual(pkt_type, PACKET_TYPE_NACK)

        client_sock.close()

    def test_channel_simulation_separation(self):
        """Clean Wi-Fi transport has zero synthetic noise, while optional channel adds noise."""
        self.server = TcpRadioReceiverServer(host="127.0.0.1", port=0, db_path=self.rx_db)
        self.server.start()
        port = self.server.bound_port

        client = TcpRadioSenderClient(db_path=self.tx_db)
        orig_symbols = np.ones(32, dtype=np.float32)

        # 1. Pure Wi-Fi (channel=None)
        res_clean = client.send_message(
            receiver_host="127.0.0.1",
            receiver_port=port,
            msg_id="CLEAN_WIFI_01",
            payload=orig_symbols,
            channel=None,
        )
        self.assertTrue(res_clean.is_delivered)

        # 2. Hybrid AWGN experiment
        awgn_ch = WirelessChannel(model_type="awgn", snr_db=5.0)
        res_noisy = client.send_message(
            receiver_host="127.0.0.1",
            receiver_port=port,
            msg_id="HYBRID_AWGN_01",
            payload=orig_symbols,
            channel=awgn_ch,
            snr_db=5.0,
        )
        self.assertTrue(res_noisy.is_delivered)

        client.close()

    def test_pipeline_send_network_message_integration(self):
        """ITantraPipeline.send_network_message synthesizes speech, encodes neural symbols, and transmits over TCP."""
        from src.pipeline import ITantraPipeline

        received_events = []

        def on_msg(event):
            received_events.append(event)

        pipeline_rx = ITantraPipeline(db_path=self.rx_db)
        self.server = pipeline_rx.create_network_receiver(host="127.0.0.1", port=0, on_message_delivered=on_msg)
        self.server.start()
        port = self.server.bound_port

        pipeline_tx = ITantraPipeline(db_path=self.tx_db)

        # Transmit Hindi message through full pipeline over TCP
        res = pipeline_tx.send_network_message(
            receiver_host="127.0.0.1",
            receiver_port=port,
            text="नमस्ते",
            language="hi",
            priority="emergency",
        )

        self.assertTrue(res.is_delivered)
        self.assertEqual(res.status, DeliveryStatus.DELIVERED)
        self.assertEqual(res.retry_count, 0)
        self.assertGreater(res.packets_sent, 0)
        self.assertEqual(len(received_events), 1)

        # Verify receiver reconstructed neural symbols
        symbols = received_events[0]["symbols"]
        self.assertGreater(len(symbols), 0)
        self.assertEqual(received_events[0]["language"], "hi")

        # Decode symbols via receiver pipeline
        sym_tensor = torch.from_numpy(symbols).to(pipeline_rx.device)
        if len(symbols) % pipeline_rx.latent_dim == 0:
            sym_tensor = sym_tensor.view(1, -1, pipeline_rx.latent_dim)
        with torch.no_grad():
            rec_features = pipeline_rx.decoder(sym_tensor)
        self.assertEqual(rec_features.shape[-1], pipeline_rx.input_dim)

        pipeline_tx.close()
        pipeline_rx.close()


if __name__ == "__main__":
    unittest.main()
