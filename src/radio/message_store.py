"""Persistent SQLite storage for store-and-forward message queuing in iTantra.

Provides durable persistence for:
  1. Queued messages when receiver is offline/unavailable.
  2. Delivery status tracking: PENDING, DELIVERED, FAILED, and QUEUED.
  3. Safe restoration of queued messages on application restart.
  4. Duplicate message suppression and idempotency via PRIMARY KEY (msg_id).
  5. Updating delivery status on successful acknowledgement (ACK).
"""

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch

from src.radio.reliable_delivery import DeliveryStatus, Message, MessagePriority


class MessageStore:
    """SQLite-backed persistent store for reliable message delivery and queuing."""

    def __init__(self, db_path: str = ":memory:"):
        """Initialize SQLite message store.

        Args:
            db_path: SQLite database file path or ':memory:' for in-memory database.
        """
        self.db_path = db_path
        if self.db_path != ":memory:" and not self.db_path.startswith("file:"):
            parent_dir = os.path.dirname(os.path.abspath(self.db_path))
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=15.0)
        self.conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:" and not self.db_path.startswith("file:"):
            try:
                self.conn.execute("PRAGMA journal_mode=WAL;")
                self.conn.execute("PRAGMA synchronous=NORMAL;")
            except Exception:
                pass
        self._init_db()

    def _init_db(self):
        """Create tables and indexes if they do not exist, and perform schema migrations."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reliable_messages (
                    msg_id TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    payload_type TEXT NOT NULL DEFAULT 'symbols',
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    created_at REAL NOT NULL,
                    delivered_at REAL,
                    error TEXT,
                    priority INTEGER NOT NULL DEFAULT 20,
                    ttl REAL DEFAULT NULL
                )
                """
            )
            # Automatic schema migration for existing databases
            try:
                cur = self.conn.execute("PRAGMA table_info(reliable_messages)")
                cols = {row["name"] for row in cur.fetchall()}
                if "priority" not in cols:
                    self.conn.execute("ALTER TABLE reliable_messages ADD COLUMN priority INTEGER NOT NULL DEFAULT 20")
                if "ttl" not in cols:
                    self.conn.execute("ALTER TABLE reliable_messages ADD COLUMN ttl REAL DEFAULT NULL")
            except Exception:
                pass

            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_msg_status ON reliable_messages(status)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_msg_priority ON reliable_messages(priority DESC, created_at ASC)"
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS received_messages (
                    msg_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    received_at REAL NOT NULL,
                    language TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    payload_type TEXT NOT NULL DEFAULT 'symbols'
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rx_node ON received_messages(node_id)"
            )

    def _serialize_payload(self, payload: Any) -> tuple[bytes, str]:
        """Convert payload into bytes and type tag."""
        if payload is None:
            return b"", "none"
        elif isinstance(payload, torch.Tensor):
            return payload.detach().cpu().numpy().astype(np.float32).tobytes(), "symbols"
        elif isinstance(payload, np.ndarray):
            return payload.astype(np.float32).tobytes(), "symbols"
        elif isinstance(payload, str):
            return payload.encode("utf-8"), "text"
        elif isinstance(payload, (bytes, bytearray)):
            return bytes(payload), "bytes"
        else:
            return np.asarray(payload, dtype=np.float32).tobytes(), "symbols"

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Construct Message instance from database row."""
        payload_type = row["payload_type"] if "payload_type" in row.keys() else "symbols"
        raw_blob = row["payload"]
        if payload_type == "none":
            payload = None
        elif payload_type == "text":
            payload = raw_blob.decode("utf-8", errors="replace")
        elif payload_type == "bytes":
            payload = raw_blob
        else:
            payload = np.frombuffer(raw_blob, dtype=np.float32).copy()

        priority_raw = row["priority"] if "priority" in row.keys() else 20
        priority = MessagePriority.from_value(priority_raw)
        ttl = row["ttl"] if "ttl" in row.keys() else None
        seq = row["rowid"] if "rowid" in row.keys() else 0

        return Message(
            msg_id=row["msg_id"],
            payload=payload,
            language=row["language"],
            status=DeliveryStatus(row["status"]),
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            created_at=row["created_at"],
            delivered_at=row["delivered_at"],
            error=row["error"],
            priority=priority,
            ttl=ttl,
            seq=seq,
        )

    def save_message(self, msg: Message) -> bool:
        """Save or update message in SQLite store.

        Prevents duplicate re-processing from overwriting an already delivered message.

        Returns:
            True if inserted or updated, False if skipped because already delivered.
        """
        existing = self.get_message(msg.msg_id)
        if existing is not None:
            if existing.status == DeliveryStatus.DELIVERED and msg.status != DeliveryStatus.DELIVERED:
                # Do not revert an already delivered message back to pending/queued
                return False

        payload_blob, payload_type = self._serialize_payload(msg.payload)
        priority_val = int(msg.priority.value if hasattr(msg.priority, "value") else msg.priority)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO reliable_messages (
                    msg_id, payload, payload_type, language, status,
                    retry_count, max_retries, created_at, delivered_at, error,
                    priority, ttl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(msg_id) DO UPDATE SET
                    payload = excluded.payload,
                    payload_type = excluded.payload_type,
                    language = excluded.language,
                    status = excluded.status,
                    retry_count = excluded.retry_count,
                    max_retries = excluded.max_retries,
                    delivered_at = excluded.delivered_at,
                    error = excluded.error,
                    priority = excluded.priority,
                    ttl = excluded.ttl
                """,
                (
                    msg.msg_id,
                    payload_blob,
                    payload_type,
                    msg.language,
                    msg.status.value,
                    msg.retry_count,
                    msg.max_retries,
                    msg.created_at,
                    msg.delivered_at,
                    msg.error,
                    priority_val,
                    msg.ttl,
                ),
            )
        return True

    def mark_delivered(self, msg_id: str, delivered_at: Optional[float] = None) -> bool:
        """Mark message as delivered in SQLite store."""
        deliv_time = time.time() if delivered_at is None else delivered_at
        with self.conn:
            cur = self.conn.execute(
                """
                UPDATE reliable_messages
                SET status = ?, delivered_at = ?, error = NULL
                WHERE msg_id = ?
                """,
                (DeliveryStatus.DELIVERED.value, deliv_time, msg_id),
            )
            return cur.rowcount > 0

    def mark_failed(self, msg_id: str, error: str, retry_count: Optional[int] = None) -> bool:
        """Mark message as failed in SQLite store."""
        with self.conn:
            if retry_count is not None:
                cur = self.conn.execute(
                    """
                    UPDATE reliable_messages
                    SET status = ?, error = ?, retry_count = ?
                    WHERE msg_id = ?
                    """,
                    (DeliveryStatus.FAILED.value, error, retry_count, msg_id),
                )
            else:
                cur = self.conn.execute(
                    """
                    UPDATE reliable_messages
                    SET status = ?, error = ?
                    WHERE msg_id = ?
                    """,
                    (DeliveryStatus.FAILED.value, error, msg_id),
                )
            return cur.rowcount > 0

    def mark_queued(self, msg_id: str, error: Optional[str] = None) -> bool:
        """Mark message as queued in SQLite store."""
        err_msg = error or "Receiver unavailable: queued for store-and-forward delivery"
        with self.conn:
            cur = self.conn.execute(
                """
                UPDATE reliable_messages
                SET status = ?, error = ?
                WHERE msg_id = ?
                """,
                (DeliveryStatus.QUEUED.value, err_msg, msg_id),
            )
            return cur.rowcount > 0

    def mark_expired(self, msg_id: str, error: Optional[str] = None) -> bool:
        """Mark message as expired in SQLite store (unless already delivered)."""
        err_msg = error or "Message TTL expired before delivery"
        with self.conn:
            cur = self.conn.execute(
                """
                UPDATE reliable_messages
                SET status = ?, error = ?
                WHERE msg_id = ? AND status != ?
                """,
                (DeliveryStatus.EXPIRED.value, err_msg, msg_id, DeliveryStatus.DELIVERED.value),
            )
            return cur.rowcount > 0

    def update_retry_count(self, msg_id: str, retry_count: int) -> bool:
        """Update retry attempt counter for message."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE reliable_messages SET retry_count = ? WHERE msg_id = ?",
                (retry_count, msg_id),
            )
            return cur.rowcount > 0

    def get_message(self, msg_id: str) -> Optional[Message]:
        """Fetch single message by msg_id."""
        cur = self.conn.execute(
            "SELECT rowid, * FROM reliable_messages WHERE msg_id = ?", (msg_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_message(row)

    def get_pending_and_queued_messages(self) -> List[Message]:
        """Fetch all messages awaiting delivery (status in PENDING, QUEUED) ordered by priority DESC, created_at ASC, rowid ASC."""
        cur = self.conn.execute(
            """
            SELECT rowid, * FROM reliable_messages
            WHERE status IN (?, ?)
            ORDER BY priority DESC, created_at ASC, rowid ASC
            """,
            (DeliveryStatus.PENDING.value, DeliveryStatus.QUEUED.value),
        )
        return [self._row_to_message(row) for row in cur.fetchall()]

    def get_messages_by_status(self, status: Union[str, DeliveryStatus]) -> List[Message]:
        """Fetch all messages matching a specific status."""
        stat_val = status.value if isinstance(status, DeliveryStatus) else str(status)
        cur = self.conn.execute(
            "SELECT rowid, * FROM reliable_messages WHERE status = ? ORDER BY created_at ASC, rowid ASC",
            (stat_val,),
        )
        return [self._row_to_message(row) for row in cur.fetchall()]

    def get_all_messages(self) -> List[Message]:
        """Fetch all messages in the store."""
        cur = self.conn.execute(
            "SELECT rowid, * FROM reliable_messages ORDER BY created_at ASC, rowid ASC"
        )
        return [self._row_to_message(row) for row in cur.fetchall()]

    def delete_message(self, msg_id: str) -> bool:
        """Remove a message from SQLite store."""
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM reliable_messages WHERE msg_id = ?", (msg_id,)
            )
            return cur.rowcount > 0

    def delete_delivered(self) -> int:
        """Remove all delivered messages from SQLite store. Returns count of deleted rows."""
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM reliable_messages WHERE status = ?",
                (DeliveryStatus.DELIVERED.value,),
            )
            return cur.rowcount

    def delete_expired(self) -> int:
        """Remove all expired messages from SQLite store. Returns count of deleted rows."""
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM reliable_messages WHERE status = ?",
                (DeliveryStatus.EXPIRED.value,),
            )
            return cur.rowcount

    def get_queued_messages(self) -> List[Message]:
        """Fetch all messages with status QUEUED."""
        return self.get_messages_by_status(DeliveryStatus.QUEUED)

    def get_pending_messages(self) -> List[Message]:
        """Fetch all messages with status PENDING."""
        return self.get_messages_by_status(DeliveryStatus.PENDING)

    def get_delivered_messages(self) -> List[Message]:
        """Fetch all messages with status DELIVERED."""
        return self.get_messages_by_status(DeliveryStatus.DELIVERED)

    def get_failed_messages(self) -> List[Message]:
        """Fetch all messages with status FAILED."""
        return self.get_messages_by_status(DeliveryStatus.FAILED)

    def get_expired_messages(self) -> List[Message]:
        """Fetch all messages with status EXPIRED."""
        return self.get_messages_by_status(DeliveryStatus.EXPIRED)

    def record_received_message(
        self,
        msg_id: str,
        node_id: str,
        payload: Any = None,
        language: str = "hi",
    ) -> bool:
        """Record received message for duplicate suppression across receiver restarts."""
        payload_blob, payload_type = self._serialize_payload(payload)
        with self.conn:
            try:
                self.conn.execute(
                    """
                    INSERT INTO received_messages (msg_id, node_id, received_at, language, payload, payload_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (msg_id, node_id, time.time(), language, payload_blob, payload_type),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def get_seen_message_ids(self, node_id: Optional[str] = None) -> Set[str]:
        """Retrieve set of seen message IDs for duplicate detection."""
        if node_id is not None:
            cur = self.conn.execute(
                "SELECT msg_id FROM received_messages WHERE node_id = ?", (node_id,)
            )
        else:
            cur = self.conn.execute("SELECT msg_id FROM received_messages")
        return {row[0] for row in cur.fetchall()}

    def clear_received(self, node_id: Optional[str] = None):
        """Clear received messages history."""
        with self.conn:
            if node_id is not None:
                self.conn.execute(
                    "DELETE FROM received_messages WHERE node_id = ?", (node_id,)
                )
            else:
                self.conn.execute("DELETE FROM received_messages")

    def clear_queued(self):
        """Remove all pending and queued messages from the store."""
        with self.conn:
            self.conn.execute(
                "DELETE FROM reliable_messages WHERE status IN (?, ?)",
                (DeliveryStatus.PENDING.value, DeliveryStatus.QUEUED.value),
            )

    def clear_all(self):
        """Remove all messages from the store."""
        with self.conn:
            self.conn.execute("DELETE FROM reliable_messages")

    def count_by_status(self, status: Union[str, DeliveryStatus]) -> int:
        """Count messages with given status."""
        stat_val = status.value if isinstance(status, DeliveryStatus) else str(status)
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM reliable_messages WHERE status = ?", (stat_val,)
        )
        return cur.fetchone()[0]

    def total_count(self) -> int:
        """Count total messages in store."""
        cur = self.conn.execute("SELECT COUNT(*) FROM reliable_messages")
        return cur.fetchone()[0]

    def close(self):
        """Close SQLite database connection."""
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
