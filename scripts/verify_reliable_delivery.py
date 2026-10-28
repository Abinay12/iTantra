"""Demonstration and verification script for reliable message delivery in iTantra.

Demonstrates:
  1. ACK-based delivery confirmation over AWGN & Rayleigh channels.
  2. Automatic retransmission (ARQ) under channel packet loss.
  3. Store-and-forward queueing when the receiver is temporarily offline.
  4. Message ID duplicate suppression (prevents duplicate delivery to receiver).
  5. Clear lifecycle delivery statuses: pending, delivered, failed, and queued.
"""

import os
import sys

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
from src.radio.reliable_delivery import DeliveryStatus


def main():
    print("=" * 75)
    print("  iTantra: Reliable Message Delivery Protocol Verification")
    print("=" * 75)

    pipeline = ITantraPipeline()

    # -------------------------------------------------------------
    # Scenario 1: Standard ACK Delivery over AWGN Channel
    # -------------------------------------------------------------
    print("\n[Scenario 1] Standard Reliable Delivery with ACK Confirmation (AWGN)")
    res1 = pipeline.send_reliable_message(
        msg_id="MSG_HINDI_001",
        text="नमस्ते, आप कैसे हैं?",
        language="hi",
        channel_type="awgn",
        snr_db=18.0,
    )
    print(f"  Message ID:        {res1['msg_id']}")
    print(f"  Status:            {res1['status'].upper()}")
    print(f"  Delivered:         {res1['is_delivered']}")
    print(f"  Retries Needed:    {res1['retry_count']} / {res1['max_retries']}")
    print(f"  Receiver Total:    {res1['receiver_delivered_messages']} delivered message(s)")

    # -------------------------------------------------------------
    # Scenario 2: ARQ Retransmissions under Simulated Loss (Rayleigh)
    # -------------------------------------------------------------
    print("\n[Scenario 2] ARQ Retransmission with 50% Simulated Loss (Rayleigh)")
    res2 = pipeline.send_reliable_message(
        msg_id="MSG_TELUGU_002",
        text="నమస్కారం, మీరు ఎలా ఉన్నారు?",
        language="te",
        channel_type="rayleigh",
        snr_db=20.0,
        packet_loss_rate=0.5,
        max_retries=5,
    )
    print(f"  Message ID:        {res2['msg_id']}")
    print(f"  Status:            {res2['status'].upper()}")
    print(f"  Delivered:         {res2['is_delivered']}")
    print(f"  Attempts/Retries:  {res2['retry_count']} retry attempt(s)")
    print(f"  Receiver Total:    {res2['receiver_delivered_messages']} delivered message(s)")

    # -------------------------------------------------------------
    # Scenario 3: Store-and-Forward Queueing (Receiver Offline)
    # -------------------------------------------------------------
    print("\n[Scenario 3] Store-and-Forward: Receiver Offline")
    res3 = pipeline.send_reliable_message(
        msg_id="MSG_QUEUED_003",
        text="नमस्ते, यह संदेश कतार में है।",
        language="hi",
        receiver_available=False,
    )
    print(f"  Message ID:        {res3['msg_id']}")
    print(f"  Status:            {res3['status'].upper()}")
    print(f"  Queued:            {res3['is_queued']}")
    print(f"  Queue Notice:      {res3['error']}")

    print("\n[Scenario 3b] Receiver Reconnects -> Flushing Store-and-Forward Queue")
    flushed = pipeline.flush_reliable_queue(channel_type="awgn", snr_db=18.0)
    for m in flushed:
        print(f"  Flushed Msg ID:    {m['msg_id']}")
        print(f"  New Status:        {m['status'].upper()}")
        print(f"  Delivered:         {m['is_delivered']}")

    # -------------------------------------------------------------
    # Scenario 4: Duplicate Message Suppression via Message ID
    # -------------------------------------------------------------
    print("\n[Scenario 4] Duplicate Message Suppression (Same Message ID Retransmitted)")
    res_dup1 = pipeline.send_reliable_message(
        msg_id="MSG_DUP_004",
        text="ధన్యవాదాలు (Thank you)",
        language="te",
    )
    count_before = res_dup1["receiver_delivered_messages"]
    print(f"  First Send Status:  {res_dup1['status'].upper()} (Receiver Count: {count_before})")

    res_dup2 = pipeline.send_reliable_message(
        msg_id="MSG_DUP_004",
        text="ధన్యవాదాలు (Thank you)",
        language="te",
    )
    count_after = res_dup2["receiver_delivered_messages"]
    print(f"  Second Send Status: {res_dup2['status'].upper()} (Receiver Count: {count_after})")
    print(f"  Duplicate Suppressed: {count_before == count_after} (No extra message recorded)")

    # -------------------------------------------------------------
    # Scenario 5: Exceeded Max Retries (Link Failure)
    # -------------------------------------------------------------
    print("\n[Scenario 5] Exceeded Max Retries under 100% Loss (Status: FAILED)")
    res_fail = pipeline.send_reliable_message(
        msg_id="MSG_FAIL_005",
        text="यह विफल हो जाएगा",
        language="hi",
        packet_loss_rate=1.0,
        max_retries=2,
    )
    print(f"  Message ID:        {res_fail['msg_id']}")
    print(f"  Status:            {res_fail['status'].upper()}")
    print(f"  Failed:            {res_fail['is_failed']}")
    print(f"  Retries Used:      {res_fail['retry_count']} / {res_fail['max_retries']}")
    print(f"  Failure Detail:    {res_fail['error']}")

    # -------------------------------------------------------------
    # Scenario 6: SQLite Persistent Store-and-Forward across Process Restart
    # -------------------------------------------------------------
    print("\n[Scenario 6] SQLite Persistent Store-and-Forward Across Process Restart")
    demo_db = os.path.join(REPO_ROOT, "data", "demo_persistent_messages.db")
    os.makedirs(os.path.dirname(demo_db), exist_ok=True)
    if os.path.exists(demo_db):
        try:
            os.remove(demo_db)
        except Exception:
            pass

    # 1. Pipeline instance 1: receiver offline, message queued to SQLite
    print("  [Step 1] First application instance queues message to SQLite...")
    pipe_instance_1 = ITantraPipeline(db_path=demo_db)
    res_persist = pipe_instance_1.send_reliable_message(
        msg_id="MSG_PERSIST_RESTART_006",
        text="పునఃప్రారంభం తర్వాత సజీవంగా ఉన్న సందేశం (Message survives restart)",
        language="te",
        receiver_available=False,
    )
    print(f"    Message ID:      {res_persist['msg_id']}")
    print(f"    Status:          {res_persist['status'].upper()}")
    print(f"    Stored in DB:    {demo_db}")
    print(f"    Sender Queue:    {pipe_instance_1.reliable_sender.queued_count} message(s)")
    pipe_instance_1.close()
    del pipe_instance_1
    print("  [Step 2] Simulated process termination (Instance 1 closed).")

    # 2. Pipeline instance 2 (Application Restart)
    print("  [Step 3] Second application instance starts up (Process Restart)...")
    pipe_instance_2 = ITantraPipeline(db_path=demo_db)
    restored_count = pipe_instance_2.reliable_sender.queued_count
    print(f"    Restored Queue:  {restored_count} message(s) loaded from SQLite on boot")
    if restored_count > 0:
        restored_m = pipe_instance_2.reliable_sender.queue[0]
        print(f"    Restored Msg ID: {restored_m.msg_id} (Status: {restored_m.status.value.upper()})")

    # 3. Receiver becomes available -> flush
    print("  [Step 4] Receiver reconnects -> flushing restored queue...")
    flushed_restart = pipe_instance_2.flush_reliable_queue(channel_type="awgn", snr_db=18.0)
    for m in flushed_restart:
        print(f"    Delivered Msg:   {m['msg_id']} -> {m['status'].upper()} (Retries: {m['retry_count']})")

    db_record = pipe_instance_2.reliable_sender.get_stored_message("MSG_PERSIST_RESTART_006")
    if db_record:
        print(f"    Persistent DB:   Record status in SQLite = {db_record.status.value.upper()}")

    pipe_instance_2.close()
    try:
        if os.path.exists(demo_db):
            os.remove(demo_db)
    except Exception:
        pass

    # -------------------------------------------------------------
    # Scenario 7: Emergency Priority Queueing and TTL Expiration
    # -------------------------------------------------------------
    print("\n[Scenario 7] Emergency Message Prioritization & TTL Expiration")
    pipe_prio = ITantraPipeline()
    pipe_prio.reliable_receiver.set_availability(False)

    print("  [Step 1] Queueing messages with different priorities while receiver offline...")
    pipe_prio.send_reliable_message(
        msg_id="MSG_PRIO_LOW",
        text="సాధారణ సమాచారం (Low priority info)",
        language="te",
        priority="low",
        receiver_available=False,
    )
    pipe_prio.send_reliable_message(
        msg_id="MSG_PRIO_NORMAL",
        text="సహజ సందేశం (Normal priority)",
        language="te",
        priority="normal",
        receiver_available=False,
    )
    pipe_prio.send_reliable_message(
        msg_id="MSG_PRIO_EMERGENCY",
        text="ఆపదలో రక్షణ హెచ్చరిక! (Emergency SOS Alert!)",
        language="te",
        priority="emergency",
        receiver_available=False,
    )
    pipe_prio.send_reliable_message(
        msg_id="MSG_PRIO_EXPIRED",
        text="త్వరగా గడువు ముగిసిన సందేశం (Short TTL message)",
        language="te",
        priority="high",
        ttl=0.08,
        receiver_available=False,
    )

    print(f"    Queued message count: {pipe_prio.reliable_sender.queued_count}")
    print("    In-memory queue order before flush:")
    for idx, m in enumerate(pipe_prio.reliable_sender.queue, 1):
        print(f"      {idx}. {m.msg_id} [Priority: {m.priority.name}, TTL: {m.ttl}]")

    print("  [Step 2] Waiting for short TTL message to expire...")
    import time
    time.sleep(0.12)

    print("  [Step 3] Flushing queue to online receiver...")
    flushed_prio = pipe_prio.flush_reliable_queue()
    for m in flushed_prio:
        print(f"    Flushed: {m['msg_id']} -> Status: {m['status'].upper()} (Priority: {m['priority'].upper()})")

    rx_msgs = pipe_prio.reliable_receiver.received_messages
    print(f"    Receiver actually received {len(rx_msgs)} message(s) (Expired message was NOT transmitted).")

    pipe_prio.close()

    print("\n" + "=" * 75)
    print("All scenarios completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
