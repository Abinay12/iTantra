"""Radio and network communication package for iTantra.

Contains packetization, framing, CRC integrity checks, and SDR interfaces
(SimulatedSDR loopback and HardwareSDRStub).
"""

from src.radio.packetizer import (
    Packetizer,
    RadioPacket,
    PREAMBLE_SYNC,
    PROTOCOL_VERSION,
    LANG_TO_ID,
    ID_TO_LANG,
)
from src.radio.sdr_interface import (
    SDRInterface,
    SimulatedSDR,
    HardwareSDRStub,
)
from src.radio.reliable_delivery import (
    DeliveryStatus,
    Message,
    MessagePriority,
    ReliablePacketizer,
    ReliableSender,
    ReliableReceiver,
)
from src.radio.message_store import MessageStore
from src.radio.network_transport import (
    TcpRadioReceiverServer,
    TcpRadioSenderClient,
    NetworkDeliveryResult,
    send_net_frame,
    recv_net_frame,
    get_local_ip_addresses,
    NET_MAGIC,
    MSG_TYPE_DATA,
    MSG_TYPE_ACK,
    MSG_TYPE_NACK,
)

__all__ = [
    "Packetizer",
    "RadioPacket",
    "PREAMBLE_SYNC",
    "PROTOCOL_VERSION",
    "LANG_TO_ID",
    "ID_TO_LANG",
    "SDRInterface",
    "SimulatedSDR",
    "HardwareSDRStub",
    "DeliveryStatus",
    "Message",
    "MessagePriority",
    "MessageStore",
    "ReliablePacketizer",
    "ReliableSender",
    "ReliableReceiver",
    "TcpRadioReceiverServer",
    "TcpRadioSenderClient",
    "NetworkDeliveryResult",
    "send_net_frame",
    "recv_net_frame",
    "get_local_ip_addresses",
    "NET_MAGIC",
    "MSG_TYPE_DATA",
    "MSG_TYPE_ACK",
    "MSG_TYPE_NACK",
]
