"""
Server Package.

Exports networking, protocol, logging, and trainer components used by the
NCA TCP service layer.
"""

from .server import NCAServer
from .trainer import NCATrainer
from .protocol import (
    send_msg,
    recv_msg,
    tensor_to_b64,
    b64_to_tensor,
    encode_message,
    decode_message,
    build_init_msg,
    parse_init_msg,
    build_state_msg,
    parse_state_msg,
    build_stop_msg,
    build_pause_msg,
    build_resume_msg,
    build_ack_msg,
    build_error_msg,
)

__all__ = [
    "NCAServer",
    "NCATrainer",
    "send_msg",
    "recv_msg",
    "tensor_to_b64",
    "b64_to_tensor",
    "encode_message",
    "decode_message",
    "build_init_msg",
    "parse_init_msg",
    "build_state_msg",
    "parse_state_msg",
    "build_stop_msg",
    "build_pause_msg",
    "build_resume_msg",
    "build_ack_msg",
    "build_error_msg",
]
