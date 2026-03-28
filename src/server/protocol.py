"""
Protocol for NCA client-server communication.

Wire format: 4-byte big-endian length header + UTF-8 JSON payload.
Tensor data is base64-encoded float32 bytes.
Model data is the raw .pt file base64-encoded.
"""

import base64
import json
import socket
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

HEADER_SIZE = 4
MAX_MSG_SIZE = 100 * 1024 * 1024  # 100 MB - accommodates large model files


def tensor_to_b64(tensor: np.ndarray) -> str:
    """Encode numpy array as base64-encoded float32 bytes.

    Args:
        tensor: NumPy array to encode.

    Returns:
        Base64-encoded string of float32 bytes.
    """
    return base64.b64encode(tensor.astype(np.float32).tobytes()).decode("ascii")


def b64_to_tensor(b64: str, shape: list) -> np.ndarray:
    """Decode base64-encoded float32 array and reshape.

    Args:
        b64: Base64-encoded string of float32 bytes.
        shape: Target shape for reshaping.

    Returns:
        NumPy array with specified shape.
    """
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape)


def encode_message(msg: Dict[str, Any]) -> bytes:
    """Encode message dict as 4-byte length header + UTF-8 JSON payload.

    Args:
        msg: Dictionary to encode.

    Returns:
        Wire-format bytes ready for socket transmission.
    """
    payload = json.dumps(msg).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def decode_message(data: bytes) -> Dict[str, Any]:
    """Decode wire-format bytes back to message dictionary.

    Args:
        data: UTF-8 JSON wire payload (without length header).

    Returns:
        Decoded message dictionary.
    """
    return json.loads(data.decode("utf-8"))


def send_msg(sock: socket.socket, msg: Dict[str, Any]) -> None:
    """Send encoded message over socket.

    Args:
        sock: Connected socket.
        msg: Message dictionary to send.
    """
    sock.sendall(encode_message(msg))


def recv_msg(sock: socket.socket) -> Optional[Dict[str, Any]]:
    """Receive and decode message from socket.

    Args:
        sock: Connected socket.

    Returns:
        Decoded message dictionary, or None if disconnected.

    Raises:
        ValueError: If message size exceeds MAX_MSG_SIZE.
    """
    header = _recv_exact(sock, HEADER_SIZE)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    if length > MAX_MSG_SIZE:
        raise ValueError(f"Message too large: {length} bytes (max {MAX_MSG_SIZE})")
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return decode_message(payload)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Receive exactly n bytes from socket, blocking until complete.

    Args:
        sock: Connected socket.
        n: Number of bytes to receive.

    Returns:
        Bytes received, or None if disconnected before n bytes received.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# --- Message Builders ---


def build_init_msg(config: dict, target: np.ndarray) -> Dict[str, Any]:
    """Build init message with config and target tensor.

    Args:
        config: Training configuration dict.
        target: Target voxel grid with shape (D,H,W,C) external format.

    Returns:
        Init message dict (type='init', config and base64 target).
    """
    return {
        "type": "init",
        "config": config,
        "target": tensor_to_b64(target),
        "target_shape": list(target.shape),
    }


def parse_init_msg(msg: Dict[str, Any]) -> Tuple[dict, np.ndarray]:
    """Extract config and target from init message.

    Args:
        msg: Init message dict from client.

    Returns:
        Tuple (config dict, target numpy array).
    """
    config = msg["config"]
    target = b64_to_tensor(msg["target"], msg["target_shape"])
    return config, target


def build_state_msg(state: np.ndarray, epoch: int, loss: float = 0.0) -> Dict[str, Any]:
    """Build state broadcast message.

    Args:
        state: Current state [B, C, X, Y, Z] or subset [C, X, Y, Z].
        epoch: Training epoch or inference step count.
        loss: Current loss value.

    Returns:
        State message dict (type='state', base64-encoded state, epoch, loss).
    """
    return {
        "type": "state",
        "data": tensor_to_b64(state),
        "shape": list(state.shape),
        "epoch": epoch,
        "loss": loss,
    }


def parse_state_msg(msg: Dict[str, Any]) -> Tuple[np.ndarray, int, float]:
    """Extract state, epoch, loss from state broadcast message.

    Args:
        msg: State message dict from server.

    Returns:
        Tuple (state numpy array, epoch, loss).
    """
    state = b64_to_tensor(msg["data"], msg["shape"])
    return state, msg.get("epoch", 0), msg.get("loss", 0.0)


def build_run_model_msg(
    model_b64: str,
    phase_steps: int,
    broadcast_every: int,
) -> Dict[str, Any]:
    """Build run_model message for inference.

    Args:
        model_b64: Previously base64-encoded model file bytes.
        phase_steps: Forward pass steps per task phase.
        broadcast_every: Broadcast interval (steps).

    Returns:
        Run model message dict.
    """
    return {
        "type": "run_model",
        "model_b64": model_b64,
        "phase_steps": phase_steps,
        "broadcast_every": broadcast_every,
    }


def parse_run_model_msg(msg: Dict[str, Any]) -> Tuple[bytes, int, int]:
    """Extract model bytes and parameters from run_model message.

    Args:
        msg: Run model message dict.

    Returns:
        Tuple (raw model PyTorch checkpoint bytes, phase_steps, broadcast_every).
    """
    model_bytes = base64.b64decode(msg["model_b64"])
    return model_bytes, msg.get("phase_steps", 32), msg.get("broadcast_every", 4)


def build_stop_msg() -> Dict[str, Any]:
    return {"type": "stop"}


def build_pause_msg() -> Dict[str, Any]:
    return {"type": "pause"}


def build_resume_msg() -> Dict[str, Any]:
    return {"type": "resume"}


def build_ack_msg(message: str) -> Dict[str, Any]:
    return {"type": "ack", "message": message}


def build_error_msg(message: str) -> Dict[str, Any]:
    return {"type": "error", "message": message}


def build_schedule_msg(events: list) -> Dict[str, Any]:
    """Build schedule update message.

    Args:
        events: List of event dictionaries.

    Returns:
        Schedule message dict.
    """
    return {"type": "update_schedule", "events": events}


def parse_schedule_msg(msg: Dict[str, Any]) -> list:
    """Extract events from schedule message.

    Args:
        msg: Schedule update message dict.

    Returns:
        List of event dictionaries.
    """
    return msg["events"]
