"""
Protocol for NCA client-server communication.

Wire format: 4-byte big-endian length header + UTF-8 JSON payload.
Tensor data is base64-encoded float32 bytes.
"""
import json
import socket
import struct
import base64
import numpy as np
from typing import Any, Dict, Optional, Tuple

HEADER_SIZE = 4
MAX_MSG_SIZE = 50 * 1024 * 1024  # 50 MB safety cap

def tensor_to_b64(tensor: np.ndarray) -> str:
    """Encode a NumPy array as a base64 string (float32)."""
    return base64.b64encode(tensor.astype(np.float32).tobytes()).decode("ascii")

def b64_to_tensor(b64: str, shape: list) -> np.ndarray:
    """Decode a base64 string back to a float32 NumPy array."""
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape)

def encode_message(msg: Dict[str, Any]) -> bytes:
    """Serialize a dict to length-prefixed JSON bytes."""
    payload = json.dumps(msg).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload

def decode_message(data: bytes) -> Dict[str, Any]:
    """Deserialize JSON bytes to a dict."""
    return json.loads(data.decode("utf-8"))

def send_msg(sock: socket.socket, msg: Dict[str, Any]) -> None:
    """Send a length-prefixed JSON message over a socket."""
    sock.sendall(encode_message(msg))

def recv_msg(sock: socket.socket) -> Optional[Dict[str, Any]]:
    """Receive one length-prefixed JSON message (blocking)."""
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
    """Read exactly n bytes or return None on disconnect."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

def build_init_msg(config: dict, target: np.ndarray) -> Dict[str, Any]:
    """Pack an init message with config and target tensor.

    Args:
        config: dict with keys cell, perception, update, grid, training.
        target: (D, H, W, C) float32 numpy array — channels-last voxel data.
    """
    return {
        "type": "init",
        "config": config,
        "target": tensor_to_b64(target),
        "target_shape": list(target.shape),
    }

def parse_init_msg(msg: Dict[str, Any]) -> Tuple[dict, np.ndarray]:
    """Unpack an init message.

    Returns:
        config: dict with cell/perception/update/grid/training settings.
        target: (D, H, W, C) float32 numpy array.
    """
    config = msg["config"]
    target = b64_to_tensor(msg["target"], msg["target_shape"])
    return config, target

def build_state_msg(
    state: np.ndarray, epoch: int, loss: float = 0.0,
) -> Dict[str, Any]:
    """Pack a state message with the current NCA state.

    Args:
        state: (B, C_total, D, H, W) or (C_total, D, H, W) float32.
        epoch: current training epoch (1-based).
        loss:  latest training loss value.
    """
    return {
        "type": "state",
        "data": tensor_to_b64(state),
        "shape": list(state.shape),
        "epoch": epoch,
        "loss": loss,
    }

def parse_state_msg(msg: Dict[str, Any]) -> Tuple[np.ndarray, int, float]:
    """Unpack a state message.

    Returns:
        state: numpy array with shape from the message (channels-first).
        epoch: current training epoch.
        loss:  latest training loss.
    """
    state = b64_to_tensor(msg["data"], msg["shape"])
    epoch = msg.get("epoch", 0)
    loss = msg.get("loss", 0.0)
    return state, epoch, loss

def build_stop_msg() -> Dict[str, Any]:
    """Pack a stop message - terminates training."""
    return {"type": "stop"}

def build_pause_msg() -> Dict[str, Any]:
    """Pack a pause message - pauses the training loop."""
    return {"type": "pause"}

def build_resume_msg() -> Dict[str, Any]:
    """Pack a resume message - resumes a paused training loop."""
    return {"type": "resume"}

def build_ack_msg(message: str) -> Dict[str, Any]:
    """Pack an ack response confirming a client command."""
    return {"type": "ack", "message": message}

def build_error_msg(message: str) -> Dict[str, Any]:
    """Pack an error response with a description."""
    return {"type": "error", "message": message}

def build_schedule_msg(events: list) -> Dict[str, Any]:
    """Pack an update_schedule message."""
    return {"type": "update_schedule", "events": events}

def parse_schedule_msg(msg: Dict[str, Any]) -> list:
    """Unpack an update_schedule message."""
    return msg["events"]
