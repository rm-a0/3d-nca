import json
import socket
import struct
import numpy as np
import base64
from typing import Any, Dict, Optional

HEADER_SIZE = 4

def encode_message(msg: Dict[str, Any]) -> bytes:
    payload = json.dumps(msg).encode('utf-8')
    return struct.pack(">I", len(payload)) + payload

def decode_message(data: bytes) -> Dict[str, Any]:
    return json.loads(data.decode('utf-8'))

def tensor_to_b64(tensor: np.ndarray) -> str:
    return base64.b64encode(tensor.astype(np.float32).tobytes()).decode("ascii")

def b64_to_tensor(b64: str, shape: list) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape)

def send_msg(sock: socket.socket, msg: Dict[str, Any]) -> None:
    sock.sendall(encode_message(msg))

def recv_msg(sock: socket.socket) -> Optional[Dict[str, Any]]:
    header = _recv_exact(sock, HEADER_SIZE)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return decode_message(payload)

def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
