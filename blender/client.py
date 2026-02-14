import threading
from .protocol import send_msg, recv_msg, tensor_to_b64, b64_to_tensor
import socket
import numpy as np
from typing import Callable, Dict, Optional

class NCAClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.host = host
        self.port = port
        self._sock = None
        self._listener_thread = None
        self._running = False

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self, timeout: float = 5.0) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect((self.host, self.port))
        self._sock.settimeout(None)

    def disconnect(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self._sock.close()
            self._sock = None
        if self._listener_thread:
            self._listener_thread.join(timeout=3)
            self._listener_thread = None

    def send_init(self, config: dict, target: np.ndarray) -> None:
        send_msg(self._sock, {
            "type": "init",
            "config": config,
            "target": tensor_to_b64(target),
            "target_shape": list(target.shape),
        })

    def send_stop(self):
        send_msg(self._sock, {"type": "stop"})

    def send_pause(self):
        send_msg(self._sock, {"type": "pause"}) 

    def send_resume(self):  
        send_msg(self._sock, {"type": "resume"})

    def send_ping(self):
        send_msg(self._sock, {"type": "ping"})

    def start_listener(
        self, 
        on_state: Optional[Callable[[np.ndarray], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
    ):
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            args=(on_state, on_error, on_disconnect),
            daemon=True
        )
        self._listener_thread.start()

    def _listen_loop(
        self, 
        on_state: Optional[Callable[[np.ndarray], None]],
        on_error: Optional[Callable[[str], None]], 
        on_disconnect: Optional[Callable[[], None]]
    ):
        while self._running:
            msg = recv_msg(self._sock)
            if msg is None:
                if on_disconnect:
                    on_disconnect()
                break

            msg_type = msg.get("type")
            if msg_type == "state" and on_state:
                state = b64_to_tensor(msg["state"], msg["state_shape"])
                on_state(state)
            elif msg_type == "error" and on_error:
                on_error(msg.get("message", "Unknown error"))
            elif msg_type == "ack":
                pass
            elif msg_type == "pong":
                pass