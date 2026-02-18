import threading
import socket
import numpy as np
from typing import Callable, Dict, Optional

from .protocol import (
    send_msg, recv_msg,
    build_init_msg, build_stop_msg, build_pause_msg, build_resume_msg,
    parse_state_msg,
)

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
        self._sock.settimeout(2.0)

    def disconnect(self) -> None:
        self._running = False
        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=5.0)
        self._listener_thread = None
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def send_init(self, config: dict, target: np.ndarray) -> None:
        send_msg(self._sock, build_init_msg(config, target))

    def send_stop(self):
        send_msg(self._sock, build_stop_msg())

    def send_pause(self):
        send_msg(self._sock, build_pause_msg())

    def send_resume(self):
        send_msg(self._sock, build_resume_msg())

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
            try:
                msg = recv_msg(self._sock)
            except socket.timeout:
                continue
            except OSError:
                break

            if msg is None:
                if on_disconnect:
                    on_disconnect()
                break

            msg_type = msg.get("type")
            if msg_type == "state" and on_state:
                state, _epoch, _loss = parse_state_msg(msg)
                on_state(state)
            elif msg_type == "error" and on_error:
                on_error(msg.get("message", "Unknown error"))
            elif msg_type == "ack":
                pass
            elif msg_type == "pong":
                pass