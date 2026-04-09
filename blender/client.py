"""
NCAClient - TCP socket client for connecting to remote NCA server.

Initiates connections and sends training/inference commands. Runs a background
listener thread to receive state broadcasts and errors from the server.
Supports pause/resume/stop control and dynamic schedule updates.
"""

import socket
import threading
import base64
from typing import Callable, Dict, Optional

import numpy as np

from .protocol import (
    send_msg,
    recv_msg,
    build_init_msg,
    build_run_model_msg,
    build_stop_msg,
    build_pause_msg,
    build_resume_msg,
    build_schedule_msg,
    parse_state_msg,
)


class NCAClient:
    """Client for remote NCA server communication."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555) -> None:
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def connected(self) -> bool:
        """Return True when socket is open and client is connected."""
        return self._sock is not None

    def connect(self, timeout: float = 5.0) -> None:
        """Open TCP connection to configured host and port.

        Args:
            timeout: Connection timeout in seconds.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect((self.host, self.port))
        self._sock.settimeout(2.0)

    def disconnect(self) -> None:
        """Stop listener thread and close socket connection safely."""
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

    # --- Send Helpers ---

    def send_init(self, config: dict, target: np.ndarray) -> None:
        """Send training initialization message.

        Args:
            config: Training configuration dict.
            target: Target voxel grid [D,H,W,C] external format.
        """
        send_msg(self._sock, build_init_msg(config, target))

    def send_run_model(
        self,
        model_path: str,
        phase_steps: int,
        broadcast_every: int,
        send_delay_ms: int = 40,
    ) -> None:
        """Send inference request with model from disk.

        Args:
            model_path: Path to .pt checkpoint file.
            phase_steps: Total forward steps to run.
            broadcast_every: Broadcast interval (steps).
            send_delay_ms: Delay before each state send in milliseconds.
        """
        with open(model_path, "rb") as f:
            model_b64 = base64.b64encode(f.read()).decode("ascii")

        send_msg(
            self._sock,
            build_run_model_msg(
                model_b64,
                phase_steps,
                broadcast_every,
                send_delay_ms,
            ),
        )

    def send_stop(self) -> None:
        """Send stop command to terminate active server session."""
        send_msg(self._sock, build_stop_msg())

    def send_pause(self) -> None:
        """Send pause command for active training session."""
        send_msg(self._sock, build_pause_msg())

    def send_resume(self) -> None:
        """Send resume command for paused training session."""
        send_msg(self._sock, build_resume_msg())

    def send_schedule(self, events: list) -> None:
        """Send schedule event updates to server.

        Args:
            events: List of event dictionaries.
        """
        send_msg(self._sock, build_schedule_msg(events))

    # --- Listener ---

    def start_listener(
        self,
        on_state: Optional[Callable[[np.ndarray], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        """Start background listener thread for server messages.

        Args:
            on_state: Callback for state broadcast [callback(state_array)].
            on_error: Callback for error messages [callback(error_string)].
            on_disconnect: Callback when connection lost [callback()].
        """
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            args=(on_state, on_error, on_disconnect),
            daemon=True,
        )
        self._listener_thread.start()

    def _listen_loop(
        self,
        on_state: Optional[Callable],
        on_error: Optional[Callable],
        on_disconnect: Optional[Callable],
    ) -> None:
        """Background message loop - runs in listener thread.

        Continuously receives protocol messages, parses them, and invokes
        appropriate callbacks. Exits on connection loss or stop event.

        Args:
            on_state: State callback.
            on_error: Error callback.
            on_disconnect: Disconnect callback.
        """
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
                pass  # acknowledgements are informational only
