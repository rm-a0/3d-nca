"""
NCAServer - TCP socket server for remote NCA training and inference.

Accepts client connections and dispatches training/inference commands via
protocol messages. Manages a single NCATrainer instance per connection for
background execution of expensive GPU operations. Supports live control:
pause, resume, stop, and dynamic schedule updates during training.

Messages are JSON-encoded with binary tensor/model data base64-encoded.
See protocol.py for full wire format specification.
"""

import socket

from .trainer import NCATrainer
from .protocol import (
    recv_msg,
    send_msg,
    parse_init_msg,
    parse_run_model_msg,
    parse_schedule_msg,
    build_ack_msg,
    build_error_msg,
)


class NCAServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5555) -> None:
        self.trainer = NCATrainer()
        self.host = host
        self.port = port
        self._sock = None

    def start(self) -> None:
        """Start TCP server and accept client connections indefinitely.

        Binds to host:port and enters blocking accept loop. Each client connection
        is handled by a dedicated message loop. Stops only if exception occurs or
        server is externally terminated.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(1)
        print(f"[Server] Listening on {self.host}:{self.port}")

        while True:
            client, addr = self._sock.accept()
            print(f"[Server] Client connected: {addr}")
            try:
                self._handle_client(client)
            except Exception as exc:
                print(f"[Server] Client error: {exc}")
            finally:
                self.trainer.stop()
                client.close()
                print(f"[Server] Client disconnected: {addr}")

    def _handle_client(self, client: socket.socket) -> None:
        """Handle protocol messages from a single client connection.

        Dispatches incoming messages (init, run_model, pause, resume, stop, etc.)
        to trainer and sends protocol responses. Runs until client disconnects.

        Args:
            client: Connected socket from accept().
        """

        def send_fn(msg):
            try:
                send_msg(client, msg)
            except Exception:
                pass

        while True:
            try:
                msg = recv_msg(client)
            except Exception as exc:
                print(f"[Server] Receive error: {exc}")
                break

            if msg is None:
                break

            msg_type = msg.get("type")

            if msg_type == "init":
                config, target = parse_init_msg(msg)
                self.trainer.init(config, target, send_fn)
                send_msg(client, build_ack_msg("Training started"))

            elif msg_type == "run_model":
                model_bytes, phase_steps, broadcast_every = parse_run_model_msg(msg)
                self.trainer.run_inference(
                    model_bytes, phase_steps, broadcast_every, send_fn
                )
                send_msg(client, build_ack_msg("Inference started"))

            elif msg_type == "stop":
                self.trainer.stop()
                send_msg(client, build_ack_msg("Stopped"))

            elif msg_type == "pause":
                self.trainer.pause()
                send_msg(client, build_ack_msg("Paused"))

            elif msg_type == "resume":
                self.trainer.resume()
                send_msg(client, build_ack_msg("Resumed"))

            elif msg_type == "update_schedule":
                events = parse_schedule_msg(msg)
                self.trainer.update_schedule(events)
                send_msg(client, build_ack_msg("Schedule updated"))

            elif msg_type == "ping":
                send_msg(client, {"type": "pong"})

            else:
                send_msg(client, build_error_msg(f"Unknown message type: {msg_type!r}"))
