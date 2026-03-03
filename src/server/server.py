import socket
import threading

from .trainer import NCATrainer 
from .protocol import (
    recv_msg, send_msg,
    parse_init_msg, build_ack_msg, build_error_msg,
)

class NCAServer:
    def __init__(self, host='127.0.0.1', port=5555):
        self.trainer = NCATrainer()
        self.host = host 
        self.port = port
        self._sock = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(1)
        print(f"Server started on {self.host}:{self.port}")

        while True:
            client, addr = self._sock.accept()
            print(f"Client connected from {addr}")
            try:
                self._handle_client(client)
            except Exception as e:
                print(f"Error handling client: {e}")
            finally:
                self.trainer.stop()
                client.close()
                print(f"Client disconnected from {addr}")

    def _handle_client(self, client):

        def send_fn(msg):
            send_msg(client, msg)

        while True:
            msg = recv_msg(client)
            print(f"Received message: {msg}")
            if msg is None:
                break
                
            msg_type = msg.get("type")

            if msg_type == "init":
                config, target = parse_init_msg(msg)
                self.trainer.init(config, target, send_fn)
                send_msg(client, build_ack_msg("Initialized"))

            elif msg_type == "stop":
                self.trainer.stop()
                break

            elif msg_type == "pause":
                self.trainer.pause()
                send_msg(client, build_ack_msg("Paused"))

            elif msg_type == "resume":
                self.trainer.resume()
                send_msg(client, build_ack_msg("Resumed"))

            elif msg_type == "ping":
                send_msg(client, {"type": "pong"})

            else:
                send_msg(client, build_error_msg(f"Unknown message type: {msg_type}"))