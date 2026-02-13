import socket
import threading

from .protocol import recv_msg, send_msg, b64_to_tensor, tensor_to_b64


class NCAServer:
    def __init__(self, trainer, host='127.0.0.1', port=5555):
        self.trainer = trainer
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
            self._client = client
            try:
                self._handle_client(client)
            except Exception as e:
                print(f"Error handling client: {e}")
            finally:
                self.trainer.stop()
                client.close()
                print(f"Client disconnected from {addr}")

    def _handle_client(self, client):
        while True:
            msg = recv_msg(client)
            if msg is None:
                break
                
            msg_type = msg.get("type")

            if msg_type == "init":
                config = msg["config"]
                target = b64_to_tensor(msg["target"], msg["target_shape"])
                self.trainer.init(config, target)
                send_msg(client, {"type": "ack", "message": "Initialized"})

            elif msg_type == "stop":
                self.trainer.stop()
                send_msg(client, {"type": "ack", "message": "Stopped"})

            elif msg_type == "pause":
                self.trainer.pause()
                send_msg(client, {"type": "ack", "message": "Paused"})

            elif msg_type == "resume":
                self.trainer.resume()
                send_msg(client, {"type": "ack", "message": "Resumed"})

            else:
                send_msg(client, {"type": "error", "message": f"Unknown message type: {msg_type}"})