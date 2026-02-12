import socket
import threading

from .protocol import recv_msg, send_msg, b64_to_tensor, tensor_to_b64


class NCAServer:
    def __init__(self, trainer, host='127.0.0.1', port=5555):
        self.trainer = trainer
        self.host = host 
        self.port = port

        self.sock = None
        self._client = None
        self._train_thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

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
                client.close()
                self._cleanup_training()
                print(f"Client disconnected from {addr}")

    def _handle_client(self, client):
        while True:
            msg = recv_msg(client)
            if msg is None:
                break

            handler = {
                "init": self._handle_init,
                "stop": self._handle_stop,
                "pause": self._handle_pause,
                "resume": self._handle_resume,
                "get_state": self._handle_get_state,
                "ping": self._handle_ping,
            }.get(msg.get("type"))

            if handler:
                handler(client, msg)
            else:
                send_msg(client, {"type": "error", "message": "Unknown message type"})

            def _handle_init(self, client, msg):
                pass

            def _handle_stop(self, client, msg):
                pass

            def _handle_pause(self, client, msg):
                pass

            def _handle_resume(self, client, msg):
                pass

            def _handle_get_state(self, client, msg):   
                pass

            def _handle_ping(self, client, msg):
                pass