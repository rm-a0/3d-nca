class NCAServer:
    def __init__(self, trainer, host='127.0.0.1', port=5555):
        self.trainer = trainer
        self.host = host 
        self.port = port

    def start(self):
        # This is a placeholder for server logic
        while True:
            pass  # Replace with actual server loop logic