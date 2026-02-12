from .trainer import NCATrainer
from .server import NCAServer

def main():
    trainer = NCATrainer()
    server = NCAServer(trainer)
    server.start()

if __name__ == "__main__":
    main()