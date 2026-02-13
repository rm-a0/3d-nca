from .server import NCAServer

def main():
    server = NCAServer()
    server.start()

if __name__ == "__main__":
    main()