"""
Server Application Entry Point.

Creates and starts the NCA TCP server process.
"""

from .server import NCAServer


def main():
    """Create and start NCAServer with default host and port."""
    server = NCAServer()
    server.start()


if __name__ == "__main__":
    main()
