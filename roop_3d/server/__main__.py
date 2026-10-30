"""Reserve the exact loopback socket before Uvicorn starts; never kill a port owner."""
from __future__ import annotations

import json
import logging
import socket
import sys

from pydantic import ValidationError
import uvicorn

from roop_3d.core.settings import Settings
from roop_3d.server.app import create_app


class PortCollisionError(RuntimeError):
    pass


def reserve_socket(settings: Settings) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind((settings.host, settings.port))
        listener.listen(128)
        listener.setblocking(False)
        return listener
    except OSError as error:
        listener.close()
        raise PortCollisionError(f"Cannot bind 127.0.0.1:{settings.port}. Another process may own the port. "
                                 "Set ROOP3D_PORT to an unused port; no existing process was stopped.") from error


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        try:
            value = json.loads(message)
            if isinstance(value, dict):
                return message
        except (ValueError, TypeError):
            pass
        return json.dumps({"level": record.levelname.lower(), "logger": record.name, "message": message})


class ReadyServer(uvicorn.Server):
    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            print(json.dumps({"event": "ready", "application": "roop_3d",
                              "url": f"http://127.0.0.1:{self.config.port}"}), flush=True)


def main() -> int:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    try:
        settings = Settings.from_env()
        listener = reserve_socket(settings)
    except (ValidationError, PortCollisionError) as error:
        message = str(error) if isinstance(error, PortCollisionError) else "Invalid ROOP3D settings; host must be 127.0.0.1 and API port must not be 5174"
        print(json.dumps({"error": {"code": "PORT_IN_USE" if isinstance(error, PortCollisionError) else "INVALID_SETTINGS", "message": message}}), flush=True)
        return 2
    try:
        config = uvicorn.Config(create_app(settings), host=settings.host, port=settings.port, log_config=None,
                                access_log=False, proxy_headers=False, timeout_graceful_shutdown=10)
        server = ReadyServer(config)
        server.run(sockets=[listener])
        return 0
    finally:
        listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
