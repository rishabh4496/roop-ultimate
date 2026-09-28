"""Run the server: ``python -m face_engine.server [--host H] [--port P] [--workspace DIR]``.

Serves the built web UI (``web_ui/dist``) at ``/`` when it exists.
Environment: ``FACE_ENGINE_WORKSPACE``, ``FACE_ENGINE_HOST``, ``FACE_ENGINE_PORT``.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from face_engine.server.api import create_app
from face_engine.server.state import ServerSettings, _default_workspace


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m face_engine.server")
    parser.add_argument("--host", default=os.environ.get("FACE_ENGINE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FACE_ENGINE_PORT", "8765")))
    parser.add_argument("--workspace", default=os.environ.get("FACE_ENGINE_WORKSPACE",
                                                              str(_default_workspace())))
    parser.add_argument("--ui", default=str(Path(__file__).resolve().parents[2] / "web_ui" / "dist"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = ServerSettings(workspace=Path(args.workspace), host=args.host, port=args.port,
                              ui_dist=Path(args.ui))
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
