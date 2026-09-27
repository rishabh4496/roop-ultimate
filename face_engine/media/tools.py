"""Locate ffmpeg/ffprobe and run ffprobe, without depending on the caller's PATH.

Resolution order: ``FACE_ENGINE_FFMPEG`` (a path to ffmpeg; ffprobe is taken
from the same folder), ``PATH``, then the Pinokio-bundled toolchains under
``PINOKIO_HOME`` (``bin/miniforge``, ``bin/miniconda``, ``bin/ffmpeg-env``).
``PINOKIO_HOME`` is resolved at run time (env var, then
``~/.pinokio/config.json``); no absolute path is written anywhere.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path
from typing import Any


class ToolNotFoundError(FileNotFoundError):
    """ffmpeg or ffprobe could not be located."""


def _pinokio_home() -> Path | None:
    home = os.environ.get("PINOKIO_HOME")
    if home and Path(home).is_dir():
        return Path(home)
    try:
        cfg = json.loads((Path.home() / ".pinokio" / "config.json").read_text(encoding="utf-8"))
        if cfg.get("home") and Path(cfg["home"]).is_dir():
            return Path(cfg["home"])
    except (OSError, ValueError):
        pass
    return None


@cache
def find_tool(name: str) -> str:
    """Absolute path to ``ffmpeg`` or ``ffprobe``.

    Raises:
        ToolNotFoundError: nothing found in any location.
    """
    exe = name + (".exe" if os.name == "nt" else "")
    override = os.environ.get("FACE_ENGINE_FFMPEG")
    if override:
        candidate = Path(override).with_name(exe)
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    home = _pinokio_home()
    if home is not None:
        for sub in ("miniforge", "miniconda", "ffmpeg-env"):
            for folder in (home / "bin" / sub / "Library" / "bin", home / "bin" / sub / "bin"):
                if (folder / exe).is_file():
                    return str(folder / exe)
    raise ToolNotFoundError(f"{name} not found (set FACE_ENGINE_FFMPEG or put it on PATH)")


def ffprobe_json(path: str | Path, *args: str, timeout: float = 120.0) -> dict[str, Any]:
    """Run ffprobe with JSON output; raises ``RuntimeError`` with stderr on failure."""
    cmd = [find_tool("ffprobe"), "-v", "error", "-of", "json", *args, str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()[-500:]}")
    return json.loads(proc.stdout or "{}")
