"""KEEP enhancer — client for the isolated sidecar process (EXPERIMENTAL).

KEEP's dependencies conflict with the main env (diffusers/huggingface_hub
pins), so the model runs in `app/sidecar_keep/.venv` as a separate HTTP
process (see sidecar_keep/README.md). This processor:

  - starts the sidecar on first use (if `setup_sidecar.py` has been run),
  - waits for /health,
  - POSTs each aligned 512px face crop to /enhance,
  - and PASSES FRAMES THROUGH UNENHANCED with one clear log line if the
    sidecar is missing or fails — it can never break a run.

Matches the standard enhancer contract: Run(...) -> (uint8 BGR frame, scale).
"""

import os
import socket
import subprocess
import threading
import time
import urllib.request

import cv2
import numpy as np

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path

_SIDECAR_DIR = None


def _sidecar_dir():
    global _SIDECAR_DIR
    if _SIDECAR_DIR is None:
        _SIDECAR_DIR = resolve_relative_path('../sidecar_keep')
    return _SIDECAR_DIR


def _sidecar_python():
    d = _sidecar_dir()
    if os.name == 'nt':
        return os.path.join(d, '.venv', 'Scripts', 'python.exe')
    return os.path.join(d, '.venv', 'bin', 'python')


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Enhance_KEEP():
    plugin_options: dict = None

    processorname = 'keep'
    type = 'enhance'

    def __init__(self):
        self._proc = None
        self._port = None
        self._ready = False
        self._warned = False
        self._lock = threading.Lock()

    # ── sidecar lifecycle ────────────────────────────────────────────────────
    def _start_sidecar(self) -> bool:
        py = _sidecar_python()
        server = os.path.join(_sidecar_dir(), 'server.py')
        if not (os.path.exists(py) and os.path.exists(server)):
            if not self._warned:
                print("[KEEP] sidecar not installed — run "
                      "'python sidecar_keep/setup_sidecar.py' (see sidecar_keep/README.md). "
                      "Frames pass through unenhanced.")
                self._warned = True
            return False
        self._port = _free_port()
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = 0x08000000  # CREATE_NO_WINDOW
        self._proc = subprocess.Popen([py, server, str(self._port)],
                                      cwd=_sidecar_dir(), **kwargs)
        # Model load can take a while (torch + checkpoint) — poll /health.
        deadline = time.time() + 120
        url = f"http://127.0.0.1:{self._port}/health"
        while time.time() < deadline:
            if self._proc.poll() is not None:
                print(f"[KEEP] sidecar exited during startup (code {self._proc.returncode}) "
                      f"— frames pass through unenhanced.")
                self._proc = None
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        print(f"[KEEP] sidecar ready on port {self._port}")
                        return True
            except Exception:
                time.sleep(0.5)
        print("[KEEP] sidecar did not become ready in 120s — frames pass through unenhanced.")
        self._stop_sidecar()
        return False

    def _stop_sidecar(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self._ready = False

    # ── enhancer contract ────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        with self._lock:
            if not self._ready and self._proc is None:
                self._ready = self._start_sidecar()

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        input_size = temp_frame.shape[1]
        if not self._ready:
            return temp_frame, 1
        try:
            ok, buf = cv2.imencode('.png', temp_frame)
            if not ok:
                return temp_frame, 1
            req = urllib.request.Request(
                f"http://127.0.0.1:{self._port}/enhance",
                data=buf.tobytes(),
                headers={'Content-Type': 'application/octet-stream'})
            # The sidecar serializes requests per connection; ThreadingHTTPServer
            # handles our worker threads concurrently.
            with urllib.request.urlopen(req, timeout=60) as r:
                out = r.read()
            img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return temp_frame, 1
            scale_factor = max(1, int(img.shape[1] / input_size))
            return img, scale_factor
        except Exception as e:
            if not self._warned:
                print(f"[KEEP] enhance call failed ({e}) — passing frames through.")
                self._warned = True
            return temp_frame, 1

    def Release(self):
        with self._lock:
            self._stop_sidecar()
