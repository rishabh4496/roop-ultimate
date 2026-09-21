"""pytest configuration for both suites (app/tests and tests/).

Markers
-------
gpu   The test needs a CUDA/TensorRT device or the full ML stack (torch,
      onnxruntime, insightface, model files). CI deselects these with
      `-m "not gpu"`; locally they run.

Light profile (ROOP_TEST_LIGHT=1)
---------------------------------
CI has no GPU, no model files and none of the heavy packages installed. With
the profile on, an import of one of those packages -- at module import or
inside a test -- is turned into a *skip* with the reason "needs the ML stack",
so the light subset runs green and the skipped count says how much was not
exercised. A test that imports the stack is also given the `gpu` marker at
collection so `-m "not gpu"` and the profile agree. Nothing is skipped when
the profile is off: a missing package is then a real failure.

Measured 2026-09-22 on the app suite: 92 modules fully light, 145 need the
stack at import, 29 import fine but reach for it in some tests.
"""
from __future__ import annotations

import importlib.abc
import os
import sys
import threading

import pytest

HEAVY_PACKAGES = frozenset({
    "torch", "torchvision", "onnxruntime", "insightface", "cv2", "tensorrt",
    "gradio", "diffusers", "transformers", "timm", "librosa", "skimage",
    "albumentations", "albucore", "pyvirtualcam", "onnx", "nvidia", "pynvml",
    "cupy", "moviepy", "imageio_ffmpeg", "scipy", "PIL", "numba", "accelerate",
})

LIGHT = os.environ.get("ROOP_TEST_LIGHT") == "1"

# Child processes some harnesses spawn (tests/test_benchmark_runner.py) print
# em-dashes; on a Windows console they come out as cp1252 bytes, pytest's fd
# capture decodes them as UTF-8 and every test after the first one errors at
# teardown. UTF-8 for every child fixes the cascade and is a no-op elsewhere.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: needs a CUDA/TensorRT device or the full ML stack (torch, onnxruntime, "
        "insightface, model files). CI runs with -m 'not gpu'.")
    if LIGHT and not any(isinstance(f, _HeavyImportSkips) for f in sys.meta_path):
        sys.meta_path.insert(0, _HeavyImportSkips())


class _HeavyImportSkips(importlib.abc.MetaPathFinder):
    """In the light profile, importing a heavy package skips instead of failing."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".", 1)[0] not in HEAVY_PACKAGES or name in sys.modules:
            return None
        reason = f"needs the ML stack ({name}); not available in the light profile"
        # A skip is only meaningful on the thread pytest is running the test
        # on. A worker thread that probes for torch/psutil inside try/except
        # gets an ordinary ImportError, which is what a machine without the
        # package would give it -- Skipped is a BaseException and would kill
        # the thread instead.
        if threading.current_thread() is not threading.main_thread():
            raise ImportError(reason)
        raise pytest.skip.Exception(reason, allow_module_level=True)


def pytest_collection_modifyitems(config, items):
    """Explicit `gpu` marks are the contract; in the light profile, tests that
    import the stack at module level never reach here (skipped at import)."""
    if not LIGHT:
        return
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(pytest.mark.skip(reason="gpu-marked test in the light profile"))
