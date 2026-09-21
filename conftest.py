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


class HeavyImportBlocked(ImportError, pytest.skip.Exception):
    """Both an ImportError and a pytest skip.

    Code that guards an optional import (`try: import torch except ImportError`)
    catches this exactly as it would on a machine without the package, so the
    light profile exercises the same degraded paths CI's runners have. An
    UNGUARDED import lets it propagate, and pytest reads it as a skip -- of the
    whole module when raised at import time, of one test otherwise.
    """

    def __init__(self, reason):
        ImportError.__init__(self, reason)
        pytest.skip.Exception.__init__(self, reason, allow_module_level=True)


class _HeavyImportSkips(importlib.abc.MetaPathFinder):
    """In the light profile, importing a heavy package skips instead of failing."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".", 1)[0] not in HEAVY_PACKAGES or name in sys.modules:
            return None
        raise HeavyImportBlocked(f"needs the ML stack ({name}); not available in the light profile")


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """A test module whose own top-level import hits the blocker is SKIPPED.

    pytest checks ImportError before Skipped when importing a test module, so
    without this the blocked import would be reported as a collection error.
    Only reports whose failure is the blocker are touched; every other
    collection error stays an error.
    """
    outcome = yield
    if not LIGHT:
        return
    report = outcome.get_result()
    if report.outcome != "failed":
        return
    text = str(report.longrepr)
    if "HeavyImportBlocked" not in text:
        return
    reason = next((line.split("HeavyImportBlocked:", 1)[1].strip()
                   for line in text.splitlines() if "HeavyImportBlocked:" in line),
                  "needs the ML stack; not available in the light profile")
    report.outcome = "skipped"
    report.longrepr = (str(getattr(collector, "path", collector.name)), 0, f"Skipped: {reason}")


def pytest_collection_modifyitems(config, items):
    """Explicit `gpu` marks are the contract; in the light profile, tests that
    import the stack at module level never reach here (skipped at import)."""
    if not LIGHT:
        return
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(pytest.mark.skip(reason="gpu-marked test in the light profile"))
