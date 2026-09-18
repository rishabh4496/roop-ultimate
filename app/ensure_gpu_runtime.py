#!/usr/bin/env python3
"""Guarantee the GPU inference stack is installed, whatever the launcher decided.

WHY THIS EXISTS

`torch.js` selects its dependency set from Pinokio's `gpu` template variable.
That is the documented mechanism and it is correct when it works, but it has
two failure modes that both end with a machine that cannot accelerate:

  1. A `when` expression that raises is SKIPPED silently, so a condition that
     touches a variable Pinokio did not define installs nothing at all.
  2. When `gpu` is reported as anything other than 'nvidia' on a machine that
     does have an NVIDIA card, the CPU branch wins and installs
     `onnxruntime` (CPU) instead of `onnxruntime-gpu` + TensorRT.

Both look identical at startup: no TensorRT, and in case 1 not even a usable
`onnxruntime` -- the import resolves to an implicit namespace package whose
`__file__` is None, which is the "exposes no provider API (loaded from None)"
report.

This script is the backstop. It asks the MACHINE what it has (nvidia-smi,
which every NVIDIA driver installs into System32, plus torch's own CUDA build)
rather than asking the launcher, and installs only what is actually missing.
It is safe to run on every start: when the environment is already correct it
performs no installs and exits immediately.

Usage:
    python ensure_gpu_runtime.py            # repair if needed
    python ensure_gpu_runtime.py --check    # report only, never install
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

# Pinned to match torch.js. onnxruntime-gpu 1.23.2's TensorRT EP is built
# against TensorRT 10.9, so these two move together.
ONNXRUNTIME_GPU = "onnxruntime-gpu==1.23.2"
TENSORRT_PACKAGES = (
    "tensorrt-cu12==10.9.0.34",
    "tensorrt-cu12-libs==10.9.0.34",
    "tensorrt-cu12-bindings==10.9.0.34",
)
NVIDIA_INDEX = "https://pypi.nvidia.com/"


def _module_is_real(name: str) -> bool:
    """True only for an importable module with actual code behind it.

    A partially removed package leaves a directory with no `__init__.py`.
    Python still resolves that as an implicit NAMESPACE package, so a plain
    `find_spec(...) is not None` reports it as present while every attribute
    access fails. `origin` is None for exactly that case.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return bool(spec and spec.origin)


def has_nvidia_gpu() -> bool:
    """Ask the machine, not the launcher."""
    smi = shutil.which("nvidia-smi")
    if not smi and os.name == "nt":
        candidate = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        smi = candidate if os.path.isfile(candidate) else None
    if smi:
        try:
            done = subprocess.run([smi, "-L"], capture_output=True, text=True,
                                  timeout=20, check=False)
            if done.returncode == 0 and "GPU" in (done.stdout or ""):
                return True
        except Exception:
            pass

    # A CUDA-enabled torch build is equally conclusive.
    try:
        import torch
        if getattr(torch.version, "cuda", None) and torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


def onnxruntime_status() -> tuple[bool, list[str]]:
    """(importable_with_provider_api, advertised_providers)."""
    if not _module_is_real("onnxruntime"):
        return False, []
    try:
        import onnxruntime as ort
        lister = getattr(ort, "get_available_providers", None)
        if not callable(lister):
            return False, []
        return True, [str(p) for p in lister()]
    except Exception:
        return False, []


def _find_uv() -> str | None:
    uv = shutil.which("uv")
    if uv:
        return uv
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (
        os.path.join(here, "..", "..", "..", "bin"),
        os.path.join(here, "..", "..", "bin"),
    ):
        for flavor in ("miniforge", "miniconda"):
            candidate = os.path.abspath(os.path.join(base, flavor, "Library", "bin", "uv.exe"))
            if os.path.isfile(candidate):
                return candidate
            candidate_nix = os.path.abspath(os.path.join(base, flavor, "bin", "uv"))
            if os.path.isfile(candidate_nix):
                return candidate_nix
    return None


def _pip(args: list[str]) -> bool:
    """Install through uv when available, else pip. Never raises."""
    uv = _find_uv()
    commands = []
    if uv:
        commands.append([uv, "pip", "install", "--python", sys.executable, *args])
    commands.append([sys.executable, "-m", "pip", "install", *args])
    for command in commands:
        try:
            print("[ensure-gpu] $ " + " ".join(command), flush=True)
            if subprocess.run(command, check=False).returncode == 0:
                return True
        except Exception as error:
            print(f"[ensure-gpu] {type(error).__name__}: {error}", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="report only; never install")
    options = parser.parse_args()

    nvidia = has_nvidia_gpu()
    usable, providers = onnxruntime_status()
    has_trt_ep = "TensorrtExecutionProvider" in providers
    trt_pkg = _module_is_real("tensorrt")

    print(f"[ensure-gpu] python      : {sys.executable}")
    print(f"[ensure-gpu] nvidia gpu  : {nvidia}")
    print(f"[ensure-gpu] onnxruntime : {'usable' if usable else 'MISSING/BROKEN'}")
    print(f"[ensure-gpu] providers   : {providers or '[]'}")
    print(f"[ensure-gpu] tensorrt pkg: {trt_pkg}")

    # Work out what is actually wrong before changing anything.
    needs_ort = not usable or (nvidia and not any(
        p in providers for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")))
    needs_trt = nvidia and not (trt_pkg and has_trt_ep)

    if not needs_ort and not needs_trt:
        print("[ensure-gpu] environment is complete; nothing to do")
        return 0

    if options.check:
        print("[ensure-gpu] repair REQUIRED (check-only mode, nothing installed)")
        return 1

    if needs_ort:
        if nvidia:
            # A CPU onnxruntime shadows the GPU build; remove it first.
            _pip(["--upgrade", ONNXRUNTIME_GPU])
        else:
            _pip(["onnxruntime==1.17.1"])

    if needs_trt:
        _pip(["--extra-index-url", NVIDIA_INDEX, *TENSORRT_PACKAGES])

    usable, providers = onnxruntime_status()
    print(f"[ensure-gpu] after repair : {providers or '[]'}")
    if not usable:
        print("[ensure-gpu] onnxruntime is still unusable; run Fix TensorRT")
        return 1
    if nvidia and "TensorrtExecutionProvider" not in providers:
        print("[ensure-gpu] TensorRT EP still absent; CUDA acceleration will be used")
    return 0


if __name__ == "__main__":
    sys.exit(main())
