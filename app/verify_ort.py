#!/usr/bin/env python3
"""Deterministic verification of ONNX Runtime installation and environment integrity.

Verifies:
1. No local onnxruntime.py or local onnxruntime/ directory shadows the package.
2. No simultaneous conflicting packages (onnxruntime vs onnxruntime-gpu).
3. NumPy version ABI compatibility (<2.0.0, pinned 1.26.4).
4. onnxruntime imports with a valid __file__ (not a PEP 420 empty namespace package).
5. onnxruntime strictly exposes a callable get_available_providers() API.
6. Execution providers are genuinely queryable and non-empty.
"""
from __future__ import annotations

import importlib.metadata
import os
import sys


def detect_shadowing(app_dir: str) -> None:
    """Detect local files or directories that shadow the onnxruntime package."""
    cwd = os.getcwd()
    check_paths = [cwd]
    if app_dir not in check_paths:
        check_paths.append(app_dir)

    for path in check_paths:
        # Check for shadow file onnxruntime.py
        py_shadow = os.path.join(path, "onnxruntime.py")
        if os.path.isfile(py_shadow):
            print(f"[FATAL] Local file shadows onnxruntime package: {py_shadow}", file=sys.stderr)
            sys.exit(1)

        # Check for local directory onnxruntime (excluding site-packages)
        dir_shadow = os.path.join(path, "onnxruntime")
        if os.path.isdir(dir_shadow):
            init_file = os.path.join(dir_shadow, "__init__.py")
            # If it's directly under app/ or cwd and not site-packages, it's shadowing
            is_site_packages = "site-packages" in os.path.normpath(dir_shadow).lower()
            if not is_site_packages:
                print(f"[FATAL] Local directory shadows onnxruntime package: {dir_shadow}", file=sys.stderr)
                sys.exit(1)

    # Detect package collision: onnxruntime (CPU) and onnxruntime-gpu cannot co-exist
    try:
        installed_names = set()
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name")
            if name:
                installed_names.add(name.lower())

        if "onnxruntime" in installed_names and "onnxruntime-gpu" in installed_names:
            print(
                "[FATAL] Package collision detected: both 'onnxruntime' and 'onnxruntime-gpu' "
                "are installed in the environment. Exactly one package contract must exist.",
                file=sys.stderr
            )
            sys.exit(1)
    except Exception as exc:
        print(f"[WARNING] Could not check distribution metadata: {exc}", file=sys.stderr)


def verify_numpy() -> None:
    """InsightFace native C-bindings require numpy<2.0.0."""
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        if major >= 2:
            print(f"[FATAL] Incompatible NumPy version {np.__version__}; required 1.26.4", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"[FATAL] Failed to verify NumPy: {exc}", file=sys.stderr)
        sys.exit(1)


def verify_onnxruntime() -> None:
    """Import and verify onnxruntime package and provider API."""
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"[FATAL] Failed to import onnxruntime: {exc}", file=sys.stderr)
        sys.exit(1)

    # 1. Check __file__ (detect namespace package)
    ort_file = getattr(ort, "__file__", None)
    if not ort_file:
        print(
            "[FATAL] onnxruntime resolved to an unpopulated namespace package (__file__ is None). "
            "Stale or corrupted package artifacts detected.",
            file=sys.stderr
        )
        sys.exit(1)

    version = getattr(ort, "__version__", "unknown")

    # 2. Check get_available_providers API
    if not hasattr(ort, "get_available_providers") or not callable(ort.get_available_providers):
        print(
            f"[FATAL] imported onnxruntime ({ort_file}) does not expose callable 'get_available_providers'.",
            file=sys.stderr
        )
        sys.exit(1)

    # 3. Deterministic query
    try:
        providers = ort.get_available_providers()
    except Exception as exc:
        print(f"[FATAL] ort.get_available_providers() raised an exception: {exc}", file=sys.stderr)
        sys.exit(1)

    if not providers:
        print("[FATAL] onnxruntime reports an empty list of execution providers.", file=sys.stderr)
        sys.exit(1)

    # 4. Deterministic output
    print(f"[Verify ORT] __file__   : {ort_file}")
    print(f"[Verify ORT] __version__: {version}")
    print(f"[Verify ORT] providers  : {providers}")
    print("[OK] ONNX Runtime installation contract verified.")


def main() -> None:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    detect_shadowing(app_dir)
    verify_numpy()
    verify_onnxruntime()


if __name__ == "__main__":
    main()
