"""Download the offline model set and pre-build TensorRT engine caches.

This is an explicit preparation step, not application startup work.  It downloads
only the three models used by the calibrated TensorRT path, then constructs one
ONNX Runtime TensorRT session per model so the engine and timing caches are
serialized under ``models/trt_cache``.

Run from the repository root::

    python tools/build_trt_engines.py

The script is hardware-aware: the RTX 4070 tier uses the established 4 GiB
workspace cap, while GPUs below 7 GiB use the 1.5 GiB laptop cap.  Set
``ROOP_TRT_WORKSPACE_BYTES`` to override that cap deliberately.  Use
``--offline`` after the models have been staged locally to guarantee that no
network request is attempted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_trt")

MODEL_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
        "shapes": {
            "min": "target:1x3x128x128,source:1x512",
            "opt": "target:4x3x128x128,source:4x512",
            "max": "target:8x3x128x128,source:8x512",
        },
    },
    "GPEN-BFR-512.onnx": {
        "url": "https://huggingface.co/yangxy/GPEN/resolve/main/GPEN-BFR-512.onnx",
        # The requested upstream mirror currently responds 401 in a clean
        # non-authenticated session. This public repository mirror is the URL
        # already used by the application's normal model preflight.
        "fallback_urls": (
            "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
        ),
        "shapes": {
            "min": "input:1x3x512x512",
            "opt": "input:2x3x512x512",
            "max": "input:4x3x512x512",
        },
    },
    "scrfd_2.5g_kps.onnx": {
        "url": "https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/scrfd_2.5g_kps.onnx",
        # The requested Hugging Face path currently responds 404. These public
        # mirrors contain the same named SCRFD KPS export and were validated
        # with HTTP HEAD before being added as fallback sources.
        "fallback_urls": (
            "https://raw.githubusercontent.com/yangjian1218/scrfd_onnx_tensorrt/master/models/scrfd/scrfd_2.5g_kps.onnx",
        ),
        "shapes": {
            "min": "input:1x3x640x640",
            "opt": "input:1x3x640x640",
            "max": "input:1x3x640x640",
        },
    },
}

# Keep the original ORT-EP preparation set stable for existing automation.
# Native precision-segmented builds cover every requested face-model family.
NATIVE_MODEL_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    **MODEL_REGISTRY,
    "gpen_bfr_256.onnx": {
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_256.onnx",
        "family": "gpen",
        "shapes": {"min": "input:1x3x256x256", "opt": "input:2x3x256x256", "max": "input:4x3x256x256"},
    },
    "restoreformer_plus_plus.onnx": {
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx",
        "family": "restoreformer_pp",
        "shapes": {"min": "input:1x3x512x512", "opt": "input:2x3x512x512", "max": "input:4x3x512x512"},
    },
    "xseg.onnx": {
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx",
        "family": "dfl_xseg",
        "shapes": {"min": "input:1x256x256x3", "opt": "input:1x256x256x3", "max": "input:1x256x256x3"},
    },
}

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("ROOP_TRT_MODELS_DIR", REPO_ROOT / "models")).resolve()
CACHE_DIR = Path(
    os.environ.get("ROOP_TRT_CACHE_DIR", MODELS_DIR / "trt_cache")
).resolve()
DOWNLOAD_TIMEOUT_S = float(os.environ.get("ROOP_TRT_DOWNLOAD_TIMEOUT", "120"))
_USER_AGENT = "roop-ultimate-trt-builder/1"
_OFFLINE = False


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not access the network; every registered model must already exist",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(NATIVE_MODEL_REGISTRY),
        help="build only this registered model; repeat the option for multiple models",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="use the native TensorRT builder with precision segmentation and validation",
    )
    return parser.parse_args(argv)


def _selected_models(
    names: Optional[list[str]], *, native: bool = False
) -> dict[str, Mapping[str, Any]]:
    registry = NATIVE_MODEL_REGISTRY if native or any(
        name not in MODEL_REGISTRY for name in (names or ())
    ) else MODEL_REGISTRY
    if not names:
        return dict(registry)
    return {name: registry[name] for name in names}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_input_names(model_path: Path) -> Optional[tuple[str, ...]]:
    """Return real graph input names, excluding ONNX initializer inputs."""
    try:
        import onnx

        model = onnx.load(str(model_path), load_external_data=False)
        initializers = {initializer.name for initializer in model.graph.initializer}
        names = tuple(
            value_info.name
            for value_info in model.graph.input
            if value_info.name not in initializers
        )
        return names or None
    except Exception as error:
        logger.warning("Could not inspect ONNX input names for %s: %s", model_path.name, error)
        return None


def _resolved_profile_shapes(model_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve registry profile names against the exact downloaded graph.

    The requested SCRFD registry uses the conventional ``input`` label, while
    some public exports name the sole graph input ``input.1``. TensorRT requires
    the actual graph name, so a one-input profile is safely remapped without
    changing its dimensions. Ambiguous multi-input mismatches fail loudly.
    """
    configured = dict(config["shapes"])
    actual = _model_input_names(model_path)
    if not actual:
        return configured
    resolved: dict[str, str] = {}
    for profile_name, shape_text in configured.items():
        entries = [entry.strip() for entry in str(shape_text).split(",") if entry.strip()]
        names = [entry.split(":", 1)[0] for entry in entries]
        if set(names).issubset(actual):
            resolved[profile_name] = shape_text
            continue
        if len(names) == 1 and len(actual) == 1:
            configured_name = names[0]
            actual_name = actual[0]
            resolved[profile_name] = shape_text.replace(
                f"{configured_name}:", f"{actual_name}:", 1
            )
            logger.warning(
                "%s %s profile input %r does not exist; using the sole graph input %r.",
                model_path.name,
                profile_name,
                configured_name,
                actual_name,
            )
            continue
        missing = sorted(set(actual) - set(names))
        raise RuntimeError(
            f"{profile_name} profile inputs for {model_path.name} do not match graph "
            f"inputs; missing profiles for {missing!r}"
        )
    return resolved


def download_file(
    url: str,
    dest_path: str | os.PathLike[str],
    fallback_urls: tuple[str, ...] = (),
) -> Path:
    """Download one model atomically, or return the existing non-empty file.

    A ``.part`` file is used so an interrupted transfer can never be mistaken
    for a complete ONNX model on the next invocation.
    """
    destination = Path(dest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        logger.info("Model %s already present.", destination.name)
        return destination
    if _OFFLINE:
        raise RuntimeError(
            f"Offline mode requires the model to exist locally: {destination}"
        )

    errors = []
    candidates = (url, *fallback_urls)
    for index, candidate_url in enumerate(candidates):
        partial = destination.with_name(destination.name + ".part")
        logger.info("Downloading %s ...", candidate_url)
        request = urllib.request.Request(candidate_url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
                expected = int(response.headers.get("Content-Length", "0") or 0)
                received = 0
                with partial.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        received += len(chunk)
            if received <= 0:
                raise IOError("download returned no bytes")
            if expected and received != expected:
                raise IOError(
                    f"incomplete download: received {received} of {expected} bytes"
                )
            os.replace(partial, destination)
            logger.info("Downloaded successfully: %s (%d bytes).", destination.name, received)
            return destination
        except (OSError, IOError, urllib.error.URLError, TimeoutError, ValueError) as error:
            errors.append(f"{candidate_url}: {error}")
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
            if index + 1 < len(candidates):
                logger.warning("Download mirror failed, trying fallback: %s", error)
    raise RuntimeError(
        f"could not download {destination}; attempts: {' | '.join(errors)}"
    )


def _prepare_runtime() -> None:
    """Register the app's packaged CUDA/TensorRT DLL directories before ORT import."""
    app_dir = str(REPO_ROOT / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    try:
        from roop.trt_session_builder import prepare_tensorrt_runtime
    except ImportError:
        # The builder can still run in an environment where the app package is
        # not installed as a module. ORT will provide the useful import error.
        return
    prepare_tensorrt_runtime()


def load_onnxruntime() -> Any:
    """Load ORT after the repository's Windows DLL search paths are prepared."""
    _prepare_runtime()
    try:
        import onnxruntime as ort
    except ImportError as error:
        if _reexec_with_app_environment():
            raise AssertionError("environment re-exec unexpectedly returned")
        raise RuntimeError(
            "onnxruntime-gpu with TensorRT support is required. Run this command "
            "with the application's app/env interpreter."
        ) from error
    lister = getattr(ort, "get_available_providers", None)
    providers = list(lister()) if callable(lister) else []
    if "TensorrtExecutionProvider" not in providers:
        if _reexec_with_app_environment():
            # os.execv replaces this process and therefore never returns.
            raise AssertionError("environment re-exec unexpectedly returned")
        raise RuntimeError(
            "TensorRTExecutionProvider is unavailable in this interpreter. "
            f"Available providers: {providers!r}"
        )
    return ort


def _reexec_with_app_environment() -> bool:
    """Replace a CPU-only system Python with the project's TensorRT venv.

    Pinokio supplies the application venv when it launches the app, but a user
    running the documented root command from a normal terminal may resolve the
    system Python first. Re-executing only when the current interpreter lacks
    TensorRT keeps ``python tools/build_trt_engines.py`` reproducible without
    changing the shell's activation state. Set ``ROOP_TRT_NO_REEXEC=1`` to
    request the stricter fail-fast behavior for diagnostics or CI.
    """
    if os.environ.get("ROOP_TRT_NO_REEXEC", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return False
    current = Path(sys.executable).resolve()
    candidates = (
        REPO_ROOT / "app" / "env" / "Scripts" / "python.exe",
        REPO_ROOT / "app" / "env" / "bin" / "python",
    )
    for candidate in candidates:
        if not candidate.is_file() or candidate.resolve() == current:
            continue
        logger.info("Re-running with the repository TensorRT environment: %s", candidate)
        os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])
    return False


def _device_id() -> int:
    try:
        return max(0, int(os.environ.get("ROOP_TRT_DEVICE_ID", "0")))
    except (TypeError, ValueError):
        return 0


def _total_vram_bytes(device_id: int) -> int:
    """Read VRAM without importing ORT after it has been initialized."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(device_id).total_memory)
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        logger.warning("Could not read GPU memory, using desktop defaults: %s", error)
    return 0


def _workspace_bytes(total_vram: int) -> int:
    override = os.environ.get("ROOP_TRT_WORKSPACE_BYTES")
    if override:
        try:
            return max(256 * 1024 * 1024, int(override))
        except ValueError as error:
            raise ValueError("ROOP_TRT_WORKSPACE_BYTES must be an integer") from error
    # The sub-7 GiB tier is the RTX 3060 laptop profile from AGENTS.md.
    return (1536 if 0 < total_vram < 7 * 1024**3 else 4096) * 1024**2


def _provider_options(model_name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    device_id = _device_id()
    total_vram = _total_vram_bytes(device_id)
    workspace = _workspace_bytes(total_vram)
    shapes = _resolved_profile_shapes(MODELS_DIR / model_name, config)
    options = {
        "device_id": device_id,
        "trt_max_workspace_size": workspace,
        "trt_fp16_enable": os.environ.get("ROOP_TRT_FP16", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(CACHE_DIR),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(CACHE_DIR),
        "trt_builder_optimization_level": 3,
        "trt_profile_min_shapes": shapes["min"],
        "trt_profile_opt_shapes": shapes["opt"],
        "trt_profile_max_shapes": shapes["max"],
        # A shared cache directory is safe because every model gets a stable
        # prefix. The model hash also prevents stale files after replacement.
        "trt_engine_cache_prefix": f"roop_{Path(model_name).stem}_{_sha256(Path(MODELS_DIR / model_name))[:12]}",
    }
    logger.info(
        "%s: GPU tier=%s, workspace=%d MiB, device=%d",
        model_name,
        "laptop-sub7GB" if 0 < total_vram < 7 * 1024**3 else "desktop/default",
        workspace // 1024**2,
        device_id,
    )
    return options


def compile_engine(model_name: str, config: Mapping[str, Any], ort: Any) -> None:
    """Construct one TensorRT-backed ORT session and release it cleanly."""
    model_path = download_file(
        config["url"], MODELS_DIR / model_name, tuple(config.get("fallback_urls", ()))
    )
    logger.info(
        "Building FP16 TensorRT engine cache for %s (this may take several minutes)...",
        model_name,
    )
    start_time = time.time()
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = None
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=[
                ("TensorrtExecutionProvider", _provider_options(model_name, config)),
                ("CUDAExecutionProvider", {}),
            ],
        )
        active = [str(provider) for provider in session.get_providers()]
        if "TensorrtExecutionProvider" not in active:
            raise RuntimeError(
                f"TensorRT was not active for {model_name}; session providers: {active!r}"
            )
    except Exception as error:
        raise RuntimeError(f"engine build failed for {model_name}: {error}") from error
    finally:
        # ORT flushes serialized TensorRT engines and timing data during session
        # teardown. Keep this explicit because the builder processes models in
        # sequence and must not retain three contexts at once.
        if session is not None:
            del session
        gc.collect()
    elapsed = time.time() - start_time
    logger.info(
        "Engine build completed in %.2f seconds for %s.", elapsed, model_name
    )


def _parse_native_profiles(shapes: Mapping[str, Any]) -> dict[str, dict[str, tuple[int, ...]]]:
    """Convert the registry's ORT profile strings to native TRT shapes."""
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for tier, text in shapes.items():
        for entry in str(text).split(","):
            if not entry.strip() or ":" not in entry:
                continue
            name, encoded = entry.split(":", 1)
            result.setdefault(name, {})[str(tier)] = tuple(
                int(value) for value in encoded.split("x")
            )
    return result


def compile_native_engine(model_name: str, config: Mapping[str, Any]) -> None:
    """Build one precision-segmented native TensorRT engine."""
    from roop.trt_graph_surgery import build_native_engine

    model_path = download_file(
        config["url"], MODELS_DIR / model_name, tuple(config.get("fallback_urls", ()))
    )
    engine_path = CACHE_DIR / "native" / f"{Path(model_name).stem}.engine"
    logger.info("Building native precision-segmented TensorRT engine for %s", model_name)
    build_native_engine(
        model_path,
        engine_path,
        model_family=config.get("family", Path(model_name).stem),
        workspace_bytes=_workspace_bytes(_total_vram_bytes(_device_id())),
        input_profiles=_parse_native_profiles(config.get("shapes", {})),
    )


def main(argv: Optional[list[str]] = None) -> int:
    global _OFFLINE
    args = _parse_args(argv)
    _OFFLINE = bool(args.offline)
    selected = _selected_models(args.model, native=bool(args.native))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Stage every download before allocating GPU contexts. A later offline run
    # therefore performs no network access and starts directly at compilation.
    for model_name, config in selected.items():
        download_file(
            config["url"],
            MODELS_DIR / model_name,
            tuple(config.get("fallback_urls", ())),
        )

    if args.native:
        for model_name, config in selected.items():
            compile_native_engine(model_name, config)
    else:
        ort = load_onnxruntime()
        for model_name, config in selected.items():
            compile_engine(model_name, config, ort)
    logger.info("All TensorRT engines successfully built and cached in %s.", CACHE_DIR)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.error("Engine build interrupted.")
        raise SystemExit(130)
    except Exception as error:
        logger.error("TensorRT builder failed: %s", error)
        raise SystemExit(1)
