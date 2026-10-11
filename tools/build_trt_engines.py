"""Download the offline model set and pre-build TensorRT engine caches.

This is an explicit preparation step, not application startup work.  It downloads
only the three models used by the calibrated TensorRT path, then constructs one
ONNX Runtime TensorRT session per model so the engine and timing caches are
serialized under ``models/trt_cache``.

Run from the repository root::

    python tools/build_trt_engines.py

The script is hardware-aware: the RTX 4070 tier builds with a 2048 MiB workspace
and GPUs below 7 GiB (the RTX 3060 laptop) with 1024 MiB, which keeps a build
from running out of memory next to a loaded app.  Set
``ROOP_TRT_WORKSPACE_BYTES`` to override that deliberately.  Use ``--offline``
after the models have been staged locally to guarantee that no network request
is attempted.

Dynamic batch.  Models marked ``"dynamic"`` in the registry (inswapper, SCRFD,
GPEN, RestoreFormer++) are static batch-1 exports, which a min 1 / opt 4 / max 8
optimization profile cannot be built against.  Each is first converted by
``tools/convert_dynamic_shapes.py`` to ``<name>_dynamic.onnx`` -- and that
conversion is only accepted when a batch of N is numerically equal to N single
runs.  A model the surgery cannot make batch-dynamic (GPEN: styled convolutions)
is built as static batch 1 and the summary says so; it is never presented as
dynamic.

Precision.  FP16 is the default.  The swapper is the exception: inswapper's
activations overflow in FP16 and TensorRT emits garbage colours without any
warning, which is why the app itself forces its swapper to FP32 (commit
52e7750).  ``ROOP_SWAP_FP16=1`` opts back in; ``ROOP_TRT_FP16=0`` builds everything
FP32.

An engine is only built at the first INFERENCE, not when the session is created,
so after creating the session this tool runs it at every batch size in the
profile (and checks the outputs are finite).  A build that wrote no engine file
fails.
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
from typing import Any, Mapping, NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))   # convert_dynamic_shapes lives beside this file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_trt")

MODEL_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
        "dynamic": True,
        "swapper": True,
        # Overflows in FP16 (rainbow output, no warning); the app forces its swapper to FP32.
        "fp16": False,
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
        "dynamic": True,
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
        "dynamic": True,
        "shapes": {
            "min": "input:1x3x640x640",
            "opt": "input:4x3x640x640",
            "max": "input:8x3x640x640",
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
        "dynamic": True,
        "shapes": {"min": "input:1x3x256x256", "opt": "input:2x3x256x256", "max": "input:4x3x256x256"},
    },
    "restoreformer_plus_plus.onnx": {
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx",
        "family": "restoreformer_pp",
        "dynamic": True,
        "shapes": {"min": "input:1x3x512x512", "opt": "input:2x3x512x512", "max": "input:4x3x512x512"},
    },
    "xseg.onnx": {
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx",
        "family": "dfl_xseg",
        "fp16": False,          # per-model FP32 (xseg mixed measured IoU min 0.81); see precision_policy
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
    return (1024 if 0 < total_vram < 7 * 1024**3 else 2048) * 1024**2


class ResolvedModel(NamedTuple):
    """What will actually be built for one registry entry."""

    path: Path
    shapes: dict           # {"min": "name:1x3x..", "opt": .., "max": ..} for THIS graph
    dynamic: bool          # True only when a verified _dynamic.onnx is being used
    note: str


def _static_profile(shapes: Mapping[str, str]) -> dict[str, str]:
    """Collapse a min/opt/max profile to its min tier (batch 1 for a static export)."""
    return {tier: str(shapes["min"]) for tier in ("min", "opt", "max")}


def _verification_size(shapes: Mapping[str, str]) -> int:
    """Spatial size to verify a dynamic-spatial model at (SCRFD: 640)."""
    for entry in str(shapes["opt"]).split(","):
        if ":" in entry:
            dims = [int(v) for v in entry.split(":", 1)[1].split("x")]
            if len(dims) == 4:
                return max(dims[2], dims[3]) if dims[1] <= 4 else max(dims[1], dims[2])
    return 640


def _resolve_model(model_name: str, config: Mapping[str, Any]) -> ResolvedModel:
    """Download if needed, then convert to a verified dynamic-batch graph when asked to."""
    model_path = download_file(
        config["url"], MODELS_DIR / model_name, tuple(config.get("fallback_urls", ()))
    )
    shapes = dict(config["shapes"])
    if not config.get("dynamic"):
        return ResolvedModel(model_path, shapes, False, "static graph, profile as registered")
    from convert_dynamic_shapes import prepare_dynamic_model

    batches = sorted({int(str(shapes[tier]).split(":", 1)[1].split("x")[0]) for tier in ("min", "opt", "max")})
    try:
        dynamic = prepare_dynamic_model(model_path, batches=batches, size=_verification_size(shapes))
    except Exception as error:   # not convertible (RuntimeError) or unreadable: either way, say it and go static
        logger.warning(
            "%s cannot be made batch-dynamic (%s). Building it as STATIC batch 1; the profile is "
            "collapsed to %s.", model_name, error, _static_profile(shapes)["min"],
        )
        return ResolvedModel(model_path, _static_profile(shapes), False,
                             f"STATIC batch 1 (not batch-convertible: {str(error)[:160]})")
    return ResolvedModel(dynamic, shapes, True, f"dynamic batch via {dynamic.name}, verified at batches {batches}")


def _fp16_for(config: Mapping[str, Any]) -> bool:
    """FP16 unless disabled globally or the model is registered as FP32-only."""
    if os.environ.get("ROOP_TRT_FP16", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    if config.get("fp16", True):
        return True
    # The swapper only: the app's own opt-in for its (broken) FP16 swapper.
    swap_opt_in = os.environ.get("ROOP_SWAP_FP16", "0").strip().lower() in {"1", "true", "yes", "on"}
    return bool(config.get("swapper")) and swap_opt_in


def _provider_options(resolved: ResolvedModel, config: Mapping[str, Any]) -> dict[str, Any]:
    device_id = _device_id()
    total_vram = _total_vram_bytes(device_id)
    workspace = _workspace_bytes(total_vram)
    shapes = _resolved_profile_shapes(resolved.path, {"shapes": resolved.shapes})
    fp16 = _fp16_for(config)
    options = {
        "device_id": device_id,
        "trt_max_workspace_size": workspace,
        "trt_fp16_enable": fp16,
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
        "trt_engine_cache_prefix": _cache_prefix(resolved.path),
    }
    logger.info(
        "%s: GPU tier=%s, workspace=%d MiB, precision=%s, device=%d",
        resolved.path.name,
        "laptop-sub7GB" if 0 < total_vram < 7 * 1024**3 else "desktop/default",
        workspace // 1024**2,
        "FP16" if fp16 else "FP32",
        device_id,
    )
    return options


def _cache_prefix(model_path: Path) -> str:
    return f"roop_{model_path.stem}_{_sha256(model_path)[:12]}"


def _engine_files(prefix: str) -> list[Path]:
    return sorted(CACHE_DIR.glob(f"{prefix}*.engine")) if CACHE_DIR.is_dir() else []


def _profile_batches(shapes: Mapping[str, str]) -> list[int]:
    """Every batch size worth exercising: the profile's min/opt/max plus 1, 4, 8 inside it."""
    firsts = {tier: int(str(shapes[tier]).split(",")[0].split(":", 1)[1].split("x")[0])
              for tier in ("min", "opt", "max")}
    low, high = firsts["min"], firsts["max"]
    wanted = set(firsts.values()) | {b for b in (1, 4, 8) if low <= b <= high}
    return sorted(wanted)


def _exercise_profile(session: Any, shapes: Mapping[str, str], resolved_name: str) -> list[int]:
    """Run the session at each batch size; a TensorRT engine is only built by the first run."""
    import numpy as np

    native = _parse_native_profiles(shapes)
    batches = _profile_batches(shapes)
    rng = np.random.RandomState(7)
    done: list[int] = []
    for batch in batches:
        feeds = {}
        for meta in session.get_inputs():
            tiers = native[meta.name]
            dims = list(tiers["opt"])
            if tiers["min"][0] != tiers["max"][0]:
                dims[0] = batch
            dtype = np.float16 if "float16" in str(meta.type) else np.float32
            feeds[meta.name] = (rng.standard_normal(dims) * 0.5).astype(dtype)
        try:
            outputs = session.run(None, feeds)
        except Exception as error:
            raise RuntimeError(
                f"{resolved_name}: batch {batch} was rejected inside the profile "
                f"{shapes['min']} .. {shapes['max']}: {error}") from error
        for index, output in enumerate(outputs):
            if np.issubdtype(output.dtype, np.floating) and not np.isfinite(output).all():
                raise RuntimeError(
                    f"{resolved_name}: output {index} is not finite at batch {batch} - the engine "
                    f"produced NaN/Inf (FP16 overflow?); rebuild it FP32")
        done.append(batch)
    return done


def compile_engine(model_name: str, config: Mapping[str, Any], ort: Any) -> dict[str, Any]:
    """Create one strict TensorRT session, build its engine across the profile, verify it."""
    from roop.strict_session import create_strictly_accelerated_session

    resolved = _resolve_model(model_name, config)
    options = _provider_options(resolved, config)
    prefix = options["trt_engine_cache_prefix"]
    before = len(_engine_files(prefix))
    logger.info(
        "Building %s TensorRT engine cache for %s (%s); this may take several minutes...",
        "FP16" if options["trt_fp16_enable"] else "FP32", model_name, resolved.note,
    )
    start_time = time.time()
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = None
    try:
        session = create_strictly_accelerated_session(
            str(resolved.path),
            providers=[("TensorrtExecutionProvider", options), ("CUDAExecutionProvider", {})],
            sess_options=sess_opts,
            tag=f"build:{Path(model_name).stem}",
        )
        active = [str(provider) for provider in session.get_providers()]
        if active[0] != "TensorrtExecutionProvider":
            raise RuntimeError(
                f"TensorRT is not the active provider for {model_name}; session providers: {active!r}"
            )
        profile = {"min": options["trt_profile_min_shapes"], "opt": options["trt_profile_opt_shapes"],
                   "max": options["trt_profile_max_shapes"]}
        exercised = _exercise_profile(session, profile, model_name)
        engines = _engine_files(prefix)
        if len(engines) <= 0:
            raise RuntimeError(
                f"{model_name}: the session ran but no engine file matching {prefix}* exists in "
                f"{CACHE_DIR}; nothing was cached")
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
    logger.info("Engine build completed in %.2f seconds for %s.", elapsed, model_name)
    return {
        "model": model_name, "built_from": resolved.path.name, "dynamic": resolved.dynamic,
        "note": resolved.note, "precision": "FP16" if options["trt_fp16_enable"] else "FP32",
        "workspace_mib": options["trt_max_workspace_size"] // 1024**2,
        "profile": profile, "batches_run": exercised,
        "engines_new": len(_engine_files(prefix)) - before, "seconds": round(elapsed, 1),
    }


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

    resolved = _resolve_model(model_name, config)
    engine_path = CACHE_DIR / "native" / f"{Path(model_name).stem}.engine"
    logger.info("Building native precision-segmented TensorRT engine for %s (%s)", model_name, resolved.note)
    build_native_engine(
        resolved.path,
        engine_path,
        model_family=config.get("family", Path(model_name).stem),
        workspace_bytes=_workspace_bytes(_total_vram_bytes(_device_id())),
        input_profiles=_parse_native_profiles(resolved.shapes),
    )


def main(argv: Optional[list[str]] = None) -> int:
    global _OFFLINE
    args = _parse_args(argv)
    _OFFLINE = bool(args.offline)
    logger.warning(
        "NOTE: the engines this tool builds are NOT read by the app. It writes a flat "
        "cache at %s for inswapper_128 / GPEN-512 / scrfd_2.5g, while the app loads "
        "app/models/trt_cache/<GPU+driver+version+precision namespace>/ for the models in "
        "config.yaml. To pre-build what the app will actually use, run "
        "tools/prebuild_engines.py instead.", CACHE_DIR)
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
        summary = [compile_engine(model_name, config, ort) for model_name, config in selected.items()]
        for row in summary:
            logger.info(
                "  %-28s %-34s %-7s ws=%dMiB batches=%s engines+%d  (%s)",
                row["model"], row["built_from"], row["precision"], row["workspace_mib"],
                row["batches_run"], row["engines_new"], "dynamic" if row["dynamic"] else row["note"])
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
