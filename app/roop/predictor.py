"""Hard execution-provider assertion and startup engine warm-up.

Two failure modes this project has been caught by repeatedly, both of which
present as "it works, just slowly":

1. **ORT drops a provider without raising.**  ``InferenceSession`` returns a
   perfectly working session when the TensorRT EP's DLLs are missing or
   version-mismatched -- it logs to stderr and quietly continues on CUDA or
   CPU.  ``face_analyser._openvino_usable`` already records the measured
   evidence for this against onnxruntime-openvino (four device requests, four
   working sessions, three of them silently on CPU).  The mechanism is not
   OpenVINO-specific.  Neither ``build_session_with_fallback`` nor any
   try/except can see it, because nothing fails: the ONLY reliable signal is
   asking the constructed session which providers it actually ended up with.

2. **TensorRT allocates on first inference, not at session build.**  A session
   that built cleanly can still spend minutes compiling an engine, or fail, on
   the first real frame.  Paying that on a dummy tensor at startup keeps the
   cost off the video clock and surfaces the failure before the render is
   already half written.

Scope of the assertion, deliberately narrow: it fires only when TensorRT was
*requested for that session* and did not register.  It therefore cannot fire on
the sub-7GB tier (``backend_manager.resolve_provider_names`` strips TensorRT
before any session is built) nor on the models ``precision_policy`` routes to
CUDA/FP32 on purpose (inswapper's FP16 smudge, the ESRGAN upscaler's black
frames).  In both of those cases TensorRT is absent from the *requested* list,
which is the input to this check.

``ROOP_STRICT_PROVIDER=0`` downgrades the raise to a loud warning plus a
recorded degradation, for the case where somebody needs a broken environment to
limp rather than stop.  ``ROOP_WARMUP=0`` skips the dummy pass.
"""

from __future__ import annotations

import glob
import os
import sys
import threading
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

TENSORRT_EP = "TensorrtExecutionProvider"
CUDA_EP = "CUDAExecutionProvider"

# Fallback spatial sizes for a model whose input axes are symbolic.  These are
# the two shapes this pipeline actually feeds: 112 for the ArcFace-family
# recognition/identity crops, 256 for the swapper and the GPEN 256 restorers.
DEFAULT_WARMUP_HW = (112, 256)

_lock = threading.RLock()
_warmed: set = set()
_asserted: set = set()


class ProviderAssertionError(RuntimeError):
    """A requested execution provider did not register on the built session."""


# --------------------------------------------------------------------------
# provider names
# --------------------------------------------------------------------------

def _name(provider) -> str:
    """Normalise a provider entry to its bare name.

    ORT accepts both ``"CUDAExecutionProvider"`` and the
    ``("CUDAExecutionProvider", {...options...})`` tuple form; this project
    uses the tuple form wherever provider options are set (TensorRT cache
    paths, precision flags), so a plain ``in`` test against the requested list
    is not enough.
    """
    if isinstance(provider, (tuple, list)) and provider:
        provider = provider[0]
    return str(provider)


def provider_names(providers: Optional[Iterable]) -> List[str]:
    return [_name(p) for p in (providers or ())]


def wants_tensorrt(providers: Optional[Iterable]) -> bool:
    return any("tensorrt" in n.lower() for n in provider_names(providers))


# --------------------------------------------------------------------------
# environment diagnostics
# --------------------------------------------------------------------------

# What has to be loadable for the TensorRT EP to register.  Globs, because the
# CUDA/cuDNN/TensorRT sonames all carry a version in the filename and the whole
# point of this report is to show WHICH version is present.
_WINDOWS_LIBS = {
    "onnxruntime TensorRT EP": ("onnxruntime_providers_tensorrt.dll",),
    "onnxruntime shared EP":   ("onnxruntime_providers_shared.dll",),
    "TensorRT":                ("nvinfer.dll", "nvinfer_*.dll"),
    "TensorRT ONNX parser":    ("nvonnxparser.dll", "nvonnxparser_*.dll"),
    "cuDNN":                   ("cudnn64_*.dll", "cudnn_graph64_*.dll"),
    "cuBLAS":                  ("cublas64_*.dll",),
    "CUDA runtime":            ("cudart64_*.dll",),
}

_POSIX_LIBS = {
    "onnxruntime TensorRT EP": ("libonnxruntime_providers_tensorrt.so",),
    "onnxruntime shared EP":   ("libonnxruntime_providers_shared.so",),
    "TensorRT":                ("libnvinfer.so*",),
    "TensorRT ONNX parser":    ("libnvonnxparser.so*",),
    "cuDNN":                   ("libcudnn.so*",),
    "cuBLAS":                  ("libcublas.so*",),
    "CUDA runtime":            ("libcudart.so*",),
}


def _search_roots() -> List[str]:
    """Directories a loader would actually look in, in rough priority order."""
    roots: List[str] = []
    seen = set()

    def add(path):
        if not path:
            return
        path = os.path.normpath(str(path))
        key = path.lower()
        if key not in seen and os.path.isdir(path):
            seen.add(key)
            roots.append(path)

    # onnxruntime ships its own EP shared objects beside the package, and on
    # Windows those are found through the DLL directories ORT adds at import
    # time rather than through PATH -- so probing PATH alone would report a
    # missing library that is in fact perfectly loadable.
    try:
        import onnxruntime
        ort_dir = os.path.dirname(os.path.abspath(onnxruntime.__file__))
        add(ort_dir)
        add(os.path.join(ort_dir, "capi"))
    except Exception:
        pass
    for var in ("CUDA_PATH", "CUDNN_PATH", "TENSORRT_PATH", "TRT_PATH"):
        base = os.environ.get(var)
        if base:
            add(base)
            add(os.path.join(base, "bin"))
            add(os.path.join(base, "lib"))
            add(os.path.join(base, "lib64"))
    # The pip-installed CUDA/TensorRT wheels put their libraries under
    # site-packages/nvidia/*/{bin,lib}; that is how a Pinokio venv usually gets
    # them, and none of those directories is on PATH.
    for site in list(sys.path):
        nvidia = os.path.join(site, "nvidia")
        if os.path.isdir(nvidia):
            try:
                entries = sorted(os.listdir(nvidia))
            except OSError:
                entries = []
            for entry in entries:
                add(os.path.join(nvidia, entry, "bin"))
                add(os.path.join(nvidia, entry, "lib"))
        for pkg in ("tensorrt", "tensorrt_libs"):
            add(os.path.join(site, pkg))
    sep = os.pathsep
    for entry in (os.environ.get("PATH", "") or "").split(sep):
        add(entry)
    for entry in (os.environ.get("LD_LIBRARY_PATH", "") or "").split(sep):
        add(entry)
    return roots


def environment_report() -> dict:
    """Which TensorRT/CUDA libraries are resolvable, and where from."""
    libs = _WINDOWS_LIBS if os.name == "nt" else _POSIX_LIBS
    roots = _search_roots()
    found, missing = {}, []
    for label, patterns in libs.items():
        hit = None
        for root in roots:
            for pattern in patterns:
                try:
                    matches = sorted(glob.glob(os.path.join(root, pattern)))
                except OSError:
                    matches = []
                if matches:
                    hit = matches[0]
                    break
            if hit:
                break
        if hit:
            found[label] = hit
        else:
            missing.append("%s (%s)" % (label, ", ".join(patterns)))

    try:
        import onnxruntime
        ort_version = onnxruntime.__version__
        available = list(onnxruntime.get_available_providers())
    except Exception as exc:  # pragma: no cover - ORT is a hard dependency
        ort_version, available = "unavailable: %s" % exc, []

    return {
        "onnxruntime": ort_version,
        "available_providers": available,
        "found": found,
        "missing": missing,
        "searched_roots": roots[:20],
        "CUDA_PATH": os.environ.get("CUDA_PATH", ""),
    }


def format_environment(report: Optional[dict] = None) -> str:
    report = report or environment_report()
    lines = [
        "  onnxruntime         : %s" % report["onnxruntime"],
        "  available providers : %s" % (", ".join(report["available_providers"]) or "(none)"),
        "  CUDA_PATH           : %s" % (report["CUDA_PATH"] or "(unset)"),
    ]
    if report["found"]:
        lines.append("  resolved libraries  :")
        for label, path in report["found"].items():
            lines.append("      %-24s %s" % (label, path))
    if report["missing"]:
        lines.append("  MISSING libraries   :")
        for item in report["missing"]:
            lines.append("      %s" % item)
    else:
        lines.append("  MISSING libraries   : (none -- every probed library resolved)")
    lines.append("  searched            :")
    for root in report["searched_roots"]:
        lines.append("      %s" % root)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the assertion
# --------------------------------------------------------------------------

def strict_enabled() -> bool:
    raw = os.environ.get("ROOP_STRICT_PROVIDER", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def warmup_enabled() -> bool:
    raw = os.environ.get("ROOP_WARMUP", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def assert_session_providers(session, requested: Optional[Iterable],
                             tag: str = "model",
                             strict: Optional[bool] = None) -> List[str]:
    """Fail loudly when a requested TensorRT session silently came up elsewhere.

    Returns the session's ACTIVE provider list so a caller can record what ran
    rather than what it asked for.  Never raises for a session that did not ask
    for TensorRT in the first place.
    """
    try:
        active = list(session.get_providers())
    except Exception:
        # A stub or pooled wrapper that does not speak the session protocol is
        # not something to fail a render over.
        return []

    if not wants_tensorrt(requested):
        return active

    if active and "tensorrt" in active[0].lower():
        with _lock:
            new = tag not in _asserted
            _asserted.add(tag)
        if new:
            print("[Provider] %s: TensorRT active (%s)." % (tag, active[0]))
        return active

    report = environment_report()
    message = (
        "%s: TensorRT was REQUESTED but is not the active execution provider.\n"
        "  requested : %s\n"
        "  active    : %s\n"
        "onnxruntime does not raise when an execution provider fails to "
        "register -- it logs and continues on the next one in the list, so this "
        "session would have run on %s at a fraction of the expected speed with "
        "nothing in the render output to show for it.\n"
        "%s\n"
        "Set ROOP_STRICT_PROVIDER=0 to downgrade this to a warning."
        % (tag,
           ", ".join(provider_names(requested)) or "(empty)",
           ", ".join(active) or "(empty)",
           active[0] if active else "an unknown provider",
           format_environment(report))
    )

    if strict is None:
        strict = strict_enabled()
    if strict:
        raise ProviderAssertionError(message)

    print("[Provider] WARNING -- %s" % message)
    try:
        from roop import backend_manager
        backend_manager._record_degradation(
            tag, TENSORRT_EP, active[0] if active else "unknown",
            ProviderAssertionError("TensorRT EP did not register"))
    except Exception:
        pass
    return active


# --------------------------------------------------------------------------
# warm-up
# --------------------------------------------------------------------------

def _concrete_shape(meta, default_hw: Sequence[int]) -> Optional[tuple]:
    """A runnable shape for one input, resolving symbolic axes conservatively."""
    shape = list(getattr(meta, "shape", None) or [])
    if not shape:
        return None
    resolved = []
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif index == 0:
            resolved.append(1)               # batch
        elif index == 1 and len(shape) == 4:
            resolved.append(3)               # channels
        elif len(shape) == 4:
            # Spatial axis: prefer the larger default so a dynamic engine is
            # built for the shape the pipeline actually feeds.
            resolved.append(int(max(default_hw)))
        elif len(shape) == 2:
            resolved.append(512)             # identity embedding
        else:
            return None
    return tuple(resolved)


def _numpy_dtype(meta):
    mapping = {
        "tensor(float)": np.float32, "tensor(float16)": np.float16,
        "tensor(double)": np.float64, "tensor(int64)": np.int64,
        "tensor(int32)": np.int32, "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(getattr(meta, "type", ""), np.float32)


def warmup_session(session, tag: str = "model",
                   default_hw: Sequence[int] = DEFAULT_WARMUP_HW,
                   once: bool = True) -> bool:
    """Push one dummy zeros tensor through *session*.

    TensorRT builds its engine and allocates its context on the FIRST
    INFERENCE, not at session construction -- so without this the first video
    frame of every run pays an engine build (measured in minutes on a cold
    cache in this project) and any build failure lands mid-render.  Returns
    True when the dummy pass completed.
    """
    if once:
        with _lock:
            if tag in _warmed:
                return True
    feed = {}
    try:
        for meta in session.get_inputs():
            shape = _concrete_shape(meta, default_hw)
            if shape is None:
                print("[Warmup] %s: input '%s' has an unresolvable shape %r; skipped."
                      % (tag, meta.name, getattr(meta, "shape", None)))
                return False
            feed[meta.name] = np.zeros(shape, dtype=_numpy_dtype(meta))
        started = time.time()
        session.run(None, feed)
        elapsed = time.time() - started
    except Exception as exc:
        print("[Warmup] %s: dummy pass FAILED (%s: %s). The first real frame "
              "would have hit this instead." % (tag, type(exc).__name__, exc))
        return False
    with _lock:
        _warmed.add(tag)
    shapes = ", ".join("%s%r" % (k, tuple(v.shape)) for k, v in feed.items())
    note = " (engine build)" if elapsed > 5.0 else ""
    print("[Warmup] %s: %s in %.2fs%s" % (tag, shapes, elapsed, note))
    return True


def verify_and_warmup(session, requested: Optional[Iterable], tag: str,
                      default_hw: Sequence[int] = DEFAULT_WARMUP_HW,
                      warmup: bool = True) -> List[str]:
    """Assert the provider, then pay the engine build on a dummy tensor.

    The single helper call sites use: assertion first, because warming a
    session that silently landed on CPU just spends time proving the wrong
    thing.
    """
    active = assert_session_providers(session, requested, tag)
    if warmup and warmup_enabled():
        warmup_session(session, tag, default_hw)
    return active


def reset() -> None:
    """Forget which tags were warmed/asserted (tests, and model reloads)."""
    with _lock:
        _warmed.clear()
        _asserted.clear()
