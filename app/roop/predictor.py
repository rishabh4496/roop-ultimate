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

Scope of the assertion, deliberately narrow, and keyed off the REQUESTED list:

* TensorRT requested and not the first active provider -> fail.  It does not
  fire on models that ``precision_policy`` intentionally routes to CUDA/FP32
  (inswapper's FP16 smudge, the ESRGAN upscaler's black frames): TensorRT is
  absent from the requested list there.
* CUDA or TensorRT requested and the session is CPU-only -> fail.  A session
  that asked for no GPU provider at all (``cpu``, ``force_cpu``, a CPU-only
  model such as LivePortrait's stock GridSample) can never trip it.  A CUDA
  session that merely runs a few unsupported nodes on the CPU EP still lists
  CUDA first and is, correctly, not flagged.

``ROOP_STRICT_PROVIDER=0`` downgrades the raise to a loud warning plus a
recorded degradation, for the case where somebody needs a broken environment to
limp rather than stop.  ``ROOP_WARMUP=0`` skips the dummy pass.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

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


def _is_gpu_name(name: str) -> bool:
    lowered = str(name).lower()
    return "tensorrt" in lowered or "cuda" in lowered


def wants_gpu(providers: Optional[Iterable]) -> bool:
    """True when a CUDA or TensorRT provider was REQUESTED for the session."""
    return any(_is_gpu_name(n) for n in provider_names(providers))


def has_gpu_active(active: Optional[Iterable]) -> bool:
    """True when the session's ACTIVE list still holds a CUDA/TensorRT provider.

    ORT lists ``CPUExecutionProvider`` last on every session, so "CPU is in the
    list" proves nothing; the defect is the absence of anything before it.
    """
    return any(_is_gpu_name(n) for n in (active or ()))


def iter_ort_sessions(obj):
    """Yield every onnxruntime-style session reachable from *obj*.

    Session factories in this project return a bare ``InferenceSession``, an
    insightface model (``.session``) or a whole ``FaceAnalysis`` (``.models``,
    each with ``.session``).  Anything that speaks no session protocol yields
    nothing, which keeps stubs and pooled wrappers out of the check.
    """
    if obj is None:
        return
    if callable(getattr(obj, "get_providers", None)):
        yield "", obj
        return
    inner = getattr(obj, "session", None)
    if inner is not None and callable(getattr(inner, "get_providers", None)):
        yield "", inner
        return
    models = getattr(obj, "models", None)
    if isinstance(models, dict):
        for name, model in models.items():
            inner = getattr(model, "session", None)
            if inner is not None and callable(getattr(inner, "get_providers", None)):
                yield str(name), inner


def verify_built(obj, requested: Optional[Iterable], tag: str) -> None:
    """Check every session reachable from a factory's return value.

    Used by ``backend_manager.build_session_with_fallback`` so the detectors and
    ``buffalo_l`` -- which build through it and never called the assertion --
    cannot come up CPU-only unnoticed.  A TensorRT->CUDA drop is recorded and
    printed but not fatal here (CUDA is still the GPU); CPU-only is fatal.
    """
    if not wants_gpu(requested):
        return
    checked = False
    for label, session in iter_ort_sessions(obj):
        checked = True
        assert_session_providers(session, requested,
                                 "%s/%s" % (tag, label) if label else tag,
                                 check_tensorrt=False)
    if checked:
        _log_memory(tag)


# --------------------------------------------------------------------------
# device memory + what is bound, for the startup log
# --------------------------------------------------------------------------

_MIB = 1024 * 1024
_BOUND: dict = {}
_last_used_mib: List[Optional[float]] = [None]


def device_memory(device_id: Optional[int] = None) -> Optional[dict]:
    """Device-wide VRAM {used, free, total} in MiB, or None when unreadable.

    Read from the driver (``cudaMemGetInfo``), so it includes every allocation
    ORT's arenas and TensorRT contexts have made -- which ``torch.cuda.memory_*``
    would not.  Only consults torch when it is already imported: this module must
    never be the thing that drags CUDA into a process.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        if device_id is None:
            try:
                import roop.globals as _g
                device_id = int(getattr(_g, "cuda_device_id", 0) or 0)
            except Exception as _degrade_error:
                _swallowed("roop/predictor.py:device_memory", _degrade_error,
                           "fallback continued")
                device_id = 0
        free, total = torch.cuda.mem_get_info(device_id)
    except Exception as _degrade_error:
        _swallowed("roop/predictor.py:device_memory", _degrade_error,
                   "fallback continued")
        return None
    return {"used": (total - free) / _MIB, "free": free / _MIB,
            "total": total / _MIB}


def bound_sessions() -> dict:
    """tag -> {requested, active, vram_used_mib} for every verified session."""
    with _lock:
        return {k: dict(v) for k, v in _BOUND.items()}


def _log_memory(tag: str) -> None:
    """Print device VRAM after *tag* finished loading, with the delta since the
    previous model.  Device-wide, so the delta is approximate when another
    thread is loading something at the same time."""
    mem = device_memory()
    if mem is None:
        return
    with _lock:
        prev = _last_used_mib[0]
        _last_used_mib[0] = mem["used"]
        if tag in _BOUND:
            _BOUND[tag]["vram_used_mib"] = round(mem["used"], 1)
    delta = "" if prev is None else " (%+.0f MiB since previous model)" % (mem["used"] - prev)
    print("[Provider] %s: VRAM %.0f / %.0f MiB in use, %.0f MiB free%s"
          % (tag, mem["used"], mem["total"], mem["free"], delta))


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
    except Exception as _degrade_error:
        _swallowed("roop/predictor.py:140", _degrade_error, "fallback continued")
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
        from roop.gpu_preflight import get_preflight_result
        preflight = get_preflight_result()
        available = preflight.get("available_providers", [])
        ort_version = preflight.get("onnxruntime_version", "")
    except Exception as exc:  # pragma: no cover - ORT is a hard dependency
        _swallowed("roop/predictor.py:199", exc, "fallback continued")
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
                             strict: Optional[bool] = None,
                             check_tensorrt: bool = True) -> List[str]:
    """Fail loudly when a requested TensorRT session silently came up elsewhere.

    Returns the session's ACTIVE provider list so a caller can record what ran
    rather than what it asked for.  Never raises for a session that did not ask
    for TensorRT in the first place.
    """
    try:
        active = list(session.get_providers())
    except Exception as _degrade_error:
        # A stub or pooled wrapper that does not speak the session protocol is
        # not something to fail a render over.
        _swallowed("roop/predictor.py:260", _degrade_error, "fallback continued")
        return []

    if not wants_gpu(requested):
        return active

    if strict is None:
        strict = strict_enabled()

    # CUDA or TensorRT was asked for and neither registered: the session is
    # CPU-only.  This is the failure with no TensorRT in it at all -- a missing
    # cuDNN/cuBLAS DLL, a CPU-only onnxruntime wheel shadowing onnxruntime-gpu,
    # or a rejected provider option (ORT drops CUDA *and* TensorRT together).
    # Checked before the TensorRT test below because that one only runs when
    # TensorRT was requested.
    if not has_gpu_active(active):
        _provider_failure(
            tag, requested, active, strict,
            headline="a GPU provider (%s) was REQUESTED but the session is "
                     "CPU-only" % ", ".join(
                         n for n in provider_names(requested) if _is_gpu_name(n)))
        return active

    _record_binding(tag, requested, active)

    if not wants_tensorrt(requested):
        return active

    if "tensorrt" in active[0].lower():
        return active

    _provider_failure(
        tag, requested, active, strict and check_tensorrt,
        headline="TensorRT was REQUESTED but is not the active execution "
                 "provider")
    return active


def _record_binding(tag: str, requested, active: List[str]) -> None:
    """Remember what a session is really bound to; print it once per tag."""
    with _lock:
        new = tag not in _asserted
        _asserted.add(tag)
        _BOUND[tag] = {"requested": provider_names(requested),
                       "active": list(active), "vram_used_mib": None}
    if new:
        print("[Provider] %s: active = %s" % (tag, ", ".join(active)))


def _provider_failure(tag: str, requested, active: List[str], strict: bool,
                      headline: str) -> None:
    """Raise (strict) or warn + record a degradation for a session that is not
    on the provider it asked for."""
    report = environment_report()
    message = (
        "%s: %s.\n"
        "  requested : %s\n"
        "  active    : %s\n"
        "onnxruntime does not raise when an execution provider fails to "
        "register -- it logs and continues on the next one in the list, so this "
        "session would have run on %s at a fraction of the expected speed with "
        "nothing in the render output to show for it.\n"
        "%s\n"
        "Set ROOP_STRICT_PROVIDER=0 to downgrade this to a warning."
        % (tag, headline,
           ", ".join(provider_names(requested)) or "(empty)",
           ", ".join(active) or "(empty)",
           active[0] if active else "an unknown provider",
           format_environment(report))
    )

    if strict:
        raise ProviderAssertionError(message)

    print("[Provider] WARNING -- %s" % message)
    try:
        from roop import backend_manager
        wanted = next((n for n in provider_names(requested) if _is_gpu_name(n)),
                      TENSORRT_EP)
        backend_manager._record_degradation(
            tag, wanted, active[0] if active else "unknown",
            ProviderAssertionError("%s EP did not register" % wanted))
    except Exception as _degrade_error:
        _swallowed("roop/predictor.py:305", _degrade_error, "fallback continued")
        pass


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
        # The provider can be dropped DURING the first run, not at construction
        # (ORT re-initialises on CPU and retries; measured 2026-08-12: the list
        # read [CUDA, CPU] before and [CPU] after a 1.9 s first call).  So the
        # construction-time check above is necessary but not sufficient.
        if wants_gpu(requested) and has_gpu_active(active):
            try:
                after = list(session.get_providers())
            except Exception as _degrade_error:
                _swallowed("roop/predictor.py:verify_and_warmup", _degrade_error,
                           "fallback continued")
                after = active
            if not has_gpu_active(after):
                _provider_failure(
                    tag, requested, after, strict_enabled(),
                    headline="the session was on a GPU provider (%s) at "
                             "construction but dropped to CPU-only during its "
                             "first inference" % ", ".join(active))
                active = after
        _log_memory(tag)
    return active


def forget(tag: str) -> None:
    """Forget ONE tag, for a session that is being released and rebuilt.

    ``warmup_session`` is once-per-tag, so a swapper released on a model
    switch and later reloaded under the same tag would otherwise skip its
    dummy pass and pay the TensorRT engine load on frame 0 of the render.
    """
    with _lock:
        _warmed.discard(tag)
        _asserted.discard(tag)
        _BOUND.pop(tag, None)


def reset() -> None:
    """Forget which tags were warmed/asserted (tests, and model reloads)."""
    with _lock:
        _warmed.clear()
        _asserted.clear()
        _BOUND.clear()
        _last_used_mib[0] = None
