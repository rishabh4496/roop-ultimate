"""Runtime hardware/provider resolution.

This module deliberately has no model or UI dependencies.  It answers one
question for the rest of the pipeline: which ONNX Runtime providers are
actually usable on this machine, and in what order should they be attempted?
Provider *availability* is not enough (CUDA can be listed while its DLLs or
device are unavailable), so capability checks are kept here and are cached for
the lifetime of the process.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import os
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Iterable, List, Dict, Mapping, Tuple, Optional, Any


_lock = threading.Lock()
_probe_cache: Dict[Tuple[str, int], bool] = {}


def _name(value) -> str:
    return value[0] if isinstance(value, (tuple, list)) else str(value)


def _available() -> List[str]:
    """Return the provider list from the one authoritative preflight.

    There is intentionally no second ONNX Runtime probe here.  A preflight
    failure is represented by an empty provider list and remains visible in
    ``get_preflight_result()`` for diagnostics.
    """
    try:
        from roop.gpu_preflight import get_preflight_result
        return list(get_preflight_result().get("available_providers", []))
    except Exception as _degrade_error:
        _swallowed("roop/backend_manager.py:33", _degrade_error, "fallback continued")
        return []


def provider_available(name: str, available: Iterable[str] | None = None) -> bool:
    """Return whether *name* is listed by the installed ORT build."""
    wanted = name.lower().replace("executionprovider", "")
    return any(wanted == p.lower().replace("executionprovider", "")
               for p in (available if available is not None else _available()))


def provider_usable(name: str, device_id: int = 0,
                    available: Iterable[str] | None = None) -> bool:
    """Cheap, side-effect-light capability probe, cached per device.

    A provider that is not listed cannot work. CUDA/TRT additionally require a
    visible CUDA device.  The probe intentionally does not instantiate a
    production model: doing that here would build TensorRT engines during UI
    startup and would make every provider query expensive.
    """
    canonical = _name(name)
    key = (canonical.lower(), int(device_id))
    with _lock:
        if key in _probe_cache:
            return _probe_cache[key]
    ok = provider_available(canonical, available)
    if ok and canonical.lower().startswith(("cuda", "tensorrt", "rocm")):
        try:
            import torch
            ok = bool(torch.cuda.is_available())
            if ok:
                count = int(torch.cuda.device_count())
                ok = 0 <= int(device_id) < max(1, count)
        except Exception as _degrade_error:
            _swallowed("roop/backend_manager.py:66", _degrade_error, "fallback continued")
            ok = False
        if ok and canonical.lower().startswith("tensorrt"):
            try:
                from roop.gpu_preflight import get_preflight_result
                ok = bool(get_preflight_result().get("tensorrt_session_usable", False))
            except Exception:
                pass
        elif ok and canonical.lower().startswith("cuda"):
            try:
                from roop.gpu_preflight import get_preflight_result
                active = get_preflight_result().get("active_provider")
                ok = active in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
            except Exception:
                ok = False
    if ok and canonical.lower().startswith("dml"):
        # ORT's DML provider is self-contained; the listing is the reliable
        # check and importing torch must not make DirectML appear unavailable.
        ok = True
    with _lock:
        _probe_cache[key] = ok
    return ok


def is_sub_7gb_gpu(device_id: int = 0) -> bool:
    """Return whether the physical CUDA device has strictly less than 7.0 GB VRAM.

    This is an intrinsic hardware check derived from physical VRAM, never from the GPU
    model name alone. Returns False for CPU-only systems or devices with >= 7.0 GB VRAM.
    """
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() <= int(device_id):
            return False
        total_gb = torch.cuda.get_device_properties(int(device_id)).total_memory / (1024 ** 3)
        return 0.0 < total_gb < 7.0
    except Exception as _degrade_error:
        _swallowed("roop/backend_manager.py:is_sub_7gb_gpu", _degrade_error, "fallback continued")
        return False


def allow_small_gpu_trt() -> bool:
    """Return whether small RTX cards are eligible for TensorRT.

    All CUDA-capable RTX devices are TensorRT candidates now.  This helper is
    retained as a compatibility field for older clients, but it no longer reads
    an opt-in environment variable.  Low-VRAM limits remain enforced by the
    runtime optimizer and session pools rather than by provider admission.
    """
    return True


def is_trt_allowed_for_device(device_id: int = 0) -> bool:
    """Return whether this device may attempt TensorRT after runtime preflight.

    VRAM size is a tuning input, not a qualification rule.  A tiny RTX card may
    still use TensorRT with one context, no pools, bounded workspace, and the
    global GPU guard.  The actual provider/session preflight remains the
    authoritative check for whether TensorRT can run on the installed driver.
    """
    return True


def _small_gpu(device_id: int = 0) -> bool:
    """Return whether this device uses the conservative low-VRAM safety tier."""
    return is_sub_7gb_gpu(device_id)


@dataclass(frozen=True)
class CanonicalProviderState:
    requested: str
    admitted: str
    available: Tuple[str, ...]
    active: str
    active_chain: Tuple[str, ...]
    degraded: bool
    degradation_reason: Optional[str]
    degradation_stage: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "admitted": self.admitted,
            "available": list(self.available),
            "active": self.active,
            "active_chain": list(self.active_chain),
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
            "degradation_stage": self.degradation_stage,
        }


_CANONICAL_STATE_CACHE: Dict[Tuple[str, int], CanonicalProviderState] = {}
_PROVIDER_DEGRADATIONS: List[Dict[str, Any]] = []


def provider_degradations() -> List[Dict[str, Any]]:
    """Return structured log of all provider degradations."""
    with _lock:
        return list(_PROVIDER_DEGRADATIONS)


def record_provider_degradation(requested: str, active: str, reason: str, stage: str) -> None:
    """Record a structured provider downgrade record."""
    entry = {
        "requested": requested,
        "active": active,
        "reason": reason,
        "stage": stage,
    }
    with _lock:
        _PROVIDER_DEGRADATIONS.append(entry)
    # A closed Pinokio/test terminal must not turn a recoverable provider
    # downgrade into a startup failure.  This is especially common on Windows
    # when a test runner replaces stdout or a detached daemon loses its pipe.
    try:
        print(f"[Provider] Downgrade: requested '{requested}' -> active '{active}' ({stage}): {reason}", flush=True)
    except (OSError, ValueError):
        pass


def canonical_provider_decision(requested: Optional[str] = None, device_id: int = 0) -> CanonicalProviderState:
    """The single authoritative provider decision following the 4-step resolution pipeline:

    1. Explicit user selection (requested)
    2. Capability / admission validation (admitted)
    3. Actual usable provider chain (available & usable)
    4. Runtime session verification (active)
    """
    raw_req = requested or os.environ.get("ROOP_EXECUTION_PROVIDER", "auto")
    req = _name(raw_req).strip().lower().replace("executionprovider", "") or "auto"
    req = {"directml": "dml"}.get(req, req)

    cache_key = (req, int(device_id))
    with _lock:
        if cache_key in _CANONICAL_STATE_CACHE:
            return _CANONICAL_STATE_CACHE[cache_key]

    from roop.gpu_preflight import get_preflight_result
    preflight = get_preflight_result()
    available_providers = tuple(preflight.get("available_providers", []))
    preflight_active = preflight.get("active_provider")

    # Step 2: Capability / Admission Validation
    admitted = req
    degradation_reason = None
    degradation_stage = None

    if req in ("auto", "tensorrt"):
        # Hardware tiering controls workspace, precision, pools, and
        # concurrency. It must not reject a qualified TensorRT provider merely
        # because the card has less than 7 GB of VRAM.
        admitted = "tensorrt"
    elif req == "cuda":
        if not preflight.get("cuda_available", False):
            admitted = "cpu"
            degradation_reason = "No CUDA device available for requested CUDA provider"
            degradation_stage = "admission_rejected"
        else:
            admitted = "cuda"
    elif req in ("rocm", "dml", "directml", "cpu"):
        admitted = req
    else:
        admitted = req

    # Step 3: Actual Usable Provider Chain
    candidates = _HIERARCHY.get(admitted, (f"{admitted.capitalize()}ExecutionProvider", "CPUExecutionProvider"))
    usable_chain: List[str] = []

    for candidate in candidates:
        cand_lower = candidate.lower()
        if cand_lower.startswith("tensorrt"):
            if preflight.get("tensorrt_session_usable", False):
                usable_chain.append(candidate)
            elif not degradation_reason and req in ("auto", "tensorrt"):
                degradation_reason = preflight.get("failure_reason") or "TensorRT runtime is not usable"
                degradation_stage = preflight.get("failure_stage") or "runtime_unavailable"
        elif cand_lower.startswith("cuda"):
            if (
                preflight.get("cuda_available", False)
                and "CUDAExecutionProvider" in available_providers
                and preflight_active in (
                    "TensorrtExecutionProvider", "CUDAExecutionProvider"
                )
            ):
                usable_chain.append(candidate)
            elif not degradation_reason and req in ("auto", "tensorrt", "cuda"):
                degradation_reason = preflight.get("failure_reason") or "CUDA is not available"
                degradation_stage = preflight.get("failure_stage") or "cuda_unavailable"
        elif cand_lower.startswith("rocm"):
            if "ROCMExecutionProvider" in available_providers:
                usable_chain.append(candidate)
        elif cand_lower.startswith("dml"):
            if "DmlExecutionProvider" in available_providers:
                usable_chain.append(candidate)
        elif cand_lower.startswith("cpu"):
            if "CPUExecutionProvider" in available_providers:
                usable_chain.append(candidate)

    if not usable_chain:
        usable_chain = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available_providers else []

    # Step 4: Runtime Session Verification
    active = usable_chain[0] if usable_chain else "none"
    active_short = active.replace("ExecutionProvider", "").lower()

    # Auto is still a request for the fastest admitted chain.  If that chain
    # steps down, record it too.  Only an explicit CPU request is inherently
    # non-degrading when it binds to CPU.
    preferred_short = _name(candidates[0]).replace("ExecutionProvider", "").lower()
    degraded = bool(degradation_reason) or active_short != preferred_short
    if degraded:
        degraded = True
        if not degradation_reason:
            degradation_reason = f"Provider '{req}' was requested but session bound to '{active_short}'"
            degradation_stage = "session_fallback"
        record_provider_degradation(req, active_short, degradation_reason, degradation_stage or "unknown")

    state = CanonicalProviderState(
        requested=req,
        admitted=admitted,
        available=available_providers,
        active=active,
        active_chain=tuple(usable_chain),
        degraded=degraded,
        degradation_reason=degradation_reason,
        degradation_stage=degradation_stage,
    )

    with _lock:
        _CANONICAL_STATE_CACHE[cache_key] = state

    return state

_HIERARCHY = {
    "auto": ("TensorrtExecutionProvider", "CUDAExecutionProvider",
             "ROCMExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"),
    "tensorrt": ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"),
    "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "rocm": ("ROCMExecutionProvider", "CPUExecutionProvider"),
    "dml": ("DmlExecutionProvider", "CPUExecutionProvider"),
    "directml": ("DmlExecutionProvider", "CPUExecutionProvider"),
    "coreml": ("CoreMLExecutionProvider", "CPUExecutionProvider"),
    "cpu": ("CPUExecutionProvider",),
}


def resolve_provider_names(requested: Iterable[str] | None,
                           device_id: int = 0) -> List[str]:
    """Resolve a requested backend through the canonical decision only."""
    requested = list(requested or ("cpu",))
    decision = canonical_provider_decision(
        requested[0] if requested else "cpu", device_id=device_id)
    return list(decision.active_chain)


def provider_admission(requested: str | None = None, device_id: int = 0) -> dict:
    """Explain the runtime admission decision for the requested backend.

    This is deliberately separate from provider resolution so diagnostics can
    distinguish "TensorRT is not installed" from "TensorRT was intentionally
    rejected for this hardware tier".  The latter is the expected Phase 4
    behavior on a sub-7GB device unless the user explicitly opts into the
    experimental override.
    """
    configured = requested or os.environ.get("ROOP_EXECUTION_PROVIDER", "auto")
    decision = canonical_provider_decision(configured, device_id)
    normalized = _name(configured).lower().replace("executionprovider", "")
    normalized = {"directml": "dml"}.get(normalized, normalized)
    admitted = decision.degradation_stage != "admission_rejected"
    small = is_sub_7gb_gpu(device_id)
    allowed = is_trt_allowed_for_device(device_id)
    return {
        "requested": configured,
        "admitted": admitted,
        "admitted_provider": decision.admitted,
        "is_sub_7gb_gpu": small,
        "tensorrt_allowed": allowed,
        "reason": decision.degradation_reason or (
            "hardware tier permits requested backend; availability is checked separately"
        ),
        # Kept for clients that rendered the old opt-in badge.  It now means
        # that the small-card safety profile is active, not that admission was
        # obtained through a hidden environment override.
        "override": bool(normalized in ("auto", "tensorrt") and small),
    }


def diagnostic_report(device_id: int = 0, requested: str | None = None) -> dict:
    """Return JSON-safe diagnostics for logs and the diagnostics panel."""
    decision = canonical_provider_decision(requested, device_id)
    available = list(decision.available)
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(device_id) if cuda else ""
        vram_gb = (torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
                   if cuda else 0.0)
    except Exception as _degrade_error:
        _swallowed("roop/backend_manager.py:175", _degrade_error, "fallback continued")
        cuda, gpu, vram_gb = False, "", 0.0
    configured = requested or os.environ.get("ROOP_EXECUTION_PROVIDER", "auto")
    return {
        "configured": configured,
        "available_providers": available,
        "cuda_visible": cuda,
        "gpu": gpu,
        "vram_gb": round(vram_gb, 2),
        "admission": {
            "requested": decision.requested,
            "admitted": decision.admitted,
            "reason": decision.degradation_reason,
            "stage": decision.degradation_stage,
        },
        "resolved": list(decision.active_chain),
        "provider_requested": decision.requested,
        "provider_admitted": decision.admitted,
        "provider_active": decision.active,
        "degradation_reason": decision.degradation_reason,
        "degradation_stage": decision.degradation_stage,
    }


def clear_probe_cache() -> None:
    with _lock:
        _probe_cache.clear()
        _CANONICAL_STATE_CACHE.clear()
        _PROVIDER_DEGRADATIONS.clear()
    try:
        from roop.gpu_preflight import clear_preflight_cache
        clear_preflight_cache()
    except Exception:
        pass
    try:
        import settings
        settings._DEFAULT_PROVIDER_CACHE = None
    except Exception:
        pass


_DRIVER_SMI_CACHE: Dict[int, str] = {}


def _driver_from_smi(device_id: int = 0) -> str:
    """Driver version from nvidia-smi, cached for the process.

    The fallback for `cache_namespace` when torch exposes no driver probe.
    Kept cheap: one short subprocess per device, memoised, and any failure
    degrades to "" so the caller records "unknown" rather than raising during
    startup.
    """
    if device_id in _DRIVER_SMI_CACHE:
        return _DRIVER_SMI_CACHE[device_id]
    value = ""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
             "--id=%d" % int(device_id)],
            text=True, timeout=10, stderr=subprocess.DEVNULL)
        value = out.strip().splitlines()[0].strip() if out.strip() else ""
    except Exception as _degrade_error:
        _swallowed("roop/backend_manager.py:215", _degrade_error, "fallback continued")
        value = ""
    _DRIVER_SMI_CACHE[device_id] = value
    return value


def cache_namespace(precision: str, device_id: int = 0) -> str:
    """Build a stable TensorRT cache namespace for this runtime and GPU.

    TensorRT engine filenames include a graph hash, but the parent directory
    must still separate precision and runtime ABI.  This namespace prevents a
    CUDA/TRT upgrade or device swap from reusing a stale engine.
    """
    precision = str(precision or "mixed").lower()
    try:
        import onnxruntime as ort
        ort_ver = str(getattr(ort, "__version__", "unknown"))
    except Exception as _degrade_error:
        _swallowed("roop/backend_manager.py:232", _degrade_error, "fallback continued")
        ort_ver = "unknown"
    cuda_ver = "unknown"
    trt_ver = "unknown"
    driver_ver = "unknown"
    gpu = "cpu"
    sm = "na"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(device_id)
            sm = "sm%02d%02d" % tuple(torch.cuda.get_device_capability(device_id))
            cuda_ver = str(getattr(torch.version, "cuda", "unknown") or "unknown")
            # DRIVER IDENTITY IS PART OF THIS CACHE KEY AND MUST ACTUALLY
            # RESOLVE. This used to rely solely on the private
            # `torch._C._cuda_getDriverVersion` probe, with a comment claiming
            # it "is available on the supported CUDA builds". It is not: on
            # torch 2.7.0+cu128 the attribute is absent, so `driver_ver` stayed
            # "unknown" and every engine directory was named `drvunknown`.
            #
            # That silently removed driver isolation from the TensorRT engine
            # cache. TensorRT engines are driver-sensitive and are not
            # guaranteed portable across driver upgrades, so a stale engine
            # could be reused after a driver change -- exactly the cache
            # invalidation error Gate C exists to prevent. Found on the
            # physical RTX 3060 Laptop, whose profile reports driver 616.56
            # while its engine cache directory said `drvunknown`.
            get_driver = getattr(getattr(torch, "_C", None),
                                 "_cuda_getDriverVersion", None)
            if get_driver is not None:
                try:
                    raw_driver = int(get_driver())
                    driver_ver = (f"{raw_driver // 1000}."
                                  f"{(raw_driver % 1000) // 10}")
                except Exception as _degrade_error:
                    _swallowed("roop/backend_manager.py:266", _degrade_error, "fallback continued")
                    driver_ver = "unknown"
            if driver_ver == "unknown":
                driver_ver = _driver_from_smi(device_id) or "unknown"
    except Exception as _degrade_error:
        _swallowed("roop/backend_manager.py:270", _degrade_error, "fallback continued")
        pass
    try:
        import tensorrt as trt
        trt_ver = str(getattr(trt, "__version__", "unknown"))
    except (ImportError, ModuleNotFoundError):
        trt_ver = "unknown"
    except Exception as _degrade_error:
        _swallowed("roop/backend_manager.py:275", _degrade_error, "fallback continued")
        pass
    raw = (f"{precision}_{gpu}_{sm}_cuda{cuda_ver}_drv{driver_ver}"
           f"_trt{trt_ver}_ort{ort_ver}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def trt_tuning_namespace(builder_optimization_level: int = 3,
                         auxiliary_streams: int = -1,
                         cuda_graph: bool = False,
                         builder_config: Mapping | None = None) -> str:
    """Return the cache suffix for TensorRT build/runtime tuning knobs.

    These options can change the generated engine or its execution schedule,
    while TensorRT's graph filename does not necessarily encode all of them.
    Keeping them in the parent directory prevents an A/B benchmark or a
    changed default from accidentally reusing an engine built with another
    tuning profile.
    """
    namespace = (f"_b{int(builder_optimization_level)}"
                 f"_a{int(auxiliary_streams)}"
                 f"_g{int(bool(cuda_graph))}")
    if builder_config is not None:
        # Keep the readable legacy knobs above, then add every effective
        # builder option in a canonical digest. ORT/TensorRT may add options
        # in future releases; a mapping keeps this identity extensible.
        canonical = json.dumps(dict(builder_config), sort_keys=True,
                               separators=(",", ":"), default=str).encode()
        namespace += "_c" + hashlib.sha256(canonical).hexdigest()[:16]
    return namespace


# ── Session-build fallback ────────────────────────────────────────────────
#
# `resolve_provider_names` answers "could this provider work on this machine".
# It cannot answer "did this particular model's engine build succeed", because
# TensorRT does not allocate or compile anything until a session is actually
# constructed -- and for the engine itself, not until the first inference.
# So a provider chain that passes every admission check here can still throw
# inside `InferenceSession` for one model: an unsupported op forcing a bad
# partition, a workspace that does not fit beside the pooled sessions already
# resident, or a corrupt entry in the engine cache.
#
# The failure mode this exists to prevent is the whole app dying at startup
# because one model could not build a TensorRT engine, when CUDA would have
# run it perfectly.  The failure mode it must NOT introduce is a silent
# downgrade: this project has repeatedly been caught by a stage that reported
# success while running somewhere slower or not at all, so every step down is
# printed once and recorded for diagnostics.

_DEGRADATIONS: List[dict] = []


def _strip_provider(providers: Iterable, token: str) -> List:
    return [p for p in list(providers or ())
            if token not in str(_name(p)).lower()]


def session_degradations() -> List[dict]:
    """Every provider downgrade taken this process, for the diagnostics panel."""
    with _lock:
        return list(_DEGRADATIONS)


def _record_degradation(tag: str, frm: str, to: str, error: BaseException,
                        requested_provider: str | None = None,
                        active_provider: str | None = None) -> None:
    entry = {"model": str(tag), "from": frm, "to": to,
             "error": f"{type(error).__name__}: {error}"[:400]}
    with _lock:
        _DEGRADATIONS.append(entry)
    record_provider_degradation(
        requested_provider or frm,
        active_provider or to,
        entry["error"],
        "session_construction",
    )
    print(f"[Backend] {tag}: {frm} session build FAILED, falling back to {to}. "
          f"{entry['error']}")


def build_session_with_fallback(build, providers, tag: str = "model"):
    """Build a session, stepping down the provider chain only on real failure.

    ``build`` is called as ``build(providers)`` and may be any session factory
    (``onnxruntime.InferenceSession``, insightface's ``get_model``, a pooled
    slot builder).  The chain is walked at most twice: TensorRT is removed
    first, then CUDA/ROCm, leaving CPU.

    The ORIGINAL exception is re-raised when even CPU fails, because at that
    point the model itself is broken and the TensorRT error is the informative
    one.  Returns ``(session, providers_used)`` so the caller can record what
    actually ran rather than what it asked for.
    """
    attempts: List[Tuple[str, List]] = [("requested", list(providers or ()))]
    if any("tensorrt" in str(_name(p)).lower() for p in (providers or ())):
        attempts.append(("cuda/cpu", _strip_provider(providers, "tensorrt")))
    without_gpu = [p for p in (providers or ())
                   if not any(token in str(_name(p)).lower()
                              for token in ("tensorrt", "cuda", "rocm"))]
    if without_gpu != list(providers or ()):
        attempts.append(("cpu", without_gpu or ["CPUExecutionProvider"]))

    first_error: BaseException | None = None
    for index, (label, chain) in enumerate(attempts):
        if not chain:
            continue
        try:
            return build(chain), chain
        except Exception as error:  # noqa: BLE001 - the point is to degrade
            _swallowed("roop/backend_manager.py:376", error, "fallback continued")
            if first_error is None:
                first_error = error
            if index + 1 >= len(attempts):
                break
            next_label, next_chain = attempts[index + 1]
            _record_degradation(
                tag,
                label,
                next_label,
                error,
                requested_provider=_name(chain[0]),
                active_provider=_name(next_chain[0]),
            )
    if first_error is not None:
        raise first_error
    raise RuntimeError(f"{tag}: no usable execution provider chain")
