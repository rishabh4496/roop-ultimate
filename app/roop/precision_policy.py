"""Model-specific precision and provider policy.

The application has one global TensorRT precision setting because that is a
useful user-facing default, but the models do not share one numerical safety
profile.  This module is the narrow decision point between that setting and a
model session.  It deliberately distinguishes:

* a known safe mode,
* a measured-but-not-yet-shipping candidate, and
* a known or unvalidated mode that must fall back to FP32.

The policy is conservative.  A faster engine is not accepted when it can
produce NaN, a flat/collapsed image, or a channel-skewed face.  BF16, INT8 and
FP8 are reported as unavailable until the installed ORT/TensorRT stack and a
model-specific calibration/quality test actually expose and validate them.

The cache identity includes the model content and the complete runtime/GPU
identity supplied by ``backend_manager.cache_namespace``.  A decision learned
on an RTX 4070 therefore cannot silently become an RTX 3060 decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from roop.backend_manager import cache_namespace


PRECISIONS = ("fp32", "fp16", "mixed", "bf16", "int8", "fp8")

# BF16 has a real provider option in this project. INT8/FP8 are represented in
# the policy matrix so future calibrated implementations can be recorded, but
# a TensorRT feature flag alone does not enable either mode here.
_IMPLEMENTED_PROVIDER_PRECISIONS = frozenset(("bf16",))


@dataclass(frozen=True)
class ModelPrecisionPolicy:
    """Static evidence-backed policy for one logical model family."""

    model: str
    backend: str
    fp32: str
    fp16: str
    mixed: str
    bf16: str
    int8: str
    fp8: str
    cuda_fallback: str
    cpu_fallback: str
    recommended: str
    trt_supported: str
    reason: str


@dataclass(frozen=True)
class PrecisionDecision:
    """Resolved request plus its provenance and cache identity."""

    model: str
    requested: str
    effective: str
    backend: str
    trt_enabled: bool
    fallback: bool
    cache_key: str
    policy: ModelPrecisionPolicy


_UNKNOWN = "not-validated"
_NO = "unsupported"
_SAFE = "safe"
_REQUIRED = "required"
_CANDIDATE = "candidate"
_UNSAFE = "unsafe"


def _policy(model, **values) -> ModelPrecisionPolicy:
    defaults = dict(
        backend="onnxruntime",
        fp32=_SAFE,
        fp16=_UNKNOWN,
        mixed=_UNKNOWN,
        bf16=_NO,
        int8=_NO,
        fp8=_NO,
        cuda_fallback="available",
        cpu_fallback="available",
        recommended="mixed",
        trt_supported="yes",
        reason="No model-specific failure has been measured; validate before changing the default.",
    )
    defaults.update(values)
    return ModelPrecisionPolicy(model=model, **defaults)


# These values are intentionally evidence labels, not guesses about what a
# GPU *could* execute.  ``mixed`` means TensorRT may use FP16 kernels while the
# graph retains the measured FP32-sensitive operations.
POLICIES = {
    "gpen_256": _policy("GPEN 256", fp16=_CANDIDATE, mixed=_SAFE,
                         reason="256px GPEN is stable on the existing mixed TensorRT path."),
    "gpen_512": _policy("GPEN 512", fp16=_CANDIDATE, mixed=_SAFE,
                         reason="Classic GPEN-512 is stable on the existing mixed TensorRT path."),
    "gpen_1024": _policy("GPEN 1024", fp16=_UNSAFE, mixed=_UNSAFE,
                          recommended="fp32", reason="Measured TensorRT FP16 activation overflow produces NaN/black faces."),
    "gpen_2048": _policy("GPEN 2048", fp16=_UNSAFE, mixed=_UNSAFE,
                          recommended="fp32", reason="Measured TensorRT FP16 activation overflow produces NaN/black faces."),
    "gpen_256_pro": _policy("GPEN 256 Pro", fp16=_CANDIDATE, mixed=_SAFE,
                             reason="Existing GPEN-256 Pro path has finite-output and collapse guards; network is small."),
    "gpen_realistic": _policy("GPEN Realistic", fp16=_CANDIDATE, mixed=_SAFE,
                               reason="Existing GPEN Realistic path is a GPEN-256/512 luminance model with output guards."),
    "codeformer": _policy("CodeFormer", fp16=_UNKNOWN, mixed=_CANDIDATE,
                           reason="FP32 graph is the reference; mixed TensorRT retains LayerNorm FP32 safeguards."),
    "codeformer_fp16": _policy("CodeFormer FP16", fp32=_SAFE, fp16=_SAFE,
                                mixed=_SAFE, recommended="mixed",
                                reason="Ships as a distinct FP16 graph and is the measured UltraMax/CodeFormer FP16 path."),
    "gfpgan": _policy("GFPGAN v1.4", fp16=_UNSAFE, mixed=_UNSAFE,
                       recommended="fp32", reason="Measured finite FP16 collapse produces a flat grey face; FP32 matches CUDA."),
    "restoreformer_pp": _policy("RestoreFormer++", fp16=_UNKNOWN, mixed=_CANDIDATE,
                                 reason="TensorRT path exists, but no model-specific FP16 quality gate is recorded."),
    "dmdnet": _policy("DMDNet", backend="pytorch", fp16=_UNSAFE, mixed=_NO,
                       recommended="fp32", trt_supported="no",
                       reason="PyTorch checkpoint path is FP32-only and is not an ONNX/TensorRT session."),
    "frame_upscaler": _policy("Frame upscaler / ESRGAN family", fp16=_UNSAFE,
                               mixed=_UNSAFE, recommended="fp32", trt_supported="no",
                               reason="Measured TensorRT mixed/FP16 ESRGAN-family output can become black; CUDA/CPU FP32 is shipped."),
    "rife": _policy("RIFE frame interpolation", fp16=_UNKNOWN, mixed=_NO,
                    recommended="fp32", trt_supported="no",
                    reason="The shipped path intentionally excludes TensorRT and uses CUDA/CPU FP32."),
    "liveportrait": _policy("LivePortrait", fp16=_UNKNOWN, mixed=_CANDIDATE,
                             bf16=_CANDIDATE,
                             reason="Patched warping graph supports TRT; stock 5-D GridSample needs CPU fallback."),
    "face_detection": _policy("Face detection", fp16=_CANDIDATE, mixed=_SAFE,
                               reason="Detection models are covered by the existing TensorRT context benchmark."),
    "recognition": _policy("Face recognition / landmarks", fp16=_CANDIDATE, mixed=_SAFE,
                            reason="Buffalo/AdaFace auxiliary sessions use the measured mixed provider chain."),
    "face_swap": _policy("Face swapping", fp16=_UNSAFE, mixed=_CANDIDATE,
                          reason="Raw FP16 has produced rainbow-smudge output; mixed is allowed only with output validation."),
    "masking": _policy("Masking models", fp16=_CANDIDATE, mixed=_SAFE,
                       reason="XSeg/BiSeNet/occluder models use the measured mixed path; SAM variants remove TRT."),
    "masking_no_trt": _policy("SAM masking models", fp16=_UNKNOWN, mixed=_NO,
                               recommended="fp32", trt_supported="no",
                               reason="FastSAM/MobileSAM/SAM2 paths intentionally use CUDA/CPU without TensorRT."),
    "frame_colorizer": _policy("Frame colorizer", fp16=_UNKNOWN, mixed=_CANDIDATE,
                                reason="No model-specific FP16 quality result is recorded."),
    "frame_masking": _policy("Frame foreground masking", fp16=_UNKNOWN, mixed=_CANDIDATE,
                              reason="No model-specific FP16 quality result is recorded."),
}


def canonical_model_key(model_key: str | None, model_path: str | None = None) -> str:
    """Map a processor/model alias to the stable policy key."""
    raw = str(model_key or "").lower().replace("\\", "/")
    path = str(model_path or "").lower().replace("\\", "/")
    value = f"{raw} {path}"
    if "gpen" in value:
        if "2048" in value:
            return "gpen_2048"
        if "1024" in value:
            return "gpen_1024"
        if "256 pro" in value or "256_pro" in value or "gpen256pro" in value:
            return "gpen_256_pro"
        if "realistic" in value or "gpenr" in value:
            return "gpen_realistic"
        if "256" in value:
            return "gpen_256"
        return "gpen_512"
    if "gfpgan" in value:
        return "gfpgan"
    if "ultramax" in value or "codeformer.fp16" in value or "codeformer_fp16" in value:
        return "codeformer_fp16"
    if "codeformer" in value:
        return "codeformer"
    if "restoreformer" in value:
        return "restoreformer_pp"
    if "dmdnet" in value:
        return "dmdnet"
    if "rife" in value:
        return "rife"
    if "liveportrait" in value or "appearance_feature" in value or "motion_extractor" in value or "warping_spade" in value or "stitching" in value:
        return "liveportrait"
    if any(token in value for token in ("upscale", "esrgan", "lsdir", "nomos8k", "span_", "ultra_sharp", "clear_reality")):
        return "frame_upscaler"
    if any(token in value for token in ("recogn", "adaface", "w600k", "1k3d68", "2d106")):
        return "recognition"
    if any(token in value for token in ("detector", "retinaface", "yoloface", "scrfd", "yunet", "det_10g")):
        return "face_detection"
    if any(token in value for token in ("swap", "inswapper", "reswapper", "hyperswap", "ghost_", "simswap", "hififace", "blendswap", "uniface")):
        return "face_swap"
    if any(token in value for token in ("fastsam", "mobilesam", "sam2", "clip2seg")):
        return "masking_no_trt"
    if any(token in value for token in ("isnet", "removebg")):
        return "frame_masking"
    if "color" in value or "deoldify" in value:
        return "frame_colorizer"
    if "frame_mask" in value or "foreground" in value:
        return "frame_masking"
    if any(token in value for token in ("mask", "xseg", "bisenet", "occluder", "resnet18")):
        return "masking"
    return "unknown"


def get_policy(model_key: str | None, model_path: str | None = None) -> ModelPrecisionPolicy:
    key = canonical_model_key(model_key, model_path)
    return POLICIES.get(key, _policy(str(model_key or "unknown"), recommended="fp32",
                                      reason="Unknown model: keep FP32 until explicitly validated."))


def _has_trt(providers: Iterable) -> bool:
    return any("tensorrt" in str(p[0] if isinstance(p, (tuple, list)) else p).lower()
               for p in (providers or ()))


def _force_fp32(providers: Iterable, tag: str):
    """Copy providers and isolate a forced-FP32 TRT engine cache."""
    patched = []
    for provider in list(providers or ()):
        if isinstance(provider, (tuple, list)) and len(provider) == 2 and "tensorrt" in str(provider[0]).lower():
            name, options = provider[0], dict(provider[1])
            options["trt_fp16_enable"] = False
            cache = options.get("trt_engine_cache_path")
            if cache:
                safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
                fp32_cache = f"{cache}_{safe_tag}_fp32"
                os.makedirs(fp32_cache, exist_ok=True)
                options["trt_engine_cache_path"] = fp32_cache
                options["trt_timing_cache_path"] = fp32_cache
            patched.append((name, options))
        else:
            patched.append(provider)
    return patched


def _enable_bf16(providers: Iterable, tag: str):
    """Enable BF16 for an explicitly validated candidate without mutation."""
    patched = []
    for provider in list(providers or ()):
        if isinstance(provider, (tuple, list)) and len(provider) == 2 and "tensorrt" in str(provider[0]).lower():
            name, options = provider[0], dict(provider[1])
            options["trt_bf16_enable"] = True
            options["trt_fp16_enable"] = False
            cache = options.get("trt_engine_cache_path")
            if cache:
                safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
                bf16_cache = f"{cache}_{safe_tag}_bf16"
                os.makedirs(bf16_cache, exist_ok=True)
                options["trt_engine_cache_path"] = bf16_cache
                options["trt_timing_cache_path"] = bf16_cache
            patched.append((name, options))
        else:
            patched.append(provider)
    return patched


def _without_trt(providers: Iterable):
    kept = [p for p in list(providers or ())
            if "tensorrt" not in str(p[0] if isinstance(p, (tuple, list)) else p).lower()]
    return kept or ["CPUExecutionProvider"]


def _model_digest(model_path: str | None) -> str:
    if not model_path or not os.path.isfile(model_path):
        return "missing"
    digest = hashlib.sha256()
    with open(model_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def decision_cache_key(model_key: str, model_path: str | None, requested: str,
                       effective: str, device_id: int = 0) -> str:
    identity = {
        "model": canonical_model_key(model_key, model_path),
        "model_digest": _model_digest(model_path),
        "requested": requested,
        "effective": effective,
        "runtime": cache_namespace(effective, device_id),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _runtime_hardware(hardware=None):
    """Return the startup profile when the application has already probed it.

    Direct policy callers may omit this optional argument and use synthetic
    provider lists. Production startup publishes the single hardware profile
    used by the workload, avoiding a second ad-hoc GPU probe.
    """
    if hardware is not None:
        return hardware
    try:
        import roop.globals
        return getattr(roop.globals, "runtime_hardware_profile", None)
    except Exception:
        return None


def _hardware_allows(precision: str, hardware) -> bool:
    if hardware is None:
        # Compatibility for direct unit/integration callers that do not own a
        # profiler. RuntimeOptimizer and production startup perform the gate.
        return True
    if not bool(getattr(hardware, "cuda_available", False)):
        return False
    if not bool(getattr(hardware, "tensorrt_available", False)):
        return False
    if precision in ("fp16", "mixed"):
        return bool(getattr(hardware, "fp16_supported", False))
    if precision == "bf16":
        return bool(getattr(hardware, "bf16_supported", False))
    if precision == "int8":
        return bool(getattr(hardware, "int8_supported", False))
    if precision == "fp8":
        return bool(getattr(hardware, "fp8_supported", False))
    return True


def resolve(model_key: str, requested: str = "mixed", providers=None,
            model_path: str | None = None, device_id: int = 0,
            hardware=None) -> PrecisionDecision:
    """Resolve one model request without mutating the caller's providers."""
    requested = str(requested or "mixed").lower()
    if requested not in PRECISIONS:
        requested = "mixed"
    policy = get_policy(model_key, model_path)
    trt = _has_trt(providers)
    effective = requested
    fallback = False
    if not trt:
        effective = "fp32"
        fallback = requested != "fp32"
        backend = "cuda" if any("cuda" in str(p).lower() for p in (providers or ())) else "cpu"
    else:
        evidence = getattr(policy, requested, _UNKNOWN)
        # Unknown and unsafe FP16/mixed candidates are never silently shipped
        # as a faster precision. Explicit FP32 remains authoritative.
        if requested in ("fp16", "mixed", "bf16", "int8", "fp8") and evidence not in (_SAFE, _CANDIDATE):
            effective = "fp32"
            fallback = True
        # A model label is not a hardware probe. Require the detected stack to
        # expose the requested mode too. INT8/FP8 additionally require a real
        # provider implementation; feature flags alone are insufficient.
        detected = _runtime_hardware(hardware)
        if (effective in ("fp16", "mixed", "bf16", "int8", "fp8") and
                not _hardware_allows(effective, detected)):
            effective = "fp32"
            fallback = True
        if (effective in ("int8", "fp8") and
                effective not in _IMPLEMENTED_PROVIDER_PRECISIONS):
            effective = "fp32"
            fallback = True
        if policy.trt_supported == "no":
            # The caller supplied TRT, but this model family deliberately
            # removes it (ESRGAN/RIFE/SAM). Report the backend that will really
            # execute after providers_for() applies that policy.
            backend = "cuda" if any("cuda" in str(p).lower() for p in (providers or ())) else "cpu"
        else:
            backend = "tensorrt"
    key = decision_cache_key(model_key, model_path, requested, effective, device_id)
    return PrecisionDecision(canonical_model_key(model_key, model_path), requested,
                             effective, backend, trt, fallback, key, policy)


def providers_for(model_key: str, providers, model_path: str | None = None,
                  requested: str | None = None, device_id: int = 0,
                  hardware=None):
    """Return provider options for one model under the active precision policy."""
    if requested is None:
        try:
            import roop.globals
            requested = getattr(getattr(roop.globals, "CFG", None), "trt_precision", "mixed")
        except Exception:
            requested = "mixed"
    decision = resolve(model_key, requested, providers, model_path, device_id,
                       hardware=hardware)
    if model_path:
        # Persist the resolved decision at session-construction time.  The key
        # contains the model digest and backend_manager's GPU/runtime
        # fingerprint, so this record cannot be reused across GPUs or model
        # revisions.
        write_decision_cache(decision)
    if not decision.trt_enabled:
        return list(providers or ()), decision
    if decision.policy.trt_supported == "no":
        return _without_trt(providers), decision
    if decision.effective == "fp32":
        return _force_fp32(providers, decision.model), decision
    if decision.effective == "bf16":
        return _enable_bf16(providers, decision.model), decision
    return list(providers or ()), decision


def write_decision_cache(decision: PrecisionDecision, directory: str | None = None) -> str:
    """Persist a JSON decision record for diagnostics and later audit."""
    root = Path(directory or os.path.join(os.path.dirname(__file__), "..", "models", "runtime_profiles"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"precision_{decision.cache_key}.json"
    payload = asdict(decision)
    payload["policy"] = asdict(decision.policy)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(root))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(path)


def matrix() -> list[dict]:
    """Return the complete auditable model-family precision matrix."""
    return [asdict(value) for value in POLICIES.values()]
