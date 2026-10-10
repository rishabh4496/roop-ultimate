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
from roop.degrade import swallowed as _swallowed

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
    # "Restore Ultra" is the RestoreFormer++ WEIGHTS with a forced-alignment
    # crop and a CPU-side anti-halo finish -- the graph, and therefore every
    # precision property of it, is identical. Its UI label does not contain
    # "restoreformer", so it is matched explicitly rather than falling through
    # to "unknown" and being pinned to FP32 by the unknown-model rule.
    if "restoreformer" in value or "restore ultra" in value or "restore_ultra" in value:
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


_DIGESTS: dict = {}


def _model_digest(model_path: str | None) -> str:
    if not model_path or not os.path.isfile(model_path):
        return "missing"
    # Memoised on (path, size, mtime): every providers_for() call for a model path re-read the whole file, and a
    # per-model precision lookup now makes that call once per buffalo file per pooled analyser (1k3d68 is 137 MB).
    st = os.stat(model_path)
    stamp = (os.path.abspath(model_path), st.st_size, st.st_mtime_ns)
    hit = _DIGESTS.get(stamp)
    if hit is not None:
        return hit
    digest = hashlib.sha256()
    with open(model_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    _DIGESTS[stamp] = digest.hexdigest()[:16]
    return _DIGESTS[stamp]


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
    except Exception as _degrade_error:
        _swallowed("roop/precision_policy.py:304", _degrade_error, "fallback continued")
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


def _global_precision() -> str:
    try:
        import roop.globals
        return getattr(getattr(roop.globals, "CFG", None), "trt_precision", "mixed")
    except Exception as _degrade_error:
        _swallowed("roop/precision_policy.py:382", _degrade_error, "fallback continued")
        return "mixed"


# ── per-model precision ────────────────────────────────────────────────────────────────────────────
#
# `trt_precision` is ONE user-facing setting, but the small models do not share a numerical profile. Measured 2026-10-10
# on 500 real faces from d4/d1/d6/Love/s7 (docs/perf/trt_precision_fidelity.md), shipped TensorRT 'mixed' against a CUDA
# FP32 reference with TF32 off, and against the same engine built with fp16 off:
#
#   model      mixed vs FP32                                                    TRT FP32 vs FP32     FP32 costs
#   xseg       IoU mean 0.9936 / min 0.81, 162 of 500 faces < 0.995, 6 < 0.95    IoU >= 0.9997       +0.6 ms  (2.23 -> 2.82)
#   2d106det   mean 0.12 px / max 1.25 px (frame)                                0.03 px (= floor)    +0.02 ms (0.50 -> 0.52)
#   1k3d68     mean 1.19 px, p95 4.5, max 8.8; refined kps mean 0.94 px, p95    0.37 px (= floor)    +1.05 ms (0.72 -> 1.77)
#              3.6 px = 10.5% of the inter-ocular distance at p95
#   w600k_r50  embedding cosine min 0.999916 (gate 0.999: PASS)                  1.000000             +1.08 ms (0.95 -> 2.04)
#
# Those fail the gates the earlier briefs set (xseg IoU >= 0.995 on >= 99% of faces and none < 0.95; landmark p95 <= 0.3 px)
# by 2-12x, while the FP32 engine sits at the engine-to-engine floor. w600k_r50 passes, so it keeps 'mixed'. An entry maps a
# model FILE STEM (lower-case) to a precision from PRECISIONS; with no entry the model follows `trt_precision`.
#
#   ROOP_TRT_MODEL_PRECISION="xseg:mixed;1k3d68:global"      env wins over the table; 'global' drops the table entry
#
# Swappers are refused (they have their own canary and ROOP_SWAP_FP32). Take an entry out of the table to hand a model back
# to the global setting; nothing else changes - the engine caches are per precision, so both engines stay on disk.
PRECISION_OVERRIDES: dict = {"xseg": "fp32", "2d106det": "fp32", "1k3d68": "fp32"}
PRECISION_OVERRIDE_ENV = "ROOP_TRT_MODEL_PRECISION"
precision_log: list = []        # every per-model precision applied or refused in this process, for harnesses and tests
_precision_announced: set = set()


def parse_precision_overrides(text: str) -> dict:
    """'xseg:fp32;1k3d68:global' -> {stem: precision | 'global'}. Raises ValueError."""
    out = {}
    for part in [p.strip() for p in str(text or "").split(";") if p.strip()]:
        stem, sep, value = part.partition(":")
        stem, value = stem.strip().lower(), value.strip().lower()
        if not stem or not sep:
            raise ValueError("expected 'stem:precision', got %r" % part)
        if value != "global" and value not in PRECISIONS:
            raise ValueError("precision must be one of %s or 'global', got %r" % (", ".join(PRECISIONS), value))
        out[stem] = value
    return out


def precision_override_for(model_key, model_path):
    """The per-model precision for this file, or None to follow the global setting. Swappers are always None."""
    stem = _stem(model_path)
    if not stem:
        return None
    table = dict(PRECISION_OVERRIDES)
    raw = os.environ.get(PRECISION_OVERRIDE_ENV, "").strip()
    if raw:
        try:
            table.update(parse_precision_overrides(raw))
        except ValueError as exc:
            if ("bad", raw) not in _precision_announced:       # loud, once: a typo must not look like "no override"
                _precision_announced.add(("bad", raw))
                print("[TRT] %s IGNORED (%s): %r" % (PRECISION_OVERRIDE_ENV, exc, raw), flush=True)
                precision_log.append({"stem": stem, "refused": "malformed: %s" % exc})
            return None
    value = table.get(stem)
    if value is None or value == "global":
        return None
    if canonical_model_key(model_key, model_path) in _PROTECTED_KEYS:
        if ("protected", stem) not in _precision_announced:
            _precision_announced.add(("protected", stem))
            print("[TRT] precision override for %s refused: swappers use ROOP_SWAP_FP32 and the swap canary" % stem,
                  flush=True)
            precision_log.append({"stem": stem, "refused": "swapper"})
        return None
    return value


def _announce_precision(stem, requested, glob):
    if requested != glob and ("applied", stem, requested, glob) not in _precision_announced:
        _precision_announced.add(("applied", stem, requested, glob))
        print("[TRT] precision for %s: %s (per-model; global %s)" % (stem, requested, glob), flush=True)
        precision_log.append({"stem": stem, "precision": requested, "global": glob})


def bundle_member_providers(model_key, providers, model_path):
    """Providers for ONE file of a multi-model bundle (buffalo_l), with ONLY its precision override applied.

    insightface builds every file of a bundle from one provider chain, so this is the seam that lets w600k_r50 stay mixed
    while 1k3d68 builds FP32. Deliberately not providers_for(): that would also attach a dynamic-batch shape profile to
    2d106det / 1k3d68 (their batch axis is 'None') and move their engines to a new cache namespace, which is a different
    change from the one being made. Returns None when no override applies (the caller keeps its chain untouched).

    An override can tighten ('fp32') or equal the global setting. Loosening a model under a global 'fp32' would need the
    pre-global chain, which a bundle caller no longer has; that case is refused loudly rather than guessed.
    """
    requested = precision_override_for(model_key, model_path)
    if requested is None:
        return None
    glob = _global_precision()
    if requested == glob:
        return None
    decision = resolve(model_key, requested, providers, model_path)
    if glob == "fp32" or not decision.trt_enabled or decision.policy.trt_supported == "no":
        if decision.trt_enabled:
            print("[TRT] precision for %s: %s ignored inside a bundle while the global setting is fp32" % (
                _stem(model_path), requested), flush=True)
        return None
    _announce_precision(_stem(model_path), decision.effective, glob)
    if decision.effective == "fp32":
        return _force_fp32(providers, decision.model)
    if decision.effective == "bf16":
        return _enable_bf16(providers, decision.model)
    return None


def providers_for(model_key: str, providers, model_path: str | None = None,
                  requested: str | None = None, device_id: int = 0,
                  hardware=None):
    """Return provider options for one model under the active precision policy.

    `requested` defaults to this model's per-model precision (PRECISION_OVERRIDES / ROOP_TRT_MODEL_PRECISION), then to the
    global `trt_precision`. An explicit `requested` is authoritative: harnesses that name a precision get exactly it.
    """
    if requested is None:
        glob = _global_precision()
        requested = precision_override_for(model_key, model_path) or glob
        if model_path:
            _announce_precision(_stem(model_path), requested, glob)
    decision = resolve(model_key, requested, providers, model_path, device_id,
                       hardware=hardware)
    if model_path:
        # Persist the resolved decision at session-construction time.  The key
        # contains the model digest and backend_manager's GPU/runtime
        # fingerprint, so this record cannot be reused across GPUs or model
        # revisions.
        write_decision_cache(decision)
    if not decision.trt_enabled:
        return _finalize(model_key, model_path, providers, device_id), decision
    if decision.policy.trt_supported == "no":
        return _finalize(model_key, model_path, _without_trt(providers),
                           device_id), decision
    if decision.effective == "fp32":
        return _finalize(model_key, model_path,
                           _force_fp32(providers, decision.model),
                           device_id), decision
    if decision.effective == "bf16":
        return _finalize(model_key, model_path,
                           _enable_bf16(providers, decision.model),
                           device_id), decision
    return _finalize(model_key, model_path, providers, device_id), decision


def _finalize(model_key, model_path, providers, device_id=0):
    """Apply the per-model provider policies that need the model identity.

    Two policies live here because neither can be decided in core.py, which
    builds one provider list for the whole process and has never seen a model
    path:

      * the device-verified cuDNN conv algo lowering (see _cudnn_algo), and
      * the TensorRT optimization profile, which is derived from THIS model's
        own dynamic axes and is skipped entirely for a static graph so that a
        fully-static install keeps its existing engine cache namespace.
    """
    providers = _cudnn_algo(model_key, model_path, providers, device_id)
    try:
        # Local import: trt_shape_profile imports canonical_model_key from
        # this module, so a module-level import here would be circular.
        from roop.trt_shape_profile import apply_shape_profile
        providers = apply_shape_profile(providers, model_key, model_path)
    except Exception as _degrade_error:
        # Shape profiling is an optimisation, never a reason to fail a build.
        _swallowed("roop/precision_policy.py:426", _degrade_error, "fallback continued")
    return apply_build_override(providers, model_key, model_path)


# ── per-model TensorRT build overrides ─────────────────────────────────────────────────────────────
#
# core.py sets ONE set of engine-build options for every model: `trt_build_heuristics_enable` exactly when precision is
# 'mixed', and `trt_builder_optimization_level` 3 (ROOP_TRT_BUILDER_OPT_LEVEL). Those decide which tactics a build
# picks, and that is not neutral: swap_canary.py records the same options minus heuristics building a swapper engine that
# runs at full speed and paints the wrong face. Whether the global choice is right for a given SMALL model (a detector,
# a mask net, a recogniser) is therefore a per-model measurement (docs/perf/trt_matrix.md), not a global setting.
#
# A model is identified by its FILE stem, lower-cased ('retinaface_r50', 'xseg', 'w600k_r50'): canonical_model_key is too
# coarse for this (every detector is 'face_detection', every recogniser and landmark net 'recognition').
#
# With no entry the providers and the cache namespace are returned UNCHANGED - an install that never opts in keeps every
# engine it has built. With an entry whose effective options differ from the global ones (or that carries a `tag`), the
# engine AND timing cache directories gain a suffix derived from the effective values, so an engine built under one
# schedule can never be loaded for another and a timing cache cannot carry tactics across them.
#
# SWAPPERS ARE REFUSED (model key 'face_swap'): heuristics stay on there until the explicit-FP32-islands work is done.
#
#   ROOP_TRT_MODEL_BUILD="retinaface_r50:h=0,l=5,tag=r1;xseg:h=1"      h = heuristics (0/1), l = level (0-5), tag = free text
#
# The env var wins over BUILD_OVERRIDES. `tag` exists for the build matrix (two builds of one config need two caches).

#: Adopted overrides, filled only from a measured result. stem -> {"heuristics": bool, "level": int}.
BUILD_OVERRIDES: dict = {}
BUILD_OVERRIDE_ENV = "ROOP_TRT_MODEL_BUILD"
_PROTECTED_KEYS = frozenset({"face_swap"})
_override_announced: set = set()
override_log: list = []         # every override applied or refused in this process, for harnesses and tests


def _stem(model_path) -> str:
    return os.path.splitext(os.path.basename(str(model_path or "")))[0].lower()


def parse_build_overrides(text: str) -> dict:
    """'stem:h=0,l=5,tag=x;stem2:h=1' -> {stem: {'heuristics': bool?, 'level': int?, 'tag': str?}}. Raises ValueError."""
    out = {}
    for part in [p.strip() for p in str(text or "").split(";") if p.strip()]:
        stem, sep, fields = part.partition(":")
        stem = stem.strip().lower()
        if not stem or not sep:
            raise ValueError("expected 'stem:h=0,l=5', got %r" % part)
        entry = {}
        for kv in [f.strip() for f in fields.split(",") if f.strip()]:
            key, eq, val = kv.partition("=")
            key, val = key.strip().lower(), val.strip()
            if not eq:
                raise ValueError("expected key=value, got %r in %r" % (kv, part))
            if key in ("h", "heuristics"):
                if val.lower() not in ("0", "1", "true", "false", "on", "off"):
                    raise ValueError("heuristics must be 0/1, got %r" % val)
                entry["heuristics"] = val.lower() in ("1", "true", "on")
            elif key in ("l", "level"):
                if not val.isdigit() or not 0 <= int(val) <= 5:
                    raise ValueError("level must be 0-5, got %r" % val)
                entry["level"] = int(val)
            elif key == "tag":
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", val):
                    raise ValueError("tag must be 1-24 of [A-Za-z0-9_-], got %r" % val)
                entry["tag"] = val
            else:
                raise ValueError("unknown override field %r in %r" % (key, part))
        if not entry:
            raise ValueError("no fields for %r" % stem)
        out[stem] = entry
    return out


def build_override_for(model_key, model_path):
    """The override entry for this model, or None. Env wins over BUILD_OVERRIDES; swappers are always None."""
    stem = _stem(model_path)
    if not stem:
        return None
    entry = dict(BUILD_OVERRIDES.get(stem) or {})
    raw = os.environ.get(BUILD_OVERRIDE_ENV, "").strip()
    if raw:
        try:
            entry.update(parse_build_overrides(raw).get(stem) or {})
        except ValueError as exc:
            # Loud, once: a malformed experiment flag must not look like a run with no override.
            if ("bad", raw) not in _override_announced:
                _override_announced.add(("bad", raw))
                print("[TRT] %s IGNORED (%s): %r" % (BUILD_OVERRIDE_ENV, exc, raw), flush=True)
                override_log.append({"stem": stem, "refused": "malformed: %s" % exc})
            return None
    if not entry:
        return None
    if canonical_model_key(model_key, model_path) in _PROTECTED_KEYS:
        if ("protected", stem) not in _override_announced:
            _override_announced.add(("protected", stem))
            print("[TRT] build override for %s refused: swappers keep the global build options (heuristics on) until "
                  "the explicit FP32-island work is done" % stem, flush=True)
            override_log.append({"stem": stem, "refused": "swapper"})
        return None
    return entry


def apply_build_override(providers, model_key, model_path):
    """Return *providers* with the per-model heuristics / builder-level override applied (see the block comment above)."""
    try:
        entry = build_override_for(model_key, model_path)
    except Exception as exc:                                    # an optimisation hook never fails a build
        _swallowed("roop/precision_policy.py:build_override", exc, "no override applied")
        return list(providers or ())
    if not entry:
        return list(providers or ())
    stem = _stem(model_path)
    patched, applied = [], False
    for provider in list(providers or ()):
        if not (isinstance(provider, (tuple, list)) and len(provider) == 2 and "tensorrt" in str(provider[0]).lower()):
            patched.append(provider)
            continue
        name, options = provider[0], dict(provider[1])
        heur = bool(options.get("trt_build_heuristics_enable")) if entry.get("heuristics") is None else entry["heuristics"]
        level = int(options.get("trt_builder_optimization_level", 3)) if entry.get("level") is None else entry["level"]
        tag = entry.get("tag")
        differs = (heur != bool(options.get("trt_build_heuristics_enable"))
                   or level != int(options.get("trt_builder_optimization_level", 3)))
        cache = options.get("trt_engine_cache_path")
        if not (differs or tag) or not cache:
            patched.append(provider)
            continue
        identity = json.dumps({"heuristics": heur, "level": level, "tag": tag}, sort_keys=True, separators=(",", ":"))
        suffix = "_ovh%dl%d%s_%s" % (int(heur), level, ("_" + tag) if tag else "",
                                    hashlib.sha256(identity.encode()).hexdigest()[:8])
        scoped = str(cache) + suffix
        try:
            os.makedirs(scoped, exist_ok=True)
        except OSError:
            patched.append(provider)            # unwritable: skip the override rather than fail the session
            continue
        options["trt_build_heuristics_enable"] = heur
        options["trt_builder_optimization_level"] = level
        options["trt_engine_cache_path"] = scoped
        if options.get("trt_timing_cache_path"):
            options["trt_timing_cache_path"] = scoped
        patched.append((name, options))
        applied = True
        if ("applied", stem, suffix) not in _override_announced:
            _override_announced.add(("applied", stem, suffix))
            print("[TRT] build override for %s: heuristics=%s level=%d%s -> cache %s" % (
                stem, heur, level, (" tag=" + tag) if tag else "", suffix), flush=True)
            override_log.append({"stem": stem, "heuristics": heur, "level": level, "tag": tag, "suffix": suffix})
    return patched if applied else list(providers or ())


def _cudnn_algo(model_key, model_path, providers, device_id=0):
    """Apply the device-verified per-model cuDNN conv algo policy.

    Only models in `cudnn_algo.SUSPECT_MODEL_KEYS` are ever probed, and the
    probe only *lowers* a model to DEFAULT when the cuDNN frontend genuinely
    fails on this device. Everything else keeps core.py's global HEURISTIC
    untouched, which matters because HEURISTIC is 1.5-3.4x faster for every
    model that can use it. See roop/cudnn_algo.py for the measurements.
    """
    try:
        from roop import cudnn_algo
        algo = cudnn_algo.probe(model_key, model_path, providers,
                                device_id=device_id)
        return cudnn_algo.apply_algo(providers, algo)
    except Exception as _degrade_error:
        # The policy is an optimisation guard, never a reason to fail a build.
        _swallowed("roop/precision_policy.py:445", _degrade_error, "fallback continued")
        return list(providers or ())


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
