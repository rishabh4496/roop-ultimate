"""TensorRT optimization profiles derived from each model's real input graph.

TensorRT needs a min/opt/max profile only for inputs that actually carry a
dynamic dimension.  A profile is not free: it becomes part of the engine, so
handing one to a model whose input is fully static either does nothing or
tells the builder to optimize for a shape the pipeline never feeds.

That distinction is not theoretical here.  Measured on the shipped weights
(2026-09-03, ``onnx.load`` over each graph's non-initializer inputs):

    hyperswap_1a_256  target  [1, 3, 256, 256]     STATIC   <- the live swapper
    gpen_bfr_256      input   [1, 3, 256, 256]     STATIC
    GPEN-BFR-512      input   [1, 3, 512, 512]     STATIC
    GFPGANv1.4        input   [1, 3, 512, 512]     STATIC
    yoloface_8n       input   [1, 3, 640, 640]     STATIC   <- the fixed export
    xseg              input   [N, 256, 256, 3]     batch only
    retinaface_r50    input   [b, 3, h, w]         FULLY DYNAMIC  <- the detector

So a 128/256/512 spatial profile over "swapper and enhancer models" is a no-op
on every shipped restorer and on the live swapper, and the one model in this
pipeline that genuinely needs a profile is the DETECTOR -- whose useful band is
the det_size dropdown (320/512/640/960/1280), not 128-512.

This module therefore does not apply a fixed band by model family.  It reads
the graph, pins every static dimension to its own value, and opens only the
dimensions the model actually leaves free, clamped to a band the pipeline can
really feed.  A model with no dynamic dimension returns None, which matters for
a second reason: emitting nothing leaves the engine cache namespace untouched,
so an install whose models are all static is not forced into a cold TensorRT
rebuild by this feature.

``yoloface_8n`` above is the reminder that this has to be read from the file
rather than assumed.  It is a fixed [1,3,640,640] export, and the 2026-08-24
session records it silently returning zero faces for an entire render at any
other det_size because the resulting InvalidArgument was swallowed upstream.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from roop.precision_policy import canonical_model_key


# The band a dynamic spatial axis may be opened to, per model family, as
# (min, opt, max).  ``opt`` is the shape the pipeline feeds most often and is
# the one TensorRT actually tunes tactics for, so it is the load-bearing value.
#
# face_detection: the UI dropdown offers 320/512/640/960/1280 and config.yaml
#   ships 512, which is the measured-fastest retinaface geometry (1.30x over
#   640 at slightly better recall, 2026-08-24).
# face_swap / enhancers: kept at the requested 128/256/512 band for any model
#   that does leave a spatial axis free.  Every shipped restorer and the live
#   swapper are static, so this is inert on them by construction -- see the
#   module docstring.
_SPATIAL_BANDS: Dict[str, Tuple[int, int, int]] = {
    "face_detection": (320, 512, 1280),
    "face_swap": (128, 256, 512),
    "masking": (128, 256, 512),
    "codeformer": (128, 256, 512),
    "codeformer_fp16": (128, 256, 512),
    "gpen_256": (128, 256, 512),
    "gpen_256_pro": (128, 256, 512),
    "gpen_realistic": (128, 256, 512),
    "gpen_512": (128, 256, 512),
    "gfpgan": (128, 256, 512),
    "restoreformer_pp": (128, 256, 512),
    "recognition": (112, 112, 224),
}

# Batch is opened separately: this pipeline batches the swapper across frames
# (``perf_batch_swap``) but never batches a detector.
_BATCH_BAND: Tuple[int, int, int] = (1, 1, 8)

_DEFAULT_SPATIAL_BAND: Tuple[int, int, int] = (128, 256, 512)

_lock = threading.Lock()
_spec_cache: Dict[str, Tuple["InputSpec", ...]] = {}


@dataclass(frozen=True)
class InputSpec:
    """One graph input; ``dims`` holds an int for static, None for dynamic."""

    name: str
    dims: Tuple[Optional[int], ...]

    @property
    def dynamic(self) -> bool:
        return any(d is None for d in self.dims)


# Per-model STATIC overrides: model file stem -> the one shape min = opt = max is pinned to.
#
# A band is the right default for a graph with free axes, but a band has a price this module's
# docstring does not mention: TensorRT sizes the execution context's activation memory for the
# profile's MAX shape.  retinaface_r50 is band (1,3,320,320) / (1,3,512,512) / (8,3,1280,1280); the
# pipeline never batches the detector and never feeds more than one det_size per render, yet every
# pooled instance's first inference grew device memory by ~1.16 GiB (2026-10-09 probe,
# tests/r50_profile_probe.py).  The export is only calibrated at 640 (its priors, and
# face_engine's detector which refuses anything else), so 1x3x640x640 is the whole useful range.
#
# The consequence is that an input of any other size is OUTSIDE the profile, so a model with an
# override also has to be fed that size: RetinaFace3Output reads `pinned_hw` and ignores det_size
# with a warning, instead of letting TensorRT reject the shape and `get_all_faces` swallow the
# error into a render with no faces (the yoloface failure of 2026-08-24).
_STATIC_OVERRIDES: Dict[str, Tuple[int, ...]] = {
    "retinaface_r50": (1, 3, 640, 640),
}

# Whether the overrides apply when ROOP_TRT_STATIC_PROFILE is unset.
_STATIC_PROFILE_DEFAULT = False


@dataclass(frozen=True)
class ShapeProfile:
    """Resolved ORT TensorRT profile options plus their cache identity."""

    min_shapes: str
    opt_shapes: str
    max_shapes: str
    namespace: str
    static_shape: Optional[Tuple[int, ...]] = None

    def as_options(self) -> Dict[str, str]:
        return {
            "trt_profile_min_shapes": self.min_shapes,
            "trt_profile_opt_shapes": self.opt_shapes,
            "trt_profile_max_shapes": self.max_shapes,
        }


def _cache_path(model_path: str) -> str:
    root = os.path.join(os.path.dirname(__file__), "..", "models",
                        "runtime_profiles", "shapes")
    stat = os.stat(model_path)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                  "%s_%d_%d" % (os.path.basename(model_path), stat.st_size,
                                int(stat.st_mtime)))
    return os.path.join(root, stem + ".json")


def _read_graph_inputs(model_path: str) -> Tuple[InputSpec, ...]:
    """Parse the ONNX graph's real (non-initializer) inputs."""
    import onnx  # local: the app must still start when onnx is unavailable

    model = onnx.load(model_path, load_external_data=False)
    initializers = {t.name for t in model.graph.initializer}
    specs: List[InputSpec] = []
    for entry in model.graph.input:
        if entry.name in initializers:
            # Some exports list every weight as a graph input.  Those are not
            # feed inputs and must never enter a profile.
            continue
        dims: List[Optional[int]] = []
        for dim in entry.type.tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                dims.append(None)
        specs.append(InputSpec(entry.name, tuple(dims)))
    return tuple(specs)


def graph_inputs(model_path: str | None) -> Tuple[InputSpec, ...]:
    """Return the model's feed inputs, memoised in-process and on disk.

    Parsing a 500MB swapper is not something to do on every session build, so
    the result is cached under the model's size and mtime.  Any failure yields
    an empty tuple, which disables profiling for that model rather than
    failing its session.
    """
    if not model_path or not os.path.isfile(model_path):
        return ()
    key = os.path.abspath(model_path)
    with _lock:
        if key in _spec_cache:
            return _spec_cache[key]
    specs: Tuple[InputSpec, ...] = ()
    path = None
    try:
        path = _cache_path(model_path)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            specs = tuple(
                InputSpec(item["name"],
                          tuple(None if d is None else int(d)
                                for d in item["dims"]))
                for item in payload["inputs"])
    except Exception as _degrade_error:
        _swallowed("roop/trt_shape_profile.py:173", _degrade_error, "fallback continued")
        specs = ()
    if not specs:
        try:
            specs = _read_graph_inputs(model_path)
        except Exception as _degrade_error:
            _swallowed("roop/trt_shape_profile.py:178", _degrade_error, "fallback continued")
            specs = ()
        if specs and path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump({"model": os.path.basename(model_path),
                               "inputs": [{"name": s.name, "dims": list(s.dims)}
                                          for s in specs]}, handle, indent=2)
                os.replace(tmp, path)
            except Exception as _degrade_error:
                _swallowed("roop/trt_shape_profile.py:189", _degrade_error, "fallback continued")
                pass
    with _lock:
        _spec_cache[key] = specs
    return specs


def _band_for(model_key: str | None,
              model_path: str | None) -> Tuple[int, int, int]:
    return _SPATIAL_BANDS.get(canonical_model_key(model_key, model_path),
                              _DEFAULT_SPATIAL_BAND)


def _dim_band(spec: InputSpec, index: int,
              spatial: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Resolve (min, opt, max) for one dimension of one input.

    A static dimension is pinned to its own value in all three profiles: that
    is what tells TensorRT the axis cannot move, and it is why a
    batch-dynamic-only model does not accidentally get its spatial axes opened.
    """
    value = spec.dims[index]
    if value is not None:
        return value, value, value
    if index == 0:
        return _BATCH_BAND
    if len(spec.dims) == 4 and index == 1:
        # A dynamic channel count is not something this pipeline feeds; keep it
        # pinned at 3 rather than inventing a range.
        return 3, 3, 3
    return spatial


def enabled() -> bool:
    """Whether shape profiling is switched on (default yes).

    ROOP_TRT_SHAPE_PROFILE=0 turns it off.  It exists for two reasons: it is
    the escape hatch if a future model's profile is ever wrong, and it is what
    makes the feature A/B-able end to end -- a profile that cannot be turned
    off cannot be measured against its own absence.
    """
    return os.environ.get("ROOP_TRT_SHAPE_PROFILE", "1").strip().lower() not in (
        "0", "off", "false", "no")


def static_enabled() -> bool:
    """Whether the per-model static overrides apply.

    ROOP_TRT_STATIC_PROFILE=1 / 0 forces it; unset follows ``_STATIC_PROFILE_DEFAULT``.  It is
    what makes the override A/B-able against the band it replaces (each arm gets its own engine
    cache namespace, so both stay warm).
    """
    raw = os.environ.get("ROOP_TRT_STATIC_PROFILE")
    if raw is None or not raw.strip():
        return _STATIC_PROFILE_DEFAULT
    return raw.strip().lower() not in ("0", "off", "false", "no")


def _static_override(model_path: str | None,
                     specs: Sequence[InputSpec]) -> Optional[Tuple[int, ...]]:
    """The override shape for *model_path*, or None when it does not apply.

    It applies only to a model with exactly one feed input whose rank matches and whose STATIC
    dimensions agree with the override: a graph that pins a different channel count is not the
    model the override was measured on, and forcing a shape onto it would build a wrong engine.
    """
    if not model_path or not static_enabled() or len(specs) != 1:
        return None
    stem = os.path.splitext(os.path.basename(str(model_path)))[0].lower()
    shape = _STATIC_OVERRIDES.get(stem)
    spec = specs[0]
    if shape is None or len(spec.dims) != len(shape):
        return None
    if any(d is not None and d != s for d, s in zip(spec.dims, shape)):
        return None
    return shape


def pinned_hw(providers: Sequence) -> Optional[Tuple[int, int]]:
    """(width, height) when the TensorRT provider in *providers* carries a STATIC profile.

    Read from the provider options themselves (min == opt == max), not from the request: a
    chain that was stepped down to CUDA/CPU has no profile, and then nothing constrains the
    input size.  Width comes first, matching ``RetinaFace3Output.input_size``.
    """
    for provider in providers or ():
        if not (isinstance(provider, (tuple, list)) and len(provider) == 2
                and "tensorrt" in str(provider[0]).lower()):
            continue
        options = provider[1] or {}
        lo = options.get("trt_profile_min_shapes")
        hi = options.get("trt_profile_max_shapes")
        mid = options.get("trt_profile_opt_shapes")
        if not lo or not (lo == hi == mid) or "," in str(lo):
            return None
        try:
            dims = [int(d) for d in str(lo).split(":", 1)[1].split("x")]
        except (IndexError, ValueError):
            return None
        return (dims[-1], dims[-2]) if len(dims) >= 2 else None
    return None


def resolve_profile(model_key: str | None,
                    model_path: str | None) -> Optional[ShapeProfile]:
    """Return the profile for *model_path*, or None when nothing is dynamic."""
    if not enabled():
        return None
    specs = graph_inputs(model_path)
    if not specs or not any(spec.dynamic for spec in specs):
        return None
    pinned = _static_override(model_path, specs)
    if pinned is not None and ":" not in specs[0].name:
        shape = "%s:%s" % (specs[0].name, "x".join(str(d) for d in pinned))
        # A distinct namespace: an engine built for one shape range is not interchangeable with
        # one built for another, and "static" in the name keeps the two caches recognisable.
        namespace = "_spstatic" + re.sub(r"[^A-Za-z0-9]+", "", shape)[-24:]
        return ShapeProfile(shape, shape, shape, namespace, static_shape=pinned)
    spatial = _band_for(model_key, model_path)
    if any(":" in spec.name for spec in specs):
        # ORT parses these options as "name:dxdxd", splitting on the FIRST
        # colon, so a tensor called "xseg_input:0" (the TensorFlow-style name
        # the shipped XSeg export uses) would be read as name "xseg_input" with
        # a shape of "0:1x256x256x3" and rejected.  There is no escaping in
        # that format, so such a model gets no profile at all rather than a
        # malformed one -- which for XSeg costs nothing, since only its batch
        # axis is dynamic.
        return None
    parts: List[List[str]] = [[], [], []]
    for spec in specs:
        if not spec.dims:
            continue
        bands = [_dim_band(spec, i, spatial) for i in range(len(spec.dims))]
        for slot in range(3):
            shape = "x".join(str(band[slot]) for band in bands)
            parts[slot].append("%s:%s" % (spec.name, shape))
    if not parts[0]:
        return None
    minimum, optimum, maximum = (",".join(part) for part in parts)
    namespace = "_sp" + re.sub(r"[^A-Za-z0-9]+", "",
                               minimum + optimum + maximum)[-24:]
    return ShapeProfile(minimum, optimum, maximum, namespace)


def apply_shape_profile(providers: Sequence, model_key: str | None,
                        model_path: str | None):
    """Return *providers* with a TensorRT profile added where one is warranted.

    The caller's list is never mutated.  When a profile is emitted the engine
    cache path gains a suffix derived from the profile itself: an engine built
    under one profile is not interchangeable with an engine built under
    another, and this project has twice shipped a benchmark that silently
    reused an engine built with different options.
    """
    try:
        profile = resolve_profile(model_key, model_path)
    except Exception as _degrade_error:
        _swallowed("roop/trt_shape_profile.py:280", _degrade_error, "fallback continued")
        profile = None
    if profile is None:
        return list(providers or ())
    patched = []
    for provider in list(providers or ()):
        if (isinstance(provider, (tuple, list)) and len(provider) == 2
                and "tensorrt" in str(provider[0]).lower()):
            name, options = provider[0], dict(provider[1])
            options.update(profile.as_options())
            cache = options.get("trt_engine_cache_path")
            if cache:
                scoped = cache + profile.namespace
                try:
                    os.makedirs(scoped, exist_ok=True)
                except OSError:
                    # An unwritable cache directory is a reason to skip the
                    # profile, not to fail the session.
                    return list(providers or ())
                options["trt_engine_cache_path"] = scoped
                if options.get("trt_timing_cache_path"):
                    options["trt_timing_cache_path"] = scoped
            patched.append((name, options))
        else:
            patched.append(provider)
    return patched


def describe(model_key: str | None, model_path: str | None) -> dict:
    """JSON-safe explanation for diagnostics and the self-verification test."""
    specs = graph_inputs(model_path)
    profile = resolve_profile(model_key, model_path)
    return {
        "model": canonical_model_key(model_key, model_path),
        "inputs": [{"name": s.name, "dims": list(s.dims), "dynamic": s.dynamic}
                   for s in specs],
        "band": list(_band_for(model_key, model_path)),
        "profiled": profile is not None,
        "static_override": (list(profile.static_shape)
                            if profile is not None and profile.static_shape else None),
        "reason": ("dynamic input dimensions present" if profile is not None
                   else "all input dimensions are static; a profile would be inert"),
        "options": profile.as_options() if profile is not None else {},
    }
