"""Startup canary for the small TensorRT models: XSeg, w600k_r50, 2d106det, 1k3d68. Verdicts are cached by ENGINE HASH.

WHY THIS EXISTS

roop/swap_canary.py closed one hole: a TensorRT engine that builds, warms up, reports the TensorRT provider, runs at full
speed and computes the wrong thing. predictor.verify_and_warmup cannot see it, the return code cannot, the test suite
cannot. The same hole is open for every other model that rides TensorRT, and these four are the ones whose output is read
as ground truth downstream: XSeg is the mask the compositor keeps, w600k_r50 is the embedding the identity gate compares,
and 1k3d68 / 2d106det are the landmarks `_refine_kps_from_68` writes back into `face.kps` before the swap net is aligned.

HOW IT WORKS

Two fixed synthetic inputs (the swap canary's "skin" and "noise" images, scaled the way each model's own pre-processing
scales a crop) go through the live session and through a transient CUDA FP32 session of the same ONNX (TF32 off, so the
reference is a real FP32 one). The outputs are compared with a metric chosen for what the model's consumer does with it:

    mask        mean |sigmoid_a - sigmoid_b| on the raw output and IoU at 0.5   (what the keep-mask is built from)
    embedding   cosine of the 512-d vectors                              (what the identity gate thresholds)
    landmarks   mean / max distance in 192-crop px after the same decode `Landmark.get` does

The floors are not the fidelity gates. Those are tests/trt_precision_fidelity.py's job and a good FP16 engine misses them
on a few percent of faces. These floors catch the swap-canary failure class - a wrong picture, a collapsed or non-finite
output - and sit between the worst real-face value of a GOOD mixed engine and the value of an engine answering for a
different face (docs/perf/aux_canary_calibration.md).

CACHED BY ENGINE HASH

The reference session costs a second of CUDA init and a transient arena, and the verdict only changes when the engine does.
So the verdict is stored under a key made of the ONNX digest, the build options that shape an engine, and the content hash
of every cached engine file in the session's engine cache directory whose name carries the model's ONNX graph name (ORT
names an engine `TensorrtExecutionProvider_TRTKernel_graph_<graph name>_<hash>_...`; several models share a graph name such
as `main_graph`, so the match is a superset, and a rebuild of ANY engine in it re-checks - spurious, never unsafe). A wiped
cache, a driver / TensorRT / ORT upgrade (they change the cache directory), a changed option, a different GPU or a
rebuilt engine all produce a new key; nothing is trusted across them. Verdicts live in models/runtime_profiles/
aux_canary.json (gitignored). Delete it to force a re-check. A check that could not run is NOT cached.

SCOPE: a session that is not on TensorRT is skipped (CUDA/CPU run the graph's own dtypes), which includes the 3060, where
TensorRT is not admitted. FP32 TensorRT engines ARE checked - the cache makes that free after the first start, and tactic
selection can go wrong at any precision. ROOP_AUX_CANARY=0 disables it.

ON FAILURE `guard` rebuilds the model on a TensorRT FP32 engine and re-checks that, then on CUDA/CPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed
from roop.env import env_bool
from roop.swap_canary import CanaryResult, _image, non_trt_providers, trt_fp16_active

CANARY_ENV = "ROOP_AUX_CANARY"
CANARY_DEFAULT = True
#: Bump when the canary's INPUTS or DECODERS change (floors are already part of every verdict's key).
VERSION = 1
STORE_NAME = "aux_canary.json"

_CASES = (("skin", 0), ("noise", 0))


@dataclass(frozen=True)
class Spec:
    kind: str                       # 'mask' | 'embedding' | 'landmarks'
    floors: Tuple[Tuple[str, float], ...]       # (metric, bound); 'cosine' and 'iou' are lower bounds, the rest upper
    points: int = 0                 # landmarks: how many trailing points the decoder keeps
    dim: int = 0                    # landmarks: values per point in the raw output (2 or 3)


#: Floors calibrated 2026-10-10 on 500 real faces plus a wrong-face control: docs/perf/aux_canary_calibration.md.
SPECS: Dict[str, Spec] = {
    "xseg": Spec("mask", (("soft_mean_abs", 0.08), ("iou", 0.60))),
    "w600k_r50": Spec("embedding", (("cosine", 0.99),)),
    "2d106det": Spec("landmarks", (("mean_px", 1.5), ("max_px", 5.0)), points=106, dim=2),
    "1k3d68": Spec("landmarks", (("mean_px", 4.0), ("max_px", 12.0)), points=68, dim=3),
}
_LOWER_BOUNDS = frozenset(("cosine", "iou"))


def enabled() -> bool:
    return env_bool(CANARY_ENV, CANARY_DEFAULT)


def stem_of(model_file: Optional[str]) -> str:
    return os.path.splitext(os.path.basename(str(model_file or "")))[0].lower()


# ── inputs and metrics ──────────────────────────────────────────────────────────────────────────────────────────────
def _shape_hw(session) -> Tuple[int, int]:
    shape = session.get_inputs()[0].shape
    dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
    if len(dims) == 4 and dims[3] == 3:             # NHWC (XSeg)
        return dims[1] or 256, dims[2] or 256
    return dims[2] or 192, dims[3] or 192


def build_feeds(stem: str, session) -> Optional[List[Dict[str, np.ndarray]]]:
    """The canary feeds for *session*, scaled the way the model's own pre-processing scales a crop."""
    spec = SPECS.get(stem)
    if spec is None or len(session.get_inputs()) != 1:
        return None
    name = session.get_inputs()[0].name
    h, w = _shape_hw(session)
    feeds = []
    for kind, seed in _CASES:
        chw = _image(kind, seed, h, w)                          # CHW float32 in [0, 1]
        if spec.kind == "mask":                                 # Mask_XSeg.Run: NHWC, /255
            x = np.transpose(chw, (1, 2, 0))[None]
        elif spec.kind == "embedding":                          # ArcFaceONNX: (x - 127.5) / 127.5, NCHW
            x = ((chw * 255.0 - 127.5) / 127.5)[None]
        else:                                                   # Landmark: mean 0, std 1 on 0..255, NCHW
            x = (chw * 255.0)[None]
        feeds.append({name: np.ascontiguousarray(x, dtype=np.float32)})
    return feeds


def _decode_points(spec: Spec, raw: np.ndarray, crop: int) -> np.ndarray:
    """Landmark.get's decode, stopping before the inverse affine: crop-space xy of the kept points."""
    pred = np.asarray(raw, np.float64).reshape(-1)
    pred = pred.reshape(-1, spec.dim)[-spec.points:, :2]
    return (pred + 1.0) * (crop // 2)


def compare(stem: str, candidate: np.ndarray, reference: np.ndarray, crop: int = 192) -> Dict[str, float]:
    """The metrics SPECS[stem].floors read. Non-finite candidate output scores as failing on every metric."""
    spec = SPECS[stem]
    a = np.asarray(candidate, np.float64)
    b = np.asarray(reference, np.float64)
    bad = float("inf")
    if a.shape != b.shape or not np.isfinite(a).all():
        return {k: (-1.0 if k in _LOWER_BOUNDS else bad) for k, _ in spec.floors}
    if spec.kind == "mask":
        ma, mb = a > 0.5, b > 0.5
        union = np.logical_or(ma, mb).sum()
        return {"soft_mean_abs": float(np.abs(a - b).mean()),
                "iou": 1.0 if union == 0 else float(np.logical_and(ma, mb).sum() / union)}
    if spec.kind == "embedding":
        x, y = a.reshape(-1), b.reshape(-1)
        return {"cosine": float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))}
    d = np.linalg.norm(_decode_points(spec, a, crop) - _decode_points(spec, b, crop), axis=1)
    return {"mean_px": float(d.mean()), "max_px": float(d.max())}


def judge(stem: str, metrics: Dict[str, float]) -> bool:
    for metric, bound in SPECS[stem].floors:
        v = metrics.get(metric)
        if v is None or not np.isfinite(v):
            return False
        if (v < bound) if metric in _LOWER_BOUNDS else (v > bound):
            return False
    return True


# ── identity: engine hash + verdict store ───────────────────────────────────────────────────────────────────────────
_LOCK = threading.RLock()
_MEMO: Dict[str, Any] = {}              # in-process: file stamp -> hash, verdict key -> CanaryResult
_ENGINE_PREFIX = "TensorrtExecutionProvider_TRTKernel_graph_"


def _store_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "runtime_profiles", STORE_NAME)


def _load_store() -> dict:
    try:
        with open(_store_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and data.get("version") == VERSION:
            return data
    except (OSError, ValueError):
        pass
    return {"version": VERSION, "verdicts": {}, "hashes": {}}


def _save_store(data: dict) -> None:
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = "%s.%d.tmp" % (path, os.getpid())
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError as exc:                                  # an unwritable store costs a re-check, never a failure
        _swallowed("roop/aux_canary.py:_save_store", exc, "verdict not persisted")


def _file_hash(path: str, store: dict) -> str:
    """blake2b of a file's content, memoised on (path, size, mtime) so each engine is read once per change."""
    st = os.stat(path)
    stamp = "%s|%d|%d" % (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    hit = _MEMO.get(stamp) or store["hashes"].get(stamp)
    if hit:
        _MEMO[stamp] = hit
        return hit
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    store["hashes"][stamp] = _MEMO[stamp] = h.hexdigest()
    return _MEMO[stamp]


def engine_hashes(cache_dir: Optional[str], graph_name: str, store: Optional[dict] = None) -> List[str]:
    """Content hashes of the cached engines that could be this session's. [] when there is no engine cache."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return []
    store = store if store is not None else _load_store()
    prefix = _ENGINE_PREFIX + graph_name + "_"
    out = []
    for name in sorted(os.listdir(cache_dir)):
        if name.startswith(prefix) and name.endswith(".engine"):
            out.append(_file_hash(os.path.join(cache_dir, name), store))
    return out


def _trt_options(session) -> dict:
    return (session.get_provider_options() or {}).get("TensorrtExecutionProvider") or {}


_KEY_OPTIONS = ("trt_fp16_enable", "trt_layer_norm_fp32_fallback", "trt_build_heuristics_enable",
                "trt_builder_optimization_level", "trt_max_workspace_size", "trt_profile_min_shapes",
                "trt_profile_opt_shapes", "trt_profile_max_shapes", "trt_bf16_enable")


def verdict_key(stem: str, session, model_file: str, store: dict) -> Optional[Tuple[str, dict]]:
    """(key, identity) for this live engine, or None when it cannot be identified (no engine cache to hash)."""
    opts = _trt_options(session)
    cache_dir = opts.get("trt_engine_cache_path")
    try:
        graph = session.get_modelmeta().graph_name
    except Exception as exc:
        _swallowed("roop/aux_canary.py:verdict_key", exc, "no engine identity; verdict not cached")
        return None
    engines = engine_hashes(cache_dir, graph, store)
    if not engines:
        return None
    from roop.precision_policy import _model_digest
    spec = SPECS[stem]          # the floors are part of the identity: edit one and every verdict made under the old one is a miss
    identity = {"v": VERSION, "stem": stem, "spec": [spec.kind, [list(f) for f in spec.floors], spec.points, spec.dim],
                "model": _model_digest(model_file),
                "engines": engines, "cache": os.path.basename(str(cache_dir)),
                "options": {k: str(opts.get(k)) for k in _KEY_OPTIONS}}
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:32], identity


# ── the check ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _reference_session(model_file: str):
    import onnxruntime
    from roop.utilities import get_onnx_session_options
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        return None
    device = 0
    try:
        import roop.globals
        device = int(getattr(roop.globals, "cuda_device_id", 0) or 0)
    except Exception as exc:
        _swallowed("roop/aux_canary.py:_reference_session", exc, "reference on device 0")
    return onnxruntime.InferenceSession(
        model_file, get_onnx_session_options(),
        providers=[("CUDAExecutionProvider", {"use_tf32": "0", "device_id": str(device)}), "CPUExecutionProvider"])


def on_tensorrt(session) -> bool:
    try:
        return bool(session.get_providers()) and session.get_providers()[0] == "TensorrtExecutionProvider"
    except Exception as exc:
        _swallowed("roop/aux_canary.py:on_tensorrt", exc, "treated as not on TensorRT")
        return False


def check_session(stem: str, session, model_file: str,
                  reference_factory: Optional[Callable[[], Any]] = None) -> CanaryResult:
    """Verdict for the live *session*: cached by engine hash, otherwise computed against a transient FP32 reference.

    Never raises. A check that cannot run is returned as skipped (passed=None) and is not cached.
    """
    tag = "aux:%s" % stem
    if not enabled():
        return CanaryResult(tag, None, "disabled by %s=0" % CANARY_ENV)
    if stem not in SPECS:
        return CanaryResult(tag, None, "no canary spec for %s" % stem)
    if not on_tensorrt(session):
        return CanaryResult(tag, None, "session is not on TensorRT; nothing can drift")
    started = time.perf_counter()
    try:
        with _LOCK:
            store = _load_store()
            ident = verdict_key(stem, session, model_file, store)
            if ident is None and _trt_options(session).get("trt_engine_cache_path"):
                # Cold start: ORT builds the engine on the first inference, so there is nothing to hash yet. Run one
                # canary feed to build it, then identify it - otherwise the first start would never be cached.
                warm = build_feeds(stem, session)
                if warm is not None:
                    session.run(None, warm[0])
                    ident = verdict_key(stem, session, model_file, store)
            key = ident[0] if ident else None
            if key is not None:
                hit = _MEMO.get("verdict:" + key)
                if hit is not None:
                    return hit
                rec = store["verdicts"].get(key)
                if rec is not None:
                    res = CanaryResult(tag, rec["passed"], rec["reason"] + " [cached by engine hash]", rec.get("min_ssim"),
                                       dict(rec.get("metrics") or {}), 0.0)
                    _MEMO["verdict:" + key] = res
                    return res
            feeds = build_feeds(stem, session)
            if feeds is None:
                return CanaryResult(tag, None, "inputs are not the single image this canary understands")
            crop = _shape_hw(session)[0]
            ref = (reference_factory or (lambda: _reference_session(model_file)))()
            if ref is None:
                return CanaryResult(tag, None, "no CUDA reference available")
            try:
                per_case: Dict[str, float] = {}
                ok = True
                worst: Dict[str, float] = {}
                for (kind, seed), feed in zip(_CASES, feeds):
                    got = session.run(None, feed)[0]
                    want = ref.run(None, feed)[0]
                    m = compare(stem, got, want, crop)
                    ok = ok and judge(stem, m)
                    for metric, v in m.items():
                        per_case["%s%d:%s" % (kind, seed, metric)] = round(float(v), 6)
                        w = worst.get(metric)
                        worst[metric] = v if w is None else (min(w, v) if metric in _LOWER_BOUNDS else max(w, v))
            finally:
                del ref
            floors = ", ".join("%s %s %g" % (k, ">=" if k in _LOWER_BOUNDS else "<=", b) for k, b in SPECS[stem].floors)
            reason = ("engine output matches the FP32 reference (%s)" % floors if ok else
                      "engine output diverges from the FP32 reference (floors: %s; worst %s)" % (
                          floors, {k: round(v, 5) for k, v in worst.items()}))
            res = CanaryResult(tag, ok, reason, None, per_case, time.perf_counter() - started)
            if key is not None:
                _MEMO["verdict:" + key] = res
                store["verdicts"][key] = {"passed": ok, "reason": reason, "metrics": per_case, "stem": stem,
                                          "engines": ident[1]["engines"], "when": time.strftime("%Y-%m-%d %H:%M:%S")}
                _save_store(store)
            return res
    except Exception as exc:
        _swallowed("roop/aux_canary.py:check_session", exc, "canary skipped")
        return CanaryResult(tag, None, "canary could not run (%s: %s)" % (type(exc).__name__, exc),
                            seconds=time.perf_counter() - started)


def guard(stem: str, session, model_file: str, providers: Sequence[Any],
          build: Callable[[Sequence[Any]], Any]) -> Tuple[Any, List[Any], CanaryResult]:
    """Check *session*; on a failed verdict rebuild via *build(chain)* on TensorRT FP32, then on CUDA/CPU.

    Returns (session, providers actually used, the verdict that decided). A skipped check keeps the session untouched.
    """
    first = check_session(stem, session, model_file)
    tag = first.tag
    if not first.ran:
        if enabled() and stem in SPECS and on_tensorrt(session):
            print("[AuxCanary] %s: skipped - %s" % (stem, first.reason), flush=True)
        return session, list(providers), first
    if first.passed:
        print("[AuxCanary] %s: OK - %s%s" % (stem, first.reason.split(" (")[0],
              "" if "cached" in first.reason else " (%.1fs)" % first.seconds), flush=True)
        return session, list(providers), first
    print("[AuxCanary] %s: FAILED - %s" % (stem, first.reason), flush=True)
    chain = list(providers)
    if trt_fp16_active(chain):
        try:
            from roop.precision_policy import _force_fp32, canonical_model_key
            # the canonical key names the FP32 cache directory providers_for(..., requested='fp32') already uses
            fp32 = _force_fp32(chain, canonical_model_key(stem, model_file))
            rebuilt = build(fp32)
            again = check_session(stem, rebuilt, model_file)
            if again.passed is not False:
                print("[AuxCanary] %s: running on the TensorRT FP32 engine instead" % stem, flush=True)
                return rebuilt, fp32, again
            print("[AuxCanary] %s: the TensorRT FP32 engine failed too - %s" % (stem, again.reason), flush=True)
        except Exception as exc:
            _swallowed("roop/aux_canary.py:guard", exc, "TensorRT FP32 rebuild failed; using CUDA/CPU")
    plain = non_trt_providers(chain) or ["CPUExecutionProvider"]
    print("[AuxCanary] %s: falling back to %s" % (stem, plain[0] if isinstance(plain[0], str) else plain[0][0]), flush=True)
    return build(plain), plain, CanaryResult(tag, False, "fell back to CUDA/CPU after a failed TensorRT engine")


__all__ = ["CANARY_ENV", "SPECS", "VERSION", "build_feeds", "check_session", "compare", "enabled", "engine_hashes",
           "guard", "judge", "stem_of", "verdict_key"]
