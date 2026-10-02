"""Startup canary for TensorRT swapper engines.

WHY THIS EXISTS

``inswapper_128`` unrolls InstanceNorm into ReduceMean / Sub / Mul(x, x) / Sqrt /
Div, and the squared terms reach ~7e6 -- far past FP16's 65504.  Whether the
TensorRT FP16/"mixed" engine keeps those nodes in FP32 is decided by tactic
selection at BUILD time, and it is sensitive to the build options.  Measured
2026-10-03 on an RTX 4070 (TRT 10.9, ORT 1.23.2), 14 real crops, SSIM against
CUDA FP32:

    production mixed options                  SSIM 0.9973 min   identity unchanged
    same, without build_heuristics            SSIM 0.7489       identity 0.762 -> 0.087
    same, without the 2 GB workspace/sequential               SSIM 0.7490       identity 0.086

Both corrupt engines build, warm up, report the TensorRT provider and run at the
same speed, so ``predictor.verify_and_warmup`` cannot tell them apart; every
swapped face is simply the wrong picture, with no error.  A driver / TensorRT /
ORT upgrade or a changed option can flip a working install into that state.

This module closes that gap: right after the swapper session is built, push two
fixed synthetic inputs through it and through a transient CUDA FP32 reference
of the same model, and compare.  Synthetic inputs are enough -- over 9
input/seed combinations a good engine scored >= 0.9958 and the corrupt one
<= 0.8653, so the 0.98 floor sits in a wide gap.

SCOPE: only a TensorRT provider with ``trt_fp16_enable`` true can drift this way
(an FP32 engine, CUDA and CPU all run the graph's own dtypes), so the canary is
skipped everywhere else -- including the secondary 3060 profile, where TensorRT
is not admitted.  ``ROOP_SWAP_CANARY=0`` disables it.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from roop.env import env_bool
from roop import predictor

#: Measured separation: good engine >= 0.9958, corrupt engine <= 0.8653.
SSIM_FLOOR = 0.98
CANARY_ENV = "ROOP_SWAP_CANARY"
CANARY_DEFAULT = True

#: Canary inputs: (kind, seed).  Two different statistics so one lucky pattern
#: cannot hide an overflow in the other.
_CASES = (("skin", 0), ("noise", 0))


@dataclass
class CanaryResult:
    """Outcome of one canary run.  ``passed`` is None when it was skipped."""

    tag: str
    passed: Optional[bool]
    reason: str
    min_ssim: Optional[float] = None
    per_case: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def failed(self) -> bool:
        return self.passed is False

    @property
    def ran(self) -> bool:
        return self.passed is not None


def enabled() -> bool:
    return env_bool(CANARY_ENV, CANARY_DEFAULT)


def _name(provider: Any) -> str:
    return str(provider[0] if isinstance(provider, (tuple, list)) else provider)


def trt_fp16_active(providers: Optional[Sequence[Any]]) -> bool:
    """True when the provider list asks TensorRT for FP16 kernels."""
    for p in providers or ():
        if "tensorrt" in _name(p).lower():
            opts = p[1] if isinstance(p, (tuple, list)) and len(p) > 1 else {}
            value = opts.get("trt_fp16_enable", False) if isinstance(opts, dict) else False
            return str(value).strip().lower() in ("1", "true", "yes", "on")
    return False


def non_trt_providers(providers: Sequence[Any]) -> List[Any]:
    return [p for p in providers if "tensorrt" not in _name(p).lower()]


def _has_cuda(providers: Sequence[Any]) -> bool:
    return any("cuda" in _name(p).lower() for p in providers)


# -- inputs -----------------------------------------------------------------

def _image(kind: str, seed: int, h: int, w: int) -> np.ndarray:
    """A deterministic CHW float32 image in [0, 1]."""
    rng = np.random.RandomState(seed)
    if kind == "noise":
        img = rng.rand(h, w, 3).astype(np.float32)
    else:  # "skin": face-like - skin tone, low-frequency shading, dark features
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yy /= max(1, h - 1)
        base = np.stack([0.80 - 0.15 * yy, 0.62 - 0.10 * yy, 0.52 - 0.10 * yy], -1)
        shade = cv2.GaussianBlur(rng.rand(h, w, 1).astype(np.float32), (0, 0), h / 12.8)
        img = np.clip(base + shade[..., None].reshape(h, w, 1) * 0.25, 0.0, 1.0)
        for fx, fy in ((0.34, 0.41), (0.66, 0.41), (0.50, 0.72)):
            cv2.circle(img, (int(fx * w), int(fy * h)), max(2, h // 18),
                       (0.10, 0.08, 0.08), -1)
        img = cv2.GaussianBlur(img, (0, 0), max(0.5, h / 106.0))
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)), dtype=np.float32)


def _latent(seed: int, shape: Sequence[int], emap: Optional[np.ndarray]) -> np.ndarray:
    """A unit-norm identity vector, projected through ``emap`` when the model
    takes the inswapper-style latent (as ``_compute_latent`` does)."""
    rng = np.random.RandomState(1000 + seed)
    dim = int(emap.shape[0]) if emap is not None else int(shape[-1])
    vec = rng.randn(1, dim).astype(np.float32)
    if emap is not None:
        vec = vec @ np.asarray(emap, dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec.reshape(shape).astype(np.float32)


def build_feeds(session: Any, emap: Optional[np.ndarray] = None) -> Optional[List[Dict[str, np.ndarray]]]:
    """Canary feed dicts for *session*, or None when its inputs are not the
    (one rank-4 image, optional rank-2 identity) shape this canary understands."""
    metas = list(session.get_inputs())
    feeds: List[Dict[str, np.ndarray]] = []
    for kind, seed in _CASES:
        feed: Dict[str, np.ndarray] = {}
        images = 0
        for meta in metas:
            shape = predictor._concrete_shape(meta, (128,))
            if shape is None:
                return None
            dtype = predictor._numpy_dtype(meta)
            if len(shape) == 4 and shape[1] == 3:
                images += 1
                value = _image(kind, seed, int(shape[2]), int(shape[3]))[None]
            elif len(shape) == 2:
                value = _latent(seed, shape, emap)
            else:
                return None
            feed[meta.name] = value.astype(dtype)
        if images != 1:
            return None
        feeds.append(feed)
    return feeds


# -- comparison -------------------------------------------------------------

def ssim(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """Mean structural similarity over an HxWxC float image pair.

    Same definition as ``skimage.metrics.structural_similarity`` with its
    defaults (7x7 uniform window, sample covariance, K1=0.01, K2=0.03), so the
    thresholds measured with skimage carry over; tests pin the agreement.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    win = 7
    npx = win * win
    cov_norm = npx / (npx - 1.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    total = 0.0
    for ch in range(a.shape[2]):
        x, y = a[..., ch], b[..., ch]
        blur = lambda im: cv2.blur(im, (win, win), borderType=cv2.BORDER_REFLECT)
        ux, uy = blur(x), blur(y)
        uxx, uyy, uxy = blur(x * x), blur(y * y), blur(x * y)
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)
        s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
        pad = (win - 1) // 2
        total += float(s[pad:-pad, pad:-pad].mean())
    return total / a.shape[2]


def _to_hwc(out: np.ndarray) -> np.ndarray:
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _compare(candidate: np.ndarray, reference: np.ndarray) -> float:
    """SSIM between two model outputs; -1.0 when the candidate is not finite."""
    cand, ref = _to_hwc(candidate), _to_hwc(reference)
    if cand.shape != ref.shape or not np.isfinite(cand).all():
        return -1.0
    # A FIXED data range from the output's convention, never the image's own
    # min/max: rescaling by the reference's range magnifies FP16 noise on a
    # low-contrast output.  The floor was measured on uint8 crops of a [0,1]
    # output, i.e. data_range = 1.0 here.
    lo, hi = float(ref.min()), float(ref.max())
    if lo >= -0.05 and hi <= 1.5:
        data_range = 1.0            # [0, 1] images (inswapper family)
    elif lo >= -1.5 and hi <= 1.5:
        data_range = 2.0            # [-1, 1] images
    else:
        data_range = 255.0          # 0..255 images
    if cand.ndim == 2:
        cand, ref = cand[..., None], ref[..., None]
    return ssim(cand, ref, data_range)


# -- public entry point -----------------------------------------------------

def check_engine(session: Any,
                 reference_factory: Callable[[], Any],
                 tag: str,
                 emap: Optional[np.ndarray] = None,
                 floor: float = SSIM_FLOOR) -> CanaryResult:
    """Compare *session* against the session ``reference_factory`` builds.

    The reference is created, used and dropped inside this call so its CUDA
    arena does not outlive the check.  Never raises: a canary that cannot run
    is reported as skipped, not as a failure.
    """
    import time
    started = time.perf_counter()
    feeds = build_feeds(session, emap)
    if feeds is None:
        return CanaryResult(tag, None, "inputs are not (one image, optional identity vector)")
    reference = None
    try:
        reference = reference_factory()
        per_case: Dict[str, float] = {}
        for (kind, seed), feed in zip(_CASES, feeds):
            ref_out = reference.run(None, feed)[0]
            got_out = session.run(None, feed)[0]
            per_case[f"{kind}{seed}"] = round(_compare(got_out, ref_out), 5)
    except Exception as exc:
        _swallowed("roop/swap_canary.py:check_engine", exc, "canary skipped")
        return CanaryResult(tag, None, f"canary could not run ({type(exc).__name__}: {exc})",
                            seconds=time.perf_counter() - started)
    finally:
        del reference
    worst = min(per_case.values())
    passed = bool(math.isfinite(worst) and worst >= floor)
    reason = ("engine output matches the FP32 reference" if passed else
              f"engine output diverges from the FP32 reference (SSIM {worst:.4f} < {floor})")
    return CanaryResult(tag, passed, reason, min_ssim=worst, per_case=per_case,
                        seconds=time.perf_counter() - started)


__all__ = ["CANARY_ENV", "CanaryResult", "SSIM_FLOOR", "build_feeds", "check_engine",
           "enabled", "non_trt_providers", "ssim", "trt_fp16_active"]
