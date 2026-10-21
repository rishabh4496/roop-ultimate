"""Point-tracking back ends for the pre-pass's optical-flow chain (`flow_coast.FlowCoaster`).

`FlowCoaster` carries one face's landmark cloud from frame to frame: it tracks ~70 points forward, tracks them
back as a consistency check, fits a similarity to the survivors and refuses the step when the fit is not one rigid
head. The only thing in that loop that touches pixels is "where did these N points go between two small grayscale
crops". That call is what a back end supplies, with `cv2.calcOpticalFlowPyrLK`'s contract:

    track(prev, nxt, pts) -> (new_pts, status, err)       pts / new_pts (N, 1, 2) float32, status (N, 1) uint8

THE LADDER (`select`, ``ROOP_FLOW_BACKEND`` = auto | cv | torch | nvof; auto = nvof when it is really there, else cv):

    nvof    `NvofPointTracker`: dense flow on the NVIDIA optical-flow ASIC (`cv2.cuda_NvidiaOpticalFlow_2_0`),
            sampled at the points. Needs a CUDA-enabled OpenCV build AND an NVIDIA GPU with the engine. The
            opencv-python wheels do not ship CUDA, so it is `None` on a stock install (this repo's venv:
            `cv2.cuda.getCudaEnabledDeviceCount() == 0`). UNVERIFIED ON HARDWARE: the wrapper follows the call
            sequence `HighFrequencyFlowStabilizer._flow_nvof` already uses and is unit-tested against a fake engine
            (the dispatch, the sampling and the refusal rules), not against a real OFA.
    torch   `TorchLK`: pyramidal Lucas-Kanade on a torch device, batched over every point of a step. Runs wherever
            torch does (CUDA, and ROCm / MPS builds), so it is the GPU option on non-NVIDIA hardware. MEASURED, not
            assumed: see `tests/probe_phase14_prepass.py --section flow` for how it compares with the CPU call on
            this workload (small crops, ~70 points: kernel-launch bound, so the CPU call is not slower).
    cv      `CvLK`: `cv2.calcOpticalFlowPyrLK`, exactly what `FlowCoaster` called before. The default fallback and the
            behaviour of every render that does not ask for another.

Below the flow ladder, `tracker.FaceTracker`'s constant-velocity Kalman prediction still covers a gap flow refused;
that fallback is unchanged and needs no pixels.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

try:
    import cv2
except ImportError:                                           # pragma: no cover - cv2 is a hard dependency
    cv2 = None


class PointTracker:
    """The contract. `name` is what the pre-pass prints."""

    name = "base"

    def track(self, prev: np.ndarray, nxt: np.ndarray, pts: np.ndarray
              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        raise NotImplementedError


# ── cv: the CPU call FlowCoaster always made ─────────────────────────────────────────────────────────
class CvLK(PointTracker):
    name = "cv"

    def __init__(self, win: int = 15, levels: int = 3, iters: int = 20, eps: float = 0.01):
        self.win, self.levels = int(win), int(levels)
        self._lk = dict(winSize=(self.win, self.win), maxLevel=self.levels,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps)))

    def track(self, prev, nxt, pts):
        return cv2.calcOpticalFlowPyrLK(prev, nxt, pts, None, **self._lk)


# ── torch: batched pyramidal Lucas-Kanade ────────────────────────────────────────────────────────────
def _torch():
    import sys
    return sys.modules.get("torch")


class TorchLK(PointTracker):
    """Pyramidal LK on a torch device. Forward-additive, one 2x2 solve per point per iteration, every point of the
    call at once. Border handling is replicate (grid_sample 'border'); a point whose window leaves the image, or whose
    gradient matrix is singular (no texture), comes back with status 0, as OpenCV reports it."""

    name = "torch"

    def __init__(self, win: int = 15, levels: int = 3, iters: int = 20, eps: float = 0.01,
                 device: Optional[str] = None, min_eig: float = 1e-3):
        import torch
        self._t = torch
        self.win, self.levels, self.iters, self.eps = int(win), int(levels), int(iters), float(eps)
        self.min_eig = float(min_eig)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        half = (self.win - 1) / 2.0
        ax = torch.arange(self.win, dtype=torch.float32, device=self.device) - half
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        self._offs = torch.stack([xx, yy], dim=-1).reshape(1, self.win * self.win, 2)    # (1, W*W, 2) in px
        k = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=self.device) / 8.0
        self._kx = k.reshape(1, 1, 3, 3)
        self._ky = k.t().reshape(1, 1, 3, 3)

    # one image -> pyramid of (image, Ix, Iy), each (1, 1, H, W)
    def _pyramid(self, gray: np.ndarray):
        torch, F = self._t, self._t.nn.functional
        img = torch.as_tensor(np.ascontiguousarray(gray), dtype=torch.float32, device=self.device)[None, None]
        out = []
        for lv in range(self.levels + 1):
            pad = F.pad(img, (1, 1, 1, 1), mode="replicate")
            out.append((img, F.conv2d(pad, self._kx), F.conv2d(pad, self._ky)))
            if lv < self.levels:
                img = F.avg_pool2d(F.pad(img, (0, img.shape[3] % 2, 0, img.shape[2] % 2), mode="replicate"), 2)
        return out

    def _sample(self, tex, centres):
        """(N, W*W) samples of `tex` (1,1,H,W) in a win x win window around each centre (N, 2) px."""
        F = self._t.nn.functional
        h, w = tex.shape[2], tex.shape[3]
        pos = centres[:, None, :] + self._offs                                   # (N, W*W, 2)
        gx = (pos[..., 0] + 0.5) / w * 2.0 - 1.0
        gy = (pos[..., 1] + 0.5) / h * 2.0 - 1.0
        grid = self._t.stack([gx, gy], dim=-1)[None]                             # (1, N, W*W, 2)
        got = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return got[0, 0]

    def track(self, prev, nxt, pts):
        torch = self._t
        if prev.shape != nxt.shape or prev.ndim != 2:
            return None, None, None
        p0 = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        n = len(p0)
        if n == 0:
            return pts.copy(), np.zeros((0, 1), np.uint8), np.zeros((0, 1), np.float32)
        with torch.no_grad():
            pa, pb = self._pyramid(prev), self._pyramid(nxt)
            base = torch.as_tensor(p0, device=self.device)
            flow = torch.zeros(n, 2, device=self.device)
            ok = torch.ones(n, dtype=torch.bool, device=self.device)
            for lv in range(self.levels, -1, -1):
                img_a, ix, iy = pa[lv]
                img_b = pb[lv][0]
                scale = 2.0 ** lv
                centre = base / scale
                a = self._sample(img_a, centre)
                gx, gy = self._sample(ix, centre), self._sample(iy, centre)
                gxx, gxy, gyy = (gx * gx).sum(1), (gx * gy).sum(1), (gy * gy).sum(1)
                det = gxx * gyy - gxy * gxy
                tr = gxx + gyy
                # smallest eigenvalue of the 2x2 normal matrix, per window pixel (OpenCV's minEigThreshold)
                min_eig = (tr - torch.sqrt(torch.clamp(tr * tr - 4.0 * det, min=0.0))) / (2.0 * self.win * self.win)
                good = (min_eig > self.min_eig) & (det > 1e-9)
                inv_det = torch.where(good, 1.0 / det.clamp(min=1e-9), torch.zeros_like(det))
                d = flow.clone()
                for _ in range(self.iters):
                    b = self._sample(img_b, centre + d)
                    it = b - a
                    bx, by = -(gx * it).sum(1), -(gy * it).sum(1)
                    dx = (gyy * bx - gxy * by) * inv_det
                    dy = (gxx * by - gxy * bx) * inv_det
                    step = torch.stack([dx, dy], dim=1)
                    d = d + step
                    if float(step.abs().max()) < self.eps:
                        break
                ok &= good
                flow = d * 2.0 if lv > 0 else d
            new = base + flow
            h, w = prev.shape
            inside = (new[:, 0] >= 0) & (new[:, 0] <= w - 1) & (new[:, 1] >= 0) & (new[:, 1] <= h - 1)
            ok &= inside
            # per-point patch error at the finest level: mean absolute grey difference, like OpenCV's err
            img_a0, _, _ = pa[0]
            err = (self._sample(img_a0, base) - self._sample(pb[0][0], new)).abs().mean(1)
            new_np = new.detach().cpu().numpy().astype(np.float32).reshape(-1, 1, 2)
            st = ok.detach().cpu().numpy().astype(np.uint8).reshape(-1, 1)
            er = err.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        return new_np, st, er


# ── nvof: the NVIDIA optical-flow engine, sampled at the points ──────────────────────────────────────
def nvof_class() -> Optional[Any]:
    """The binding class, or None. The same two spellings `HighFrequencyFlowStabilizer._nvof_class` accepts."""
    if cv2 is None:
        return None
    cuda = getattr(cv2, "cuda", None)
    return getattr(cuda, "NvidiaOpticalFlow_2_0", None) or getattr(cv2, "cuda_NvidiaOpticalFlow_2_0", None)


def cuda_opencv_available() -> bool:
    try:
        return bool(cv2 is not None and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0)
    except Exception as exc:                                # cv2.error on a build without the module
        _swallowed("roop/flow_backend.py:cuda_opencv_available", exc, "the NVOF rung is not offered")
        return False


def nvof_unavailable_reason() -> Optional[str]:
    """Why the NVOF rung is not usable here, in words (None when it is)."""
    if cv2 is None:
        return "OpenCV is not importable"
    if not cuda_opencv_available():
        return "this OpenCV build has no CUDA device (the opencv-python wheels ship without CUDA)"
    if nvof_class() is None:
        return "this OpenCV build has no NvidiaOpticalFlow_2_0 binding"
    return None


class NvofPointTracker(PointTracker):
    """Dense flow on the OFA engine for the crop pair, read at each point (bilinear), forward only.

    NVOF returns S10.5 fixed-point vectors; `convertToFloat` brings them to float pixels. The engine sees the
    two crops FlowCoaster already cuts (a face, at most ~192 px a side), so this is a small dense job, not a
    whole-frame one. `engine_factory(size)` is injectable so the dispatch and sampling can be tested without a
    GPU; the default builds the real engine and so raises on a machine without one.
    """

    name = "nvof"

    def __init__(self, engine_factory=None):
        self._factory = engine_factory or self._real_engine
        self._engines = {}

    @staticmethod
    def _real_engine(size):                                  # pragma: no cover - needs a CUDA OpenCV + an NVIDIA GPU
        cls = nvof_class()
        const = lambda name, default: getattr(cls, name, getattr(getattr(cv2, "cuda", None), name, default))  # noqa: E731
        try:
            return cls.create(size, const("NV_OF_PERF_LEVEL_FAST", 20), const("NV_OF_OUTPUT_VECTOR_GRID_SIZE_1", 1),
                              const("NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED", 0), False, False, False, 0)
        except TypeError:
            return cls.create(size)

    def _engine(self, shape):
        key = (int(shape[1]), int(shape[0]))                 # OpenCV Size is (width, height)
        if key not in self._engines:
            self._engines[key] = self._factory(key)
        return self._engines[key]

    def _dense(self, prev, nxt):                             # pragma: no cover - needs the real engine
        engine = self._engine(prev.shape)
        a, b, out = cv2.cuda_GpuMat(), cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        a.upload(prev)
        b.upload(nxt)
        raw = engine.calc(a, b, None)
        flow = engine.convertToFloat(raw, out)
        return np.asarray((flow if flow is not None else out).download(), dtype=np.float32)

    def track(self, prev, nxt, pts):
        flow = self._dense(prev, nxt)
        return self.sample(flow, pts, prev.shape)

    @staticmethod
    def sample(flow, pts, shape):
        """Move each point by the flow under it (bilinear). status 0 outside the image or where flow is not finite."""
        if flow is None or flow.ndim != 3 or flow.shape[2] != 2 or flow.shape[:2] != tuple(shape[:2]):
            return None, None, None
        p = np.asarray(pts, np.float32).reshape(-1, 2)
        h, w = shape[:2]
        inside = (p[:, 0] >= 0) & (p[:, 0] <= w - 1) & (p[:, 1] >= 0) & (p[:, 1] <= h - 1)
        x0 = np.clip(np.floor(p[:, 0]).astype(int), 0, w - 2)
        y0 = np.clip(np.floor(p[:, 1]).astype(int), 0, h - 2)
        fx, fy = (p[:, 0] - x0)[:, None], (p[:, 1] - y0)[:, None]
        v = ((1 - fx) * (1 - fy) * flow[y0, x0] + fx * (1 - fy) * flow[y0, x0 + 1]
             + (1 - fx) * fy * flow[y0 + 1, x0] + fx * fy * flow[y0 + 1, x0 + 1])
        finite = np.isfinite(v).all(axis=1)
        ok = (inside & finite).astype(np.uint8).reshape(-1, 1)
        new = (p + np.where(finite[:, None], v, 0.0)).astype(np.float32).reshape(-1, 1, 2)
        return new, ok, np.zeros((len(p), 1), np.float32)


# ── selection ────────────────────────────────────────────────────────────────────────────────────────
_CHOICES = ("auto", "cv", "torch", "nvof")


def requested() -> str:
    raw = str(os.environ.get("ROOP_FLOW_BACKEND", "auto") or "auto").strip().lower()
    return raw if raw in _CHOICES else "auto"


def select(choice: Optional[str] = None, *, win: int = 15, levels: int = 3, log=None) -> PointTracker:
    """The back end for this run. Anything that cannot be built falls DOWN the ladder (never raises) and says so."""
    choice = (choice or requested()).lower()
    say = log or (lambda *_a, **_k: None)

    def cv_tracker():
        return CvLK(win=win, levels=levels)

    if choice in ("nvof", "auto"):
        reason = nvof_unavailable_reason()
        if reason is None:
            try:
                return NvofPointTracker()
            except Exception as exc:                                 # pragma: no cover - needs a CUDA OpenCV
                _swallowed("roop/flow_backend.py:nvof", exc, "falling back to CPU Lucas-Kanade")
                reason = "%s: %s" % (type(exc).__name__, exc)
        if choice == "nvof":
            say("[Flow] ROOP_FLOW_BACKEND=nvof is not available (%s); using CPU Lucas-Kanade." % reason)
    if choice == "torch":
        try:
            return TorchLK(win=win, levels=levels)
        except Exception as exc:
            _swallowed("roop/flow_backend.py:torch", exc, "falling back to CPU Lucas-Kanade")
            say("[Flow] ROOP_FLOW_BACKEND=torch could not start (%s); using CPU Lucas-Kanade." % exc)
    return cv_tracker()
