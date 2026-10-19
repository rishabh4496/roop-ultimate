"""Flow-advected mask fusion with Schmitt-trigger hysteresis, and a guided matte.

Masks use this project's convention: 1 = keep the ORIGINAL pixel (occluded),
0 = use the swap. So the brief's occlusion probability P_occ is the mask itself.

WHAT EXISTS ALREADY, AND WHAT THIS ADDS
---------------------------------------
`roop/occlusion_mask.TemporalMaskSmoother` is a flow-warped EMA with ONE global
residual test (reset the whole state if the mean warped-vs-current difference is
large) and a fixed blend. It has no per-pixel notion of "the warped history is
wrong HERE", and no hysteresis: a pixel whose probability hovers around 0.5 (a
finger, a hair strand, a glasses frame) still flips between keep and swap.

`FlowHysteresisFuser` adds both:

  P_f = a * c * warp(P_f_prev) + (1 - a * c) * P_raw          (per-pixel c)
  c   = exp(-|I_t - warp(I_{t-1})|_1 / sigma)                 flow consistency

where I are the aligned crops. Where the flow does not explain the pixel (an
object arriving or leaving) c -> 0 and the history is ignored at once: no ghost.
The fused probability then goes through a dual-threshold soft ramp whose
position depends on the pixel's previous state: a pixel that was occluded stays
occluded until P_f falls under `t_low`, a pixel that was clear needs P_f over
`t_high` to turn occluded. A probability hovering between them cannot chatter.

`guided_matte` is the edge-preserving snap (He et al.): the mask is re-expressed as
a local linear function of the guide image, so its edge follows the real edge in
the crop. `cv2.ximgproc` is contrib-only and absent here, so it is the box-filter
form, the same one `frequency_split.guided_filter` uses.

Pure cv2/numpy; one small state dict per track key; no model, no GPU.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

FLOW_SIZE = 128


def _gray(crop: np.ndarray, size: int) -> np.ndarray:
    g = crop if crop.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(crop[..., :3]), cv2.COLOR_BGR2GRAY)
    if g.shape[0] != size or g.shape[1] != size:
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g


def guided_matte(mask: np.ndarray, guide: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """Guided image filter of `mask` (HxW float 0..1) by `guide` (HxW or HxWx3 uint8/float)."""
    p = np.asarray(mask, np.float32)
    g = np.asarray(guide)
    if g.ndim == 3:
        g = cv2.cvtColor(np.ascontiguousarray(g[..., :3]).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32) / (255.0 if g.dtype != np.float32 or g.max() > 1.5 else 1.0)
    if g.shape != p.shape:
        g = cv2.resize(g, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_LINEAR)
    k = (2 * int(radius) + 1, 2 * int(radius) + 1)
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mg, mp = box(g), box(p)
    cov = box(g * p) - mg * mp
    var = box(g * g) - mg * mg
    a = cov / (var + float(eps))
    b = mp - a * mg
    return np.clip(box(a) * g + box(b), 0.0, 1.0)


def hysteresis_ramp(p: np.ndarray, prev_on: np.ndarray, t_high: float = 0.60,
                    t_low: float = 0.38, width: float = 0.08) -> np.ndarray:
    """Soft Schmitt trigger: the ramp sits at `t_low` for pixels that were occluded
    and at `t_high` for pixels that were clear; `width` is its half-width."""
    centre = np.where(prev_on, t_low, t_high).astype(np.float32)
    lo = centre - width
    return np.clip((p - lo) / (2.0 * width), 0.0, 1.0).astype(np.float32)


class FlowHysteresisFuser:
    """Per-track temporal fusion. `update` is called once per frame IN ORDER."""

    def __init__(self, alpha: float = 0.7, sigma: float = 14.0, t_high: float = 0.60,
                 t_low: float = 0.38, width: float = 0.08, flow_size: int = FLOW_SIZE,
                 max_tracks: int = 64, max_flow_frac: float = 0.03):
        self.alpha = float(alpha)
        self.sigma = float(sigma)            # 8-bit grey levels
        self.t_high, self.t_low, self.width = float(t_high), float(t_low), float(width)
        self.flow_size = int(flow_size)
        self.max_tracks = int(max_tracks)
        self.max_flow_frac = float(max_flow_frac)
        self._state: Dict[Any, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.applied = 0
        self.seeded = 0
        self.skipped = 0
        self.reset_pixels = 0.0

    def reset(self) -> None:
        with self._lock:
            self._state.clear()

    def _flow(self, cur_small: np.ndarray, prev_small: np.ndarray) -> np.ndarray:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        return dis.calc(cur_small, prev_small, None)       # samples prev at cur's pixels

    @staticmethod
    def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def update(self, p_raw: np.ndarray, crop: np.ndarray, key: Any = 0,
               frame_index: Optional[int] = None) -> np.ndarray:
        """Fuse `p_raw` (HxW, 1 = occluded) with the advected history of `key`."""
        raw = np.clip(np.asarray(p_raw, np.float32), 0.0, 1.0)
        if raw.ndim != 2 or crop is None:
            self.skipped += 1
            return raw
        h, w = raw.shape
        small = _gray(crop, self.flow_size)
        with self._lock:
            st = self._state.get(key)
            if (st is not None and frame_index is not None and st['frame_index'] == frame_index
                    and st['out'].shape == raw.shape):
                # ProcessMgr hands the stabilizer the same frame's mask a second
                # time (it was already applied inside process_mask). Fusing it
                # with its own history again would double-count the frame.
                return st['out'].copy()
            contiguous = (st is not None and frame_index is not None
                          and st['frame_index'] == frame_index - 1 and st['pf'].shape == raw.shape)
            if not contiguous:
                self.seeded += 1 if st is None else 0
                self.skipped += 1 if st is not None else 0
                out = hysteresis_ramp(raw, np.zeros_like(raw, bool), self.t_high, self.t_low, self.width)
                self._state[key] = {'pf': raw.copy(), 'out': out.copy(), 'gray': small,
                                    'full': _gray(crop, h), 'frame_index': frame_index}
                return out
            flow = self._flow(small, st['gray'])
            if self.flow_size != h:
                flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) * (w / float(self.flow_size))
            # Dense flow will happily "explain" a patch that is simply different
            # (an object that has left) by warping unrelated pixels onto it, which
            # keeps the consistency weight high and the old occlusion alive. Aligned
            # crops cancel the head's motion, so what remains is small; bound it.
            lim = self.max_flow_frac * w
            flow = np.clip(flow, -lim, lim)
            full = _gray(crop, h)
            warped_pf = self._warp(st['pf'], flow)
            warped_out = self._warp(st['out'], flow)
            warped_img = self._warp(st['full'], flow)
            c = np.exp(-np.abs(full.astype(np.float32) - warped_img.astype(np.float32)) / self.sigma)
            c = cv2.GaussianBlur(c, (0, 0), 1.5)
            k = self.alpha * c
            pf = k * warped_pf + (1.0 - k) * raw
            out = hysteresis_ramp(pf, warped_out > 0.5, self.t_high, self.t_low, self.width)
            self.applied += 1
            self.reset_pixels += float((c < 0.3).mean())
            self._state[key] = {'pf': pf.astype(np.float32), 'out': out, 'gray': small,
                                'full': full, 'frame_index': frame_index}
            if len(self._state) > self.max_tracks:
                self._state.pop(next(iter(self._state)))
            return out



from roop.one_euro import MaskStabilizer                                  # noqa: E402


class FusionMaskStabilizer(MaskStabilizer):
    """`MaskStabilizer` with the flow-advected, hysteresis fusion in place of its EMA.

    Same constructor and the same `apply(mask, kps, t, guide, track_id)` contract, so
    the per-chunk factory, warm-up frame count and the parallel stabilizer path work
    unchanged. A call without a guide crop or a track id falls back to the parent.
    """

    def __init__(self, *args, fusion=None, guided=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._fuser = FlowHysteresisFuser(**(fusion or {}))
        self.guided = bool(guided)

    def reset(self):
        super().reset()
        self._fuser.reset()

    def apply(self, mask, kps, t, guide=None, track_id=None):
        if mask is None or guide is None or track_id is None:
            return super().apply(mask, kps, t, guide=guide, track_id=track_id)
        m = np.asarray(mask, dtype=np.float32)
        shape = m.shape
        m2 = m.reshape(m.shape[:2])
        out = self._fuser.update(m2, guide, key=track_id, frame_index=int(t))
        if self.guided:
            out = guided_matte(out, guide, 4, 1e-3)
        return np.clip(out, 0.0, 1.0).astype(np.float32).reshape(shape)
