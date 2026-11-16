"""Whole-crop, optical-flow-registered temporal blend of the ENHANCED swap crop.

Where it sits. The pipeline order for one face is

    Swapper -> Enhancer (GFPGAN / CodeFormer / GPEN ...) -> [this blend] -> motion blur / grain
            -> BiSeNet + occlusion mask -> multi-band paste

and nothing enhances after the paste (``ProcessMgr.process_face``: the restorers run inside the processor loop on the
aligned crop; the paste, ``procmgr_masking.composite``, only composites). This stage is the last thing that touches the
enhanced crop's pixels before the realism ops.

What it adds over what already exists. ``one_euro.EnhancerStabilizer`` blends whole crops with ONE scalar and no flow,
so it has to switch itself off on motion; ``temporal_smoother.HighFrequencyFlowStabilizer`` registers first but carries
only the HIGH band. This one registers the WHOLE previous output to the current crop with Farneback flow and blends it
in at ``temporal_enhancer_blend`` (default 0.20), gated per pixel by how well the warp explains the current crop, so a
disocclusion or a mouth opening is taken from the current frame alone:

    flow      = Farneback(cur_small -> prev_small)                      # 64 px grid, upsampled
    warped    = remap(prev_out, flow)                                   # prev registered to cur
    gate      = clip(1 - max(|warped_small - cur_small| - GATE_FLOOR, 0) / (PHOTO_T - GATE_FLOOR), 0, 1)
    out       = (1 - blend * gate) * cur + (blend * gate) * warped

The gate has a dead zone (``GATE_FLOOR``) because the restorer's own per-frame re-draw is already part of
``|warped - cur|`` on the 64 px grid, and that is exactly the noise this stage exists to average; without the dead zone
the gate throttles the blend on a perfectly registered face. Only error beyond the noise floor (a disocclusion, a
mouth opening, flow that missed) takes weight away.

The recursion is on the OUTPUT (an EMA), so a face that is barely moving converges to a ``1/(1-blend)``-frame average
of the restorer's per-frame texture draw while a moving one is registered before it is averaged.

It RESETS (drops the track's history and passes the current crop through) on

  * a non-contiguous frame (a parallel block's first frame, a skipped frame, a seek),
  * a crop-shape change,
  * a SHOT BOUNDARY. The authoritative signal is the pipeline's own cut set (``cut=True``, from
    ``ProcessMgr._shot_boundaries``), the same one that resets the mask state. A second, photometric line catches an
    exposure or lighting STEP the cut set missed: the flow-warped previous crop still disagrees with the current one by
    more than ``SHOT_RESIDUAL`` mean levels on the 64 px grid. It is a heuristic and has a known blind spot: an
    aligned crop of the same swapped face before and after a cut at similar lighting looks like head motion to it, so
    it does not replace the cut set,
  * HIGH OCCLUSION: more than ``OCCLUSION_LIMIT`` of the FACE INTERIOR is the plate's (``keep`` read inside
    ``region``, the landmark hull), so what the swap drew under the occluder is not a continuation of what it drew a
    frame ago. The interior, not the whole crop: an aligned crop is mostly neck, hair and background, which the paste
    restores from the plate BY DESIGN, so a whole-crop share sits above any sensible limit on every clean frame. (That
    is the defect the first regression render exposed: 299 of 312 faces "occluded", the blend never applied.)

Every decision is counted; ``summary_line`` prints beside the other temporal engines' lines, because a filter that
returns its input untouched without saying so is indistinguishable from one that never ran (AGENTS.md).

The class follows the contract of the other temporal engines (``enabled``, ``reset``, ``warmup_frames``,
``clone_for_block``, ``summary_line``) so ``ProcessMgr`` can hand each parallel block its own clone.
"""
from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed
from roop.one_euro import ema_warmup_frames

DEFAULT_BLEND = 0.20
FLOW_SIZE = 64
# Mean |warped_prev - cur| on the 64 px grid, 0-255, above which the pair is treated as a lighting / exposure step.
# Synthetic only (tests/test_enhancer_and_realism.py): a +40 level step reads ~37, a restorer re-draw of a still face
# ~4. Not measured on real footage, so it is the second line behind the pipeline's own cut set.
SHOT_RESIDUAL = 24.0
# Per-pixel photometric gate on the 64 px grid: no penalty below GATE_FLOOR (the restorer's own re-draw lives there),
# then the blend weight falls linearly to zero at PHOTO_T.
GATE_FLOOR = 6.0
PHOTO_T = 40.0
# Share of the crop that is the plate's (keep mask >= 0.5) above which the history is dropped.
OCCLUSION_LIMIT = 0.35
MAX_TRACKS = 16


def occluded_share(keep: Optional[np.ndarray], region: Optional[np.ndarray] = None) -> float:
    """Share of ``region`` (a mask of the face interior, any size) where ``keep`` (0..1, 1 = the plate is restored)
    is at least 0.5. With no ``region`` it is the share of the whole mask, which is only meaningful for a mask that
    already covers the face and nothing else."""
    if keep is None:
        return 0.0
    k = np.asarray(keep)
    if k.ndim == 3:
        k = k[..., 0]
    if region is not None:
        r = np.asarray(region)
        if r.ndim == 3:
            r = r[..., 0]
        if r.shape[:2] != k.shape[:2]:
            r = cv2.resize(r.astype(np.uint8), (k.shape[1], k.shape[0]), interpolation=cv2.INTER_NEAREST)
        sel = r > 0
        if sel.any():
            return float(np.mean(k[sel] >= 0.5))
    return float(np.mean(k >= 0.5))
MIN_CROP = 32

_COUNTERS = ("applied", "seeded", "reset_noncontiguous", "reset_shape", "reset_shot", "reset_occlusion",
             "skipped_no_key", "skipped_small", "errors")


class _Tally:
    """Counters shared by an engine and every per-block clone of it (a shallow copy passes this by reference, so a
    clone's work is not lost with the block; see ``temporal_smoother._Counters``)."""

    __slots__ = ("_lock", "_v", "gate_sum", "gate_n")

    def __init__(self):
        self._lock = threading.Lock()
        self._v: Dict[str, int] = {}
        self.gate_sum = 0.0
        self.gate_n = 0

    def bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._v[name] = self._v.get(name, 0) + n

    def gate(self, mean_gate: float) -> None:
        with self._lock:
            self.gate_sum += float(mean_gate)
            self.gate_n += 1

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            out: Dict[str, float] = {k: self._v.get(k, 0) for k in _COUNTERS}
            out["mean_gate"] = (self.gate_sum / self.gate_n) if self.gate_n else 0.0
            return out

    def clear(self) -> None:
        with self._lock:
            self._v.clear()
            self.gate_sum, self.gate_n = 0.0, 0


def _small_gray(crop: np.ndarray, size: int) -> np.ndarray:
    img = crop if crop.dtype == np.uint8 else np.clip(crop, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


class TemporalEnhancerBlend:
    """Per-track flow-registered EMA of the enhanced aligned crop. See the module docstring."""

    def __init__(self, blend: Optional[float] = None, enabled: bool = True, flow_size: int = FLOW_SIZE,
                 shot_residual: float = SHOT_RESIDUAL, photo_t: float = PHOTO_T,
                 occlusion_limit: float = OCCLUSION_LIMIT, max_tracks: int = MAX_TRACKS):
        self.enabled = bool(enabled)
        self.blend = DEFAULT_BLEND if blend is None else float(blend)
        self.flow_size = int(flow_size)
        self.shot_residual = float(shot_residual)
        self.photo_t = max(1.0, float(photo_t))
        self.occlusion_limit = float(occlusion_limit)
        self.max_tracks = max(1, int(max_tracks))
        self._states: "OrderedDict[Any, dict]" = OrderedDict()
        self._lock = threading.RLock()
        self._tally = _Tally()

    # ``weight`` mirrors HighFrequencyFlowStabilizer so the shared engine plumbing can read either.
    @property
    def blend(self) -> float:
        return self._blend

    @blend.setter
    def blend(self, value: float) -> None:
        self._blend = float(min(0.9, max(0.0, value)))

    weight = blend

    # -- lifecycle ----------------------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._states.clear()
        self._tally.clear()

    def flush(self) -> None:
        """Drop the histories but keep the counters (a shot cut mid-render must not zero the report)."""
        with self._lock:
            self._states.clear()

    def warmup_frames(self, eps: float = 0.01) -> int:
        """The seed decays as ``blend ** W`` (the current crop enters with ``1 - blend``)."""
        return ema_warmup_frames(1.0 - self._blend, eps)

    def clone_for_block(self) -> "TemporalEnhancerBlend":
        clone = copy.copy(self)
        clone._lock = threading.RLock()
        clone._states = OrderedDict()
        return clone

    # -- the filter ---------------------------------------------------------------------------------
    def blend_crop(self, crop: np.ndarray, track_id: Any = None, frame_index: Optional[int] = None,
                   keep: Optional[np.ndarray] = None, cut: bool = False,
                   region: Optional[np.ndarray] = None) -> np.ndarray:
        """``crop`` blended with its flow-registered predecessor. Never raises; a crop that could not be blended is
        returned unchanged and counted. ``keep`` is the 0..1 mask where 1 = restore the plate (any size) and
        ``region`` the face interior it is read inside (see ``occluded_share``); ``cut`` says ``frame_index`` is the
        first frame of a new shot (the pipeline's own cut set)."""
        try:
            return self._blend_crop(crop, track_id, frame_index, keep, cut, region)
        except Exception as exc:                            # a smoothing stage must never cost the frame
            self._tally.bump("errors")
            _swallowed("roop/enhancer_stabilizer.py:blend_crop", exc,
                       "this face was not temporally blended; the render continues")
            return crop

    def _blend_crop(self, crop, track_id, frame_index, keep, cut=False, region=None):
        if not self.enabled or self._blend <= 0.0 or crop is None:
            return crop
        if frame_index is None:
            self._tally.bump("skipped_no_key")
            return crop
        h, w = crop.shape[:2]
        if min(h, w) < MIN_CROP or crop.ndim != 3:
            self._tally.bump("skipped_small")
            return crop

        key = track_id if track_id is not None else "_"
        small = _small_gray(crop, self.flow_size)
        occluded = occluded_share(keep, region)

        with self._lock:
            state = self._states.get(key)
            reason = None
            if state is None:
                reason = "seeded"
            elif int(frame_index) != state["frame"] + 1:
                reason = "reset_noncontiguous"
            elif cut:
                reason = "reset_shot"
            elif state["crop"].shape != crop.shape:
                reason = "reset_shape"
            elif occluded > self.occlusion_limit:
                reason = "reset_occlusion"
            if reason is not None:
                self._seed(key, crop, small, frame_index)
                self._tally.bump(reason)
                return crop
            prev_crop, prev_small = state["crop"], state["small"]

        # Farneback on the 64 px grid: flow[y, x] = where in PREV the pixel (x, y) of CUR came from.
        flow = cv2.calcOpticalFlowFarneback(small, prev_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        gx, gy = np.meshgrid(np.arange(self.flow_size, dtype=np.float32),
                             np.arange(self.flow_size, dtype=np.float32))
        warped_small = cv2.remap(prev_small, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        err = np.abs(warped_small.astype(np.float32) - small.astype(np.float32))
        if float(err.mean()) > self.shot_residual:          # the flow cannot explain it: another shot
            with self._lock:
                self._seed(key, crop, small, frame_index)
            self._tally.bump("reset_shot")
            return crop

        flow_up = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        fx = np.float32(w) / np.float32(self.flow_size)
        fy = np.float32(h) / np.float32(self.flow_size)
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        warped = cv2.remap(prev_crop, mx + flow_up[..., 0] * fx, my + flow_up[..., 1] * fy, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        gate = np.clip(1.0 - np.maximum(err - GATE_FLOOR, 0.0) / max(self.photo_t - GATE_FLOOR, 1.0), 0.0, 1.0)
        gate = cv2.resize(gate, (w, h), interpolation=cv2.INTER_LINEAR)
        wgt = (self._blend * gate)[..., None]
        out = (1.0 - wgt) * crop.astype(np.float32) + wgt * warped
        if crop.dtype == np.uint8:
            out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        else:
            out = out.astype(crop.dtype)
        with self._lock:
            self._seed(key, out, _small_gray(out, self.flow_size), frame_index)
        self._tally.bump("applied")
        self._tally.gate(float(gate.mean()))
        return out

    def _seed(self, key, crop, small, frame_index) -> None:
        stored = crop if crop.dtype == np.uint8 else np.clip(crop, 0, 255).astype(np.uint8)
        self._states[key] = {"crop": stored.copy(), "small": small, "frame": int(frame_index)}
        self._states.move_to_end(key)
        while len(self._states) > self.max_tracks:
            self._states.popitem(last=False)

    # -- reporting ----------------------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        s = self._tally.snapshot()
        s["blend"] = self._blend
        return s

    def summary_line(self) -> Optional[str]:
        s = self.stats()
        total = sum(s[k] for k in _COUNTERS if k != "errors")
        if total == 0:
            return None
        return ("[EnhancerBlend] flow-registered whole-crop EMA w=%.2f: applied %d/%d (%.1f%%), seeded %d, "
                "non-contiguous %d, shape %d, shot-cut %d, occlusion %d, no-key %d, small %d, mean gate %.2f, "
                "errors %d" % (s["blend"], s["applied"], total, 100.0 * s["applied"] / total, s["seeded"],
                               s["reset_noncontiguous"], s["reset_shape"], s["reset_shot"],
                               s["reset_occlusion"], s["skipped_no_key"], s["skipped_small"], s["mean_gate"],
                               s["errors"]))
