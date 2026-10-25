"""Motion-Compensated Temporal ROI Delta-Skipper for near-static faces.

During Stage 2 Pre-Pass:
- Computes frame-to-frame inner-face landmark displacement (`delta_eyes_mouth_px`)
  and ROI illumination delta.
- If a face has `delta_eyes_mouth_px < 0.15` and is not on a keyframe or scene cut
  (capped at max 2 consecutive skipped frames to prevent drift), marks the frame-face
  entry as `REUSE_FLOW_WARP`.

In Stage 3:
- For `REUSE_FLOW_WARP` entries, bypasses Swapper + Restorer neural inference and warps
  the previous frame's final composited face patch forward using the dense NVOFA
  (NVIDIA Optical Flow) vector field.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import roop.globals
from roop.degrade import swallowed as _swallowed

# Inner face keypoint indices for 5-point alignment:
# 0: left eye, 1: right eye, 2: nose, 3: left mouth corner, 4: right mouth corner
INNER_LANDMARK_INDICES = (0, 1, 3, 4)
MAX_CONSECUTIVE_SKIPS = 2
DISPLACEMENT_THRESHOLD_PX = 0.15
ILLUMINATION_DELTA_THRESHOLD = 0.005  # < 0.5% local illumination change


def is_smart_skip_enabled() -> bool:
    """Return whether Temporal Smart-Skip (static reuse) is active."""
    env = os.environ.get("ROOP_TEMPORAL_SMART_SKIP", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return bool(getattr(roop.globals, "temporal_smart_skip", False))


def compute_inner_landmark_displacement(
    curr_kps: Any, prev_kps: Any
) -> float:
    """Compute frame-to-frame inner-face landmark displacement in pixels (eyes and mouth)."""
    if curr_kps is None or prev_kps is None:
        return float("inf")
    ck = np.asarray(curr_kps, dtype=np.float32).reshape(-1, 2)
    pk = np.asarray(prev_kps, dtype=np.float32).reshape(-1, 2)
    if ck.shape[0] < 5 or pk.shape[0] < 5:
        return float("inf")
    if not (np.isfinite(ck).all() and np.isfinite(pk).all()):
        return float("inf")

    idx = list(INNER_LANDMARK_INDICES)
    disp = np.linalg.norm(ck[idx] - pk[idx], axis=1)
    return float(np.mean(disp))


def compute_roi_luma(frame: np.ndarray, bbox: Optional[Any] = None) -> float:
    """Compute normalized mean luminance (0.0 to 1.0) of a face ROI."""
    if frame is None or frame.size == 0:
        return 0.5
    h, w = frame.shape[:2]
    if bbox is not None:
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in bbox[:4]]
            x0 = max(0, min(w - 1, x0))
            x1 = max(x0 + 1, min(w, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(y0 + 1, min(h, y1))
            crop = frame[y0:y1, x0:x1]
        except (ValueError, TypeError, IndexError, AttributeError):
            crop = frame
    else:
        crop = frame

    if crop.size == 0:
        return 0.5
    if crop.ndim == 2:
        y = crop
    else:
        y = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)[..., 0]
    return float(y.mean()) / 255.0


def compute_illumination_delta(curr_luma: float, prev_luma: float) -> float:
    """Absolute difference in normalized ROI illumination."""
    return abs(float(curr_luma) - float(prev_luma))


class DenseFlowWarper:
    """Dense optical flow estimation with NVIDIA Optical Flow (NVOFA) priority."""

    FLOW_TIERS = ("nvof", "cuda_farneback", "dis")

    def __init__(self, flow_tier: str = "auto", engine_factory=None) -> None:
        self.flow_tier = flow_tier
        self.last_flow_tier: Optional[str] = None
        self._engine_factory = engine_factory
        self._tls = threading.local()

    @staticmethod
    def _cuda_available() -> bool:
        try:
            return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except (AttributeError, cv2.error):
            return False

    @staticmethod
    def _nvof_class() -> Optional[Any]:
        cuda = getattr(cv2, "cuda", None)
        return (
            getattr(cuda, "NvidiaOpticalFlow_2_0", None)
            or getattr(cv2, "cuda_NvidiaOpticalFlow_2_0", None)
        )

    def _nvof_engine(self, shape: Tuple[int, int]) -> Optional[Any]:
        key = (int(shape[1]), int(shape[0]))  # (width, height)
        if self._engine_factory is not None:
            return self._engine_factory(key)
        if not self._cuda_available():
            return None
        nvof_cls = self._nvof_class()
        if nvof_cls is None:
            return None
        engines = getattr(self._tls, "nvof", None)
        if engines is None:
            engines = {}
            self._tls.nvof = engines
        if key not in engines:
            try:
                cuda = getattr(cv2, "cuda", None)
                perf = getattr(nvof_cls, "NV_OF_PERF_LEVEL_FAST", getattr(cuda, "NV_OF_PERF_LEVEL_FAST", 20))
                grid = getattr(nvof_cls, "NV_OF_OUTPUT_VECTOR_GRID_SIZE_1", getattr(cuda, "NV_OF_OUTPUT_VECTOR_GRID_SIZE_1", 1))
                hint = getattr(nvof_cls, "NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED", getattr(cuda, "NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED", 0))
                try:
                    engines[key] = nvof_cls.create(key, perf, grid, hint, False, False, False, 0)
                except TypeError:
                    engines[key] = nvof_cls.create(key)
            except (cv2.error, AttributeError, RuntimeError, TypeError):
                engines[key] = False
        return engines[key] or None

    def _flow_nvof(self, cur_small: np.ndarray, prev_small: np.ndarray) -> Optional[np.ndarray]:
        engine = self._nvof_engine(cur_small.shape[:2])
        if engine is None:
            return None
        try:
            if self._cuda_available():
                gpu_mat_cls = getattr(cv2, "cuda_GpuMat", getattr(getattr(cv2, "cuda", None), "GpuMat", None))
                if gpu_mat_cls is not None:
                    cur_gpu = gpu_mat_cls()
                    prev_gpu = gpu_mat_cls()
                    cur_gpu.upload(cur_small)
                    prev_gpu.upload(prev_small)
                    raw = engine.calc(cur_gpu, prev_gpu, None)
                    float_gpu = gpu_mat_cls()
                    converted = engine.convertToFloat(raw, float_gpu)
                    if converted is None:
                        converted = float_gpu
                    flow = converted.download()
                    return np.asarray(flow, dtype=np.float32)
            if hasattr(engine, "calc"):
                raw = engine.calc(cur_small, prev_small, None)
                if hasattr(engine, "convertToFloat"):
                    flow = engine.convertToFloat(raw, None)
                else:
                    flow = raw
                return np.asarray(flow, dtype=np.float32)
            return None
        except Exception as exc:
            _swallowed('roop/temporal_smart_skip.py:nvof', exc, 'NVOF failed; falling back')
            return None

    def _flow_cuda_farneback(self, cur_small: np.ndarray, prev_small: np.ndarray) -> Optional[np.ndarray]:
        if not self._cuda_available():
            return None
        try:
            gpu_mat_cls = getattr(cv2, "cuda_GpuMat", getattr(getattr(cv2, "cuda", None), "GpuMat", None))
            cur_gpu = gpu_mat_cls()
            prev_gpu = gpu_mat_cls()
            cur_gpu.upload(cur_small)
            prev_gpu.upload(prev_small)
            direct = getattr(getattr(cv2, "cuda", None), "calcOpticalFlowFarneback", None)
            if direct is not None:
                flow_gpu = direct(cur_gpu, prev_gpu, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                return np.asarray(flow_gpu.download(), dtype=np.float32)
        except Exception as exc:
            _swallowed('roop/temporal_smart_skip.py:cuda_farneback', exc, 'CUDA Farneback failed; falling back')
            return None

    def _flow_dis_engine(self) -> Optional[Any]:
        engine = getattr(self._tls, "dis", None)
        if engine is None:
            try:
                engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            except (cv2.error, AttributeError, RuntimeError):
                engine = False
            self._tls.dis = engine
        return engine or None

    def compute_dense_flow(self, cur_img: np.ndarray, prev_img: np.ndarray) -> np.ndarray:
        """Calculate forward dense optical flow from cur_img to prev_img."""
        cur_gray = cur_img if cur_img.ndim == 2 else cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        prev_gray = prev_img if prev_img.ndim == 2 else cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        if self.flow_tier in ("auto", "nvof"):
            flow = self._flow_nvof(cur_gray, prev_gray)
            if flow is not None and np.isfinite(flow).all():
                self.last_flow_tier = "nvof"
                return flow

        if self.flow_tier in ("auto", "cuda_farneback"):
            flow = self._flow_cuda_farneback(cur_gray, prev_gray)
            if flow is not None and np.isfinite(flow).all():
                self.last_flow_tier = "cuda_farneback"
                return flow

        if self.flow_tier in ("auto", "dis"):
            dis = self._flow_dis_engine()
            if dis is not None:
                try:
                    flow = dis.calc(cur_gray, prev_gray, None)
                    if flow is not None and np.isfinite(flow).all():
                        self.last_flow_tier = "dis"
                        return np.asarray(flow, dtype=np.float32)
                except Exception as exc:
                    _swallowed('roop/temporal_smart_skip.py:dis', exc, 'DIS flow failed; falling back')

        self.last_flow_tier = "fallback_zero"
        return np.zeros(cur_gray.shape[:2] + (2,), dtype=np.float32)

    def warp(self, source: np.ndarray, flow: np.ndarray, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Warp source image using dense optical flow vector field."""
        h, w = (target_shape[:2] if target_shape is not None else source.shape[:2])
        fh, fw = flow.shape[:2]
        scaled_flow = flow
        if (fh, fw) != (h, w):
            scaled_flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR).copy()
            scaled_flow[..., 0] *= float(w) / float(fw)
            scaled_flow[..., 1] *= float(h) / float(fh)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + scaled_flow[..., 0]
        map_y = grid_y + scaled_flow[..., 1]
        return cv2.remap(source, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_DEFAULT_WARPER: Optional[DenseFlowWarper] = None


def get_dense_flow_warper() -> DenseFlowWarper:
    global _DEFAULT_WARPER
    if _DEFAULT_WARPER is None:
        _DEFAULT_WARPER = DenseFlowWarper()
    return _DEFAULT_WARPER


# ── Track patch cache for Stage 3 ──────────────────────────────────────────
_PATCH_CACHE: Dict[int, Dict[str, Any]] = {}
_PATCH_LOCK = threading.Lock()


def set_track_patch(track_id: int, patch: np.ndarray, aligned_img: np.ndarray, frame_idx: int, **kwargs) -> None:
    """Record final composited face patch for a track."""
    if track_id is None:
        return
    with _PATCH_LOCK:
        _PATCH_CACHE[int(track_id)] = {
            "patch": patch.copy(),
            "aligned_img": aligned_img.copy() if aligned_img is not None else None,
            "frame_idx": int(frame_idx),
            **kwargs,
        }


def get_track_patch(track_id: int, expected_frame_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Retrieve previous frame's composited face patch for a track.

    If expected_frame_idx is provided, returns None unless the cached patch matches that frame index.
    """
    if track_id is None:
        return None
    with _PATCH_LOCK:
        entry = _PATCH_CACHE.get(int(track_id))
        if entry is None:
            return None
        if expected_frame_idx is not None:
            entry_fidx = entry.get("frame_idx")
            if entry_fidx is None or int(entry_fidx) != int(expected_frame_idx):
                return None
        return entry


def clear_track_patches() -> None:
    """Clear all track patches between jobs."""
    with _PATCH_LOCK:
        _PATCH_CACHE.clear()


def mark_smart_skips_in_track(
    merged: Dict[int, Any],
    cuts: Optional[set[int]] = None,
    keyframes: Optional[set[int]] = None,
    max_consecutive: int = MAX_CONSECUTIVE_SKIPS,
    disp_threshold: float = DISPLACEMENT_THRESHOLD_PX,
    illum_threshold: float = ILLUMINATION_DELTA_THRESHOLD,
) -> int:
    """Evaluate consecutive observations of a track and mark REUSE_FLOW_WARP candidates.

    Returns the number of marked frames.
    """
    cuts = cuts or set()
    keyframes = keyframes or set()
    idxs = sorted(merged)
    if len(idxs) < 2:
        return 0

    marked_count = 0
    consecutive_skips = 0

    for idx_pos in range(len(idxs)):
        i = idxs[idx_pos]
        face = merged[i]

        if idx_pos == 0:
            face["reuse_flow_warp"] = False
            consecutive_skips = 0
            continue

        prev_i = idxs[idx_pos - 1]
        prev_face = merged[prev_i]

        # Must be strictly consecutive frames
        if i != prev_i + 1:
            face["reuse_flow_warp"] = False
            consecutive_skips = 0
            continue

        # Check scene cut or keyframe
        if i in cuts or i in keyframes:
            face["reuse_flow_warp"] = False
            consecutive_skips = 0
            continue

        # Check consecutive skip cap
        if consecutive_skips >= max_consecutive:
            face["reuse_flow_warp"] = False
            consecutive_skips = 0
            continue

        disp = compute_inner_landmark_displacement(
            face.get("kps"), prev_face.get("kps")
        )
        luma_curr = float(face.get("roi_luma", 0.5))
        luma_prev = float(prev_face.get("roi_luma", 0.5))
        delta_illum = compute_illumination_delta(luma_curr, luma_prev)

        if disp < disp_threshold and delta_illum < illum_threshold:
            face["reuse_flow_warp"] = True
            face["mode"] = "REUSE_FLOW_WARP"
            face["delta_eyes_mouth_px"] = disp
            face["roi_illum_delta"] = delta_illum
            consecutive_skips += 1
            marked_count += 1
        else:
            face["reuse_flow_warp"] = False
            consecutive_skips = 0

    return marked_count
