"""Content-aware scene cut detection and temporal buffer management.

Provides:
1. Content-aware scene boundary detection using:
   - PySceneDetect (when scenedetect package is available)
   - GPU-accelerated or vectorized multi-channel HSV/RGB histogram-difference thresholds
   - Adaptive rolling-median ratio gating with refractory guards
2. Unified temporal buffer flushing across optical flow, landmark tracking,
   high-frequency texture stabilizers, and occlusion mask caches to prevent
   warp smearing and phantom landmark drift across cut transitions.
"""

from collections import deque
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed

# Optional GPU support via PyTorch CUDA
_TORCH_AVAILABLE = False
_TORCH_CUDA_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
    _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except Exception as _exc:
    _swallowed("scene_detector.py:import_torch", _exc, "torch unavailable")
    torch = None

# Optional PySceneDetect support
_PYSCENEDETECT_AVAILABLE = False
try:
    import scenedetect
    from scenedetect import ContentDetector, SceneManager, open_video
    _PYSCENEDETECT_AVAILABLE = True
except Exception as _exc:
    _swallowed("scene_detector.py:import_pyscenedetect", _exc, "pyscenedetect unavailable")
    scenedetect = None


def is_pyscenedetect_available() -> bool:
    """Return True if PySceneDetect is installed and importable."""
    return _PYSCENEDETECT_AVAILABLE


def flush_pipeline_temporal_buffers(mgr: Optional[Any] = None) -> Dict[str, bool]:
    """Flush optical flow and landmark tracking buffers across all pipeline stages.

    Called at detected scene boundaries to prevent warp smearing and landmark
    carry-over across cut transitions.
    """
    flushed = {
        'face_swapper_temporal': False,
        'landmark_smoother': False,
        'hf_stabilizer': False,
        'target_appearance': False,
        'occlusion_mask': False,
    }

    # 1. Flush face_swapper's global smoother, motion blur, and injected maskers
    try:
        from roop.processors.frame.face_swapper import clear_temporal_state
        clear_temporal_state()
        flushed['face_swapper_temporal'] = True
    except Exception as _exc:
        _swallowed("scene_detector.py:flush_face_swapper", _exc, "fallback continued")

    # 2. Flush ProcessMgr smoothers and stabilizers if provided
    if mgr is not None:
        landmark_smoother = getattr(mgr, '_landmark_smoother', None)
        if landmark_smoother is not None and hasattr(landmark_smoother, 'reset'):
            try:
                landmark_smoother.reset()
                flushed['landmark_smoother'] = True
            except Exception as _exc:
                _swallowed("scene_detector.py:flush_landmark_smoother", _exc, "fallback continued")

        hf_stabilizer = getattr(mgr, '_hf_stabilizer', None)
        if hf_stabilizer is not None and hasattr(hf_stabilizer, 'reset'):
            try:
                hf_stabilizer.reset()
                flushed['hf_stabilizer'] = True
            except Exception as _exc:
                _swallowed("scene_detector.py:flush_hf_stabilizer", _exc, "fallback continued")

        target_app = getattr(mgr, '_target_appearance', None)
        if target_app is not None and hasattr(target_app, 'reset'):
            try:
                target_app.reset()
                flushed['target_appearance'] = True
            except Exception as _exc:
                _swallowed("scene_detector.py:flush_target_appearance", _exc, "fallback continued")

    return flushed


class ContentAwareSceneDetector:
    """Robust content-aware scene cut detector.

    Combines:
    - 3-channel color histogram difference (HSV/RGB) for content changes
    - GPU acceleration when PyTorch CUDA is available
    - Adaptive rolling-median thresholding with refractory suppression to avoid
      false cut cascades on motion or camera shake
    - Optional PySceneDetect integration
    """

    DEFAULT_CUT_FLOOR = 0.045
    DEFAULT_CUT_RATIO = 4.5
    DEFAULT_WINDOW = 45
    DEFAULT_REFRACTORY = 3
    DEFAULT_COLD_START = 0.15
    DEFAULT_MIN_BASELINE = 6

    def __init__(
        self,
        cut_floor: Optional[float] = None,
        cut_ratio: Optional[float] = None,
        window: int = DEFAULT_WINDOW,
        refractory_frames: int = DEFAULT_REFRACTORY,
        use_gpu: bool = True,
        use_pyscenedetect: bool = False,
    ):
        self.cut_floor = float(cut_floor if cut_floor is not None
                               else float(os.environ.get('ROOP_CUT_FLOOR', self.DEFAULT_CUT_FLOOR)))
        self.cut_ratio = float(cut_ratio if cut_ratio is not None
                               else float(os.environ.get('ROOP_CUT_RATIO', self.DEFAULT_CUT_RATIO)))
        self.window = int(window)
        self.refractory_frames = int(refractory_frames)
        self.use_gpu = bool(use_gpu and _TORCH_CUDA_AVAILABLE)
        self.use_pyscenedetect = bool(use_pyscenedetect and _PYSCENEDETECT_AVAILABLE)

        self._diffs: deque = deque(maxlen=self.window)
        self._since_cut: int = 1 << 24
        self._last_hist: Optional[np.ndarray] = None
        self._last_hist_gpu: Optional[Any] = None
        self.scene_cuts: List[int] = []

    def reset(self) -> None:
        """Reset running detector state."""
        self._diffs.clear()
        self._since_cut = 1 << 24
        self._last_hist = None
        self._last_hist_gpu = None
        self.scene_cuts.clear()

    @staticmethod
    def compute_hsv_hist_cpu(frame: np.ndarray, bins: Tuple[int, int, int] = (16, 8, 8)) -> np.ndarray:
        """Compute normalized 3D HSV histogram on CPU."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return np.zeros(bins, dtype=np.float32)

        # Downsample for high-speed calculation while preserving color semantics
        h, w = frame.shape[:2]
        stride = max(1, int(max(h, w) / 128))
        sample = frame[::stride, ::stride]

        if sample.shape[2] == 4:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGRA2BGR)

        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.astype(np.float32)

    def compute_hist_gpu(self, frame: np.ndarray) -> Optional[Any]:
        """Compute histogram representation on GPU if PyTorch CUDA is enabled."""
        if not self.use_gpu or not _TORCH_CUDA_AVAILABLE or torch is None:
            return None
        try:
            h, w = frame.shape[:2]
            stride = max(1, int(max(h, w) / 128))
            small = frame[::stride, ::stride, :3]
            t_frame = torch.from_numpy(small).cuda().float() / 255.0
            # Compute channel-wise marginals and mean color distribution
            # 32 bins per channel on GPU
            hists = []
            for c in range(3):
                ch = t_frame[:, :, c].contiguous()
                hists.append(torch.histc(ch, bins=32, min=0.0, max=1.0))
            hist_tensor = torch.cat(hists)
            hist_norm = hist_tensor / (torch.norm(hist_tensor, p=1) + 1e-7)
            return hist_norm
        except Exception as _exc:
            _swallowed("scene_detector.py:compute_hist_gpu", _exc, "fallback to cpu")
            return None

    def calculate_difference(self, frame: np.ndarray) -> float:
        """Calculate content-aware histogram difference against previous frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return 0.0

        # Try GPU path first if active
        if self.use_gpu:
            hist_gpu = self.compute_hist_gpu(frame)
            if hist_gpu is not None:
                if self._last_hist_gpu is None:
                    self._last_hist_gpu = hist_gpu
                    return 0.0
                # L1 distance on GPU
                diff_gpu = torch.sum(torch.abs(hist_gpu - self._last_hist_gpu)).item() * 0.5
                self._last_hist_gpu = hist_gpu
                return float(diff_gpu)

        # CPU Vectorized HSV Histogram Bhattacharyya / L1 distance
        hist = self.compute_hsv_hist_cpu(frame)
        if self._last_hist is None:
            self._last_hist = hist
            return 0.0

        # Bhattacharyya distance in [0, 1]
        try:
            bhat = float(cv2.compareHist(self._last_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            if not np.isfinite(bhat):
                bhat = float(np.mean(np.abs(self._last_hist - hist)))
        except Exception as _exc:
            _swallowed("scene_detector.py:compare_hist", _exc, "fallback continued")
            bhat = float(np.mean(np.abs(self._last_hist - hist)))

        self._last_hist = hist
        return float(bhat)

    def is_cut(self, diff: float) -> bool:
        """Adaptive cut evaluation: absolute floor + rolling-median ratio + refractory guard."""
        if diff < self.cut_floor:
            return False

        if self._since_cut < self.refractory_frames:
            return False

        if len(self._diffs) < self.DEFAULT_MIN_BASELINE:
            # Cold-start period before baseline is formed
            return diff >= self.DEFAULT_COLD_START

        baseline = float(np.median(np.fromiter(self._diffs, dtype=np.float64)))
        threshold = max(self.cut_floor, self.cut_ratio * max(baseline, 1e-4))
        return diff >= threshold

    def observe_frame(self, frame: np.ndarray, frame_idx: int) -> bool:
        """Process one frame observation. Returns True if frame_idx is a cut boundary."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False

        diff = self.calculate_difference(frame)

        if self._since_cut == 0 and len(self.scene_cuts) > 0 and self.scene_cuts[-1] == frame_idx - 1:
            # Immediate next frame after a cut: register difference to build new shot baseline
            self._since_cut = 1
            self._diffs.append(diff)
            return False

        cut = self.is_cut(diff)
        if cut:
            self.scene_cuts.append(frame_idx)
            self._diffs.clear()
            self._since_cut = 0
        else:
            self._since_cut += 1
            self._diffs.append(diff)

        return cut

    def scan_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        time_budget: Optional[float] = None,
        on_progress: Optional[Any] = None,
    ) -> Set[int]:
        """Scan a video file and return a set of frame indices marking scene cuts."""
        if not video_path or not os.path.isfile(video_path):
            return set()

        # If PySceneDetect is explicitly requested and installed, use it
        if self.use_pyscenedetect and _PYSCENEDETECT_AVAILABLE:
            try:
                cuts = self._scan_pyscenedetect(video_path)
                return cuts
            except Exception as _exc:
                _swallowed("scene_detector.py:scan_pyscenedetect", _exc, "fallback to histogram")

        cuts: Set[int] = set()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return cuts

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit = min(total, max_frames) if max_frames else total

        idx = 0
        import time
        t0 = time.time()
        self.reset()

        while True:
            if limit and idx >= limit:
                break
            if time_budget and (time.time() - t0) > time_budget:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if self.observe_frame(frame, idx):
                cuts.add(idx)

            idx += 1
            if on_progress and idx % 25 == 0:
                on_progress(idx, limit or total)

        cap.release()
        return cuts

    def _scan_pyscenedetect(self, video_path: str) -> Set[int]:
        """Scan using PySceneDetect ContentDetector."""
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        cuts: Set[int] = set()
        for scene in scene_list:
            start_frame = scene[0].get_frames()
            if start_frame > 0:
                cuts.add(start_frame)
        return cuts
