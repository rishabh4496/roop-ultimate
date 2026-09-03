"""Face swapper frame processor with foreground occlusion parsing, temporal smoothing,
eyelid blink retention (EAR), and teeth/inner-mouth passthrough.

Features:
1. Occlusion Parsing Pipeline:
   - Integrates an ONNX-based face occlusion / parsing model at 256x256 resolution.
   - Segments foreground occlusions (hands, hair, objects, food).
   - Generates effective blend mask: Mask_blend = Mask_face * (1.0 - Mask_occlusion).
2. Temporal Mask Smoothing (Optical Flow / EMA):
   - NVIDIA Optical Flow hardware when available, with CUDA Farneback and
     CPU DIS fallbacks.
   - Warps Mask_{t-1} to frame t and applies EMA:
     Mask_t = 0.85 * Mask_t + 0.15 * WarpedMask_{t-1}.
3. Eye Aspect Ratio (EAR) Blink Detection & Multi-Scale Eyelid Blending:
   - Evaluates EAR for both eyes: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||).
   - If EAR < 0.21 (blink/eye closing):
     - Extracts target actor's eyelid region.
     - Attenuates/bypasses restorer enhancement on eye bounding box.
     - Blends target's original eyelid back onto swap using multi-scale frequency decomposition.
4. Teeth & Inner Mouth Passthrough:
   - Extracts inner lip landmark polygon (points 61-68 in 68-pt scheme, 0-indexed 60-67).
   - If lip separation > 8 pixels:
     - Feathers inner-mouth contour by 3 pixels.
   - Passes through target's native teeth, tongue, and oral cavity.
5. Dermal Detail, Tone Mapping & Multi-band Compositing:
   - Separates a crop into guided-filter base/detail layers (r=4, eps=0.01).
   - Matches only skin-pixel LAB luminance distributions for low-light scenes.
   - Restores target and optional FaceSet V2 dermal residuals before a
     three-level Laplacian-pyramid composite.  No Poisson clone is used.
"""

import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime

try:
    import roop.globals
    from roop.typing import Face, Frame
    from roop.utilities import resolve_relative_path, conditional_download
except ImportError:
    try:
        import globals as roop_globals
        roop = type('RoopModule', (), {'globals': roop_globals})
    except ImportError:
        class _Globals:
            enable_occlusion_mask = True
            execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        roop = type('RoopModule', (), {'globals': _Globals()})
    Face = Any
    Frame = Any

    def resolve_relative_path(path: str) -> str:
        return os.path.abspath(path)

    def conditional_download(download_directory_path: str, urls: List[str]) -> None:
        pass

NAME = 'ROOP.FACE-SWAPPER'
PROCESSOR_NAME = 'face_swapper'

FACE_SWAPPER: Optional[onnxruntime.InferenceSession] = None
FACE_OCCLUDER: Optional[onnxruntime.InferenceSession] = None
THREAD_LOCK_SWAPPER = threading.Lock()
THREAD_LOCK_OCCLUDER = threading.Lock()

OCCLUSION_INPUT_SIZE = 256
DEFAULT_EMA_ALPHA = 0.85  # Mask_t = 0.85 * Mask_t + 0.15 * WarpedMask_{t-1}

# The values below intentionally keep the extra CPU work small: two 128 px
# guided decompositions and a three-level image pyramid are safe on both the
# 12 GB desktop and 6 GB laptop profiles, and introduce no model contexts.
DERMAL_GUIDED_RADIUS = 4
DERMAL_GUIDED_EPSILON = 0.01
DERMAL_TARGET_DETAIL_WEIGHT = 0.60
DERMAL_SWAP_DETAIL_WEIGHT = 0.25
DERMAL_PATCH_WEIGHT = 0.30

# Facial dynamics thresholds
EAR_BLINK_THRESHOLD = 0.21
MIN_LIP_SEPARATION_PX = 8.0
MOUTH_FEATHER_PX = 3

_OCCLUDER_URL = 'https://github.com/rishabh4496/roop-sam-weights/releases/download/v1/face_occluder.onnx'
_SWAPPER_URL = 'https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'

ARCFACE_DST_128 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


# ==============================================================================
# 1. Temporal Mask Smoothing (Optical Flow / EMA)
# ==============================================================================

class TemporalMaskSmoother:
    """Flow-warped Exponential Moving Average (EMA) over temporal mask sequences.

    Warping Mask_{t-1} to frame t along dense optical flow prevents edge jitter
    and boundary chatter around moving foreground objects (hands, food, microphones).

    Formula:
        Mask_t = 0.85 * Mask_t + 0.15 * WarpedMask_{t-1}

    Optical-flow priority is deliberately fixed: NVIDIA Optical Flow hardware
    (NVOF) first, CUDA Farneback second, then CPU DIS FAST.  NVOF uses the
    GPU's dedicated optical-flow engine; it does not schedule the flow solve
    on CUDA compute cores.  The single resulting vector field is reused to
    remap both the previous face mask and its seam boundary.
    """

    FLOW_SIZE = 128
    RESET_RESIDUAL = 0.50
    FLOW_TIERS = ('nvof', 'cuda_farneback', 'dis')

    def __init__(self, alpha: float = DEFAULT_EMA_ALPHA,
                 flow_tier: str = 'auto'):
        self.alpha = float(alpha)
        if flow_tier not in ('auto',) + self.FLOW_TIERS:
            raise ValueError("flow_tier must be 'auto', 'nvof', "
                             "'cuda_farneback', or 'dis'")
        # A forced tier is principally useful for deterministic self-tests.
        # Production uses auto and always retains the fallback hierarchy.
        self.flow_tier = flow_tier
        self.last_flow_tier: Optional[str] = None
        self._states: Dict[Any, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._tls = threading.local()

    @staticmethod
    def _cuda_available() -> bool:
        """Return whether this OpenCV build can address a CUDA device."""
        try:
            return (hasattr(cv2, 'cuda') and
                    cv2.cuda.getCudaEnabledDeviceCount() > 0)
        except (AttributeError, cv2.error):
            return False

    @staticmethod
    def _flow_is_valid(flow: Any) -> bool:
        return (isinstance(flow, np.ndarray) and flow.ndim == 3 and
                flow.shape[2] == 2 and flow.size > 0 and
                np.isfinite(flow).all())

    @staticmethod
    def _nvof_class() -> Optional[Any]:
        """Find either spelling used by CUDA-enabled OpenCV Python wheels."""
        cuda = getattr(cv2, 'cuda', None)
        return (getattr(cuda, 'NvidiaOpticalFlow_2_0', None) or
                getattr(cv2, 'cuda_NvidiaOpticalFlow_2_0', None))

    @staticmethod
    def _nvof_constant(nvof_class: Any, name: str, default: int) -> int:
        """Read a binding-specific NVOF enum without assuming its namespace."""
        cuda = getattr(cv2, 'cuda', None)
        return getattr(nvof_class, name,
                       getattr(cuda, name, getattr(
                           cv2, 'cuda_NvidiaOpticalFlow_2_0_' + name,
                           getattr(cv2, name, default))))

    def _nvof_engine(self, shape: Tuple[int, int]) -> Optional[Any]:
        """Create one NVOF context per worker and input resolution.

        NVOF contexts own device buffers, so sharing one between parallel frame
        workers risks both cross-track temporal hints and unsafe CUDA access.
        """
        if not self._cuda_available():
            return None
        nvof_class = self._nvof_class()
        if nvof_class is None:
            return None
        key = (int(shape[1]), int(shape[0]))  # OpenCV Size is (width, height)
        engines = getattr(self._tls, 'nvof', None)
        if engines is None:
            engines = {}
            self._tls.nvof = engines
        if key not in engines:
            try:
                perf = self._nvof_constant(
                    nvof_class, 'NV_OF_PERF_LEVEL_FAST', 20)
                output_grid = self._nvof_constant(
                    nvof_class, 'NV_OF_OUTPUT_VECTOR_GRID_SIZE_1', 1)
                hint_grid = self._nvof_constant(
                    nvof_class, 'NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED', 0)
                create = nvof_class.create
                try:
                    engines[key] = create(key, perf, output_grid, hint_grid,
                                          False, False, False, 0)
                except TypeError:
                    # Some Python bindings expose only the default arguments.
                    engines[key] = create(key)
            except (AttributeError, cv2.error, TypeError):
                engines[key] = False
        return engines[key] or None

    @staticmethod
    def _gpu_mat() -> Any:
        gpu_mat_class = getattr(cv2, 'cuda_GpuMat', None)
        if gpu_mat_class is None:
            gpu_mat_class = getattr(getattr(cv2, 'cuda', None), 'GpuMat', None)
        return gpu_mat_class()

    def _flow_nvof(self, cur_small: np.ndarray,
                   prev_small: np.ndarray) -> Optional[np.ndarray]:
        """Calculate backward vectors on the dedicated NVIDIA OFA engine."""
        engine = self._nvof_engine(cur_small.shape[:2])
        if engine is None:
            return None
        try:
            cur_gpu = self._gpu_mat()
            prev_gpu = self._gpu_mat()
            cur_gpu.upload(cur_small)
            prev_gpu.upload(prev_small)
            raw_flow = engine.calc(cur_gpu, prev_gpu, None)

            # NVOF returns fixed-point CV_16FC2 vectors.  Convert on-device
            # before the one required download for CPU-side mask remapping.
            float_gpu = self._gpu_mat()
            converted = engine.convertToFloat(raw_flow, float_gpu)
            if converted is None:
                converted = float_gpu
            flow = converted.download()
            return np.asarray(flow, dtype=np.float32)
        except (AttributeError, cv2.error, TypeError):
            return None

    def _cuda_farneback_engine(self) -> Optional[Any]:
        if not self._cuda_available():
            return None
        engine = getattr(self._tls, 'cuda_farneback', None)
        if engine is None:
            try:
                cuda = getattr(cv2, 'cuda', None)
                klass = (getattr(cv2, 'cuda_FarnebackOpticalFlow', None) or
                         getattr(cuda, 'FarnebackOpticalFlow', None))
                engine = klass.create(3, 0.5, False, 15, 3, 5, 1.2, 0)
            except (AttributeError, cv2.error, TypeError):
                engine = False
            self._tls.cuda_farneback = engine
        return engine or None

    def _flow_cuda_farneback(self, cur_small: np.ndarray,
                             prev_small: np.ndarray) -> Optional[np.ndarray]:
        """CUDA-core fallback for builds without the Optical Flow SDK module."""
        if not self._cuda_available():
            return None
        try:
            cur_gpu = self._gpu_mat()
            prev_gpu = self._gpu_mat()
            cur_gpu.upload(cur_small)
            prev_gpu.upload(prev_small)

            # Prefer the direct binding where a wheel exposes it.  Other
            # CUDA-enabled OpenCV builds expose the equivalent algorithm class
            # below instead.
            direct = getattr(getattr(cv2, 'cuda', None),
                             'calcOpticalFlowFarneback', None)
            if direct is not None:
                flow_gpu = direct(cur_gpu, prev_gpu, None, 0.5, 3, 15, 3,
                                  5, 1.2, 0)
                return np.asarray(flow_gpu.download(), dtype=np.float32)

            engine = self._cuda_farneback_engine()
            if engine is None:
                return None
            flow_gpu = engine.calc(cur_gpu, prev_gpu, None)
            return np.asarray(flow_gpu.download(), dtype=np.float32)
        except (AttributeError, cv2.error, TypeError):
            return None

    def _flow_dis_engine(self) -> Optional[Any]:
        """Return a thread-local CPU DIS FAST instance."""
        engine = getattr(self._tls, 'dis', None)
        if engine is None:
            try:
                engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            except (AttributeError, cv2.error):
                engine = False
            self._tls.dis = engine
        return engine or None

    def _dense_flow(self, cur_small: np.ndarray, prev_small: np.ndarray) -> np.ndarray:
        """Compute current-to-previous vectors using the configured hierarchy."""
        candidates = (
            ('nvof', self._flow_nvof),
            ('cuda_farneback', self._flow_cuda_farneback),
        )
        for tier, calculate in candidates:
            if self.flow_tier not in ('auto', tier):
                continue
            flow = calculate(cur_small, prev_small)
            if self._flow_is_valid(flow):
                self.last_flow_tier = tier
                return flow

        if self.flow_tier in ('auto', 'dis'):
            engine = self._flow_dis_engine()
            if engine is not None:
                try:
                    flow = engine.calc(cur_small, prev_small, None)
                    if self._flow_is_valid(flow):
                        self.last_flow_tier = 'dis'
                        return flow
                except cv2.error:
                    pass

        # A build without DIS cannot estimate tier-3 motion.  Keeping the
        # mask fixed is safer than introducing an undeclared fourth tier that
        # could smear a cut or consume unexpected CPU time.
        self.last_flow_tier = 'dis_unavailable'
        return np.zeros(cur_small.shape + (2,), dtype=np.float32)

    def calculate_motion_vectors(self, current_crop: np.ndarray,
                                 previous_crop: np.ndarray) -> np.ndarray:
        """Return finite backward motion vectors for two crop-sized frames.

        The vectors are solved at ``FLOW_SIZE``.  This public, side-effect-free
        method exists for diagnostics and the NVOF pipeline self-test.
        """
        cur_small = self._to_gray_small(current_crop, self.FLOW_SIZE)
        prev_small = self._to_gray_small(previous_crop, self.FLOW_SIZE)
        return self._dense_flow(cur_small, prev_small)

    def _warp(self, prev_mask: np.ndarray, flow: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Warp previous mask using dense optical flow field."""
        h, w = int(shape[0]), int(shape[1])
        fh, fw = flow.shape[:2]
        scaled_flow = flow
        if (fh, fw) != (h, w):
            scaled_flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR).copy()
            scaled_flow[..., 0] *= float(w) / float(fw)
            scaled_flow[..., 1] *= float(h) / float(fh)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        map_x = grid_x + scaled_flow[..., 0]
        map_y = grid_y + scaled_flow[..., 1]
        return cv2.remap(prev_mask, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _seam_boundary(mask: np.ndarray) -> np.ndarray:
        """Represent the soft face-mask rim that must follow temporal motion."""
        source = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.morphologyEx(source, cv2.MORPH_GRADIENT, kernel)

    @staticmethod
    def _to_gray_small(crop: np.ndarray, size: int = 128) -> np.ndarray:
        img = np.asarray(crop)
        if img.ndim == 3:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    def smooth(self, mask: np.ndarray, crop: np.ndarray, track_id: Any = 0,
               alpha: Optional[float] = None) -> np.ndarray:
        if mask is None:
            return mask
        cur_mask = np.asarray(mask, dtype=np.float32)
        if cur_mask.ndim != 2:
            return mask

        eff_alpha = self.alpha if alpha is None else float(alpha)
        gray_small = self._to_gray_small(crop, self.FLOW_SIZE)

        with self._lock:
            state = self._states.get(track_id)
            if state is not None:
                prev_mask = state['mask']
                prev_gray = state['gray']
                if prev_mask.shape != cur_mask.shape:
                    prev_mask = cv2.resize(
                        prev_mask, (cur_mask.shape[1], cur_mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR)

                flow = self._dense_flow(gray_small, prev_gray)
                warped = self._warp(prev_mask, flow, cur_mask.shape)
                # Reuse the same hardware/compute flow: boundary motion is
                # never estimated separately, so NVOF adds no second pass.
                warped_seam = self._warp(state['seam'], flow, cur_mask.shape)

                # Reset on large flow residual (scene cut or sudden teleportation)
                residual = float(np.mean(np.abs(warped - cur_mask)))
                if residual <= self.RESET_RESIDUAL:
                    out_mask = np.clip(eff_alpha * cur_mask + (1.0 - eff_alpha) * warped, 0.0, 1.0)
                    out_seam = np.clip(eff_alpha * self._seam_boundary(cur_mask) +
                                       (1.0 - eff_alpha) * warped_seam, 0.0, 1.0)
                else:
                    out_mask = cur_mask
                    out_seam = self._seam_boundary(cur_mask)
            else:
                out_mask = cur_mask
                out_seam = self._seam_boundary(cur_mask)

            self._states[track_id] = {
                'mask': out_mask.copy(),
                'gray': gray_small,
                'seam': out_seam.copy(),
            }
            return out_mask

    def reset(self, track_id: Optional[Any] = None) -> None:
        with self._lock:
            if track_id is None:
                self._states.clear()
            else:
                self._states.pop(track_id, None)


_GLOBAL_SMOOTHER = TemporalMaskSmoother(alpha=DEFAULT_EMA_ALPHA)


# ==============================================================================
# 2. Eye Aspect Ratio (EAR) Blink Detection & Multi-Scale Eyelid Blending
# ==============================================================================

def calculate_ear(eye_points: np.ndarray) -> float:
    """Calculate Eye Aspect Ratio (EAR) from 6 landmark points.

    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    """
    pts = np.asarray(eye_points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 6:
        return 0.30
    p1, p2, p3, p4, p5, p6 = pts[:6]
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))
    if h < 1e-6:
        return 0.30
    return float((v1 + v2) / (2.0 * h))


def compute_eye_aspect_ratios(landmarks_68: np.ndarray) -> Tuple[float, float, float]:
    """Compute (left_ear, right_ear, mean_ear) from standard 68-point landmarks.

    0-indexed scheme:
        Left eye (viewer-left): points 36..41
        Right eye (viewer-right): points 42..47
    """
    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return 0.30, 0.30, 0.30
    left_ear = calculate_ear(pts[36:42])
    right_ear = calculate_ear(pts[42:48])
    mean_ear = float(0.5 * (left_ear + right_ear))
    return left_ear, right_ear, mean_ear


def build_blink_eyelid_mask(
    landmarks_68: np.ndarray,
    shape: Tuple[int, int],
    ear_threshold: float = EAR_BLINK_THRESHOLD
) -> np.ndarray:
    """Extract smooth eyelid mask for eyes whose EAR is below the blink threshold (< 0.21)."""
    h, w = int(shape[0]), int(shape[1])
    eyelid_mask = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return eyelid_mask

    left_ear, right_ear, _ = compute_eye_aspect_ratios(pts)

    eye_indices = [
        (36, 42, left_ear),
        (42, 48, right_ear),
    ]

    for start_idx, end_idx, ear in eye_indices:
        if ear < ear_threshold:
            eye_pts = pts[start_idx:end_idx]
            # Center of eye
            cx = float(np.mean(eye_pts[:, 0]))
            cy = float(np.mean(eye_pts[:, 1]))
            # Width and height of eye
            ew = float(np.linalg.norm(eye_pts[0] - eye_pts[3]))
            eh = float(max(np.linalg.norm(eye_pts[1] - eye_pts[5]),
                           np.linalg.norm(eye_pts[2] - eye_pts[4])))

            # Eyelid region: extend upper lid upwards to capture the folding lid
            axis_x = max(6, int(ew * 0.75))
            axis_y = max(4, int(max(eh * 1.5, ew * 0.35)))

            # Blink weight smoothly transitions from 0 at threshold down to 1.0 when fully shut
            closure_weight = np.clip((ear_threshold - ear) / (ear_threshold - 0.12), 0.0, 1.0)

            sub_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(sub_mask, (int(round(cx)), int(round(cy - axis_y * 0.15))),
                        (axis_x, axis_y), 0, 0, 360, float(closure_weight), -1)

            # Feather boundary softly
            k = max(3, int(round(axis_x * 0.35)) | 1)
            sub_mask = cv2.GaussianBlur(sub_mask, (k, k), 0)
            eyelid_mask = np.maximum(eyelid_mask, sub_mask)

    return np.clip(eyelid_mask, 0.0, 1.0)


def blend_eyelid_multiscale(
    target_crop: np.ndarray,
    swapped_crop: np.ndarray,
    eyelid_mask: np.ndarray
) -> np.ndarray:
    """Multi-scale frequency blend to composite original eyelid onto swapped crop seamlessly."""
    if eyelid_mask is None or not np.any(eyelid_mask > 1e-4):
        return swapped_crop.copy()

    t_f = target_crop.astype(np.float32)
    s_f = swapped_crop.astype(np.float32)

    # Multi-scale mask decomposition (fine texture vs coarse illumination)
    m_fine = cv2.GaussianBlur(eyelid_mask, (5, 5), 1.5)[..., np.newaxis]
    m_coarse = cv2.GaussianBlur(eyelid_mask, (15, 15), 5.0)[..., np.newaxis]

    # Frequency split
    t_low = cv2.GaussianBlur(t_f, (15, 15), 5.0)
    s_low = cv2.GaussianBlur(s_f, (15, 15), 5.0)
    t_high = t_f - t_low
    s_high = s_f - s_low

    # Blend low and high frequency components independently
    blended_low = s_low * (1.0 - m_coarse) + t_low * m_coarse
    blended_high = s_high * (1.0 - m_fine) + t_high * m_fine

    blended = blended_low + blended_high
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def get_closed_eyes_attenuation(
    landmarks_68: np.ndarray,
    shape: Tuple[int, int],
    ear_threshold: float = EAR_BLINK_THRESHOLD
) -> Tuple[bool, np.ndarray]:
    """Generate attenuation mask over closed eye bounding boxes to bypass GPEN/restorer hallucination."""
    h, w = int(shape[0]), int(shape[1])
    attenuation_mask = np.ones((h, w), dtype=np.float32)
    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return False, attenuation_mask

    left_ear, right_ear, _ = compute_eye_aspect_ratios(pts)
    is_blinking = (left_ear < ear_threshold or right_ear < ear_threshold)

    for start_idx, end_idx, ear in ((36, 42, left_ear), (42, 48, right_ear)):
        if ear < ear_threshold:
            eye_pts = pts[start_idx:end_idx]
            x1 = max(0, int(np.min(eye_pts[:, 0]) - 8))
            y1 = max(0, int(np.min(eye_pts[:, 1]) - 12))
            x2 = min(w, int(np.max(eye_pts[:, 0]) + 8))
            y2 = min(h, int(np.max(eye_pts[:, 1]) + 8))
            # Attenuate enhancement within closed eye bounding box to 0.0
            attenuation_mask[y1:y2, x1:x2] = 0.0

    if is_blinking:
        attenuation_mask = cv2.GaussianBlur(attenuation_mask, (7, 7), 2.0)
    return is_blinking, attenuation_mask


# ==============================================================================
# 3. Teeth & Inner Mouth Passthrough
# ==============================================================================

def extract_inner_mouth_mask(
    landmarks_68: np.ndarray,
    shape: Tuple[int, int],
    min_separation: float = MIN_LIP_SEPARATION_PX,
    feather_px: int = MOUTH_FEATHER_PX
) -> np.ndarray:
    """Extract inner-mouth contour mask (points 61-68 / 0-indexed 60-67) when mouth is open (> 8px).

    The contour is softly feathered by 3 pixels to allow clean teeth, tongue, and oral cavity
    passthrough without static or blurry mouth artifacts.
    """
    h, w = int(shape[0]), int(shape[1])
    mouth_mask = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return mouth_mask

    # Inner lips: indices 60 to 67
    inner_lips = pts[60:68]

    # Vertical separation: between upper center inner lip (62) and lower center inner lip (66)
    vertical_sep = float(np.linalg.norm(pts[62] - pts[66]))

    if vertical_sep > min_separation:
        poly = np.asarray(inner_lips, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mouth_mask, [poly], 1.0)

        # 3-pixel contour feathering
        k = max(3, int(feather_px * 2 + 1))
        mouth_mask = cv2.GaussianBlur(mouth_mask, (k, k), 1.5)

    return np.clip(mouth_mask, 0.0, 1.0)


def blend_inner_mouth_passthrough(
    target_crop: np.ndarray,
    composite_crop: np.ndarray,
    mouth_mask: np.ndarray
) -> np.ndarray:
    """Passthrough target actor's native teeth, tongue, and oral cavity into composite crop."""
    if mouth_mask is None or not np.any(mouth_mask > 1e-4):
        return composite_crop.copy()

    t_f = target_crop.astype(np.float32)
    c_f = composite_crop.astype(np.float32)
    m = mouth_mask[..., np.newaxis] if mouth_mask.ndim == 2 else mouth_mask

    blended = c_f * (1.0 - m) + t_f * m
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def apply_facial_dynamics(
    target_crop: np.ndarray,
    swapped_crop: np.ndarray,
    landmarks_68: Optional[np.ndarray],
    blend_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Unified handler for eyelid blinks (EAR < 0.21) and inner-mouth retention (sep > 8px)."""
    meta = {
        'is_blinking': False,
        'left_ear': 0.30,
        'right_ear': 0.30,
        'mouth_open': False,
        'lip_separation': 0.0,
        'eyelid_mask': None,
        'mouth_mask': None,
        'attenuation_mask': None,
    }

    if landmarks_68 is None:
        return swapped_crop.copy(), meta

    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return swapped_crop.copy(), meta

    shape = target_crop.shape[:2]

    # 1. EAR Blink Detection & Eyelid blending
    l_ear, r_ear, _ = compute_eye_aspect_ratios(pts)
    meta['left_ear'] = l_ear
    meta['right_ear'] = r_ear
    is_blink, att_mask = get_closed_eyes_attenuation(pts, shape, ear_threshold=EAR_BLINK_THRESHOLD)
    meta['is_blinking'] = is_blink
    meta['attenuation_mask'] = att_mask

    current_result = swapped_crop
    if is_blink:
        eyelid_mask = build_blink_eyelid_mask(pts, shape, ear_threshold=EAR_BLINK_THRESHOLD)
        meta['eyelid_mask'] = eyelid_mask
        current_result = blend_eyelid_multiscale(target_crop, current_result, eyelid_mask)

    # 2. Teeth & Inner Mouth Passthrough
    sep = float(np.linalg.norm(pts[62] - pts[66]))
    meta['lip_separation'] = sep
    if sep > MIN_LIP_SEPARATION_PX:
        meta['mouth_open'] = True
        mouth_mask = extract_inner_mouth_mask(pts, shape, min_separation=MIN_LIP_SEPARATION_PX, feather_px=MOUTH_FEATHER_PX)
        meta['mouth_mask'] = mouth_mask
        current_result = blend_inner_mouth_passthrough(target_crop, current_result, mouth_mask)

    return current_result, meta


# ==============================================================================
# 4. Occlusion Parsing Pipeline
# ==============================================================================

def _find_model_file(filename: str) -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', filename),
        os.path.join(os.path.dirname(__file__), '..', '..', 'models', filename),
        os.path.join(os.path.dirname(__file__), '..', 'models', filename),
        resolve_relative_path(os.path.join('../models', filename)),
        resolve_relative_path(os.path.join('models', filename)),
        os.path.abspath(os.path.join('app', 'models', filename)),
        os.path.abspath(os.path.join('models', filename)),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return candidates[0]


def get_face_swapper() -> Optional[onnxruntime.InferenceSession]:
    global FACE_SWAPPER
    with THREAD_LOCK_SWAPPER:
        if FACE_SWAPPER is None:
            model_path = _find_model_file('inswapper_128.onnx')
            if not os.path.isfile(model_path):
                model_dir = os.path.dirname(model_path)
                os.makedirs(model_dir, exist_ok=True)
                conditional_download(model_dir, [_SWAPPER_URL])
            if os.path.isfile(model_path):
                providers = getattr(roop.globals, 'execution_providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                FACE_SWAPPER = onnxruntime.InferenceSession(model_path, providers=providers)
    return FACE_SWAPPER


def get_face_occluder() -> Optional[onnxruntime.InferenceSession]:
    global FACE_OCCLUDER
    with THREAD_LOCK_OCCLUDER:
        if FACE_OCCLUDER is None:
            model_path = None
            for name in ('face_occluder.onnx', 'xseg.onnx', 'resnet18.onnx'):
                path = _find_model_file(name)
                if os.path.isfile(path):
                    model_path = path
                    break
            if model_path is None:
                model_path = _find_model_file('face_occluder.onnx')
                model_dir = os.path.dirname(model_path)
                os.makedirs(model_dir, exist_ok=True)
                conditional_download(model_dir, [_OCCLUDER_URL])

            if os.path.isfile(model_path):
                providers = getattr(roop.globals, 'execution_providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                FACE_OCCLUDER = onnxruntime.InferenceSession(model_path, providers=providers)
    return FACE_OCCLUDER


def _heuristic_occlusion_mask(crop_frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Fallback heuristic detection for foreign occlusions (hands, black bars, objects)."""
    h, w = crop_frame.shape[:2]
    crop_256 = cv2.resize(crop_frame, (OCCLUSION_INPUT_SIZE, OCCLUSION_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop_256, cv2.COLOR_BGR2GRAY) if crop_256.ndim == 3 else crop_256

    dark = (gray < 35).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    if (h, w) != (OCCLUSION_INPUT_SIZE, OCCLUSION_INPUT_SIZE):
        dark = cv2.resize(dark, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(dark, 0.0, 1.0).astype(np.float32)


def compute_occlusion_mask(
    crop_frame: np.ndarray,
    occluder_session: Optional[onnxruntime.InferenceSession] = None,
    face_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Segment foreground occlusions (hands, hair, objects, food) at 256x256 resolution.

    Performance Guardrail:
        Strictly executes at 256x256 resolution to keep GPU inference under 4ms.

    Returns:
        Mask_occlusion in [0.0, 1.0], where 1.0 = occluded, 0.0 = clear face.
    """
    if not getattr(roop.globals, 'enable_occlusion_mask', True):
        return np.zeros(crop_frame.shape[:2], dtype=np.float32)

    h, w = crop_frame.shape[:2]
    crop_256 = cv2.resize(crop_frame, (OCCLUSION_INPUT_SIZE, OCCLUSION_INPUT_SIZE), interpolation=cv2.INTER_AREA)

    sess = occluder_session
    if sess is None:
        try:
            sess = get_face_occluder()
        except Exception:
            sess = None

    if sess is not None:
        try:
            inp_meta = sess.get_inputs()[0]
            inp_shape = inp_meta.shape
            inp_name = inp_meta.name

            temp_frame = crop_256.astype(np.float32) / 255.0
            if len(inp_shape) == 4 and inp_shape[1] == 3:
                temp_frame = np.transpose(temp_frame, (2, 0, 1))
            temp_frame = np.expand_dims(temp_frame, axis=0)

            with THREAD_LOCK_OCCLUDER:
                ort_outs = sess.run(None, {inp_name: temp_frame})

            raw_out = ort_outs[0]
            if raw_out.ndim == 4:
                raw_out = raw_out[0]
            if raw_out.ndim == 3 and raw_out.shape[-1] == 1:
                raw_out = raw_out[..., 0]
            elif raw_out.ndim == 3 and raw_out.shape[0] == 1:
                raw_out = raw_out[0]

            vis_face = np.clip(raw_out / 0.70, 0.0, 1.0)
            occ_256 = np.clip(1.0 - vis_face, 0.0, 1.0)

            if (h, w) != (OCCLUSION_INPUT_SIZE, OCCLUSION_INPUT_SIZE):
                occlusion_mask = cv2.resize(occ_256, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                occlusion_mask = occ_256
            return np.clip(occlusion_mask, 0.0, 1.0).astype(np.float32)
        except Exception:
            pass

    return _heuristic_occlusion_mask(crop_frame, face_mask)


def apply_occlusion_blend(
    face_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    enable_occlusion: Optional[bool] = None
) -> np.ndarray:
    """Generate effective blend mask: Mask_blend = Mask_face * (1.0 - Mask_occlusion)."""
    if enable_occlusion is None:
        enable_occlusion = getattr(roop.globals, 'enable_occlusion_mask', True)

    f_mask = np.asarray(face_mask, dtype=np.float32)
    if not enable_occlusion or occlusion_mask is None:
        return np.clip(f_mask, 0.0, 1.0)

    occ = np.asarray(occlusion_mask, dtype=np.float32)
    if occ.shape != f_mask.shape:
        occ = cv2.resize(occ, (f_mask.shape[1], f_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    blend = f_mask * (1.0 - occ)
    return np.clip(blend, 0.0, 1.0)


def smooth_temporal_mask(
    mask: np.ndarray,
    crop: np.ndarray,
    track_id: Any = 0,
    alpha: float = DEFAULT_EMA_ALPHA
) -> np.ndarray:
    """Warp Mask_{t-1} to frame t and apply Exponential Moving Average:

        Mask_t = 0.8 * Mask_t + 0.2 * WarpedMask_{t-1}
    """
    return _GLOBAL_SMOOTHER.smooth(mask, crop, track_id=track_id, alpha=alpha)


def clear_temporal_state(track_id: Optional[Any] = None) -> None:
    _GLOBAL_SMOOTHER.reset(track_id)


def guided_filter_decompose(
    image: np.ndarray,
    radius: int = DERMAL_GUIDED_RADIUS,
    epsilon: float = DERMAL_GUIDED_EPSILON,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(base, detail)`` using a grayscale-guided filter.

    The computation follows ``detail = image - GuidedFilter(image, r=4,
    eps=0.01)`` with image values normalized to [0, 1].  OpenCV's optional
    ximgproc module is deliberately not required: this box-filter form is
    portable across the project's supported installations.
    """
    src = np.asarray(image)
    if src.ndim != 3 or src.shape[2] < 3 or src.size == 0:
        raise ValueError("guided_filter_decompose expects a non-empty BGR image")
    value = np.clip(src[..., :3], 0, 255).astype(np.float32) / 255.0
    guide = cv2.cvtColor((value * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    guide = guide.astype(np.float32) / 255.0
    r = max(1, int(radius))
    window = (2 * r + 1, 2 * r + 1)
    mean_guide = cv2.boxFilter(guide, cv2.CV_32F, window, normalize=True,
                                borderType=cv2.BORDER_REFLECT)
    var_guide = (cv2.boxFilter(guide * guide, cv2.CV_32F, window,
                               normalize=True, borderType=cv2.BORDER_REFLECT)
                 - mean_guide * mean_guide)
    base = np.empty_like(value)
    denom = np.maximum(var_guide + max(1e-6, float(epsilon)), 1e-6)
    for channel in range(3):
        p = value[..., channel]
        mean_p = cv2.boxFilter(p, cv2.CV_32F, window, normalize=True,
                               borderType=cv2.BORDER_REFLECT)
        covariance = (cv2.boxFilter(guide * p, cv2.CV_32F, window,
                                    normalize=True, borderType=cv2.BORDER_REFLECT)
                      - mean_guide * mean_p)
        a = covariance / denom
        b = mean_p - a * mean_guide
        mean_a = cv2.boxFilter(a, cv2.CV_32F, window, normalize=True,
                               borderType=cv2.BORDER_REFLECT)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, window, normalize=True,
                               borderType=cv2.BORDER_REFLECT)
        base[..., channel] = mean_a * guide + mean_b
    base *= 255.0
    return base.astype(np.float32), (value * 255.0 - base).astype(np.float32)


def derive_skin_mask(target_crop: np.ndarray,
                     face_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate a soft skin-only support mask, excluding background and hair.

    YCrCb is used instead of brightness thresholds so it remains usable in
    dark scenes.  If a very dim/colour-neutral frame leaves too few candidate
    pixels, the existing face matte is the conservative fallback rather than
    sampling the surrounding background.
    """
    target = np.asarray(target_crop)
    if target.ndim != 3 or target.shape[2] < 3:
        raise ValueError("derive_skin_mask expects a BGR image")
    h, w = target.shape[:2]
    if face_mask is None:
        support = create_static_face_mask((h, w))
    else:
        support = np.asarray(face_mask, dtype=np.float32)
        if support.ndim == 3:
            support = support[..., 0]
        if support.shape != (h, w):
            support = cv2.resize(support, (w, h), interpolation=cv2.INTER_LINEAR)
        support = np.clip(support, 0.0, 1.0)
    ycrcb = cv2.cvtColor(np.clip(target[..., :3], 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    candidate = ((cr >= 120) & (cr <= 180) & (cb >= 70) & (cb <= 145)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    candidate = cv2.GaussianBlur(candidate.astype(np.float32), (0, 0), 1.1)
    skin = np.clip(candidate, 0.0, 1.0) * support
    if float(np.count_nonzero(skin > 0.2)) < max(32.0, 0.02 * float(np.count_nonzero(support > 0.2))):
        return support.astype(np.float32)
    return np.clip(skin, 0.0, 1.0).astype(np.float32)


def skin_masked_lab_tone_map(
    swap_crop: np.ndarray,
    target_crop: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    """Map swap LAB L* CDF to the target's CDF using skin pixels only.

    Hair and the dark background never enter either histogram.  A bounded
    skin-only a*/b* mean offset follows the L* match: it keeps a dim target's
    chroma from making an otherwise correct L* conversion look too bright when
    returned to BGR, without allowing a broad colour cast.
    """
    swap = np.asarray(swap_crop)
    target = np.asarray(target_crop)
    if swap.ndim != 3 or target.ndim != 3 or swap.shape[2] < 3 or target.shape[2] < 3:
        raise ValueError("skin_masked_lab_tone_map expects BGR images")
    if target.shape[:2] != swap.shape[:2]:
        target = cv2.resize(target, (swap.shape[1], swap.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = np.asarray(skin_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != swap.shape[:2]:
        mask = cv2.resize(mask, (swap.shape[1], swap.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0)
    sample = mask > 0.35
    if int(np.count_nonzero(sample)) < 32:
        return swap.copy()

    s_lab = cv2.cvtColor(np.clip(swap[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    t_lab = cv2.cvtColor(np.clip(target[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    source_l = s_lab[..., 0]
    target_l = t_lab[..., 0]
    source_hist = np.bincount(source_l[sample], minlength=256).astype(np.float64)
    target_hist = np.bincount(target_l[sample], minlength=256).astype(np.float64)
    if source_hist.sum() <= 0.0 or target_hist.sum() <= 0.0:
        return swap.copy()
    source_cdf = np.cumsum(source_hist) / source_hist.sum()
    target_cdf = np.cumsum(target_hist) / target_hist.sum()
    # A nearly flat synthetic/restored swap has no usable rank ordering.  A
    # literal CDF inverse maps every one of those pixels to the target's
    # brightest sample, visibly lifting a night face.  Preserve the target
    # profile's centre in this degenerate case; normal images use the full CDF.
    if float(np.std(source_l[sample])) < 1.0:
        mapped_l = np.clip(source_l.astype(np.float32) +
                           (float(target_l[sample].mean()) - float(source_l[sample].mean())),
                           0.0, 255.0).astype(np.uint8)
    else:
        lookup = np.searchsorted(target_cdf, source_cdf, side='left').clip(0, 255).astype(np.uint8)
        mapped_l = lookup[source_l]
    out_lab = s_lab.copy()
    # Feathering the operation with the same skin mask avoids a hard cheek or
    # hairline change while retaining exact CDF mapping in the skin interior.
    out_lab[..., 0] = np.clip(source_l.astype(np.float32) +
                              (mapped_l.astype(np.float32) - source_l.astype(np.float32)) * mask,
                              0.0, 255.0).astype(np.uint8)
    chroma_delta = np.clip(t_lab[..., 1:3][sample].mean(axis=0) -
                           s_lab[..., 1:3][sample].mean(axis=0), -32.0, 32.0)
    out_lab[..., 1:3] = np.clip(s_lab[..., 1:3].astype(np.float32) +
                                chroma_delta.reshape(1, 1, 2) * mask[..., None],
                                0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def resolve_dermal_patch(source: Any) -> Optional[Dict[str, Any]]:
    """Find an optional V2 dermal patch on a face or its loaded FaceSet."""
    if source is None:
        return None
    candidates = [
        _field(source, 'dermal_patch'),
        (_field(source, 'faceset_metadata') or {}).get('dermal_patch')
        if isinstance(_field(source, 'faceset_metadata'), dict) else None,
    ]
    for owner_name in ('faceset', 'source_faceset', 'face_set'):
        owner = _field(source, owner_name)
        if owner is not None:
            candidates.append(_field(owner, 'dermal_patch'))
            metadata = _field(owner, 'faceset_metadata')
            if isinstance(metadata, dict):
                candidates.append(metadata.get('dermal_patch'))
    for patch in candidates:
        if isinstance(patch, dict) and isinstance(patch.get('texture') or patch, dict):
            return patch
    return None


def warp_dermal_patch(
    dermal_patch: Optional[Dict[str, Any]],
    output_shape: Tuple[int, int],
    target_landmarks: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Warp a FaceSet V2 residual from its UV anchors into target crop space."""
    if not isinstance(dermal_patch, dict):
        return None, None
    try:
        from roop.identity_detail import decode_detail
        decoded = decode_detail(dermal_patch.get('texture') or dermal_patch)
        anchors = dermal_patch.get('uv_anchors') or {}
        source_uv = np.asarray(anchors.get('uv'), dtype=np.float32)
        if decoded is None or source_uv.shape != (5, 2) or not np.isfinite(source_uv).all():
            return None, None
        h, w = int(output_shape[0]), int(output_shape[1])
        dh, dw = decoded['residual'].shape
        source_points = source_uv * np.array([dw - 1, dh - 1], dtype=np.float32)
        if target_landmarks is None:
            destination = source_uv * np.array([w - 1, h - 1], dtype=np.float32)
        else:
            destination = np.asarray(target_landmarks, dtype=np.float32).reshape(-1, 2)[:5]
            if destination.shape != (5, 2) or not np.isfinite(destination).all():
                return None, None
            if float(np.max(destination)) <= 1.2:
                destination = destination * np.array([w - 1, h - 1], dtype=np.float32)
        transform, _ = cv2.estimateAffinePartial2D(source_points, destination, method=cv2.LMEDS)
        if transform is None:
            return None, None
        residual = cv2.warpAffine(decoded['residual'], transform, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        confidence = decoded['confidence'] * decoded['mask']
        weight = cv2.warpAffine(confidence, transform, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        return residual.astype(np.float32), np.clip(weight, 0.0, 1.0).astype(np.float32)
    except (ImportError, TypeError, ValueError, cv2.error):
        return None, None


def restore_dermal_and_tone(
    target_crop: np.ndarray,
    enhanced_swap_crop: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    dermal_patch: Optional[Dict[str, Any]] = None,
    target_landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Tone-map skin and restore photograph-derived high-frequency detail.

    Both crops are decomposed before recomposition.  The target's residual
    keeps pores/moles photographed in the destination lighting; an available
    V2 patch contributes only its confidence-weighted source residual after
    UV-landmark warping.
    """
    target = np.asarray(target_crop)
    swap = np.asarray(enhanced_swap_crop)
    if target.ndim != 3 or swap.ndim != 3 or target.shape[2] < 3 or swap.shape[2] < 3:
        raise ValueError("restore_dermal_and_tone expects BGR images")
    if target.shape[:2] != swap.shape[:2]:
        target = cv2.resize(target, (swap.shape[1], swap.shape[0]), interpolation=cv2.INTER_LINEAR)
    skin = derive_skin_mask(target, face_mask)
    toned = skin_masked_lab_tone_map(swap, target, skin)
    target_base, target_detail = guided_filter_decompose(target)
    swap_base, swap_detail = guided_filter_decompose(toned)
    del target_base  # Detail is the target contribution; its base stays untouched.
    smooth_restored = toned.astype(np.float32) * (1.0 - skin[..., None]) + swap_base * skin[..., None]
    detail = (DERMAL_TARGET_DETAIL_WEIGHT * target_detail +
              DERMAL_SWAP_DETAIL_WEIGHT * swap_detail)
    patch_residual, patch_weight = warp_dermal_patch(dermal_patch, target.shape[:2], target_landmarks)
    if patch_residual is not None and patch_weight is not None:
        detail += (DERMAL_PATCH_WEIGHT * patch_residual * patch_weight)[..., None]
    # Keep detail strictly inside the skin support.  Smoothly cap it so a noisy
    # source cannot turn a mole into a clipped black/white pixel.
    detail = 24.0 * np.tanh(detail / 24.0)
    out = smooth_restored + detail * skin[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def laplacian_pyramid_blend(
    target_buffer: np.ndarray,
    swap_buffer: np.ndarray,
    blend_mask: np.ndarray,
    levels: int = 3,
) -> np.ndarray:
    """Blend target/swap buffers with a bounded multi-level Laplacian pyramid."""
    target = np.asarray(target_buffer, dtype=np.float32)
    swap = np.asarray(swap_buffer, dtype=np.float32)
    mask = np.asarray(blend_mask, dtype=np.float32)
    if target.ndim != 3 or target.shape[2] < 3:
        raise ValueError("laplacian_pyramid_blend expects a BGR target")
    if swap.shape != target.shape:
        swap = cv2.resize(swap, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != target.shape[:2]:
        mask = cv2.resize(mask, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0)
    depth = max(1, min(int(levels), 3))
    target_gaussian, swap_gaussian, mask_gaussian = [target], [swap], [mask]
    for _ in range(1, depth):
        if min(target_gaussian[-1].shape[:2]) < 2:
            break
        target_gaussian.append(cv2.pyrDown(target_gaussian[-1]))
        swap_gaussian.append(cv2.pyrDown(swap_gaussian[-1]))
        mask_gaussian.append(cv2.pyrDown(mask_gaussian[-1]))
    target_laplacian, swap_laplacian = [], []
    for index in range(len(target_gaussian) - 1):
        size = (target_gaussian[index].shape[1], target_gaussian[index].shape[0])
        target_laplacian.append(target_gaussian[index] - cv2.pyrUp(target_gaussian[index + 1], dstsize=size))
        swap_laplacian.append(swap_gaussian[index] - cv2.pyrUp(swap_gaussian[index + 1], dstsize=size))
    blended = (target_gaussian[-1] * (1.0 - mask_gaussian[-1][..., None]) +
               swap_gaussian[-1] * mask_gaussian[-1][..., None])
    for index in range(len(target_laplacian) - 1, -1, -1):
        size = (target_laplacian[index].shape[1], target_laplacian[index].shape[0])
        blended = cv2.pyrUp(blended, dstsize=size)
        m = mask_gaussian[index][..., None]
        blended += target_laplacian[index] * (1.0 - m) + swap_laplacian[index] * m
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def blend_swap_buffer(
    target_buffer: np.ndarray,
    swap_buffer: np.ndarray,
    blend_mask: np.ndarray
) -> np.ndarray:
    """Composite with the three-level Laplacian blend used instead of Poisson."""
    return laplacian_pyramid_blend(target_buffer, swap_buffer, blend_mask, levels=3)


def create_static_face_mask(shape: Tuple[int, int], radius_factor: float = 0.40) -> np.ndarray:
    """Generate default elliptical face mask with soft boundary feathering."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.float32)
    center = (int(w / 2), int(h / 2))
    axes = (int(w * radius_factor), int(h * radius_factor * 1.25))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    k = max(3, int(min(h, w) * 0.10) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


# ==============================================================================
# 5. Core Swap Pipeline
# ==============================================================================

def swap_face(
    source_face: Face,
    target_face: Face,
    temp_frame: Frame,
    track_id: Any = 0
) -> Frame:
    """Core face swapping pipeline with occlusion subtraction, temporal smoothing,
    eyelid blink retention, and teeth/inner-mouth passthrough."""
    target_frame = temp_frame.copy()
    kps = getattr(target_face, 'kps', None)
    if kps is None and isinstance(target_face, dict):
        kps = target_face.get('kps')

    crop_size = 128
    if kps is not None and len(kps) == 5:
        M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, dtype=np.float32), ARCFACE_DST_128)
        if M is None:
            return temp_frame
        crop_frame = cv2.warpAffine(temp_frame, M, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
    else:
        bbox = getattr(target_face, 'bbox', None)
        if bbox is None and isinstance(target_face, dict):
            bbox = target_face.get('bbox')
        if bbox is None:
            return temp_frame
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(temp_frame.shape[1], x2), min(temp_frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return temp_frame
        crop_frame = temp_frame[y1:y2, x1:x2]
        M = None

    # Step 1: Face Swapper ONNX inference
    swapper = get_face_swapper()
    embedding = getattr(source_face, 'embedding', None)
    if embedding is None and isinstance(source_face, dict):
        embedding = source_face.get('embedding')

    if embedding is not None and swapper is not None:
        source_emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        crop_blob = crop_frame.astype(np.float32) / 255.0
        crop_blob = crop_blob[..., ::-1]  # BGR to RGB
        crop_blob = np.transpose(crop_blob, (2, 0, 1))
        crop_blob = np.expand_dims(crop_blob, axis=0)

        with THREAD_LOCK_SWAPPER:
            swapped_blob = swapper.run(None, {
                'target': crop_blob,
                'source': source_emb
            })[0]

        swapped_crop = np.squeeze(swapped_blob, axis=0)
        swapped_crop = np.transpose(swapped_crop, (1, 2, 0))
        swapped_crop = swapped_crop[..., ::-1]  # RGB to BGR
        swapped_crop = np.clip(swapped_crop * 255.0, 0, 255).astype(np.uint8)
    else:
        swapped_crop = crop_frame.copy()

    # Step 2: Facial dynamics (Blinks & Inner Mouth retention)
    landmarks_68 = getattr(target_face, 'landmark_3d_68', None)
    if landmarks_68 is None and hasattr(target_face, 'landmarks'):
        landmarks_68 = getattr(target_face, 'landmarks')
    if landmarks_68 is None and isinstance(target_face, dict):
        landmarks_68 = target_face.get('landmark_3d_68') or target_face.get('landmarks_68') or target_face.get('landmarks')

    if landmarks_68 is not None:
        # Transform landmarks into crop space
        pts_full = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
        if M is not None:
            pts_crop = cv2.transform(pts_full.reshape(-1, 1, 2), M).reshape(-1, 2)
        else:
            pts_crop = pts_full - np.array([x1, y1], dtype=np.float32)
        swapped_crop, _ = apply_facial_dynamics(crop_frame, swapped_crop, pts_crop)

    # Step 3: Base face mask
    face_mask = create_static_face_mask(crop_frame.shape[:2])

    # Step 3a: Skin-only LAB tone mapping plus guided-filter dermal recovery.
    # A loaded V2 FaceSet can expose its patch directly on the chosen source
    # face or through its FaceSet owner.  V1/missing metadata is intentionally
    # a no-op for the source contribution while target micro-detail remains.
    crop_kps = None
    if kps is not None and len(kps) == 5 and M is not None:
        crop_kps = cv2.transform(np.asarray(kps, dtype=np.float32).reshape(-1, 1, 2), M).reshape(-1, 2)
    swapped_crop = restore_dermal_and_tone(
        crop_frame, swapped_crop, face_mask=face_mask,
        dermal_patch=resolve_dermal_patch(source_face), target_landmarks=crop_kps)

    # Step 4: Occlusion Parsing Pipeline at 256x256
    occlusion_mask = compute_occlusion_mask(crop_frame, face_mask=face_mask)
    blend_mask = apply_occlusion_blend(face_mask, occlusion_mask)

    # Step 5: Temporal Mask Smoothing (Optical Flow / EMA)
    smoothed_mask = smooth_temporal_mask(blend_mask, crop_frame, track_id=track_id)

    # Step 6: Composite and paste back.  For an aligned crop the pyramid runs
    # against a full-frame source that is explicitly filled with the target
    # outside the matte.  Feeding its black warp border into a coarse pyramid
    # level would create precisely the gray halo this replaces Poisson to avoid.
    if M is not None:
        inv_M = cv2.invertAffineTransform(M)
        h, w = temp_frame.shape[:2]
        warped_crop = cv2.warpAffine(swapped_crop, inv_M, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask = cv2.warpAffine(smoothed_mask, inv_M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        pyramid_source = temp_frame.copy()
        valid = warped_mask > 1e-4
        pyramid_source[valid] = warped_crop[valid]
        return laplacian_pyramid_blend(temp_frame, pyramid_source, warped_mask, levels=3)
    else:
        blended_crop = blend_swap_buffer(crop_frame, swapped_crop, smoothed_mask)
        target_frame[y1:y2, x1:x2] = cv2.resize(blended_crop, (x2 - x1, y2 - y1))
        return target_frame


def pre_check() -> bool:
    return True


def pre_start() -> bool:
    return True


def post_process() -> None:
    clear_temporal_state()


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    return swap_face(source_face, reference_face, temp_frame)


def process_frames(source_path: str, temp_frame_paths: List[str], update_progress: Optional[Any] = None) -> None:
    clear_temporal_state()
    for frame_path in temp_frame_paths:
        frame = cv2.imread(frame_path)
        cv2.imwrite(frame_path, frame)
        if update_progress:
            update_progress()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    clear_temporal_state()
    target_frame = cv2.imread(target_path)
    cv2.imwrite(output_path, target_frame)


def process_video(source_path: str, frame_paths: List[str]) -> None:
    process_frames(source_path, frame_paths)
