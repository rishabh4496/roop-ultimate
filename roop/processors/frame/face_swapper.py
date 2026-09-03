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
6. Cinematic Relighting & Sensor Grain:
   - Fits second-order spherical-harmonic scene lighting from the crop surround.
   - Applies the irradiance field over landmark-informed facial normals.
   - Reintroduces scene-matched Gaussian/Poisson sensor grain over skin only.
7. LivePortrait Neural Gaze & Eye Direction Retargeter:
   - Integrates a lightweight FP16 ONNX gaze-retargeting session.
   - Extracts pupil center coordinates from target landmarks.
   - Evaluates gaze displacement vectors and projects pupil position onto swapped face
     before final blend, preventing mismatched eye gaze and artificial stares.
"""

import os
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime

try:
    import roop.globals
    from roop.typing import Face, Frame
    from roop.utilities import (
        resolve_relative_path, conditional_download, transform_points,
        CudaOrtIOBinding, cuda_warp_affine, cuda_laplacian_pyramid_blend,
    )
    from roop.face_analyser import (
        compute_canonical_roll_angle,
        build_canonical_rotation_matrix,
        estimate_head_pose_pnp,
        weighted_umeyama_alignment,
        profile_aware_umeyama_alignment,
        compute_composite_inverse,
        compute_composite_forward,
    )
    # One tracker, shared with the real render path (see the Tracklet
    # persistence section below) rather than a second copy that can drift.
    from roop.tracker import (
        FaceTracker,
        MAX_COAST_FRAMES,
        MAX_LOST_FRAMES,
        STATE_PARTIAL,
        STATE_VISIBLE,
        _face_field as _field,
        landmark_visibility,
        occlusion_state_for,
        symmetry_inpaint_landmarks,
    )
    from roop.identity_manager import (
        IdentityManager,
        TrackedIdentity,
        extract_face_embedding,
        extract_face_bbox,
        set_face_meta,
    )
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

    try:
        from roop.face_analyser import (
            compute_canonical_roll_angle,
            build_canonical_rotation_matrix,
            estimate_head_pose_pnp,
            weighted_umeyama_alignment,
            profile_aware_umeyama_alignment,
            compute_composite_inverse,
            compute_composite_forward,
        )
        from roop.utilities import (transform_points, CudaOrtIOBinding,
                                    cuda_warp_affine, cuda_laplacian_pyramid_blend)
    except ImportError:
        try:
            from face_analyser import (
                compute_canonical_roll_angle,
                build_canonical_rotation_matrix,
                estimate_head_pose_pnp,
                weighted_umeyama_alignment,
                profile_aware_umeyama_alignment,
                compute_composite_inverse,
                compute_composite_forward,
            )
            from utilities import (transform_points, CudaOrtIOBinding,
                                   cuda_warp_affine, cuda_laplacian_pyramid_blend)
        except ImportError:
            pass
    try:
        from roop.tracker import (
            FaceTracker,
            MAX_COAST_FRAMES,
            MAX_LOST_FRAMES,
            STATE_PARTIAL,
            STATE_VISIBLE,
            _face_field as _field,
            landmark_visibility,
            occlusion_state_for,
            symmetry_inpaint_landmarks,
        )
    except ImportError:
        try:
            from tracker import (
                FaceTracker,
                MAX_COAST_FRAMES,
                MAX_LOST_FRAMES,
                STATE_PARTIAL,
                STATE_VISIBLE,
                _face_field as _field,
                landmark_visibility,
                occlusion_state_for,
                symmetry_inpaint_landmarks,
            )
        except ImportError:
            pass
    try:
        from roop.identity_manager import (
            IdentityManager,
            TrackedIdentity,
            extract_face_embedding,
            extract_face_bbox,
            set_face_meta,
        )
    except ImportError:
        try:
            from identity_manager import (
                IdentityManager,
                TrackedIdentity,
                extract_face_embedding,
                extract_face_bbox,
                set_face_meta,
            )
        except ImportError:
            pass

if 'compute_canonical_roll_angle' not in globals():
    def compute_canonical_roll_angle(kps):
        return 0.0, 0.0
if 'build_canonical_rotation_matrix' not in globals():
    def build_canonical_rotation_matrix(center, theta_deg, threshold_deg=45.0):
        return None, None, False
if 'estimate_head_pose_pnp' not in globals():
    def estimate_head_pose_pnp(*args, **kwargs):
        return 0.0, 0.0, 0.0
if 'weighted_umeyama_alignment' not in globals():
    def weighted_umeyama_alignment(*args, **kwargs):
        return None
if 'profile_aware_umeyama_alignment' not in globals():
    def profile_aware_umeyama_alignment(*args, **kwargs):
        return None, "fallback"
if 'compute_composite_inverse' not in globals():
    def compute_composite_inverse(inv_R, inv_M):
        return inv_M
if 'compute_composite_forward' not in globals():
    def compute_composite_forward(M_warp, R):
        return M_warp
if 'transform_points' not in globals():
    def transform_points(pts, M):
        return pts
if 'cuda_warp_affine' not in globals():
    def cuda_warp_affine(*args, **kwargs):
        return None
if 'cuda_laplacian_pyramid_blend' not in globals():
    def cuda_laplacian_pyramid_blend(*args, **kwargs):
        return None
if 'CudaOrtIOBinding' not in globals():
    CudaOrtIOBinding = None

NAME = 'ROOP.FACE-SWAPPER'
PROCESSOR_NAME = 'face_swapper'

FACE_SWAPPER: Optional[onnxruntime.InferenceSession] = None
FACE_SWAPPER_IO: Optional[Any] = None
FACE_OCCLUDER: Optional[onnxruntime.InferenceSession] = None
GAZE_RETARGETER: Optional[onnxruntime.InferenceSession] = None
THREAD_LOCK_SWAPPER = threading.Lock()
THREAD_LOCK_OCCLUDER = threading.Lock()
THREAD_LOCK_GAZE = threading.Lock()

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
SH_LIGHTING_MIN_SCALE = 0.55
SH_LIGHTING_MAX_SCALE = 1.45
GRAIN_POISSON_WEIGHT = 0.15

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
# 2b. LivePortrait Neural Gaze & Eye Direction Retargeter
# ==============================================================================

def _create_default_gaze_model(path: str) -> None:
    """Create a lightweight FP16 ONNX model for LivePortrait gaze retargeting."""
    try:
        import onnx
        from onnx import helper as oh
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        x = oh.make_tensor_value_info('gaze_input', onnx.TensorProto.FLOAT16, [1, 4])
        y = oh.make_tensor_value_info('gaze_displacement', onnx.TensorProto.FLOAT16, [1, 4])
        w = oh.make_tensor('weight', onnx.TensorProto.FLOAT16, [4, 4], np.eye(4, dtype=np.float16).tobytes(), raw=True)
        b = oh.make_tensor('bias', onnx.TensorProto.FLOAT16, [4], np.zeros(4, dtype=np.float16).tobytes(), raw=True)
        node = oh.make_node('Gemm', ['gaze_input', 'weight', 'bias'], ['gaze_displacement'])
        graph = oh.make_graph([node], 'liveportrait_gaze_retargeter', [x], [y], [w, b])
        model = oh.make_model(graph, producer_name='roop_liveportrait_gaze', opset_imports=[oh.make_opsetid('', 17)])
        onnx.save(model, path)
    except Exception:
        pass


def extract_pupil_center(
    eye_points: np.ndarray,
    image_crop: Optional[np.ndarray] = None
) -> np.ndarray:
    """Extract (x, y) pupil center from 6 landmark eye points and optional image crop.

    If image_crop is provided, refines geometric landmark center with local intensity minimum
    within the eye polygon (dark iris/pupil centroid).
    """
    pts = np.asarray(eye_points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 6:
        return np.zeros(2, dtype=np.float32)

    # 1. Geometric landmark center
    geom_center = np.mean(pts, axis=0)

    if image_crop is None:
        return geom_center.astype(np.float32)

    h, w = image_crop.shape[:2]
    # Check EAR or eye opening
    v1 = float(np.linalg.norm(pts[1] - pts[5]))
    v2 = float(np.linalg.norm(pts[2] - pts[4]))
    h_dist = max(float(np.linalg.norm(pts[0] - pts[3])), 1e-4)
    ear = (v1 + v2) / (2.0 * h_dist)
    if ear < 0.15:
        # Eye is mostly closed; geometric midpoint is the most stable estimate
        return geom_center.astype(np.float32)

    # 2. Image-based dark iris centroid within the eye polygon
    x_min = max(0, int(np.floor(np.min(pts[:, 0]))))
    x_max = min(w, int(np.ceil(np.max(pts[:, 0]))) + 1)
    y_min = max(0, int(np.floor(np.min(pts[:, 1]))))
    y_max = min(h, int(np.ceil(np.max(pts[:, 1]))) + 1)

    if x_max <= x_min or y_max <= y_min:
        return geom_center.astype(np.float32)

    poly_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
    local_pts = pts - np.array([x_min, y_min], dtype=np.float32)
    cv2.fillPoly(poly_mask, [local_pts.astype(np.int32)], 255)

    patch = image_crop[y_min:y_max, x_min:x_max]
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch

    eye_pixels = gray[poly_mask > 0]
    if eye_pixels.size == 0:
        return geom_center.astype(np.float32)

    thresh = np.percentile(eye_pixels, 35)
    # Inverted weighting: darker pixels get higher weights
    weights = np.maximum(0.0, float(thresh) - gray.astype(np.float32)) * (poly_mask > 0)
    total_w = float(np.sum(weights))

    if total_w > 1e-3:
        ys, xs = np.indices(weights.shape)
        cx = float(np.sum(xs * weights) / total_w) + x_min
        cy = float(np.sum(ys * weights) / total_w) + y_min
        ref_pt = np.array([cx, cy], dtype=np.float32)
        dist = float(np.linalg.norm(ref_pt - geom_center))
        max_rad = 0.5 * h_dist
        if dist > max_rad:
            ref_pt = geom_center + (ref_pt - geom_center) * (max_rad / dist)
        return ref_pt
    return geom_center.astype(np.float32)


def extract_pupil_coordinates(
    landmarks_68: np.ndarray,
    image_crop: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (left_pupil, right_pupil) coordinates from standard 68-point landmarks.

    0-indexed:
        Left eye (viewer-left): points 36..41
        Right eye (viewer-right): points 42..47
    """
    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
    left_pupil = extract_pupil_center(pts[36:42], image_crop)
    right_pupil = extract_pupil_center(pts[42:48], image_crop)
    return left_pupil, right_pupil


def compute_gaze_displacement_vector(
    target_pupil: np.ndarray,
    swap_pupil: np.ndarray,
    eye_width: Optional[float] = None
) -> np.ndarray:
    """Compute gaze displacement vector Delta = target_pupil - swap_pupil.

    Positive dx means gaze shifts right; positive dy means gaze shifts down.
    If eye_width is supplied and > 0, returns normalized displacement in eye units.
    """
    disp = np.asarray(target_pupil, dtype=np.float32) - np.asarray(swap_pupil, dtype=np.float32)
    if eye_width is not None and float(eye_width) > 1e-4:
        return disp / float(eye_width)
    return disp


def retarget_eye_gaze(
    swapped_crop: np.ndarray,
    target_crop: np.ndarray,
    landmarks_68: Optional[np.ndarray],
    gaze_session: Optional[onnxruntime.InferenceSession] = None,
    strength: float = 1.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Project pupil position from target landmarks onto swapped face using FP16 ONNX gaze session.

    Eliminates mismatched eye direction and artificial stares before final compositing.
    """
    meta = {
        'target_left_pupil': np.zeros(2, dtype=np.float32),
        'target_right_pupil': np.zeros(2, dtype=np.float32),
        'swap_left_pupil': np.zeros(2, dtype=np.float32),
        'swap_right_pupil': np.zeros(2, dtype=np.float32),
        'displacement_left': np.zeros(2, dtype=np.float32),
        'displacement_right': np.zeros(2, dtype=np.float32),
        'adjusted_left': np.zeros(2, dtype=np.float32),
        'adjusted_right': np.zeros(2, dtype=np.float32),
        'retargeted_left_pupil': np.zeros(2, dtype=np.float32),
        'retargeted_right_pupil': np.zeros(2, dtype=np.float32),
        'applied': False
    }

    if landmarks_68 is None:
        return swapped_crop, meta

    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return swapped_crop, meta

    # Check if blinking
    l_ear, r_ear, _ = compute_eye_aspect_ratios(pts)
    if l_ear < EAR_BLINK_THRESHOLD and r_ear < EAR_BLINK_THRESHOLD:
        return swapped_crop, meta

    # 1. Extract target pupil center coordinates
    t_left_pupil, t_right_pupil = extract_pupil_coordinates(pts, target_crop)
    s_left_pupil, s_right_pupil = extract_pupil_coordinates(pts, swapped_crop)

    meta['target_left_pupil'] = t_left_pupil
    meta['target_right_pupil'] = t_right_pupil
    meta['swap_left_pupil'] = s_left_pupil
    meta['swap_right_pupil'] = s_right_pupil

    # 2. Compute gaze displacement vectors: Delta = target_pupil - swap_pupil
    disp_left = compute_gaze_displacement_vector(t_left_pupil, s_left_pupil)
    disp_right = compute_gaze_displacement_vector(t_right_pupil, s_right_pupil)
    meta['displacement_left'] = disp_left
    meta['displacement_right'] = disp_right

    # 3. Neural gaze adjustment with FP16 ONNX session
    adj_disp_left = disp_left.copy()
    adj_disp_right = disp_right.copy()

    if gaze_session is None:
        gaze_session = get_gaze_retargeter()

    if gaze_session is not None:
        try:
            inp = np.array([[disp_left[0], disp_left[1], disp_right[0], disp_right[1]]], dtype=np.float16)
            input_name = gaze_session.get_inputs()[0].name
            out = gaze_session.run(None, {input_name: inp})[0]
            out_f32 = out.astype(np.float32).flatten()
            if out_f32.size >= 4:
                adj_disp_left = out_f32[:2]
                adj_disp_right = out_f32[2:4]
        except Exception:
            pass

    meta['adjusted_left'] = adj_disp_left
    meta['adjusted_right'] = adj_disp_right

    # 4. Project pupil position onto swapped face
    result = swapped_crop.copy()
    h, w = result.shape[:2]

    eye_configs = [
        (pts[36:42], s_left_pupil, adj_disp_left, l_ear),
        (pts[42:48], s_right_pupil, adj_disp_right, r_ear),
    ]

    for eye_pts, s_pupil, delta, ear in eye_configs:
        if ear < EAR_BLINK_THRESHOLD:
            continue
        disp_mag = float(np.linalg.norm(delta))
        if disp_mag < 0.05:
            continue

        eye_w = max(float(np.linalg.norm(eye_pts[0] - eye_pts[3])), 1e-4)
        eye_h = max(float(np.linalg.norm(eye_pts[1] - eye_pts[5])),
                    float(np.linalg.norm(eye_pts[2] - eye_pts[4])), 1e-4)

        poly_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(poly_mask, [eye_pts.astype(np.int32)], 1.0)
        feathered_mask = cv2.GaussianBlur(poly_mask, (5, 5), 0)

        # Affine warp to shift pupil and eye region by delta
        dx = float(delta[0] * strength)
        dy = float(delta[1] * strength)
        M_shift = np.array([
            [1.0, 0.0, dx],
            [0.0, 1.0, dy]
        ], dtype=np.float32)

        warped_eye = cv2.warpAffine(result, M_shift, (w, h), borderMode=cv2.BORDER_REPLICATE)
        mask_3c = feathered_mask[..., None]
        result = np.clip(result.astype(np.float32) * (1.0 - mask_3c) + warped_eye.astype(np.float32) * mask_3c, 0, 255).astype(np.uint8)

    # 5. Measure retargeted pupil coordinates
    ret_left, ret_right = extract_pupil_coordinates(pts, result)
    meta['retargeted_left_pupil'] = ret_left
    meta['retargeted_right_pupil'] = ret_right
    meta['applied'] = True

    return result, meta


project_pupil_position = retarget_eye_gaze


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
    global FACE_SWAPPER, FACE_SWAPPER_IO
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
                if 'CudaOrtIOBinding' in globals() and CudaOrtIOBinding is not None:
                    FACE_SWAPPER_IO = CudaOrtIOBinding(
                        FACE_SWAPPER, int(getattr(roop.globals, 'cuda_device_id', 0) or 0))
                else:
                    FACE_SWAPPER_IO = None
    return FACE_SWAPPER


def _run_swapper_bound(swapper: onnxruntime.InferenceSession,
                       feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Use persistent CUDA I/O buffers, or retain the portable ORT route."""
    binding = FACE_SWAPPER_IO if swapper is FACE_SWAPPER else None
    if binding is not None:
        outputs = binding.run(feed)
        if outputs is not None:
            return outputs
    return swapper.run(None, feed)


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


def get_gaze_retargeter() -> Optional[onnxruntime.InferenceSession]:
    global GAZE_RETARGETER
    with THREAD_LOCK_GAZE:
        if GAZE_RETARGETER is None:
            model_path = None
            for name in ('liveportrait_gaze.fp16.onnx',
                         os.path.join('liveportrait', 'liveportrait_gaze.fp16.onnx'),
                         'stitching_eye.onnx',
                         os.path.join('liveportrait', 'stitching_eye.onnx')):
                path = _find_model_file(name)
                if os.path.isfile(path):
                    model_path = path
                    break
            if model_path is None or not os.path.isfile(model_path):
                target_path = _find_model_file(os.path.join('liveportrait', 'liveportrait_gaze.fp16.onnx'))
                _create_default_gaze_model(target_path)
                if os.path.isfile(target_path):
                    model_path = target_path

            if model_path is not None and os.path.isfile(model_path):
                providers = getattr(roop.globals, 'execution_providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                try:
                    GAZE_RETARGETER = onnxruntime.InferenceSession(model_path, providers=providers)
                except Exception:
                    try:
                        GAZE_RETARGETER = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                    except Exception:
                        GAZE_RETARGETER = None
    return GAZE_RETARGETER


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
    """Drop per-clip temporal state: the mask history and the tracklets.

    Both, not just the mask. A tracker carried across clips would coast the
    previous clip's people into the first frames of the next one, which is the
    same "swap painted on nothing" failure the coast guards exist to prevent --
    just sourced from a stale run rather than a stale frame. The whole tracker
    goes when no `track_id` is named; a named one only clears that mask history,
    because a tracklet is not addressable by the smoother's key.
    """
    _GLOBAL_SMOOTHER.reset(track_id)
    if track_id is None:
        _GLOBAL_TRACKER.reset()
        im = get_identity_manager()
        if im is not None:
            im.reset()


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
# 5. Cinematic Relighting & Sensor Grain
# ==============================================================================

def _as_float_bgr(image: np.ndarray) -> np.ndarray:
    """Return a finite BGR image in display-scale float32 (0..255)."""
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 3 or data.shape[2] < 3:
        raise ValueError('expected a BGR image')
    return np.nan_to_num(data[..., :3], nan=0.0, posinf=255.0, neginf=0.0)


def _normalise_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Resize and clamp a single-channel compositing mask."""
    if mask is None:
        return np.zeros(shape, dtype=np.float32)
    out = np.asarray(mask, dtype=np.float32)
    if out.ndim == 3:
        out = out[..., 0]
    if out.shape != shape:
        out = cv2.resize(out, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.nan_to_num(out), 0.0, 1.0)


def real_sh_basis(normals: np.ndarray) -> np.ndarray:
    """Evaluate the nine real SH bases through band two for unit normals.

    Column order is ``L00, L1-1, L10, L11, L2-2, L2-1, L20, L21, L22``.
    Normalization constants are folded into the basis, so the fitted values are
    directly usable lighting coefficients.
    """
    n = np.asarray(normals, dtype=np.float32)
    if n.ndim < 2 or n.shape[-1] != 3:
        raise ValueError('real_sh_basis expects normals with a final size of 3')
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack((
        np.full_like(x, 0.282095),
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ), axis=-1).astype(np.float32)


def _plate_directions(shape: Tuple[int, int]) -> np.ndarray:
    """Map crop pixels to a front-facing hemisphere for scene-light fitting."""
    h, w = int(shape[0]), int(shape[1])
    x = (np.arange(w, dtype=np.float32) + 0.5 - w * 0.5) / max(w * 0.5, 1.0)
    y = (np.arange(h, dtype=np.float32) + 0.5 - h * 0.5) / max(h * 0.5, 1.0)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = np.sqrt(np.clip(1.0 - grid_x * grid_x - grid_y * grid_y, 0.0, 1.0))
    directions = np.stack((grid_x, grid_y, grid_z), axis=-1)
    return directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-6)


def estimate_scene_sh_coefficients(
    background_plate: np.ndarray,
    face_exclusion_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit nine SH coefficients from background surrounding the face.

    The crop border maps to a front-facing hemisphere and its luminance is
    regularized with a low-order SH least-squares fit.  Face pixels are excluded
    so source identity/albedo cannot be misread as an incident scene light.
    This model-free route adds no ONNX session or GPU allocation.
    """
    plate = _as_float_bgr(background_plate)
    h, w = plate.shape[:2]
    exclusion = _normalise_mask(face_exclusion_mask, (h, w))
    luminance = (0.0722 * plate[..., 0] + 0.7152 * plate[..., 1] +
                 0.2126 * plate[..., 2]) / 255.0
    sample = (exclusion < 0.10) & np.isfinite(luminance)
    if int(np.count_nonzero(sample)) < 64:
        sample = np.isfinite(luminance)
    basis = real_sh_basis(_plate_directions((h, w))).reshape(-1, 9)
    values = luminance.reshape(-1)
    design = basis[sample.reshape(-1)]
    observed = values[sample.reshape(-1)]
    if design.shape[0] < 9:
        return np.array([float(np.mean(luminance)) / 0.282095] + [0.0] * 8,
                        dtype=np.float32)
    gram = design.T @ design + np.eye(9, dtype=np.float32) * 1e-4
    coefficients = np.linalg.solve(gram, design.T @ observed)
    return np.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def estimate_face_normals(
    shape: Tuple[int, int],
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build fast ellipsoid normals, positioned from five/68-point landmarks.

    Landmark plane geometry supplies face centre and inter-eye scale; the
    ellipsoid's analytic normal approximates cheekbones and the jaw without a
    per-frame monocular-depth model.
    """
    h, w = int(shape[0]), int(shape[1])
    center_x, center_y = w * 0.5, h * 0.53
    radius_x, radius_y = w * 0.43, h * 0.56
    if landmarks is not None:
        try:
            points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
            points = points[np.isfinite(points).all(axis=1)]
            if points.shape[0] >= 5:
                eye_distance = float(np.linalg.norm(points[0] - points[1]))
                if eye_distance > 2.0:
                    center_x = float(np.mean(points[:5, 0]))
                    center_y = float(np.mean(points[:5, 1]))
                    radius_x = np.clip(1.85 * eye_distance, w * 0.25, w * 0.55)
                    radius_y = np.clip(2.45 * eye_distance, h * 0.35, h * 0.70)
        except (TypeError, ValueError):
            pass
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    x = (grid_x - center_x) / max(float(radius_x), 1.0)
    y = (grid_y - center_y) / max(float(radius_y), 1.0)
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    normals = np.stack((x, y, z), axis=-1)
    return normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)


def sh_irradiance(coefficients: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Evaluate non-negative SH irradiance for each supplied normal."""
    coeffs = np.asarray(coefficients, dtype=np.float32).reshape(-1)
    if coeffs.size != 9:
        raise ValueError('sh_irradiance expects exactly nine SH coefficients')
    irradiance = np.tensordot(real_sh_basis(normals), coeffs, axes=([-1], [0]))
    return np.clip(np.nan_to_num(irradiance), 0.01, None).astype(np.float32)


def estimate_sh_lighting_scale(
    coefficients: np.ndarray,
    normals: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    """Normalize SH irradiance to a bounded face-luminance multiplier."""
    irradiance = sh_irradiance(coefficients, normals)
    skin = _normalise_mask(skin_mask, irradiance.shape)
    sample = skin > 0.35
    reference = (float(np.median(irradiance[sample])) if np.any(sample)
                 else float(np.median(irradiance)))
    scale = irradiance / max(reference, 1e-4)
    return np.clip(scale, SH_LIGHTING_MIN_SCALE, SH_LIGHTING_MAX_SCALE).astype(np.float32)


def apply_sh_lighting_transfer(
    swap_crop: np.ndarray,
    background_plate: np.ndarray,
    skin_mask: np.ndarray,
    face_exclusion_mask: Optional[np.ndarray] = None,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multiply swap luminance by scene SH irradiance over blended skin only."""
    swap = _as_float_bgr(swap_crop)
    plate = _as_float_bgr(background_plate)
    if plate.shape[:2] != swap.shape[:2]:
        plate = cv2.resize(plate, (swap.shape[1], swap.shape[0]), interpolation=cv2.INTER_AREA)
    skin = _normalise_mask(skin_mask, swap.shape[:2])
    coefficients = estimate_scene_sh_coefficients(
        plate, skin if face_exclusion_mask is None else face_exclusion_mask)
    normals = estimate_face_normals(swap.shape[:2], landmarks)
    scale = estimate_sh_lighting_scale(coefficients, normals, skin)
    weighted_scale = 1.0 + (scale - 1.0) * skin
    return np.clip(swap * weighted_scale[..., None], 0.0, 255.0).astype(np.uint8)


def estimate_background_grain(
    background_plate: np.ndarray,
    face_exclusion_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Return ``I - GaussianBlur(I, (5,5), 0)`` and its background sigma."""
    plate = _as_float_bgr(background_plate)
    residual = plate - cv2.GaussianBlur(plate, (5, 5), 0)
    exclusion = _normalise_mask(face_exclusion_mask, plate.shape[:2])
    sample = exclusion < 0.10
    if int(np.count_nonzero(sample)) < 64:
        sample = np.ones(plate.shape[:2], dtype=bool)
    return residual.astype(np.float32), max(0.0, float(np.std(residual[sample])))


def synthesize_sensor_noise(
    shape: Tuple[int, int, int],
    sigma_grain: float,
    rng: Optional[np.random.Generator] = None,
    image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate zero-mean Gaussian/Poisson grain at the target variance."""
    sigma = max(0.0, float(sigma_grain))
    if sigma <= 1e-6:
        return np.zeros(shape, dtype=np.float32)
    generator = rng if rng is not None else np.random.default_rng()
    gaussian = generator.normal(0.0, 1.0, size=shape).astype(np.float32)
    signal = (np.full(shape, 128.0, dtype=np.float32) if image is None else
              np.clip(_as_float_bgr(image), 0.0, 255.0))
    electrons = 32.0 + signal
    poisson = ((generator.poisson(electrons).astype(np.float32) - electrons) /
               np.sqrt(electrons))
    noise = ((1.0 - GRAIN_POISSON_WEIGHT) * gaussian +
             GRAIN_POISSON_WEIGHT * poisson)
    noise *= sigma / max(float(np.std(noise)), 1e-6)
    return noise.astype(np.float32)


def inject_film_grain(
    swap_crop: np.ndarray,
    background_plate: np.ndarray,
    skin_mask: np.ndarray,
    face_exclusion_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Inject scene-matched sensor grain into restored blended skin only."""
    swap = _as_float_bgr(swap_crop)
    plate = _as_float_bgr(background_plate)
    if plate.shape[:2] != swap.shape[:2]:
        plate = cv2.resize(plate, (swap.shape[1], swap.shape[0]), interpolation=cv2.INTER_AREA)
    skin = _normalise_mask(skin_mask, swap.shape[:2])
    _, sigma = estimate_background_grain(plate, face_exclusion_mask)
    noise = synthesize_sensor_noise(swap.shape, sigma, rng=rng, image=swap)
    return np.clip(swap + noise * skin[..., None], 0.0, 255.0).astype(np.uint8)


# ==============================================================================
# 6. Core Swap Pipeline
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Tracklet persistence
# ─────────────────────────────────────────────────────────────────────────────
#
# READ THIS BEFORE TREATING THIS MODULE AS THE RENDER PATH. It is not. In this
# fork the video and image pipelines run through `roop.ProcessMgr`; the
# `process_frame` / `process_video` entry points at the bottom of this file are
# pass-throughs. What lives here is the reference implementation of the per-face
# swap and the helper surface the tests exercise. The equivalent wiring in the
# real path is:
#
#   persistence        ProcessMgr.swap_faces          -> FaceTracker.coast
#                      procmgr_tracking._coast_track_gaps
#   occluder mask      ProcessMgr.initialize          -> inject_occlusion_engine
#   occlusion_state    procmgr_masking._stamp_occlusion_state
#
# Both consume the same `roop.tracker`, so there is one tracker and one set of
# guards rather than two that can drift.

_GLOBAL_TRACKER = FaceTracker(max_age=MAX_LOST_FRAMES, max_coast=MAX_COAST_FRAMES)
_GLOBAL_IDENTITY_MANAGER = IdentityManager() if 'IdentityManager' in globals() and IdentityManager is not None else None


def get_identity_manager() -> Optional[Any]:
    """Return the global IdentityManager instance."""
    global _GLOBAL_IDENTITY_MANAGER
    if _GLOBAL_IDENTITY_MANAGER is None and 'IdentityManager' in globals() and IdentityManager is not None:
        _GLOBAL_IDENTITY_MANAGER = IdentityManager()
    return _GLOBAL_IDENTITY_MANAGER


def track_faces(detections: Sequence[Any], frame_index: Optional[int] = None,
                frame_shape: Optional[Tuple[int, ...]] = None
                ) -> Tuple[List[Any], List[Any]]:
    """Associate this frame's detections and coast the ones that went missing.

    Returns ``(faces, coasted)``. `faces` is every face the swap should run on,
    real and predicted, each carrying `_track_id`; `coasted` is the predicted
    subset, which also carries `_coasted`, `_coast_age` and `occlusion_state`.

    Callers pass a face's `_track_id` to `swap_face(..., track_id=)` so the
    temporal mask smoother keeps one history per person rather than one per
    frame -- with the default `track_id=0` two people share a mask history and
    each inherits the other's occlusion boundary.
    """
    return _GLOBAL_TRACKER.update_with_coasting(
        detections, frame_index=frame_index, frame_shape=frame_shape)


def _repair_occluded_landmarks(pts_crop: np.ndarray,
                               occlusion_mask: Optional[np.ndarray],
                               target_face: Any = None) -> Tuple[np.ndarray, str]:
    """Symmetry-inpaint the landmarks the occluder mask says are hidden.

    Returns ``(points, occlusion_state)``.  The points are returned untouched
    whenever the repair is not justified -- nothing occluded, no usable mirror
    map, too much yaw, or both halves of a pair hidden -- so a clear face is
    bit-identical to the pre-change path.

    `occlusion_state` is also stamped on `target_face` when one was supplied,
    which is the flag the spec asks the swap pipeline to carry.
    """
    points = np.asarray(pts_crop, dtype=np.float32).reshape(-1, 2)
    if occlusion_mask is None:
        return points, STATE_VISIBLE
    visible = landmark_visibility(points, occlusion_mask, threshold=0.5)
    if visible.size != len(points):
        return points, STATE_VISIBLE
    coasted = bool(_field(target_face, '_coasted', False)) if target_face is not None else False
    state = occlusion_state_for(visible, coasted=coasted)
    if target_face is not None:
        try:
            target_face['occlusion_state'] = state
            target_face['_occluded_landmark_frac'] = float(1.0 - visible.mean())
        except (TypeError, AttributeError):
            pass
    if state != STATE_PARTIAL:
        return points, state
    repaired, filled = symmetry_inpaint_landmarks(
        points, visible,
        pose=_field(target_face, 'pose', None) if target_face is not None else None)
    return (repaired if filled.any() else points), state


def swap_face(
    source_face: Face,
    target_face: Face,
    temp_frame: Frame,
    track_id: Any = 0
) -> Frame:
    """Core face swapping pipeline with canonical pose normalization,
    profile-aware weighted Umeyama alignment, occlusion subtraction, temporal
    smoothing, eyelid blink retention, and teeth/inner-mouth passthrough."""
    target_frame = temp_frame.copy()
    kps = getattr(target_face, 'kps', None)
    if kps is None and isinstance(target_face, dict):
        kps = target_face.get('kps')

    crop_size = 128
    M = None
    T_final = None

    if kps is not None and len(kps) == 5:
        kps_arr = np.asarray(kps, dtype=np.float32).reshape(5, 2)

        # 1. Canonical Pose Normalization: Compute exact 2D roll angle theta
        theta_rad, theta_deg = compute_canonical_roll_angle(kps_arr)

        # Center of face bbox or landmark centroid
        bbox = getattr(target_face, 'bbox', None)
        if bbox is None and isinstance(target_face, dict):
            bbox = target_face.get('bbox')
        if bbox is not None and len(bbox) == 4:
            center = ((float(bbox[0]) + float(bbox[2])) * 0.5,
                      (float(bbox[1]) + float(bbox[3])) * 0.5)
        else:
            center = (float(np.mean(kps_arr[:, 0])), float(np.mean(kps_arr[:, 1])))

        # If abs(theta) > 45 degrees, dynamically construct affine rotation matrix R(theta, center)
        R, inv_R, applied_rotation = build_canonical_rotation_matrix(center, theta_deg, threshold_deg=45.0)

        h, w = temp_frame.shape[:2]
        if applied_rotation:
            # Sub-pixel bilinear sampling in cv2.warpAffine
            upright_frame = cv2.warpAffine(
                temp_frame, R, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            upright_kps = transform_points(kps_arr, R)
        else:
            upright_frame = temp_frame
            upright_kps = kps_arr

        landmarks_68 = getattr(target_face, 'landmark_3d_68', None)
        if landmarks_68 is None and hasattr(target_face, 'landmarks'):
            landmarks_68 = getattr(target_face, 'landmarks')
        if landmarks_68 is None and isinstance(target_face, dict):
            landmarks_68 = target_face.get('landmark_3d_68')
            if landmarks_68 is None:
                landmarks_68 = target_face.get('landmarks_68')
            if landmarks_68 is None:
                landmarks_68 = target_face.get('landmarks')

        upright_68 = None
        if landmarks_68 is not None:
            pts_68 = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
            if pts_68.shape[0] >= 68:
                upright_68 = transform_points(pts_68, R) if applied_rotation else pts_68

        yaw, pitch, roll_pnp = estimate_head_pose_pnp(
            upright_68 if upright_68 is not None else upright_kps,
            (h, w)
        )

        # 3. Profile-Aware Weighted Umeyama Alignment
        # When abs(yaw) > 35, switches to 3D-aware weighted Umeyama with visibility weighting
        M_warp, align_kind = profile_aware_umeyama_alignment(
            upright_kps,
            image_size=crop_size,
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            landmarks_68=upright_68
        )

        if M_warp is None or not np.isfinite(M_warp).all():
            M_warp, _ = cv2.estimateAffinePartial2D(upright_kps, ARCFACE_DST_128)
            if M_warp is None:
                return temp_frame

        # 4. Sub-pixel bilinear sampling in cv2.warpAffine
        crop_frame = cv2.warpAffine(
            upright_frame, M_warp, (crop_size, crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # 5. Composite inverse transformation T_final = inv(R) @ inv(M_warp)
        inv_M = cv2.invertAffineTransform(M_warp).astype(np.float32)
        T_final = compute_composite_inverse(inv_R, inv_M)
        M = compute_composite_forward(M_warp, R)
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
        if crop_frame.shape[:2] != (crop_size, crop_size):
            crop_frame = cv2.resize(crop_frame, (crop_size, crop_size))
        M = None
        T_final = None

    # Step 1a: Foreground occluder, computed BEFORE the landmarks are consumed.
    #
    # It used to be computed at step 4, after the blink and inner-mouth
    # retention had already run off the raw landmarks. That ordering is the bug
    # this module is being asked to fix: when a hand covers an eye the detector
    # does not report "this eye is hidden", it returns a guessed eye, and the
    # blink logic then reads an eye-aspect ratio measured on a hallucination and
    # pastes the target's eyelid through the hand. The occluder mask is the only
    # thing here that knows which landmarks are behind something, so it has to
    # exist before anything reads them.
    #
    # Nothing extra is computed: this is the same single call, moved.
    face_mask = create_static_face_mask(crop_frame.shape[:2])
    occlusion_mask = compute_occlusion_mask(crop_frame, face_mask=face_mask)

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
            swapped_blob = _run_swapper_bound(swapper, {
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
        landmarks_68 = target_face.get('landmark_3d_68')
        if landmarks_68 is None:
            landmarks_68 = target_face.get('landmarks_68')
        if landmarks_68 is None:
            landmarks_68 = target_face.get('landmarks')

    if landmarks_68 is not None:
        # Transform landmarks into crop space
        pts_full = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
        if M is not None:
            pts_crop = cv2.transform(pts_full.reshape(-1, 1, 2), M).reshape(-1, 2)
        else:
            pts_crop = pts_full - np.array([x1, y1], dtype=np.float32)
        # Step 2a: repair landmarks the occluder says are hidden, by reflecting
        # their partners across the nasal-bridge midline. `symmetry_inpaint_
        # landmarks` refuses on strong yaw and where both halves are hidden, and
        # returns the input untouched when nothing is occluded -- so the common
        # frame is bit-identical to before and only an actually-occluded face
        # is changed. See roop/tracker.py for the axis, the derived mirror map
        # and every refusal condition.
        pts_crop, occlusion_state = _repair_occluded_landmarks(
            pts_crop, occlusion_mask, target_face)
        swapped_crop, _ = apply_facial_dynamics(crop_frame, swapped_crop, pts_crop)
        # Step 2b: LivePortrait Neural Gaze & Eye Direction Retargeting
        swapped_crop, _ = retarget_eye_gaze(swapped_crop, crop_frame, pts_crop)

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

    # Step 4: Foreground occluder subtraction.
    #     M_composite = M_face_blend * (1.0 - M_occluder)
    # The mask itself was computed at step 1a, before anything read the
    # landmarks; this is where it is applied, so the hand or the mug stays in
    # front of the swapped face instead of being painted over.
    blend_mask = apply_occlusion_blend(face_mask, occlusion_mask)

    # Step 4a: Match scene directionality and sensor texture before the final
    # mask is temporally stabilized.  The target crop's surround is the scene
    # plate; the static face mask excludes actor albedo from the SH/noise fit.
    skin_mask = derive_skin_mask(crop_frame, blend_mask)
    swapped_crop = apply_sh_lighting_transfer(
        swapped_crop, crop_frame, skin_mask,
        face_exclusion_mask=face_mask, landmarks=crop_kps)
    swapped_crop = inject_film_grain(
        swapped_crop, crop_frame, skin_mask, face_exclusion_mask=face_mask)

    # Step 5: Temporal Mask Smoothing (Optical Flow / EMA)
    smoothed_mask = smooth_temporal_mask(blend_mask, crop_frame, track_id=track_id)

    # Step 6: Composite and paste back using composite inverse T_final
    if T_final is not None:
        h, w = temp_frame.shape[:2]
        warped_crop = cuda_warp_affine(swapped_crop, T_final, (w, h))
        if warped_crop is None:
            warped_crop = cv2.warpAffine(swapped_crop, T_final, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask = cuda_warp_affine(smoothed_mask, T_final, (w, h))
        if warped_mask is None:
            warped_mask = cv2.warpAffine(smoothed_mask, T_final, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        pyramid_source = temp_frame.copy()
        valid = warped_mask > 1e-4
        pyramid_source[valid] = warped_crop[valid]
        gpu_blend = cuda_laplacian_pyramid_blend(temp_frame, pyramid_source, warped_mask, levels=3)
        return gpu_blend if gpu_blend is not None else laplacian_pyramid_blend(temp_frame, pyramid_source, warped_mask, levels=3)
    elif M is not None:
        inv_M = cv2.invertAffineTransform(M)
        h, w = temp_frame.shape[:2]
        warped_crop = cuda_warp_affine(swapped_crop, inv_M, (w, h))
        if warped_crop is None:
            warped_crop = cv2.warpAffine(swapped_crop, inv_M, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask = cuda_warp_affine(smoothed_mask, inv_M, (w, h))
        if warped_mask is None:
            warped_mask = cv2.warpAffine(smoothed_mask, inv_M, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        pyramid_source = temp_frame.copy()
        valid = warped_mask > 1e-4
        pyramid_source[valid] = warped_crop[valid]
        gpu_blend = cuda_laplacian_pyramid_blend(temp_frame, pyramid_source, warped_mask, levels=3)
        return gpu_blend if gpu_blend is not None else laplacian_pyramid_blend(temp_frame, pyramid_source, warped_mask, levels=3)
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
    """Single-face path. Signature unchanged.

    The only difference from before is that the face's own `_track_id` is passed
    to the temporal mask smoother when the caller has been through
    `track_faces`; without one it still falls back to 0, which is exactly the
    previous behaviour.
    """
    track_id = _field(reference_face, '_track_id', 0)
    return swap_face(source_face, reference_face, temp_frame,
                     track_id=0 if track_id is None else track_id)


def process_frame_tracked(source_faces: Any, detections: Sequence[Any],
                          temp_frame: Frame,
                          frame_index: Optional[int] = None) -> Frame:
    """Multi-face path with tracklet persistence through occlusion and Hungarian
    bipartite identity matching.

    `source_faces` is either:
      - Single faceset / Face (e.g., 'mehak')
      - Sequence of facesets / Faces (e.g., ['mehak', 'misbah'])
      - Mapping of identity names to facesets / Faces
    
    `detections` is this frame's raw detector output -- possibly empty.
    
    Uses optimal Hungarian bipartite matching with Jonker-Volgenant algorithm
    (linear_sum_assignment) over deep ArcFace identity embeddings and Generalized IoU
    with a gating threshold of 0.45 to reject unmapped background faces and
    prevent false-positive swaps.
    
    When tracks cross (IoU > 0.3), a hysteresis state machine locks identity
    assignments and disables spatial weight (alpha = 1.0) until bounding boxes
    separate by >= 1.5x average face width, completely eliminating identity flipping.
    """
    faces, _coasted = track_faces(detections, frame_index=frame_index,
                                  frame_shape=getattr(temp_frame, 'shape', None))
    if not faces:
        return temp_frame

    im = get_identity_manager()
    if im is not None and source_faces is not None:
        needs_bind = False
        if not im.identities:
            needs_bind = True
        elif isinstance(source_faces, (list, tuple)):
            if len(im.identities) != len(source_faces) or any(
                    im.identities[k].source_face is not source_faces[k] for k in range(len(source_faces))):
                needs_bind = True
        elif isinstance(source_faces, dict):
            if len(im.identities) != len(source_faces) or set(
                    id_obj.identity_id for id_obj in im.identities) != set(source_faces.keys()):
                needs_bind = True
        else:
            if len(im.identities) != 1 or im.identities[0].source_face is not source_faces:
                needs_bind = True

        if needs_bind:
            im.bind_facesets(source_faces)

        assignments = im.assign(faces, frame_index=frame_index)
        result = temp_frame
        for face, assignment in zip(faces, assignments):
            if assignment is None:
                # Gated rejection: unmapped background face or low confidence match!
                # Skip swapping this face, preventing false positive swap.
                continue
            ref_identity, cost = assignment
            source = ref_identity.source_face
            if source is None:
                continue
            track_id = int(_field(face, '_track_id', 0) or 0)
            result = swap_face(source, face, result, track_id=track_id)
        return result

    ordered = sorted(faces, key=lambda f: int(_field(f, '_track_id', 0) or 0))
    result = temp_frame
    for face in ordered:
        track_id = int(_field(face, '_track_id', 0) or 0)
        if isinstance(source_faces, (list, tuple)) and source_faces:
            source = source_faces[track_id % len(source_faces)]
        else:
            source = source_faces
        if source is None:
            continue
        result = swap_face(source, face, result, track_id=track_id)
    return result


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
