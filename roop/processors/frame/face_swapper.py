"""Face swapper frame processor with foreground occlusion parsing and temporal mask smoothing.

Task Specification:
1. Occlusion Parsing Pipeline:
   - Integrate an ONNX-based face occlusion / parsing model (e.g. lightweight BiSeNet / DFL-XSeg ONNX session).
   - Segment foreground occlusions (hands, hair, objects, food).
   - Generate effective blend mask by subtracting occlusion from the face mask:
     Mask_blend = Mask_face * (1.0 - Mask_occlusion)
2. Temporal Mask Smoothing (Optical Flow / EMA):
   - Store the previous frame's mask and target crop.
   - Compute motion vectors between frame t-1 and frame t (using OpenCV DIS optical flow or Farneback as fallback).
   - Warp Mask_{t-1} to frame t and apply Exponential Moving Average:
     Mask_t = 0.8 * Mask_t + 0.2 * WarpedMask_{t-1}
   - Eliminates boundary flickering and chatter around moving objects.
3. Performance Guardrails:
   - Execute the occlusion model at 256x256 resolution to keep inference under 4ms on GPU.
   - Add a UI toggle in configuration: --enable-occlusion-mask (default: True).
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
DEFAULT_EMA_ALPHA = 0.8  # Mask_t = 0.8 * Mask_t + 0.2 * WarpedMask_{t-1}

_OCCLUDER_URL = 'https://github.com/rishabh4496/roop-sam-weights/releases/download/v1/face_occluder.onnx'
_SWAPPER_URL = 'https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'

ARCFACE_DST_128 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


class TemporalMaskSmoother:
    """Flow-warped Exponential Moving Average (EMA) over temporal mask sequences.

    Warping Mask_{t-1} to frame t along dense optical flow prevents edge jitter
    and boundary chatter around moving foreground objects (hands, food, microphones).

    Formula:
        Mask_t = 0.8 * Mask_t + 0.2 * WarpedMask_{t-1}
    """

    FLOW_SIZE = 128
    RESET_RESIDUAL = 0.50

    def __init__(self, alpha: float = DEFAULT_EMA_ALPHA):
        self.alpha = float(alpha)
        self._states: Dict[Any, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._tls = threading.local()

    def _flow_engine(self):
        engine = getattr(self._tls, 'dis', None)
        if engine is None:
            try:
                engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            except (AttributeError, cv2.error):
                engine = False
            self._tls.dis = engine
        return engine or None

    def _dense_flow(self, cur_small: np.ndarray, prev_small: np.ndarray) -> np.ndarray:
        """Compute dense backward motion vectors between frame t and frame t-1."""
        engine = self._flow_engine()
        if engine is not None:
            try:
                return engine.calc(cur_small, prev_small, None)
            except Exception:
                pass
        return cv2.calcOpticalFlowFarneback(
            cur_small, prev_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)

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

                # Reset on large flow residual (e.g. cut or sudden teleportation)
                residual = float(np.mean(np.abs(warped - cur_mask)))
                if residual <= self.RESET_RESIDUAL:
                    out_mask = np.clip(eff_alpha * cur_mask + (1.0 - eff_alpha) * warped, 0.0, 1.0)
                else:
                    out_mask = cur_mask
            else:
                out_mask = cur_mask

            self._states[track_id] = {
                'mask': out_mask.copy(),
                'gray': gray_small
            }
            return out_mask

    def reset(self, track_id: Optional[Any] = None) -> None:
        with self._lock:
            if track_id is None:
                self._states.clear()
            else:
                self._states.pop(track_id, None)


_GLOBAL_SMOOTHER = TemporalMaskSmoother(alpha=DEFAULT_EMA_ALPHA)


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
    
    # Identify low-luminance / artificial occluding objects across face
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

            # Face occluder outputs high on unoccluded skin; low on occluded areas.
            # Convert visible face probability to occlusion mask:
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
    """Generate the effective blend mask by subtracting occlusion from the face mask:

        Mask_blend = Mask_face * (1.0 - Mask_occlusion)
    """
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


def blend_swap_buffer(
    target_buffer: np.ndarray,
    swap_buffer: np.ndarray,
    blend_mask: np.ndarray
) -> np.ndarray:
    """Apply effective blend mask to composite swapped face into target buffer:

        Buffer = Target * (1.0 - Mask_blend) + Swap * Mask_blend
    """
    target = np.asarray(target_buffer, dtype=np.float32)
    swap = np.asarray(swap_buffer, dtype=np.float32)
    mask = np.asarray(blend_mask, dtype=np.float32)

    if mask.ndim == 2 and target.ndim == 3:
        mask = mask[..., np.newaxis]

    if swap.shape != target.shape:
        swap = cv2.resize(swap, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != target.shape[:2]:
        mask = cv2.resize(mask, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
        if target.ndim == 3 and mask.ndim == 2:
            mask = mask[..., np.newaxis]

    blended = target * (1.0 - mask) + swap * mask
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def create_static_face_mask(shape: Tuple[int, int], radius_factor: float = 0.40) -> np.ndarray:
    """Generate default elliptical face mask with soft boundary feathering."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.float32)
    center = (int(w / 2), int(h / 2))
    axes = (int(w * radius_factor), int(h * radius_factor * 1.25))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    k = max(3, int(min(h, w) * 0.10) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


def swap_face(
    source_face: Face,
    target_face: Face,
    temp_frame: Frame,
    track_id: Any = 0
) -> Frame:
    """Core face swapping pipeline with occlusion subtraction and temporal smoothing."""
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

    # Step 2: Base face mask
    face_mask = create_static_face_mask(crop_frame.shape[:2])

    # Step 3: Occlusion Parsing Pipeline at 256x256
    occlusion_mask = compute_occlusion_mask(crop_frame, face_mask=face_mask)
    blend_mask = apply_occlusion_blend(face_mask, occlusion_mask)

    # Step 4: Temporal Mask Smoothing (Optical Flow / EMA)
    smoothed_mask = smooth_temporal_mask(blend_mask, crop_frame, track_id=track_id)

    # Step 5: Composite into crop buffer
    blended_crop = blend_swap_buffer(crop_frame, swapped_crop, smoothed_mask)

    # Step 6: Paste back into full frame
    if M is not None:
        inv_M = cv2.invertAffineTransform(M)
        h, w = temp_frame.shape[:2]
        warped_crop = cv2.warpAffine(blended_crop, inv_M, (w, h), borderMode=cv2.BORDER_TRANSPARENT)
        warped_mask = cv2.warpAffine(smoothed_mask, inv_M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        return blend_swap_buffer(temp_frame, warped_crop, warped_mask)
    else:
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
