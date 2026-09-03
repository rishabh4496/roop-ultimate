import glob
import json
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib
from urllib.parse import urlparse
import torch
import gradio
import tempfile
import cv2
import zipfile
import traceback

from pathlib import Path
from typing import List, Any, Optional, Tuple, Dict
from tqdm import tqdm
from scipy.spatial import distance
from datetime import datetime
import numpy as np

import roop.template_parser as template_parser

import roop.globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"


# ---------------------------------------------------------------------------
# CUDA transport and compositing primitives
# ---------------------------------------------------------------------------

class CudaOrtIOBinding:
    """Reusable CUDA buffers for an ONNX Runtime session.

    ORT's normal ``session.run`` path accepts NumPy arrays and returns NumPy
    arrays.  With CUDA/TensorRT that means an upload and download at *every*
    call, even when the next stage is also on CUDA.  This helper owns
    contiguous torch allocations, exposes their pointers to ORT through
    ``bind_input``/``bind_output`` and reuses those allocations for every
    compatible call.

    It is deliberately a best-effort acceleration.  CPU-only installs, a
    provider that rejects a user allocation, dynamic output shapes we cannot
    prove, and any ORT error all return ``None`` so callers retain their
    established ``session.run`` path.  The class is per-session: TensorRT
    contexts must never share an I/O binding or an output allocation.
    """

    def __init__(self, session, device_id: int = 0):
        self.session = session
        self.device_id = int(device_id)
        self.enabled = self._has_cuda_provider(session)
        self._inputs: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}
        self._outputs: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}
        self._lock = threading.RLock()
        self._failure_reported = False

    @staticmethod
    def _has_cuda_provider(session) -> bool:
        try:
            providers = session.get_providers()
            return (torch.cuda.is_available() and any(
                name in ('CUDAExecutionProvider', 'TensorrtExecutionProvider')
                for name in providers))
        except Exception:
            return False

    @staticmethod
    def _shape_for(meta, batch: int) -> Optional[Tuple[int, ...]]:
        shape = list(meta.shape or [])
        if not shape:
            return None
        result = []
        for index, dimension in enumerate(shape):
            if isinstance(dimension, int) and dimension > 0:
                result.append(int(dimension))
            elif index == 0:
                result.append(int(batch))
            else:
                # User-allocated output needs a concrete shape.  Do not guess
                # non-batch dynamic axes; ordinary ORT remains the fallback.
                return None
        return tuple(result)

    @staticmethod
    def _as_float32(value) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        return np.ascontiguousarray(array)

    def _buffer(self, cache, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        key = (name, shape)
        tensor = cache.get(key)
        if tensor is None:
            tensor = torch.empty(shape, dtype=torch.float32,
                                 device=f'cuda:{self.device_id}').contiguous()
            cache[key] = tensor
        return tensor

    def run_gpu(self, feed: Dict[str, np.ndarray]) -> Optional[List[torch.Tensor]]:
        """Run one concrete-shape request and leave every output on CUDA."""
        if not self.enabled or not feed:
            return None
        try:
            with self._lock, torch.cuda.device(self.device_id):
                binding = self.session.io_binding()
                batch = None
                for name, value in feed.items():
                    host = self._as_float32(value)
                    if host.ndim < 1:
                        return None
                    batch = int(host.shape[0]) if batch is None else batch
                    if int(host.shape[0]) != batch:
                        return None
                    device_tensor = self._buffer(self._inputs, name, tuple(host.shape))
                    # This is the sole unavoidable host -> device transfer for
                    # an external NumPy crop.  The ORT call itself does not copy.
                    device_tensor.copy_(torch.from_numpy(host), non_blocking=False)
                    binding.bind_input(name, 'cuda', self.device_id, np.float32,
                                       tuple(host.shape), device_tensor.data_ptr())

                outputs: List[torch.Tensor] = []
                for meta in self.session.get_outputs():
                    shape = self._shape_for(meta, batch or 1)
                    if shape is None:
                        return None
                    output_tensor = self._buffer(self._outputs, meta.name, shape)
                    binding.bind_output(meta.name, 'cuda', self.device_id, np.float32,
                                        shape, output_tensor.data_ptr())
                    outputs.append(output_tensor)
                self.session.run_with_iobinding(binding)
                # The caller may immediately feed outputs to torch compositing;
                # do not synchronize or copy them to the host here.
                return outputs
        except Exception as exc:
            self.enabled = False
            if not self._failure_reported:
                self._failure_reported = True
                print(f'[CUDA I/O binding] disabled for this session: {exc}', flush=True)
            return None

    def run(self, feed: Dict[str, np.ndarray]) -> Optional[List[np.ndarray]]:
        """Compatibility form of :meth:`run_gpu` for NumPy-based callers."""
        outputs = self.run_gpu(feed)
        if outputs is None:
            return None
        # One final D2H transfer is unavoidable for legacy CPU consumers.  It
        # replaces ORT's per-output transfer and happens only after all bound
        # GPU work for this inference has completed.
        return [tensor.detach().cpu().numpy().copy() for tensor in outputs]


def cuda_warp_affine(image: np.ndarray, matrix: np.ndarray,
                     output_size: Tuple[int, int], *,
                     border_mode: str = 'zeros',
                     interpolation: str = 'bilinear') -> Optional[np.ndarray]:
    """GPU equivalent of the common OpenCV affine warp, with safe fallback.

    ``matrix`` has OpenCV's source->destination pixel convention.  PyTorch's
    ``grid_sample`` wants destination->source normalized coordinates, so the
    conversion is explicit rather than relying on a subtly incompatible affine
    shortcut.  The result is returned to NumPy solely because the surrounding
    mature pipeline is still CPU-frame based.
    """
    if not torch.cuda.is_available() or image is None:
        return None
    try:
        data = np.asarray(image)
        if data.ndim not in (2, 3):
            return None
        h, w = data.shape[:2]
        out_w, out_h = int(output_size[0]), int(output_size[1])
        if min(h, w, out_h, out_w) <= 0:
            return None
        channels = 1 if data.ndim == 2 else data.shape[2]
        source = torch.from_numpy(np.ascontiguousarray(data)).to(
            device='cuda', dtype=torch.float32).permute(
                2, 0, 1).unsqueeze(0) if channels > 1 else torch.from_numpy(
                    np.ascontiguousarray(data)).to(device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        affine = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
        inverse = cv2.invertAffineTransform(affine)
        ys, xs = torch.meshgrid(
            torch.arange(out_h, device='cuda', dtype=torch.float32),
            torch.arange(out_w, device='cuda', dtype=torch.float32), indexing='ij')
        src_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
        src_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
        grid = torch.stack((2.0 * (src_x + 0.5) / w - 1.0,
                            2.0 * (src_y + 0.5) / h - 1.0), dim=-1).unsqueeze(0)
        padding = 'border' if border_mode == 'replicate' else 'zeros'
        result = torch.nn.functional.grid_sample(
            source, grid, mode=interpolation, padding_mode=padding,
            align_corners=False)
        result = result.squeeze(0).permute(1, 2, 0) if channels > 1 else result[0, 0]
        if np.issubdtype(data.dtype, np.integer):
            result = result.clamp(0, 255).to(torch.uint8)
        else:
            result = result.to(torch.float32)
        return result.cpu().numpy()
    except Exception:
        return None


def cuda_laplacian_pyramid_blend(target: np.ndarray, swap: np.ndarray,
                                 mask: np.ndarray, levels: int = 3) -> Optional[np.ndarray]:
    """Blend BGR buffers on CUDA using a bounded Laplacian pyramid.

    This keeps the feather/multiband compositing arithmetic on the GPU and
    performs one final download.  It intentionally returns ``None`` on any
    unsupported input so the established OpenCV implementation remains exact.
    """
    if not torch.cuda.is_available():
        return None
    try:
        target_a = np.ascontiguousarray(np.asarray(target, dtype=np.float32))
        swap_a = np.ascontiguousarray(np.asarray(swap, dtype=np.float32))
        mask_a = np.ascontiguousarray(np.asarray(mask, dtype=np.float32))
        if target_a.ndim != 3 or target_a.shape[2] < 3:
            return None
        h, w = target_a.shape[:2]
        if swap_a.shape[:2] != (h, w):
            return None
        if mask_a.ndim == 3:
            mask_a = mask_a[..., 0]
        if mask_a.shape != (h, w):
            return None
        to_t = lambda a: torch.from_numpy(a).to('cuda', non_blocking=False)
        a = to_t(target_a[..., :3]).permute(2, 0, 1).unsqueeze(0)
        b = to_t(swap_a[..., :3]).permute(2, 0, 1).unsqueeze(0)
        m = to_t(np.clip(mask_a, 0.0, 1.0)).unsqueeze(0).unsqueeze(0)
        ga, gb, gm = [a], [b], [m]
        for _ in range(1, max(1, min(int(levels), 3))):
            if min(ga[-1].shape[-2:]) < 2:
                break
            ga.append(torch.nn.functional.avg_pool2d(ga[-1], 2, 2))
            gb.append(torch.nn.functional.avg_pool2d(gb[-1], 2, 2))
            gm.append(torch.nn.functional.avg_pool2d(gm[-1], 2, 2))
        la, lb = [], []
        for index in range(len(ga) - 1):
            size = ga[index].shape[-2:]
            la.append(ga[index] - torch.nn.functional.interpolate(ga[index + 1], size=size,
                                                                    mode='bilinear', align_corners=False))
            lb.append(gb[index] - torch.nn.functional.interpolate(gb[index + 1], size=size,
                                                                    mode='bilinear', align_corners=False))
        blended = ga[-1] * (1.0 - gm[-1]) + gb[-1] * gm[-1]
        for index in range(len(la) - 1, -1, -1):
            blended = torch.nn.functional.interpolate(blended, size=la[index].shape[-2:],
                                                       mode='bilinear', align_corners=False)
            blended = blended + la[index] * (1.0 - gm[index]) + lb[index] * gm[index]
        return blended.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    except Exception:
        return None


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to Nx2 points without mutating the input."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    affine = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
    return pts @ affine[:, :2].T + affine[:, 2]


def compose_affines(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Return the affine equivalent of applying ``before`` then ``after``."""
    a = np.vstack((np.asarray(after, dtype=np.float32).reshape(2, 3),
                   [0.0, 0.0, 1.0]))
    b = np.vstack((np.asarray(before, dtype=np.float32).reshape(2, 3),
                   [0.0, 0.0, 1.0]))
    return (a @ b)[:2].astype(np.float32)


def rotation_affines(bbox: np.ndarray, roll_degrees: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build forward/inverse target-space rotations around a face bbox centre.

    ``roll_degrees`` is the image-space roll to apply.  The caller passes
    ``-roll`` to upright the face.  Keeping both matrices together is important:
    the inverse is composed into the final paste affine, so alpha blending is
    performed back in the original target coordinate system.
    """
    x0, y0, x1, y1 = np.asarray(bbox, dtype=np.float32).reshape(4)
    centre = ((float(x0) + float(x1)) * 0.5, (float(y0) + float(y1)) * 0.5)
    forward = cv2.getRotationMatrix2D(centre, float(roll_degrees), 1.0)
    return forward.astype(np.float32), cv2.invertAffineTransform(forward).astype(np.float32)


def pre_rotate_face_crop(image: np.ndarray, bbox: np.ndarray, roll_degrees: float,
                         threshold: float = 30.0):
    """Upright a target image around ``bbox`` when its roll exceeds ``threshold``.

    Returns ``(upright_image, forward, inverse, applied)``.  For a small roll,
    identity matrices preserve the existing hot path exactly.  ``BORDER_REPLICATE``
    matches the normal alignment crop and avoids introducing a black wedge at an
    image edge.
    """
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    try:
        roll = float(roll_degrees)
    except (TypeError, ValueError):
        return image, identity, identity, False
    if image is None or not np.isfinite(roll) or abs(roll) <= float(threshold):
        return image, identity, identity, False
    forward, inverse = rotation_affines(bbox, -roll)
    upright = cv2.warpAffine(image, forward, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return upright, forward, inverse, True


def _similarity_matrix(source: np.ndarray, destination: np.ndarray) -> Optional[np.ndarray]:
    """Estimate a rigid, uniformly scaled 2D transform (never an anisotropic affine)."""
    src = np.asarray(source, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(destination, dtype=np.float32).reshape(-1, 2)
    if src.shape != dst.shape or src.shape[0] < 3 or not np.isfinite(src).all() or not np.isfinite(dst).all():
        return None
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        return None
    # estimateAffinePartial2D is constrained to a similarity transform.  This
    # explicit check makes a future OpenCV/API change fail safe rather than
    # silently compressing a lateral face.
    linear = matrix[:, :2]
    scales = np.linalg.norm(linear, axis=1)
    if min(scales) <= 1e-7 or max(scales) / min(scales) > 1.0001:
        return None
    return matrix.astype(np.float32)


def profile_alignment_matrix(profile_anchors: np.ndarray, image_size: int,
                             yaw_degrees: float) -> Optional[np.ndarray]:
    """Return a similarity alignment from visible-ear, nose and chin anchors.

    The profile template intentionally has a lateral ear-to-nose baseline and a
    vertical nose-to-chin baseline.  Because the fit is similarity-only, a
    foreshortened visible profile can be scaled and rotated but can never be
    squeezed in one axis.  ``yaw_degrees`` selects the matching visible side.
    """
    anchors = np.asarray(profile_anchors, dtype=np.float32).reshape(-1, 2)
    if anchors.shape != (3, 2):
        return None
    side = 1.0 if float(yaw_degrees) >= 0.0 else -1.0
    size = float(image_size)
    destination = np.array([
        [0.50 * size + side * 0.25 * size, 0.48 * size],  # visible tragus/ear
        [0.50 * size,                         0.52 * size],  # nose tip
        [0.50 * size,                         0.80 * size],  # chin centre
    ], dtype=np.float32)
    return _similarity_matrix(anchors, destination)


def get_small_card_safe_providers(providers=None, model_path=None, stage=None):
    """Choose a safe provider for one heavy mask/enhancer session.

    On the sub-7GB tier the decision is made from the *current* CUDA free
    memory and the model's on-disk size. A small model is admitted to CUDA
    when there is enough measured headroom; an unknown or inadequately sized
    model stays on CPU. This keeps the RTX 3060 path adaptive without
    assuming that every heavy model has the same activation footprint.

    ``ROOP_ALLOW_CUDA_HEAVY_STAGES_SMALL_GPU=1`` remains an explicit diagnostic
    override. ``ROOP_SMALL_CARD_HEAVY_PROVIDER=cpu`` can reproduce the
    conservative baseline; the default is ``auto``.
    """
    selected = list(providers if providers is not None
                    else getattr(roop.globals, 'execution_providers', []) or [])
    if os.environ.get('ROOP_ALLOW_CUDA_HEAVY_STAGES_SMALL_GPU', '').strip().lower() in (
            '1', 'true', 'yes', 'on'):
        return selected
    has_cuda = any('cuda' in str(p[0] if isinstance(p, (tuple, list)) else p).lower()
                   for p in selected)
    if not has_cuda:
        return selected
    try:
        device_id = getattr(roop.globals, 'cuda_device_id', 0)
        props = torch.cuda.get_device_properties(device_id)
        total_bytes = int(getattr(props, 'total_memory', 0) or 0)
        small = (torch.cuda.is_available() and
                 total_bytes > 0 and total_bytes / (1024 ** 3) < 7.0)
    except Exception:
        small = False
    if not small:
        return selected

    policy = os.environ.get('ROOP_SMALL_CARD_HEAVY_PROVIDER', 'auto').strip().lower()
    if policy == 'cpu':
        return ['CPUExecutionProvider']
    if policy not in ('', 'auto', 'cuda'):
        print(f"[Runtime] unknown ROOP_SMALL_CARD_HEAVY_PROVIDER={policy!r}; "
              "using CPU for safety", flush=True)
        return ['CPUExecutionProvider']

    model_bytes = 0
    try:
        if model_path:
            model_bytes = int(os.path.getsize(model_path))
    except (OSError, TypeError, ValueError):
        model_bytes = 0
    if model_bytes <= 0:
        print(f"[Runtime] small-card admission: {stage or 'heavy stage'} "
              "has no measurable model size; using CPU", flush=True)
        return ['CPUExecutionProvider']

    try:
        free_bytes, live_total_bytes = torch.cuda.mem_get_info(device_id)
        free_bytes = int(free_bytes)
        live_total_bytes = int(live_total_bytes or total_bytes)
    except Exception:
        print(f"[Runtime] small-card admission: live VRAM unavailable for "
              f"{stage or 'heavy stage'}; using CPU", flush=True)
        return ['CPUExecutionProvider']

    # ORT may hold graph, allocator, and activation memory in addition to the
    # file mapping. The multiplier is a model-footprint safety factor, not a
    # GPU-specific capacity assumption. Requiring more free memory than the
    # estimate leaves the already-running swap/detection sessions untouched.
    estimated_bytes = model_bytes * 6
    if free_bytes <= estimated_bytes:
        print(f"[Runtime] small-card admission: {stage or 'heavy stage'} "
              f"CUDA rejected (free={free_bytes / 2**20:.0f}MB, "
              f"estimate={estimated_bytes / 2**20:.0f}MB, "
              f"total={live_total_bytes / 2**30:.2f}GB); using CPU", flush=True)
        return ['CPUExecutionProvider']

    print(f"[Runtime] small-card admission: {stage or 'heavy stage'} "
          f"CUDA admitted (free={free_bytes / 2**20:.0f}MB, "
          f"model={model_bytes / 2**20:.1f}MB, "
          f"estimate={estimated_bytes / 2**20:.0f}MB, "
          f"total={live_total_bytes / 2**30:.2f}GB)", flush=True)
    return selected

# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


# https://github.com/facefusion/facefusion/blob/master/facefusion
def detect_fps(target_path: str) -> float:
    # Animated WebP: OpenCV returns 0 FPS — derive from PIL frame durations instead
    if target_path and target_path.lower().endswith('.webp'):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                n = getattr(img, 'n_frames', 1)
                if n > 1:
                    durations = []
                    for i in range(n):
                        img.seek(i)
                        d = img.info.get('duration', None)
                        durations.append(d)
                    print(f"[detect_fps] WebP '{os.path.basename(target_path)}': "
                          f"{n} frames, raw durations (ms) = {durations}")
                    # Treat None or 0 as 100 ms (browsers use ~100 ms as the
                    # effective minimum for animated WebP, similar to GIF).
                    cleaned = [(d if d and d > 0 else 100) for d in durations]
                    avg_ms = sum(cleaned) / len(cleaned)
                    fps = round(1000.0 / avg_ms, 2)
                    print(f"[detect_fps] avg_ms={avg_ms:.1f} → fps={fps}")
                    return fps
        except Exception as exc:
            print(f"[detect_fps] WebP duration read failed: {exc}")
        return 10.0  # safe fallback: 100 ms per frame
    fps = 24.0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def detect_dimensions(target_path: str):
    """Returns (width, height) for images and videos. Returns (0, 0) on failure."""
    if is_image(target_path):
        img = cv2.imread(target_path)
        if img is not None:
            return img.shape[1], img.shape[0]
        return 0, 0
    # Animated WebP: OpenCV VideoCapture returns 0x0 — use PIL instead
    if target_path and target_path.lower().endswith('.webp') and is_animated_webp(target_path):
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                return img.width, img.height
        except Exception:
            return 0, 0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h
    cap.release()
    return 0, 0


# Gradio wants Images in RGB
def convert_to_gradio(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def sort_filenames_ignore_path(filenames):
    """Sorts a list of filenames containing a complete path by their filename,
    while retaining their original path.

    Args:
      filenames: A list of filenames containing a complete path.

    Returns:
      A sorted list of filenames containing a complete path.
    """
    filename_path_tuples = [
        (os.path.split(filename)[1], filename) for filename in filenames
    ]
    sorted_filename_path_tuples = sorted(filename_path_tuples, key=lambda x: x[0])
    return [
        filename_path_tuple[1] for filename_path_tuple in sorted_filename_path_tuples
    ]


def sort_rename_frames(path: str):
    filenames = os.listdir(path)
    filenames.sort()
    for i in range(len(filenames)):
        of = os.path.join(path, filenames[i])
        newidx = i + 1
        new_filename = os.path.join(
            path, f"{newidx:06d}." + roop.globals.CFG.output_image_format
        )
        os.rename(of, new_filename)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob(
        (
            os.path.join(
                glob.escape(temp_directory_path),
                f"*.{roop.globals.CFG.output_image_format}",
            )
        )
    )


def get_temp_frame_paths_from_dir(directory: str) -> List[str]:
    """Return sorted frame image paths from an arbitrary directory.

    Used to get originals from _frames_orig/ for per-frame mask re-processing.
    Tries the configured output_image_format first, then falls back to common formats.
    """
    if not directory or not os.path.isdir(directory):
        return []
    fmt = roop.globals.CFG.output_image_format
    paths = sorted(glob.glob(os.path.join(glob.escape(directory), f'*.{fmt}')))
    if not paths:
        for fallback in ('png', 'jpg', 'jpeg'):
            paths = sorted(glob.glob(os.path.join(glob.escape(directory), f'*.{fallback}')))
            if paths:
                break
    return paths


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def get_destfilename_from_path(
    srcfilepath: str, destfilepath: str, extension: str
) -> str:
    fn, ext = os.path.splitext(os.path.basename(srcfilepath))
    if "." in extension:
        return os.path.join(destfilepath, f"{fn}{extension}")
    return os.path.join(destfilepath, f"{fn}{extension}{ext}")


def replace_template(file_path: str, index: int = 0) -> str:
    fn, ext = os.path.splitext(os.path.basename(file_path))

    # Remove the "__temp" placeholder that was used as a temporary filename
    fn = fn.replace("__temp", "")

    template = roop.globals.CFG.output_template
    replaced_filename = template_parser.parse(
        template, {"index": str(index), "file": fn, "timestamp": datetime.now().strftime('%Y%m%d%H%M%S')}
    )

    return os.path.join(roop.globals.output_path, f"{replaced_filename}{ext}")


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def delete_temp_frames(filename: str) -> None:
    dir = os.path.dirname(os.path.dirname(filename))
    shutil.rmtree(dir)


def get_frames_output_path(target_path: str) -> str:
    """Return the directory where extracted frames are saved when keep_frames is enabled.
    Frames are placed in a <videoname>_frames sub-folder inside the configured output directory."""
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    return os.path.join(roop.globals.output_path, f"{target_name}_frames")


def move_frames_to_output(target_path: str, fps: float = 0.0) -> None:
    """Move the extracted temp frames to a persistent sub-folder in the output directory.

    When fps > 0 a meta.json sidecar is written inside the frames folder so the
    Frame Editor tab can auto-populate FPS and image format without user input.
    """
    temp_dir = get_temp_directory_path(target_path)
    frames_out_dir = get_frames_output_path(target_path)
    if not os.path.isdir(temp_dir):
        return
    # Remove any stale frames folder from a previous run before moving
    if os.path.isdir(frames_out_dir):
        shutil.rmtree(frames_out_dir)
    shutil.move(temp_dir, frames_out_dir)
    # Write metadata sidecar for the Frame Editor
    if fps > 0:
        write_frames_metadata(
            frames_out_dir,
            fps=fps,
            source_name=target_path,
            image_format=roop.globals.CFG.output_image_format,
        )
    # Clean up the now-empty parent temp directory if nothing else uses it
    parent = os.path.dirname(temp_dir)
    if os.path.exists(parent) and not os.listdir(parent):
        os.rmdir(parent)


def write_frames_metadata(frames_dir: str, fps: float, source_name: str, image_format: str) -> None:
    """Write a meta.json sidecar inside *frames_dir* for use by the Frame Editor."""
    meta = {
        "fps": fps,
        "source": os.path.basename(source_name),
        "source_path": source_name,
        "image_format": image_format,
    }
    try:
        with open(os.path.join(frames_dir, 'meta.json'), 'w') as fh:
            json.dump(meta, fh)
    except Exception as exc:
        print(f"write_frames_metadata: {exc}")


def read_frames_metadata(frames_dir: str) -> dict:
    """Read meta.json from *frames_dir*; return empty dict if absent or corrupt."""
    meta_path = os.path.join(frames_dir, 'meta.json')
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r') as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def get_frames_orig_path(target_path: str) -> str:
    """Return the directory where unswapped original frames are stored when keep_frames is enabled.
    Stored alongside the processed frames as <videoname>_frames_orig/ in the output directory."""
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    return os.path.join(roop.globals.output_path, f"{target_name}_frames_orig")


def save_original_frames(target_path: str) -> None:
    """Copy the extracted temp frames to a _frames_orig/ folder BEFORE run_batch overwrites them.

    Called from core.py when keep_frames is True, so the Frame Editor always has
    access to the unswapped source frames for per-frame reprocessing.
    """
    temp_dir = get_temp_directory_path(target_path)
    frames_orig_dir = get_frames_orig_path(target_path)
    if not os.path.isdir(temp_dir):
        return
    if os.path.isdir(frames_orig_dir):
        shutil.rmtree(frames_orig_dir)
    shutil.copytree(temp_dir, frames_orig_dir)


def get_frame_mask_path(frames_orig_dir: str, frame_filename: str) -> str:
    """Return the path for the per-frame mask JSON sidecar.

    frame_filename is the basename of the frame image (e.g. '000001.png').
    The sidecar is stored as '000001_mask.json' in the same _frames_orig/ directory.
    """
    base, _ = os.path.splitext(frame_filename)
    return os.path.join(frames_orig_dir, f"{base}_mask.json")


def save_frame_mask(frames_orig_dir: str, frame_filename: str, mask_data: dict) -> None:
    """Persist per-frame mask settings to a JSON sidecar inside *frames_orig_dir*.

    mask_data is a dict containing any combination of:
      - slider keys: top, bottom, left, right, face_mask_blend,
                     mouth_mask_blend, mouth_top, mouth_bottom,
                     mouth_left, mouth_right (all floats)
      - 'mask_json': the canvas mask JSON string from the mask editor
    """
    mask_path = get_frame_mask_path(frames_orig_dir, frame_filename)
    try:
        with open(mask_path, 'w') as fh:
            json.dump(mask_data, fh)
    except Exception as exc:
        print(f"save_frame_mask: {exc}")


def load_frame_mask(frames_orig_dir: str, frame_filename: str) -> dict:
    """Load per-frame mask settings from the JSON sidecar; return {} if absent or corrupt."""
    mask_path = get_frame_mask_path(frames_orig_dir, frame_filename)
    if os.path.isfile(mask_path):
        try:
            with open(mask_path, 'r') as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg", "webp"))


def has_extension(filepath: str, extensions: List[str]) -> bool:
    return filepath.lower().endswith(tuple(extensions))


def is_animated_webp(image_path: str) -> bool:
    """Return True if the file is an animated (multi-frame) WebP."""
    if not image_path or not image_path.lower().endswith(".webp"):
        return False
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return getattr(img, "n_frames", 1) > 1
    except Exception:
        return False


def is_animated_gif(image_path: str) -> bool:
    """Return True if the file is an animated (multi-frame) GIF."""
    if not image_path or not image_path.lower().endswith(".gif"):
        return False
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return getattr(img, "n_frames", 1) > 1
    except Exception:
        return False


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        if image_path.lower().endswith(".webp"):
            # Animated webp is not a static image
            return not is_animated_webp(image_path)
        if image_path.lower().endswith(".gif"):
            # Animated gif is not a static image
            return not is_animated_gif(image_path)
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


_ONLINE_STATE = None
_ONLINE_STATE_AT = 0.0
_ONLINE_HOST_STATE = {}
_ONLINE_CACHE_SECONDS = 30.0


def reset_online_state() -> None:
    """Clear the short-lived connectivity cache.

    Connectivity is intentionally not a process-lifetime fact: a laptop can
    reconnect after startup.  This hook also keeps tests from sharing network
    state with one another.
    """
    global _ONLINE_STATE, _ONLINE_STATE_AT
    _ONLINE_STATE = None
    _ONLINE_STATE_AT = 0.0
    _ONLINE_HOST_STATE.clear()


def is_online(timeout: float = 2.5, hosts=None) -> bool:
    """Best-effort, short-lived connectivity probe for model acquisition.

    ``hosts`` lets a caller check the host it will actually use.  The result is
    cached briefly because startup may ask for many models, but it expires so a
    connection restored later in the same process can be used.  This function
    is never called from a per-frame processing path.
    """
    global _ONLINE_STATE, _ONLINE_STATE_AT
    import socket

    default_hosts = ("huggingface.co", "github.com")
    explicit_hosts = hosts is not None
    host_values = (hosts,) if isinstance(hosts, str) else (hosts or default_hosts)
    target_hosts = tuple(str(host).strip() for host in host_values
                         if str(host).strip())
    if not target_hosts:
        return False
    now = time.monotonic()
    cache_key = tuple(dict.fromkeys(target_hosts))
    if explicit_hosts:
        cached = _ONLINE_HOST_STATE.get(cache_key)
        if cached and now - cached[1] < _ONLINE_CACHE_SECONDS:
            return cached[0]
    elif _ONLINE_STATE is not None and (
            _ONLINE_STATE_AT == 0.0 or now - _ONLINE_STATE_AT < _ONLINE_CACHE_SECONDS):
        return _ONLINE_STATE

    result = False
    for host in cache_key:
        try:
            with socket.create_connection((host, 443), timeout=timeout):
                result = True
                break
        except OSError:
            continue
    if explicit_hosts:
        _ONLINE_HOST_STATE[cache_key] = (result, now)
    else:
        _ONLINE_STATE = result
        _ONLINE_STATE_AT = now
    return result


def _handle_missing_model(download_file_path: str, download_directory_path: str,
                          required: bool, reason: str) -> None:
    """Shared policy for a model that is absent and could not be downloaded.

    required=True  -> raise a clear, actionable error. A feature the user just
                      selected needs this model and it is not on disk.
    required=False -> warn and continue. Used by the startup pre-warm so the app
                      still boots offline with whatever partial model set exists;
                      the missing model only surfaces if its feature is used."""
    name = os.path.basename(download_file_path)
    detail = "you appear to be offline" if reason == "offline" else f"download failed: {reason}"
    msg = (
        f"Model '{name}' is not available locally and could not be downloaded "
        f"({detail}). Place the file in '{download_directory_path}' to use this "
        f"feature offline."
    )
    if required:
        raise RuntimeError(msg)
    try:
        print(f"\033[93m[OFFLINE] {msg}\033[0m")
    except Exception:
        print(f"[OFFLINE] {msg}")


def conditional_download(download_directory_path: str, urls: List[str], required: bool = True) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)

    if hasattr(ssl, '_create_unverified_context'):
        ssl._create_default_https_context = ssl._create_unverified_context

    for url in urls:
        download_file_path = os.path.join(
            download_directory_path, os.path.basename(url)
        )
        if os.path.exists(download_file_path):
            continue

        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            _handle_missing_model(download_file_path, download_directory_path,
                                  required, reason="model URL has no host")
            continue

        # Auto offline mode: with no connectivity to THIS model host, don't
        # block on a download timeout. Fall back to local files and let
        # _handle_missing_model apply the required policy.
        if not is_online(hosts=(host,)):
            _handle_missing_model(download_file_path, download_directory_path, required, reason="offline")
            continue

        # Download to a .part file and rename only on success. Writing the
        # final filename directly means an interrupted download leaves a
        # truncated file that the exists() check above then treats as a
        # complete model forever (cryptic ONNX load error until the user
        # deletes it by hand).
        partial_path = download_file_path + ".part"
        try:
            total = 0
            try:
                with urllib.request.urlopen(url) as response:
                    total = int(response.headers.get("Content-Length", 0))
            except Exception:
                pass
            with tqdm(
                total=total,
                desc=f"Downloading {url}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(url, partial_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]
            if total and os.path.getsize(partial_path) < total:
                raise IOError(f"Incomplete download: got {os.path.getsize(partial_path)} of {total} bytes")
            os.replace(partial_path, download_file_path)
        except Exception as exc:
            if os.path.exists(partial_path):
                try:
                    os.remove(partial_path)
                except OSError:
                    pass
            # A failed download (transient network error, host down, partial
            # transfer) is handled the same way as offline: clear error if the
            # model is required now, otherwise warn and move on.
            _handle_missing_model(download_file_path, download_directory_path, required, reason=str(exc))


def get_local_files_from_folder(folder: str) -> List[str]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
    return files


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_device() -> str:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()

    if len(roop.globals.execution_providers) < 1:
        if 'CUDAExecutionProvider' in available_providers:
            roop.globals.execution_providers = ['CUDAExecutionProvider']
        else:
            roop.globals.execution_providers = ["CPUExecutionProvider"]

    prov = roop.globals.execution_providers[0]
    if "CoreMLExecutionProvider" in prov:
        return "mps"
    if "CUDAExecutionProvider" in prov or "ROCMExecutionProvider" in prov or "TensorrtExecutionProvider" in prov:
        return "cuda"
    if "OpenVINOExecutionProvider" in prov:
        return "mkl"
    return "cpu"


def str_to_class(module_name, class_name) -> Any:
    from importlib import import_module

    class_ = None
    try:
        module_ = import_module(module_name)
        try:
            class_ = getattr(module_, class_name)()
        except AttributeError:
            print(f"Class {class_name} does not exist")
    except ImportError:
        print(f"Module {module_name} does not exist")
    return class_

def is_installed(name:str) -> bool:
    return shutil.which(name);

# Taken from https://stackoverflow.com/a/68842705
def get_platform() -> str:
    if sys.platform == "linux":
        try:
            proc_version = open("/proc/version").read()
            if "Microsoft" in proc_version:
                return "wsl"
        except:
            pass
    return sys.platform

def open_with_default_app(filename:str):
    if filename == None:
        return
    platform = get_platform()
    if platform == "darwin":
        subprocess.call(("open", filename))
    elif platform in ["win64", "win32"]:        os.startfile(filename.replace("/", "\\"))
    elif platform == "wsl":
        subprocess.call("cmd.exe /C start".split() + [filename])
    else:  # linux variants
        subprocess.call("xdg-open", filename)


def prepare_for_batch(target_files) -> str:
    print("Preparing temp files")
    tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
    if os.path.exists(tempfolder):
        shutil.rmtree(tempfolder)
    Path(tempfolder).mkdir(parents=True, exist_ok=True)
    for f in target_files:
        newname = os.path.basename(f.name)
        shutil.move(f.name, os.path.join(tempfolder, newname))
    return tempfolder


def zip(files, zipname):
    with zipfile.ZipFile(zipname, "w") as zip_file:
        for f in files:
            zip_file.write(f, os.path.basename(f))


def unzip(zipfilename: str, target_path: str):
    with zipfile.ZipFile(zipfilename, "r") as zip_file:
        zip_file.extractall(target_path)


def mkdir_with_umask(directory):
    oldmask = os.umask(0)
    # mode needs octal
    os.makedirs(directory, mode=0o775, exist_ok=True)
    os.umask(oldmask)


def open_folder(path: str):
    platform = get_platform()
    try:
        if platform == "darwin":
            subprocess.call(("open", path))
        elif platform in ["win64", "win32"]:
            open_with_default_app(path)
        elif platform == "wsl":
            subprocess.call("cmd.exe /C start".split() + [path])
        else:  # linux variants
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        traceback.print_exc()
        pass
        # import webbrowser
        # webbrowser.open(url)


def create_version_html() -> str:
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    versions_html = f"""
python: <span title="{sys.version}">{python_version}</span>
•
torch: {getattr(torch, '__long_version__',torch.__version__)}
•
gradio: {gradio.__version__}
"""
    return versions_html


def compute_cosine_distance(emb1, emb2) -> float:
    if emb1 is None or emb2 is None:
        return 1.0
    u = np.asarray(emb1, dtype=np.float32)
    v = np.asarray(emb2, dtype=np.float32)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-7 or nv < 1e-7:
        return 1.0
    dot = float(np.dot(u, v))
    return max(0.0, min(2.0, 1.0 - (dot / (nu * nv))))

def has_cuda_device():
    return torch.cuda is not None and torch.cuda.is_available()


def print_cuda_info():
    try:
        print(f'Number of CUDA devices: {torch.cuda.device_count()} Currently used Id: {torch.cuda.current_device()} Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    except:
       print('No CUDA device found!')


def get_onnx_session_options(optimization_level=None):
    """Return memory-bounded SessionOptions for ONNX Runtime.

    Disables the unbounded CPU memory arena (BFCArena) and memory pattern caching
    which cause steady RAM accumulation over long video processing runs.
    """
    import onnxruntime

    def _session_threads(name, default, upper):
        # An explicit process environment value wins over the optimizer hint.
        # The defaults are intentionally serial per session: Python workers
        # provide the outer parallelism, so ORT must not create a hidden pool
        # for every worker.
        raw = os.environ.get(name)
        if raw is None or str(raw).strip().lower() in ('', 'auto', 'default'):
            raw = os.environ.get('ROOP_RUNTIME_' + name[5:])
        try:
            return max(1, min(upper, int(raw)))
        except (TypeError, ValueError):
            return default

    try:
        opts = onnxruntime.SessionOptions()
        if hasattr(opts, 'enable_cpu_mem_arena'):
            opts.enable_cpu_mem_arena = False
        if hasattr(opts, 'enable_mem_pattern'):
            opts.enable_mem_pattern = False
        if hasattr(opts, 'execution_mode') and hasattr(onnxruntime, 'ExecutionMode'):
            try:
                opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            except Exception:
                pass
        if hasattr(opts, 'intra_op_num_threads'):
            opts.intra_op_num_threads = _session_threads(
                'ROOP_ORT_INTRA_THREADS', 1, 4)
        if hasattr(opts, 'inter_op_num_threads'):
            opts.inter_op_num_threads = _session_threads(
                'ROOP_ORT_INTER_THREADS', 1, 2)
        # ORT's constant-cost scheduler can produce high latency variance on
        # Windows. Keep one thread per session, but distribute uneven CPU work
        # more fairly between repeated calls.
        if hasattr(opts, 'add_session_config_entry'):
            try:
                opts.add_session_config_entry(
                    'session.dynamic_block_base',
                    str(os.environ.get('ROOP_ORT_DYNAMIC_BLOCK', '4')))
            except Exception:
                pass
        if hasattr(opts, 'log_severity_level'):
            opts.log_severity_level = 3
        if optimization_level is not None:
            if hasattr(onnxruntime, 'GraphOptimizationLevel') and hasattr(onnxruntime.GraphOptimizationLevel, 'ORT_ENABLE_EXTENDED'):
                if optimization_level == 1 or optimization_level == onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED:
                    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                else:
                    try:
                        opts.graph_optimization_level = optimization_level
                    except Exception:
                        pass
            else:
                try:
                    opts.graph_optimization_level = optimization_level
                except Exception:
                    pass
        return opts
    except Exception:
        return None

print_cuda_info()
