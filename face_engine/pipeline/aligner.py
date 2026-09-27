"""5-point similarity alignment, crop warps and paste-back.

Templates
---------
Each template is the five landmark positions (left eye, right eye, nose, left
and right mouth corner) a model was trained on, normalised to ``[0, 1]`` of
the crop side. Values match roop-ultimate's ``face_util.WARP_TEMPLATES`` and
InsightFace's ``arcface_dst``:

=====================  ======  =====================================================
name                   size    used by
=====================  ======  =====================================================
``arcface_112``        112     ArcFace / w600k embeddings (InsightFace ``arcface_dst``)
``arcface_128``        256     HyperSwap, inswapper (``arcface_dst`` at 128 + 8px x-shift)
``ffhq_512``           512     GPEN-BFR 512/1024, RestoreFormer++, CodeFormer
``arcface_112_v1``     512     SimSwap 512
=====================  ======  =====================================================

:data:`CANONICAL_TEMPLATES` maps the three standard input sizes to a default;
SimSwap and GPEN disagree at 512 (different templates), so 512 defaults to
``ffhq_512`` and SimSwap callers pass ``template="arcface_112_v1"``.

Robustness
----------
* :func:`estimate_similarity_transform` raises :class:`AlignmentError` only
  for genuinely degenerate input (non-finite, coincident points).
  :func:`align_face` never raises; it returns ``None`` instead.
* Every warp computes the region of interest the crop covers in the frame and
  clips it, so faces cut by the frame edge, or lying entirely outside it,
  are handled without exceptions; the crop's out-of-frame area is reported in
  :attr:`AlignedFace.valid` so masks can exclude it.
* ``fit_error`` (RMS landmark residual as a fraction of the crop) is large for
  sharp profiles and mirrored landmark sets; callers can gate on it.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

_ARCFACE_DST_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)

TEMPLATES: dict[str, np.ndarray] = {
    "arcface_112": _ARCFACE_DST_112 / 112.0,
    "arcface_128": (_ARCFACE_DST_112 + np.array([8.0, 0.0])) / 128.0,
    "ffhq_512": np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813],
                          [0.50123859, 0.61331904], [0.39308822, 0.72541100],
                          [0.61150205, 0.72490465]], dtype=np.float64),
    "arcface_112_v1": np.array([[0.35473214, 0.45658929], [0.64526786, 0.45658929],
                                [0.50000000, 0.61154464], [0.37913393, 0.77687500],
                                [0.62086607, 0.77687500]], dtype=np.float64),
}

CANONICAL_TEMPLATES: dict[int, str] = {112: "arcface_112", 256: "arcface_128", 512: "ffhq_512"}


class AlignmentError(ValueError):
    """The landmarks cannot define a similarity transform."""


def template_points(crop_size: int, template: str | None = None) -> np.ndarray:
    """Template landmarks in crop pixels, ``(5, 2)`` float64.

    Args:
        crop_size: Side of the square crop.
        template: A :data:`TEMPLATES` key; defaults to the canonical template
            for ``crop_size`` (112/256/512) and raises for other sizes.
    """
    if template is None:
        if crop_size not in CANONICAL_TEMPLATES:
            raise KeyError(f"no canonical template for {crop_size}px; pass template=")
        template = CANONICAL_TEMPLATES[crop_size]
    return TEMPLATES[template] * float(crop_size)


def estimate_similarity_transform(source_points: np.ndarray,
                                  target_points: np.ndarray) -> np.ndarray:
    """Least-squares similarity (rotation, uniform scale, translation) via SVD.

    Umeyama (1991): the rotation comes from the SVD of the cross-covariance,
    with the smallest singular direction flipped when needed so the result is
    a proper rotation — a mirrored landmark set gets the best *rotation*, and
    a large residual, never a reflection.

    Returns:
        ``(2, 3)`` float64 matrix mapping source to target.

    Raises:
        AlignmentError: fewer than 2 points, mismatched shapes, non-finite
            values, or source points that are (numerically) coincident.
    """
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape or src.shape[0] < 2:
        raise AlignmentError(f"need matching (N>=2, 2) point sets, got {src.shape} and {dst.shape}")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise AlignmentError("landmarks contain NaN or inf")
    n = src.shape[0]
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = float((sc ** 2).sum() / n)
    if var_s < 1e-12 * max(1.0, float(np.abs(src).max()) ** 2):
        raise AlignmentError("source landmarks are coincident")
    cov = dc.T @ sc / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1.0
    rotation = u @ np.diag(d) @ vt
    scale = float((s * d).sum() / var_s)
    if not np.isfinite(scale) or scale <= 0:
        raise AlignmentError(f"degenerate scale {scale}")
    matrix = np.empty((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = mu_d - scale * rotation @ mu_s
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a ``(2, 3)`` affine to ``(..., 2)`` points."""
    pts = np.asarray(points, dtype=np.float64)
    return pts @ matrix[:, :2].T + matrix[:, 2]


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a ``(2, 3)`` affine."""
    return cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float64))


def matrix_scale(matrix: np.ndarray) -> float:
    """Uniform scale of a similarity matrix (crop px per frame px)."""
    return float(np.sqrt(abs(np.linalg.det(np.asarray(matrix)[:, :2]))))


@dataclass(frozen=True)
class AlignedFace:
    """A warped face crop and how to get back to the frame.

    Attributes:
        crop: ``(S, S, 3)`` uint8 crop.
        matrix: ``(2, 3)`` frame -> crop similarity.
        valid: ``(S, S)`` float32, 1 where the crop samples inside the frame,
            0 where it was padded (face cut by the frame edge).
        fit_error: RMS landmark residual / crop size. ~0.01-0.03 frontal;
            grows with yaw; mirrored or garbage landmarks read far higher.
        template: Template name used.
    """

    crop: np.ndarray
    matrix: np.ndarray
    valid: np.ndarray
    fit_error: float
    template: str

    @property
    def crop_size(self) -> int:
        return int(self.crop.shape[0])

    @property
    def fully_inside(self) -> bool:
        return bool(self.valid.min() >= 0.999)


def _frame_roi(corners: np.ndarray, frame_h: int, frame_w: int,
               pad: int = 2) -> tuple[int, int, int, int] | None:
    """Clipped integer ``x0, y0, x1, y1`` covering ``corners`` (+pad), or None if empty."""
    if not np.all(np.isfinite(corners)):
        return None
    x0 = max(int(np.floor(corners[:, 0].min())) - pad, 0)
    y0 = max(int(np.floor(corners[:, 1].min())) - pad, 0)
    x1 = min(int(np.ceil(corners[:, 0].max())) + pad, frame_w)
    y1 = min(int(np.ceil(corners[:, 1].max())) + pad, frame_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _crop_corners(crop_size: int) -> np.ndarray:
    s = float(crop_size)
    return np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)


def warp_face_by_translation(frame: np.ndarray, matrix: np.ndarray, crop_size: int,
                             antialias: bool = True,
                             border_mode: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Warp the face region of ``frame`` into a ``crop_size`` square crop.

    Only the frame region the crop covers is touched. With ``antialias``, a
    face larger than the crop is Gaussian-prefiltered before resampling
    (bilinear sampling of a >2x downscale aliases skin texture and hair).

    Args:
        frame: ``(H, W, C)`` image.
        matrix: ``(2, 3)`` frame -> crop affine.
        crop_size: Output side.
        border_mode: How the crop is padded where it leaves the frame.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    h, w = frame.shape[:2]
    scale = matrix_scale(matrix)
    sigma = 0.5 * (1.0 / scale - 1.0) if antialias and 0 < scale < 0.5 else 0.0
    # The ROI margin covers the blur footprint so the prefilter sees real
    # neighbours, not the ROI's own reflected edge.
    pad = 4 + int(np.ceil(3.0 * sigma))
    roi = _frame_roi(transform_points(_crop_corners(crop_size), invert_affine(matrix)), h, w,
                     pad=pad)
    if roi is None:
        return np.zeros((crop_size, crop_size) + frame.shape[2:], dtype=frame.dtype)
    x0, y0, x1, y1 = roi
    source = frame[y0:y1, x0:x1]
    if sigma > 0:
        source = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Sampling only inside the ROI: where the ROI was clipped, its edge IS the
    # frame edge, so border_mode behaves exactly as on the full frame.
    shifted = matrix.copy()
    shifted[:, 2] += matrix[:, :2] @ np.array([x0, y0], dtype=np.float64)
    return cv2.warpAffine(source, shifted, (crop_size, crop_size), flags=cv2.INTER_LINEAR,
                          borderMode=border_mode)


def crop_valid_mask(frame_shape: tuple[int, ...], matrix: np.ndarray,
                    crop_size: int) -> np.ndarray:
    """``(S, S)`` float32: 1 where the crop samples inside the frame."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    inside = np.ones((h, w), dtype=np.uint8)
    matrix = np.asarray(matrix, dtype=np.float64)
    roi = _frame_roi(transform_points(_crop_corners(crop_size), invert_affine(matrix)), h, w,
                     pad=4)
    if roi is None:
        return np.zeros((crop_size, crop_size), dtype=np.float32)
    x0, y0, x1, y1 = roi
    shifted = matrix.copy()
    shifted[:, 2] += matrix[:, :2] @ np.array([x0, y0], dtype=np.float64)
    # Bilinear taps straddling the frame edge come out fractional: a soft edge.
    region = inside[y0:y1, x0:x1] * 255
    valid = cv2.warpAffine(region, shifted, (crop_size, crop_size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (valid.astype(np.float32) / 255.0)


def align_face(frame: np.ndarray, kps: np.ndarray, crop_size: int = 256,
               template: str | None = None, antialias: bool = True) -> AlignedFace | None:
    """Fit ``kps`` to the template and warp the crop. Returns None when the fit is degenerate."""
    try:
        name = template or CANONICAL_TEMPLATES[crop_size]
        dst = template_points(crop_size, name)
        matrix = estimate_similarity_transform(kps, dst)
    except (AlignmentError, KeyError):
        return None
    residual = transform_points(np.asarray(kps, dtype=np.float64).reshape(-1, 2), matrix) - dst
    fit_error = float(np.sqrt((residual ** 2).sum(axis=1).mean()) / crop_size)
    crop = warp_face_by_translation(frame, matrix, crop_size, antialias=antialias)
    valid = crop_valid_mask(frame.shape, matrix, crop_size)
    return AlignedFace(crop=crop, matrix=matrix, valid=valid, fit_error=fit_error, template=name)


def paste_mask_to_canvas(mask: np.ndarray, matrix: np.ndarray,
                         frame_shape: tuple[int, ...]) -> np.ndarray:
    """Inverse-warp a crop-space mask to a full ``(H, W)`` float32 frame mask."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    canvas = np.zeros((h, w), dtype=np.float32)
    placed = _inverse_roi(np.asarray(mask, dtype=np.float32), matrix, h, w, cv2.BORDER_CONSTANT)
    if placed is not None:
        (x0, y0, x1, y1), warped = placed
        canvas[y0:y1, x0:x1] = np.clip(warped, 0.0, 1.0)
    return canvas


def _inverse_roi(crop: np.ndarray, matrix: np.ndarray, h: int, w: int,
                 border_mode: int) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    size = crop.shape[0]
    inverse = invert_affine(np.asarray(matrix, dtype=np.float64))
    roi = _frame_roi(transform_points(_crop_corners(size), inverse), h, w)
    if roi is None:
        return None
    x0, y0, x1, y1 = roi
    shifted = inverse.copy()
    shifted[:, 2] -= np.array([x0, y0], dtype=np.float64)
    warped = cv2.warpAffine(crop, shifted, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=border_mode, borderValue=0)
    return roi, warped


def warp_face_inverse(target_frame: np.ndarray, swapped_crop: np.ndarray, matrix: np.ndarray,
                      mask: np.ndarray | None = None) -> np.ndarray:
    """Paste ``swapped_crop`` back into a copy of ``target_frame``.

    Args:
        target_frame: ``(H, W, C)`` frame (not modified).
        swapped_crop: ``(S, S, C)`` crop in the same space ``matrix`` maps to.
        matrix: ``(2, 3)`` frame -> crop affine used to cut the crop.
        mask: ``(S, S)`` blend weight in ``[0, 1]``; defaults to all ones. The
            area outside the crop always gets weight 0.

    Returns:
        A new frame of ``target_frame``'s dtype; an unchanged copy when the
        crop does not overlap the frame.
    """
    out = target_frame.copy()
    h, w = target_frame.shape[:2]
    size = swapped_crop.shape[0]
    weight = np.ones((size, size), np.float32) if mask is None else \
        np.clip(np.asarray(mask, dtype=np.float32).reshape(size, size), 0.0, 1.0)
    placed_w = _inverse_roi(weight, matrix, h, w, cv2.BORDER_CONSTANT)
    if placed_w is None:
        return out
    (x0, y0, x1, y1), alpha = placed_w
    fast = (out.dtype == np.uint8 and swapped_crop.dtype == np.uint8
            and out.ndim == swapped_crop.ndim and out.shape[2:] == swapped_crop.shape[2:])
    if fast:
        # uint8 end to end: bilinear-warp the crop as uint8 and let OpenCV blend
        # with float weights. 1080p, one ~600px face: 10.0 ms -> 2.4 ms
        # (2026-09-27), output within 1 level of the float path.
        placed_img = _inverse_roi(swapped_crop, matrix, h, w, cv2.BORDER_REPLICATE)
        assert placed_img is not None
        image = placed_img[1].reshape(out[y0:y1, x0:x1].shape)
        out[y0:y1, x0:x1] = cv2.blendLinear(image, out[y0:y1, x0:x1], alpha,
                                            np.ascontiguousarray(1.0 - alpha))
        return out
    placed_img = _inverse_roi(swapped_crop.astype(np.float32), matrix, h, w, cv2.BORDER_REPLICATE)
    assert placed_img is not None
    image = placed_img[1]
    if image.ndim == 3:
        alpha = alpha[:, :, None]
    region = out[y0:y1, x0:x1].astype(np.float32)
    blended = region + (image.reshape(region.shape) - region) * alpha
    if np.issubdtype(out.dtype, np.integer):
        info = np.iinfo(out.dtype)
        blended = np.clip(np.rint(blended), info.min, info.max)
    out[y0:y1, x0:x1] = blended.astype(out.dtype)
    return out


# ---------------------------------------------------------------------------- GPU (kornia)
_TF32_LOCK = threading.Lock()


@contextmanager
def _exact_fp32() -> Iterator[None]:
    """Disable TF32 for the duration of a kornia warp, then restore it.

    kornia builds the sampling grid with float32 matmuls. With TF32 enabled
    (roop-ultimate's ``core.py`` turns it on process-wide) the grid carries a
    10-bit mantissa: on the insightface sample faces the crop error grew from
    max 2 / mean 0.06 to max 17 / mean 0.40 levels (2026-09-27). The flags
    are process-global, so the switch is serialized; other threads' matmuls
    run in exact FP32 for those microseconds, which is only slower.
    """
    import torch

    with _TF32_LOCK:
        previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _as_matrix_tensor(matrix: Any, batch: int, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
    import torch

    m = torch.as_tensor(np.asarray(matrix) if not isinstance(matrix, torch.Tensor) else matrix,
                        device=device, dtype=dtype)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] == 1 and batch > 1:
        m = m.expand(batch, 2, 3)
    if m.shape != (batch, 2, 3):
        raise ValueError(f"expected matrices of shape ({batch}, 2, 3), got {tuple(m.shape)}")
    return m


def warp_face_gpu(frames: torch.Tensor, matrix: Any, crop_size: int,
                  padding_mode: str = "border") -> torch.Tensor:
    """Batched crop warp on the frames' device with ``kornia.geometry.transform.warp_affine``.

    Args:
        frames: ``(B, C, H, W)`` float tensor.
        matrix: ``(B, 2, 3)`` or ``(2, 3)`` frame -> crop affine (numpy or tensor).
        crop_size: Output side.
        padding_mode: ``"border"`` (like ``BORDER_REPLICATE``) or ``"zeros"``.

    Returns:
        ``(B, C, S, S)`` tensor. Uses the same pixel-centre convention as
        OpenCV (``align_corners=True``) and matches :func:`warp_face_by_translation`
        with ``antialias=False`` to within bilinear rounding.
    """
    from kornia.geometry.transform import warp_affine

    m = _as_matrix_tensor(matrix, frames.shape[0], frames.device, frames.dtype)
    with _exact_fp32():
        return warp_affine(frames, m, dsize=(crop_size, crop_size), mode="bilinear",
                           padding_mode=padding_mode, align_corners=True)


def warp_face_inverse_gpu(target_frames: torch.Tensor, swapped_crops: torch.Tensor,
                          matrix: Any, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Batched paste-back on the GPU; returns a new ``(B, C, H, W)`` tensor.

    ``mask`` is ``(B, 1, S, S)`` in ``[0, 1]`` (default ones). Outside the
    crop the weight is zero.
    """
    import torch
    from kornia.geometry.transform import invert_affine_transform, warp_affine

    b, _, h, w = target_frames.shape
    size = swapped_crops.shape[-1]
    m = _as_matrix_tensor(matrix, b, target_frames.device, target_frames.dtype)
    if mask is None:
        mask = torch.ones((b, 1, size, size), device=target_frames.device,
                          dtype=target_frames.dtype)
    with _exact_fp32():
        inverse = invert_affine_transform(m)
        image = warp_affine(swapped_crops.to(target_frames.dtype), inverse, dsize=(h, w),
                            mode="bilinear", padding_mode="border", align_corners=True)
        alpha = warp_affine(mask.to(target_frames.dtype).clamp(0, 1), inverse, dsize=(h, w),
                            mode="bilinear", padding_mode="zeros", align_corners=True)
    return target_frames + (image - target_frames) * alpha


def frames_to_tensor(frames: np.ndarray | list[np.ndarray], device: str = "cuda") -> torch.Tensor:
    """``(H, W, C)`` or a list of them (uint8) -> ``(B, C, H, W)`` float32 in ``[0, 255]``."""
    import torch

    batch = np.stack(frames) if isinstance(frames, list) else np.asarray(frames)[None]
    return torch.from_numpy(np.ascontiguousarray(batch)).to(device).permute(0, 3, 1, 2).float()


def tensor_to_frames(tensor: torch.Tensor) -> np.ndarray:
    """``(B, C, H, W)`` float in ``[0, 255]`` -> ``(B, H, W, C)`` uint8."""
    return tensor.clamp(0, 255).round().to("cpu").permute(0, 2, 3, 1).numpy().astype(np.uint8)
