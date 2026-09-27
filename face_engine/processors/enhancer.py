"""Generative face restoration: GPEN-BFR 512/1024/2048 and RestoreFormer++.

Pipeline (per face)
-------------------
1. Cut an ``ffhq_512``-template crop at the model's native size from the
   frame that already holds the swapped face. (Re-warping the tighter swap
   crop would run off its border: the FFHQ template frames the whole head.)
   Lanczos when the crop upsamples the face.
2. RGB, ``[-1, 1]``; run the network; back to BGR uint8.
3. Output guard: non-finite or collapsed output (a flat face, std < 2
   levels — the failure GFPGAN showed under FP16 in roop-ultimate) is
   rejected and the input kept (``status="rejected"``).
4. Colour stabilisation against a reference crop (see :class:`ColorMode`).
5. ``Enhanced = (1 - alpha) * input + alpha * restored``, pasted back with a
   feathered box mask (optionally multiplied by a caller mask).

Precision
---------
GPEN-1024/2048 overflow under TensorRT FP16 (NaN / black faces) and
RestoreFormer++ has no FP16 quality record, so those sessions are FP32;
GPEN-512 is FP16-safe (roop-ultimate ``precision_policy``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from face_engine.core.execution import ExecutionEngine, ManagedSession
from face_engine.pipeline.aligner import (
    AlignmentError,
    crop_valid_mask,
    estimate_similarity_transform,
    matrix_scale,
    template_points,
    warp_face_by_translation,
    warp_face_inverse,
)
from face_engine.pipeline.detector import Face, as_bgr
from face_engine.pipeline.masker import BoxMaskConfig, box_mask

logger = logging.getLogger(__name__)

TEMPLATE = "ffhq_512"
COLLAPSE_STD = 2.0  # levels; a restored face flatter than this is a failed pass


@dataclass(frozen=True)
class EnhancerSpec:
    zoo_name: str
    size: int
    fp16_safe: bool


ENHANCER_MODELS: dict[str, EnhancerSpec] = {
    "gpen_bfr_512": EnhancerSpec("gpen_bfr_512", 512, True),
    "gpen_bfr_1024": EnhancerSpec("gpen_bfr_1024", 1024, False),
    "gpen_bfr_2048": EnhancerSpec("gpen_bfr_2048", 2048, False),
    "restoreformer_plus_plus": EnhancerSpec("restoreformer_plus_plus", 512, False),
}


class ColorMode(str, Enum):
    """How the restored face's colour is tied back to a reference.

    NONE        the network's colour as-is.
    LAB_MEAN    shift the LAB channel means to the reference's (face region).
    REINHARD    LAB mean AND standard deviation (Reinhard et al. 2001). Matching
                L's spread also scales back texture contrast the restorer added.
    KEEP_CHROMA take L from the restored face and a/b from the reference: all
                restored luminance detail, none of its colour cast.
    """

    NONE = "none"
    LAB_MEAN = "lab_mean"
    REINHARD = "reinhard"
    KEEP_CHROMA = "keep_chroma"


def face_region_mask(size: int, grow: float = 1.0) -> np.ndarray:
    """Ellipse over the face in an ``ffhq_512`` crop, uint8 0/255.

    ``grow`` scales the axes (1.0 = cheeks-to-chin interior used for colour
    statistics; the paste mask uses a larger, feathered one).
    """
    pts = template_points(size, TEMPLATE)
    center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()) + 0.02 * size)
    axes = (int(0.26 * size * grow), int(0.34 * size * grow))
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (int(center[0]), int(center[1])), axes, 0, 0, 360, 255, -1)
    return mask


def paste_mask(size: int, blur: float) -> np.ndarray:
    """Feathered face ellipse x feathered box, float32 in [0, 1].

    The FFHQ template frames the whole head, so the crop routinely holds a
    NEIGHBOUR's face when two people are close; restoring and pasting the
    whole square altered that person too (seen on a two-person frame,
    2026-09-27). The ellipse keeps the paste on the target face.
    """
    ellipse = face_region_mask(size, grow=1.2).astype(np.float32) / 255.0
    ellipse = cv2.GaussianBlur(ellipse, (0, 0), 0.03 * size)
    return ellipse * box_mask(size, BoxMaskConfig(blur=blur))


def transfer_color(image: np.ndarray, reference: np.ndarray, mode: ColorMode,
                   region: np.ndarray | None = None) -> np.ndarray:
    """Match ``image``'s colour to ``reference`` in LAB (both uint8 BGR, same size).

    Statistics are taken inside ``region`` (uint8/bool mask) when given; the
    correction is applied to every pixel.
    """
    if mode is ColorMode.NONE:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = np.ones(image.shape[:2], bool) if region is None else np.asarray(region) > 0
    if sel.sum() < 16:
        return image
    if mode is ColorMode.KEEP_CHROMA:
        lab[:, :, 1:] = ref[:, :, 1:]
    else:
        for c in range(3):
            mu_i, mu_r = lab[:, :, c][sel].mean(), ref[:, :, c][sel].mean()
            if mode is ColorMode.REINHARD:
                sd_i, sd_r = lab[:, :, c][sel].std(), ref[:, :, c][sel].std()
                lab[:, :, c] = (lab[:, :, c] - mu_i) * (sd_r / max(sd_i, 1e-3)) + mu_r
            else:
                lab[:, :, c] += mu_r - mu_i
    return cv2.cvtColor(np.clip(np.rint(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


@dataclass(frozen=True)
class EnhanceResult:
    """Attributes:
        frame: The output frame (a copy; unchanged input if rejected).
        crop_in: The ffhq crop fed to the network.
        crop_out: Restored, colour-corrected crop (before alpha).
        matrix: frame -> crop affine.
        status: ``"ok"``, ``"rejected"`` (bad network output) or ``"no_face"``.
    """

    frame: np.ndarray
    crop_in: np.ndarray | None
    crop_out: np.ndarray | None
    matrix: np.ndarray | None
    status: str


class FaceEnhancer:
    """One restoration model.

    Args:
        engine: Session provider.
        model: An :data:`ENHANCER_MODELS` key.
        model_path: ONNX file.
        color: Default colour mode (:class:`ColorMode`).
        blur: Feather of the paste-back box (fraction of the crop, as in
            :class:`~face_engine.pipeline.masker.BoxMaskConfig`).
    """

    def __init__(self, engine: ExecutionEngine, model: str, model_path: Path | str,
                 color: ColorMode = ColorMode.LAB_MEAN, blur: float = 0.3) -> None:
        if model not in ENHANCER_MODELS:
            raise KeyError(f"unknown enhancer {model!r}; known: {sorted(ENHANCER_MODELS)}")
        self.spec = ENHANCER_MODELS[model]
        self.name = model
        self.engine = engine
        self.model_path = Path(model_path)
        self.color = color
        self._paste_mask = paste_mask(self.spec.size, blur)
        self._region = face_region_mask(self.spec.size)

    @property
    def size(self) -> int:
        return self.spec.size

    @property
    def session(self) -> ManagedSession:
        return self.engine.get_session(self.model_path,
                                       trt_fp16=None if self.spec.fp16_safe else False)

    def crop(self, frame: np.ndarray, face: Face | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(crop, matrix)`` on the ``ffhq_512`` template at the model's size."""
        kps = face.kps if isinstance(face, Face) else np.asarray(face, np.float64)
        matrix = estimate_similarity_transform(kps, template_points(self.size, TEMPLATE))
        interpolation = cv2.INTER_LANCZOS4 if matrix_scale(matrix) > 1.0 else cv2.INTER_LINEAR
        return warp_face_by_translation(frame, matrix, self.size,
                                        interpolation=interpolation), matrix

    def restore(self, crop: np.ndarray) -> np.ndarray | None:
        """Network pass on a native-size crop; None if the output is unusable."""
        blob = crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
        handle = self.session
        out = handle.session.run(None, {handle.input_names[0]: np.ascontiguousarray(blob[None])})[0]
        img = np.asarray(out, np.float32).reshape(3, self.size, self.size)
        if not np.all(np.isfinite(img)):
            logger.warning("%s returned non-finite output; keeping the input", self.name)
            return None
        img = np.clip(img, -1.0, 1.0).transpose(1, 2, 0)[:, :, ::-1]
        restored = np.rint((img + 1.0) * 127.5).astype(np.uint8)
        if restored[self._region > 0].std() < COLLAPSE_STD:
            logger.warning("%s returned a collapsed (flat) face; keeping the input", self.name)
            return None
        return restored

    def enhance(self, frame: np.ndarray, face: Face | np.ndarray, alpha: float = 1.0,
                reference_frame: np.ndarray | None = None, color: ColorMode | None = None,
                mask: np.ndarray | None = None) -> EnhanceResult:
        """Restore one face in ``frame``.

        Args:
            frame: Frame holding the (swapped) face.
            face: Its landmarks.
            alpha: ``(1 - alpha) * input + alpha * restored``, in ``[0, 1]``.
            reference_frame: Colour reference, normally the ORIGINAL target
                frame; defaults to ``frame``.
            color: Override the colour mode.
            mask: Extra ``(S, S)`` paste mask in this enhancer's crop space.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        bgr = as_bgr(frame)
        if bgr is None:
            return EnhanceResult(frame, None, None, None, "no_face")
        try:
            crop_in, matrix = self.crop(bgr, face)
        except (AlignmentError, ValueError):
            return EnhanceResult(bgr.copy(), None, None, None, "no_face")
        valid = crop_valid_mask(bgr.shape, matrix, self.size)
        if valid.max() == 0:
            return EnhanceResult(bgr.copy(), crop_in, None, matrix, "no_face")
        restored = self.restore(crop_in)
        if restored is None:
            return EnhanceResult(bgr.copy(), crop_in, None, matrix, "rejected")
        ref_frame = as_bgr(reference_frame) if reference_frame is not None else bgr
        reference = crop_in if ref_frame is bgr else self.crop(ref_frame, face)[0]
        region = ((self._region > 0) & (valid > 0.99)).astype(np.uint8)
        restored = transfer_color(restored, reference, color or self.color, region)
        blended = restored if alpha == 1.0 else cv2.addWeighted(restored, alpha, crop_in,
                                                               1.0 - alpha, 0.0)
        paste = self._paste_mask * valid
        if mask is not None:
            paste = paste * cv2.resize(np.asarray(mask, np.float32), (self.size, self.size))
        out = warp_face_inverse(bgr, blended, matrix, paste)
        return EnhanceResult(out, crop_in, restored, matrix, "ok")
