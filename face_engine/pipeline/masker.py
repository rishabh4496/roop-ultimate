"""Tri-layer composite face mask: box x occlusion (XSeg) x semantic regions (BiSeNet).

``M = Feather(Box) * XSeg * Regions * Valid``, computed in the swap crop
(``crop_size``, default 256 on the ``arcface_128`` template) and inverse-warped
to a full-frame canvas mask. HIGH (1.0) means "take the swapped pixel".

Layer A — box
    A rectangle with per-side padding (fractions of the crop side), Gaussian
    feathered. The feather is kept inside the padded rectangle, so the mask is
    exactly 0 at the crop border and a paste never shows the crop's edge.

Layer B — XSeg occlusion
    ``xseg.onnx`` takes the 256px crop as NHWC BGR in ``[0, 1]`` and outputs
    the probability of **visible face** — measured 2026-09-27: with a mask
    texture pasted over the mouth of the six insightface sample faces, the
    mouth band read 0.00 and the eyes 0.75-1.00 (uncovered mouth 0.64-1.00).
    The map is therefore used as-is, NOT inverted: inverting it would keep
    the occluder and drop the face. Values below ``threshold`` are zeroed.
    TensorRT FP16 vs CUDA FP32 differ by 0.0004-0.003 mean on those crops.

Layer C — BiSeNet regions
    BiSeNet-ResNet34 (19 CelebAMask-HQ classes) runs on its OWN 512px crop cut
    from the frame on the ``ffhq_512`` template — the whole head, as in its
    training data — rather than an upscaled copy of the tight swap crop. The
    selected classes are mapped into the swap crop through the two
    alignment matrices. Normalization: RGB, ImageNet mean/std.
    Left/right eye and eyebrow labels are NOT reliable: on the sample faces
    class 4 (``l_eye``) sometimes covered both eyes and class 5 was absent.
    Select both sides together unless you have checked your footage.

Every step is guarded: a degenerate landmark fit, a face outside the frame,
or a failing model never raises out of :meth:`CompositeMasker.generate`; the
result carries ``status`` and ``failed_layers`` instead.
"""
from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from face_engine.core.execution import ExecutionEngine, ManagedSession
from face_engine.pipeline.aligner import (
    AlignedFace,
    _exact_fp32,
    align_face,
    compose_affine_cuda,
    crop_valid_mask_cuda,
    invert_affine,
    invert_affine_cuda,
    paste_mask_to_canvas,
    similarity_is_valid,
    similarity_matrices_cuda,
    warp_face_cuda,
)
from face_engine.pipeline.detector import Face, as_bgr

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

XSEG_SIZE = 256
PARSER_SIZE = 512
PARSER_TEMPLATE = "ffhq_512"
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


class FaceRegion(str, Enum):
    """CelebAMask-HQ classes (BiSeNet output channels)."""

    BACKGROUND = "background"
    SKIN = "skin"
    LEFT_EYEBROW = "left_eyebrow"
    RIGHT_EYEBROW = "right_eyebrow"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    GLASSES = "glasses"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    EARRING = "earring"
    NOSE = "nose"
    INNER_MOUTH = "inner_mouth"
    UPPER_LIP = "upper_lip"
    LOWER_LIP = "lower_lip"
    NECK = "neck"
    NECKLACE = "necklace"
    CLOTH = "cloth"
    HAIR = "hair"
    HAT = "hat"


REGION_CLASS: dict[FaceRegion, int] = {region: index for index, region in enumerate(FaceRegion)}

DEFAULT_REGIONS: frozenset[FaceRegion] = frozenset({
    FaceRegion.SKIN, FaceRegion.NOSE, FaceRegion.LEFT_EYE, FaceRegion.RIGHT_EYE,
    FaceRegion.LEFT_EYEBROW, FaceRegion.RIGHT_EYEBROW, FaceRegion.UPPER_LIP,
    FaceRegion.LOWER_LIP, FaceRegion.INNER_MOUTH,
})


class BoxMaskConfig(BaseModel):
    """Layer A. Paddings and blur are fractions of the crop side."""

    model_config = ConfigDict(frozen=True)

    padding_top: float = Field(default=0.0, ge=0.0, le=0.5)
    padding_bottom: float = Field(default=0.0, ge=0.0, le=0.5)
    padding_left: float = Field(default=0.0, ge=0.0, le=0.5)
    padding_right: float = Field(default=0.0, ge=0.0, le=0.5)
    blur: float = Field(default=0.3, ge=0.0, le=1.0)


class XSegConfig(BaseModel):
    """Layer B."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    threshold: float = Field(default=0.1, ge=0.0, lt=1.0)
    feather_sigma: float = Field(default=1.0, ge=0.0)  # px at 256


class RegionConfig(BaseModel):
    """Layer C."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    regions: frozenset[FaceRegion] = DEFAULT_REGIONS
    feather_sigma: float = Field(default=5.0, ge=0.0)  # px at 512, blurred inward


class MaskerConfig(BaseModel):
    """Composite masker settings."""

    model_config = ConfigDict(frozen=True)

    crop_size: int = Field(default=256, ge=32)
    template: str | None = None  # None -> canonical for crop_size
    box: BoxMaskConfig = Field(default_factory=BoxMaskConfig)
    xseg: XSegConfig = Field(default_factory=XSegConfig)
    regions: RegionConfig = Field(default_factory=RegionConfig)
    concurrent: bool = True


@dataclass
class MaskResult:
    """Output of :meth:`CompositeMasker.generate`.

    Attributes:
        crop_mask: ``(S, S)`` float32 composite in the swap crop.
        canvas_mask: ``(H, W)`` float32 composite in frame space.
        aligned: The swap crop and its matrix (None if alignment failed).
        layers: Each layer that ran, in crop space (``box``, ``xseg``,
            ``regions``, ``valid``).
        labels: ``(512, 512)`` BiSeNet class map on the parser crop, if run.
        status: ``"ok"``, ``"degraded"`` (a layer failed and was skipped), or
            ``"no_face"`` (unusable input; both masks are zero).
        failed_layers: Names of layers that raised.
    """

    crop_mask: np.ndarray
    canvas_mask: np.ndarray
    aligned: AlignedFace | None
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    labels: np.ndarray | None = None
    status: str = "ok"
    failed_layers: tuple[str, ...] = ()

    def as_tensor(self, device: str = "cuda") -> torch.Tensor:
        """``crop_mask`` as a ``(1, 1, S, S)`` float32 tensor."""
        import torch

        return torch.from_numpy(self.crop_mask)[None, None].to(device)


def box_mask(crop_size: int, config: BoxMaskConfig) -> np.ndarray:
    """Layer A: padded, feathered rectangle, exactly 0 on the crop border."""
    s = crop_size
    blur_px = config.blur * 0.5 * s
    # Keep the feather inside the rectangle: inset every side by the blur
    # reach so the blurred edge has decayed to ~0 at the padded boundary.
    inset = int(round(blur_px * 0.5))
    top = max(int(round(config.padding_top * s)), inset, 1)
    bottom = max(int(round(config.padding_bottom * s)), inset, 1)
    left = max(int(round(config.padding_left * s)), inset, 1)
    right = max(int(round(config.padding_right * s)), inset, 1)
    mask = np.zeros((s, s), dtype=np.float32)
    if top + bottom < s and left + right < s:
        mask[top:s - bottom, left:s - right] = 1.0
    if blur_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_px * 0.25)
    mask[[0, -1], :] = 0.0
    mask[:, [0, -1]] = 0.0
    return np.clip(mask, 0.0, 1.0)


class CompositeMasker:
    """Builds the three layers for one face and composites them.

    Args:
        engine: Session provider for both models.
        xseg_path: ``xseg.onnx``; None disables Layer B.
        bisenet_path: ``bisenet_resnet_34.onnx``; None disables Layer C.
        config: :class:`MaskerConfig`.

    Thread-safe for concurrent :meth:`generate` calls (ORT sessions are).
    Use as a context manager, or call :meth:`close`, to stop the worker pool.
    """

    def __init__(self, engine: ExecutionEngine, xseg_path: Path | str | None = None,
                 bisenet_path: Path | str | None = None,
                 config: MaskerConfig | None = None) -> None:
        self.engine = engine
        self.config = config or MaskerConfig()
        self.xseg_path = xseg_path
        self.bisenet_path = bisenet_path
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face-mask") \
            if self.config.concurrent else None
        self._box_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ sessions
    def _session(self, path: Path | str | None) -> ManagedSession | None:
        return None if path is None else self.engine.get_session(path)

    @property
    def xseg_enabled(self) -> bool:
        return self.config.xseg.enabled and self.xseg_path is not None

    @property
    def regions_enabled(self) -> bool:
        return (self.config.regions.enabled and self.bisenet_path is not None
                and bool(self.config.regions.regions))

    # ------------------------------------------------------------------ layers
    def box(self, crop_size: int | None = None) -> np.ndarray:
        size = crop_size or self.config.crop_size
        if size not in self._box_cache:
            self._box_cache[size] = box_mask(size, self.config.box)
        return self._box_cache[size]

    def xseg(self, crop: np.ndarray) -> np.ndarray:
        """Layer B on a swap crop of any size; returns visible-face probability at its size."""
        handle = self._session(self.xseg_path)
        if handle is None:
            raise RuntimeError("XSeg model not configured")
        size = crop.shape[0]
        x = crop if size == XSEG_SIZE else cv2.resize(
            crop, (XSEG_SIZE, XSEG_SIZE),
            interpolation=cv2.INTER_AREA if size > XSEG_SIZE else cv2.INTER_CUBIC)
        blob = (x.astype(np.float32) / 255.0)[None]
        out = np.asarray(handle.run({handle.input_names[0]: blob})[0]).reshape(XSEG_SIZE, XSEG_SIZE)
        prob = np.clip(out.astype(np.float32), 0.0, 1.0)
        prob[prob < self.config.xseg.threshold] = 0.0
        if self.config.xseg.feather_sigma > 0:
            prob = cv2.GaussianBlur(prob, (0, 0), self.config.xseg.feather_sigma)
        if size != XSEG_SIZE:
            prob = cv2.resize(prob, (size, size), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob, 0.0, 1.0)

    def parse(self, parser_crop: np.ndarray) -> np.ndarray:
        """BiSeNet class map ``(512, 512)`` int64 for a 512px ``ffhq_512`` crop."""
        handle = self._session(self.bisenet_path)
        if handle is None:
            raise RuntimeError("BiSeNet model not configured")
        x = parser_crop if parser_crop.shape[0] == PARSER_SIZE else cv2.resize(
            parser_crop, (PARSER_SIZE, PARSER_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = x[:, :, ::-1].astype(np.float32)
        blob = ((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
        logits = handle.run({handle.input_names[0]: np.ascontiguousarray(blob, np.float32)})[0]
        return np.asarray(logits)[0].argmax(0)

    def region_mask_from_labels(self, labels: np.ndarray) -> np.ndarray:
        """Selected classes as a soft ``(512, 512)`` mask, feathered inward."""
        ids = [REGION_CLASS[r] for r in self.config.regions.regions]
        mask = np.isin(labels, ids).astype(np.float32)
        sigma = self.config.regions.feather_sigma
        if sigma > 0:
            # Blur, then keep only the inner half of the ramp: the edge moves
            # inward by ~1 sigma and softens, so a paste never bleeds past the
            # parsed boundary into hair or background.
            mask = cv2.GaussianBlur(mask, (0, 0), sigma)
            mask = (np.clip(mask, 0.5, 1.0) - 0.5) * 2.0
        return mask

    # ------------------------------------------------------------------ composite
    def generate(self, frame: np.ndarray, face: Face | np.ndarray) -> MaskResult:
        """All layers for one face; never raises for bad frames or landmarks."""
        size = self.config.crop_size
        bgr = as_bgr(frame)
        kps = face.kps if isinstance(face, Face) else np.asarray(face, dtype=np.float32)
        h, w = (bgr.shape[:2] if bgr is not None else
                (frame.shape[:2] if isinstance(frame, np.ndarray) and frame.ndim >= 2 else (0, 0)))
        empty = MaskResult(crop_mask=np.zeros((size, size), np.float32),
                           canvas_mask=np.zeros((h, w), np.float32), aligned=None,
                           status="no_face")
        if bgr is None or kps.shape != (5, 2) or not np.all(np.isfinite(kps)):
            return empty
        aligned = align_face(bgr, kps, size, self.config.template)
        if aligned is None:
            return empty

        failed: list[str] = []
        layers: dict[str, np.ndarray] = {"box": self.box(size), "valid": aligned.valid}
        labels: np.ndarray | None = None

        jobs: dict[str, Future[Any] | None] = {}
        if self.xseg_enabled:
            jobs["xseg"] = self._submit(self.xseg, aligned.crop)
        if self.regions_enabled:
            jobs["regions"] = self._submit(self._regions_in_crop, bgr, kps, aligned)
        for name, job in jobs.items():
            try:
                value = job.result() if job is not None else None
            except Exception as exc:  # noqa: BLE001 - a failed layer degrades, never raises
                logger.warning("mask layer %s failed: %s", name, exc)
                failed.append(name)
                continue
            if name == "regions":
                layers["regions"], labels = value
            else:
                layers[name] = value

        composite = np.ones((size, size), dtype=np.float32)
        for mask in layers.values():
            composite *= mask
        composite = np.clip(composite, 0.0, 1.0)
        canvas = paste_mask_to_canvas(composite, aligned.matrix, (h, w))
        return MaskResult(crop_mask=composite, canvas_mask=canvas, aligned=aligned,
                          layers=layers, labels=labels,
                          status="degraded" if failed else "ok", failed_layers=tuple(failed))

    def _submit(self, fn: Any, *args: Any) -> Future[Any]:
        if self._pool is not None:
            return self._pool.submit(fn, *args)
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    def _regions_in_crop(self, frame: np.ndarray, kps: np.ndarray,
                         aligned: AlignedFace) -> tuple[np.ndarray, np.ndarray]:
        """Parse a whole-head ffhq_512 crop and map the region mask into the swap crop."""
        parser = align_face(frame, kps, PARSER_SIZE, PARSER_TEMPLATE)
        if parser is None:
            raise RuntimeError("parser crop alignment failed")
        labels = self.parse(parser.crop)
        region = self.region_mask_from_labels(labels) * parser.valid
        # parser crop -> frame -> swap crop
        m_parser_inv = np.vstack([invert_affine(parser.matrix), [0.0, 0.0, 1.0]])
        m_swap = np.vstack([aligned.matrix, [0.0, 0.0, 1.0]])
        to_swap = (m_swap @ m_parser_inv)[:2]
        size = aligned.crop_size
        mapped = cv2.warpAffine(region, to_swap, (size, size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return np.clip(mapped, 0.0, 1.0), labels

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> CompositeMasker:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------- CUDA-resident
@dataclass
class GPUMaskResult:
    """Output of :meth:`GPUMasker.generate`; every tensor stays on the frames' device.

    Attributes:
        mask: ``(N, 1, S, S)`` float32 composite alpha, ready for
            :func:`~face_engine.pipeline.aligner.warp_face_inverse_cuda`.
        crops: ``(N, 3, S, S)`` the aligned swap crops the mask belongs to.
        matrices: ``(N, 2, 3)`` frame -> crop affines.
        layers: Each layer that ran, ``(N, 1, S, S)`` (``box`` is ``(1, 1, S, S)``).
        labels: ``(N, 512, 512)`` int64 BiSeNet class map, if run.
        failed_layers: Names of layers that raised (their factor was skipped).
    """

    mask: Any
    crops: Any
    matrices: Any
    layers: dict[str, Any] = field(default_factory=dict)
    labels: Any = None
    failed_layers: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return "degraded" if self.failed_layers else "ok"


class GPUMasker:
    """The tri-layer composite of :class:`CompositeMasker`, computed in VRAM.

    ``mask = box x xseg x regions x valid``, batched over faces:

    * **box** — the same padded, feathered rectangle, built once per crop
      size, feathered with a cached separable Gaussian (``kornia.filters``);
    * **xseg** — the 256px crop through XSeg (``xseg_3.onnx`` by default)
      via ``run_binding``. Thresholded, NOT inverted: XSeg outputs the
      probability of visible face (see the module docstring and
      ``test_gpu_xseg_keeps_face_and_drops_the_occluder``);
    * **regions** — BiSeNet on each face's own ``ffhq_512`` crop, the selected
      classes (default: skin, brows, eyes, nose, lips, inner mouth; never hair,
      background, glasses, hat...) feathered inward and mapped into the swap
      crop through both matrices;
    * **valid** — 0 where the crop samples outside the frame.

    Frames are ``(B, 3, H, W)`` BGR ``[0, 255]`` tensors; ``frame_index`` says
    which frame each face is on when ``B > 1``. Nothing is copied to the host.
    """

    def __init__(self, engine: ExecutionEngine, xseg_path: Path | str | None = None,
                 bisenet_path: Path | str | None = None,
                 config: MaskerConfig | None = None) -> None:
        self.engine = engine
        self.config = config or MaskerConfig()
        self.xseg_path = xseg_path
        self.bisenet_path = bisenet_path
        self._box_cache: dict[tuple[int, str], Any] = {}
        self._class_ids: dict[str, Any] = {}
        self._constants: dict[str, Any] = {}

    @property
    def xseg_enabled(self) -> bool:
        return self.config.xseg.enabled and self.xseg_path is not None

    @property
    def regions_enabled(self) -> bool:
        return (self.config.regions.enabled and self.bisenet_path is not None
                and bool(self.config.regions.regions))

    # ------------------------------------------------------------------ layers
    def box(self, crop_size: int, device: Any) -> torch.Tensor:
        """Layer A ``(1, 1, S, S)``: padded rectangle, Gaussian feathered, 0 on the border."""
        import torch
        key = (crop_size, str(device))
        if key not in self._box_cache:
            c, s = self.config.box, crop_size
            blur_px = c.blur * 0.5 * s
            inset = int(round(blur_px * 0.5))
            top = max(int(round(c.padding_top * s)), inset, 1)
            bottom = max(int(round(c.padding_bottom * s)), inset, 1)
            left = max(int(round(c.padding_left * s)), inset, 1)
            right = max(int(round(c.padding_right * s)), inset, 1)
            mask = torch.zeros((1, 1, s, s), device=device)
            if top + bottom < s and left + right < s:
                mask[..., top:s - bottom, left:s - right] = 1.0
            if blur_px > 0:
                mask = _gaussian(mask, blur_px * 0.25)
            mask[..., [0, -1], :] = 0.0
            mask[..., :, [0, -1]] = 0.0
            self._box_cache[key] = mask.clamp(0, 1)
        return self._box_cache[key]

    def xseg(self, crops: torch.Tensor) -> torch.Tensor:
        """Layer B on ``(N, 3, 256, 256)`` BGR crops -> visible-face probability ``(N, 1, 256, 256)``."""
        import torch

        if self.xseg_path is None:
            raise RuntimeError("XSeg model not configured")
        handle = self.engine.get_session(self.xseg_path)
        blob = (crops.permute(0, 2, 3, 1) / 255.0).contiguous()  # NHWC, BGR, [0, 1]
        out = handle.run_binding({handle.input_names[0]: blob})[handle.output_names[0]]
        prob = out.reshape(crops.shape[0], 1, XSEG_SIZE, XSEG_SIZE).float().clamp(0, 1)
        prob = torch.where(prob < self.config.xseg.threshold, torch.zeros_like(prob), prob)
        if self.config.xseg.feather_sigma > 0:
            prob = _gaussian(prob, self.config.xseg.feather_sigma)
        return prob.clamp(0, 1)

    def parse(self, parser_crops: torch.Tensor) -> torch.Tensor:
        """BiSeNet class maps ``(N, 512, 512)`` int64 for ``ffhq_512`` BGR crops."""
        import torch

        if self.bisenet_path is None:
            raise RuntimeError("BiSeNet model not configured")
        handle = self.engine.get_session(self.bisenet_path)
        n = parser_crops.shape[0]
        key = f"imagenet:{parser_crops.device}"
        if key not in self._constants:
            self._constants[key] = (
                torch.as_tensor(_IMAGENET_MEAN, device=parser_crops.device).view(1, 3, 1, 1),
                torch.as_tensor(_IMAGENET_STD, device=parser_crops.device).view(1, 3, 1, 1))
        mean, std = self._constants[key]
        blob = ((parser_crops.flip(1) - mean) / std).contiguous()  # RGB, ImageNet
        name = handle.output_names[0]
        logits = handle.run_binding(
            {handle.input_names[0]: blob},
            output_shapes={name: (n, len(FaceRegion), PARSER_SIZE, PARSER_SIZE)},
            unreturned_outputs=handle.output_names[1:])[name]
        return logits.argmax(1)

    def region_mask_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Selected classes as a soft ``(N, 1, 512, 512)`` mask, feathered inward."""
        import torch

        key = str(labels.device)
        if key not in self._class_ids:
            self._class_ids[key] = torch.as_tensor(
                sorted(REGION_CLASS[r] for r in self.config.regions.regions), device=labels.device)
        mask = torch.isin(labels, self._class_ids[key]).float()[:, None]
        sigma = self.config.regions.feather_sigma
        if sigma > 0:
            mask = _gaussian(mask, sigma)
            mask = (mask.clamp(0.5, 1.0) - 0.5) * 2.0
        return mask

    # ------------------------------------------------------------------ composite
    def generate(self, frames: torch.Tensor, kps: torch.Tensor, *,
                 frame_index: torch.Tensor | None = None) -> GPUMaskResult:
        """Crops, matrices and composite masks for ``(N, 5, 2)`` landmarks.

        A failing model layer is skipped (its factor is 1) and reported in
        ``failed_layers``, as on the CPU path; faces with a degenerate
        landmark fit get an all-zero mask.
        """
        from kornia.geometry.transform import warp_affine

        f = frames if frames.ndim == 4 else frames[None]
        f = f if f.is_floating_point() else f.float()
        size = self.config.crop_size
        h, w = f.shape[-2:]
        kps = kps.to(device=f.device, dtype=f.dtype)
        n = kps.shape[0]
        if n == 0:
            empty = f.new_zeros((0, 1, size, size))
            return GPUMaskResult(empty, f.new_zeros((0, 3, size, size)), f.new_zeros((0, 2, 3)))
        matrices = similarity_matrices_cuda(kps, size, self.config.template)
        ok = similarity_is_valid(matrices)
        crops = warp_face_cuda(f, matrices, size, frame_index=frame_index)
        layers: dict[str, Any] = {"box": self.box(size, f.device),
                                  "valid": crop_valid_mask_cuda((h, w), matrices, size)
                                  * ok.float().view(-1, 1, 1, 1)}
        labels = None
        failed: list[str] = []
        if self.xseg_enabled:
            try:
                if size == XSEG_SIZE:
                    xseg_crops = crops
                else:  # same framing, resampled from the frame at 256
                    xseg_crops = warp_face_cuda(f, matrices * (XSEG_SIZE / size), XSEG_SIZE,
                                                frame_index=frame_index)
                prob = self.xseg(xseg_crops)
                if size != XSEG_SIZE:
                    import torch.nn.functional as F

                    prob = F.interpolate(prob, size=(size, size), mode="bilinear",
                                         align_corners=False)
                layers["xseg"] = prob
            except Exception as exc:  # noqa: BLE001 - a failed layer degrades, never raises
                logger.warning("mask layer xseg failed: %s", exc)
                failed.append("xseg")
        if self.regions_enabled:
            try:
                parser_m = similarity_matrices_cuda(kps, PARSER_SIZE, PARSER_TEMPLATE)
                parser_crops = warp_face_cuda(f, parser_m, PARSER_SIZE, frame_index=frame_index)
                labels = self.parse(parser_crops)
                region = (self.region_mask_from_labels(labels)
                          * crop_valid_mask_cuda((h, w), parser_m, PARSER_SIZE))
                to_swap = compose_affine_cuda(matrices, invert_affine_cuda(parser_m))
                with _exact_fp32():
                    layers["regions"] = warp_affine(region, to_swap, dsize=(size, size),
                                                    mode="bilinear", padding_mode="zeros",
                                                    align_corners=True).clamp(0, 1)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mask layer regions failed: %s", exc)
                failed.append("regions")
        composite = layers["box"].expand(n, -1, -1, -1).clone()
        for name, layer in layers.items():
            if name != "box":
                composite = composite * layer
        return GPUMaskResult(composite.clamp(0, 1), crops, matrices, layers, labels, tuple(failed))


_GAUSSIAN_KERNELS: dict[tuple[float, str], Any] = {}


def _gaussian(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur sized like ``cv2.GaussianBlur`` on float images
    (4-sigma radius) with reflect-101 borders, so the GPU layers match the CPU
    ones. The kernel is cached per sigma: ``kornia.filters.gaussian_blur2d``
    rebuilds it from host values on every call, a copy and a sync each time.
    """
    import torch
    from kornia.filters import filter2d_separable, get_gaussian_kernel1d

    key = (float(sigma), str(image.device))
    kernel = _GAUSSIAN_KERNELS.get(key)
    if kernel is None:
        radius = max(1, int(round(4.0 * sigma)))
        kernel = get_gaussian_kernel1d(2 * radius + 1, float(sigma), device=image.device,
                                       dtype=torch.float32)
        _GAUSSIAN_KERNELS[key] = kernel
    return filter2d_separable(image, kernel, kernel, border_type="reflect")
