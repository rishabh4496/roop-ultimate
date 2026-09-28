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
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from face_engine.processors.color import ColorMode, transfer_color

if TYPE_CHECKING:
    import torch

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

# Default precision of the batched GPU enhancer (TensorRT engines), per model.
# Measured 2026-09-28, RTX 4070, 37 real-clip faces (identity kept = cosine of the
# restored face to the original):
#   gpen_bfr_512             fp32 0.8068   fp16 0.7835 (30 dB vs fp32)  -> fp32
#   restoreformer_plus_plus  fp32 0.8082   fp16 0.8080 (49 dB), original graph -> fp16
#   gpen_bfr_1024 / 2048     fp16 collapses (flat / non-finite faces)          -> fp32
# The HOST FaceEnhancer still builds GPEN-512 with TensorRT FP16 (fp16_safe=True
# in ENHANCER_MODELS); by the numbers above that costs identity. Not changed here.
#
# gpen_bfr_1024 "auto": the AOT FP16 engine when compiled (tools/compile_engines.py
# pins the encoder's final linear and the style pixel-norm to FP32, the only
# layers that leave FP16's range), else fp32. Measured 2026-09-28, RTX 4070, 28
# real faces: 32.7 vs 57.2 ms/face, identity 0.9344 vs 0.9343, PSNR vs fp32
# median 66.7 dB (min 61.4), no non-finite value. ONNX Runtime's own FP16 build
# is still NaN, so "auto" never picks fp16 without the compiled engine.
ENHANCER_PRECISION: dict[str, str] = {name: "fp32" for name in ENHANCER_MODELS}
ENHANCER_PRECISION["restoreformer_plus_plus"] = "fp16"
ENHANCER_PRECISION["gpen_bfr_1024"] = "auto"


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


# ---------------------------------------------------------------------------- CUDA, batched
@dataclass
class BatchedEnhanceResult:
    """Attributes (tensors on the frames' device):
        frames: ``(B, 3, H, W)`` output frames (float BGR ``[0, 255]``).
        crops_in: ``(N, 3, S, S)`` ffhq crops fed to the network.
        crops_out: ``(N, 3, S, S)`` restored + colour-corrected crops (before alpha;
            equal to ``crops_in`` where rejected).
        matrices: ``(N, 2, 3)`` frame -> crop affines.
        ok: ``(N,)`` bool; False = rejected (non-finite or collapsed output),
            nothing pasted for that face.
        paste: ``(crops, matrices, weights)`` still to paste when
            :meth:`BatchedFaceEnhancer.enhance` ran with ``paste_back=False``
            (``frames`` is then the unmodified input); None once pasted.
    """

    frames: Any
    crops_in: Any
    crops_out: Any
    matrices: Any
    ok: Any
    paste: Any = None


class BatchedFaceEnhancer:
    """:class:`FaceEnhancer` over batches of faces, on the GPU.

    Faces run as one batch when the model batches (RestoreFormer++); GPEN's
    StyleGAN2 modulated convolutions fix the batch inside the graph (see
    :mod:`face_engine.utils.onnx_batch`), so GPEN runs one face per
    ``run_binding`` call, still without leaving the device.

    Args:
        precision: ``"fp32"`` / ``"fp16"`` / ``"auto"`` (the AOT FP16 engine when
            compiled, else fp32); default :data:`ENHANCER_PRECISION`.
        batching: False = original graph, one face per call.
    """

    def __init__(self, engine: ExecutionEngine, model: str, model_path: Path | str, *,
                 precision: str | None = None, color: ColorMode = ColorMode.LAB_MEAN,
                 blur: float = 0.3, max_batch: int = 4, batching: bool = True) -> None:
        from face_engine.utils.onnx_batch import batched_model

        if model not in ENHANCER_MODELS:
            raise KeyError(f"unknown enhancer {model!r}; known: {sorted(ENHANCER_MODELS)}")
        self.spec = ENHANCER_MODELS[model]
        self.name = model
        self.engine = engine
        self.precision = precision or ENHANCER_PRECISION.get(model, "fp32")
        if self.precision == "auto":
            from face_engine.core.trt_compiler import aot_available

            self.precision = ("fp16" if batching and aot_available(model_path, "fp16",
                                                                   engine.config)
                              else "fp32")
        if self.precision not in ("fp32", "fp16"):
            raise ValueError("precision must be 'fp32', 'fp16' or 'auto'")
        self.color = color
        self.max_batch = max_batch
        from face_engine.core.trt_compiler import aot_engine

        self.aot = aot_engine(model_path, self.precision, engine.config) if batching else None
        if self.aot is not None:  # a compiled engine (tools/compile_engines.py)
            self.batched = self.aot.max_batch > 1
            self.max_batch = min(max_batch, self.aot.max_batch)
            self.model_path = Path(model_path)
        else:
            # FP16 + a rewritten InstanceNorm (RestoreFormer++) = the original
            # graph, one face per call; see batched_model.
            batched = (batched_model(model_path, fp16=self.precision == "fp16")
                       if batching else None)
            self.batched = batched is not None
            self.model_path = batched or Path(model_path)
        self._paste_np = paste_mask(self.spec.size, blur)
        self._region_np = (face_region_mask(self.spec.size) > 0).astype(np.float32)
        self._constants: dict[str, Any] = {}

    @property
    def size(self) -> int:
        return self.spec.size

    @property
    def session(self) -> Any:
        if self.aot is not None:
            return self.aot
        from face_engine.utils.onnx_batch import batch_shape_profile

        profile = (batch_shape_profile(self.model_path, max_batch=self.max_batch)
                   if self.batched else None)
        return self.engine.get_session(self.model_path, shape_profile=profile,
                                       trt_fp16=self.precision == "fp16")

    def _masks(self, device: Any) -> tuple[Any, Any]:
        import torch

        key = str(device)
        if key not in self._constants:
            self._constants[key] = (torch.as_tensor(self._paste_np, device=device)[None, None],
                                    torch.as_tensor(self._region_np, device=device)[None, None])
        return self._constants[key]

    def restore(self, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Network pass on ``(N, 3, s, s)`` BGR crops of any size.

        Crops are resampled to the native size with GPU bicubic (antialiased
        when shrinking). Returns ``(restored, ok)``: ``(N, 3, S, S)`` BGR float
        and ``(N,)`` bool; a non-finite or collapsed face (region std <
        ``COLLAPSE_STD`` levels, GFPGAN's FP16 failure mode) is ``ok=False`` and
        its output is the (resampled) input.
        """
        import torch

        from face_engine.processors.swapper import _chunks, resample_crops

        size = self.size
        x = resample_crops(crops.float(), size).clamp(0, 255)
        blob = (x.flip(1) / 127.5 - 1.0).contiguous()
        handle = self.session
        first = handle.output_names[0]
        step = self.max_batch if self.batched else 1
        outs = []
        for part in _chunks(blob.shape[0], step):
            b = part.stop - part.start
            outs.append(handle.run_binding({handle.input_names[0]: blob[part]},
                                           output_shapes={first: (b, 3, size, size)},
                                           unreturned_outputs=handle.output_names[1:])[first])
        raw = torch.cat(outs).float()
        finite = torch.isfinite(raw).flatten(1).all(1)
        restored = ((raw.nan_to_num(0.0).clamp(-1, 1) + 1.0) * 127.5).flip(1)
        _, region = self._masks(raw.device)
        weight = region.expand(raw.shape[0], 1, size, size)
        total = weight.sum(dim=(1, 2, 3))
        mean = (restored * weight).sum(dim=(2, 3), keepdim=True) / total.view(-1, 1, 1, 1)
        std = ((((restored - mean) ** 2) * weight).sum(dim=(1, 2, 3))
               / (total * 3)).clamp_min(0).sqrt()
        ok = finite & (std >= COLLAPSE_STD)
        return torch.where(ok.view(-1, 1, 1, 1), restored, x), ok

    def enhance(self, frames: torch.Tensor, kps: torch.Tensor, *,
                reference: torch.Tensor | None = None, frame_index: torch.Tensor | None = None,
                alpha: float = 1.0, color: ColorMode | None = None,
                mask: torch.Tensor | None = None, paste_back: bool = True) -> BatchedEnhanceResult:
        """Restore every face ``kps`` ``(N, 5, 2)`` in ``frames`` ``(B, 3, H, W)``.

        Args:
            reference: Colour reference frames (the ORIGINAL targets), same
                layout as ``frames``; default ``frames``.
            alpha: ``Restored = (1 - alpha) * input + alpha * enhanced``.
            mask: Extra ``(N, 1, s, s)`` paste mask in this enhancer's crop
                space (any ``s``; resized).
            paste_back: False leaves the paste-back to the caller (the stream
                pipeline runs it on its encode stream): ``result.paste`` holds
                what :func:`warp_face_inverse_cuda` would have pasted.
        """
        import torch
        import torch.nn.functional as F

        from face_engine.pipeline.aligner import (
            crop_valid_mask_cuda,
            similarity_is_valid,
            similarity_matrices_cuda,
            warp_face_cuda,
            warp_face_inverse_cuda,
        )
        from face_engine.processors.color import transfer_color_cuda

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        f = frames if frames.ndim == 4 else frames[None]
        f = f.float()
        size = self.size
        matrices = similarity_matrices_cuda(kps.to(f.device, torch.float32), size, TEMPLATE)
        crops_in = warp_face_cuda(f, matrices, size, frame_index=frame_index,
                                  padding_mode="border", antialias=True,
                                  mode="bicubic").clamp(0, 255)
        valid = crop_valid_mask_cuda(f.shape[-2:], matrices, size)
        restored, ok = self.restore(crops_in)
        ok = ok & similarity_is_valid(matrices) & (valid.flatten(1).amax(1) > 0)
        paste, region = self._masks(f.device)
        mode = color or self.color
        if mode is not ColorMode.NONE:
            ref = f if reference is None else (reference if reference.ndim == 4
                                               else reference[None]).float()
            ref_crops = crops_in if reference is None else warp_face_cuda(
                ref, matrices, size, frame_index=frame_index, padding_mode="border",
                antialias=True, mode="bicubic").clamp(0, 255)
            restored = transfer_color_cuda(restored, ref_crops, mode,
                                           region * (valid > 0.99).float())
        restored = torch.where(ok.view(-1, 1, 1, 1), restored, crops_in)
        blended = restored if alpha == 1.0 else alpha * restored + (1.0 - alpha) * crops_in
        weight = paste * valid * ok.float().view(-1, 1, 1, 1)
        if mask is not None:
            weight = weight * F.interpolate(mask.float(), size=(size, size), mode="bilinear",
                                            align_corners=False)
        if not paste_back:
            return BatchedEnhanceResult(f, crops_in, restored, matrices, ok,
                                        paste=(blended, matrices, weight))
        out = warp_face_inverse_cuda(f, blended, matrices, weight, frame_index=frame_index)
        return BatchedEnhanceResult(out, crops_in, restored, matrices, ok)
