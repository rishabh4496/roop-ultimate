"""Identity transfer: HyperSwap 1a/1b/1c and inswapper-128, with Pixel Boost.

Identity
--------
:class:`IdentityEncoder` aligns the source face to ``arcface_112``, feeds it
to ``arcface_w600k_r50`` as RGB ``(x - 127.5) / 127.5`` and L2-normalises the
512-d output. HyperSwap consumes that vector directly; inswapper consumes
``normalize(v @ emap)``, where ``emap`` is the 512x512 projection stored as an
initializer inside ``inswapper_128.onnx``.

Pixel Boost
-----------
The swap networks have fixed inputs (256 or 128). Upsampling a model-sized
crop and feeding the same network cannot add detail, so Pixel Boost follows
FaceFusion / roop-ultimate: the target is cut from the frame at
``boost = k * model_size`` (Lanczos when that crop upsamples the face),
split *polyphase* into ``k*k`` interleaved model-sized sub-images (sub-image
``(a, b)`` holds pixels ``(k*i + a, k*j + b)``), each is swapped with the same
identity, and the outputs are re-interleaved into one ``boost``-sized face.
Every sub-image is a complete face at model resolution, so each pass is
in-distribution; together they carry ``k*k`` times the pixels.

Precision
---------
Swap sessions are built with ``trt_fp16=False``. roop-ultimate measured FP16
overflow in these networks (inswapper "rainbow" smudges under TensorRT FP16;
HyperSwap identity distance 0.352 -> 0.407); an FP32 engine is kept in its
own cache directory.

AlphaFace
---------
Registered but unavailable: no public release of ``alphaface_256.onnx`` was
located, and its I/O contract is unknown, so no spec is guessed for it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from face_engine.core.execution import ExecutionEngine, ManagedSession
from face_engine.core.registry import ModelUnavailableError
from face_engine.pipeline.aligner import (
    AlignmentError,
    estimate_similarity_transform,
    template_points,
    warp_face_by_translation,
    warp_face_inverse,
)
from face_engine.pipeline.detector import Face, as_bgr

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class SwapError(RuntimeError):
    """The swap could not run (bad input, degenerate landmarks, bad model output)."""


@dataclass(frozen=True)
class SwapModelSpec:
    """How to feed and read one swap network.

    Attributes:
        size: Native input/output side.
        template: Alignment template name (:mod:`face_engine.pipeline.aligner`).
        mean, std: Per-channel RGB normalization of the ``[0, 1]`` input.
        denormalize: Output is ``[-1, 1]`` (else ``[0, 1]``).
        embedding: ``"normed"`` (L2-normalised ArcFace) or ``"normed_emap"``.
        available: False when no model file source is known.
    """

    zoo_name: str
    size: int
    template: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    denormalize: bool
    embedding: str
    available: bool = True


SWAP_MODELS: dict[str, SwapModelSpec] = {
    **{f"hyperswap_{v}_256": SwapModelSpec(f"hyperswap_{v}_256", 256, "arcface_128",
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True, "normed")
       for v in ("1a", "1b", "1c")},
    "inswapper_128": SwapModelSpec("inswapper_128", 128, "arcface_128", (0.0, 0.0, 0.0),
                                   (1.0, 1.0, 1.0), False, "normed_emap"),
    # FaceFusion's FP16-weight export: FP32 inputs/outputs, FP32 emap.
    "inswapper_128_fp16": SwapModelSpec("inswapper_128_fp16", 128, "arcface_128",
                                        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), False, "normed_emap"),
    "alphaface_256": SwapModelSpec("alphaface_256", 256, "arcface_128", (0.5, 0.5, 0.5),
                                   (0.5, 0.5, 0.5), True, "normed", available=False),
}

# Default precision of the batched GPU swapper (TensorRT engines), per model.
# Measured 2026-09-28, RTX 4070, 222 real-clip swaps (37 faces from three clips x
# six sources; identity = cosine of the re-detected swapped face to the source):
#   hyperswap_1a_256    fp32 0.5875   fp16 0.5867 on the original graph, but FP16
#                       cannot batch (see onnx_batch.batched_model) and batched
#                       FP32 is faster than unbatched FP16 (bench_stage3)  -> fp32
#   inswapper_128       fp32 0.8406   fp16 0.8406 (0.23 lv), 16.0 -> 5.3 ms/face -> fp16
#   inswapper_128_fp16  fp32 engine 0.8407, fp16 engine 0.8086 (p5 0.38)     -> fp32
# hyperswap_1b/1c were not measured; they keep fp32.
SWAP_PRECISION: dict[str, str] = {name: "fp32" for name in SWAP_MODELS}
SWAP_PRECISION["inswapper_128"] = "fp16"


@dataclass(frozen=True)
class Identity:
    """A source identity.

    Attributes:
        embedding: ``(512,)`` float32, unit L2 norm.
        raw_norm: Norm of the network output before normalization (a rough
            quality signal: blurry or profile crops read lower).
    """

    embedding: np.ndarray
    raw_norm: float

    def similarity(self, other: Identity | np.ndarray) -> float:
        """Cosine similarity with another identity or embedding."""
        vec = other.embedding if isinstance(other, Identity) else np.asarray(other).reshape(-1)
        return float(np.dot(self.embedding, vec / (np.linalg.norm(vec) or 1.0)))


def _kps(face: Face | np.ndarray) -> np.ndarray:
    kps = face.kps if isinstance(face, Face) else np.asarray(face, dtype=np.float64)
    kps = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
    if kps.shape != (5, 2) or not np.all(np.isfinite(kps)):
        raise SwapError(f"need (5, 2) finite landmarks, got {kps.shape}")
    return kps


class IdentityEncoder:
    """ArcFace (w600k R50) 512-d identity embeddings."""

    size = 112
    template = "arcface_112"

    def __init__(self, engine: ExecutionEngine, model_path: Path | str) -> None:
        self.engine = engine
        self.model_path = model_path

    @property
    def session(self) -> ManagedSession:
        return self.engine.get_session(self.model_path)

    def align(self, frame: np.ndarray, face: Face | np.ndarray) -> np.ndarray:
        """112x112 ``arcface_112`` crop."""
        bgr = as_bgr(frame)
        if bgr is None:
            raise SwapError("unusable source frame")
        try:
            matrix = estimate_similarity_transform(_kps(face), template_points(self.size,
                                                                               self.template))
        except AlignmentError as exc:
            raise SwapError(str(exc)) from exc
        return warp_face_by_translation(bgr, matrix, self.size)

    def embed_crop(self, crop: np.ndarray) -> Identity:
        """Identity from an already aligned 112x112 BGR crop."""
        blob = ((crop[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        handle = self.session
        raw = np.asarray(handle.session.run(None, {handle.input_names[0]:
                                                   np.ascontiguousarray(blob)})[0]).reshape(-1)
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm == 0.0:
            raise SwapError("recognizer returned a zero or non-finite embedding")
        return Identity(embedding=(raw / norm).astype(np.float32), raw_norm=norm)

    def embed(self, frame: np.ndarray, face: Face | np.ndarray) -> Identity:
        return self.embed_crop(self.align(frame, face))

    @staticmethod
    def average(identities: list[Identity]) -> Identity:
        """Mean of several identities (e.g. a faceset), re-normalised."""
        if not identities:
            raise ValueError("no identities to average")
        mean = np.mean([i.embedding for i in identities], axis=0)
        norm = float(np.linalg.norm(mean))
        return Identity(embedding=(mean / norm).astype(np.float32),
                        raw_norm=float(np.mean([i.raw_norm for i in identities])))


@dataclass(frozen=True)
class SwapResult:
    """One swapped face.

    Attributes:
        crop: ``(B, B, 3)`` uint8 swapped face at the Pixel Boost size ``B``.
        target_crop: The same crop of the original frame.
        matrix: ``(2, 3)`` frame -> crop affine.
        model_mask: ``(B, B)`` float32 mask the network emitted (HyperSwap), or None.
        pixel_boost: ``B``.
        tiles: Sub-images swapped (``(B / model_size) ** 2``).
    """

    crop: np.ndarray
    target_crop: np.ndarray
    matrix: np.ndarray
    model_mask: np.ndarray | None
    pixel_boost: int
    tiles: int


def implode_pixel_boost(crop: np.ndarray, model_size: int, factor: int) -> np.ndarray:
    """``(k*s, k*s, C)`` -> ``(k*k, s, s, C)`` polyphase sub-images."""
    c = crop.shape[2]
    return (crop.reshape(model_size, factor, model_size, factor, c)
            .transpose(1, 3, 0, 2, 4).reshape(factor * factor, model_size, model_size, c))


def explode_pixel_boost(tiles: np.ndarray, model_size: int, factor: int) -> np.ndarray:
    """Inverse of :func:`implode_pixel_boost`."""
    c = tiles.shape[-1]
    return (tiles.reshape(factor, factor, model_size, model_size, c)
            .transpose(2, 0, 3, 1, 4).reshape(model_size * factor, model_size * factor, c))


class FaceSwapper:
    """Runs one swap model.

    Args:
        engine: Session provider.
        model: A :data:`SWAP_MODELS` key.
        model_path: The model's ONNX file.

    Raises:
        ModelUnavailableError: for a registered model with no known source.
    """

    def __init__(self, engine: ExecutionEngine, model: str, model_path: Path | str) -> None:
        if model not in SWAP_MODELS:
            raise KeyError(f"unknown swap model {model!r}; known: {sorted(SWAP_MODELS)}")
        self.spec = SWAP_MODELS[model]
        if not self.spec.available:
            raise ModelUnavailableError(
                f"{model}: no public model release located and its I/O contract is unknown")
        self.name = model
        self.engine = engine
        self.model_path = Path(model_path)
        self._emap: np.ndarray | None = None
        self._mean = np.asarray(self.spec.mean, np.float32).reshape(3, 1, 1)
        self._std = np.asarray(self.spec.std, np.float32).reshape(3, 1, 1)

    @property
    def session(self) -> ManagedSession:
        return self.engine.get_session(self.model_path, trt_fp16=False)

    # ------------------------------------------------------------------ identity
    @property
    def emap(self) -> np.ndarray:
        """inswapper's 512x512 identity projection (the last 512x512 initializer)."""
        if self._emap is None:
            import onnx
            from onnx import numpy_helper

            graph = onnx.load(str(self.model_path)).graph
            for init in reversed(graph.initializer):
                if tuple(init.dims) == (512, 512):
                    self._emap = numpy_helper.to_array(init).astype(np.float32)
                    break
            else:
                raise SwapError(f"{self.model_path.name} has no 512x512 emap initializer")
        return self._emap

    def latent(self, identity: Identity) -> np.ndarray:
        """``(1, 512)`` float32 source input for this model."""
        vec = identity.embedding.reshape(1, -1).astype(np.float32)
        if self.spec.embedding == "normed_emap":
            vec = vec @ self.emap
            vec = vec / np.linalg.norm(vec)
        return np.ascontiguousarray(vec, dtype=np.float32)

    # ------------------------------------------------------------------ swap
    def _blob(self, crop: np.ndarray) -> np.ndarray:
        rgb = crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(((rgb - self._mean) / self._std)[None], dtype=np.float32)

    def _decode(self, out: np.ndarray) -> np.ndarray:
        img = np.asarray(out, dtype=np.float32).reshape(3, self.spec.size, self.spec.size)
        if self.spec.denormalize:
            img = (img + 1.0) * 0.5
        return img.transpose(1, 2, 0)[:, :, ::-1] * 255.0  # float BGR in [0, 255]

    def run_crop(self, crop: np.ndarray, identity: Identity) -> tuple[np.ndarray, np.ndarray | None]:
        """Swap one model-sized aligned crop. Returns (float BGR image, mask or None)."""
        handle = self.session
        feeds = {}
        for name in handle.input_names:
            feeds[name] = self.latent(identity) if name == "source" else self._blob(crop)
        outputs = handle.session.run(None, feeds)
        image = self._decode(outputs[0])
        if not np.all(np.isfinite(image)):
            raise SwapError(f"{self.name} produced non-finite output")
        mask = None
        if len(outputs) > 1 and np.asarray(outputs[1]).size == self.spec.size ** 2:
            mask = np.clip(np.asarray(outputs[1], np.float32).reshape(self.spec.size,
                                                                     self.spec.size), 0, 1)
        return image, mask

    def swap(self, frame: np.ndarray, target: Face | np.ndarray, identity: Identity,
             pixel_boost: int | None = None, weight: float = 1.0) -> SwapResult:
        """Swap ``identity`` onto the ``target`` face of ``frame``.

        Args:
            pixel_boost: Output crop side; a multiple of the model size
                (256/512/768/1024 for HyperSwap). None = model size.
            weight: ``out = weight * swapped + (1 - weight) * target`` in ``[0, 1]``.

        Raises:
            SwapError: unusable frame or landmarks, or bad model output.
        """
        size = self.spec.size
        boost = pixel_boost or size
        if boost % size or boost < size:
            raise ValueError(f"pixel_boost must be a multiple of {size}, got {boost}")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        bgr = as_bgr(frame)
        if bgr is None:
            raise SwapError("unusable target frame")
        try:
            matrix = estimate_similarity_transform(_kps(target),
                                                   template_points(boost, self.spec.template))
        except AlignmentError as exc:
            raise SwapError(str(exc)) from exc
        scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
        interpolation = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_LINEAR
        target_crop = warp_face_by_translation(bgr, matrix, boost, interpolation=interpolation)
        factor = boost // size
        tiles = implode_pixel_boost(target_crop, size, factor)
        swapped = np.empty(tiles.shape, np.float32)
        masks: list[np.ndarray | None] = []
        for i, tile in enumerate(tiles):
            swapped[i], m = self.run_crop(tile, identity)
            masks.append(m)
        image = explode_pixel_boost(swapped, size, factor)
        if weight < 1.0:
            image = weight * image + (1.0 - weight) * target_crop.astype(np.float32)
        crop = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        model_mask = None
        if all(m is not None for m in masks):
            model_mask = explode_pixel_boost(np.stack(masks)[..., None], size, factor)[..., 0]
        return SwapResult(crop=crop, target_crop=target_crop, matrix=matrix,
                          model_mask=model_mask, pixel_boost=boost, tiles=factor * factor)

    @staticmethod
    def paste(frame: np.ndarray, result: SwapResult, mask: np.ndarray | None = None) -> np.ndarray:
        """Composite a :class:`SwapResult` into a copy of ``frame``.

        ``mask`` may be any square size (e.g. a 256px
        :class:`~face_engine.pipeline.masker.MaskResult` ``crop_mask``); it is
        resized to the boost size.
        """
        b = result.pixel_boost
        if mask is not None and mask.shape[:2] != (b, b):
            mask = cv2.resize(np.asarray(mask, np.float32), (b, b), interpolation=cv2.INTER_LINEAR)
        return warp_face_inverse(frame, result.crop, result.matrix, mask)


# ---------------------------------------------------------------------------- CUDA, batched
MAX_BATCH = 8


def implode_pixel_boost_cuda(crops: torch.Tensor, model_size: int, factor: int) -> torch.Tensor:
    """``(N, C, k*s, k*s)`` -> ``(N*k*k, C, s, s)`` polyphase sub-images (GPU view ops).

    Sub-image ``(a, b)`` of face ``n`` is row ``n*k*k + a*k + b`` and holds
    pixels ``(k*i + a, k*j + b)``: the same layout as :func:`implode_pixel_boost`.
    """
    n, c = crops.shape[:2]
    return (crops.reshape(n, c, model_size, factor, model_size, factor)
            .permute(0, 3, 5, 1, 2, 4).reshape(n * factor * factor, c, model_size, model_size))


def explode_pixel_boost_cuda(tiles: torch.Tensor, model_size: int, factor: int) -> torch.Tensor:
    """Inverse of :func:`implode_pixel_boost_cuda`."""
    c = tiles.shape[1]
    n = tiles.shape[0] // (factor * factor)
    return (tiles.reshape(n, factor, factor, c, model_size, model_size)
            .permute(0, 3, 4, 1, 5, 2).reshape(n, c, model_size * factor, model_size * factor))


def resample_crops(crops: torch.Tensor, size: int) -> torch.Tensor:
    """Resize ``(N, C, S, S)`` crops with GPU bicubic (antialiased when shrinking)."""
    import torch.nn.functional as F

    if crops.shape[-1] == size:
        return crops
    return F.interpolate(crops, size=(size, size), mode="bicubic", align_corners=False,
                         antialias=crops.shape[-1] > size)


def _chunks(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


class GPUIdentityEncoder:
    """ArcFace embeddings for a batch of faces, on the GPU.

    ``embed(frames, kps)`` warps each face to ``arcface_112``, runs the
    batched recognizer and L2-normalises on the device
    (``v / ||v||``). Returns ``(N, 512)`` unit vectors.
    """

    size = 112
    template = "arcface_112"

    def __init__(self, engine: ExecutionEngine, model_path: Path | str,
                 max_batch: int = MAX_BATCH, batching: bool = True) -> None:
        from face_engine.utils.onnx_batch import batched_model

        self.engine = engine
        self.max_batch = max_batch
        batched = batched_model(model_path) if batching else None
        self.model_path = batched or Path(model_path)
        self.batched = batched is not None

    @property
    def session(self) -> ManagedSession:
        from face_engine.utils.onnx_batch import batch_shape_profile

        profile = (batch_shape_profile(self.model_path, max_batch=self.max_batch)
                   if self.batched else None)
        return self.engine.get_session(self.model_path, shape_profile=profile)

    def embed_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """``(N, 3, 112, 112)`` BGR ``[0, 255]`` crops -> ``(N, 512)`` unit embeddings."""
        import torch

        handle = self.session
        blob = ((crops.flip(1) - 127.5) / 127.5).contiguous()  # RGB
        out_name = handle.output_names[0]
        step = self.max_batch if self.batched else 1
        vectors = []
        for part in _chunks(blob.shape[0], step):
            b = part.stop - part.start
            vectors.append(handle.run_binding({handle.input_names[0]: blob[part]},
                                              output_shapes={out_name: (b, 512)})[out_name])
        raw = torch.cat(vectors).float()
        return raw / torch.norm(raw, dim=-1, keepdim=True).clamp_min(1e-12)

    def embed(self, frames: torch.Tensor, kps: torch.Tensor, *,
              frame_index: torch.Tensor | None = None) -> torch.Tensor:
        from face_engine.pipeline.aligner import (
            similarity_matrices_cuda,
            warp_face_cuda,
        )

        matrices = similarity_matrices_cuda(kps, self.size, self.template)
        crops = warp_face_cuda(frames, matrices, self.size, frame_index=frame_index,
                               padding_mode="border", antialias=True)
        return self.embed_crops(crops)


@dataclass
class BatchedSwapResult:
    """Swapped faces, as tensors on the frames' device.

    Attributes:
        crops: ``(N, 3, P, P)`` swapped faces, BGR float ``[0, 255]``, P = Pixel Boost size.
        target_crops: ``(N, 3, P, P)`` the same crops of the original frames.
        matrices: ``(N, 2, 3)`` frame -> crop affines.
        model_mask: ``(N, 1, P, P)`` the network's own mask (HyperSwap) or None.
        ok: ``(N,)`` bool; False where the network output was non-finite (that
            face's crop is its target crop, so a paste changes nothing).
        pixel_boost: ``P``.
    """

    crops: Any
    target_crops: Any
    matrices: Any
    model_mask: Any
    ok: Any
    pixel_boost: int


class BatchedFaceSwapper:
    """One swap model over batches of faces, entirely on the GPU.

    * **Dynamic batching**: the model is used through
      :func:`~face_engine.utils.onnx_batch.batched_model` (a verified
      dynamic-batch rewrite), so ``N`` faces x ``k*k`` Pixel Boost tiles run
      as ``ceil(N*k*k / max_batch)`` inferences.
    * **Identity**: :meth:`set_source` caches the source latent on the device
      (inswapper's ``emap`` projection applied and re-normalised once); it is
      expanded (a view, no copy) to each batch.
    * **Pixel Boost**: the target is cut from the frame at ``P = k * model``
      (bicubic, supersampled when the face is larger than the crop), split
      polyphase into ``k*k`` model-sized faces that ride in the same batch,
      and re-interleaved. See the module docstring for why a resampled crop
      cannot replace cutting from the frame.
    * **Zero copy**: inputs and outputs are bound with ``run_binding``.

    Args:
        precision: ``"fp32"`` or ``"fp16"``; default :data:`SWAP_PRECISION`.
        batching: False = the model's original graph, one face (tile) per
            call: the single-face baseline the benchmark compares against.
    """

    def __init__(self, engine: ExecutionEngine, model: str, model_path: Path | str, *,
                 precision: str | None = None, max_batch: int = MAX_BATCH,
                 batching: bool = True) -> None:
        from face_engine.utils.onnx_batch import batched_model

        if model not in SWAP_MODELS:
            raise KeyError(f"unknown swap model {model!r}; known: {sorted(SWAP_MODELS)}")
        self.spec = SWAP_MODELS[model]
        if not self.spec.available:
            raise ModelUnavailableError(
                f"{model}: no public model release located and its I/O contract is unknown")
        self.name = model
        self.engine = engine
        self.source_path = Path(model_path)
        self.precision = precision or SWAP_PRECISION.get(model, "fp32")
        if self.precision not in ("fp32", "fp16"):
            raise ValueError("precision must be 'fp32' or 'fp16'")
        self.max_batch = max_batch
        # FP16 + a rewritten InstanceNorm (HyperSwap) = the original graph,
        # one face per call; see batched_model.
        batched = (batched_model(self.source_path, fp16=self.precision == "fp16")
                   if batching else None)
        self.batched = batched is not None
        self.model_path = batched or self.source_path
        self._host = FaceSwapper(engine, model, self.source_path)  # emap, spec helpers
        self._source: Any = None
        self._constants: dict[str, Any] = {}

    # ------------------------------------------------------------------ session
    @property
    def session(self) -> ManagedSession:
        from face_engine.utils.onnx_batch import batch_shape_profile

        profile = (batch_shape_profile(self.model_path, max_batch=self.max_batch)
                   if self.batched else None)
        return self.engine.get_session(self.model_path, shape_profile=profile,
                                       trt_fp16=self.precision == "fp16")

    # ------------------------------------------------------------------ identity
    def latent_cuda(self, embeddings: torch.Tensor) -> torch.Tensor:
        """``(N, 512)`` unit ArcFace embeddings -> this model's source input."""
        import torch

        vec = embeddings.float()
        if vec.ndim == 1:
            vec = vec[None]
        if self.spec.embedding == "normed_emap":
            key = f"emap:{vec.device}"
            if key not in self._constants:
                self._constants[key] = torch.as_tensor(self._host.emap, device=vec.device)
            vec = vec @ self._constants[key]
            vec = vec / torch.norm(vec, dim=-1, keepdim=True).clamp_min(1e-12)
        return vec.contiguous()

    def set_source(self, source: torch.Tensor | Identity, device: Any = "cuda") -> torch.Tensor:
        """Cache the source identity on the GPU; returns the ``(1, 512)`` latent."""
        import torch

        emb = source.embedding if isinstance(source, Identity) else source
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32).reshape(1, -1)
        emb = emb / torch.norm(emb, dim=-1, keepdim=True).clamp_min(1e-12)
        self._source = self.latent_cuda(emb)
        return self._source

    # ------------------------------------------------------------------ network
    def _normalize(self, crops: torch.Tensor) -> torch.Tensor:
        import torch

        key = f"norm:{crops.device}"
        if key not in self._constants:
            self._constants[key] = (
                torch.as_tensor(self.spec.mean, device=crops.device).view(1, 3, 1, 1),
                torch.as_tensor(self.spec.std, device=crops.device).view(1, 3, 1, 1))
        mean, std = self._constants[key]
        return ((crops.flip(1) / 255.0 - mean) / std).contiguous()  # RGB

    def run_crops(self, crops: torch.Tensor,
                  latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Swap ``(N, 3, s, s)`` model-sized BGR crops with ``(N or 1, 512)`` latents.

        Returns ``(images, masks)``: BGR float ``[0, 255]`` and ``(N, 1, s, s)``
        masks (None when the model has no mask output).
        """
        import torch

        handle = self.session
        size = self.spec.size
        n = crops.shape[0]
        blob = self._normalize(crops.float())
        lat = latents.expand(n, -1) if latents.shape[0] == 1 else latents
        step = self.max_batch if self.batched else 1
        names = handle.output_names
        images, masks = [], []
        for part in _chunks(n, step):
            b = part.stop - part.start
            feeds = {name: (lat[part].contiguous() if name == "source" else blob[part])
                     for name in handle.input_names}
            shapes = {names[0]: (b, 3, size, size)}
            if len(names) > 1:
                shapes[names[1]] = (b, 1, size, size)
            out = handle.run_binding(feeds, output_shapes=shapes)
            images.append(out[names[0]])
            if len(names) > 1:
                masks.append(out[names[1]])
        image = torch.cat(images).float()
        if self.spec.denormalize:
            image = (image + 1.0) * 0.5
        image = image.flip(1) * 255.0
        mask = torch.cat(masks).float().clamp(0, 1) if masks else None
        return image, mask

    # ------------------------------------------------------------------ swap
    def swap(self, frames: torch.Tensor, kps: torch.Tensor, *,
             source: torch.Tensor | None = None, frame_index: torch.Tensor | None = None,
             pixel_boost: int | None = None, weight: float = 1.0) -> BatchedSwapResult:
        """Swap every face ``kps`` (``(N, 5, 2)``) in ``frames`` (``(B, 3, H, W)``).

        Args:
            source: ``(N or 1, 512)`` unit ArcFace embeddings; default the
                cached :meth:`set_source` identity.
            frame_index: ``(N,)`` frame of each face when ``B > 1``.
            pixel_boost: Output crop side, a multiple of the model size.
            weight: ``weight * swapped + (1 - weight) * target``.
        """
        import torch

        from face_engine.pipeline.aligner import (
            similarity_is_valid,
            similarity_matrices_cuda,
            warp_face_cuda,
        )

        size = self.spec.size
        boost = pixel_boost or size
        if boost % size or boost < size:
            raise ValueError(f"pixel_boost must be a multiple of {size}, got {boost}")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        if source is not None:
            latents = self.latent_cuda(source.to(frames.device))
        elif self._source is not None:
            latents = self._source
        else:
            raise SwapError("no source identity: pass source= or call set_source()")
        f = frames if frames.ndim == 4 else frames[None]
        matrices = similarity_matrices_cuda(kps.to(f.device, torch.float32), boost,
                                            self.spec.template)
        target = warp_face_cuda(f, matrices, boost, frame_index=frame_index,
                                padding_mode="border", antialias=True, mode="bicubic")
        target = target.clamp(0, 255)
        factor = boost // size
        tiles = implode_pixel_boost_cuda(target, size, factor)
        tile_latents = (latents if latents.shape[0] == 1
                        else latents.repeat_interleave(factor * factor, dim=0))
        swapped, mask = self.run_crops(tiles, tile_latents)
        image = explode_pixel_boost_cuda(swapped, size, factor)
        model_mask = explode_pixel_boost_cuda(mask, size, factor) if mask is not None else None
        ok = torch.isfinite(image).flatten(1).all(1) & similarity_is_valid(matrices)
        image = torch.where(ok.view(-1, 1, 1, 1), image.nan_to_num(0.0), target)
        if weight < 1.0:
            image = weight * image + (1.0 - weight) * target
        return BatchedSwapResult(image.clamp(0, 255), target, matrices, model_mask, ok, boost)
