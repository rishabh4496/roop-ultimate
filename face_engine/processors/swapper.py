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
    "alphaface_256": SwapModelSpec("alphaface_256", 256, "arcface_128", (0.5, 0.5, 0.5),
                                   (0.5, 0.5, 0.5), True, "normed", available=False),
}


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

