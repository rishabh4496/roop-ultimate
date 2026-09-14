"""Streaming face pre-pass with temporal reuse and compact RAM output.

The production renderer already has several temporal helpers, but they are
coupled to ``ProcessMgr`` and to the InsightFace object model.  This module is
the small, explicit contract needed by a memory-streaming processor:

* only detector/key-frame images are resized and sent to the detector;
* non-key frames coast from a track prediction instead of re-running SCRFD;
* detector calls can be supplied as one batch without requiring a particular
  face-detector implementation;
* 5-point landmarks, track ids, and alignment matrices are stored as compact
  contiguous arrays rather than serialized per-frame files.

The detector and optional landmark/embedding functions are dependency-injected
because InsightFace exports differ between model versions.  ``from_roop``
provides the repository's normal detector as a safe default while keeping this
module usable in a small benchmark process.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import math
import os
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .env import env_bool

# OFF here, deliberately. This prepass is the general-purpose path and must keep
# working against a batch-one detector; strict mode RAISES rather than falling
# back (see _detect_batch below). `roop.vectorized_pipeline` reads the same
# ROOP_OPT_STRICT_TRT flag with STRICT_TRT_DEFAULT = True because it is the
# strict-only pipeline. Both defaults are named so the divergence is explicit;
# tests/test_env_flags.py pins that they are declared, not inlined.
STRICT_TRT_DEFAULT = False


Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
UInt8Array = NDArray[np.uint8]

DetectorOne = Callable[[UInt8Array], Sequence[Any]]
DetectorBatch = Callable[[Sequence[UInt8Array]], Sequence[Sequence[Any]]]
LandmarkExtractor = Callable[
    [UInt8Array, Float32Array, Float32Array], Optional[Float32Array]
]
EmbeddingExtractor = Callable[
    [UInt8Array, Float32Array, Float32Array], Optional[Float32Array]
]
MatrixEstimator = Callable[[Float32Array, int, str], Float32Array]


def _is_detector_batch_error(error: BaseException) -> bool:
    """Identify detector errors that are safe to retry with fewer frames."""

    message = str(error).lower()
    return any(
        token in message
        for token in (
            "out of memory",
            "cudaerror",
            "cudnn_status_alloc_failed",
            "shape",
            "dimension",
            "invalid argument",
        )
    )


def _env_int(name: str, default: int) -> int:
    """Read a positive integer environment value without raising in startup."""

    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def _env_float(name: str, default: float) -> float:
    """Read a finite floating-point environment value."""

    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    return value if math.isfinite(value) else float(default)


@dataclass(frozen=True)
class PrepassConfig:
    """Hardware-safe temporal pre-pass settings.

    ``detector_batch_size`` is a ceiling, not a promise.  The detector adapter
    may only support batch one and the main processor's VRAM governor may lower
    it further.  The defaults mirror the repository's two hardware tiers:
    one context and a small batch on sub-7 GB GPUs, bounded multi-item work on
    the 12 GB desktop card.
    """

    detection_interval: int = 5
    max_detection_dimension: int = 640
    scene_cut_threshold: float = 0.40
    max_track_gap: int = 4
    min_iou: float = 0.10
    max_faces: int = 32
    alignment_size: int = 128
    alignment_template: str = "arcface"
    detector_batch_size: int = 8
    min_detection_score: float = 0.0
    enable_scene_cuts: bool = True
    strict_trt: bool = False

    @classmethod
    def from_environment(cls) -> "PrepassConfig":
        """Resolve settings while preserving the sub-7 GB safety profile."""

        batch = _env_int("ROOP_OPT_PREPASS_BATCH", 8)
        strict_trt = env_bool("ROOP_OPT_STRICT_TRT", STRICT_TRT_DEFAULT)
        try:
            import torch

            if torch.cuda.is_available():
                total_gb = float(
                    torch.cuda.get_device_properties(0).total_memory
                ) / float(1024 ** 3)
                if total_gb < 7.0:
                    batch = 1
        except Exception as _degrade_error:
            # CPU-only environments can still use detector batching when their
            # adapter supports it; the memory governor handles the GPU case.
            _swallowed("roop/optimized_prepass.py:135", _degrade_error, "fallback continued")
            pass

        if strict_trt:
            batch = max(2, batch)

        return cls(
            detection_interval=_env_int("ROOP_OPT_DETECT_INTERVAL", 5),
            max_detection_dimension=_env_int("ROOP_OPT_DETECT_DIM", 640),
            scene_cut_threshold=_env_float("ROOP_OPT_SCENE_CUT", 0.40),
            max_track_gap=_env_int("ROOP_OPT_TRACK_GAP", 4),
            min_iou=min(1.0, max(0.0, _env_float("ROOP_OPT_TRACK_IOU", 0.10))),
            max_faces=_env_int("ROOP_OPT_MAX_FACES", 32),
            alignment_size=_env_int("ROOP_OPT_ALIGNMENT_SIZE", 128),
            alignment_template=os.environ.get(
                "ROOP_OPT_ALIGNMENT_TEMPLATE", "arcface"
            ),
            detector_batch_size=batch,
            min_detection_score=_env_float("ROOP_OPT_DET_SCORE", 0.0),
            enable_scene_cuts=os.environ.get("ROOP_OPT_SCENE_CUT_ENABLED", "1")
            .strip()
            .lower()
            not in ("0", "false", "no", "off"),
            strict_trt=strict_trt,
        )


def _field(face: Any, name: str, default: Any = None) -> Any:
    """Read an InsightFace ``Face`` or a dictionary without side effects."""

    if isinstance(face, dict):
        return face.get(name, default)
    try:
        return getattr(face, name, default)
    except Exception as _degrade_error:
        _swallowed("roop/optimized_prepass.py:171", _degrade_error, "fallback continued")
        return default


def _as_float_array(value: Any, shape: Optional[Tuple[int, ...]] = None) -> Optional[Float32Array]:
    """Return a finite, owned float32 array or ``None``."""

    if value is None:
        return None
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    except (TypeError, ValueError):
        return None
    if shape is not None and array.shape != shape:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array.copy()


def _normalise_bbox(face: Any) -> Optional[Float32Array]:
    """Extract a valid ``x1,y1,x2,y2`` box."""

    box = _as_float_array(_field(face, "bbox"))
    if box is None or box.size != 4:
        return None
    box = box.reshape(4)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _fallback_landmarks(box: Float32Array) -> Float32Array:
    """Create a deterministic five-point fallback for incomplete adapters.

    Normal InsightFace detections already contain ``kps``.  This fallback only
    keeps a detector that returns boxes usable; it is never used when kps are
    present and therefore does not alter the normal alignment contract.
    """

    x1, y1, x2, y2 = [float(value) for value in box]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return np.asarray(
        [
            [x1 + width * 0.32, y1 + height * 0.38],
            [x1 + width * 0.68, y1 + height * 0.38],
            [x1 + width * 0.50, y1 + height * 0.55],
            [x1 + width * 0.37, y1 + height * 0.72],
            [x1 + width * 0.63, y1 + height * 0.72],
        ],
        dtype=np.float32,
    )


def _normalise_landmarks(face: Any, box: Float32Array) -> Float32Array:
    """Extract five keypoints, preferring the detector's exact ``kps`` field."""

    for name in ("kps", "landmark_2d_5", "landmarks_5"):
        points = _as_float_array(_field(face, name))
        if points is not None and points.size == 10:
            return np.ascontiguousarray(points.reshape(5, 2), dtype=np.float32)
    return _fallback_landmarks(box)


def _normalise_embedding(face: Any) -> Optional[Float32Array]:
    """Read and L2-normalize a cached identity vector, if one exists."""

    value = _field(face, "normed_embedding")
    if value is None:
        value = _field(face, "embedding")
    embedding = _as_float_array(value)
    if embedding is None or embedding.ndim != 1 or embedding.size == 0:
        return None
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-7:
        return None
    return np.ascontiguousarray(embedding / norm, dtype=np.float32)


def _score(face: Any) -> float:
    """Read the detector confidence using InsightFace's field variants."""

    value = _field(face, "det_score", _field(face, "score", 1.0))
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = 1.0
    return result if math.isfinite(result) else 0.0


def _iou(first: Float32Array, second: Float32Array) -> float:
    """Compute continuous-coordinate intersection over union."""

    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-7 else 0.0


def _cosine_distance(first: Optional[Float32Array], second: Optional[Float32Array]) -> float:
    """Return a bounded cosine distance, or one when embeddings are absent."""

    if first is None or second is None or first.shape != second.shape:
        return 1.0
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-7:
        return 1.0
    similarity = float(np.dot(first, second) / denominator)
    return 1.0 - max(-1.0, min(1.0, similarity))


def _histogram_signature(frame: UInt8Array) -> Float32Array:
    """Build the same small BGR histogram used for inexpensive cut detection."""

    height, width = frame.shape[:2]
    if max(height, width) > 160:
        small_width = 160
        small_height = max(1, int(round(height * small_width / width)))
        frame = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
    histogram = cv2.calcHist(
        [frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(histogram, histogram, 1.0, 0.0, cv2.NORM_L1)
    return np.ascontiguousarray(histogram.reshape(-1), dtype=np.float32)


def _histogram_difference(first: Optional[Float32Array], second: Optional[Float32Array]) -> float:
    """Return Bhattacharyya distance in ``[0, 1]``."""

    if first is None or second is None:
        return 1.0
    try:
        return float(
            cv2.compareHist(
                first.reshape(-1, 1), second.reshape(-1, 1), cv2.HISTCMP_BHATTACHARYYA
            )
        )
    except Exception as _degrade_error:
        _swallowed("roop/optimized_prepass.py:320", _degrade_error, "fallback continued")
        return 1.0


def _resize_for_detection(frame: UInt8Array, maximum: int) -> UInt8Array:
    """Downscale only the detector input and leave the source frame untouched."""

    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= maximum:
        return frame
    scale = float(maximum) / float(longest)
    resized = cv2.resize(
        frame,
        (max(16, int(round(width * scale))), max(16, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized, dtype=np.uint8)


def _restore_detection_coordinates(
    face: Any,
    original: UInt8Array,
    detection_input: UInt8Array,
) -> Dict[str, Any]:
    """Map a detector result from the resized image back to source pixels."""

    source_height, source_width = original.shape[:2]
    detect_height, detect_width = detection_input.shape[:2]
    scale_x = float(source_width) / float(max(1, detect_width))
    scale_y = float(source_height) / float(max(1, detect_height))
    result: Dict[str, Any] = {
        "bbox": _field(face, "bbox"),
        "kps": _field(face, "kps"),
        "landmark_2d_5": _field(face, "landmark_2d_5"),
        "landmarks_5": _field(face, "landmarks_5"),
        "det_score": _field(face, "det_score", _field(face, "score", 1.0)),
        "embedding": _field(face, "embedding"),
        "normed_embedding": _field(face, "normed_embedding"),
    }
    bbox = result.get("bbox")
    if bbox is not None:
        values = np.asarray(bbox, dtype=np.float32).reshape(-1).copy()
        if values.size >= 4:
            values[[0, 2]] *= scale_x
            values[[1, 3]] *= scale_y
            result["bbox"] = values[:4]
    for name in ("kps", "landmark_2d_5", "landmarks_5"):
        points = result.get(name)
        if points is None:
            continue
        values = np.asarray(points, dtype=np.float32).copy()
        if values.size >= 2 and values.shape[-1] == 2:
            values[..., 0] *= scale_x
            values[..., 1] *= scale_y
            result[name] = values
    return result


def _estimate_similarity(points: Float32Array, size: int, template: str) -> Float32Array:
    """Use the repository's exact alignment fit, with a dependency-safe fallback."""

    points = np.ascontiguousarray(points.reshape(5, 2), dtype=np.float32)
    try:
        from roop.face_util import estimate_norm

        matrix = estimate_norm(points, int(size), template)
        return np.ascontiguousarray(np.asarray(matrix, dtype=np.float32).reshape(2, 3))
    except Exception:
        # The fallback is a least-squares similarity fit with the standard
        # ArcFace five-point template.  It is only reached when this module is
        # used outside the repository's app environment.
        if template != "arcface":
            raise RuntimeError(
                "exact alignment template is unavailable outside roop.face_util"
            )
        destination = np.asarray(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float64,
        )
        destination *= float(size) / 112.0
        source = points.astype(np.float64)
        source_mean = source.mean(axis=0)
        destination_mean = destination.mean(axis=0)
        source_centered = source - source_mean
        destination_centered = destination - destination_mean
        covariance = source_centered.T @ destination_centered
        left, singular, right_t = np.linalg.svd(covariance)
        rotation = left @ right_t
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            singular[-1] *= -1.0
            rotation = left @ right_t
        denominator = float(np.sum(source_centered ** 2))
        scale = float(np.sum(singular) / max(denominator, 1e-9))
        linear = scale * rotation.T
        translation = destination_mean - source_mean @ linear
        return np.asarray(
            [
                [linear[0, 0], linear[0, 1], translation[0]],
                [linear[1, 0], linear[1, 1], translation[1]],
            ],
            dtype=np.float32,
        )


@dataclass
class FaceObservation:
    """One compact, compositor-ready face record."""

    track_id: int
    bbox: Float32Array
    landmarks: Float32Array
    matrix: Float32Array
    score: float
    embedding: Optional[Float32Array] = None
    detected: bool = True

    def __post_init__(self) -> None:
        self.bbox = np.ascontiguousarray(self.bbox, dtype=np.float32).reshape(4).copy()
        self.landmarks = np.ascontiguousarray(self.landmarks, dtype=np.float32).reshape(5, 2).copy()
        self.matrix = np.ascontiguousarray(self.matrix, dtype=np.float32).reshape(2, 3).copy()
        if self.embedding is not None:
            self.embedding = np.ascontiguousarray(self.embedding, dtype=np.float32).reshape(-1).copy()
        self.score = float(self.score)


@dataclass
class FrameAnalysis:
    """Ordered analysis for a single frame, suitable for a compositor callback."""

    frame_index: int
    faces: List[FaceObservation] = field(default_factory=list)
    detection_run: bool = False
    scene_cut: bool = False


@dataclass
class PrepassResult:
    """Contiguous result arrays for a complete scan.

    ``frame_offsets`` indexes all face arrays.  For frame ``i``, faces occupy
    ``[frame_offsets[i], frame_offsets[i + 1])``.  This avoids an object array
    per frame and makes the result cheap to pass to a vectorized consumer.
    """

    frame_indices: Int64Array
    frame_offsets: Int64Array
    boxes: Float32Array
    landmarks: Float32Array
    matrices: Float32Array
    track_ids: Int32Array
    scores: Float32Array
    detected: BoolArray
    scene_cuts: BoolArray
    embeddings: Optional[Float32Array] = None
    embedding_valid: Optional[BoolArray] = None

    def frame(self, position: int) -> FrameAnalysis:
        """Materialize one frame's compact slice without decoding any image."""

        index = int(position)
        start = int(self.frame_offsets[index])
        end = int(self.frame_offsets[index + 1])
        observations: List[FaceObservation] = []
        for row in range(start, end):
            embedding = None
            if self.embeddings is not None and self.embedding_valid is not None:
                if bool(self.embedding_valid[row]):
                    embedding = self.embeddings[row].copy()
            observations.append(
                FaceObservation(
                    track_id=int(self.track_ids[row]),
                    bbox=self.boxes[row],
                    landmarks=self.landmarks[row],
                    matrix=self.matrices[row],
                    score=float(self.scores[row]),
                    embedding=embedding,
                    detected=bool(self.detected[index]),
                )
            )
        return FrameAnalysis(
            frame_index=int(self.frame_indices[index]),
            faces=observations,
            detection_run=bool(self.detected[index]),
            scene_cut=bool(self.scene_cuts[index]),
        )

    @classmethod
    def from_analyses(cls, analyses: Sequence[FrameAnalysis]) -> "PrepassResult":
        """Pack ordered frame analyses into contiguous NumPy arrays."""

        frame_indices = np.asarray([item.frame_index for item in analyses], dtype=np.int64)
        frame_offsets = np.zeros(len(analyses) + 1, dtype=np.int64)
        observations = [face for item in analyses for face in item.faces]
        for index, item in enumerate(analyses):
            frame_offsets[index + 1] = frame_offsets[index] + len(item.faces)

        if observations:
            boxes = np.ascontiguousarray([face.bbox for face in observations], dtype=np.float32)
            landmarks = np.ascontiguousarray(
                [face.landmarks for face in observations], dtype=np.float32
            )
            matrices = np.ascontiguousarray(
                [face.matrix for face in observations], dtype=np.float32
            )
            track_ids = np.asarray([face.track_id for face in observations], dtype=np.int32)
            scores = np.asarray([face.score for face in observations], dtype=np.float32)
            embedding_sizes = [
                int(face.embedding.size)
                for face in observations
                if face.embedding is not None
            ]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            landmarks = np.zeros((0, 5, 2), dtype=np.float32)
            matrices = np.zeros((0, 2, 3), dtype=np.float32)
            track_ids = np.zeros((0,), dtype=np.int32)
            scores = np.zeros((0,), dtype=np.float32)
            embedding_sizes = []

        embeddings: Optional[Float32Array] = None
        embedding_valid: Optional[BoolArray] = None
        if embedding_sizes and len(set(embedding_sizes)) == 1:
            width = embedding_sizes[0]
            embeddings = np.zeros((len(observations), width), dtype=np.float32)
            embedding_valid = np.zeros((len(observations),), dtype=np.bool_)
            for row, face in enumerate(observations):
                if face.embedding is not None and face.embedding.size == width:
                    embeddings[row] = face.embedding
                    embedding_valid[row] = True

        return cls(
            frame_indices=frame_indices,
            frame_offsets=frame_offsets,
            boxes=boxes,
            landmarks=landmarks,
            matrices=matrices,
            track_ids=track_ids,
            scores=scores,
            detected=np.asarray([item.detection_run for item in analyses], dtype=np.bool_),
            scene_cuts=np.asarray([item.scene_cut for item in analyses], dtype=np.bool_),
            embeddings=embeddings,
            embedding_valid=embedding_valid,
        )

    def to_torch(
        self,
        device: str = "cuda",
        pin_memory: bool = True,
    ) -> Dict[str, Any]:
        """Copy compact trajectory arrays to one device in one transfer each.

        The prepass never stores decoded frames.  A compositor that consumes
        many faces can call this once per scan chunk and reuse the returned
        tensors for all alignment and paste operations.  Pinned staging is
        used for CUDA transfers when requested, which lets the caller overlap
        the copy with the previous TensorRT enqueue on a non-default stream.
        """

        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for GPU prepass tensors") from error
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA prepass tensors requested but CUDA is unavailable")

        def transfer(array: NDArray[Any]) -> Any:
            source = torch.from_numpy(np.ascontiguousarray(array))
            if device.startswith("cuda") and pin_memory:
                source = source.pin_memory()
            return source.to(device=device, non_blocking=bool(pin_memory))

        tensors: Dict[str, Any] = {
            "frame_indices": transfer(self.frame_indices),
            "frame_offsets": transfer(self.frame_offsets),
            "boxes": transfer(self.boxes),
            "landmarks": transfer(self.landmarks),
            "matrices": transfer(self.matrices),
            "track_ids": transfer(self.track_ids),
            "scores": transfer(self.scores),
            "detected": transfer(self.detected),
            "scene_cuts": transfer(self.scene_cuts),
        }
        if self.embeddings is not None:
            tensors["embeddings"] = transfer(self.embeddings)
        if self.embedding_valid is not None:
            tensors["embedding_valid"] = transfer(self.embedding_valid)
        return tensors


class PinnedPrepassCache:
    """Reusable device cache for trajectory and alignment tensors.

    Updating the cache replaces only the arrays whose shape changed.  This is
    useful for ordered video chunks: the detector state remains on the CPU,
    while the compositor sees already-contiguous GPU matrices without a
    per-face allocation or a per-frame serialization roundtrip.
    """

    def __init__(self, device: str = "cuda", pin_memory: bool = True) -> None:
        self.device = str(device)
        self.pin_memory = bool(pin_memory)
        self.tensors: Dict[str, Any] = {}
        self._shapes: Dict[str, Tuple[int, ...]] = {}

    def update(self, result: PrepassResult) -> Mapping[str, Any]:
        """Upload a packed result and return the persistent tensor mapping."""

        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for the prepass device cache") from error
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA prepass cache requested but CUDA is unavailable")

        arrays: Dict[str, Optional[NDArray[Any]]] = {
            "frame_indices": result.frame_indices,
            "frame_offsets": result.frame_offsets,
            "boxes": result.boxes,
            "landmarks": result.landmarks,
            "matrices": result.matrices,
            "track_ids": result.track_ids,
            "scores": result.scores,
            "detected": result.detected,
            "scene_cuts": result.scene_cuts,
            "embeddings": result.embeddings,
            "embedding_valid": result.embedding_valid,
        }
        for name, array in arrays.items():
            if array is None:
                self.tensors.pop(name, None)
                self._shapes.pop(name, None)
                continue
            contiguous = np.ascontiguousarray(array)
            shape = tuple(int(item) for item in contiguous.shape)
            if self._shapes.get(name) == shape and name in self.tensors:
                target = self.tensors[name]
                source = torch.from_numpy(contiguous)
                if self.device.startswith("cuda"):
                    if self.pin_memory:
                        source = source.pin_memory()
                    target.copy_(source, non_blocking=self.pin_memory)
                else:
                    target.copy_(source)
                continue
            source = torch.from_numpy(contiguous)
            if self.device.startswith("cuda"):
                if self.pin_memory:
                    source = source.pin_memory()
                target = source.to(self.device, non_blocking=self.pin_memory)
            else:
                target = source.to(self.device)
            self.tensors[name] = target
            self._shapes[name] = shape
        return dict(self.tensors)

    def clear(self) -> None:
        """Release cached device references before a new video or model tier."""

        self.tensors.clear()
        self._shapes.clear()


class BatchedTensorDetector:
    """Adapter that connects a GPU TensorRT runner to :class:`FacePrepass`.

    ``preprocess`` must return one CUDA tensor per input frame and
    ``postprocess`` may perform model-specific decode/NMS.  The adapter does
    not call the runner once per face: it pads only at the runner boundary and
    returns one detection sequence per original frame.
    """

    def __init__(
        self,
        runner: Any,
        input_name: str,
        preprocess: Callable[[Sequence[UInt8Array]], Any],
        postprocess: Callable[[Sequence[Any], Sequence[UInt8Array]], Sequence[Sequence[Any]]],
        batch_size: int = 8,
    ) -> None:
        self.runner = runner
        self.input_name = str(input_name)
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.batch_size = max(1, int(batch_size))

    def detect_batch(self, frames: Sequence[UInt8Array]) -> List[Sequence[Any]]:
        """Run one dynamic-batch inference for a detector frame group."""

        if not frames:
            return []
        inputs = self.preprocess(frames)
        shape = getattr(inputs, "shape", None)
        if shape is None or len(shape) == 0 or int(shape[0]) != len(frames):
            raise ValueError("detector preprocess must return a batch matching frames")
        run_gpu = getattr(self.runner, "run_gpu", None)
        if not callable(run_gpu):
            raise TypeError("TensorRT detector runner must expose run_gpu")
        outputs = run_gpu(
            {self.input_name: inputs},
            batch_size=self.batch_size,
            pad_to_batch=True,
        )
        decoded = self.postprocess(outputs, frames)
        result = [list(item or []) for item in decoded]
        if len(result) != len(frames):
            raise ValueError("detector postprocess returned the wrong frame count")
        return result


@dataclass
class _Track:
    track_id: int
    bbox: Float32Array
    landmarks: Float32Array
    score: float
    embedding: Optional[Float32Array]
    velocity: Float32Array = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_frame: int = -1
    misses: int = 0
    hits: int = 1

    def predict(self, frame_index: int) -> Float32Array:
        """Predict a box using constant velocity between detector frames."""

        delta = max(0, int(frame_index) - int(self.last_frame))
        if delta <= 0:
            return self.bbox.copy()
        predicted = self.bbox + self.velocity * float(delta)
        if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
            return self.bbox.copy()
        return np.ascontiguousarray(predicted, dtype=np.float32)


class FacePrepass:
    """Stateful temporal detector and alignment cache.

    The class is safe to call from one ordered worker, which is intentional:
    track state is sequential while the expensive detector adapter can still
    receive a batch of key frames.  It never retains full-resolution frames.
    """

    def __init__(
        self,
        detector: Any,
        config: Optional[PrepassConfig] = None,
        landmark_extractor: Optional[LandmarkExtractor] = None,
        embedding_extractor: Optional[EmbeddingExtractor] = None,
        matrix_estimator: Optional[MatrixEstimator] = None,
    ) -> None:
        self.detector = detector
        self.config = config or PrepassConfig.from_environment()
        self.landmark_extractor = landmark_extractor
        self.embedding_extractor = embedding_extractor
        self.matrix_estimator = matrix_estimator or _estimate_similarity
        self._tracks: List[_Track] = []
        self._next_track_id = 0
        self._last_histogram: Optional[Float32Array] = None
        self._last_detection_frame: Optional[int] = None
        self._next_frame_index: Optional[int] = None
        self._lock = RLock()

    @classmethod
    def from_roop(cls, config: Optional[PrepassConfig] = None) -> "FacePrepass":
        """Build a detector adapter for the repository's InsightFace stack."""

        def detect(frame: UInt8Array) -> Sequence[Any]:
            from roop.face_util import get_all_faces

            return get_all_faces(frame) or []

        return cls(detect, config=config)

    def reset(self) -> None:
        """Drop track state before a new video or a non-sequential seek."""

        with self._lock:
            self._tracks.clear()
            self._next_track_id = 0
            self._last_histogram = None
            self._last_detection_frame = None
            self._next_frame_index = None

    def _detect_many(self, frames: Sequence[UInt8Array]) -> List[Sequence[Any]]:
        """Run the adapter in batch when it exposes that contract."""

        if not frames:
            return []
        batch_method = getattr(self.detector, "detect_batch", None)
        if callable(batch_method):
            results: List[Sequence[Any]] = []
            ceiling = max(1, int(self.config.detector_batch_size))
            for start in range(0, len(frames), ceiling):
                group = frames[start : start + ceiling]
                try:
                    result = batch_method(group)
                except (TypeError, ValueError):
                    stacked = np.stack(group, axis=0)
                    result = batch_method(stacked)
                except RuntimeError as error:
                    if len(group) <= 1 or not _is_detector_batch_error(error):
                        raise
                    middle = max(1, len(group) // 2)
                    results.extend(self._detect_many(group[:middle]))
                    results.extend(self._detect_many(group[middle:]))
                    continue
                group_results = [list(item or []) for item in result]
                if len(group_results) != len(group):
                    raise ValueError(
                        "detector.detect_batch returned a result for a different "
                        "number of frames"
                    )
                results.extend(group_results)
            return results

        one_method = getattr(self.detector, "detect", None)
        if callable(one_method):
            if self.config.strict_trt:
                raise RuntimeError(
                    "strict TensorRT prepass requires detector.detect_batch; "
                    "detector.detect would execute a batch-one loop"
                )
            return [list(one_method(frame) or []) for frame in frames]
        if callable(self.detector):
            if self.config.strict_trt:
                raise RuntimeError(
                    "strict TensorRT prepass requires a detector.detect_batch adapter; "
                    "the repository's legacy callable detector is batch-one"
                )
            return [list(self.detector(frame) or []) for frame in frames]
        raise TypeError("detector must be callable or expose detect/detect_batch")

    def _scheduled_positions(
        self,
        frames: Sequence[UInt8Array],
        start_index: int,
    ) -> Tuple[List[int], List[bool]]:
        """Plan detector calls while computing scene-cut state in one pass."""

        positions: List[int] = []
        cuts: List[bool] = []
        planned_last = self._last_detection_frame
        planned_has_tracks = bool(self._tracks)
        previous_hist = self._last_histogram

        for offset, frame in enumerate(frames):
            frame_index = start_index + offset
            histogram = _histogram_signature(frame)
            cut = bool(
                self.config.enable_scene_cuts
                and previous_hist is not None
                and _histogram_difference(previous_hist, histogram)
                > self.config.scene_cut_threshold
            )
            cuts.append(cut)
            previous_hist = histogram
            interval_due = (
                planned_last is None
                or frame_index - int(planned_last) >= self.config.detection_interval
            )
            needs_detection = cut or not planned_has_tracks or interval_due
            if needs_detection:
                positions.append(offset)
                planned_last = frame_index
                # A key frame is expected to seed tracks; if it is empty the
                # actual update will leave the state empty and the next chunk
                # will schedule recovery again.
                planned_has_tracks = True

        self._last_histogram = previous_hist
        return positions, cuts

    def _match_detections(
        self,
        detections: Sequence[Any],
        frame: UInt8Array,
        frame_index: int,
    ) -> List[FaceObservation]:
        """Assign detector results to tracks and return current observations."""

        parsed: List[Tuple[Float32Array, Float32Array, float, Optional[Float32Array]]] = []
        for face in detections:
            bbox = _normalise_bbox(face)
            if bbox is None or _score(face) < self.config.min_detection_score:
                continue
            landmarks = _normalise_landmarks(face, bbox)
            score = _score(face)
            embedding = _normalise_embedding(face)
            if self.landmark_extractor is not None and _field(face, "kps") is None:
                extracted = self.landmark_extractor(frame, bbox, landmarks)
                if extracted is not None and np.asarray(extracted).size == 10:
                    landmarks = np.ascontiguousarray(
                        np.asarray(extracted, dtype=np.float32).reshape(5, 2)
                    )
            if embedding is None and self.embedding_extractor is not None:
                embedding = self.embedding_extractor(frame, bbox, landmarks)
                if embedding is not None:
                    embedding = _normalise_embedding({"embedding": embedding})
            parsed.append((bbox, landmarks, score, embedding))

        parsed.sort(key=lambda item: (-item[2], float(item[0][0]), float(item[0][1])))
        predicted = [track.predict(frame_index) for track in self._tracks]
        candidates: List[Tuple[float, int, int]] = []
        for det_index, (bbox, _landmarks, _score_value, embedding) in enumerate(parsed):
            for track_index, track in enumerate(self._tracks):
                overlap = _iou(bbox, predicted[track_index])
                appearance = _cosine_distance(embedding, track.embedding)
                if overlap < self.config.min_iou and appearance > 0.35:
                    continue
                first_center = (bbox[:2] + bbox[2:]) * 0.5
                second_center = (predicted[track_index][:2] + predicted[track_index][2:]) * 0.5
                scale = max(
                    1.0,
                    math.hypot(
                        float(predicted[track_index][2] - predicted[track_index][0]),
                        float(predicted[track_index][3] - predicted[track_index][1]),
                    ),
                )
                center_cost = min(2.0, float(np.linalg.norm(first_center - second_center)) / scale)
                cost = 0.60 * (1.0 - overlap) + 0.20 * center_cost + 0.20 * appearance
                candidates.append((cost, det_index, track_index))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        assigned_detections: Dict[int, int] = {}
        assigned_tracks: Dict[int, int] = {}
        for cost, det_index, track_index in candidates:
            if det_index in assigned_detections or track_index in assigned_tracks:
                continue
            if cost > 0.95:
                continue
            assigned_detections[det_index] = track_index
            assigned_tracks[track_index] = det_index

        for track_index, track in enumerate(self._tracks):
            if track_index not in assigned_tracks:
                previous_box = track.bbox.copy()
                track.bbox = predicted[track_index]
                delta = track.bbox - previous_box
                track.landmarks = np.ascontiguousarray(
                    track.landmarks + delta[:2], dtype=np.float32
                )
                track.last_frame = frame_index
                track.misses += 1

        observations: List[FaceObservation] = []
        for det_index, (bbox, landmarks, score, embedding) in enumerate(parsed):
            track_index = assigned_detections.get(det_index)
            if track_index is None:
                track = _Track(
                    track_id=self._next_track_id,
                    bbox=bbox.copy(),
                    landmarks=landmarks.copy(),
                    score=score,
                    embedding=None if embedding is None else embedding.copy(),
                    last_frame=frame_index,
                )
                self._next_track_id += 1
                self._tracks.append(track)
            else:
                track = self._tracks[track_index]
                previous_box = track.bbox.copy()
                track.velocity = np.ascontiguousarray(bbox - previous_box, dtype=np.float32)
                track.bbox = bbox.copy()
                track.landmarks = landmarks.copy()
                track.score = score
                if embedding is not None:
                    track.embedding = embedding.copy()
                track.last_frame = frame_index
                track.misses = 0
                track.hits += 1
            observations.append(self._observation(track, frame_index, detected=True))

        survivors: List[_Track] = []
        matched_ids = {observation.track_id for observation in observations}
        for track in self._tracks:
            if track.track_id in matched_ids or track.misses <= self.config.max_track_gap:
                survivors.append(track)
        self._tracks = survivors
        observations.sort(key=lambda item: (float(item.bbox[0]), int(item.track_id)))
        return observations[: self.config.max_faces]

    def _coast(self, frame_index: int) -> List[FaceObservation]:
        """Predict active tracks and reuse their cached landmarks/embeddings."""

        observations: List[FaceObservation] = []
        for track in self._tracks:
            if track.misses >= self.config.max_track_gap:
                continue
            previous_box = track.bbox.copy()
            predicted = track.predict(frame_index)
            delta = predicted - previous_box
            track.bbox = predicted
            track.landmarks = np.ascontiguousarray(
                track.landmarks + delta[:2], dtype=np.float32
            )
            track.last_frame = frame_index
            track.misses += 1
            observations.append(self._observation(track, frame_index, detected=False))
        return sorted(observations, key=lambda item: (float(item.bbox[0]), int(item.track_id)))

    def _observation(self, track: _Track, frame_index: int, detected: bool) -> FaceObservation:
        """Build the exact alignment matrix for a track's current 5 points."""

        matrix = self.matrix_estimator(
            track.landmarks, self.config.alignment_size, self.config.alignment_template
        )
        return FaceObservation(
            track_id=track.track_id,
            bbox=track.bbox,
            landmarks=track.landmarks,
            matrix=matrix,
            score=track.score * (1.0 if detected else 0.96 ** max(1, track.misses)),
            embedding=track.embedding,
            detected=detected,
        )

    def process_batch(
        self,
        frames: Sequence[UInt8Array],
        start_index: Optional[int] = None,
    ) -> List[FrameAnalysis]:
        """Analyze an ordered batch while making only the planned detections."""

        if not frames:
            return []
        with self._lock:
            first_index = (
                int(start_index)
                if start_index is not None
                else int(self._next_frame_index or 0)
            )
            if self._next_frame_index is not None and first_index != self._next_frame_index:
                self.reset()
            positions, cuts = self._scheduled_positions(frames, first_index)
            detection_inputs = [
                _resize_for_detection(frames[position], self.config.max_detection_dimension)
                for position in positions
            ]
            detection_batches = self._detect_many(detection_inputs)
            detection_by_position: Dict[int, List[Dict[str, Any]]] = {}
            for index, position in enumerate(positions):
                detection_by_position[position] = [
                    _restore_detection_coordinates(
                        face, frames[position], detection_inputs[index]
                    )
                    for face in detection_batches[index]
                ]

            analyses: List[FrameAnalysis] = []
            for offset, frame in enumerate(frames):
                frame_index = first_index + offset
                if cuts[offset]:
                    self._tracks.clear()
                if offset in detection_by_position:
                    observations = self._match_detections(
                        detection_by_position[offset], frame, frame_index
                    )
                    self._last_detection_frame = frame_index
                    analysis = FrameAnalysis(
                        frame_index=frame_index,
                        faces=observations,
                        detection_run=True,
                        scene_cut=cuts[offset],
                    )
                else:
                    observations = self._coast(frame_index)
                    analysis = FrameAnalysis(
                        frame_index=frame_index,
                        faces=observations,
                        detection_run=False,
                        scene_cut=cuts[offset],
                    )
                analyses.append(analysis)
            self._next_frame_index = first_index + len(frames)
            return analyses

    def process_frame(
        self,
        frame: UInt8Array,
        frame_index: Optional[int] = None,
    ) -> FrameAnalysis:
        """Single-frame convenience wrapper for existing render loops."""

        return self.process_batch([frame], start_index=frame_index)[0]

    def scan(
        self,
        frames: Iterable[UInt8Array],
        start_index: int = 0,
    ) -> PrepassResult:
        """Scan a frame iterator and return only compact analysis arrays.

        At most ``detector_batch_size * detection_interval`` full frames are
        held while key-frame detection is coalesced.  The returned object holds
        no source image data, so a long clip does not multiply its RAM usage by
        the video duration.
        """

        self.reset()
        analyses: List[FrameAnalysis] = []
        chunk_size = max(1, self.config.detector_batch_size * self.config.detection_interval)
        iterator = iter(frames)
        current_index = int(start_index)
        while True:
            chunk: List[UInt8Array] = []
            for _ in range(chunk_size):
                try:
                    chunk.append(next(iterator))
                except StopIteration:
                    break
            if not chunk:
                break
            analyses.extend(self.process_batch(chunk, start_index=current_index))
            current_index += len(chunk)
        return PrepassResult.from_analyses(analyses)


__all__ = [
    "BatchedTensorDetector",
    "FaceObservation",
    "FacePrepass",
    "FrameAnalysis",
    "PinnedPrepassCache",
    "PrepassConfig",
    "PrepassResult",
]
