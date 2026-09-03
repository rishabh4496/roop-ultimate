"""Multi-shot ArcFace reference construction and pose-aware identity lookup.

This module is deliberately detector/session agnostic.  The caller supplies
the faces produced by the already-initialised buffalo/ArcFace analyser, so a
folder upload never creates a second GPU model context.  It also makes the
identity-bank contract usable by the API, legacy UI and command-line callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ARCFACE_DIMENSION = 512
DEFAULT_MIN_COSINE = 0.65


def _field(face: Any, name: str, default=None):
    if isinstance(face, dict):
        return face.get(name, default)
    return getattr(face, name, default)


def normalized_arcface_embedding(face: Any) -> Optional[np.ndarray]:
    """Return a finite, unit-length 512-D ArcFace embedding or ``None``."""
    value = _field(face, "normed_embedding")
    if value is None:
        value = _field(face, "embedding")
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size != ARCFACE_DIMENSION or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 1e-8 else None


def face_pose(face: Any) -> Tuple[float, float]:
    """Estimate yaw/pitch once while ingesting a reference image.

    A detector may already supply these values.  Otherwise the project's own
    five-landmark solve is used; failures are represented by the neutral pose,
    which keeps a valid identity usable instead of silently dropping it.
    """
    yaw, pitch = _field(face, "yaw"), _field(face, "pitch")
    try:
        if yaw is not None and pitch is not None:
            return float(yaw), float(pitch)
    except (TypeError, ValueError):
        pass
    try:
        from roop.face_util import solve_pose_5pt
        result = solve_pose_5pt(_field(face, "kps"))
        if result is not None:
            yaw, pitch = float(result[0]), float(result[1])
            if np.isfinite((yaw, pitch)).all():
                return yaw, pitch
    except Exception:
        pass
    return 0.0, 0.0


def reference_weight(face: Any) -> float:
    """Cheap quality weight; no extra inference or image pass is required."""
    score = _field(face, "det_score", _field(face, "score", 1.0))
    bbox = _field(face, "bbox")
    try:
        confidence = max(0.05, min(1.0, float(score)))
    except (TypeError, ValueError):
        confidence = 1.0
    try:
        box = np.asarray(bbox, dtype=np.float32).reshape(4)
        pixels = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
        # Root area rewards a well-resolved face without allowing one 4K photo
        # to completely suppress all pose coverage.
        return confidence * max(0.25, min(4.0, np.sqrt(pixels) / 160.0))
    except (TypeError, ValueError):
        return confidence


@dataclass(frozen=True)
class ReferenceSample:
    face: Any
    image: Any
    path: str
    embedding: np.ndarray
    yaw: float
    pitch: float
    weight: float


@dataclass
class ClusteredReferences:
    samples: List[ReferenceSample]
    embedding: np.ndarray
    rejected: List[dict]

    @property
    def poses(self) -> List[Tuple[float, float]]:
        return [(sample.yaw, sample.pitch) for sample in self.samples]

    @property
    def embeddings(self) -> List[np.ndarray]:
        return [sample.embedding for sample in self.samples]

    def embedding_for_pose(self, yaw: float, pitch: float, blend: bool = True) -> np.ndarray:
        """Return the nearest reference identity, blending an ambiguous pair.

        Blending only happens for two nearby pose samples; a profile therefore
        cannot be diluted by a frontal reference merely because it is present
        in the same faceset.
        """
        distances = np.asarray([
            (float(yaw) - item.yaw) ** 2 + (float(pitch) - item.pitch) ** 2
            for item in self.samples
        ], dtype=np.float32)
        if distances.size == 0:
            return self.embedding
        order = np.argsort(distances)
        first = int(order[0])
        if not blend or len(order) < 2:
            return self.samples[first].embedding
        second = int(order[1])
        # At a pose boundary a weighted blend avoids a frame-to-frame source
        # discontinuity.  Else select the closest true reference exactly.
        if float(distances[second] - distances[first]) > 100.0:
            return self.samples[first].embedding
        weights = np.asarray([
            self.samples[first].weight / (5.0 + float(distances[first])),
            self.samples[second].weight / (5.0 + float(distances[second])),
        ], dtype=np.float32)
        vector = np.average(np.stack((self.samples[first].embedding,
                                      self.samples[second].embedding)),
                            axis=0, weights=weights)
        norm = float(np.linalg.norm(vector))
        return (vector / norm).astype(np.float32) if norm > 1e-8 else self.embedding


def cluster_references(samples: Iterable[ReferenceSample],
                       min_cosine: float = DEFAULT_MIN_COSINE) -> ClusteredReferences:
    """Discard identity outliers and calculate an L2-normalized weighted mean."""
    candidates = list(samples)
    if not candidates:
        raise ValueError("no valid 512-D ArcFace references were detected")
    initial = np.mean(np.stack([item.embedding for item in candidates]), axis=0)
    initial /= max(float(np.linalg.norm(initial)), 1e-8)
    kept, rejected = [], []
    for item in candidates:
        similarity = float(np.dot(initial, item.embedding))
        if similarity < float(min_cosine):
            rejected.append({"path": item.path, "reason": "identity_outlier",
                             "cosine_similarity": round(similarity, 6),
                             "threshold": float(min_cosine)})
        else:
            kept.append(item)
    # A malformed set should never become an empty face identity.  One valid
    # reference is still preferable to an unexplained upload failure.
    if not kept:
        kept, rejected = candidates, []
    weights = np.asarray([item.weight for item in kept], dtype=np.float32)
    centroid = np.average(np.stack([item.embedding for item in kept]), axis=0,
                          weights=weights)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
    return ClusteredReferences(kept, centroid.astype(np.float32), rejected)


def clustered_faceset(samples: Sequence[Tuple[Any, Any, str]],
                      min_cosine: float = DEFAULT_MIN_COSINE) -> ClusteredReferences:
    """Build a cluster from ``(detected_face, original_bgr_image, path)`` data."""
    valid = []
    for face, image, path in samples:
        embedding = normalized_arcface_embedding(face)
        if embedding is None:
            continue
        yaw, pitch = face_pose(face)
        valid.append(ReferenceSample(face, image, str(path), embedding, yaw, pitch,
                                     reference_weight(face)))
    return cluster_references(valid, min_cosine=min_cosine)


__all__ = ["ARCFACE_DIMENSION", "DEFAULT_MIN_COSINE", "ReferenceSample",
           "ClusteredReferences", "normalized_arcface_embedding", "face_pose",
           "cluster_references", "clustered_faceset"]
