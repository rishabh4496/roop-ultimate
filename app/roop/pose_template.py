"""Pose-projected ArcFace templates for target-face alignment.

The swapper still receives five landmarks in its trained canonical layout. At
high yaw, however, a frontal destination forces both eye and mouth pairs to
occupy frontal widths. This module projects those offsets with the estimated
head pose and, once the profile is visible, gives the near-side silhouette and
chin a gradual vote in the same similarity fit. All functions are CPU-only and
return values; no filter or worker state lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np


def _pose3(pose: Any) -> tuple[float, float, float]:
    try:
        values = np.asarray(pose, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        values = np.zeros(3, dtype=np.float64)
    if not np.isfinite(values).all():
        values = np.zeros(3, dtype=np.float64)
    return tuple(float(np.clip(value, -180.0, 180.0)) for value in values)


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class PoseTemplate:
    """One crop-size-specific template and its optional profile anchors."""

    points: np.ndarray
    silhouette: np.ndarray
    anchor_weights: np.ndarray
    source_anchors: np.ndarray
    yaw: float
    pitch: float
    roll: float
    size: int
    mode: str = "arcface"

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float32).reshape(5, 2).copy()
        silhouette = np.asarray(self.silhouette, dtype=np.float32).reshape(3, 2).copy()
        weights = np.asarray(self.anchor_weights, dtype=np.float32).reshape(3).copy()
        anchors = np.asarray(self.source_anchors, dtype=np.float32).reshape(3, 2).copy()
        for value in (points, silhouette, weights, anchors):
            value.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "silhouette", silhouette)
        object.__setattr__(self, "anchor_weights", weights)
        object.__setattr__(self, "source_anchors", anchors)
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "yaw", float(self.yaw))
        object.__setattr__(self, "pitch", float(self.pitch))
        object.__setattr__(self, "roll", float(self.roll))
        object.__setattr__(self, "mode", str(self.mode or "arcface"))

    @property
    def has_silhouette_anchors(self) -> bool:
        return bool(np.isfinite(self.source_anchors[[0, 2]]).all()
                    and np.all(self.anchor_weights[[0, 2]] > 0.0))

    def alignment_matrix(self, landmarks: Any, *, sar: Optional[float] = None) -> np.ndarray:
        """Fit the target crop to this warped template with a strict similarity.

        The five detector points retain unit weight. Ear and chin constraints
        enter gradually above 35 degrees and cannot introduce shear or
        reflection. Missing dense silhouette landmarks leave the five-point
        fit intact.
        """
        from roop.alignment_safety import landmarks5

        kps = landmarks5(landmarks)
        if self.has_silhouette_anchors:
            from roop.face_analyser import weighted_umeyama_alignment

            src = np.concatenate((kps, self.source_anchors[[0, 2]]), axis=0)
            dst = np.concatenate((self.points, self.silhouette[[0, 2]]), axis=0)
            weights = np.concatenate((np.ones(5, dtype=np.float64),
                                      self.anchor_weights[[0, 2]].astype(np.float64)))
            return weighted_umeyama_alignment(src, dst, weights=weights, sar=sar)

        from roop.face_util import estimate_norm
        return estimate_norm(kps, self.size, self.mode, sar=sar,
                             template_points=self.points)


def warp_template(pose: Any, size: int, mode: str = "arcface", *,
                  source_anchors: Any = None) -> PoseTemplate:
    """Project the destination five-point template by yaw, pitch, and roll.

    X offsets around the eye midpoint shrink by ``cos(yaw)``. The nose and
    silhouette anchors also move along the yaw projection vector, modelling
    their depth. Yaw is bounded below 90 degrees and the projected span has a
    positive floor, so no eye or mouth pair can invert. Roll remains an image
    coordinate rotation handled by the ordinary similarity alignment.
    """
    image_size = int(size)
    if image_size <= 0:
        raise ValueError("template size must be positive")
    from roop.face_util import swap_template_points

    yaw, pitch, roll = _pose3(pose)
    base = np.asarray(swap_template_points(image_size, mode), dtype=np.float64).reshape(5, 2).copy()
    eye_mid = (base[0] + base[1]) * 0.5
    eye_span = max(1e-6, float(abs(base[1, 0] - base[0, 0])))
    yaw_rad = math.radians(float(np.clip(yaw, -85.0, 85.0)))
    pitch_rad = math.radians(float(np.clip(pitch, -80.0, 80.0)))
    x_scale = max(0.12, math.cos(yaw_rad))
    y_scale = max(0.55, math.cos(pitch_rad))
    sin_yaw = math.sin(yaw_rad)

    # A weak-perspective projection: the face's horizontal dimensions shorten,
    # while points closer to the camera (nose, chin) move along the yaw vector.
    projected = base.copy()
    projected[:, 0] = eye_mid[0] + (base[:, 0] - eye_mid[0]) * x_scale
    projected[:, 1] = eye_mid[1] + (base[:, 1] - eye_mid[1]) * y_scale
    projected[3:, 0] += sin_yaw * eye_span * 0.06
    nose_shift = float(np.clip(sin_yaw * eye_span * 0.35,
                               -0.42 * eye_span * x_scale,
                               0.42 * eye_span * x_scale))
    projected[2, 0] += nose_shift

    # Anchor target points follow the same projection. The ear is the
    # near-side lateral contour; the chin stabilizes the lower silhouette.
    side = 1.0 if yaw >= 0.0 else -1.0
    ear_base = np.asarray([eye_mid[0] + side * eye_span * 1.08,
                           eye_mid[1] + image_size * 0.20], dtype=np.float64)
    chin_y = float(base[2, 1] + 1.65 * (base[3, 1] - base[2, 1]))
    chin_base = np.asarray([eye_mid[0], chin_y], dtype=np.float64)
    ear = np.asarray([eye_mid[0] + (ear_base[0] - eye_mid[0]) * x_scale
                      + sin_yaw * eye_span * -0.12,
                      eye_mid[1] + (ear_base[1] - eye_mid[1]) * y_scale])
    nose = projected[2].copy()
    chin = np.asarray([eye_mid[0] + (chin_base[0] - eye_mid[0]) * x_scale
                       + sin_yaw * eye_span * 0.12,
                       eye_mid[1] + (chin_base[1] - eye_mid[1]) * y_scale])
    silhouette = np.stack((ear, nose, chin)).astype(np.float32)

    # A continuous transition avoids a hard gate at the profile threshold.
    blend = _smoothstep((abs(yaw) - 35.0) / 40.0)
    anchor_weights = np.asarray([0.35 * blend, 0.0, 0.50 * blend], dtype=np.float32)
    anchors = (np.full((3, 2), np.nan, dtype=np.float32) if source_anchors is None
               else np.asarray(source_anchors, dtype=np.float32).reshape(3, 2))
    return PoseTemplate(projected.astype(np.float32), silhouette, anchor_weights,
                        anchors, yaw, pitch, roll, image_size, mode)


def source_silhouette_anchors(face: Any, yaw: float) -> np.ndarray:
    """Return near-ear/nose/chin observations, or NaNs when dense points are absent."""
    try:
        from roop.face_analyser import profile_anchors
        anchors = profile_anchors(face, float(yaw))
        if anchors is not None:
            return np.asarray(anchors, dtype=np.float32).reshape(3, 2)
    except (ImportError, TypeError, ValueError, AttributeError):
        pass
    return np.full((3, 2), np.nan, dtype=np.float32)


__all__ = ["PoseTemplate", "warp_template", "source_silhouette_anchors"]
