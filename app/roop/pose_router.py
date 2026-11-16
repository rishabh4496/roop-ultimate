"""Offline head-pose estimation and pose-matched .fsz embedding routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import cv2
import numpy as np

_FACE_MODEL_5 = np.asarray([
    [-0.342, -0.412, 0.169], [0.342, -0.412, 0.169],
    [0.000, -0.095, 0.487], [-0.368, 0.547, 0.299],
    [0.368, 0.547, 0.299],
], dtype=np.float64)
_YAW_BUCKETS = ("profile_left", "half_left", "frontal", "half_right", "profile_right")


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _unit(vector):
    try:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(value)) if np.isfinite(value).all() else 0.0
    return value / norm if norm > 1e-8 else None


def estimate_head_pose(landmarks: Any, frame_shape=None) -> Optional[tuple[float, float, float]]:
    """Estimate yaw/pitch/roll in degrees using the canonical 3D model + solvePnP."""
    try:
        points = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 5 or points.shape[1] < 2:
        return None
    image_points = points[:5, :2]
    if not np.isfinite(image_points).all():
        return None
    if frame_shape is None or len(frame_shape) < 2:
        width = max(1.0, float(np.ptp(image_points[:, 0]) * 6.0))
        height = max(1.0, float(np.ptp(image_points[:, 1]) * 6.0))
    else:
        height, width = float(frame_shape[0]), float(frame_shape[1])
    focal = max(width, height)
    camera = np.asarray([[focal, 0.0, width * 0.5],
                         [0.0, focal, height * 0.5],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        _FACE_MODEL_5, image_points, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                _FACE_MODEL_5, image_points, camera, distortion, rvec, tvec)
        except cv2.error:
            pass
    rotation, _ = cv2.Rodrigues(rvec)
    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if sy > 1e-7:
        pitch_x = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw_y = math.atan2(-float(rotation[2, 0]), sy)
        roll_z = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        pitch_x = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        yaw_y = math.atan2(-float(rotation[2, 0]), sy)
        roll_z = 0.0
    # OpenCV's camera Y axis points down; negating X rotation makes positive
    # pitch mean looking down, matching roop.head_pose's public convention.
    values = (math.degrees(yaw_y), -math.degrees(pitch_x), math.degrees(roll_z))
    return tuple(float(np.clip(v, -180.0, 180.0)) for v in values)


def pose_bucket(yaw: float, pitch: float = 0.0) -> str:
    """Return one of the five required yaw bins; pitch is indexed separately."""
    yaw = float(yaw)
    if abs(yaw) < 20.0:
        return "frontal"
    if yaw < -50.0:
        return "profile_left"
    if yaw > 50.0:
        return "profile_right"
    return "half_left" if yaw < 0.0 else "half_right"


def pitch_bucket(pitch: float) -> str:
    return "up" if float(pitch) < -15.0 else "down" if float(pitch) > 15.0 else "level"


@dataclass(frozen=True)
class PoseRoute:
    embedding: np.ndarray
    bucket: str
    source_indices: tuple[int, ...]
    weights: tuple[float, ...]
    pitch_bucket: str = "level"


class PoseRouter:
    """Index source embeddings by pose and smoothly interpolate per track.

    The router reads source metadata from a loaded .fsz archive, performs
    NumPy weighting, and makes no model or network calls. The default EMA is
    the requested 0.25.
    """

    def __init__(self, *, alpha: float = 0.25, top_k: int = 3):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.top_k = max(1, int(top_k))
        self._ema: dict[Any, tuple[int, np.ndarray]] = {}
        self._pose_ema: dict[Any, tuple[int, float, float]] = {}
        self._entry_cache: dict[int, tuple[Any, list[tuple[int, np.ndarray, float, float]]]] = {}
        self._bucket_cache: dict[int, tuple[Any, dict[str, list[tuple[int, np.ndarray, float, float]]]]] = {}

    def reset(self, track_id: Any = None) -> None:
        if track_id is None:
            self._ema.clear()
            self._pose_ema.clear()
        else:
            self._ema.pop(track_id, None)
            self._pose_ema.pop(track_id, None)

    def index(self, faceset: Any) -> dict[str, list[tuple[int, np.ndarray, float, float]]]:
        """Build the yaw-bin index plus the independent up/down pitch weights."""
        key = id(faceset)
        cached = self._bucket_cache.get(key)
        if cached is not None and cached[0] is faceset:
            return {name: list(entries) for name, entries in cached[1].items()}
        indexed = {name: [] for name in
                   ("frontal", "half_left", "half_right", "profile_left", "profile_right", "up", "down")}
        for entry in self._indexed_entries(faceset):
            source_index, vector, yaw, pitch = entry
            indexed[pose_bucket(yaw, pitch)].append(entry)
            vertical = pitch_bucket(pitch)
            if vertical in ("up", "down"):
                indexed[vertical].append(entry)
        self._bucket_cache[key] = (faceset, indexed)
        return {name: list(entries) for name, entries in indexed.items()}

    def _indexed_buckets(self, faceset: Any):
        key = id(faceset)
        cached = self._bucket_cache.get(key)
        if cached is not None and cached[0] is faceset:
            return cached[1]
        self.index(faceset)
        cached = self._bucket_cache.get(key)
        return cached[1] if cached is not None and cached[0] is faceset else {}

    def _indexed_entries(self, faceset: Any):
        key = id(faceset)
        cached = self._entry_cache.get(key)
        if cached is not None and cached[0] is faceset:
            return cached[1]
        entries = self._entries(faceset)
        self._entry_cache[key] = (faceset, entries)
        return entries

    @staticmethod
    def _entries(faceset: Any):
        metadata = _get(faceset, "faceset_metadata")
        if metadata is None and isinstance(faceset, dict):
            metadata = faceset
        if metadata is not None:
            sources = metadata.get("sources", []) if isinstance(metadata, dict) else []
            result = []
            for fallback_index, source in enumerate(sources or []):
                identity = source.get("identity") or {}
                embedding = (identity.get("normalized_embedding")
                             if isinstance(identity, dict) else None)
                if embedding is None and isinstance(identity, dict):
                    embedding = identity.get("arcface_embedding", identity.get("embedding"))
                geometry = source.get("geometry") or {}
                pose = source.get("pose")
                if pose is None:
                    pose = (geometry.get("yaw"), geometry.get("pitch"))
                # Older .fsz files stored solve_pose_5pt's angles, while new
                # files store solvePnP angles. Refit archived 2D points when
                # their camera geometry is recoverable so both sides of the
                # target/source comparison use the same convention.
                if geometry.get("pose_method") != "solvePnP/canonical5":
                    points = geometry.get("landmarks_5")
                    if points is None:
                        points = geometry.get("landmarks_2d")
                    if points is not None:
                        points_array = np.asarray(points, dtype=np.float32)
                        if points_array.ndim == 2 and points_array.shape[0] != 5:
                            landmarks68 = geometry.get("landmarks_68")
                            if landmarks68 is None and points_array.shape[0] >= 68:
                                landmarks68 = points_array[:68, :2]
                            if landmarks68 is not None:
                                lm68_array = np.asarray(landmarks68, dtype=np.float32)
                                if lm68_array.ndim == 2 and lm68_array.shape[0] >= 68:
                                    points = np.asarray([
                                        lm68_array[36:42, :2].mean(axis=0),
                                        lm68_array[42:48, :2].mean(axis=0),
                                        lm68_array[30, :2], lm68_array[48, :2],
                                        lm68_array[54, :2],
                                    ], dtype=np.float32)
                                else:
                                    points = None
                    size = geometry.get("camera_size")
                    face_scale = geometry.get("face_scale") or {}
                    if size is None:
                        pixels = float(face_scale.get("pixels") or 0.0)
                        rel_h = float(face_scale.get("relative_height") or 0.0)
                        rel_w = float(face_scale.get("relative_width") or 0.0)
                        width = pixels / rel_w if pixels > 0.0 and rel_w > 0.0 else 0.0
                        height = pixels / rel_h if pixels > 0.0 and rel_h > 0.0 else 0.0
                        if width > 0.0 and height > 0.0:
                            size = (width, height)
                    if points is not None and size is not None:
                        pose = estimate_head_pose(points, frame_shape=(size[1], size[0]))
                vector = _unit(embedding) if embedding is not None else None
                if vector is None:
                    continue
                try:
                    yaw, pitch = float(pose[0] or 0.0), float(pose[1] or 0.0)
                except (TypeError, ValueError, IndexError):
                    yaw, pitch = 0.0, 0.0
                result.append((int(source.get("source_index", fallback_index)), vector, yaw, pitch))
            if result:
                return result

        faces = _get(faceset, "faces", []) or []
        poses = _get(faceset, "face_poses", []) or []
        result = []
        for index, face in enumerate(faces):
            embedding = _get(face, "normed_embedding")
            if embedding is None:
                embedding = _get(face, "embedding")
            vector = _unit(embedding)
            if vector is None:
                continue
            pose = poses[index] if index < len(poses) and poses[index] is not None else (0.0, 0.0)
            try:
                yaw, pitch = float(pose[0]), float(pose[1])
            except (TypeError, ValueError, IndexError):
                yaw, pitch = 0.0, 0.0
            result.append((index, vector, yaw, pitch))
        return result

    def route(self, faceset: Any, yaw: float, pitch: float, *, track_id: Any = 0,
              frame_index: Optional[int] = None, scene_cut: bool = False) -> Optional[PoseRoute]:
        if scene_cut:
            self.reset(track_id)
        previous = self._ema.get(track_id)
        if (previous is not None and frame_index is not None
                and previous[0] >= 0 and int(frame_index) != previous[0] + 1):
            self.reset(track_id)
            previous = None
        prior_pose = self._pose_ema.get(track_id)
        if (prior_pose is not None and frame_index is not None
                and prior_pose[0] >= 0 and int(frame_index) != prior_pose[0] + 1):
            self.reset(track_id)
            previous = prior_pose = None
        buckets = self._indexed_buckets(faceset)
        if not buckets:
            return None
        target_yaw, target_pitch = float(yaw), float(pitch)
        if prior_pose is not None:
            yaw_delta = (target_yaw - prior_pose[1] + 180.0) % 360.0 - 180.0
            target_yaw = prior_pose[1] + self.alpha * yaw_delta
            target_pitch = prior_pose[2] + self.alpha * (target_pitch - prior_pose[2])
        # Search the matching yaw bucket first, widening to neighboring bins
        # only when it lacks enough samples for top-K interpolation. The
        # independent pitch bucket contributes candidates for steep up/down
        # poses; final ranking uses the full continuous angle distance.
        yaw_order = [pose_bucket(target_yaw)]
        center = _YAW_BUCKETS.index(yaw_order[0])
        for radius in range(1, len(_YAW_BUCKETS)):
            neighbors = [i for i in (center - radius, center + radius)
                         if 0 <= i < len(_YAW_BUCKETS)]
            neighbors.sort(key=lambda i: abs(i - center))
            yaw_order.extend(_YAW_BUCKETS[i] for i in neighbors)
        candidate_entries = []
        seen_sources = set()

        def add_bucket(name):
            for entry in buckets.get(name, ()):
                if entry[0] not in seen_sources:
                    seen_sources.add(entry[0])
                    candidate_entries.append(entry)

        add_bucket(yaw_order[0])
        for bucket_name in yaw_order[1:]:
            if len(candidate_entries) >= self.top_k:
                break
            add_bucket(bucket_name)
        vertical = pitch_bucket(target_pitch)
        if vertical in ("up", "down"):
            add_bucket(vertical)
        if not candidate_entries:
            return None
        ordered = sorted(candidate_entries, key=lambda item: math.hypot(
            item[2] - target_yaw, 0.75 * (item[3] - target_pitch)))[:self.top_k]
        distances = [math.hypot(item[2] - target_yaw,
                                0.75 * (item[3] - target_pitch)) for item in ordered]
        if distances[0] < 1e-5:
            weights = np.zeros(len(ordered), dtype=np.float64)
            weights[0] = 1.0
        else:
            # A 12-degree floor avoids hard source changes between sparse pose
            # buckets while still preferring a genuinely matching profile.
            inv = np.asarray([1.0 / max(12.0, d) for d in distances], dtype=np.float64)
            weights = inv / max(1e-12, float(inv.sum()))
        vectors = [item[1] for item in ordered]
        if any(vector.shape != vectors[0].shape for vector in vectors):
            return None
        candidate = _unit(np.sum(np.stack(vectors) * weights[:, None], axis=0))
        if candidate is None:
            return None
        if previous is not None and previous[1].shape == candidate.shape:
            candidate = _unit((1.0 - self.alpha) * previous[1] + self.alpha * candidate)
            if candidate is None:
                return None
        stored_frame = -1 if frame_index is None else int(frame_index)
        self._ema[track_id] = (stored_frame, candidate.copy())
        self._pose_ema[track_id] = (stored_frame, target_yaw, target_pitch)
        return PoseRoute(candidate.astype(np.float32), pose_bucket(target_yaw, target_pitch),
                         tuple(int(item[0]) for item in ordered),
                         tuple(float(w) for w in weights), pitch_bucket(target_pitch))


__all__ = ["PoseRouter", "PoseRoute", "estimate_head_pose", "pose_bucket", "pitch_bucket"]
