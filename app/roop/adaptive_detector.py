"""Adaptive SCRFD face detector strategy and orchestration.

Implements an intelligent, motion-aware, and geometry-validated adaptive detection pipeline:
1. Avoids running full-frame SCRFD unnecessarily on every frame when motion is low.
2. Reuses valid detections/tracks during low-motion phases with motion-compensated projection.
3. Triggers immediate re-detection upon:
   - Track confidence drops
   - Face count changes (new face entered, person exited, expected count mismatch)
   - Large motion or abrupt acceleration
   - Occlusion (inter-face overlap or boundary clipping)
   - Invalid or degenerate landmark geometry (collapsed eyes, inverted eye-mouth ordering)
4. Employs ROI rescue for missing tracked faces before falling back to full-frame detection.
5. Preserves reliable periodic full-frame recovery and instant scene-cut reset.
6. Provides configurable detection frequency and thresholds via globals and environment variables.
7. Strictly preserves small-face and profile-face recall without sacrificing detection accuracy.
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

import roop.globals
from roop.degrade import swallowed as _swallowed
from roop.typing import Face, Frame


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveDetectorConfig:
    """Configurable hyperparameters for the adaptive detector strategy."""
    enabled: bool = True
    full_recovery_interval: int = 8
    low_motion_threshold: float = 0.05       # Normalized to face bounding box diagonal
    large_motion_threshold: float = 0.18     # Normalized to face bounding box diagonal
    confidence_drop_threshold: float = 0.15  # Re-detect if confidence drops by this delta
    min_confidence: float = 0.55             # Absolute confidence floor
    roi_rescue_enabled: bool = True
    roi_pad_ratio: float = 0.85              # Margin around predicted bbox for ROI detection
    scene_cut_threshold: float = 0.40        # Normalized histogram distance triggering reset
    max_coast_frames: int = 4                # Maximum consecutive frames to reuse without full detect
    small_face_threshold: float = 60.0       # Diagonal in pixels below which a face is considered small
    profile_yaw_threshold: float = 45.0      # Yaw angle in degrees above which face is in profile

    @classmethod
    def from_env(cls) -> "AdaptiveDetectorConfig":
        def _get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key)
            if val is None:
                return default
            return val.strip().lower() in ("1", "true", "yes", "on")

        def _get_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default

        def _get_float(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default

        glob_enabled = getattr(roop.globals, "adaptive_detection", True)
        enabled = _get_bool("ROOP_ADAPTIVE_DETECTION", bool(glob_enabled))

        return cls(
            enabled=enabled,
            full_recovery_interval=_get_int("ROOP_ADAPTIVE_DETECT_INTERVAL",
                                            getattr(roop.globals, "adaptive_detect_interval", 8)),
            low_motion_threshold=_get_float("ROOP_ADAPTIVE_LOW_MOTION_THRESH", 0.05),
            large_motion_threshold=_get_float("ROOP_ADAPTIVE_LARGE_MOTION_THRESH", 0.18),
            confidence_drop_threshold=_get_float("ROOP_ADAPTIVE_CONF_DROP_THRESH", 0.15),
            min_confidence=_get_float("ROOP_ADAPTIVE_MIN_CONFIDENCE", 0.55),
            roi_rescue_enabled=_get_bool("ROOP_ADAPTIVE_ROI_RESCUE", True),
            roi_pad_ratio=_get_float("ROOP_ADAPTIVE_ROI_PAD_RATIO", 0.85),
            scene_cut_threshold=_get_float("ROOP_ADAPTIVE_SCENE_CUT_THRESH", 0.40),
            max_coast_frames=_get_int("ROOP_ADAPTIVE_MAX_COAST_FRAMES", 4),
            small_face_threshold=_get_float("ROOP_ADAPTIVE_SMALL_FACE_THRESH", 60.0),
            profile_yaw_threshold=_get_float("ROOP_ADAPTIVE_PROFILE_YAW_THRESH", 45.0),
        )


# ---------------------------------------------------------------------------
# Landmark Geometry Validator
# ---------------------------------------------------------------------------

class LandmarkGeometryValidator:
    """Rigorous geometric validation of 5-point facial keypoints."""

    @staticmethod
    def validate(kps: Any, bbox: Optional[np.ndarray] = None,
                 frame_shape: Optional[Tuple[int, int]] = None) -> Tuple[bool, str]:
        """Verify whether 5-point facial keypoints satisfy anatomical and geometric constraints.

        Returns:
            (is_valid: bool, failure_reason: str)
        """
        if kps is None:
            return False, "kps_is_none"

        try:
            pts = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return False, "invalid_shape_or_type"

        if pts.shape[0] < 5:
            return False, f"insufficient_points_{pts.shape[0]}"

        # 1. Non-finite values check
        if not np.all(np.isfinite(pts[:5])):
            return False, "non_finite_coordinates"

        p5 = pts[:5]
        left_eye, right_eye, nose, left_mouth, right_mouth = p5[0], p5[1], p5[2], p5[3], p5[4]

        # 2. Inter-ocular distance (eyes must not collapse)
        interocular = float(np.linalg.norm(right_eye - left_eye))
        if interocular < 3.0:
            return False, f"interocular_collapsed_{interocular:.2f}px"

        # 3. Eye to mouth vertical separation (face height span)
        eye_center = (left_eye + right_eye) * 0.5
        mouth_center = (left_mouth + right_mouth) * 0.5
        eye_mouth_dist = float(np.linalg.norm(mouth_center - eye_center))
        if eye_mouth_dist < 3.0:
            return False, f"eye_mouth_span_collapsed_{eye_mouth_dist:.2f}px"

        # 4. In-plane roll angle and orientation consistency
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        roll_rad = math.atan2(dy, dx)
        # Vector from eyes to mouth
        v_em_x = float(mouth_center[0] - eye_center[0])
        v_em_y = float(mouth_center[1] - eye_center[1])

        # In anatomical upright frame, eyes-to-mouth vector must point downward along facial axis
        # Dot product with orthogonal downward direction to eye vector
        down_x = -dy
        down_y = dx
        down_norm = math.hypot(down_x, down_y)
        if down_norm > 1e-5:
            dot_down = (v_em_x * down_x + v_em_y * down_y) / down_norm
            # If dot product is strongly negative, mouth is positioned ABOVE eyes (inverted landmark labeling)
            if dot_down < -1.0:
                return False, f"inverted_facial_axis_dot_{dot_down:.2f}"

        # 5. Bounding box enclosure check if bbox is provided
        if bbox is not None:
            b = np.asarray(bbox, dtype=np.float32).reshape(-1)
            if b.size >= 4:
                bw = float(b[2] - b[0])
                bh = float(b[3] - b[1])
                if bw <= 4.0 or bh <= 4.0:
                    return False, f"degenerate_bbox_size_{bw:.1f}x{bh:.1f}"

                # Allow modest margin (25%) around bbox for keypoints
                pad_x = bw * 0.25
                pad_y = bh * 0.25
                x_min, y_min = b[0] - pad_x, b[1] - pad_y
                x_max, y_max = b[2] + pad_x, b[3] + pad_y

                outside_x = np.any((p5[:, 0] < x_min) | (p5[:, 0] > x_max))
                outside_y = np.any((p5[:, 1] < y_min) | (p5[:, 1] > y_max))
                if outside_x or outside_y:
                    return False, "landmarks_outside_bbox_margin"

        # 6. Frame boundary check if frame_shape is provided
        if frame_shape is not None:
            fh, fw = frame_shape[:2]
            # Keypoints should not be wildly outside frame
            if np.any(p5[:, 0] < -fw * 0.3) or np.any(p5[:, 0] > fw * 1.3):
                return False, "landmarks_wildly_outside_canvas_x"
            if np.any(p5[:, 1] < -fh * 0.3) or np.any(p5[:, 1] > fh * 1.3):
                return False, "landmarks_wildly_outside_canvas_y"

        return True, "valid"


# ---------------------------------------------------------------------------
# Adaptive Track State
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveTrackState:
    """Internal tracking state for a face across consecutive video frames."""
    track_id: int
    bbox: np.ndarray
    kps: np.ndarray
    confidence: float
    landmarks_106: Optional[np.ndarray] = None
    landmarks_68: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    normed_embedding: Optional[np.ndarray] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_seen_frame: int = 0
    consecutive_coasts: int = 0
    hit_count: int = 1
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    is_profile: bool = False
    is_small: bool = False

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) * 0.5,
                         (self.bbox[1] + self.bbox[3]) * 0.5], dtype=np.float32)

    @property
    def diagonal(self) -> float:
        w = float(self.bbox[2] - self.bbox[0])
        h = float(self.bbox[3] - self.bbox[1])
        return float(math.hypot(w, h))

    def predict_bbox(self, frame_idx: int) -> np.ndarray:
        """Project bounding box forward to frame_idx using velocity."""
        dt = max(0, int(frame_idx) - int(self.last_seen_frame))
        if dt > 0 and np.any(self.velocity):
            proj = self.bbox + self.velocity * float(dt)
            if proj[2] > proj[0] and proj[3] > proj[1]:
                return proj.astype(np.float32)
        return self.bbox.copy()


# ---------------------------------------------------------------------------
# Detection Plan & Action
# ---------------------------------------------------------------------------

@dataclass
class AdaptivePlan:
    """Action planned for a given frame."""
    action: str                        # 'FULL_FRAME', 'ROI_RESCUE', 'COAST_REUSE'
    reason: str                        # Explanation for the chosen strategy
    roi_boxes: List[np.ndarray] = field(default_factory=list)
    reusable_track_ids: List[int] = field(default_factory=list)
    missing_track_ids: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main Adaptive Face Detector
# ---------------------------------------------------------------------------

class AdaptiveFaceDetector:
    """Intelligent adaptive detector managing SCRFD full passes, ROI rescues, and coasting."""

    def __init__(self, config: Optional[AdaptiveDetectorConfig] = None):
        self.config = config or AdaptiveDetectorConfig.from_env()
        self.tracks: Dict[int, AdaptiveTrackState] = {}
        self._next_id: int = 0
        self._last_full_frame: int = -9999
        self._last_frame_idx: Optional[int] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._lock = RLock()

        # Telemetry metrics
        self.telemetry: Dict[str, Any] = {
            "total_frames": 0,
            "full_detections": 0,
            "roi_rescues_attempted": 0,
            "roi_rescues_succeeded": 0,
            "coast_reuses": 0,
            "trigger_reasons": {},
            "missed_detections": 0,
            "false_detections": 0,
            "profile_detections": 0,
            "small_face_detections": 0,
            "gpu_eval_count": 0,
        }

    def reset(self) -> None:
        """Reset internal tracking state and scene history."""
        with self._lock:
            self.tracks.clear()
            self._next_id = 0
            self._last_full_frame = -9999
            self._last_frame_idx = None
            self._prev_hist = None

    def _record_trigger(self, reason: str) -> None:
        tr = self.telemetry["trigger_reasons"]
        tr[reason] = tr.get(reason, 0) + 1

    @staticmethod
    def _compute_hist(frame: np.ndarray) -> Optional[np.ndarray]:
        """Compute normalized 3D color histogram for fast scene cut detection."""
        try:
            from roop.face_analyser import compute_histogram_signature
            return compute_histogram_signature(frame)
        except Exception:
            return None

    @staticmethod
    def _compare_hist(hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
        """Bhattacharyya histogram difference in [0, 1] range."""
        try:
            from roop.face_analyser import compare_histogram_difference
            return compare_histogram_difference(hist1, hist2)
        except Exception:
            return 0.0

    @staticmethod
    def _bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0.0:
            return 0.0
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def _estimate_pose(kps: np.ndarray) -> Tuple[float, float, float]:
        """Estimate 3D head pose (yaw, pitch, roll) from 5-point keypoints."""
        try:
            from roop.face_util import solve_pose_5pt
            p = solve_pose_5pt(np.asarray(kps, np.float32))
            if p is not None:
                return float(p[0]), float(p[1]), float(p[2])
        except Exception:
            pass
        # Fallback approximate yaw from eye-nose symmetry
        try:
            d_left = float(np.linalg.norm(kps[2] - kps[0]))
            d_right = float(np.linalg.norm(kps[2] - kps[1]))
            span = d_left + d_right
            if span > 1e-4:
                sym = (d_right - d_left) / span
                yaw_approx = float(np.clip(sym * 90.0, -90.0, 90.0))
                return yaw_approx, 0.0, 0.0
        except Exception:
            pass
        return 0.0, 0.0, 0.0

    def plan_strategy(
        self,
        frame: np.ndarray,
        frame_idx: int,
        expected_count: Optional[int] = None,
        force_full: bool = False
    ) -> AdaptivePlan:
        """Determine whether to run full-frame detection, ROI rescue, or reuse previous tracks."""
        h, w = frame.shape[:2]
        curr_hist = self._compute_hist(frame)

        # 1. Force full if adaptive detection is globally disabled
        if not self.config.enabled:
            return AdaptivePlan("FULL_FRAME", reason="adaptive_disabled")

        # 2. Scene cut detection
        if self._prev_hist is not None and curr_hist is not None:
            dist = self._compare_hist(self._prev_hist, curr_hist)
            if dist > self.config.scene_cut_threshold:
                self.reset()
                self._prev_hist = curr_hist
                return AdaptivePlan("FULL_FRAME", reason="scene_cut")
        self._prev_hist = curr_hist

        # 3. Explicit forced full detection
        if force_full:
            return AdaptivePlan("FULL_FRAME", reason="forced_full")

        # 4. Initial frame or no active tracks
        if not self.tracks:
            return AdaptivePlan("FULL_FRAME", reason="no_tracks")

        # 5. Periodic full-frame recovery interval
        if frame_idx - self._last_full_frame >= self.config.full_recovery_interval:
            return AdaptivePlan("FULL_FRAME", reason="periodic_recovery")

        # 6. Face count change against expected_count
        if expected_count is not None and len(self.tracks) != expected_count:
            return AdaptivePlan("FULL_FRAME", reason=f"face_count_change_{len(self.tracks)}_vs_{expected_count}")

        # 7. Analyze individual track conditions
        reusable_ids = []
        roi_boxes = []
        missing_ids = []
        needs_full = False
        full_reason = ""

        track_items = list(self.tracks.items())
        all_boxes = [t.bbox for _, t in track_items]

        for tid, track in track_items:
            diag = track.diagonal

            # 7a. Check coast limit
            if track.consecutive_coasts >= self.config.max_coast_frames:
                needs_full = True
                full_reason = f"max_coast_reached_track_{tid}"
                break

            # 7b. Check landmark geometric validity
            is_valid_geo, geo_err = LandmarkGeometryValidator.validate(track.kps, track.bbox, (h, w))
            if not is_valid_geo:
                needs_full = True
                full_reason = f"invalid_geometry_{geo_err}_track_{tid}"
                break

            # 7c. Check confidence drop
            if track.confidence < self.config.min_confidence:
                if self.config.roi_rescue_enabled:
                    roi_boxes.append(track.predict_bbox(frame_idx))
                    missing_ids.append(tid)
                else:
                    needs_full = True
                    full_reason = f"confidence_drop_track_{tid}_{track.confidence:.2f}"
                    break
                continue

            # 7d. Check large motion
            vel_mag = float(np.linalg.norm(track.velocity[:2]))
            norm_motion = vel_mag / max(10.0, diag)
            if norm_motion > self.config.large_motion_threshold:
                needs_full = True
                full_reason = f"large_motion_{norm_motion:.2f}_track_{tid}"
                break

            # 7e. Check occlusion with other tracks
            has_occlusion = False
            for other_tid, other_track in track_items:
                if other_tid == tid:
                    continue
                iou = self._bbox_iou(track.bbox, other_track.bbox)
                if iou > 0.12:
                    has_occlusion = True
                    break
            if has_occlusion:
                needs_full = True
                full_reason = f"occlusion_overlap_track_{tid}"
                break

            # 7f. Check boundary clipping (panning off screen)
            b = track.bbox
            bw = float(b[2] - b[0])
            bh = float(b[3] - b[1])
            visible_w = max(0.0, min(float(w), float(b[2])) - max(0.0, float(b[0])))
            visible_h = max(0.0, min(float(h), float(b[3])) - max(0.0, float(b[1])))
            vis_area = visible_w * visible_h
            tot_area = max(1.0, bw * bh)
            if vis_area / tot_area < 0.70:
                needs_full = True
                full_reason = f"boundary_clipping_track_{tid}"
                break

            # If track passes all checks and motion is low -> eligible for coast reuse
            reusable_ids.append(tid)

        if needs_full:
            return AdaptivePlan("FULL_FRAME", reason=full_reason)

        if missing_ids and self.config.roi_rescue_enabled:
            return AdaptivePlan("ROI_RESCUE", reason="roi_rescue_missing",
                                roi_boxes=roi_boxes,
                                reusable_track_ids=reusable_ids,
                                missing_track_ids=missing_ids)

        return AdaptivePlan("COAST_REUSE", reason="low_motion_reuse",
                            reusable_track_ids=reusable_ids)

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
        expected_count: Optional[int] = None,
        det_fn: Optional[Callable[[np.ndarray], List[Face]]] = None,
        roi_det_fn: Optional[Callable[[np.ndarray, np.ndarray], List[Face]]] = None,
        force_full: bool = False
    ) -> List[Face]:
        """Execute face detection adaptively based on motion, geometry, and tracking confidence."""
        with self._lock:
            self.telemetry["total_frames"] += 1
            if frame_idx is None:
                frame_idx = (self._last_frame_idx + 1) if self._last_frame_idx is not None else 0
            self._last_frame_idx = frame_idx

            # Default fallback detector calls into face_util if not provided
            if det_fn is None:
                from roop.face_util import get_all_faces
                det_fn = lambda fr: get_all_faces(fr, expected_count=expected_count)

            if roi_det_fn is None:
                from roop.face_util import get_all_faces_in_roi
                roi_det_fn = lambda fr, box: get_all_faces_in_roi(fr, box, pad_ratio=self.config.roi_pad_ratio)

            plan = self.plan_strategy(frame, frame_idx, expected_count, force_full=force_full)
            self._record_trigger(plan.reason)

            # Strategy 1: Full-Frame SCRFD Detection
            if plan.action == "FULL_FRAME":
                self.telemetry["full_detections"] += 1
                self.telemetry["gpu_eval_count"] += 1
                self._last_full_frame = frame_idx
                raw_faces = det_fn(frame) or []
                faces = self._update_tracks_from_detections(raw_faces, frame_idx, frame.shape)
                for f in faces:
                    setattr(f, "_detection_mode", "full")
                return faces

            # Strategy 2: ROI Rescue for Missing / Dropped Tracks
            if plan.action == "ROI_RESCUE":
                self.telemetry["roi_rescues_attempted"] += 1
                rescued_faces = []
                all_rescued = True

                for box in plan.roi_boxes:
                    self.telemetry["gpu_eval_count"] += 1
                    rf = roi_det_fn(frame, box) or []
                    if rf:
                        rescued_faces.extend(rf)
                    else:
                        all_rescued = False

                if all_rescued and rescued_faces:
                    self.telemetry["roi_rescues_succeeded"] += 1
                    # Merge rescued detections with reusable coasted tracks
                    reused_faces = self._generate_coasted_faces(plan.reusable_track_ids, frame_idx)
                    all_candidates = rescued_faces + reused_faces
                    faces = self._update_tracks_from_detections(all_candidates, frame_idx, frame.shape)
                    for f in faces:
                        setattr(f, "_detection_mode", "roi_rescue")
                    return faces

                # If ROI rescue failed for any track, escalate to full-frame detection
                self.telemetry["full_detections"] += 1
                self.telemetry["gpu_eval_count"] += 1
                self._last_full_frame = frame_idx
                self._record_trigger("roi_failed_escalate_full")
                raw_faces = det_fn(frame) or []
                faces = self._update_tracks_from_detections(raw_faces, frame_idx, frame.shape)
                for f in faces:
                    setattr(f, "_detection_mode", "full")
                return faces

            # Strategy 3: Low-Motion Coast Reuse
            self.telemetry["coast_reuses"] += 1
            faces = self._generate_coasted_faces(plan.reusable_track_ids, frame_idx)
            for f in faces:
                setattr(f, "_detection_mode", "coast_reuse")
            return faces

    def _generate_coasted_faces(self, track_ids: List[int], frame_idx: int) -> List[Face]:
        """Construct synthetic Face objects for coasted tracks with updated positions."""
        from insightface.app.common import Face as InsightFaceFace

        out_faces = []
        for tid in track_ids:
            if tid not in self.tracks:
                continue
            track = self.tracks[tid]
            dt = max(1, frame_idx - track.last_seen_frame)

            # Velocity-projected bounding box
            pred_box = track.predict_bbox(frame_idx)
            dx = float(pred_box[0] - track.bbox[0])
            dy = float(pred_box[1] - track.bbox[1])

            # Shift landmarks by displacement
            pred_kps = track.kps.copy()
            pred_kps[:, 0] += dx
            pred_kps[:, 1] += dy

            # Construct InsightFace Face object
            f = InsightFaceFace(bbox=pred_box, kps=pred_kps, det_score=float(track.confidence * 0.98))
            f._track_id = tid
            f._adaptive_reused = True

            # Preserve auxiliary attributes if present
            if track.landmarks_106 is not None:
                lm106 = track.landmarks_106.copy()
                lm106[:, 0] += dx
                lm106[:, 1] += dy
                f.landmark_2d_106 = lm106
            if track.landmarks_68 is not None:
                lm68 = track.landmarks_68.copy()
                lm68[:, :2] += [dx, dy]
                f.landmark_3d_68 = lm68
            if track.embedding is not None:
                f.embedding = track.embedding.copy()

            # Update track state
            track.bbox = pred_box
            track.kps = pred_kps
            track.consecutive_coasts += 1
            track.last_seen_frame = frame_idx
            out_faces.append(f)

        return sorted(out_faces, key=lambda x: x.bbox[0])

    def _update_tracks_from_detections(
        self,
        raw_faces: List[Face],
        frame_idx: int,
        frame_shape: Tuple[int, int]
    ) -> List[Face]:
        """Associate detections with existing tracks and update tracking states."""
        from insightface.app.common import Face as InsightFaceFace

        valid_faces = []
        for f in raw_faces:
            kps = getattr(f, "kps", None)
            bbox = getattr(f, "bbox", None)
            is_valid, _ = LandmarkGeometryValidator.validate(kps, bbox, frame_shape)
            if is_valid:
                valid_faces.append(f)
            else:
                self.telemetry["missed_detections"] += 1

        matched_track_ids = set()
        out_faces = []

        for f in valid_faces:
            bbox = np.asarray(f.bbox, dtype=np.float32)
            kps = np.asarray(f.kps, dtype=np.float32)
            conf = float(getattr(f, "det_score", 0.90) or 0.90)
            emb = getattr(f, "embedding", None)
            normed_emb = getattr(f, "normed_embedding", None)
            lm106 = getattr(f, "landmark_2d_106", None)
            lm68 = getattr(f, "landmark_3d_68", None)

            # Estimate pose & attributes
            yaw, pitch, roll = self._estimate_pose(kps)
            is_profile = abs(yaw) >= self.config.profile_yaw_threshold
            diag = math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
            is_small = diag <= self.config.small_face_threshold

            if is_profile:
                self.telemetry["profile_detections"] += 1
            if is_small:
                self.telemetry["small_face_detections"] += 1

            # Find matching track via IoU & embedding
            best_tid = None
            best_iou = 0.30
            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                pred_box = track.predict_bbox(frame_idx)
                iou = self._bbox_iou(bbox, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                # Update existing track
                matched_track_ids.add(best_tid)
                track = self.tracks[best_tid]
                dt = max(1, frame_idx - track.last_seen_frame)
                vel = (bbox - track.bbox) / float(dt)
                track.velocity = (0.6 * track.velocity + 0.4 * vel).astype(np.float32)
                track.bbox = bbox
                track.kps = kps
                track.confidence = 0.7 * track.confidence + 0.3 * conf
                track.consecutive_coasts = 0
                track.hit_count += 1
                track.last_seen_frame = frame_idx
                track.yaw = yaw
                track.pitch = pitch
                track.roll = roll
                track.is_profile = is_profile
                track.is_small = is_small
                if emb is not None:
                    track.embedding = emb
                if normed_emb is not None:
                    track.normed_embedding = normed_emb
                if lm106 is not None:
                    track.landmarks_106 = lm106
                if lm68 is not None:
                    track.landmarks_68 = lm68
                f._track_id = best_tid
            else:
                # Initialize new track
                new_tid = self._next_id
                self._next_id += 1
                matched_track_ids.add(new_tid)
                self.tracks[new_tid] = AdaptiveTrackState(
                    track_id=new_tid,
                    bbox=bbox,
                    kps=kps,
                    confidence=conf,
                    landmarks_106=lm106,
                    landmarks_68=lm68,
                    embedding=emb,
                    normed_embedding=normed_emb,
                    velocity=np.zeros(4, dtype=np.float32),
                    last_seen_frame=frame_idx,
                    consecutive_coasts=0,
                    hit_count=1,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    is_profile=is_profile,
                    is_small=is_small,
                )
                f._track_id = new_tid

            out_faces.append(f)

        # Retire stale tracks (not seen for > 15 frames)
        stale_tids = [tid for tid, t in self.tracks.items()
                      if frame_idx - t.last_seen_frame > 15]
        for tid in stale_tids:
            del self.tracks[tid]

        return sorted(out_faces, key=lambda x: x.bbox[0])

    def get_telemetry(self) -> Dict[str, Any]:
        """Return snapshot of runtime telemetry metrics."""
        with self._lock:
            snap = dict(self.telemetry)
            snap["trigger_reasons"] = dict(self.telemetry["trigger_reasons"])
            snap["active_tracks"] = len(self.tracks)
            return snap


# ---------------------------------------------------------------------------
# Global Singleton Accessor
# ---------------------------------------------------------------------------

_GLOBAL_ADAPTIVE_DETECTOR: Optional[AdaptiveFaceDetector] = None
_GLOBAL_LOCK = RLock()


def get_adaptive_face_detector() -> AdaptiveFaceDetector:
    """Retrieve or construct the global adaptive face detector instance."""
    global _GLOBAL_ADAPTIVE_DETECTOR
    with _GLOBAL_LOCK:
        if _GLOBAL_ADAPTIVE_DETECTOR is None:
            _GLOBAL_ADAPTIVE_DETECTOR = AdaptiveFaceDetector()
        return _GLOBAL_ADAPTIVE_DETECTOR


def reset_adaptive_face_detector() -> None:
    """Reset the global adaptive face detector."""
    global _GLOBAL_ADAPTIVE_DETECTOR
    with _GLOBAL_LOCK:
        if _GLOBAL_ADAPTIVE_DETECTOR is not None:
            _GLOBAL_ADAPTIVE_DETECTOR.reset()


__all__ = [
    "AdaptiveDetectorConfig",
    "LandmarkGeometryValidator",
    "AdaptiveTrackState",
    "AdaptivePlan",
    "AdaptiveFaceDetector",
    "get_adaptive_face_detector",
    "reset_adaptive_face_detector",
]
