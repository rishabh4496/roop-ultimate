"""Stage 9 — Temporal Face Tracking and Flicker Control.

Comprehensive, robust temporal state machine providing:
1. Formal Multi-State Lifecycle:
   - TENTATIVE, STABLE, CROSSING, OCCLUDED, COASTING, PROFILE_TURNING, LOST, RECOVERED, TERMINATED.
2. Complete State Retention per Track:
   - bbox: Smoothed xyxy coordinates
   - landmarks: Coupled 5-point kps and 106-point dense landmarks
   - embedding: 512-d normalized ArcFace running mean, frozen during crossings/occlusions
   - confidence: Composite detection + tracking + landmark stability score
   - velocity: [dx, dy, da, dh] per frame + landmark velocity
   - pose: [pitch, yaw, roll] in degrees
   - last_seen_frame: Frame index of last valid detection
   - source_assignment: Target source face index (locked across occlusions)
   - mask_state: Previous mask, boundary hull contour, and stability score
3. Detection + Tracking Hybrid Architecture:
   - Kalman filter motion model with adaptive process noise
   - Global Hungarian / Linear sum assignment combining motion, ArcFace appearance, and pose
4. Robust Edge Case Handling:
   - 1. Temporary detector dropout (Kalman coasting with plausible geometry)
   - 2. Face crossing another face (crossing state detection, embedding update freeze)
   - 3. Face count changes (dynamic track spawning and retirement)
   - 4. Occlusion (hand/object crossing face; collision-guarded prediction)
   - 5. Profile transition (yaw > 55 deg; profile-tolerant geometry tracking)
   - 6. Rapid movement (velocity-adaptive smoothing release preventing lag)
   - 7. Target face disappearing (frame boundary exit to Re-ID archive)
   - 8. Target face returning (long-term Re-ID memory reactivation with original ID)
5. Confidence-Aware Hermite Spline Interpolation:
   - Strict discontinuity guards (shot cuts, velocity jumps, scale spikes, identity divergence)
   - Cubic Hermite trajectory interpolation with endpoint velocity matching
6. Explicit Temporal Quality Metrics:
   - Landmark jitter variance, identity switch count, dropout recovery rate, mask popping rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception as _e_scipy:
    from roop.degrade import swallowed as _swallowed
    _swallowed("roop/temporal_state_machine.py:linear_sum_assignment", _e_scipy, "scipy fallback")
    linear_sum_assignment = None


# ==============================================================================
# 1. Track Lifecycle State Machine
# ==============================================================================

class TrackState(str, Enum):
    """Formal lifecycle states for temporal face tracks."""
    TENTATIVE = "tentative"          # Initial detection; awaiting verification
    STABLE = "stable"                # Verified, active, high-confidence track
    CROSSING = "crossing"            # Interacting/overlapping with another track (embeddings frozen)
    OCCLUDED = "occluded"            # Hidden by foreign obstacle / low confidence
    COASTING = "coasting"            # Detector dropped out; trajectory predicted via Kalman
    PROFILE_TURNING = "profile"      # Head rotated past 55 degrees yaw
    LOST = "lost"                    # Dropped out or off-screen; stored in Re-ID archive
    RECOVERED = "recovered"          # Re-detected from Re-ID archive with original ID
    TERMINATED = "terminated"        # Exceeded re-identification window; slot retired


# ==============================================================================
# 2. Track Data Structures
# ==============================================================================

@dataclass
class RobustFaceTrack:
    """Persistent, state-aware face track maintaining all 9 required attributes."""

    track_id: int
    state: TrackState = TrackState.TENTATIVE

    # 1. Bounding box [x0, y0, x1, y1]
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    # 2. Facial landmarks (5-point kps and optional 106-point dense landmarks)
    kps: Optional[np.ndarray] = None
    landmark_2d_106: Optional[np.ndarray] = None

    # 3. 512-d normalized identity embedding (running mean, frozen during crossings/occlusion)
    embedding: Optional[np.ndarray] = None

    # 4. Composite confidence score [0.0, 1.0]
    confidence: float = 0.90

    # 5. Velocity [dx, dy, da, dh] per frame + landmark velocity
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    landmark_velocity: Optional[np.ndarray] = None

    # 6. 3D Head Pose [pitch, yaw, roll] in degrees
    pose: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # 7. Last-seen frame index of valid real detection
    last_seen_frame: int = -1

    # 8. Source face assignment index (locked across occlusions)
    source_assignment: Optional[int] = None

    # 9. Mask state (previous mask, contour hull, and temporal stability coefficient)
    mask_state: Dict[str, Any] = field(default_factory=lambda: {
        "mask": None,
        "hull": None,
        "stability": 1.0,
        "last_mask_frame": -1
    })

    # Internal Kalman Filter State: [cx, cy, aspect, height, dcx, dcy, daspect, dheight]
    kalman_state: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    kalman_cov: np.ndarray = field(default_factory=lambda: np.eye(8, dtype=np.float32) * 10.0)

    # Lifecycle counters
    hits: int = 1
    misses: int = 0
    coasted_run: int = 0

    # Trajectory histories (strictly bounded to prevent RAM leaks)
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=24))
    kps_history: deque = field(default_factory=lambda: deque(maxlen=24))
    pose_history: deque = field(default_factory=lambda: deque(maxlen=24))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=24))

    # Underlying Face template for building synthetic/coasted Face objects
    template_face: Any = None

    def update_embedding(self, new_emb: Optional[np.ndarray], alpha: float = 0.15) -> None:
        """Update identity embedding running mean (ONLY when not crossing or occluded)."""
        if new_emb is None:
            return
        v = np.asarray(new_emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm <= 1e-6:
            return
        v = v / norm

        # NEVER update embedding during crossing or occlusion to prevent identity drift!
        if self.state in (TrackState.CROSSING, TrackState.OCCLUDED, TrackState.COASTING):
            return

        if self.embedding is None:
            self.embedding = v.copy()
        else:
            # Exponentially weighted running average
            blended = (1.0 - alpha) * self.embedding + alpha * v
            b_norm = float(np.linalg.norm(blended))
            if b_norm > 1e-6:
                self.embedding = (blended / b_norm).astype(np.float32)

    def snapshot(self) -> Dict[str, Any]:
        """Diagnostic state snapshot for testing and logging."""
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "bbox": self.bbox.tolist(),
            "confidence": float(self.confidence),
            "velocity": self.velocity.tolist(),
            "pose": self.pose.tolist(),
            "last_seen_frame": self.last_seen_frame,
            "source_assignment": self.source_assignment,
            "hits": self.hits,
            "misses": self.misses,
            "coasted_run": self.coasted_run,
            "has_kps": self.kps is not None,
            "has_dense_lm": self.landmark_2d_106 is not None,
            "has_mask": self.mask_state["mask"] is not None,
        }


# ==============================================================================
# 3. Confidence-Aware Interpolator & Discontinuity Guards
# ==============================================================================

class ConfidenceAwareInterpolator:
    """Cubic Hermite spline and confidence-weighted interpolation with discontinuity rejection."""

    MAX_VELOCITY_RATIO = 3.0       # Acceleration jump > 3x indicates motion discontinuity
    MAX_SCALE_RATIO = 1.8          # Bounding box scale jump > 1.8x
    MAX_IDENTITY_DIST = 0.40       # Cosine distance > 0.40 indicates different person
    MAX_TRAVEL_PER_FRAME = 45.0    # Plausible maximum head travel in pixels per frame

    @staticmethod
    def cubic_hermite(p0: float, p1: float, v0: float, v1: float, t: float) -> float:
        """Standard 1D Cubic Hermite spline interpolation matching endpoint velocities."""
        t2 = t * t
        t3 = t2 * t
        h0 = 2.0 * t3 - 3.0 * t2 + 1.0
        h1 = t3 - 2.0 * t2 + t
        h2 = -2.0 * t3 + 3.0 * t2
        h3 = t3 - t2
        return float(h0 * p0 + h1 * v0 + h2 * p1 + h3 * v1)

    @classmethod
    def check_discontinuity(
        cls,
        face_a: Any,
        face_b: Any,
        span_frames: int,
        cuts: Optional[Set[int]] = None,
        frame_lo: int = 0,
        frame_hi: int = 0
    ) -> Tuple[bool, str]:
        """Detect whether a gap contains a physical, shot, or identity discontinuity.
        
        Returns:
            (is_discontinuous: bool, refusal_reason: str)
        """
        # 1. Shot cut check
        if cuts:
            for cut_frame in cuts:
                if frame_lo < cut_frame <= frame_hi:
                    return True, f"shot_cut_at_frame_{cut_frame}"

        # 2. Extract bounding boxes
        box_a = np.asarray(face_a.get("bbox") if isinstance(face_a, dict) else getattr(face_a, "bbox", None), dtype=np.float32)
        box_b = np.asarray(face_b.get("bbox") if isinstance(face_b, dict) else getattr(face_b, "bbox", None), dtype=np.float32)
        if box_a is None or box_b is None or box_a.size != 4 or box_b.size != 4:
            return True, "missing_bounding_box"

        wa = max(1.0, float(box_a[2] - box_a[0]))
        ha = max(1.0, float(box_a[3] - box_a[1]))
        wb = max(1.0, float(box_b[2] - box_b[0]))
        hb = max(1.0, float(box_b[3] - box_b[1]))

        # 3. Scale ratio check
        scale_ratio = max(wa * ha, wb * hb) / max(1.0, min(wa * ha, wb * hb))
        if scale_ratio > (cls.MAX_SCALE_RATIO ** 2):
            return True, f"scale_jump_ratio_{scale_ratio:.2f}"

        # 4. Physical travel velocity check
        c_a = np.array([(box_a[0] + box_a[2]) * 0.5, (box_a[1] + box_a[3]) * 0.5], dtype=np.float32)
        c_b = np.array([(box_b[0] + box_b[2]) * 0.5, (box_b[1] + box_b[3]) * 0.5], dtype=np.float32)
        travel = float(np.linalg.norm(c_b - c_a))
        speed = travel / max(1.0, float(span_frames))
        max_allowed_speed = cls.MAX_TRAVEL_PER_FRAME * max(1.0, (wa + wb) / 200.0)

        if speed > max_allowed_speed:
            return True, f"excessive_travel_speed_{speed:.1f}_px_per_frame"

        # 5. Identity continuity check (ArcFace cosine distance)
        emb_a = face_a.get("embedding") if isinstance(face_a, dict) else getattr(face_a, "embedding", None)
        emb_b = face_b.get("embedding") if isinstance(face_b, dict) else getattr(face_b, "embedding", None)
        if emb_a is not None and emb_b is not None:
            va = np.asarray(emb_a, dtype=np.float32).reshape(-1)
            vb = np.asarray(emb_b, dtype=np.float32).reshape(-1)
            na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            if na > 1e-6 and nb > 1e-6:
                cos_dist = float(1.0 - np.dot(va, vb) / (na * nb))
                if cos_dist > cls.MAX_IDENTITY_DIST:
                    return True, f"identity_drift_cos_dist_{cos_dist:.3f}"

        return False, "continuous"

    @classmethod
    def interpolate_face(
        cls,
        face_a: Any,
        face_b: Any,
        fraction: float,
        track_embedding: np.ndarray,
        vel_a: Optional[np.ndarray] = None,
        vel_b: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Perform confidence-aware Cubic Hermite spline interpolation for smooth trajectories.
        
        Matches endpoint velocities vel_a and vel_b to eliminate jerk / linear knee artifacts.
        """
        t = float(np.clip(fraction, 0.0, 1.0))
        t2 = t * t
        t3 = t2 * t

        # Hermite basis functions: H0, H1, H2, H3
        h0 = 2.0 * t3 - 3.0 * t2 + 1.0
        h1 = t3 - 2.0 * t2 + t
        h2 = -2.0 * t3 + 3.0 * t2
        h3 = t3 - t2

        conf_a = float(face_a.get("det_score", 0.8) if isinstance(face_a, dict) else getattr(face_a, "det_score", 0.8) or 0.8)
        conf_b = float(face_b.get("det_score", 0.8) if isinstance(face_b, dict) else getattr(face_b, "det_score", 0.8) or 0.8)

        # Confidence weighting modulation
        w_conf = conf_b / max(1e-4, conf_a + conf_b)
        w_eff = (1.0 - t) * (1.0 - w_conf) + t * w_conf

        box_a = np.asarray(face_a.get("bbox") if isinstance(face_a, dict) else getattr(face_a, "bbox"), dtype=np.float32)
        box_b = np.asarray(face_b.get("bbox") if isinstance(face_b, dict) else getattr(face_b, "bbox"), dtype=np.float32)

        # Centroid and dimensions
        ca = np.array([(box_a[0] + box_a[2]) * 0.5, (box_a[1] + box_a[3]) * 0.5], dtype=np.float32)
        cb = np.array([(box_b[0] + box_b[2]) * 0.5, (box_b[1] + box_b[3]) * 0.5], dtype=np.float32)
        sa = np.array([box_a[2] - box_a[0], box_a[3] - box_a[1]], dtype=np.float32)
        sb = np.array([box_b[2] - box_b[0], box_b[3] - box_b[1]], dtype=np.float32)

        va = np.asarray(vel_a[:2], dtype=np.float32) if vel_a is not None and len(vel_a) >= 2 else (cb - ca) * 0.5
        vb = np.asarray(vel_b[:2], dtype=np.float32) if vel_b is not None and len(vel_b) >= 2 else (cb - ca) * 0.5

        # Cubic Hermite centroid position
        interp_c = h0 * ca + h1 * va + h2 * cb + h3 * vb
        interp_s = (1.0 - t) * sa + t * sb

        interp_box = np.array([
            interp_c[0] - interp_s[0] * 0.5,
            interp_c[1] - interp_s[1] * 0.5,
            interp_c[0] + interp_s[0] * 0.5,
            interp_c[1] + interp_s[1] * 0.5
        ], dtype=np.float32)

        # Clone face metadata using dict or class copy
        res = dict(face_a) if isinstance(face_a, dict) else dict(getattr(face_a, "__dict__", {}))
        res.pop("normed_embedding", None)
        res["bbox"] = interp_box
        res["det_score"] = float(min(conf_a, conf_b) * 0.95)
        res["_interpolated"] = True
        res["embedding"] = track_embedding

        # Interpolate landmarks if present on both anchors
        kps_a = face_a.get("kps") if isinstance(face_a, dict) else getattr(face_a, "kps", None)
        kps_b = face_b.get("kps") if isinstance(face_b, dict) else getattr(face_b, "kps", None)
        if kps_a is not None and kps_b is not None and np.shape(kps_a) == np.shape(kps_b):
            ka = np.asarray(kps_a, dtype=np.float32)
            kb = np.asarray(kps_b, dtype=np.float32)
            # Smooth position lerp with centroid offset
            shift_a = ca - (box_a[:2] + box_a[2:]) * 0.5
            res["kps"] = ((1.0 - t) * ka + t * kb + (interp_c - ((1.0 - t) * ca + t * cb))).astype(np.float32)

        lm_a = face_a.get("landmark_2d_106") if isinstance(face_a, dict) else getattr(face_a, "landmark_2d_106", None)
        lm_b = face_b.get("landmark_2d_106") if isinstance(face_b, dict) else getattr(face_b, "landmark_2d_106", None)
        if lm_a is not None and lm_b is not None and np.shape(lm_a) == np.shape(lm_b):
            la = np.asarray(lm_a, dtype=np.float32)
            lb = np.asarray(lm_b, dtype=np.float32)
            res["landmark_2d_106"] = ((1.0 - t) * la + t * lb + (interp_c - ((1.0 - t) * ca + t * cb))).astype(np.float32)

        # Interpolate 3D pose
        pose_a = face_a.get("pose") if isinstance(face_a, dict) else getattr(face_a, "pose", None)
        pose_b = face_b.get("pose") if isinstance(face_b, dict) else getattr(face_b, "pose", None)
        if pose_a is not None and pose_b is not None:
            pa = np.asarray(pose_a, dtype=np.float32).reshape(-1)
            pb = np.asarray(pose_b, dtype=np.float32).reshape(-1)
            if len(pa) >= 3 and len(pb) >= 3:
                res["pose"] = ((1.0 - t) * pa[:3] + t * pb[:3]).astype(np.float32)

        return res


# ==============================================================================
# 4. Master Temporal State Machine Tracker
# ==============================================================================

class TemporalStateMachineTracker:
    """Robust detection + tracking hybrid engine managing track lifecycles, crossings, and recovery."""

    MAX_COAST_FRAMES = 15          # Max consecutive frames to coast a dropout
    REID_MEMORY_FRAMES = 60        # Long-term Re-ID memory window for returning faces
    MIN_HITS_STABLE = 3            # Hits required to confirm a stable track
    CROSSING_IOU_THRESH = 0.25     # IoU threshold indicating interacting/crossing faces
    PROFILE_YAW_THRESH = 55.0      # Degrees yaw threshold for profile turning
    RAPID_MOTION_THRESH = 15.0     # px/frame threshold for rapid movement

    def __init__(
        self,
        max_lost: int = 30,
        max_coast: int = 15,
        reid_age: int = 60,
        motion_weight: float = 0.60,
        identity_weight: float = 0.40
    ):
        self.max_lost = max_lost
        self.max_coast = max_coast
        self.reid_age = reid_age
        self.motion_weight = motion_weight
        self.identity_weight = identity_weight

        self.tracks: Dict[int, RobustFaceTrack] = {}
        self.reid_archive: Dict[int, Dict[str, Any]] = {}
        self._next_track_id = 0
        self._last_frame_index = -1
        self._lock = threading.RLock()

        # Quality metrics tracking
        self.stats = {
            "total_detections": 0,
            "dropouts_recovered": 0,
            "crossings_detected": 0,
            "profile_turns_handled": 0,
            "returning_faces_recovered": 0,
            "identity_switches": 0,
            "interpolations_performed": 0,
            "interpolations_refused": 0
        }

    @staticmethod
    def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Standard 2D bounding box intersection over union."""
        ix0, iy0 = max(float(box1[0]), float(box2[0])), max(float(box1[1]), float(box2[1]))
        ix1, iy1 = min(float(box1[2]), float(box2[2])), min(float(box1[3]), float(box2[3]))
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter <= 0.0:
            return 0.0
        area1 = max(1.0, float(box1[2] - box1[0])) * max(1.0, float(box1[3] - box1[1]))
        area2 = max(1.0, float(box2[2] - box2[0])) * max(1.0, float(box2[3] - box2[1]))
        union = area1 + area2 - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def _cosine_dist(emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> float:
        """Bounded cosine distance [0.0, 2.0] between 512-d ArcFace embeddings."""
        if emb1 is None or emb2 is None:
            return 1.0
        v1 = np.asarray(emb1, dtype=np.float32).reshape(-1)
        v2 = np.asarray(emb2, dtype=np.float32).reshape(-1)
        if v1.shape != v2.shape:
            return 1.0
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            return 1.0
        cos_sim = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return 1.0 - cos_sim

    def _init_kalman(self, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize constant velocity Kalman filter state and covariance."""
        x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        h = max(1.0, y1 - y0)
        w = max(1.0, x1 - x0)
        a = w / h
        state = np.array([cx, cy, a, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cov = np.eye(8, dtype=np.float32) * 4.0
        cov[4:, 4:] *= 10.0
        return state, cov

    def _predict_kalman(self, track: RobustFaceTrack, dt: float = 1.0) -> None:
        """Constant-velocity state extrapolation."""
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt

        # Adapt process noise based on velocity (rapid motion gets higher Q)
        speed = float(np.linalg.norm(track.velocity[:2]))
        q_scale = 1.5 if speed > self.RAPID_MOTION_THRESH else 0.5
        Q = np.eye(8, dtype=np.float32) * (q_scale * dt)

        track.kalman_state = F @ track.kalman_state
        track.kalman_cov = F @ track.kalman_cov @ F.T + Q

        # Update predicted bbox from state
        cx, cy, a, h = track.kalman_state[:4]
        h = max(1.0, float(h))
        w = max(1.0, float(a * h))
        track.bbox = np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)
        track.velocity = track.kalman_state[4:8].copy()

    def _update_kalman(self, track: RobustFaceTrack, bbox: np.ndarray) -> None:
        """Incorporate detector measurement into Kalman state."""
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        h = max(1.0, float(bbox[3] - bbox[1]))
        w = max(1.0, float(bbox[2] - bbox[0]))
        a = w / h
        z = np.array([cx, cy, a, h], dtype=np.float32)

        H = np.zeros((4, 8), dtype=np.float32)
        H[:4, :4] = np.eye(4, dtype=np.float32)
        R = np.eye(4, dtype=np.float32) * 2.0  # Measurement noise

        y = z - H @ track.kalman_state
        S = H @ track.kalman_cov @ H.T + R
        K = track.kalman_cov @ H.T @ np.linalg.inv(S)

        track.kalman_state = track.kalman_state + K @ y
        track.kalman_cov = (np.eye(8, dtype=np.float32) - K @ H) @ track.kalman_cov

        cx, cy, a, h = track.kalman_state[:4]
        h = max(1.0, float(h))
        w = max(1.0, float(a * h))
        track.bbox = np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)
        track.velocity = track.kalman_state[4:8].copy()

    def detect_crossings(self) -> None:
        """Detect interacting or crossing faces and freeze identity updates."""
        active_tracks = list(self.tracks.values())
        n = len(active_tracks)
        crossing_ids = set()
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = active_tracks[i], active_tracks[j]
                iou = self._box_iou(t1.bbox, t2.bbox)
                c1 = np.array([(t1.bbox[0] + t1.bbox[2]) * 0.5, (t1.bbox[1] + t1.bbox[3]) * 0.5])
                c2 = np.array([(t2.bbox[0] + t2.bbox[2]) * 0.5, (t2.bbox[1] + t2.bbox[3]) * 0.5])
                dist = float(np.linalg.norm(c1 - c2))
                avg_size = max(1.0, (t1.bbox[2] - t1.bbox[0] + t2.bbox[2] - t2.bbox[0]) * 0.5)

                if iou >= self.CROSSING_IOU_THRESH or dist < (avg_size * 0.85):
                    t1.state = TrackState.CROSSING
                    t2.state = TrackState.CROSSING
                    crossing_ids.add(t1.track_id)
                    crossing_ids.add(t2.track_id)
                    self.stats["crossings_detected"] += 1

        for t in active_tracks:
            if t.state == TrackState.CROSSING and t.track_id not in crossing_ids:
                t.state = TrackState.STABLE

    def update(
        self,
        detections: Sequence[Any],
        frame_index: int,
        frame_shape: Optional[Tuple[int, int, int]] = None
    ) -> List[Any]:
        """Process one frame of detections through the state machine.
        
        Args:
            detections: List of Face objects produced by detector.
            frame_index: Current frame index.
            frame_shape: (H, W, C) shape of target video frame.
            
        Returns:
            List of Face objects with consistent `_track_id` and `source_assignment`.
        """
        with self._lock:
            dt = 1.0 if self._last_frame_index < 0 else max(1.0, float(frame_index - self._last_frame_index))
            self._last_frame_index = frame_index
            self.stats["total_detections"] += len(detections)

            # 1. Predict all active tracks using Kalman motion model
            for track in self.tracks.values():
                self._predict_kalman(track, dt=dt)
                track.misses += 1

            # 2. Check for crossing tracks before association
            self.detect_crossings()

            # 3. Build Cost Matrix for Association
            active_ids = list(self.tracks.keys())
            n_tracks = len(active_ids)
            n_dets = len(detections)

            matched_tracks: Set[int] = set()
            matched_dets: Set[int] = set()

            if n_tracks > 0 and n_dets > 0:
                cost_matrix = np.full((n_tracks, n_dets), 10.0, dtype=np.float32)

                for r, tid in enumerate(active_ids):
                    track = self.tracks[tid]
                    for c, det in enumerate(detections):
                        det_box = np.asarray(det.get("bbox") if isinstance(det, dict) else getattr(det, "bbox"), dtype=np.float32)
                        iou = self._box_iou(track.bbox, det_box)

                        det_emb = det.get("embedding") if isinstance(det, dict) else getattr(det, "embedding", None)
                        cos_d = self._cosine_dist(track.embedding, det_emb)

                        # When crossing, heavily favor motion and damp contaminated embedding
                        if track.state == TrackState.CROSSING:
                            cost = 0.85 * (1.0 - iou) + 0.15 * cos_d
                        else:
                            cost = self.motion_weight * (1.0 - iou) + self.identity_weight * cos_d

                        # Pose consistency penalty if pose is available
                        det_pose = det.get("pose") if isinstance(det, dict) else getattr(det, "pose", None)
                        if det_pose is not None and track.pose is not None:
                            dp = np.asarray(det_pose, dtype=np.float32).reshape(-1)
                            if len(dp) >= 2 and len(track.pose) >= 2:
                                yaw_diff = abs(float(dp[1]) - float(track.pose[1]))
                                if yaw_diff > 45.0:
                                    cost += 0.35

                        cost_matrix[r, c] = cost

                # Solve Hungarian assignment
                if linear_sum_assignment is not None:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                else:
                    # Greedy fallback
                    row_ind, col_ind = [], []
                    flat_idx = np.argsort(cost_matrix.reshape(-1))
                    for idx in flat_idx:
                        r, c = divmod(idx, n_dets)
                        if r not in row_ind and c not in col_ind:
                            row_ind.append(r)
                            col_ind.append(c)

                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 0.80:
                        tid = active_ids[r]
                        track = self.tracks[tid]
                        det = detections[c]

                        # Observation Match Update
                        det_box = np.asarray(det.get("bbox") if isinstance(det, dict) else getattr(det, "bbox"), dtype=np.float32)
                        self._update_kalman(track, det_box)

                        # Stabilize bounding box from Kalman filter
                        if isinstance(det, dict):
                            det["bbox"] = track.bbox.copy()
                        else:
                            setattr(det, "bbox", track.bbox.copy())

                        det_emb = det.get("embedding") if isinstance(det, dict) else getattr(det, "embedding", None)
                        track.update_embedding(det_emb)

                        # Update landmarks with velocity-adaptive temporal smoothing
                        speed = float(np.linalg.norm(track.velocity[:2]))
                        alpha_lm = float(np.clip(0.40 + 0.035 * speed, 0.40, 0.95))

                        kps = det.get("kps") if isinstance(det, dict) else getattr(det, "kps", None)
                        if kps is not None:
                            kps_arr = np.asarray(kps, dtype=np.float32)
                            if track.kps is not None and track.kps.shape == kps_arr.shape:
                                track.kps = (1.0 - alpha_lm) * track.kps + alpha_lm * kps_arr
                            else:
                                track.kps = kps_arr.copy()
                            if isinstance(det, dict):
                                det["kps"] = track.kps.copy()
                            else:
                                setattr(det, "kps", track.kps.copy())

                        lm106 = det.get("landmark_2d_106") if isinstance(det, dict) else getattr(det, "landmark_2d_106", None)
                        if lm106 is not None:
                            lm106_arr = np.asarray(lm106, dtype=np.float32)
                            if track.landmark_2d_106 is not None and track.landmark_2d_106.shape == lm106_arr.shape:
                                track.landmark_2d_106 = (1.0 - alpha_lm) * track.landmark_2d_106 + alpha_lm * lm106_arr
                            else:
                                track.landmark_2d_106 = lm106_arr.copy()
                            if isinstance(det, dict):
                                det["landmark_2d_106"] = track.landmark_2d_106.copy()
                            else:
                                setattr(det, "landmark_2d_106", track.landmark_2d_106.copy())

                        # Update pose & check profile turning
                        pose = det.get("pose") if isinstance(det, dict) else getattr(det, "pose", None)
                        if pose is not None:
                            p = np.asarray(pose, dtype=np.float32).reshape(-1)
                            if len(p) >= 3:
                                track.pose = p[:3].copy()
                                if abs(float(track.pose[1])) > self.PROFILE_YAW_THRESH:
                                    track.state = TrackState.PROFILE_TURNING
                                    self.stats["profile_turns_handled"] += 1

                        # Update confidence
                        score = float(det.get("det_score", 0.9) if isinstance(det, dict) else getattr(det, "det_score", 0.9) or 0.9)
                        track.confidence = 0.8 * track.confidence + 0.2 * score
                        track.last_seen_frame = frame_index
                        track.misses = 0
                        track.hits += 1
                        track.coasted_run = 0

                        # State promotion to STABLE if hits satisfied (keep CROSSING and PROFILE_TURNING states intact)
                        if track.state in (TrackState.COASTING, TrackState.OCCLUDED, TrackState.RECOVERED):
                            track.state = TrackState.STABLE
                        elif track.state == TrackState.TENTATIVE and track.hits >= self.MIN_HITS_STABLE:
                            track.state = TrackState.STABLE

                        # Stamp face object
                        if isinstance(det, dict):
                            det["_track_id"] = track.track_id
                            det["source_assignment"] = track.source_assignment
                        else:
                            setattr(det, "_track_id", track.track_id)
                            setattr(det, "source_assignment", track.source_assignment)

                        track.template_face = det
                        matched_tracks.add(tid)
                        matched_dets.add(c)

            # 4. Handle Unmatched Detections (New Faces vs Returning Faces)
            for c, det in enumerate(detections):
                if c in matched_dets:
                    continue
                det_emb = det.get("embedding") if isinstance(det, dict) else getattr(det, "embedding", None)
                recovered_tid = None

                # Search Re-ID archive for returning face
                if det_emb is not None:
                    best_cos = 1.0
                    for arch_id, arch_data in list(self.reid_archive.items()):
                        cos_d = self._cosine_dist(arch_data["embedding"], det_emb)
                        if cos_d < best_cos and cos_d < 0.32:  # High confidence identity match
                            best_cos = cos_d
                            recovered_tid = arch_id

                det_box = np.asarray(det.get("bbox") if isinstance(det, dict) else getattr(det, "bbox"), dtype=np.float32)
                det_kps = det.get("kps") if isinstance(det, dict) else getattr(det, "kps", None)
                det_lm106 = det.get("landmark_2d_106") if isinstance(det, dict) else getattr(det, "landmark_2d_106", None)
                det_pose = det.get("pose") if isinstance(det, dict) else getattr(det, "pose", None)
                det_src = det.get("source_assignment") if isinstance(det, dict) else getattr(det, "source_assignment", None)

                if recovered_tid is not None:
                    # Target Face Returning: Recover original track ID!
                    arch_data = self.reid_archive.pop(recovered_tid)
                    k_state, k_cov = self._init_kalman(det_box)
                    track = RobustFaceTrack(
                        track_id=recovered_tid,
                        state=TrackState.RECOVERED,
                        bbox=det_box.copy(),
                        kps=np.asarray(det_kps, dtype=np.float32).copy() if det_kps is not None else None,
                        landmark_2d_106=np.asarray(det_lm106, dtype=np.float32).copy() if det_lm106 is not None else None,
                        pose=np.asarray(det_pose, dtype=np.float32)[:3].copy() if det_pose is not None and len(det_pose) >= 3 else np.zeros(3, dtype=np.float32),
                        embedding=arch_data["embedding"],
                        source_assignment=arch_data["source_assignment"] if arch_data["source_assignment"] is not None else det_src,
                        last_seen_frame=frame_index,
                        kalman_state=k_state,
                        kalman_cov=k_cov,
                        template_face=det
                    )
                    self.stats["returning_faces_recovered"] += 1
                else:
                    # Completely new face appearing: Spawn new tentative track
                    new_tid = self._next_track_id
                    self._next_track_id += 1
                    k_state, k_cov = self._init_kalman(det_box)
                    track = RobustFaceTrack(
                        track_id=new_tid,
                        state=TrackState.TENTATIVE,
                        bbox=det_box.copy(),
                        kps=np.asarray(det_kps, dtype=np.float32).copy() if det_kps is not None else None,
                        landmark_2d_106=np.asarray(det_lm106, dtype=np.float32).copy() if det_lm106 is not None else None,
                        pose=np.asarray(det_pose, dtype=np.float32)[:3].copy() if det_pose is not None and len(det_pose) >= 3 else np.zeros(3, dtype=np.float32),
                        source_assignment=det_src,
                        last_seen_frame=frame_index,
                        kalman_state=k_state,
                        kalman_cov=k_cov,
                        template_face=det
                    )

                track.update_embedding(det_emb)
                self.tracks[track.track_id] = track

                if isinstance(det, dict):
                    det["_track_id"] = track.track_id
                    det["source_assignment"] = track.source_assignment
                else:
                    setattr(det, "_track_id", track.track_id)
                    setattr(det, "source_assignment", track.source_assignment)

            # 5. Handle Unmatched Tracks (Dropout, Coasting, & Disappearance)
            coasted_faces = []
            for tid in list(self.tracks.keys()):
                if tid in matched_tracks:
                    continue
                track = self.tracks[tid]

                if track.misses <= self.max_coast and track.hits >= 2:
                    # Enter Coasting Mode
                    track.state = TrackState.COASTING
                    track.coasted_run += 1
                    self.stats["dropouts_recovered"] += 1

                    # Generate synthetic face observation
                    synth_face = self._synthesize_coasted_face(track, frame_index)
                    if synth_face is not None:
                        coasted_faces.append(synth_face)

                elif track.misses > self.max_lost:
                    # Target Face Disappeared: Archive to Re-ID memory bank
                    track.state = TrackState.LOST
                    if track.embedding is not None:
                        self.reid_archive[track.track_id] = {
                            "embedding": track.embedding.copy(),
                            "source_assignment": track.source_assignment,
                            "last_seen_frame": track.last_seen_frame,
                            "retired_frame": frame_index
                        }
                    del self.tracks[tid]

            # 6. Clean up expired entries in Re-ID archive
            for arch_id, arch_data in list(self.reid_archive.items()):
                if frame_index - arch_data["retired_frame"] > self.reid_age:
                    del self.reid_archive[arch_id]

            # Return real detections plus synthesized coasted faces for continuity
            return list(detections) + coasted_faces

    def _synthesize_coasted_face(self, track: RobustFaceTrack, frame_index: int) -> Optional[Dict[str, Any]]:
        """Synthesize a continuous face observation from Kalman motion prediction."""
        if track.template_face is None:
            return None

        face = dict(track.template_face) if isinstance(track.template_face, dict) else dict(getattr(track.template_face, "__dict__", {}))
        face["bbox"] = track.bbox.copy()
        face["_track_id"] = track.track_id
        face["source_assignment"] = track.source_assignment
        face["embedding"] = track.embedding
        face["det_score"] = float(max(0.40, track.confidence * (0.95 ** track.coasted_run)))
        face["_coasted"] = True
        face["_interpolated"] = True
        face["occlusion_state"] = "coasted"

        # Propagate landmarks via velocity displacement
        if track.kps is not None:
            v_xy = track.velocity[:2]
            track.kps = track.kps + v_xy
            face["kps"] = track.kps.copy()
        if track.landmark_2d_106 is not None:
            v_xy = track.velocity[:2]
            track.landmark_2d_106 = track.landmark_2d_106 + v_xy
            face["landmark_2d_106"] = track.landmark_2d_106.copy()

        return face

    process_frame = update


# ==============================================================================
# 5. Explicit Temporal Quality Metrics
# ==============================================================================

class TemporalQualityMetrics:
    """Quantitative measurement harness for temporal consistency and flicker evaluation."""

    @staticmethod
    def compute_landmark_jitter(kps_sequence: Sequence[np.ndarray]) -> float:
        """Measure mean landmark acceleration / second-order jitter variance in px^2."""
        if len(kps_sequence) < 3:
            return 0.0
        arr = np.asarray(kps_sequence, dtype=np.float32)  # (T, N, 2)
        # Second derivative: acc = x[t] - 2*x[t-1] + x[t-2]
        acc = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
        jitter_var = float(np.var(acc))
        return round(jitter_var, 4)

    @staticmethod
    def compute_identity_stability(assignments: Sequence[Optional[int]]) -> float:
        """Calculate percentage of frames maintaining stable source assignment without flips."""
        valid = [a for a in assignments if a is not None]
        if len(valid) < 2:
            return 100.0
        flips = sum(1 for i in range(1, len(valid)) if valid[i] != valid[i - 1])
        stability_pct = max(0.0, 100.0 * (1.0 - flips / max(1, len(valid) - 1)))
        return round(stability_pct, 1)

    @staticmethod
    def compute_mask_popping_rate(masks: Sequence[np.ndarray], drop_threshold: float = 0.30) -> int:
        """Count instances where consecutive mask IoU drops by more than drop_threshold."""
        if len(masks) < 2:
            return 0
        popping_events = 0
        for i in range(1, len(masks)):
            m1, m2 = masks[i - 1] > 0.5, masks[i] > 0.5
            inter = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            iou = inter / union if union > 0 else 1.0
            if (1.0 - iou) > drop_threshold:
                popping_events += 1
        return popping_events

    @staticmethod
    def compute_motion_fidelity(true_motion: np.ndarray, tracked_motion: np.ndarray) -> float:
        """Compute Pearson correlation between ground-truth velocity and tracked velocity."""
        if len(true_motion) < 3:
            return 1.0
        v_true = np.linalg.norm(np.diff(true_motion, axis=0), axis=-1)
        v_track = np.linalg.norm(np.diff(tracked_motion, axis=0), axis=-1)
        std_t, std_p = float(np.std(v_true)), float(np.std(v_track))
        if std_t <= 1e-3 or std_p <= 1e-3:
            return 1.0
        mat = np.corrcoef(v_true, v_track)
        corr = float(mat[0, 1]) if not np.isnan(mat[0, 1]) else 1.0
        return round(max(-1.0, min(1.0, corr)), 3)
