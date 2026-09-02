"""Face analysis interface and early fast-path bypass.

Provides detection, recognition, and early fast-path bypass routing for non-target
and empty frames (0 faces or similarity < threshold), routing raw frames directly
to encoder queues and completely bypassing face enhancement, restorer modules,
stabilizer matrices, and GPU tensor allocations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import (compose_affines, pre_rotate_face_crop,
                            profile_alignment_matrix, transform_points)
from roop.face_util import (
    FACE_ANALYSER,
    FACE_ANALYSER_POOL,
    get_face_analyser,
    lease_face_analyser,
    release_face_analyser,
    release_face_analyser_aux,
    get_all_faces,
    get_first_face,
    get_all_faces_in_roi,
    detect_boxes_in_roi,
    get_all_faces_hires,
    align_crop,
    analysis_pooled,
    estimate_norm,
)


def _face_field(face: Any, name: str, default: Any = None) -> Any:
    """Read an InsightFace ``Face``, mapping, or small test-double uniformly."""
    if isinstance(face, dict):
        return face.get(name, default)
    try:
        value = face.get(name, default)
    except (AttributeError, TypeError):
        value = getattr(face, name, default)
    return default if value is None else value


def _set_face_field(face: Any, name: str, value: Any) -> None:
    """Stamp metadata without imposing a concrete detector-face type."""
    try:
        face[name] = value
        return
    except (TypeError, AttributeError):
        pass
    setattr(face, name, value)


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection-over-union for ``[left, top, right, bottom]`` boxes."""
    ax0, ay0, ax1, ay1 = (float(v) for v in first)
    bx0, by0, bx1, by1 = (float(v) for v in second)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    second_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-6 else 0.0


def _cosine_similarity(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> float:
    """Return a bounded ArcFace cosine similarity; missing embeddings are neutral."""
    if first is None or second is None:
        return 0.0
    a = np.asarray(first, dtype=np.float32).reshape(-1)
    b = np.asarray(second, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or not len(a):
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-6:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


@dataclass
class FaceTrack:
    """One constant-velocity face track in centre/aspect/height space."""

    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    embedding: Optional[np.ndarray]
    hits: int = 1
    missed: int = 0


class FaceTracker:
    """Small deterministic Kalman/Hungarian tracker for detector dispatch.

    The state is exactly ``[x, y, a, h, dx, dy, da, dh]``: box centre,
    aspect ratio, height, and their constant velocities.  ArcFace appearance
    and predicted-box IoU are solved together, preventing the left-to-right
    detector ordering from exchanging sources when people cross.
    """

    _MOTION_WEIGHT = 0.6
    _IDENTITY_WEIGHT = 0.4

    def __init__(self, max_age: int = 30, max_cost: float = 0.85,
                 process_noise: float = 1.0, measurement_noise: float = 4.0):
        self.max_age = max(0, int(max_age))
        self.max_cost = float(max_cost)
        self.process_noise = max(1e-6, float(process_noise))
        self.measurement_noise = max(1e-6, float(measurement_noise))
        self.tracks: Dict[int, FaceTrack] = {}
        self._next_track_id = 0
        self._last_frame_index: Optional[int] = None
        self._lock = RLock()

    @staticmethod
    def bbox_to_measurement(bbox: Sequence[float]) -> np.ndarray:
        """Convert an xyxy box to the Kalman measurement ``[x, y, a, h]``."""
        x0, y0, x1, y1 = (float(v) for v in bbox)
        height = max(1e-3, y1 - y0)
        width = max(1e-3, x1 - x0)
        return np.asarray(((x0 + x1) * 0.5, (y0 + y1) * 0.5,
                           width / height, height), dtype=np.float32)

    @staticmethod
    def measurement_to_bbox(measurement: Sequence[float]) -> np.ndarray:
        """Convert ``[x, y, a, h]`` back to an xyxy box."""
        x, y, aspect, height = (float(v) for v in measurement)
        height = max(1e-3, height)
        width = max(1e-3, aspect * height)
        return np.asarray((x - width * 0.5, y - height * 0.5,
                           x + width * 0.5, y + height * 0.5), dtype=np.float32)

    @staticmethod
    def _embedding(face: Any) -> Optional[np.ndarray]:
        embedding = _face_field(face, 'embedding')
        if embedding is None:
            return None
        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            return vector.copy() if vector.size and np.isfinite(vector).all() else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        transition = np.eye(8, dtype=np.float32)
        transition[0, 4] = transition[1, 5] = dt
        transition[2, 6] = transition[3, 7] = dt
        return transition

    def _predict(self, track: FaceTrack, dt: float) -> None:
        transition = self._transition(dt)
        # Velocity uncertainty should grow faster than measurement uncertainty.
        q = self.process_noise * np.diag((dt * dt, dt * dt, 0.01 * dt * dt,
                                           dt * dt, dt, dt, 0.01 * dt, dt)).astype(np.float32)
        track.state = transition @ track.state
        track.state[2] = max(track.state[2], 1e-3)
        track.state[3] = max(track.state[3], 1e-3)
        track.covariance = transition @ track.covariance @ transition.T + q

    def _update(self, track: FaceTrack, measurement: np.ndarray,
                embedding: Optional[np.ndarray]) -> None:
        observation = np.zeros((4, 8), dtype=np.float32)
        observation[np.arange(4), np.arange(4)] = 1.0
        innovation = measurement - observation @ track.state
        innovation_covariance = (observation @ track.covariance @ observation.T +
                                 np.eye(4, dtype=np.float32) * self.measurement_noise)
        gain = track.covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
        track.state = track.state + gain @ innovation
        track.state[2] = max(track.state[2], 1e-3)
        track.state[3] = max(track.state[3], 1e-3)
        track.covariance = (np.eye(8, dtype=np.float32) - gain @ observation) @ track.covariance
        track.hits += 1
        track.missed = 0
        if embedding is not None:
            # Never blend an identity discontinuity into a stable ArcFace mean.
            if track.embedding is None or _cosine_similarity(track.embedding, embedding) >= 0.5:
                base = embedding if track.embedding is None else 0.9 * track.embedding + 0.1 * embedding
                norm = float(np.linalg.norm(base))
                track.embedding = (base / norm if norm > 1e-6 else base).astype(np.float32)

    def _new_track(self, measurement: np.ndarray,
                   embedding: Optional[np.ndarray]) -> FaceTrack:
        track = FaceTrack(
            track_id=self._next_track_id,
            state=np.concatenate((measurement, np.zeros(4, dtype=np.float32))).astype(np.float32),
            covariance=np.diag((16.0, 16.0, 0.04, 16.0,
                                64.0, 64.0, 0.16, 64.0)).astype(np.float32),
            embedding=embedding,
        )
        self.tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def association_cost_matrix(self, detections: Sequence[Any]) -> np.ndarray:
        """Return the specified IoU/ArcFace cost for active tracks and detections."""
        active = [self.tracks[track_id] for track_id in sorted(self.tracks)]
        matrix = np.empty((len(active), len(detections)), dtype=np.float32)
        for row, track in enumerate(active):
            predicted_bbox = self.measurement_to_bbox(track.state[:4])
            for column, face in enumerate(detections):
                detection_bbox = _face_field(face, 'bbox')
                if detection_bbox is None:
                    matrix[row, column] = np.inf
                    continue
                iou = _bbox_iou(predicted_bbox, detection_bbox)
                similarity = _cosine_similarity(track.embedding, self._embedding(face))
                matrix[row, column] = (self._MOTION_WEIGHT * (1.0 - iou) +
                                       self._IDENTITY_WEIGHT * (1.0 - similarity))
        return matrix

    def update(self, detections: Sequence[Any], frame_index: Optional[int] = None) -> List[Any]:
        """Associate detections, stamp ``_track_id``, and expire stale tracks."""
        faces = list(detections or ())
        with self._lock:
            if frame_index is None:
                frame_index = 0 if self._last_frame_index is None else self._last_frame_index + 1
            frame_index = int(frame_index)
            if self._last_frame_index is None:
                dt = 1.0
            else:
                # A worker may finish an old frame after a newer one.  It may use
                # the current prediction, but must not rewind the persistent state.
                dt = max(1.0, float(frame_index - self._last_frame_index))
            for track in self.tracks.values():
                self._predict(track, dt)
                track.missed += 1

            cost = self.association_cost_matrix(faces)
            matched_detections = set()
            if cost.size:
                rows, columns = linear_sum_assignment(cost)
                active_ids = sorted(self.tracks)
                for row, column in zip(rows, columns):
                    value = float(cost[row, column])
                    if not np.isfinite(value) or value > self.max_cost:
                        continue
                    track = self.tracks[active_ids[int(row)]]
                    bbox = _face_field(faces[int(column)], 'bbox')
                    self._update(track, self.bbox_to_measurement(bbox),
                                 self._embedding(faces[int(column)]))
                    _set_face_field(faces[int(column)], '_track_id', track.track_id)
                    matched_detections.add(int(column))

            for index, face in enumerate(faces):
                if index in matched_detections:
                    continue
                bbox = _face_field(face, 'bbox')
                if bbox is None:
                    continue
                track = self._new_track(self.bbox_to_measurement(bbox), self._embedding(face))
                _set_face_field(face, '_track_id', track.track_id)

            for track_id in tuple(self.tracks):
                if self.tracks[track_id].missed > self.max_age:
                    del self.tracks[track_id]
            if self._last_frame_index is None or frame_index > self._last_frame_index:
                self._last_frame_index = frame_index
        return faces


def face_roll_degrees(face: Any) -> float:
    """Return the resolved roll when present, otherwise derive it from the eyes."""
    try:
        value = face.get('roll_deg') if isinstance(face, dict) else getattr(face, 'roll_deg', None)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    except (TypeError, ValueError):
        pass
    try:
        kps = face.get('kps') if isinstance(face, dict) else getattr(face, 'kps', None)
        left_eye, right_eye = np.asarray(kps, dtype=np.float32).reshape(5, 2)[:2]
        return float(np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                           right_eye[0] - left_eye[0])))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def face_yaw_pitch(face: Any) -> Tuple[float, float]:
    """Read target pose stamped by the tracker, with a 5-point solver fallback."""
    def _value(name):
        return face.get(name) if isinstance(face, dict) else getattr(face, name, None)
    try:
        yaw, pitch = _value('_adaptive_yaw'), _value('_adaptive_pitch')
        if yaw is not None and pitch is not None:
            return float(yaw), float(pitch)
    except (TypeError, ValueError):
        pass
    try:
        from roop.face_util import solve_pose_5pt
        pose = solve_pose_5pt(_value('kps'))
        if pose is not None:
            return float(pose[0]), float(pose[1])
    except Exception:
        pass
    return 0.0, 0.0


def profile_anchors(face: Any, yaw_degrees: float) -> Optional[np.ndarray]:
    """Find visible tragus/ear, nose tip and chin centre from refined landmarks.

    A detector may optionally stamp ``profile_anchors`` explicitly.  Otherwise
    the lateral contour and lowest refined point supply the visible ear/tragus
    and chin.  Without 106-point refinement this returns ``None`` so callers
    retain the well-defined five-point path rather than inventing an ear.
    """
    def _value(name):
        return face.get(name) if isinstance(face, dict) else getattr(face, name, None)
    explicit = _value('profile_anchors')
    if explicit is not None:
        try:
            anchors = np.asarray(explicit, dtype=np.float32).reshape(3, 2)
            return anchors if np.isfinite(anchors).all() else None
        except (TypeError, ValueError):
            return None
    try:
        refined = np.asarray(_value('landmark_2d_106'), dtype=np.float32).reshape(-1, 2)
        kps = np.asarray(_value('kps'), dtype=np.float32).reshape(5, 2)
        if refined.shape[0] < 20 or not np.isfinite(refined).all():
            return None
        # Keep the lower face from being chosen as the lateral ear contour.
        upper = refined[refined[:, 1] <= np.percentile(refined[:, 1], 80)]
        if len(upper) == 0:
            return None
        ear = upper[np.argmax(upper[:, 0]) if yaw_degrees >= 0.0 else np.argmin(upper[:, 0])]
        chin = refined[np.argmax(refined[:, 1])]
        return np.asarray((ear, kps[2], chin), dtype=np.float32)
    except (TypeError, ValueError, AttributeError):
        return None


def adaptive_alignment_matrix(kps: np.ndarray, image_size: int, mode: str,
                              yaw_degrees: float = 0.0,
                              anchors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
    """Choose standard five-point or non-squashing profile alignment."""
    if abs(float(yaw_degrees)) > 45.0 and anchors is not None:
        matrix = profile_alignment_matrix(anchors, image_size, yaw_degrees)
        if matrix is not None:
            return matrix, 'profile_3pt'
    return estimate_norm(np.asarray(kps, dtype=np.float32).reshape(5, 2), image_size, mode), 'five_point'


def canonicalize_face_alignment(image: Frame, face: Any, image_size: int, mode: str,
                                dst: Optional[np.ndarray] = None):
    """Pre-upright a rolled face and return its canonical crop plus paste affine.

    ``paste_matrix`` maps original target coordinates directly to canonical crop
    coordinates.  Its inverse is therefore the exact un-rotation/paste mapping
    used before alpha blending, with no second resample of the swapped result.
    """
    def _value(name):
        return face.get(name) if isinstance(face, dict) else getattr(face, name, None)
    kps = np.asarray(_value('kps'), dtype=np.float32).reshape(5, 2)
    bbox = np.asarray(_value('bbox'), dtype=np.float32).reshape(4)
    yaw, pitch = face_yaw_pitch(face)
    roll = face_roll_degrees(face)
    upright, pre_rotation, inverse_pre_rotation, applied = pre_rotate_face_crop(image, bbox, roll)
    upright_kps = transform_points(kps, pre_rotation)
    anchors = profile_anchors(face, yaw)
    if anchors is not None:
        anchors = transform_points(anchors, pre_rotation)
    local_matrix, alignment_kind = adaptive_alignment_matrix(
        upright_kps, image_size, mode, yaw_degrees=yaw, anchors=anchors)
    paste_matrix = compose_affines(local_matrix, pre_rotation)
    if dst is not None:
        cv2.warpAffine(upright, local_matrix, (image_size, image_size), dst=dst,
                       borderMode=cv2.BORDER_REPLICATE)
        aligned = dst
    else:
        aligned = cv2.warpAffine(upright, local_matrix, (image_size, image_size),
                                 borderMode=cv2.BORDER_REPLICATE)
    return aligned, paste_matrix, {
        'yaw': yaw, 'pitch': pitch, 'roll': roll, 'pre_rotation': pre_rotation,
        'inverse_pre_rotation': inverse_pre_rotation, 'applied_roll_prerotation': applied,
        'alignment_kind': alignment_kind,
    }


def has_face(frame: Frame) -> bool:
    """Fast check whether any face is present in the frame."""
    if frame is None:
        return False
    face = get_first_face(frame)
    return face is not None


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity in [0, 1] range (1.0 = identical)."""
    if emb1 is None or emb2 is None:
        return 0.0
    v1 = np.asarray(emb1, dtype=np.float32).flatten()
    v2 = np.asarray(emb2, dtype=np.float32).flatten()
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    dot = float(np.dot(v1, v2))
    sim = dot / (n1 * n2)
    # Clamp to [-1.0, 1.0] then map to [0, 1] or return direct cosine similarity
    return max(-1.0, min(1.0, sim))


def check_face_matches_target(
    face: Any,
    target_faces: List[Any],
    threshold: Optional[float] = None
) -> Tuple[bool, float]:
    """Check if face matches any target face above similarity threshold (or below distance threshold).
    
    Returns (matches: bool, best_score: float).
    """
    if not target_faces:
        return True, 1.0
    emb = getattr(face, "embedding", None)
    if emb is None and isinstance(face, dict):
        emb = face.get("embedding")
    if emb is None:
        return False, 0.0

    # Default distance threshold from globals/options (lower distance = higher similarity)
    dist_thresh = float(threshold if threshold is not None
                        else getattr(roop.globals, "face_distance_threshold", 0.65) or 0.65)

    best_sim = -1.0
    min_dist = float("inf")

    for target in target_faces:
        t_emb = getattr(target, "embedding", None)
        if t_emb is None and isinstance(target, dict):
            t_emb = target.get("embedding")
        if t_emb is None:
            continue
        
        sim = compute_cosine_similarity(emb, t_emb)
        dist = 1.0 - sim  # Cosine distance
        if dist < min_dist:
            min_dist = dist
            best_sim = sim

    # Match passes if distance <= threshold
    matches = bool(min_dist <= dist_thresh)
    return matches, best_sim


def evaluate_fast_path(
    frame: Frame,
    target_faces: Optional[List[Any]] = None,
    threshold: Optional[float] = None,
    precomputed_faces: Optional[List[Any]] = None,
) -> Tuple[bool, List[Any]]:
    """Determine whether the frame should take the early fast-path bypass.
    
    Returns (should_bypass: bool, detected_faces: List[Face]).
    If should_bypass is True:
        The frame has 0 faces or similarity score < threshold.
        The caller must route the raw input frame directly to the encoder output queue,
        completely bypassing face enhancement, restorer modules, stabilizer matrices,
        and GPU tensor allocations.
    """
    if frame is None:
        return True, []

    # 1. Check precomputed faces if available
    if precomputed_faces is not None:
        faces = list(precomputed_faces)
    else:
        faces = get_all_faces(frame)

    # If 0 faces found -> immediate bypass
    if not faces:
        return True, []

    # If target faces are specified, check if ANY detected face matches
    if target_faces:
        any_matched = False
        for face in faces:
            matched, _ = check_face_matches_target(face, target_faces, threshold=threshold)
            if matched:
                any_matched = True
                break
        if not any_matched:
            # All tracks/faces failed to match target source -> bypass!
            return True, faces

    return False, faces


# ── Heterogeneous detector offload (OpenVINO NPU / integrated GPU) ─────────
#
# The detector, its landmark models and the occlusion mask are small graphs
# that run once per face, while the swapper and the restorer are the large
# ones.  On an Intel machine with an NPU or an integrated GPU there is idle
# silicon that could take the small graphs and leave the discrete card to the
# swap and restoration models, which is where this pipeline's GPU time
# actually goes.
#
# THIS IS OFF BY DEFAULT, and the reason is worth stating rather than leaving
# as an accident:
#
#   1. Neither validated machine can execute it.  The RTX 4070 desktop and the
#      RTX 3060 laptop both run an ORT build whose providers are exactly
#      ('TensorrtExecutionProvider', 'CUDAExecutionProvider',
#      'CPUExecutionProvider') -- no OpenVINOExecutionProvider, and no
#      `openvino` package.  Every number in this repo's records comes from one
#      of those two boxes, so there is no measurement to support switching it
#      on and this project's standing rule is that a default change must prove
#      no regression.
#
#   2. The blast radius is the whole pipeline, not one stage.  Detection
#      output feeds every identity gate, the track binding and the alignment
#      matrices.  A different kernel's rounding here does not show up as a
#      slower render, it shows up as a different face being swapped -- and the
#      swap audit counts intent, not outcome, so it would read 100% either way.
#
# So this ships as a real, wired, testable path that a user on Intel hardware
# can turn on explicitly, and as an honest no-op everywhere else.  When it is
# requested but cannot be satisfied, it says so once instead of pretending.

_OPENVINO_ENV = "ROOP_DETECTOR_OPENVINO"

# Only the small per-face graphs are ever offloaded.  The swapper and the
# restorers stay on the discrete GPU: moving them is the opposite of the point.
_OFFLOADABLE = frozenset(("face_detection", "masking", "recognition"))

_openvino_state: Dict[str, Any] = {}
_openvino_lock = RLock()


def openvino_available() -> bool:
    """Whether the installed ONNX Runtime build LISTS the OpenVINO EP.

    Listed is not usable, and for this EP the gap is unusually dangerous --
    see openvino_device_usable().
    """
    try:
        from roop.backend_manager import provider_available
        return provider_available("OpenVINOExecutionProvider")
    except Exception:
        return False


def _probe_model() -> bytes:
    """A one-node ONNX graph, built in memory, for the usability probe."""
    from onnx import TensorProto, helper
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node], "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    return model.SerializeToString()


def openvino_device_usable(device: str) -> bool:
    """Whether the OpenVINO EP genuinely ACTIVATES for *device* on this box.

    THIS EP FAILS OPEN, which is why a probe is needed rather than a listing.
    Measured 2026-09-03 against onnxruntime-openvino 1.23.0 on this machine,
    asking for a device the system does not have:

        device_type=CPU    build 0.4s  EP active   -> ran on OpenVINO
        device_type=GPU    build 0.3s  EP ABSENT   -> ran on CPUExecutionProvider
        device_type=NPU    build 0.3s  EP ABSENT   -> ran on CPUExecutionProvider
        device_type=GPU.1  build 0.4s  EP ABSENT   -> ran on CPUExecutionProvider

    In every failing row `InferenceSession` returned a working session and
    raised NOTHING; ORT logged to stderr and quietly dropped the provider.  A
    mismatched OpenVINO runtime version does the same thing.  So neither
    `build_session_with_fallback` nor any try/except can see this: the only
    reliable signal is asking the constructed session which providers it ended
    up with, which is what this does -- once per device, on a one-node graph.
    """
    key = f"usable::{device}"
    with _openvino_lock:
        if key in _openvino_state:
            return _openvino_state[key]
    usable = False
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(
            _probe_model(),
            providers=[("OpenVINOExecutionProvider", {"device_type": device}),
                       "CPUExecutionProvider"])
        usable = "OpenVINOExecutionProvider" in session.get_providers()
    except Exception:
        usable = False
    with _openvino_lock:
        _openvino_state[key] = usable
    return usable


def openvino_devices() -> Tuple[str, ...]:
    """Device strings OpenVINO reports on this machine, e.g. NPU, GPU.1, CPU.

    Queried through the ``openvino`` runtime when it is importable.  The ORT
    provider can be present without it, so an empty tuple means "cannot
    enumerate", not "no devices" -- callers treat an explicitly requested
    device as usable in that case and let the session build decide.
    """
    with _openvino_lock:
        if "devices" in _openvino_state:
            return _openvino_state["devices"]
    devices: Tuple[str, ...] = ()
    try:
        import openvino as ov
        devices = tuple(str(d) for d in ov.Core().available_devices)
    except Exception:
        devices = ()
    with _openvino_lock:
        _openvino_state["devices"] = devices
    return devices


def detector_offload_target(requested: Optional[str] = None) -> Optional[str]:
    """Resolve the configured offload device, or None when it is not usable.

    ``requested`` accepts an explicit device string ("NPU", "GPU.1"), "auto"
    to take the best available accelerator, or an off value.  Unset is off.
    """
    raw = requested if requested is not None else os.environ.get(_OPENVINO_ENV, "")
    value = str(raw or "").strip()
    if value.lower() in ("", "0", "off", "false", "no", "none"):
        return None
    if not openvino_available():
        _warn_once("openvino_missing",
                   f"[Detector] {_OPENVINO_ENV}={value} but this ONNX Runtime "
                   f"build has no OpenVINOExecutionProvider; detection stays on "
                   f"the primary provider.")
        return None
    devices = openvino_devices()
    if value.lower() in ("auto", "1", "on", "true", "yes"):
        # Prefer the NPU, then a secondary GPU.  GPU.0 is skipped deliberately:
        # OpenVINO 2026.3 on this machine enumerates GPU as
        # "NVIDIA GeForce RTX 4070 (dGPU)", i.e. the first GPU slot really can
        # be the discrete card this offload exists to keep free.  Taking it
        # would move detection onto the very device we are trying to unload,
        # through a slower runtime than the TensorRT path it already has.
        for candidate in ("NPU", "GPU.1"):
            if candidate in devices and openvino_device_usable(candidate):
                return candidate
        _warn_once("openvino_auto_none",
                   f"[Detector] {_OPENVINO_ENV}=auto found no usable NPU or "
                   f"secondary GPU (devices: {', '.join(devices) or 'none reported'}); "
                   f"detection stays on the primary provider.")
        return None
    if devices and value not in devices:
        _warn_once(f"openvino_absent_{value}",
                   f"[Detector] {_OPENVINO_ENV}={value} is not among the "
                   f"OpenVINO devices on this machine "
                   f"({', '.join(devices) or 'none reported'}); detection stays "
                   f"on the primary provider.")
        return None
    # The listing can be absent or stale, and the EP fails OPEN on a device it
    # cannot serve -- a session that silently runs on CPU while every log line
    # says the detector was offloaded. Confirm the EP actually activates for
    # this device before routing anything to it.
    if not openvino_device_usable(value):
        _warn_once(f"openvino_inactive_{value}",
                   f"[Detector] {_OPENVINO_ENV}={value}: the OpenVINO provider "
                   f"is listed but does not activate for that device (ORT would "
                   f"fall back to CPU without raising); detection stays on the "
                   f"primary provider.")
        return None
    return value


def _warn_once(key: str, message: str) -> None:
    with _openvino_lock:
        if key in _openvino_state:
            return
        _openvino_state[key] = True
    print(message)


def detector_providers(providers: Sequence, model_key: str = "face_detection",
                       requested: Optional[str] = None) -> List:
    """Prepend the OpenVINO EP for a small per-face model, when enabled.

    The primary chain is kept behind it rather than replaced, so any operator
    OpenVINO cannot take still falls back to CUDA/CPU inside the same session.
    Models outside ``_OFFLOADABLE`` are returned untouched.
    """
    chain = list(providers or ())
    family = str(model_key or "").split(":", 1)[0].strip().lower()
    if family not in _OFFLOADABLE:
        return chain
    device = detector_offload_target(requested)
    if device is None:
        return chain
    if any("openvino" in str(p[0] if isinstance(p, (tuple, list)) else p).lower()
           for p in chain):
        return chain
    options = {"device_type": device}
    precision = os.environ.get("ROOP_DETECTOR_OPENVINO_PRECISION", "").strip()
    if precision:
        options["precision"] = precision
    _warn_once(f"openvino_active_{device}_{family}",
               f"[Detector] routing {family} models to OpenVINO {device}; the "
               f"primary provider is retained as in-session fallback.")
    return [("OpenVINOExecutionProvider", options)] + chain


def offload_report(requested: Optional[str] = None) -> dict:
    """JSON-safe diagnostics for the panel and the self-verification test."""
    target = detector_offload_target(requested)
    return {
        "env": _OPENVINO_ENV,
        "configured": (requested if requested is not None
                       else os.environ.get(_OPENVINO_ENV, "") or "off"),
        "provider_available": openvino_available(),
        "devices": list(openvino_devices()),
        "target": target,
        "offloadable_models": sorted(_OFFLOADABLE),
        "active": target is not None,
    }
