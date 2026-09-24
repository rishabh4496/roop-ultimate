"""Low-latency webcam capture, tracking, and virtual-camera output.

The batch renderer intentionally keeps its existing decode and temporal pipeline.
This module is a separate live path with a latest-frame mailbox, bounded optical
flow tracking, and no disk-backed intermediate frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import platform
import threading
import time
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed


@dataclass(frozen=True)
class LiveCaptureConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    buffer_size: int = 1


def capture_backend_for_system(system: Optional[str] = None) -> int:
    """Return the native OpenCV backend for the current desktop OS."""
    name = (system or platform.system()).lower()
    if name == "windows":
        return cv2.CAP_DSHOW
    if name == "linux":
        return cv2.CAP_V4L2
    if name == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


class LatestFrameMailbox:
    """A one-slot mailbox that always exposes the newest captured frame.

    A normal queue is the wrong primitive for live video. If inference falls
    behind, queued frames become stale and glass-to-glass latency grows without
    bound. ``put`` replaces the previous item and counts that replacement.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self._timestamp = 0.0
        self._sequence = 0
        self._closed = False
        self.dropped = 0

    def put(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        if frame is None:
            return
        with self._condition:
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self._timestamp = float(timestamp if timestamp is not None else time.perf_counter())
            self._sequence += 1
            self._condition.notify()

    def get(self, timeout: Optional[float] = None):
        with self._condition:
            if self._frame is None and not self._closed:
                self._condition.wait(timeout=timeout)
            if self._frame is None:
                return None
            item = (self._frame, self._timestamp, self._sequence)
            self._frame = None
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class LiveCapture:
    """Native webcam capture with a bounded latest-frame mailbox."""

    def __init__(self, config: LiveCaptureConfig):
        self.config = config
        self.mailbox = LatestFrameMailbox()
        self.capture = None
        self.thread = None
        self.active = False
        self.error = None
        self.backend = capture_backend_for_system()

    def open(self) -> None:
        self.capture = cv2.VideoCapture(int(self.config.camera_index), self.backend)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(
                f"Could not open camera {self.config.camera_index} with backend {self.backend}")

        # Drivers may ignore one or more of these requests. The mailbox remains
        # the hard latency bound even when the backend cannot expose its queue.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.height))
        self.capture.set(cv2.CAP_PROP_FPS, float(self.config.fps))
        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, int(self.config.buffer_size))
        except Exception as _degrade_error:
            _swallowed("live_mode.py:capture_buffer_size", _degrade_error,
                       "driver does not expose buffer size")

    def start(self) -> None:
        if self.active:
            return
        self.open()
        self.mailbox = LatestFrameMailbox()
        self.error = None
        self.active = True
        self.thread = threading.Thread(target=self._run, name="roop-live-capture", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while self.active and self.capture is not None:
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    self.error = "camera read failed"
                    break
                self.mailbox.put(frame)
        except Exception as exc:
            self.error = str(exc)
            _swallowed("live_mode.py:capture_loop", exc, "camera stopped")
        finally:
            self.active = False
            self.mailbox.close()

    def stop(self) -> None:
        self.active = False
        self.mailbox.close()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.5)
        self.thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None


@dataclass
class _TrackedFace:
    points: np.ndarray
    mask: np.ndarray
    output: Optional[np.ndarray] = None


@dataclass
class LiveFrameResult:
    frame: np.ndarray
    frame_index: int
    keyframe: bool
    tracking: bool
    scene_cut: bool
    latency_ms: float


class LiveFrameProcessor:
    """Run a swap on detector keyframes and Lucas-Kanade between them."""

    def __init__(
        self,
        swap_frame: Callable[[np.ndarray], np.ndarray],
        detect_faces: Callable[[np.ndarray], Iterable[object]],
        detector_interval: int = 6,
        scene_detector=None,
        flush_callback: Optional[Callable[[], object]] = None,
    ):
        self.swap_frame = swap_frame
        self.detect_faces = detect_faces
        self.detector_interval = max(1, int(detector_interval))
        self.scene_detector = scene_detector
        self.flush_callback = flush_callback
        self.frame_index = 0
        self.force_keyframe = True
        self.previous_gray = None
        self.last_output = None
        self.tracks: list[_TrackedFace] = []
        self.last_latency_ms = 0.0
        self.keyframes = 0
        self.tracked_frames = 0
        self.scene_cuts = 0

    @staticmethod
    def _face_points(face: object) -> Optional[np.ndarray]:
        for name in ("landmark_2d_106", "landmark_3d_68", "kps"):
            value = getattr(face, name, None)
            if value is None:
                continue
            points = np.asarray(value, dtype=np.float32)
            if points.ndim == 2 and points.shape[1] >= 2 and len(points) >= 3:
                points = points[:, :2]
                if np.isfinite(points).all():
                    return points.copy()
        return None

    @staticmethod
    def _mask_for_points(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        hull = cv2.convexHull(np.round(points).astype(np.int32))
        if len(hull) >= 3:
            cv2.fillConvexPoly(mask, hull, 255)
            # A small feather stops the low-resolution LK warp from producing
            # a hard polygon edge without paying for a second model pass.
            mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        return mask

    def _flush_tracking(self) -> None:
        self.previous_gray = None
        self.last_output = None
        self.tracks = []
        self.force_keyframe = True
        if self.flush_callback is not None:
            try:
                self.flush_callback()
            except Exception as _degrade_error:
                _swallowed("live_mode.py:flush_callback", _degrade_error,
                           "tracking reset continued")

    def reset(self) -> None:
        self._flush_tracking()
        self.frame_index = 0
        self.keyframes = 0
        self.tracked_frames = 0
        self.scene_cuts = 0
        if self.scene_detector is not None and hasattr(self.scene_detector, "reset"):
            self.scene_detector.reset()

    def _track_one(self, previous_gray: np.ndarray, current_gray: np.ndarray,
                   track: _TrackedFace):
        old = track.points.reshape(-1, 1, 2).astype(np.float32)
        new, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, old, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if new is None or status is None:
            return None
        keep = status.reshape(-1).astype(bool)
        if int(keep.sum()) < max(3, int(len(old) * 0.55)):
            return None
        old_good = old.reshape(-1, 2)[keep]
        new_good = new.reshape(-1, 2)[keep]
        if error is not None:
            err = error.reshape(-1)[keep]
            if len(err) and float(np.median(err)) > 35.0:
                return None
        matrix, _ = cv2.estimateAffinePartial2D(
            old_good, new_good, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if matrix is None or not np.isfinite(matrix).all():
            return None
        h, w = current_gray.shape[:2]
        warped_output = cv2.warpAffine(track.output if track.output is not None else self.last_output,
                                        matrix, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
        warped_mask = cv2.warpAffine(track.mask, matrix, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)
        track.points = new_good
        track.mask = warped_mask
        return warped_output, warped_mask

    def _tracked_frame(self, frame: np.ndarray):
        if self.previous_gray is None or self.last_output is None or not self.tracks:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = frame.copy()
        for track in self.tracks:
            old_points = track.points
            track.output = self.last_output
            tracked = self._track_one(self.previous_gray, gray, track)
            if tracked is None:
                self.force_keyframe = True
                return None
            warped_output, warped_mask = tracked
            alpha = (warped_mask.astype(np.float32) / 255.0)[..., None]
            output = (output.astype(np.float32) * (1.0 - alpha) +
                      warped_output.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
            if len(old_points) < 3:
                self.force_keyframe = True
                return None
        self.previous_gray = gray
        self.last_output = output
        self.tracked_frames += 1
        return output

    def _keyframe(self, frame: np.ndarray):
        output = self.swap_frame(frame)
        if output is None:
            output = frame
        faces = list(self.detect_faces(frame) or [])
        h, w = frame.shape[:2]
        tracks = []
        for face in faces:
            points = self._face_points(face)
            if points is None:
                continue
            mask = self._mask_for_points(points, (h, w))
            if int(np.count_nonzero(mask)) > 0:
                tracks.append(_TrackedFace(points=points, mask=mask))
        self.tracks = tracks
        self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_output = np.asarray(output).copy()
        self.force_keyframe = False
        self.keyframes += 1
        return self.last_output

    def process(self, frame: np.ndarray, captured_at: Optional[float] = None) -> LiveFrameResult:
        started = time.perf_counter()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("live frame is empty")

        scene_cut = False
        if self.scene_detector is not None:
            try:
                scene_cut = bool(self.scene_detector.observe_frame(frame, self.frame_index))
            except Exception as _degrade_error:
                _swallowed("live_mode.py:scene_detector", _degrade_error,
                           "scene detection disabled for this frame")
        if scene_cut:
            self.scene_cuts += 1
            self._flush_tracking()

        keyframe = self.force_keyframe or (self.frame_index % self.detector_interval == 0)
        tracking = False
        output = None
        if not keyframe:
            output = self._tracked_frame(frame)
            tracking = output is not None
        if output is None:
            keyframe = True
            output = self._keyframe(frame)

        now = time.perf_counter()
        capture_latency = (now - captured_at) * 1000.0 if captured_at else 0.0
        # captured_at already spans capture wait plus processing. Do not add the
        # processing interval a second time, or the audio delay would be doubled.
        latency_ms = max(0.0, capture_latency or (now - started) * 1000.0)
        self.last_latency_ms = latency_ms
        result = LiveFrameResult(output, self.frame_index, keyframe, tracking,
                                 scene_cut, latency_ms)
        self.frame_index += 1
        return result


@dataclass
class LiveStats:
    latency_budget_ms: float = 45.0
    frames: int = 0
    keyframes: int = 0
    tracked_frames: int = 0
    scene_cuts: int = 0
    dropped_capture_frames: int = 0
    last_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    processing_fps: float = 0.0
    budget_exceeded: int = 0
    started_at: float = 0.0

    def update(self, result: LiveFrameResult, dropped: int) -> None:
        self.frames += 1
        self.keyframes += int(result.keyframe)
        self.tracked_frames += int(result.tracking)
        self.scene_cuts += int(result.scene_cut)
        self.dropped_capture_frames = int(dropped)
        self.last_latency_ms = float(result.latency_ms)
        self.budget_exceeded += int(self.last_latency_ms > self.latency_budget_ms)
        if self.frames == 1:
            self.average_latency_ms = self.last_latency_ms
        else:
            self.average_latency_ms = self.average_latency_ms * 0.9 + self.last_latency_ms * 0.1
        elapsed = max(1e-6, time.perf_counter() - self.started_at)
        self.processing_fps = self.frames / elapsed
