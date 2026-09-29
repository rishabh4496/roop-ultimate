"""Temporal pre-pass scanner: where does the selected face appear in a video?

When the user picks a face in the React UI, one frame is a weak seed. This
module walks the target's timeline with a strided sequential decode, runs the
app's configured detector on each sampled frame (``face_util.get_all_faces``:
the same engine, TensorRT session and rescue ladder the render uses), threads
the detections through ``TemporalFaceTracker`` and reports, per tracklet, every
frame index, box, 5-point landmark set and detector score of that face.

Membership is decided per TRACKLET, not per frame. A same-person profile sits
0.7-1.0 cosine DISTANCE from a frontal seed (see api.target_auto_angles), so a
per-frame "similarity > 0.65" gate would silently drop every turned head and
the index would read complete while missing the hard poses. Instead the tracker
carries identity through the turn (motion + IoU + its own EMA embedding), and a
tracklet joins the reference when its best frames clear the threshold
(mean of its top ``_TOP_K`` similarities). Each detection still records its own
similarity so a caller can see which frames carried the decision.

Decoding never seeks: ``grab()`` advances over skipped frames and ``retrieve()``
converts only the sampled ones, so the long-GOP/HEVC seek defects documented in
``roop/capturer.py`` cannot return the wrong frame here. Nothing here allocates
GPU memory; the detector's sessions are the app's shared ones, and every
decoded frame and Face object is dropped before the next frame is read.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed
from roop.temporal_tracker import TemporalFaceTracker

logger = logging.getLogger("roop.scanner")

DEFAULT_SIMILARITY = 0.65
# Mean of a tracklet's best K frame similarities decides membership: one lucky
# frame on a look-alike is not enough, one frontal frame on a short tracklet is.
_TOP_K = 3
# Past this many frames the default stride widens from 3 to 5 (~5 min at 30 fps).
_LONG_VIDEO_FRAMES = 9000
# grab() failures in a row before the scan gives up on a file it believes has
# more frames (a corrupt packet costs one or two; a truncated file costs all).
_MAX_CONSECUTIVE_FAILURES = 25

# One scan at a time per process: the detector sessions are shared with the
# preview and the render, and a second concurrent scan only contends for them.
_SCAN_LOCK = threading.Lock()


def auto_step(frame_total: int) -> int:
    """The stride used when the caller passes step_frames <= 0 / None."""
    return 5 if frame_total > _LONG_VIDEO_FRAMES else 3


def _normed(vec: Any) -> Optional[np.ndarray]:
    if vec is None:
        return None
    try:
        out = np.asarray(vec, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if out.size == 0 or not np.all(np.isfinite(out)):
        return None
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 1e-7 else None


def _normed_bank(value: Any) -> Optional[np.ndarray]:
    """(k, d) unit rows from one embedding or a bank of them; None if unusable.

    A person captured at several angles is matched against their CLOSEST
    angle, the same min-distance rule the swap applies to an angle bank."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    rows = [r for r in (_normed(row) for row in arr) if r is not None]
    return np.stack(rows) if rows else None


def _face_field(face: Any, name: str) -> Any:
    if isinstance(face, dict):
        return face.get(name)
    try:
        return getattr(face, name, None)
    except Exception as exc:  # insightface Face.__getattr__ can raise on odd keys
        _swallowed("roop/scanner.py:_face_field", exc, "field read as missing")
        return None


def _face_embedding(face: Any) -> Optional[np.ndarray]:
    emb = _face_field(face, "normed_embedding")
    if emb is None:
        emb = _face_field(face, "embedding")
    return _normed(emb)


def _as_list(value: Any, shape: Tuple[int, ...]) -> Optional[list]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(shape)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return [round(float(v), 2) for v in arr] if len(shape) == 1 else \
        [[round(float(x), 2) for x in row] for row in arr]


@contextmanager
def open_capture(video_path: str) -> Iterator[cv2.VideoCapture]:
    """cv2.VideoCapture that is released on every exit path."""
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise IOError(f"cannot open video: {video_path}")
        yield capture
    finally:
        capture.release()


def ffprobe_warnings(video_path: str, limit: int = 20) -> List[str]:
    """Container/stream warnings ffprobe prints for this file (header level:
    a full decode check would cost as much as the scan)."""
    try:
        from roop.capturer import _ffprobe_binary, _popen_kwargs
        binary, kwargs = _ffprobe_binary(), _popen_kwargs()
    except Exception as exc:
        _swallowed("roop/scanner.py:ffprobe_warnings", exc, "bare ffprobe used")
        binary, kwargs = "ffprobe", {}
    try:
        proc = subprocess.run(
            [binary, "-v", "warning", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,nb_frames", "-of", "json", video_path],
            capture_output=True, text=True, timeout=20, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"ffprobe unavailable: {exc}"]
    lines = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
    return lines[:limit]


class TemporalPrePassScanner:
    """Index every occurrence of a reference face across a video.

    ``detect_fn(frame) -> list[Face]`` defaults to the app's configured
    detector; tests and benches inject their own. Faces must carry ``bbox``,
    ``kps`` and ``det_score``, and an embedding (``normed_embedding`` or
    ``embedding``) for membership to be decided.
    """

    def __init__(self, detect_fn: Optional[Callable[[np.ndarray], list]] = None,
                 similarity_threshold: float = DEFAULT_SIMILARITY,
                 rescue: bool = True,
                 progress: Optional[Callable[[int, int], None]] = None):
        self._detect_fn = detect_fn
        self.similarity_threshold = float(similarity_threshold)
        self.rescue = bool(rescue)
        self.progress = progress
        self._cancel = threading.Event()
        self._tracks: Dict[int, List[Dict[str, Any]]] = {}
        # Updated in place during a scan; read by ``progress`` callbacks.
        # tracklet_size is the detections in the largest tracklet that has
        # already cleared the threshold on some frame - a live estimate, the
        # final membership is decided on the top-K mean at the end.
        self.live: Dict[str, Any] = {}
        self.last_result: Optional[Dict[str, Any]] = None

    # ── public API ─────────────────────────────────────────────────────────
    async def scan_media(self, video_path: str, reference_embedding: np.ndarray,
                         step_frames: int = 3) -> Dict[str, Any]:
        """Scan off the event loop; see _scan for the result schema."""
        return await asyncio.to_thread(self._scan, video_path, reference_embedding, step_frames)

    def scan(self, video_path: str, reference_embedding: Any, step_frames: Optional[int] = 3) -> Dict[str, Any]:
        """Blocking form of scan_media, for a caller already on a worker thread.
        ``reference_embedding`` may be one vector or a (k, d) bank; a face's
        similarity is its best match over the bank."""
        return self._scan(video_path, reference_embedding, step_frames)

    def extract_tracklet_candidates(self, track_id: int) -> List[Dict[str, Any]]:
        """Every detection of ``track_id`` from the last scan, ranked as capture
        candidates: highest reference similarity first, then detector score,
        then face size. Unknown ids return []."""
        dets = self._tracks.get(int(track_id))
        if not dets:
            return []

        def _key(d):
            sim = d["similarity"] if d["similarity"] is not None else -2.0
            box = d["bbox"]
            return (sim, d["det_score"], (box[2] - box[0]) * (box[3] - box[1]))

        return [dict(d) for d in sorted(dets, key=_key, reverse=True)]

    def cancel(self) -> None:
        self._cancel.set()

    # ── implementation ─────────────────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> list:
        if self._detect_fn is not None:
            return list(self._detect_fn(frame) or [])
        from roop.face_util import get_all_faces
        return list(get_all_faces(frame, rescue=self.rescue) or [])

    def _scan(self, video_path: str, reference_embedding: Any, step_frames: Optional[int]) -> Dict[str, Any]:
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(video_path)
        reference = _normed_bank(reference_embedding)
        if reference is None:
            raise ValueError("reference_embedding is empty or not finite")
        self._cancel.clear()
        with _SCAN_LOCK:
            return self._scan_locked(video_path, reference, step_frames)

    def _scan_locked(self, video_path: str, reference: np.ndarray, step_frames: Optional[int]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self._tracks = {}
        tracks: Dict[int, List[Dict[str, Any]]] = {}
        failed_frames: List[int] = []
        dim_mismatch = 0
        frames_scanned = faces_seen = 0
        cancelled = False
        track_best: Dict[int, float] = {}
        tracklet_size = 0

        self._warn_if_cv2_differs(video_path)
        with open_capture(video_path) as capture:
            frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_total <= 0:
                from roop.capturer import get_video_frame_total
                frame_total = int(get_video_frame_total(video_path) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            step = int(step_frames) if step_frames and int(step_frames) > 0 else auto_step(frame_total)
            # The tracker counts misses in FRAMES; at stride N a single missed
            # sample is already N frames, so scale its tolerance to ~2 samples
            # or every detector blink fragments the tracklet.
            tracker = TemporalFaceTracker(max_misses=max(3, 2 * step),
                                          reid_age=max(45, 15 * step))
            frame_shape: Tuple[int, int] = (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                                            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
            self.live = {"frame_idx": 0, "frame_total": frame_total, "step_frames": step,
                         "frames_scanned": 0, "faces_seen": 0, "tracklet_size": 0,
                         "scan_fps": 0.0, "elapsed_s": 0.0}

            idx = -1
            decoded = 0
            pending_fail: List[int] = []
            while True:
                if self._cancel.is_set():
                    cancelled = True
                    break
                idx += 1
                if not capture.grab():
                    # EOF, or a corrupt packet short of the known end. Only a
                    # later successful grab proves it was a broken frame: the
                    # container's frame count routinely overstates by a few.
                    if frame_total and idx < frame_total and len(pending_fail) < _MAX_CONSECUTIVE_FAILURES:
                        pending_fail.append(idx)
                        continue
                    break
                decoded += 1
                if pending_fail:
                    failed_frames.extend(pending_fail)
                    pending_fail = []
                if idx % step:
                    continue
                ok, frame = capture.retrieve()
                if not ok or frame is None or frame.size == 0:
                    failed_frames.append(idx)
                    continue
                try:
                    faces = self._detect(frame)
                except Exception as exc:  # one bad frame must not end the scan
                    logger.warning("scanner: detector failed on frame %d: %s", idx, exc)
                    failed_frames.append(idx)
                    faces = []
                frames_scanned += 1
                faces_seen += len(faces)
                if faces:
                    result = tracker.update(faces, idx, frame.shape, detection_mode="full")
                    for det_index, track_id in result["assignments"].items():
                        record = self._record(faces[det_index], idx, reference)
                        if record is None:
                            continue
                        if record.pop("_dim_mismatch", False):
                            dim_mismatch += 1
                        tracks.setdefault(int(track_id), []).append(record)
                        tid = int(track_id)
                        if record["similarity"] is not None:
                            track_best[tid] = max(track_best.get(tid, -2.0), record["similarity"])
                        if track_best.get(tid, -2.0) >= self.similarity_threshold:
                            tracklet_size = max(tracklet_size, len(tracks[tid]))
                else:
                    tracker.update([], idx, frame.shape, detection_mode="full")
                # Frames are 6-25 MB each at 1080p-4K; do not let the loop
                # variables pin one across the next decode.
                del frame, faces
                frame_shape = frame_shape if all(frame_shape) else (int(frame.shape[0]), int(frame.shape[1]))
                if frames_scanned % 10 == 0:
                    elapsed_now = time.perf_counter() - t0
                    self.live.update(frame_idx=idx, frames_scanned=frames_scanned, faces_seen=faces_seen,
                                     tracklet_size=tracklet_size, elapsed_s=round(elapsed_now, 2),
                                     scan_fps=round(frames_scanned / elapsed_now, 1) if elapsed_now > 0 else 0.0)
                if self.progress is not None and frames_scanned % 10 == 0:
                    try:
                        self.progress(idx, frame_total)
                    except Exception as exc:
                        _swallowed("roop/scanner.py:progress", exc, "scan continued")
            tracker_stats = dict(tracker.stats)
            del tracker

        warnings: List[str] = []
        if failed_frames:
            warnings = ffprobe_warnings(video_path)
            logger.warning("scanner: %d unreadable frame(s) in %s (first %s); ffprobe: %s",
                           len(failed_frames), video_path, failed_frames[:5],
                           "; ".join(warnings) or "no warnings")
        if dim_mismatch:
            logger.warning("scanner: %d face embedding(s) did not match the reference's "
                           "dimension %d - is the reference from another recognizer?",
                           dim_mismatch, reference.shape[1])

        index, rejected = self._classify(tracks)
        self._tracks = tracks
        elapsed = time.perf_counter() - t0
        result = {
            "video_path": video_path,
            "frame_total": frame_total,
            "fps": fps,
            "frame_shape": list(frame_shape),
            "step_frames": step,
            "similarity_threshold": self.similarity_threshold,
            "frames_decoded": decoded,
            "frames_scanned": frames_scanned,
            "faces_seen": faces_seen,
            "failed_frames": failed_frames[:200],
            "failed_frame_count": len(failed_frames),
            # > 0 when decoding stopped short of the container's frame count
            # (truncated file, or a count that overstated the stream).
            "frames_short_of_total": max(0, frame_total - decoded) if frame_total and not cancelled else 0,
            "ffprobe_warnings": warnings,
            "cancelled": cancelled,
            "elapsed_s": round(elapsed, 3),
            # decode rate (every frame is grabbed) and detector rate, separately:
            # a fast scan that found fewer faces has not got faster.
            "decode_fps": round(decoded / elapsed, 1) if elapsed > 0 else 0.0,
            "scan_fps": round(frames_scanned / elapsed, 1) if elapsed > 0 else 0.0,
            "tracks": index,
            "rejected_tracks": rejected,
            "tracker_stats": tracker_stats,
        }
        self.last_result = result
        logger.info("scanner: %s - %d/%d frames sampled (step %d), %d faces, %d matching "
                    "tracklet(s) over %d frames, %d rejected, %.1f s",
                    os.path.basename(video_path), frames_scanned, decoded, step, faces_seen,
                    len(index), sum(len(t["frame_indices"]) for t in index),
                    len(rejected), elapsed)
        return result

    @staticmethod
    def _warn_if_cv2_differs(video_path: str) -> None:
        # HDR / high-bit-depth sources render through the ffmpeg working view;
        # cv2 shows them as flat gamma, so embeddings here run slightly off the
        # render's. Detection geometry is unaffected.
        try:
            from roop.capturer import _hdr_spec
            if _hdr_spec(video_path) is not None:
                logger.warning("scanner: %s is HDR - cv2 decodes it without the render's "
                               "tone mapping; similarities may read low", video_path)
        except Exception as exc:
            _swallowed("roop/scanner.py:_warn_if_cv2_differs", exc, "HDR check skipped")

    @staticmethod
    def _record(face: Any, frame_idx: int, reference: np.ndarray) -> Optional[Dict[str, Any]]:
        bbox = _as_list(_face_field(face, "bbox"), (4,))
        if bbox is None:
            return None
        kps = _as_list(_face_field(face, "kps"), (5, 2))
        score = _face_field(face, "det_score")
        emb = _face_embedding(face)
        similarity = None
        record: Dict[str, Any] = {}
        if emb is not None:
            if emb.shape[0] == reference.shape[1]:
                similarity = round(float(np.max(reference @ emb)), 4)
            else:
                record["_dim_mismatch"] = True
        record.update({
            "frame_idx": int(frame_idx),
            "bbox": bbox,
            "kps": kps,
            "det_score": round(float(score), 4) if score is not None else 0.0,
            "similarity": similarity,
        })
        return record

    def _classify(self, tracks: Dict[int, List[Dict[str, Any]]]):
        index, rejected = [], []
        for track_id, dets in sorted(tracks.items()):
            sims = sorted((d["similarity"] for d in dets if d["similarity"] is not None), reverse=True)
            score = float(np.mean(sims[:_TOP_K])) if sims else None
            entry = {
                "track_id": int(track_id),
                "score": None if score is None else round(score, 4),
                "frames": len(dets),
            }
            if score is not None and score >= self.similarity_threshold:
                entry.update({
                    "frame_indices": [d["frame_idx"] for d in dets],
                    # Frames below the threshold that the tracker carried: the
                    # turned/occluded poses a per-frame gate would have lost.
                    "carried_frames": sum(1 for d in dets if d["similarity"] is None
                                          or d["similarity"] < self.similarity_threshold),
                    "detections": dets,
                })
                index.append(entry)
            else:
                rejected.append(entry)
        index.sort(key=lambda t: t["score"], reverse=True)
        return index, rejected


__all__ = ["TemporalPrePassScanner", "open_capture", "ffprobe_warnings", "auto_step",
           "DEFAULT_SIMILARITY"]
