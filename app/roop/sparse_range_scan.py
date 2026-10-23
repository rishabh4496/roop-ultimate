"""Sparse auto-range pre-scan (Stage 1): which frames can a swap possibly touch?

One sequential demuxer thread steps through the clip (or the trimmed
[frame_start, frame_end) interval) with ``cap.grab()`` and ``cap.retrieve()``s only the
sampled frames, so no worker ever seeks. A pool of detector workers, one per resident
FaceAnalysis instance, consumes a bounded queue; the main thread collects results.

- Base stride ``S = max(1, min(12, floor(0.20 * fps)))``.
- Every sample is compared with the previous one on a 64x36 luma thumbnail; a mean
  absolute difference above ``CUT_DELTA_Y`` means a hard cut fell between them, and the
  midpoint ``t - floor(S/2)`` is inspected too so a flash shorter than ``S`` is never
  stepped over. The midpoint frame is retrieved (a colour conversion, not a decode) and
  held one interval deep, because a sequential decoder cannot go back for it.
- Positive frames, with relaxed hysteresis floors:
    universal mode  any face with det_score >= max(0.25, det_thresh - 0.15)
    targeted mode   max cosine(face, target) >= max(0.15, tau_swap - 0.12), on w600k
                    embeddings, so B-roll of other people does not open Stage 2
- Each island of positives is dilated by ``M = ceil(1.25 * S) + 2`` frames on both ends and
  ranges whose gap is <= ``MERGE_GAP`` frames are coalesced.

During rendering and the tracking pre-pass active frames get the full pipeline; inactive
frames go straight to the encoder queue and skip every GPU network.

A clean scan that finds NOTHING returns ``[]`` and an all-False mask (the video really has
no frame to swap). A scan that could not look properly (unreadable file, detector or
decode errors, user stop) fails SAFE to the full range instead - "found nothing" and
"could not look" are different answers.
"""

from __future__ import annotations

import math
import os
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import roop
from roop.degrade import swallowed as _swallowed

STRIDE_FPS_FRACTION = 0.20      # S = floor(0.20 * fps) ...
STRIDE_MAX = 12                 # ... capped here
CUT_THUMB_SIZE = (64, 36)       # (w, h) of the luma thumbnail the cut test reads
CUT_DELTA_Y = 22.0              # mean abs luma difference (0-255) that means a hard cut
DET_RELAX = 0.15                # universal gate: det_thresh - 0.15 ...
DET_FLOOR = 0.25                # ... but never below this
SIM_RELAX = 0.12                # targeted gate: tau_swap - 0.12 ...
SIM_FLOOR = 0.15                # ... but never below this
PAD_FACTOR = 1.25               # dilation M = ceil(1.25 * S) + 2
PAD_EXTRA = 2
MERGE_GAP = 45                  # coalesce dilated ranges at most this many frames apart

_SENTINEL = object()            # demuxer -> worker: no more frames
_DONE = object()                # worker -> collector: this worker has exited
_POLL_S = 0.1                   # how often a blocked queue operation re-checks the stop flag


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested without a GPU or a video)
# --------------------------------------------------------------------------------------

def sparse_stride(fps: float) -> int:
    """Base sampling stride S = max(1, min(12, floor(0.20 * fps)))."""
    try:
        value = float(fps)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value <= 0:
        value = 25.0
    # fps / 5 rather than 0.2 * fps: 0.2 is not exact in binary and 30 * 0.2 must give 6.
    return max(1, min(STRIDE_MAX, int(math.floor(value / 5.0 + 1e-9))))


def dilation_pad(stride: int) -> int:
    """Symmetric pad M = ceil(1.25 * S) + 2 applied to both ends of every island."""
    return int(math.ceil(PAD_FACTOR * max(1, int(stride)))) + PAD_EXTRA


def det_score_floor(det_thresh: float) -> float:
    """Universal-mode face gate: max(0.25, det_thresh - 0.15)."""
    return max(DET_FLOOR, float(det_thresh) - DET_RELAX)


def similarity_floor(swap_similarity: float) -> float:
    """Targeted-mode embedding gate: max(0.15, tau_swap - 0.12), cosine SIMILARITY."""
    return max(SIM_FLOOR, float(swap_similarity) - SIM_RELAX)


def luma_thumb(frame: np.ndarray) -> np.ndarray:
    """64x36 float32 luma of a BGR (or gray) frame. Shrinks first, converts second."""
    small = cv2.resize(frame, CUT_THUMB_SIZE, interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small.astype(np.float32)


def luma_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two thumbnails (0-255)."""
    return float(np.mean(np.abs(a - b)))


def _field(face: Any, name: str) -> Any:
    value = getattr(face, name, None)
    if value is None and isinstance(face, dict):
        value = face.get(name)
    return value


def unit_embeddings(faces: Iterable[Any]) -> Optional[np.ndarray]:
    """(n, d) matrix of L2-normalised embeddings of `faces`, or None when none has one."""
    rows = []
    for face in faces or ():
        emb = _field(face, 'embedding')
        if emb is None:
            continue
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if vec.size and np.isfinite(norm) and norm > 0:
            rows.append(vec / norm)
    if not rows:
        return None
    width = rows[0].shape[0]
    rows = [r for r in rows if r.shape[0] == width]
    return np.stack(rows)


def frame_is_positive(faces: Sequence[Any], det_floor: float,
                      target_matrix: Optional[np.ndarray] = None,
                      sim_floor: Optional[float] = None) -> bool:
    """Relaxed hysteresis gate for one sampled frame.

    Universal (``target_matrix`` None): any face scoring >= ``det_floor``.
    Targeted: the best cosine similarity of any face against any target >= ``sim_floor``.
    A face with no embedding cannot vouch for a target and is ignored.
    """
    if not faces:
        return False
    if target_matrix is None:
        for face in faces:
            score = _field(face, 'det_score')
            if score is not None and float(score) >= det_floor:
                return True
        return False
    floor = SIM_FLOOR if sim_floor is None else float(sim_floor)
    face_matrix = unit_embeddings(faces)
    if face_matrix is None or face_matrix.shape[1] != target_matrix.shape[1]:
        return False
    return bool(float(np.max(face_matrix @ target_matrix.T)) >= floor)


def build_active_ranges(positives: Iterable[int], total_frames: int, stride: int,
                        trim_in: int = 0, trim_out: Optional[int] = None,
                        merge_gap: int = MERGE_GAP,
                        scene_cuts: Optional[Sequence[int]] = None) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Positive sample indices -> dilated, coalesced inclusive ranges plus a bool mask.

    Indices are clip-relative (0 = first frame of the trimmed interval), so the trim bounds
    default to [0, total_frames - 1]. Positives closer than ``stride`` frames form one
    island [a, b]; each island becomes [max(trim_in, a - M), min(trim_out, b + M)].

    When ``scene_cuts`` are provided, any island containing positive sample(s) is expanded
    to span the full shot(s) from the scene cut boundary where the shot started up to the
    scene cut boundary where the shot ended, preventing face unswaps at the beginning and
    end of cut scenes.

    Two ranges whose gap ``next_start - prev_end`` is <= ``merge_gap`` are merged. No positives
    -> ``[]`` and an all-False mask, never an exception.
    """
    if total_frames <= 0:
        return [], np.zeros(0, dtype=bool)
    lo = max(0, int(trim_in))
    hi = total_frames - 1 if trim_out is None else min(total_frames - 1, int(trim_out))
    mask = np.zeros(total_frames, dtype=bool)
    points = sorted({int(p) for p in positives if lo <= int(p) <= hi})
    if not points:
        return [], mask

    stride = max(1, int(stride))
    islands: List[Tuple[int, int]] = []
    a = b = points[0]
    for p in points[1:]:
        if p - b <= stride:
            b = p
        else:
            islands.append((a, b))
            a = b = p
    islands.append((a, b))

    pad = dilation_pad(stride)
    cuts_list = sorted({int(c) for c in (scene_cuts or ()) if lo <= int(c) <= hi})

    merged: List[Tuple[int, int]] = []
    for a, b in islands:
        start, end = max(lo, a - pad), min(hi, b + pad)
        if cuts_list:
            prior_cuts = [c for c in cuts_list if c <= a]
            if prior_cuts:
                shot_start = prior_cuts[-1]
                start = min(start, shot_start)
            post_cuts = [c for c in cuts_list if c > b]
            if post_cuts:
                shot_end = post_cuts[0] - 1
                end = max(end, shot_end)

        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for start, end in merged:
        mask[start:end + 1] = True
    return merged, mask


def _sample_grid_size(total_frames: int, stride: int) -> int:
    """Number of grid samples the demuxer will take: every S-th frame plus the last."""
    n = (total_frames + stride - 1) // stride
    return n + (0 if (total_frames - 1) % stride == 0 else 1)


# --------------------------------------------------------------------------------------
# Sequential demuxer
# --------------------------------------------------------------------------------------

def run_demux(cap: Any, frame_start: int, total_frames: int, stride: int,
              put: Callable[[Tuple[str, int, Optional[np.ndarray]]], bool],
              stopped: Callable[[], bool]) -> Dict[str, Any]:
    """Step through `total_frames` frames from `frame_start`, grab() everything,
    retrieve() only the sampled frames (plus the one midpoint a possible cut needs).

    ``put(('sample' | 'mid', clip_relative_index, frame_or_None))`` hands an item on and
    returns False when the scan was stopped (the demuxer then ends). The only seek is the
    single initial one to `frame_start`; there is no per-sample seeking. Returns telemetry.
    """
    info: Dict[str, Any] = dict(decoded=0, samples=0, mids=0, cuts=0, read_failures=0,
                                stopped=False, start_mismatch=False, scene_cuts=[])
    if frame_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        landed = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if landed is not None and int(landed) != frame_start:
            info['start_mismatch'] = True

    last = total_frames - 1
    prev_thumb: Optional[np.ndarray] = None
    mid_off: Optional[int] = None       # frame index to retrieve as the next interval's midpoint
    mid_frame: Optional[np.ndarray] = None
    mid_idx: Optional[int] = None       # index of the frame held in mid_frame
    next_sample = 0
    cuts_list: List[int] = []

    for off in range(total_frames):
        if stopped():
            info['stopped'] = True
            break
        if not cap.grab():
            break
        info['decoded'] = off + 1

        if off == mid_off:
            ok, fr = cap.retrieve()
            mid_frame, mid_idx = (fr, off) if ok else (None, None)
            mid_off = None

        if off != next_sample:
            continue

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            info['read_failures'] += 1
            frame = None
        else:
            thumb = luma_thumb(frame)
            if prev_thumb is not None and luma_delta(prev_thumb, thumb) > CUT_DELTA_Y:
                info['cuts'] += 1
                cuts_list.append(off)
                if mid_frame is not None:
                    info['mids'] += 1
                    if not put(('mid', mid_idx, mid_frame)):
                        info['stopped'] = True
                        break
            prev_thumb = thumb
        mid_frame = mid_idx = None          # a midpoint is only good for its own interval
        info['samples'] += 1
        if not put(('sample', off, frame)):
            info['stopped'] = True
            break

        if off >= last:
            next_sample = -1
        else:
            next_sample = min(off + stride, last)
            gap = next_sample - off
            mid_off = next_sample - gap // 2 if gap >= 2 else None

    info['scene_cuts'] = cuts_list
    return info


# --------------------------------------------------------------------------------------
# Detector backend
# --------------------------------------------------------------------------------------

class FaceAnalyserProbe:
    """Production backend: leases a FaceAnalysis instance per call (roop.face_util).

    ``prepare()`` runs on the calling (main) thread BEFORE any worker exists. Building the
    pool there is what runs the engine-hash-cached aux canary (face_util._guard_aux_models,
    synchronous inside _build_face_analyser) and loads every TensorRT engine once, so the
    workers never trigger a build, a canary or an engine-cache write concurrently.
    """

    def __init__(self, det_floor: float, target_matrix: Optional[np.ndarray] = None,
                 sim_floor: Optional[float] = None, rescue: bool = True):
        self.det_floor = float(det_floor)
        self.target_matrix = target_matrix
        self.sim_floor = sim_floor
        self.rescue = bool(rescue)
        self._guard = None
        self._pooled = None

    def prepare(self) -> int:
        from roop import face_util
        from roop.procmgr_runtime import _gpu_guard
        workers = face_util.prescan_pool_size()
        if self.target_matrix is not None and not face_util.prescan_has_recognition():
            print("[SparseScan] recognition model is not loaded; scanning in universal "
                  "(any face) mode instead of per-identity.", flush=True)
            self.target_matrix = None
        self._guard = _gpu_guard
        self._pooled = face_util.analysis_pooled
        return workers

    def _pass(self, frame: np.ndarray, turn: Optional[str]) -> bool:
        from roop import face_util
        with self._guard(pooled=self._pooled(), owner='analysis'):
            faces = face_util.detect_faces_for_prescan(
                frame, det_thresh=self.det_floor, embed=self.target_matrix is not None, turn=turn)
        return frame_is_positive(faces, self.det_floor, self.target_matrix, self.sim_floor)

    def positive(self, frame: np.ndarray) -> bool:
        """Upright pass first; if the frame is not positive, the three cardinal turns.

        Turns are tried on "not positive", not on "empty": at the relaxed threshold a junk
        detection makes a frame non-empty, which would turn the rescue off exactly when a
        sideways or inverted target needs it.
        """
        if self._pass(frame, None):
            return True
        if self.rescue:
            from roop import face_util
            for turn in face_util.PRESCAN_TURNS:
                if self._pass(frame, turn):
                    return True
        return False

    @property
    def targeted(self) -> bool:
        return self.target_matrix is not None


def _full_range(total: int, mode: str, **extra: Any) -> Tuple[List[Tuple[int, int]], np.ndarray, Dict[str, Any]]:
    total = max(1, int(total))
    stats = {"active_frames": total, "total_frames": total, "skipped_frames": 0,
             "percentage": 100.0, "ranges": [(0, total - 1)], "mode": mode}
    stats.update(extra)
    return [(0, total - 1)], np.ones(total, dtype=bool), stats


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def _scan_active_ranges_hw(
    source_video: str,
    target_faces: Optional[Sequence[Any]] = None,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_count: Optional[int] = None,
    fps: float = 25.0,
    stride: Optional[int] = None,
    swap_distance_threshold: Optional[float] = None,
    det_thresh: Optional[float] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    detector: Any = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray, Dict[str, Any]]:
    import time
    import roop.globals
    from roop.video_io_hw import NvSparseKeyframeReader, HwFrame
    from roop.scene_detector import HardwareBitstreamCutDetector
    import PyNvVideoCodec as nvc
    import torch

    t0 = time.perf_counter()
    with NvSparseKeyframeReader(source_video, adaptive_midpoints=True, max_gop_gap=60, enable_decode_stats=True) as reader:
        table = reader.keyframe_table()
        total_packets = reader._total_packet_count or (frame_count or 0)
        actual_fps = reader.fps if reader.fps > 0 else (fps or 25.0)
        if frame_end is None or frame_end <= 0:
            frame_end = total_packets if total_packets > 0 else (frame_start + max(1, int(frame_count or 1)))
        total_frames = max(1, frame_end - frame_start)

        S = max(1, int(stride)) if stride is not None else sparse_stride(actual_fps)
        M = dilation_pad(S)

        if total_frames <= S * 2:
            return _full_range(total_frames, "full_short_clip")

        if det_thresh is None:
            det_thresh = float(getattr(roop.globals, 'face_detector_threshold', 0.60))
        det_floor = det_score_floor(det_thresh)
        target_matrix = unit_embeddings(target_faces or ())

        sim_floor = None
        if target_matrix is not None:
            distance = swap_distance_threshold
            if distance is None:
                distance = float(getattr(roop.globals, 'face_distance_threshold', 0.65) or 0.65)
            sim_floor = similarity_floor(1.0 - float(distance))

        probe = detector if detector is not None else FaceAnalyserProbe(det_floor, target_matrix, sim_floor)
        worker_count = max(1, int(probe.prepare()))

        cut_detector = HardwareBitstreamCutDetector()
        positives: List[int] = []
        samples_done = 0
        detector_errors = 0

        # Build sample schedule: sample GOP keyframes (plus adaptive midpoints when GOP > 60)
        rel_kfs = [k - frame_start for k, _ in table if frame_start <= k < frame_end]
        gaps = [rel_kfs[i+1] - rel_kfs[i] for i in range(len(rel_kfs)-1)] if len(rel_kfs) > 1 else [total_frames]
        median_gap = np.median(gaps) if gaps else total_frames

        if median_gap < S:
            scheduled_samples = list(range(0, total_frames, S))
            if (total_frames - 1) not in scheduled_samples:
                scheduled_samples.append(total_frames - 1)
        else:
            scheduled_samples = list(rel_kfs) if rel_kfs else [0]
            expanded = []
            for i, idx in enumerate(scheduled_samples):
                expanded.append(idx)
                next_idx = scheduled_samples[i+1] if i + 1 < len(scheduled_samples) else total_frames
                if next_idx - idx > 60:
                    expanded.append(idx + (next_idx - idx) // 2)
            scheduled_samples = sorted(set(expanded))

        dec = nvc.CreateSimpleDecoder(source_video, reader.gpu_id, 0, 0, True, 0, 0, False, 1,
                                     nvc.OutputColorType.RGBP, False, reader.enable_decode_stats)

        total_samples = len(scheduled_samples)
        prev_idx = None

        def test_frame(abs_f, rel_f):
            nonlocal samples_done, detector_errors
            raw = dec[abs_f]
            t = torch.from_dlpack(raw).clone()
            stats = raw.ParseDecodeStats() if reader.enable_decode_stats else None
            is_cut = cut_detector.observe(abs_f, tensor=t, stats=stats)
            hw_f = HwFrame(abs_f, None, tensor=t, decode_stats=stats)
            bgr = hw_f.bgr()
            try:
                if probe.positive(bgr):
                    positives.append(rel_f)
            except Exception as exc:
                _swallowed("sparse_range_scan:hw_detect", exc, "face detect failed")
                detector_errors += 1
            samples_done += 1
            if progress_cb is not None:
                try:
                    progress_cb(samples_done, total_samples)
                except Exception as exc:
                    _swallowed("sparse_range_scan:progress_cb", exc, "progress callback error")
            return is_cut

        for s_idx in scheduled_samples:
            if should_stop and should_stop():
                break
            abs_idx = frame_start + s_idx
            is_cut = test_frame(abs_idx, s_idx)
            # If a scene cut fell between samples, inspect the midpoint and boundary frames
            if is_cut and prev_idx is not None and (s_idx - prev_idx) >= 2:
                mid_rel = prev_idx + (s_idx - prev_idx) // 2
                mid_abs = frame_start + mid_rel
                test_frame(mid_abs, mid_rel)
                if (s_idx - prev_idx) > 8:
                    if s_idx > 0 and (s_idx - 1) not in positives:
                        test_frame(frame_start + s_idx - 1, s_idx - 1)
                    if prev_idx + 1 < mid_rel and (prev_idx + 1) not in positives:
                        test_frame(frame_start + prev_idx + 1, prev_idx + 1)
            prev_idx = s_idx

        elapsed = time.perf_counter() - t0
        common = dict(
            stride=S, pad=M, workers=worker_count, det_floor=det_floor,
            sim_floor=sim_floor, targeted=bool(getattr(probe, 'targeted', target_matrix is not None)),
            samples_total=total_samples, samples_done=samples_done,
            cuts=len(cut_detector.scene_cuts), scene_cuts=list(cut_detector.scene_cuts),
            detector_errors=detector_errors, hits_count=len(positives),
            elapsed_s=round(elapsed, 3),
            scan_fps=round(samples_done / elapsed, 1) if elapsed > 0 else None,
            mode="nv_hardware_keyframes",
        )

        effective_stride = S
        effective_merge_gap = MERGE_GAP
        hw_cuts = [int(c) - frame_start for c in cut_detector.scene_cuts if frame_start <= int(c) < frame_end]

        ranges, mask = build_active_ranges(
            positives, total_frames, effective_stride,
            merge_gap=effective_merge_gap,
            scene_cuts=hw_cuts,
        )
        active_count = int(np.sum(mask))
        percentage = round((active_count / total_frames) * 100.0, 1)
        stats = dict(common, active_frames=active_count, total_frames=total_frames,
                     skipped_frames=total_frames - active_count, percentage=percentage,
                     ranges=ranges,
                     mode="sparse_auto_range" if ranges else "no_target_found")
        print(f"[SparseScan] Hardware Keyframe Pre-Scan complete in {elapsed:.2f}s: "
              f"{len(ranges)} active range(s), {len(cut_detector.scene_cuts)} scene cuts.", flush=True)
        return ranges, mask, stats


def scan_active_ranges(
    source_video: str,
    target_faces: Optional[Sequence[Any]] = None,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_count: Optional[int] = None,
    fps: float = 25.0,
    stride: Optional[int] = None,
    swap_distance_threshold: Optional[float] = None,
    det_thresh: Optional[float] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    detector: Any = None,
    capture: Any = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray, Dict[str, Any]]:
    """Scan the clip sparsely and return ``(ranges, mask, stats)``.

    Parameters
    ----------
    target_faces : faces whose identity must be present for a frame to count. Empty/None
        (or none carrying an embedding) means universal mode: any face counts. Only pass the
        faces being REPLACED (the video's targets), never the source faces.
    frame_start, frame_end, frame_count : the trimmed interval, absolute frame indices.
    stride : override of the fps-derived S (tests, diagnostics).
    swap_distance_threshold : the swap's cosine-DISTANCE threshold (options.face_distance_
        threshold); tau_swap = 1 - that.
    det_thresh : the detector's configured score threshold (default roop.globals).
    progress_cb : called on the calling thread as ``(samples_done, samples_total)``.
    should_stop : polled by the demuxer each frame; true aborts and fails safe to full.
    detector, capture : injection points for tests (a probe with ``prepare()``/``positive()``,
        a cv2.VideoCapture look-alike).

    Returns ranges relative to the clip start (inclusive), a bool mask of the clip length and
    telemetry. See the module docstring for the empty / fail-safe rules.
    """
    frame_start = max(0, int(frame_start or 0))

    if capture is None:
        if not source_video or not os.path.isfile(source_video):
            return _full_range(frame_count or 1, "fail_safe_full", error="file_not_found")
        from roop.video_io_hw import hw_io_wanted, HwUnavailable
        if hw_io_wanted():
            try:
                return _scan_active_ranges_hw(
                    source_video,
                    target_faces=target_faces,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    frame_count=frame_count,
                    fps=fps,
                    stride=stride,
                    swap_distance_threshold=swap_distance_threshold,
                    det_thresh=det_thresh,
                    progress_cb=progress_cb,
                    should_stop=should_stop,
                    detector=detector,
                )
            except HwUnavailable as exc:
                print(f"[SparseScan] Hardware keyframe pre-scan unavailable ({exc}); using fallback demuxer", flush=True)
            except Exception as exc:
                _swallowed("sparse_range_scan:hw_scan", exc, "hardware scan failed; using fallback demuxer")
        capture = cv2.VideoCapture(source_video)
    cap = capture
    if not cap.isOpened():
        cap.release()
        return _full_range(frame_count or 1, "fail_safe_full", error="cannot_open_video")

    demux_thread: Optional[threading.Thread] = None
    workers: List[threading.Thread] = []
    stop = threading.Event()
    try:
        video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 25.0)
        if actual_fps <= 0 or not np.isfinite(actual_fps):
            actual_fps = 25.0

        if frame_end is None or frame_end <= 0:
            frame_end = video_total_frames if video_total_frames > 0 else (
                frame_start + max(1, int(frame_count or 1)))
        total_frames = max(1, frame_end - frame_start)

        S = max(1, int(stride)) if stride is not None else sparse_stride(actual_fps)
        M = dilation_pad(S)

        # A clip this short is cheaper to just process.
        if total_frames <= S * 2:
            return _full_range(total_frames, "full_short_clip")

        if det_thresh is None:
            det_thresh = float(getattr(roop.globals, 'face_detector_threshold', 0.60))
        det_floor = det_score_floor(det_thresh)
        target_matrix = unit_embeddings(target_faces or ())
        sim_floor = None
        if target_matrix is not None:
            distance = swap_distance_threshold
            if distance is None:
                distance = float(getattr(roop.globals, 'face_distance_threshold', 0.65) or 0.65)
            sim_floor = similarity_floor(1.0 - float(distance))
        probe = detector if detector is not None else FaceAnalyserProbe(det_floor, target_matrix, sim_floor)

        # Barrier. Everything below this line may start threads; everything above runs on
        # the calling thread. prepare() returns only once the analyser pool is built, which
        # is after the aux canary has judged (and if need be rebuilt) every aux engine.
        worker_count = max(1, int(probe.prepare()))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 360
        warm = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(worker_count):       # one pass per instance: each leases a different one
            try:
                probe.positive(warm)
            except Exception as exc:
                _swallowed("sparse_range_scan:warmup", exc, "warm-up detect skipped")

        work_q: Queue = Queue(maxsize=worker_count * 2)
        result_q: Queue = Queue()
        info_box: Dict[str, Any] = {}

        def _put(item: Any) -> bool:
            while not stop.is_set():
                try:
                    work_q.put(item, timeout=_POLL_S)
                    return True
                except Full:
                    continue
            return False

        def _demux() -> None:
            try:
                info_box.update(run_demux(
                    cap, frame_start, total_frames, S, _put,
                    lambda: stop.is_set() or bool(should_stop and should_stop())))
            except Exception as exc:
                _swallowed("sparse_range_scan:demux", exc, "demuxer error")
                info_box['error'] = repr(exc)
            finally:
                for _ in range(worker_count):
                    _put(_SENTINEL)
                cap.release()

        busy_s: List[float] = []          # one entry per worker, appended as it exits (no lock needed)

        def _work() -> None:
            busy = 0.0
            try:
                while True:
                    try:
                        item = work_q.get(timeout=_POLL_S)
                    except Empty:
                        if stop.is_set():
                            return
                        continue
                    if item is _SENTINEL:
                        return
                    kind, idx, frame = item
                    if frame is None:
                        result_q.put((kind, idx, False, True))
                        continue
                    t_det = time.perf_counter()
                    try:
                        result_q.put((kind, idx, bool(probe.positive(frame)), False))
                    except Exception as exc:
                        _swallowed("sparse_range_scan:detect", exc, "sample frame skipped")
                        result_q.put((kind, idx, False, True))
                    busy += time.perf_counter() - t_det
            finally:
                busy_s.append(busy)
                result_q.put(_DONE)

        t0 = time.perf_counter()
        demux_thread = threading.Thread(target=_demux, name="sparse_demux", daemon=True)
        workers = [threading.Thread(target=_work, name=f"sparse_det{i}", daemon=True)
                   for i in range(worker_count)]
        for t in workers:
            t.start()
        demux_thread.start()

        positives: List[int] = []
        detector_errors = 0
        samples_done = 0
        total_samples = _sample_grid_size(total_frames, S)
        live_workers = worker_count
        while live_workers:
            item = result_q.get()
            if item is _DONE:
                live_workers -= 1
                continue
            kind, idx, positive, errored = item
            detector_errors += int(errored)
            if positive:
                positives.append(idx)
            if kind == 'sample':
                samples_done += 1
                if progress_cb is not None:
                    try:
                        progress_cb(samples_done, total_samples)
                    except Exception as exc:
                        _swallowed("sparse_range_scan:progress_cb", exc, "progress callback failed")
        demux_thread.join(timeout=5.0)
        elapsed = time.perf_counter() - t0

        info = dict(info_box)
        common = dict(stride=S, pad=M, workers=worker_count, det_floor=det_floor,
                      sim_floor=sim_floor, targeted=bool(getattr(probe, 'targeted', target_matrix is not None)),
                      samples_total=total_samples, samples_done=samples_done,
                      cuts=info.get('cuts', 0), mids_inspected=info.get('mids', 0),
                      decoded_frames=info.get('decoded', 0), read_failures=info.get('read_failures', 0),
                      detector_errors=detector_errors, hits_count=len(positives),
                      start_mismatch=bool(info.get('start_mismatch')),
                      elapsed_s=round(elapsed, 3),
                      scan_fps=round(info.get('decoded', 0) / elapsed, 1) if elapsed > 0 else None,
                      # Detector time summed over the pool / (workers x wall clock): 100% = no worker ever
                      # waited on the demuxer. Low means the pool is starved by decode, not that it is slow.
                      worker_busy_s=round(sum(busy_s), 3),
                      worker_utilization_pct=(round(100.0 * sum(busy_s) / (worker_count * elapsed), 1)
                                              if elapsed > 0 and worker_count else None))

        # A scan that did not finish cannot say which frames are inactive, so it never
        # returns a partial answer. One that finished but could not look at some samples
        # (read or detector errors, or nothing decoded) is trusted only if it found something.
        reason = None
        if info.get('stopped'):
            reason = "stopped"
        elif info.get('error'):
            reason = "demux error"
        elif info.get('start_mismatch'):
            reason = "initial seek landed on another frame"    # every range would be shifted
        elif not positives:
            if info.get('decoded', 0) == 0:
                reason = "no frame decoded"
            elif detector_errors or info.get('read_failures', 0):
                reason = f"{detector_errors} detector / {info.get('read_failures', 0)} read error(s)"
        if reason is not None:
            print(f"[SparseScan] Scan not usable ({reason}); safely processing the full video.",
                  flush=True)
            return _full_range(total_frames, "fail_safe_full", reason=reason, **common)

        ranges, mask = build_active_ranges(positives, total_frames, S, scene_cuts=info.get('scene_cuts'))
        active_count = int(np.sum(mask))
        percentage = round((active_count / total_frames) * 100.0, 1)
        stats = dict(common, active_frames=active_count, total_frames=total_frames,
                     skipped_frames=total_frames - active_count, percentage=percentage,
                     ranges=ranges,
                     mode="sparse_auto_range" if ranges else "no_target_found")
        if not ranges:
            print("[SparseScan] No target face found in any sampled frame; every frame will "
                  "pass through unchanged.", flush=True)
        else:
            print(f"[SparseScan] Active ranges: {len(ranges)} segment(s) covering "
                  f"{active_count}/{total_frames} frames ({percentage}%). "
                  f"{total_frames - active_count} non-target frames will pass through at maximum speed.",
                  flush=True)
        return ranges, mask, stats

    finally:
        stop.set()
        for t in workers:
            t.join(timeout=5.0)
        if demux_thread is not None:
            demux_thread.join(timeout=5.0)
        cap.release()
