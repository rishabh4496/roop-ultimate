"""Sparse auto-range pre-scan for timeline active range selection.

Rapidly samples video keyframes (e.g. 1 frame every 1-2 seconds) using
face detection and embedding similarity matching against target faces.
Frames where the target face is found are clustered into contiguous active ranges
with safety padding.

During rendering and tracking pre-pass:
- Active frames: full pipeline (face detection, swapping, enhancement, stabilization).
- Inactive frames: instant passthrough to output encoder queue (300-800+ FPS),
  bypassing all GPU neural network inference.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

import roop
from roop import session_pool
from roop.degrade import swallowed as _swallowed
from roop.face_analyser import check_face_matches_target
from roop.face_util import get_all_faces


def cluster_hits_into_ranges(
    hits: Sequence[int],
    total_frames: int,
    stride: int,
    padding: int,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Group hit indices into padded, merged (start, end) frame intervals and a boolean mask.

    Parameters
    ----------
    hits : Sequence[int]
        Frame indices (0-based) where a target face was detected.
    total_frames : int
        Total number of frames in the active clip.
    stride : int
        Sampling interval in frames.
    padding : int
        Safety margin in frames added to both sides of each active range.

    Returns
    -------
    ranges : List[Tuple[int, int]]
        List of merged (start_frame, end_frame) inclusive intervals.
    mask : np.ndarray
        Boolean array of shape (total_frames,) where True indicates an active frame.
    """
    if total_frames <= 0:
        return [], np.zeros(0, dtype=bool)

    if not hits:
        # Fail-safe: if no hits found, mark all frames as active to never erroneously skip
        return [(0, total_frames - 1)], np.ones(total_frames, dtype=bool)

    sorted_hits = sorted(hits)
    clusters: List[List[int]] = []
    current_cluster: List[int] = [sorted_hits[0]]

    # Cluster hits that are within 2 * stride of each other
    cluster_gap = max(stride * 2, 30)
    for h in sorted_hits[1:]:
        if h - current_cluster[-1] <= cluster_gap:
            current_cluster.append(h)
        else:
            clusters.append(current_cluster)
            current_cluster = [h]
    clusters.append(current_cluster)

    # Pad each cluster into [start, end] inclusive, clamped to bounds
    raw_ranges: List[Tuple[int, int]] = []
    for cl in clusters:
        start = max(0, min(cl) - padding)
        end = min(total_frames - 1, max(cl) + padding)
        raw_ranges.append((start, end))

    # Merge overlapping or touching ranges
    merged: List[Tuple[int, int]] = []
    for start, end in raw_ranges:
        if not merged:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

    # Build boolean mask
    mask = np.zeros(total_frames, dtype=bool)
    for start, end in merged:
        mask[start : end + 1] = True

    return merged, mask


def scan_active_ranges(
    source_video: str,
    target_faces: Optional[Sequence[Any]] = None,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_count: Optional[int] = None,
    fps: float = 25.0,
    stride: Optional[int] = None,
    padding: Optional[int] = None,
    threshold: Optional[float] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray, Dict[str, Any]]:
    """Scan video keyframes sparsely to determine active face-swap ranges.

    Parameters
    ----------
    source_video : str
        Path to the target video file.
    target_faces : Sequence[Any], optional
        List of reference target face objects/embeddings to match against.
        If empty or None, any detected face is treated as a hit.
    frame_start : int
        Starting frame index (0-based) of the clip.
    frame_end : int, optional
        Ending frame index of the clip.
    frame_count : int, optional
        Total frames in the clip (frame_end - frame_start).
    fps : float
        Frames per second of the video.
    stride : int, optional
        Step between sampled frames (default: ~1 sec = max(15, round(fps))).
    padding : int, optional
        Safety margin in frames added to active ranges (default: ~1 sec = max(15, round(fps))).
    threshold : float, optional
        Cosine distance threshold for face matching (lower is stricter).
    progress_cb : Callable[[int, int], None], optional
        Optional callback reporting (current_sample_idx, total_samples).

    Returns
    -------
    ranges : List[Tuple[int, int]]
        List of active (start, end) frame intervals relative to clip start (0-based).
    mask : np.ndarray
        Boolean array of length `total_frames` where True indicates active frames.
    stats : Dict[str, Any]
        Dictionary with telemetry metadata.
    """
    frame_start = max(0, int(frame_start or 0))

    if not source_video or not os.path.isfile(source_video):
        total = max(1, int(frame_count or 1))
        return [(0, total - 1)], np.ones(total, dtype=bool), {"error": "file_not_found"}

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        total = max(1, int(frame_count or 1))
        return [(0, total - 1)], np.ones(total, dtype=bool), {"error": "cannot_open_video"}

    try:
        video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 25.0)
        if actual_fps <= 0 or not np.isfinite(actual_fps):
            actual_fps = 25.0

        if frame_end is None or frame_end <= 0:
            frame_end = video_total_frames if video_total_frames > 0 else (frame_start + max(1, int(frame_count or 1)))

        total_frames = max(1, frame_end - frame_start)

        effective_stride = max(15, int(round(actual_fps))) if stride is None else max(1, int(stride))
        effective_padding = max(15, int(round(actual_fps))) if padding is None else max(1, int(padding))

        # If clip is very short, no need for sparse pre-scan
        if total_frames <= effective_stride * 2:
            return [(0, total_frames - 1)], np.ones(total_frames, dtype=bool), {
                "active_frames": total_frames,
                "total_frames": total_frames,
                "percentage": 100.0,
                "ranges": [(0, total_frames - 1)],
                "mode": "full_short_clip",
            }

        # Generate sample frame offsets
        sample_offsets = list(range(0, total_frames, effective_stride))
        if (total_frames - 1) not in sample_offsets:
            sample_offsets.append(total_frames - 1)

        # Multi-worker parallel scanning over sample keyframes using FaceAnalysis pool
        pool_size = (
            session_pool.detmask_pool_size()
            if hasattr(session_pool, 'detmask_pool_size') and session_pool.detmask_pooling_enabled()
            else 1
        )
        workers = max(1, min(pool_size * 2, 4))

        sample_q: Queue[Tuple[Optional[int], Optional[np.ndarray]]] = Queue(maxsize=16)

        def _read_samples():
            c = cv2.VideoCapture(source_video)
            try:
                for offset in sample_offsets:
                    abs_frame_idx = frame_start + offset
                    c.set(cv2.CAP_PROP_POS_FRAMES, abs_frame_idx)
                    ret, fr = c.read()
                    sample_q.put((offset, fr if ret else None))
            except Exception as _re:
                _swallowed("sparse_range_scan:read_samples", _re, "reader error")
            finally:
                sample_q.put((None, None))
                c.release()

        reader_t = Thread(target=_read_samples, daemon=True)
        reader_t.start()

        def _eval_sample(offset: int, frame: Optional[np.ndarray]) -> Tuple[int, bool]:
            if frame is None:
                return offset, False
            try:
                faces = get_all_faces(frame)
            except Exception as _exc:
                _swallowed("sparse_range_scan:get_all_faces", _exc, "sample frame skipped")
                faces = []
            if not faces:
                return offset, False
            if target_faces:
                for f in faces:
                    matched, _ = check_face_matches_target(f, list(target_faces), threshold=threshold)
                    if matched:
                        return offset, True
                return offset, False
            return offset, True

        hits: List[int] = []
        samples_done = 0
        total_samples = len(sample_offsets)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sparse_det") as executor:
            futures = []
            while True:
                offset, fr = sample_q.get()
                if offset is None:
                    break
                fut = executor.submit(_eval_sample, offset, fr)
                futures.append(fut)

            for fut in as_completed(futures):
                offset, is_hit = fut.result()
                if is_hit:
                    hits.append(offset)
                samples_done += 1
                if progress_cb is not None:
                    try:
                        progress_cb(samples_done, total_samples)
                    except Exception as _cb_exc:
                        _swallowed("sparse_range_scan:progress_cb", _cb_exc, "progress callback failed")

        reader_t.join(timeout=5.0)

        # If no hits detected, fail-safe to keeping all frames active
        if not hits:
            print("[SparseScan] No target face detected in sparse samples; safely processing full video.")
            return [(0, total_frames - 1)], np.ones(total_frames, dtype=bool), {
                "active_frames": total_frames,
                "total_frames": total_frames,
                "percentage": 100.0,
                "ranges": [(0, total_frames - 1)],
                "mode": "fail_safe_full",
            }

        ranges, mask = cluster_hits_into_ranges(hits, total_frames, effective_stride, effective_padding)
        active_count = int(np.sum(mask))
        percentage = round((active_count / total_frames) * 100.0, 1)

        stats = {
            "active_frames": active_count,
            "total_frames": total_frames,
            "skipped_frames": total_frames - active_count,
            "percentage": percentage,
            "ranges": ranges,
            "hits_count": len(hits),
            "samples_total": len(sample_offsets),
            "stride": effective_stride,
            "padding": effective_padding,
            "mode": "sparse_auto_range",
        }

        print(
            f"[SparseScan] Active ranges: {len(ranges)} segment(s) covering "
            f"{active_count}/{total_frames} frames ({percentage}%). "
            f"{total_frames - active_count} non-target frames will pass through at maximum speed."
        )

        return ranges, mask, stats

    finally:
        cap.release()
