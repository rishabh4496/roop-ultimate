"""Which frames SAM 2 is allowed to look at.

The SAM 2 mask engine used to decode and propagate the WHOLE trimmed clip from a memory bank seeded on frame 0,
whatever Stage 1 and Stage 2 had already proved: on the Phase 5 test clip 444 of 600 frames hold no target face
and were tracked anyway, the frames were buffered in RAM (12.6 MB each at 1080p), and the one memory bank ran
across every cut and every empty stretch, so a mask could drift from one appearance of a person into the next.

A RUN is a maximal stretch of frames that are consecutive in the Stage 2 frame plan (`a_{i+1} - b_i > 1` starts a
new one). Each run is propagated on its own fresh memory bank, seeded with the boxes of the faces the plan holds
for the run's FIRST frame - the matched targets, not whatever happened to be on frame 0 - and nothing outside a
run is decoded, buffered or tracked. Without a plan (temporal detection off) the Stage 1 active ranges are the
runs, seeded by a detection on each range's first frame; without either, the clip is one run, as before.

Pure functions, no models: `tests/test_sam2_runs.py`.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Run = Tuple[int, int, Optional[List[np.ndarray]]]       # (first, last inclusive, seed boxes or None = detect)


def runs_from_plan(plan: Mapping[int, Sequence[Any]], frame_count: Optional[int] = None) -> List[Run]:
    """Runs of consecutive plan frames, seeded with the plan's own boxes at each run's first frame."""
    frames = sorted(int(f) for f in plan if f is not None and int(f) >= 0
                    and (frame_count is None or int(f) < int(frame_count)) and plan[f])
    spans: List[List[int]] = []
    for f in frames:
        if spans and f - spans[-1][1] == 1:
            spans[-1][1] = f
        else:
            spans.append([f, f])
    runs: List[Run] = []
    for first, last in spans:
        boxes = []
        for rec in plan[first]:
            bbox = getattr(rec, "bbox", None)
            if bbox is None and isinstance(rec, Mapping):
                bbox = rec.get("bbox")
            if bbox is not None:
                boxes.append(np.asarray(bbox, dtype=np.float32).reshape(4).copy())
        runs.append((first, last, boxes))
    return runs


def runs_from_ranges(ranges: Optional[Sequence[Sequence[int]]], frame_count: int) -> List[Run]:
    """Stage 1's inclusive ranges as runs with no seeds (the caller detects on each first frame)."""
    out: List[Run] = []
    for a, b in ranges or []:
        lo, hi = max(int(a), 0), min(int(b), int(frame_count) - 1)
        if hi >= lo:
            out.append((lo, hi, None))
    return out


def whole_clip(frame_count: int) -> List[Run]:
    return [(0, max(int(frame_count) - 1, 0), None)]


def choose(plan: Optional[Mapping[int, Sequence[Any]]], ranges: Optional[Sequence[Sequence[int]]],
           frame_count: int) -> Tuple[List[Run], str]:
    """The runs SAM 2 will use and where they came from: 'plan' | 'ranges' | 'clip'."""
    if plan:
        runs = runs_from_plan(plan, frame_count)
        if runs:
            return runs, "plan"
    if ranges:
        runs = runs_from_ranges(ranges, frame_count)
        if runs:
            return runs, "ranges"
    return whole_clip(frame_count), "clip"


def coverage(runs: Sequence[Run], frame_count: int) -> Tuple[int, float]:
    n = sum(last - first + 1 for first, last, _ in runs)
    return n, (100.0 * n / frame_count if frame_count else 0.0)
