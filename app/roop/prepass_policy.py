"""The Stage 2 pre-pass knobs that are read in more than one place, in one place.

`prepass_detect_interval` (K)
    Run the face detector on every K-th frame of the tracking pre-pass (frame 0, K, 2K, ... counted from the
    last full detection, so a shot cut restarts the count) and carry the faces between by optical flow, then
    bridge each hole between two real detections (`procmgr_tracking._build_temporal_faces`). 1 detects on
    every frame, exactly as the pre-pass always did. It is NOT `temporal_step`, which skips the frames
    outright and linearly interpolates them (measured 6x worse landmark error on turned heads at 2).

    Read at USE time from ``ROOP_PREPASS_DETECT_INTERVAL`` (the setting exports it; see settings.py).

`prepass_landmark_filter`
    Which smoother the pre-pass runs over each track: ``adaptive`` (the shipped causal velocity-ramped EMA),
    ``zero_phase`` (the same EMA forward + backward, Phase 7) or ``one_euro`` (`roop/one_euro_vec.py`).
    ``ROOP_PREPASS_FILTER`` chooses; unset, the legacy ``ROOP_STAB_ZERO_PHASE`` flag (read by
    `temporal_smoother.zero_phase_default`) still turns on ``zero_phase``; otherwise `FILTER_DEFAULT`.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

DETECT_INTERVAL_MIN = 1
DETECT_INTERVAL_MAX = 8
#: What a fresh install runs. See docs/CHANGELOG.md (2026-10-21, Phase 14) for the measurement behind it.
DETECT_INTERVAL_DEFAULT = 1

FILTER_KINDS = ('adaptive', 'zero_phase', 'one_euro')
FILTER_DEFAULT = 'adaptive'


def clamp_interval(value) -> int:
    try:
        k = int(float(value))
    except (TypeError, ValueError):
        return DETECT_INTERVAL_DEFAULT
    return max(DETECT_INTERVAL_MIN, min(DETECT_INTERVAL_MAX, k))


def detect_interval(env: Optional[Mapping[str, str]] = None) -> int:
    """K, clamped to 1..8. Unset or unparsable -> `DETECT_INTERVAL_DEFAULT`."""
    env = os.environ if env is None else env
    raw = env.get('ROOP_PREPASS_DETECT_INTERVAL')
    if raw is None or str(raw).strip() == '' or str(raw).strip().lower() == 'auto':
        return DETECT_INTERVAL_DEFAULT
    return clamp_interval(raw)


def max_misses_for(interval: int, base: int) -> int:
    """How many consecutive undetected frames a track survives before the tracker calls it lost.

    With a detection every K frames a healthy track is undetected for K - 1 frames in a row, and one missed
    key frame (an occlusion) doubles that. The default of 3 would declare every track of K >= 5 lost between
    two key frames and force a full-frame recovery on each, which is the work K exists to remove."""
    k = clamp_interval(interval)
    return max(int(base), 2 * k - 1) if k > 1 else int(base)


def filter_kind(env: Optional[Mapping[str, str]] = None) -> str:
    """The explicitly chosen smoother, or `FILTER_DEFAULT`. The pre-pass lets the legacy
    `temporal_smoother.zero_phase_default()` flag override an unset choice (see `_build_temporal_faces`)."""
    env = os.environ if env is None else env
    raw = str(env.get('ROOP_PREPASS_FILTER', '') or '').strip().lower().replace('-', '_')
    return raw if raw in FILTER_KINDS else FILTER_DEFAULT
