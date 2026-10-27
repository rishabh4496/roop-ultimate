"""Identity crossover lock: user pins, cross-frame hysteresis and the confidence readout.

What already exists, and what this adds
---------------------------------------
Already shipped (names and tests, not assumptions):
  * velocity-predicted IoU + embedding association in the pre-pass tracker
    (`temporal_tracker.TemporalFaceTracker._candidate`, `_assign`), and the Stage 16 composite
    cost with a keypoint term (`association_cost.composite_cost`);
  * per-frame source vetoes in `ProcessMgr` (a source is refused when "another person fits
    better" or the face is "far from its assigned person").

What none of that does is REMEMBER. Every frame's veto is an independent decision on one noisy
embedding, so when two people cross and one crop is half the other for three frames, the veto
fires on those three frames and the swap blinks off the person who never stopped being themselves.
This module supplies the three missing pieces:

``TrackLock``       hysteresis. A track whose identity is established is not stripped of its source
                    by a challenge burst of up to `hold` frames, and a challenge only counts at all
                    if it beats the incumbent by `margin` MORE than the veto already demands - and
                    by `traj_bonus` more again while the track's box is following its own
                    velocity-predicted path (the body is demonstrably the same one). A challenge
                    that persists is honoured: this delays a genuine switch by `hold` frames, it
                    never forbids one.
``resolve_pin_tracks``  a user pin ("this person is track #N, source S") is ground truth. It names a
                    box on a frame; the pre-pass track passing through that box is bound to that
                    source before any automatic gate runs.
``confidence``      the number the timeline strip shows: how far inside its match threshold a face
                    sits, and when that is close enough to the edge to be worth a look.

State discipline. Hysteresis needs frames IN ORDER. Rendering is ordered inside a block but blocks
run on several workers, so state lives per worker (see ``LockBank``), and any non-monotonic or
gappy frame sequence RESETS a track's state to "new", which has no hold - i.e. the old, memoryless
behaviour. The failure mode of misordered input is therefore "no worse than before", never "a
stale decision applied to the wrong frame".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ── configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LockConfig:
    margin: float = 0.08        # extra cosine-distance advantage a challenger needs, beyond the veto's own
    hold: int = 3               # a challenge burst of up to this many frames is absorbed; the veto lands on the next
    settle: int = 3             # unchallenged frames before an identity counts as established
    release: int = 2            # unchallenged frames to leave the CHALLENGED phase
    traj_iou: float = 0.45      # IoU with the predicted box above which the body is "the same one"
    traj_bonus: float = 0.10    # extra margin an incumbent on a continuous trajectory is owed
    max_gap: int = 6            # a jump of more than this many frames (or backwards) resets the track
    max_defend: int = 12        # ceiling: no run of vetoes is defended longer than this, whatever the bar says


NEW, SETTLED, CHALLENGED = "new", "settled", "challenged"


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / union if union > 1e-9 else 0.0


# ── hysteresis ───────────────────────────────────────────────────────────────


@dataclass
class _State:
    phase: str = NEW
    last_frame: int = -10 ** 9
    last_bbox: Optional[np.ndarray] = None
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    challenge_run: int = 0
    clean_run: int = 0
    veto_run: int = 0           # consecutive frames the memoryless rule wanted to veto, counted or not


@dataclass(frozen=True)
class Verdict:
    hold: bool                  # True: do NOT veto this frame, the identity is being defended
    phase: str
    traj_iou: Optional[float]
    reason: str


class TrackLock:
    """Per-track hysteresis over the veto the matcher already computes."""

    def __init__(self, cfg: LockConfig = LockConfig()):
        self.cfg = cfg
        self._tracks: Dict[Any, _State] = {}
        self.suppressed = 0         # vetoes this lock turned into holds
        self.honoured = 0           # sustained challenges it let through

    def reset(self) -> None:
        self._tracks.clear()

    def phase(self, tid) -> str:
        st = self._tracks.get(tid)
        return st.phase if st else NEW

    def _predict(self, st: _State, frame: int) -> Optional[np.ndarray]:
        if st.last_bbox is None:
            return None
        dt = frame - st.last_frame
        return st.last_bbox + st.vel * float(dt)

    def update(self, tid, frame: int, bbox: Sequence[float], instant_veto: bool,
               excess: float = 0.0) -> Verdict:
        """Feed one observation of track `tid` and get the verdict for THIS frame.

        `instant_veto` is what the memoryless matcher decided. `excess` is how far past its own
        threshold that veto was (relative test: ``(d_own - d_other) - veto_margin``; absolute
        test: ``d_own - gate``), never negative. Returns ``hold=True`` only when the identity is
        established and the challenge has not yet persisted.
        """
        cfg = self.cfg
        frame = int(frame)
        box = np.asarray(bbox[:4], np.float32)
        st = self._tracks.get(tid)
        if st is not None and (frame <= st.last_frame or frame - st.last_frame > cfg.max_gap):
            st = None                                   # out of order or a long gap: forget everything
        if st is None:
            st = self._tracks[tid] = _State()

        pred = self._predict(st, frame)
        t_iou = iou(pred, box) if pred is not None else None
        continuous = t_iou is not None and t_iou >= cfg.traj_iou

        counts = bool(instant_veto) and float(excess) >= cfg.margin + (cfg.traj_bonus if continuous else 0.0)
        st.veto_run = st.veto_run + 1 if instant_veto else 0
        if counts:
            st.challenge_run += 1
            st.clean_run = 0
        elif instant_veto:
            # A veto that did not beat the raised bar: neither a clean frame nor a counted challenge.
            st.challenge_run = 0
            st.clean_run = 0
        else:
            st.challenge_run = 0
            st.clean_run += 1

        if st.phase == NEW and st.clean_run >= cfg.settle:
            st.phase = SETTLED
        elif st.phase == SETTLED and (st.challenge_run > cfg.hold or st.veto_run > cfg.max_defend):
            # Either a counted challenge persisted past the hold window, or the memoryless rule has
            # wanted to veto for so long that no raised bar may keep overriding it: a lock that can
            # defend an identity forever has turned "delay" into "forbid".
            st.phase = CHALLENGED
        elif st.phase == CHALLENGED and st.clean_run >= cfg.release:
            st.phase = SETTLED
            st.challenge_run = 0

        hold = False
        reason = "no veto" if not instant_veto else "veto stands"
        if instant_veto:
            if st.phase == SETTLED:
                hold = True
                reason = ("below the raised bar" if not counts
                          else f"challenge {st.challenge_run}/{cfg.hold}")
                self.suppressed += 1
            elif st.phase == CHALLENGED:
                reason = "challenge sustained"
                self.honoured += 1
            else:
                reason = "identity not yet established"

        # advance the trajectory model with THIS frame's box
        if st.last_bbox is not None and frame > st.last_frame:
            raw = (box - st.last_bbox) / float(frame - st.last_frame)
            st.vel = (0.55 * st.vel + 0.45 * raw).astype(np.float32)
        st.last_bbox, st.last_frame = box, frame
        return Verdict(hold=hold, phase=st.phase, traj_iou=t_iou, reason=reason)


class LockBank:
    """One `TrackLock` per worker thread (hysteresis is order-dependent; see the module docstring)."""

    def __init__(self, cfg: LockConfig = LockConfig()):
        import threading
        self.cfg = cfg
        self._tls = threading.local()

    def lock(self) -> TrackLock:
        lk = getattr(self._tls, "lock", None)
        if lk is None:
            lk = self._tls.lock = TrackLock(self.cfg)
        return lk


# ── pins ─────────────────────────────────────────────────────────────────────


def sanitize_pins(raw: Any, limit: int = 64) -> List[Dict[str, Any]]:
    """Validated user pins. A pin names a FRAME (UI numbering, 1-based absolute), a place on it
    (`bbox` [x0,y0,x1,y1] or `point` [x,y], frame pixels) and the identity to lock: a `source_index`
    (what the matcher binds) and/or a `person_id` (what the UI shows). Malformed pins are dropped."""
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        try:
            frame = int(item.get("frame"))
        except (TypeError, ValueError):
            continue
        box = item.get("bbox")
        pt = item.get("point")
        centre = None
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                b = [float(v) for v in box[:4]]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in b) or b[2] <= b[0] or b[3] <= b[1]:
                continue
            centre = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            box = b
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                centre = (float(pt[0]), float(pt[1]))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in centre):
                continue
            box = None
        else:
            continue
        si = item.get("source_index")
        src = int(si) if isinstance(si, (int, float)) and not isinstance(si, bool) and si >= 0 else None
        pid = item.get("person_id")
        if src is None and not pid:
            continue                                    # a pin that locks nothing
        out.append({"id": str(item.get("id") or f"pin_{len(out)}"), "frame": frame, "bbox": box,
                    "centre": centre, "source_index": src,
                    "person_id": str(pid) if pid else None,
                    "track_id": item.get("track_id")})
    return out


def resolve_pin_tracks(pins: Sequence[Dict[str, Any]], per_frame: Dict[int, Sequence[Tuple[Any, Any]]],
                       to_key=lambda ui_frame: ui_frame - 1, tol_frames: int = 3):
    """(bound, unresolved): which pre-pass track each pin names.

    `per_frame` is the scan's ``{frame: [(centroid, track_id)]}``; `to_key` maps the pin's UI frame
    onto its keys. The entry whose centroid is nearest the pin's place wins, searching the pinned
    frame first and then outward by up to `tol_frames` (a pin on a frame the scan skipped still
    finds its track). With a `bbox` the centroid must fall inside it grown by 15%; with a bare
    `point` it must be within 60 px. Two pins naming one track with different sources is a
    conflict: the first wins and the loser is reported unresolved.
    """
    bound: Dict[Any, Dict[str, Any]] = {}
    unresolved: List[Dict[str, Any]] = []
    for pin in pins:
        key = int(to_key(pin["frame"]))
        cx, cy = pin["centre"]
        best = None
        for off in range(0, tol_frames + 1):
            for k in ((key,) if off == 0 else (key - off, key + off)):
                for cent, tid in per_frame.get(k, ()):
                    px, py = float(cent[0]), float(cent[1])
                    if pin["bbox"] is not None:
                        x0, y0, x1, y1 = pin["bbox"]
                        gx, gy = (x1 - x0) * 0.15, (y1 - y0) * 0.15
                        if not (x0 - gx <= px <= x1 + gx and y0 - gy <= py <= y1 + gy):
                            continue
                    dist = math.hypot(px - cx, py - cy)
                    if pin["bbox"] is None and dist > 60.0:
                        continue
                    if best is None or dist < best[0]:
                        best = (dist, tid)
            if best is not None:
                break                                   # nearest frame that has a candidate wins
        if best is None:
            unresolved.append(pin)
            continue
        tid = best[1]
        prior = bound.get(tid)
        if prior is not None and prior["source_index"] != pin["source_index"]:
            unresolved.append(pin)
            continue
        bound[tid] = pin
    return bound, unresolved


# ── confidence ───────────────────────────────────────────────────────────────

NEAR_BAND = 0.25            # confidence at or below this is "near the threshold"
MARGIN_BAND = 0.06          # a runner-up this close behind the match is ambiguous


def confidence(distance: Optional[float], threshold: float) -> Optional[float]:
    """1 = identical, 0 = exactly at the match threshold (or beyond). None when unmeasurable."""
    if distance is None or threshold is None or threshold <= 0 or not math.isfinite(distance):
        return None
    return float(min(1.0, max(0.0, 1.0 - distance / threshold)))


def is_amber(conf: Optional[float], margin: Optional[float] = None) -> bool:
    """Worth a look: close to the threshold, or barely ahead of the next-best person."""
    if conf is None:
        return False
    if conf <= NEAR_BAND:
        return True
    return margin is not None and margin <= MARGIN_BAND


def confidence_lane(samples: Iterable[Tuple[int, Optional[float], Optional[float]]], threshold: float,
                    frame_base: int = 1) -> Dict[str, Any]:
    """One person's strip. `samples` is ``[(frame, own_distance, runner_up_distance|None)]`` in
    0-based sampled frames; the result is in UI numbering (`frame_base` shifts 0-based to 1-based).

    ``points``  every measured sample as ``[frame, confidence, amber(0/1)]`` for the strip itself;
    ``amber``   the sample frames that are amber, for jump-to-next. Unmeasurable samples (no
                embedding) are dropped, never drawn as a number they do not have.
    """
    points: List[List[float]] = []
    amber: List[int] = []
    for frame, own, other in sorted(samples, key=lambda s: s[0]):
        c = confidence(own, threshold)
        if c is None:
            continue
        margin = None if other is None else float(other) - float(own)
        flag = is_amber(c, margin)
        f = int(frame) + int(frame_base)
        points.append([f, round(c, 4), 1 if flag else 0])
        if flag:
            amber.append(f)
    return {"points": points, "amber": amber, "threshold": float(threshold)}
