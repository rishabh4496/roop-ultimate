"""Identity crossover lock (Stage 11): hysteresis, trajectory, pins and the confidence strip.

The crossing is SIMULATED with known truth: two bodies on straight paths cross at frame 40, and for
a few frames around the crossing the crop is half the neighbour, so the memoryless matcher says
"another person fits better" about the person who never changed. The number that matters is how
many frames of the right person the veto strips, with and without the lock - and that a REAL
switch (the track genuinely jumping to the other body) is still followed.
"""
import os
import sys

import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import identity_lock as il  # noqa: E402

VETO_MARGIN = 0.10     # the matcher's own relative margin (d_other + margin < d_own)


def _box(cx, cy=100.0, w=60.0):
    return [cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2]


def _instant(d_own, d_other):
    """The memoryless rule the lock sits on top of, and how far past its bar it is."""
    veto = d_other is not None and d_other + VETO_MARGIN < d_own
    return veto, (max(0.0, (d_own - d_other) - VETO_MARGIN) if veto else 0.0)


def _run(frames, lock=None):
    """frames: [(frame, bbox, d_own, d_other)]. Returns the set of frames the face loses its source on."""
    lost = []
    for f, box, d_own, d_other in frames:
        veto, excess = _instant(d_own, d_other)
        if lock is None:
            if veto:
                lost.append(f)
            continue
        v = lock.update("A", f, box, veto, excess)
        if veto and not v.hold:
            lost.append(f)
    return lost


def _crossing(burst=(38, 39, 40), n=80, jump_at=None, seed=2, contam=(0.62, 0.30)):
    """Person A walks right across a still frame; during `burst` its crop is contaminated and a
    different person looks closer. With `jump_at` the track GENUINELY becomes the other person
    from that frame on (and its box jumps with it)."""
    rng = np.random.default_rng(seed)
    out = []
    for f in range(n):
        cx = 100.0 + 4.0 * f                              # smooth, continuous path
        d_own = float(0.30 + rng.normal(0, 0.02))
        d_other = float(0.75 + rng.normal(0, 0.03))
        if f in burst:
            d_own, d_other = contam                        # contamination: B suddenly fits better by 0.32
        box = _box(cx)
        if jump_at is not None and f >= jump_at:
            box = _box(cx + 140.0)                         # a different body entirely
            d_own, d_other = 0.70, 0.28                    # and it really is the other person
        out.append((f, box, d_own, d_other))
    return out


# ── hysteresis ───────────────────────────────────────────────────────────────

def test_memoryless_matcher_strips_the_burst_the_lock_absorbs():
    seq = _crossing()
    baseline = _run(seq)
    held = _run(seq, il.TrackLock())
    assert baseline == [38, 39, 40]                       # the defect: three frames of A lose the swap
    assert held == []                                     # the lock rides through it


def test_a_longer_burst_is_eventually_honoured():
    seq = _crossing(burst=tuple(range(38, 48)))           # ten frames: not a glitch
    cfg = il.LockConfig()
    lost = _run(seq, il.TrackLock(cfg))
    assert lost[0] == 38 + cfg.hold                       # frames 38..40 absorbed (`hold` of them), the veto lands on 41
    assert lost == list(range(38 + cfg.hold, 48))         # and stays until the burst ends


def test_a_challenge_below_the_raised_bar_is_still_not_defended_forever():
    # excess 0.12 clears the veto (0.10 margin) but not the lock's bar on a continuous track (0.18):
    # the lock absorbs it, but only up to max_defend frames - past that the veto must land.
    cfg = il.LockConfig()
    seq = _crossing(burst=tuple(range(30, 70)), contam=(0.60, 0.38))
    lost = _run(seq, il.TrackLock(cfg))
    assert lost and lost[0] == 30 + cfg.max_defend        # 12 frames defended, then the veto lands
    assert lost[-1] == 69


def test_a_genuine_switch_is_followed_not_forbidden():
    seq = _crossing(jump_at=40)
    lost = _run(seq, il.TrackLock())
    assert lost, "a real identity switch must still be vetoed"
    assert lost[0] <= 40 + il.LockConfig().hold           # delayed by at most the hold window
    assert lost[-1] == 79                                 # and it stays vetoed while it persists


def test_an_unsettled_track_gets_no_protection():
    lock = il.TrackLock()
    seq = _crossing(burst=(1, 2))                         # contamination on the track's FIRST frames
    lost = _run(seq, lock)
    assert lost == [1, 2]                                 # nothing is established yet: behave as before


def test_trajectory_continuity_raises_the_bar_for_a_borderline_challenge():
    cfg = il.LockConfig(margin=0.04, traj_bonus=0.10, hold=1, settle=2)
    # an excess of 0.08 beats margin 0.04 alone but not margin + trajectory bonus (0.14)
    def run(jumpy):
        lock = il.TrackLock(cfg)
        res = []
        for f in range(12):
            cx = 100.0 + 4.0 * f
            if jumpy and f >= 6:
                cx += 160.0 * (f % 2 or -1)               # box no longer follows its own path
            veto = f >= 6
            v = lock.update("A", f, _box(cx), veto, 0.08 if veto else 0.0)
            res.append(veto and not v.hold)
        return sum(res)
    assert run(jumpy=False) == 0                          # continuous body: the borderline challenge never counts
    assert run(jumpy=True) > 0                            # jumping box: the same numbers DO count


def test_out_of_order_or_gappy_frames_reset_to_memoryless():
    lock = il.TrackLock()
    for f in range(10):                                   # settle the track
        lock.update("A", f, _box(100 + 4 * f), False)
    assert lock.phase("A") == il.SETTLED
    assert lock.update("A", 5, _box(120), True, 0.5).hold is False        # backwards: forgotten
    assert lock.phase("A") == il.NEW
    for f in range(20, 30):
        lock.update("A", f, _box(100 + 4 * f), False)
    assert lock.update("A", 30 + il.LockConfig().max_gap + 1, _box(400), True, 0.5).hold is False   # long gap


def test_phase_recovers_after_the_challenge_ends():
    lock = il.TrackLock()
    seq = _crossing(burst=tuple(range(38, 48)))
    for f, box, d_own, d_other in seq:
        veto, ex = _instant(d_own, d_other)
        lock.update("A", f, box, veto, ex)
    assert lock.phase("A") == il.SETTLED                   # clean frames after the burst release it


def test_tracks_are_independent_and_banks_are_per_thread():
    import threading
    lock = il.TrackLock()
    for f in range(6):
        lock.update("A", f, _box(100), False)
    assert lock.phase("A") == il.SETTLED and lock.phase("B") == il.NEW
    bank = il.LockBank()
    seen = {}
    bank.lock().update("A", 0, _box(100), False)
    t = threading.Thread(target=lambda: seen.setdefault("other", bank.lock()))
    t.start()
    t.join()
    assert seen["other"] is not bank.lock()


# ── pins ─────────────────────────────────────────────────────────────────────

def test_sanitize_pins_keeps_only_well_formed_pins():
    raw = [
        {"frame": 10, "bbox": [0, 0, 50, 50], "source_index": 1, "person_id": "tp_1", "id": "x"},
        {"frame": 11, "point": [20, 20], "person_id": "tp_0"},
        {"frame": 12, "bbox": [50, 50, 0, 0], "source_index": 0},        # inverted box
        {"frame": "n/a", "bbox": [0, 0, 5, 5], "source_index": 0},       # no frame
        {"frame": 3, "bbox": [0, 0, 5, 5]},                              # locks nothing
        {"frame": 4, "point": [float("nan"), 1], "source_index": 0},
        "junk",
    ]
    out = il.sanitize_pins(raw)
    assert [p["frame"] for p in out] == [10, 11]
    assert out[0]["source_index"] == 1 and out[0]["centre"] == (25.0, 25.0)
    assert out[1]["source_index"] is None and out[1]["person_id"] == "tp_0"


def _scan():
    # two tracks; at frame index 20 (UI frame 21) track 7 is at x=100, track 9 at x=300
    return {k: [((100.0 + k, 100.0), 7), ((300.0 - k, 100.0), 9)] for k in range(0, 60)}


def test_pin_binds_the_track_through_its_box():
    pins = il.sanitize_pins([{"frame": 21, "bbox": [70, 70, 150, 150], "source_index": 1}])
    bound, unresolved = il.resolve_pin_tracks(pins, _scan())
    assert list(bound) == [7] and bound[7]["source_index"] == 1 and not unresolved


def test_pin_on_a_skipped_frame_searches_outward_and_a_miss_is_reported():
    scan = {k: v for k, v in _scan().items() if k != 20}
    pins = il.sanitize_pins([{"frame": 21, "bbox": [70, 70, 150, 150], "source_index": 1}])
    bound, unresolved = il.resolve_pin_tracks(pins, scan)
    assert list(bound) == [7]
    far = il.sanitize_pins([{"frame": 21, "bbox": [500, 500, 600, 600], "source_index": 1}])
    assert il.resolve_pin_tracks(far, _scan()) == ({}, far)


def test_two_pins_that_disagree_on_one_track_keep_the_first():
    pins = il.sanitize_pins([
        {"frame": 21, "bbox": [70, 70, 150, 150], "source_index": 0},
        {"frame": 25, "bbox": [70, 70, 160, 160], "source_index": 1},
    ])
    bound, unresolved = il.resolve_pin_tracks(pins, _scan())
    assert bound[7]["source_index"] == 0 and len(unresolved) == 1


def test_point_pin_has_a_distance_limit():
    near = il.sanitize_pins([{"frame": 21, "point": [105, 110], "source_index": 0}])
    far = il.sanitize_pins([{"frame": 21, "point": [200, 100], "source_index": 0}])
    assert list(il.resolve_pin_tracks(near, _scan())[0]) == [7]
    assert il.resolve_pin_tracks(far, _scan())[0] == {}


# ── confidence ───────────────────────────────────────────────────────────────

def test_confidence_scale_and_amber_rule():
    assert il.confidence(0.0, 0.75) == 1.0
    assert il.confidence(0.75, 0.75) == 0.0 and il.confidence(1.4, 0.75) == 0.0
    assert il.confidence(None, 0.75) is None and il.confidence(0.3, 0.0) is None
    assert il.is_amber(il.confidence(0.60, 0.75))                       # 0.20 of the way in: near the edge
    assert not il.is_amber(il.confidence(0.30, 0.75))
    assert il.is_amber(il.confidence(0.30, 0.75), margin=0.03)           # strong match but a close runner-up
    assert not il.is_amber(None, margin=0.0)


def test_confidence_lane_marks_exactly_the_dips():
    samples = [(0, 0.20, 0.80), (10, 0.25, 0.70), (20, 0.62, 0.90), (30, 0.30, 0.32), (40, None, None), (50, 0.28, 0.9)]
    lane = il.confidence_lane(samples, threshold=0.75)
    assert [p[0] for p in lane["points"]] == [1, 11, 21, 31, 51]         # UI numbering, the unmeasured one dropped
    assert lane["amber"] == [21, 31]                                     # a far match, and a close runner-up
    assert all(0.0 <= p[1] <= 1.0 for p in lane["points"])
    assert lane["threshold"] == 0.75


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
