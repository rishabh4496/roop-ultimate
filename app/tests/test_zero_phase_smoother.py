"""ZeroPhaseTrackSmoother (Phase 7): the pre-pass's landmark smoother, run forward AND backward.

The claims, each pinned:
  * a causal filter trails fast motion and a zero-phase one does not (against EXACT truth: the stored real
    detections of a face with a known motion path, tests/data/synthetic_face_jitter.npz)
  * it removes at least as much still-face jitter as the shipped filter
  * nothing is averaged across a shot cut, a missing frame or a non-finite frame
  * kps, the 106 and the 68 landmarks share ONE beta per frame
  * short runs and missing arrays pass through untouched
CPU only.
"""
import os
import sys

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.temporal_smoother import (AdaptiveLandmarkSmoother, ZeroPhaseTrackSmoother,  # noqa: E402
                                    zero_phase_default)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic_face_jitter.npz")
BASE = np.array([[-70.0, -45.0], [70.0, -45.0], [0.0, 0.0], [-55.0, 80.0], [55.0, 80.0]])


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def causal(frames):
    sm = AdaptiveLandmarkSmoother()
    return np.stack([sm.smooth(np.asarray(k, np.float32), None, track_id=0, frame_index=t)[0]
                     for t, k in enumerate(frames)]).astype(np.float64)


def zero_phase(frames, cuts=(), first=0, **kw):
    zp = ZeroPhaseTrackSmoother(**kw)
    out, _ = zp.smooth_track(list(range(first, first + len(frames))), list(frames), {}, cuts=cuts)
    return np.stack(out['kps']).astype(np.float64)


def d2(x):
    return float(np.sqrt(np.mean((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2)))


def lag_frames(y, truth):
    cy, ct = y.mean(1)[:, 0], truth.mean(1)[:, 0]
    t = np.arange(30, 140)
    return min(np.arange(-4, 8.01, 0.25),
               key=lambda dl: np.mean((cy[t] - np.interp(t - dl, np.arange(len(ct)), ct)) ** 2))


# ── against exact truth, on real detector output ─────────────────────────────────────────────────────────────
def test_known_motion_has_no_lag_where_the_shipped_filter_trails():
    d = np.load(DATA)
    frames = list(d["moving_kps"].astype(np.float64))
    truth = d["base_kps"].astype(np.float64)[None] + d["moving_shift"].astype(np.float64)[:, None, :]
    shipped, zp = causal(frames), zero_phase(frames)
    assert lag_frames(shipped, truth) >= 0.75                    # the shipped filter's measured lag (1.0 f)
    assert abs(lag_frames(zp, truth)) <= 0.5
    err = lambda y: float(np.sqrt(np.mean(np.linalg.norm(y - truth, axis=2).mean(1)[30:] ** 2)))      # noqa: E731
    assert err(zp) < err(shipped)


def test_a_still_real_face_is_at_least_as_quiet_as_with_the_shipped_filter():
    d = np.load(DATA)
    frames = list(d["static_kps"].astype(np.float64))
    raw = np.stack(frames)
    shipped, zp = causal(frames), zero_phase(frames)
    assert d2(zp[20:-20]) < d2(shipped[20:-20]) < d2(raw[20:-20])      # measured 0.231 < 0.249 < 1.024
    assert d2(zp[20:-20]) < 0.3 * d2(raw[20:-20])                      # 77% of the detector's jitter gone


def test_a_lower_beta_buys_jitter_with_error_on_real_slow_motion_which_is_why_the_default_stays_035():
    """Measured against exact truth on the stored detections (rms error px / still jitter):
    0.35 -> 1.16 / 0.231, 0.25 -> 2.06 / 0.153, 0.20 -> 3.02 / 0.118, 0.12 -> 6.43 / 0.067; the shipped
    causal filter is 1.94 / 0.249. Only the shipped beta beats it on BOTH. A face that is still and a face
    that sways slowly are the same thing to a speed-keyed filter, so 'zero jitter when still' costs accuracy."""
    d = np.load(DATA)
    mv, st = list(d["moving_kps"].astype(np.float64)), list(d["static_kps"].astype(np.float64))
    truth = d["base_kps"].astype(np.float64)[None] + d["moving_shift"].astype(np.float64)[:, None, :]
    err = lambda y: float(np.sqrt(np.mean(np.linalg.norm(y - truth, axis=2).mean(1)[30:] ** 2)))      # noqa: E731
    shipped_err = err(causal(mv))
    assert err(zero_phase(mv, beta_slow=0.35)) < shipped_err
    assert err(zero_phase(mv, beta_slow=0.12)) > 2 * shipped_err
    assert d2(zero_phase(st, beta_slow=0.12)[20:-20]) < d2(zero_phase(st, beta_slow=0.35)[20:-20])
    assert ZeroPhaseTrackSmoother().beta_slow == AdaptiveLandmarkSmoother.BETA_SLOW


def test_a_steady_pan_is_preserved_exactly_in_the_interior():
    """A constant-velocity pan: forward lag and backward lead cancel. Fast enough to hold beta at beta_fast."""
    clean = np.stack([BASE + np.array([300.0 + 6.0 * t, 200.0]) for t in range(120)])
    out = zero_phase(list(clean))
    assert np.abs(out[10:-10] - clean[10:-10]).max() < 1e-3


def test_a_slow_ramp_is_preserved_in_the_interior_too():
    clean = np.stack([BASE + np.array([300.0 + 0.2 * t, 200.0]) for t in range(300)])
    out = zero_phase(list(clean))
    assert np.abs(out[60:-60] - clean[60:-60]).max() < 1e-3     # float32 output


# ── bounding ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_nothing_is_averaged_across_a_shot_cut():
    a = BASE + np.array([200.0, 200.0])
    b = BASE + np.array([900.0, 450.0])                          # a different shot: the face is elsewhere
    frames = [a + 0.0] * 30 + [b + 0.0] * 30
    plain = zero_phase(frames)
    cut = zero_phase(frames, cuts=(30,))
    assert np.abs(cut[30:] - b).max() < 1e-9 and np.abs(cut[:30] - a).max() < 1e-9
    assert np.abs(plain[28:32] - np.concatenate([np.repeat(a[None], 2, 0), np.repeat(b[None], 2, 0)])).max() > 1.0, \
        'without the cut the step must bleed (control)'


def test_a_missing_frame_ends_the_run():
    a = BASE + 100.0
    frames = [a] * 20 + [BASE + 800.0] * 20
    zp = ZeroPhaseTrackSmoother()
    frame_numbers = list(range(20)) + list(range(25, 45))        # frames 20-24 are not in the track
    out, stats = zp.smooth_track(frame_numbers, frames, {})
    assert stats['runs'] == 2
    assert np.abs(np.stack(out['kps'][:20]) - a).max() < 1e-6
    assert np.abs(np.stack(out['kps'][20:]) - (BASE + 800.0)).max() < 1e-6


def test_a_non_finite_or_missing_frame_ends_the_run_and_is_left_alone():
    frames = [BASE + 50.0] * 12 + [None] + [BASE + 500.0] * 12
    bad = np.array(BASE + 50.0)
    bad[0, 0] = np.nan
    frames[5] = bad
    zp = ZeroPhaseTrackSmoother()
    out, stats = zp.smooth_track(list(range(len(frames))), frames, {})
    assert out['kps'][12] is None
    assert np.isnan(out['kps'][5][0, 0])
    assert np.abs(np.stack(out['kps'][13:]) - (BASE + 500.0)).max() < 1e-6


def test_runs_shorter_than_three_frames_are_returned_unchanged():
    frames = [BASE + 1.0, BASE + 9.0]
    zp = ZeroPhaseTrackSmoother()
    out, stats = zp.smooth_track([0, 1], frames, {})
    assert stats['runs'] == 0
    assert out['kps'][0] is frames[0] and out['kps'][1] is frames[1]


# ── coupling, shapes, dtypes ─────────────────────────────────────────────────────────────────────────────────
def test_all_landmark_sets_ride_one_beta():
    """Dense landmarks that are a rigid copy of the keypoints' motion must stay a rigid copy after smoothing."""
    rng = np.random.default_rng(0)
    n = 80
    shift = np.stack([np.cumsum(rng.normal(0, 0.3, n)), np.cumsum(rng.normal(0, 0.3, n))], axis=1)
    kps = [BASE + s for s in shift]
    d106 = [np.tile(BASE.mean(0), (106, 1)) + np.linspace(-1, 1, 106)[:, None] * 40 + s for s in shift]
    d68 = [np.concatenate([np.tile(BASE.mean(0), (68, 1)) + np.linspace(-1, 1, 68)[:, None] * 40 + s,
                           np.zeros((68, 1))], axis=1) for s in shift]
    zp = ZeroPhaseTrackSmoother()
    out, _ = zp.smooth_track(list(range(n)), kps, {'lm106': d106, 'lm68': d68})
    k, a, b = (np.stack(out[x]).astype(np.float64) for x in ('kps', 'lm106', 'lm68'))
    # the centroid motion of all three sets is identical (one beta)
    ck, ca, cb = k.mean(1), a.mean(1), b[..., :2].mean(1)
    assert np.abs(ck - ca).max() < 1e-3 and np.abs(ck - cb).max() < 1e-3
    assert out['lm68'][0].shape == (68, 3) and out['lm68'][0].dtype == np.float32
    assert np.abs(b[..., 2]).max() < 1e-6                         # z stays what it was (zeros here)


def test_outputs_are_float32_and_inputs_are_not_mutated():
    frames = [(BASE + t * 0.5).astype(np.float32) for t in range(20)]
    snapshot = [f.copy() for f in frames]
    zp = ZeroPhaseTrackSmoother()
    out, _ = zp.smooth_track(list(range(20)), frames, {})
    assert all(o.dtype == np.float32 for o in out['kps'])
    assert all(np.array_equal(f, s) for f, s in zip(frames, snapshot))


def test_an_array_missing_on_some_frames_is_left_unsmoothed_for_that_run_only():
    n = 30
    kps = [BASE + 0.0 for _ in range(n)]
    d = [np.zeros((106, 2)) + float(i % 2) for i in range(n)]
    d[10] = None
    zp = ZeroPhaseTrackSmoother()
    out, _ = zp.smooth_track(list(range(n)), kps, {'lm106': d})
    assert out['lm106'][10] is None
    assert out['lm106'][0] is d[0]                                # the whole run was skipped for this array, kps was not
    assert out['kps'][5].dtype == np.float32


def test_velocity_is_normalised_by_face_size():
    """The same motion in face-widths gives the same beta at any resolution."""
    zp = ZeroPhaseTrackSmoother()
    small = np.stack([BASE * 0.25 + np.array([10.0 + 1.5 * t, 5.0]) for t in range(10)])
    big = np.stack([BASE * 1.0 + np.array([40.0 + 6.0 * t, 20.0]) for t in range(10)])
    assert np.allclose(zp.velocities(small), zp.velocities(big), rtol=1e-6)


def test_environment_and_default(monkeypatch=None):
    os.environ.pop('ROOP_STAB_ZERO_PHASE', None)
    from roop import temporal_smoother as ts
    assert zero_phase_default() is bool(ts.ZERO_PHASE_DEFAULT)
    os.environ['ROOP_STAB_ZERO_PHASE'] = '1'
    try:
        assert zero_phase_default() is True
        os.environ['ROOP_STAB_ZERO_PHASE'] = '0'
        assert zero_phase_default() is False
        os.environ['ROOP_STAB_ZERO_PHASE_BETA'] = '0.2'
        assert abs(ZeroPhaseTrackSmoother.from_env().beta_slow - 0.2) < 1e-9
    finally:
        os.environ.pop('ROOP_STAB_ZERO_PHASE', None)
        os.environ.pop('ROOP_STAB_ZERO_PHASE_BETA', None)
