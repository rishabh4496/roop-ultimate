"""The vectorized zero-phase 1-euro landmark filter (`roop/one_euro_vec.py`).

The measured claims are asserted against the stored detections in tests/data/synthetic_face_jitter.npz (a still
and a moving 266 px face with sensor noise through x264, the same fixture that scores the shipped smoother and the
Kalman option in test_kalman_kps_stabilizer.py), so nothing here is recalled:

    smoother                          static >2 Hz   >4 Hz    sway lag   head-turn peak   rms
    AdaptiveLandmarkSmoother (shipped)   -67.4 %     -73.8 %    1.00 f      2.30 px       1.94 px
    KalmanKpsStabilizer (option)         -77.0 %     -81.0 %    0.00 f      1.79 px       1.01 px
    1-euro 0.6 / 25 / 3 (this, default)  -90.9 %     -95.2 %    0.00 f      2.59 px       0.83 px     raw 1.41 / 0.90
    1-euro 0.8 / 50 / 3 (more open)      -82.1 %     -89.3 %    0.00 f      1.58 px       0.75 px

The head-turn peak is NOT better than the shipped filter's with the default parameters (2.59 vs 2.30 px at a 60 px
turn in 8 frames, a harder turn than footage usually has); everything else is, by a wide margin. The defaults were
chosen with real tracks as well (see one_euro_vec.py): a fixture-only choice (beta 50) is no better than the shipped
filter on a still face in real footage.

and the properties that make it a filter you can trust on a track: no lag, no dependence on where the face is or
how big it is, nothing averaged across a cut or a gap, the 68 3-D landmarks and the 106 moving with the same
alphas as the 5 keypoints, and a scan that equals the loop it replaces.
"""
import os
import sys

import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import one_euro_vec as ev                                    # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic_face_jitter.npz")
FPS = 24.0
BASE = np.array([[-70.0, -45.0], [70.0, -45.0], [0.0, 0.0], [-55.0, 80.0], [55.0, 80.0]])
SIGMA = 0.00195 * 266.0


def smooth(k, dense=None, bounds=None, **kw):
    t = len(k)
    dense = np.zeros((t, 0)) if dense is None else dense
    bounds = [(0, t)] if bounds is None else bounds
    kw.setdefault('fps', FPS)
    kw.setdefault('device', 'cpu')
    return ev.smooth_runs(np.asarray(k, np.float64), np.asarray(dense, np.float64), bounds, **kw)


# ── the scan ─────────────────────────────────────────────────────────────────────────────────────
def test_the_doubling_scan_equals_the_sequential_recurrence():
    rng = np.random.default_rng(0)
    for t in (1, 2, 3, 7, 64, 500):
        a = rng.uniform(0, 1, (t, 3))
        b = rng.normal(size=(t, 3))
        want = np.zeros_like(a)
        prev = np.zeros(3)
        for i in range(t):
            prev = a[i] * prev + b[i]
            want[i] = prev
        assert np.abs(ev._scan(a, b) - want).max() < 1e-12, t


def test_an_ema_with_alpha_one_at_a_start_restarts_there():
    x = np.arange(10, dtype=float)[:, None] * 5.0
    alpha = np.full((10, 1), 0.3)
    alpha[0] = 1.0
    alpha[5] = 1.0
    y = ev._ema(x, alpha)
    assert y[0, 0] == x[0, 0] and y[5, 0] == x[5, 0]
    assert y[4, 0] < x[4, 0]                 # trailing before the restart


# ── the measured claims, on the stored detections ─────────────────────────────────────────────────
def _highpass_rms(x, cutoff_hz, fps=24.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, cutoff_hz / (fps / 2), btype="high")
    return float(np.sqrt(np.mean(filtfilt(b, a, x.reshape(len(x), -1), axis=0)[40:-40] ** 2)))


def _lag_peak_rms(y, truth):
    err = np.linalg.norm(y - truth, axis=2).mean(1)
    cy, ct = y.mean(1)[:, 0], truth.mean(1)[:, 0]
    t = np.arange(30, 140)
    lag = min(np.arange(0, 12.01, 0.25),
              key=lambda dl: np.mean((cy[t] - np.interp(t - dl, np.arange(len(ct)), ct)) ** 2))
    return lag, err[148:176].max(), float(np.sqrt(np.mean(err[30:] ** 2)))


def _production(frames):
    from roop.temporal_smoother import AdaptiveLandmarkSmoother
    sm = AdaptiveLandmarkSmoother()
    return np.stack([sm.smooth(np.asarray(k, np.float32), None, track_id=0, frame_index=t)[0]
                     for t, k in enumerate(frames)]).astype(np.float64)


def _kalman(frames):
    from roop.one_euro import KalmanKpsStabilizer
    stab = KalmanKpsStabilizer()
    return np.stack([stab.apply(k, t) for t, k in enumerate(frames)]).astype(np.float64)


def test_a_still_face_loses_most_of_its_jitter_and_more_than_either_existing_smoother():
    raw = np.load(DATA)["static_kps"].astype(np.float64)
    out = smooth(raw)[0]
    kal, prod = _kalman(list(raw)), _production(list(raw))
    red = lambda y, c: 100 * (1 - _highpass_rms(y, c) / _highpass_rms(raw, c))      # noqa: E731
    assert red(out, 2.0) >= 88.0 and red(out, 4.0) >= 93.0                          # measured 90.9 / 95.2
    assert red(out, 2.0) > red(kal, 2.0) + 8.0 and red(out, 4.0) > red(kal, 4.0) + 8.0
    assert red(out, 4.0) > red(prod, 4.0) + 15.0


def test_known_motion_has_no_lag_and_a_small_error_against_the_exact_truth():
    d = np.load(DATA)
    raw = d["moving_kps"].astype(np.float64)
    truth = d["base_kps"].astype(np.float64)[None] + d["moving_shift"].astype(np.float64)[:, None, :]
    lag, peak, rms = _lag_peak_rms(smooth(raw)[0], truth)
    klag, kpeak, krms = _lag_peak_rms(_kalman(list(raw)), truth)
    plag, ppeak, prms = _lag_peak_rms(_production(list(raw)), truth)
    assert lag <= 0.25 and peak < 2.8 and rms < 0.9                  # measured 0.00 / 2.59 / 0.83 (raw 1.41 / 0.90)
    assert rms < krms                                                # Kalman: 1.01 (and a 1.79 px peak: not claimed)
    assert plag >= 0.75 and rms < prms / 2                           # shipped: 1.00 f / 2.30 / 1.94
    assert peak < 1.25 * ppeak                                       # the one metric it does not win: within 25%


def test_a_constant_velocity_ramp_comes_out_unshifted():
    """Zero phase: a causal filter trails a ramp by a constant; forward + backward cancels it."""
    t = np.arange(300)
    clean = np.stack([BASE + np.array([300.0 + 3.0 * i, 200.0 + 1.0 * i]) for i in t])
    out = smooth(clean)[0]
    assert np.abs(out[40:-40] - clean[40:-40]).max() < 0.05


def test_a_head_turn_is_followed_not_smeared():
    n = 120
    shift = 60.0 * np.clip((np.arange(n) - 60) / 8.0, 0, 1)
    clean = np.stack([BASE + np.array([500.0 + s, 300.0]) for s in shift])
    rng = np.random.default_rng(9)
    frames = np.stack([c + rng.normal(0, SIGMA, c.shape) for c in clean])
    out = smooth(frames)[0]
    err = np.linalg.norm(out - clean, axis=2).mean(1)
    assert err[60:80].max() < 8.0
    assert err[90:].mean() < 0.8


# ── invariances (exact, because the speed is extent-normalised and the filter is linear) ───────────
def _noisy(n=200, seed=3, scale=1.0, offset=(900.0, 500.0)):
    rng = np.random.default_rng(seed)
    return np.stack([BASE * scale + np.asarray(offset) + rng.normal(0, SIGMA * scale, BASE.shape) for _ in range(n)])


def test_where_the_face_is_in_the_frame_does_not_matter():
    a = smooth(_noisy(offset=(0.0, 0.0)))[0]
    b = smooth(_noisy(offset=(1500.0, 800.0)))[0]
    assert np.abs((b - np.array([1500.0, 800.0])) - a).max() < 1e-8


def test_how_big_the_face_is_does_not_matter():
    """The same RELATIVE noise at twice the size gets the same relative smoothing: output scales exactly."""
    a = smooth(_noisy(scale=1.0, offset=(0.0, 0.0)))[0]
    k2 = _noisy(scale=2.0, offset=(0.0, 0.0))
    b = smooth(k2)[0]
    assert np.abs(b - 2.0 * a).max() < 1e-8


def test_frame_rate_changes_the_cutoff_in_hz_not_the_answer_per_second():
    """The same motion sampled at 2x the rate with the cutoffs in Hz keeps the same smoothing of a slow sway."""
    t1 = np.arange(240) / 24.0
    t2 = np.arange(480) / 48.0
    sway = lambda tt: np.stack([BASE + np.array([500.0 + 25.0 * np.sin(2 * np.pi * 0.4 * x), 300.0]) for x in tt])  # noqa: E731
    a = smooth(sway(t1), fps=24.0)[0]
    b = smooth(sway(t2), fps=48.0)[0]
    # compare at the same instants (every other sample of b)
    assert np.abs(a[40:-40] - b[::2][40:-40]).max() < 0.2


# ── bounded: nothing crosses a cut or a gap ────────────────────────────────────────────────────────
def test_one_side_of_a_cut_cannot_bleed_into_the_other():
    k = _noisy(120, seed=5)
    cut = 60
    bounds = [(0, cut), (cut, 120)]
    base = smooth(k, bounds=bounds)[0]
    k2 = k.copy()
    k2[cut:] += np.array([400.0, -250.0])                       # the other shot: somewhere else entirely
    other = smooth(k2, bounds=bounds)[0]
    assert np.array_equal(base[:cut], other[:cut])              # bit for bit: the first shot never saw it
    assert np.abs((other[cut:] - np.array([400.0, -250.0])) - base[cut:]).max() < 1e-8


def test_frames_outside_every_run_are_returned_untouched_and_short_runs_too():
    k = _noisy(40, seed=6)
    out = smooth(k, bounds=[(5, 20), (22, 24), (30, 40)])[0]    # (22, 24) is 2 frames: left alone
    for lo, hi in [(0, 5), (20, 30)]:
        assert np.array_equal(out[lo:hi], k[lo:hi])
    assert not np.array_equal(out[5:20], k[5:20]) and not np.array_equal(out[30:40], k[30:40])


# ── every landmark set rides the same alphas ───────────────────────────────────────────────────────
def test_dense_landmarks_are_smoothed_with_exactly_the_keypoints_alphas():
    k = _noisy(150, seed=8)
    dense = np.concatenate([k.reshape(150, -1)] * 3, axis=1)     # the keypoints, repeated: must come out identical
    ks, ds, info = smooth(k, dense)
    assert np.abs(ds[:, :10] - ks.reshape(150, -1)).max() < 1e-9
    assert np.abs(ds[:, 10:20] - ks.reshape(150, -1)).max() < 1e-9


# ── the track-level contract (same as ZeroPhaseTrackSmoother) ─────────────────────────────────────
def _track(n=60, seed=1):
    rng = np.random.default_rng(seed)
    frames = list(range(100, 100 + n))
    kps = [(BASE + 700.0 + rng.normal(0, SIGMA, BASE.shape)).astype(np.float32) for _ in frames]
    lm = [(rng.normal(0, 30, (106, 2)) + 700.0 + kk.mean(0) - 700.0).astype(np.float32) for kk in kps]
    lm68 = [(rng.normal(0, 30, (68, 3)) + 700.0).astype(np.float32) for _ in frames]
    return frames, kps, lm, lm68


def test_smooth_track_returns_lists_aligned_with_frames_and_smooths_all_three_sets():
    frames, kps, lm, lm68 = _track()
    sm = ev.OneEuroTrackSmoother(fps=FPS, device='cpu')
    out, stats = sm.smooth_track(frames, kps, {'landmark_2d_106': lm, 'landmark_3d_68': lm68})
    assert len(out['kps']) == len(out['landmark_2d_106']) == len(out['landmark_3d_68']) == len(frames)
    assert stats['runs'] == 1 and stats['frames'] == len(frames) and stats['device'] == 'cpu'
    assert out['kps'][5].shape == (5, 2) and out['landmark_2d_106'][5].shape == (106, 2) \
        and out['landmark_3d_68'][5].shape == (68, 3)
    assert out['kps'][5].dtype == np.float32
    d2 = lambda seq: float(np.sqrt(np.mean(np.diff(np.stack(seq), n=2, axis=0) ** 2)))      # noqa: E731
    assert d2(out['kps'][10:]) < 0.5 * d2(kps[10:])
    assert d2(out['landmark_3d_68'][10:]) < 0.5 * d2(lm68[10:])      # the 68 were never smoothed before


def test_a_missing_frame_ends_a_run_and_a_none_passes_through():
    frames, kps, lm, lm68 = _track(40)
    frames = frames[:20] + [f + 5 for f in frames[20:]]            # frames 120..124 missing
    sm = ev.OneEuroTrackSmoother(fps=FPS, device='cpu')
    kps2 = list(kps)
    kps2[3] = None
    out, stats = sm.smooth_track(frames, kps2, {'landmark_2d_106': lm})
    assert stats['runs'] == 3                                       # [0,3), [4,20), [20,40)
    assert out['kps'][3] is None


def test_a_dense_set_with_a_hole_inside_a_run_is_left_alone_while_the_keypoints_are_smoothed():
    frames, kps, lm, lm68 = _track(30)
    lm = list(lm)
    lm[10] = None
    sm = ev.OneEuroTrackSmoother(fps=FPS, device='cpu')
    out, stats = sm.smooth_track(frames, kps, {'landmark_2d_106': lm, 'landmark_3d_68': lm68})
    assert all(np.array_equal(a, b) for a, b in zip(out['landmark_2d_106'][11:], lm[11:]))
    assert not np.array_equal(out['kps'][10], kps[10])
    assert not np.array_equal(out['landmark_3d_68'][10], lm68[10])


def test_tracks_shorter_than_three_frames_come_back_unchanged():
    frames, kps, lm, _ = _track(2)
    out, stats = ev.OneEuroTrackSmoother(fps=FPS, device='cpu').smooth_track(frames, kps, {'landmark_2d_106': lm})
    assert stats['runs'] == 0 and all(np.array_equal(a, b) for a, b in zip(out['kps'], kps))


def test_a_cut_frame_starts_a_new_run():
    frames, kps, lm, _ = _track(40)
    out, stats = ev.OneEuroTrackSmoother(fps=FPS, device='cpu').smooth_track(
        frames, kps, {'landmark_2d_106': lm}, cuts={frames[20]})
    assert stats['runs'] == 2


def test_the_summary_line_says_what_ran():
    frames, kps, lm, _ = _track(30)
    sm = ev.OneEuroTrackSmoother(fps=FPS, device='cpu')
    _, stats = sm.smooth_track(frames, kps, {})
    line = sm.summary_line(stats)
    assert '1-euro' in line and 'zero-phase' in line and 'cpu' in line and '24.0 fps' in line
    assert sm.summary_line({'runs': 0}) is None


def test_the_environment_sets_the_three_parameters_within_bounds(monkeypatch):
    monkeypatch.setenv('ROOP_ONE_EURO_MIN_CUTOFF', '2.5')
    monkeypatch.setenv('ROOP_ONE_EURO_BETA', '999999')
    monkeypatch.setenv('ROOP_ONE_EURO_D_CUTOFF', 'junk')
    sm = ev.OneEuroTrackSmoother.from_env(fps=30.0)
    assert sm.min_cutoff == 2.5 and sm.beta == 200.0 and sm.d_cutoff == ev.D_CUTOFF_HZ and sm.fps == 30.0


# ── the device ────────────────────────────────────────────────────────────────────────────────────
def test_the_device_choice():
    assert ev.resolve_device(10 ** 9, 'cpu') == 'cpu'
    assert ev.resolve_device(10, 'auto') == 'cpu'                  # tiny jobs never leave the CPU
    if ev._torch_cuda() is None:
        assert ev.resolve_device(10 ** 9, 'cuda') == 'cpu'         # asked for a GPU that is not there


def test_the_gpu_agrees_with_the_cpu():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    k = _noisy(400, seed=2)
    dense = np.random.default_rng(1).normal(500, 3, (400, 212))
    cpu = smooth(k, dense, bounds=[(0, 200), (210, 400)], device='cpu')
    gpu = smooth(k, dense, bounds=[(0, 200), (210, 400)], device='cuda')
    assert gpu[2]['device'] == 'cuda'
    assert np.abs(cpu[0] - gpu[0]).max() < 1e-2 and np.abs(cpu[1] - gpu[1]).max() < 1e-2     # float32 vs float64


# ── wired into the pre-pass ───────────────────────────────────────────────────────────────────────
def _prepass(filter_kind, monkeypatch):
    from insightface.app.common import Face
    from roop.procmgr_tracking import TrackingMixin

    monkeypatch.setenv('ROOP_PREPASS_FILTER', filter_kind)

    class Options:
        stabilize_face = True
        stabilize_method = 'one_euro'
        stabilize_min_cutoff = 0.05
        stabilize_beta = 0.02
        stabilize_landmarks = True

    class Mgr(TrackingMixin):
        def __init__(self):
            self.options = Options()
            self.input_face_datas = []
            self._track_pose_source_map = {}
            self._video_fps = 24.0

    rng = np.random.default_rng(21)
    emb = rng.normal(size=512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    lm_rel = rng.normal(0, 40, (106, 2))
    raw, obs = {}, {}
    for i in range(80):
        kps = (BASE + np.array([800.0, 500.0]) + rng.normal(0, SIGMA, BASE.shape)).astype(np.float32)
        face = Face(bbox=np.array([650.0, 350.0, 950.0, 650.0], np.float32), kps=kps, det_score=0.9, embedding=emb)
        face['landmark_2d_106'] = (lm_rel + kps.mean(0)).astype(np.float32)
        obs[i] = face
        raw[i] = kps.copy()
    mgr = Mgr()
    out = mgr._build_temporal_faces([{'id': 0, 'obs': obs, 'emb_mean': emb, 'emb_n': 80}], gap_max=10)
    return raw, out, mgr


def test_the_prepass_runs_the_one_euro_filter_when_selected(monkeypatch):
    raw, out, mgr = _prepass('one_euro', monkeypatch)
    k_raw = np.stack([raw[i] for i in range(80)])
    k_out = np.stack([out[i][0]['kps'] for i in range(80)])
    d2 = lambda x: float(np.sqrt(np.mean((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2)))      # noqa: E731
    assert d2(k_out[10:-10].reshape(-1, 10)) < 0.35 * d2(k_raw[10:-10].reshape(-1, 10))
    assert any('1-euro' in line for line in mgr._landmark_smooth_lines), mgr._landmark_smooth_lines


def test_the_default_prepass_filter_is_still_the_shipped_one(monkeypatch):
    _, out_default, mgr_default = _prepass('adaptive', monkeypatch)
    assert any('adaptive EMA' in line for line in mgr_default._landmark_smooth_lines)
    _, out_euro, _ = _prepass('one_euro', monkeypatch)
    a = np.stack([out_default[i][0]['kps'] for i in range(20, 60)])
    b = np.stack([out_euro[i][0]['kps'] for i in range(20, 60)])
    assert not np.allclose(a, b, atol=1e-4)
