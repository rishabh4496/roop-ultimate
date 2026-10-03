"""KalmanKpsStabilizer (`stabilize_method: kalman`): a drop-in, opt-in keypoint smoother.

Measured 2026-10-03 on a static 266 px face rendered with sensor noise through x264
(detector sigma 0.52 px) and a known motion path (0.4 Hz sway plus a 60 px head turn in 8
frames): the shipped default (AdaptiveLandmarkSmoother) removes 67% of the >2 Hz jitter
(74% above 4 Hz) with a frame of lag on the slow sway; this filter removes 77% / 81% with
no measurable lag. Those detections are stored in tests/data/synthetic_face_jitter.npz so
the numbers are asserted, not recalled. On pure WHITE noise the same filter removes only
~40% of the >4 Hz band: the 81% belongs to the real detector's noise spectrum (concentrated
at 4-12 Hz). On real conversational footage it measured no better than the default
(6-12% of the >4 Hz band against 9-12%), so it is an option and not the default.
"""
import os
import sys

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.one_euro import KalmanKpsStabilizer  # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


SIZE = 266.0
BASE = np.array([[-70.0, -45.0], [70.0, -45.0], [0.0, 0.0], [-55.0, 80.0], [55.0, 80.0]])  # ~266 px extent
SIGMA = 0.00195 * SIZE


def _run(stab, frames):
    return np.stack([stab.apply(k, t) for t, k in enumerate(frames)]).astype(np.float64)


def _d2(x):
    return float(np.sqrt(np.mean((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2)))


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic_face_jitter.npz")


def _highpass_rms(x, cutoff_hz, fps=24.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, cutoff_hz / (fps / 2), btype="high")
    return float(np.sqrt(np.mean(filtfilt(b, a, x.reshape(len(x), -1), axis=0)[40:-40] ** 2)))


def _production(frames):
    from roop.temporal_smoother import AdaptiveLandmarkSmoother
    sm = AdaptiveLandmarkSmoother()
    return np.stack([sm.smooth(np.asarray(k, np.float32), None, track_id=0, frame_index=t)[0]
                     for t, k in enumerate(frames)]).astype(np.float64)


def _noisy_static(n=300, scale=1.0, offset=(900.0, 500.0), seed=3):
    rng = np.random.default_rng(seed)
    clean = BASE * scale + np.asarray(offset)
    return clean, [clean + rng.normal(0, SIGMA * scale, clean.shape) for _ in range(n)]


def test_white_noise_is_smoothed_but_this_is_not_the_measured_claim():
    """White noise: ~40% of the >4 Hz band and ~24% of the std. Pinned loosely, only so a
    regression to 'no smoothing' or to amplification is caught; the real claim is below."""
    clean, frames = _noisy_static(n=400)
    out = _run(KalmanKpsStabilizer(), frames)
    raw = np.stack(frames)
    assert _highpass_rms(out, 4.0) < 0.75 * _highpass_rms(raw, 4.0)
    assert np.std(out[40:] - clean, axis=0).mean() < 0.9 * np.std(raw[40:] - clean, axis=0).mean()


def test_a_still_face_through_the_real_detector_loses_80_percent_of_the_4hz_band():
    """The measured claim, on the stored detections: static face + sensor noise + x264."""
    d = np.load(DATA)
    frames = list(d["static_kps"].astype(np.float64))
    raw = np.stack(frames)
    kal = _run(KalmanKpsStabilizer(), frames)
    prod = _production(frames)
    red4 = lambda y: 100 * (1 - _highpass_rms(y, 4.0) / _highpass_rms(raw, 4.0))      # noqa: E731
    red2 = lambda y: 100 * (1 - _highpass_rms(y, 2.0) / _highpass_rms(raw, 2.0))      # noqa: E731
    assert red4(kal) >= 78.0 and red2(kal) >= 74.0                  # measured 81.0 / 77.0
    assert red4(kal) > red4(prod) + 4.0                             # shipped default: 73.8 / 67.4


def test_known_motion_through_the_real_detector_has_no_lag_and_a_small_error():
    """Moving clip (0.4 Hz sway + a 60 px turn in 8 frames) against the exact truth."""
    d = np.load(DATA)
    frames = list(d["moving_kps"].astype(np.float64))
    truth = d["base_kps"].astype(np.float64)[None] + d["moving_shift"].astype(np.float64)[:, None, :]
    kal = _run(KalmanKpsStabilizer(), frames)
    prod = _production(frames)

    def lag_and_errors(y):
        err = np.linalg.norm(y - truth, axis=2).mean(1)
        cy, ct = y.mean(1)[:, 0], truth.mean(1)[:, 0]
        t = np.arange(30, 140)
        best = min(np.arange(0, 12.01, 0.25),
                   key=lambda dl: np.mean((cy[t] - np.interp(t - dl, np.arange(len(ct)), ct)) ** 2))
        return best, err[148:176].max(), float(np.sqrt(np.mean(err[30:] ** 2)))
    lag, peak, rms = lag_and_errors(kal)
    plag, ppeak, prms = lag_and_errors(prod)
    assert lag <= 0.25 and peak < 2.2 and rms < 1.15                # measured 0.0 / 1.79 / 1.01
    assert plag >= 0.75 and rms < prms                              # shipped: 1.0 f / 2.30 / 1.94


def test_steady_motion_has_no_lag():
    """Constant velocity (3 px/frame): a constant-velocity filter converges to zero error."""
    clean = np.stack([BASE + np.array([300.0 + 3.0 * t, 200.0 + 1.0 * t]) for t in range(200)])
    rng = np.random.default_rng(5)
    frames = [c + rng.normal(0, SIGMA, c.shape) for c in clean]
    out = _run(KalmanKpsStabilizer(), frames)
    assert np.abs(out[60:] - clean[60:]).mean() < 0.35           # px; a lagging EMA sits ~3 px behind


def test_a_head_turn_is_followed_not_smeared():
    """60 px in 8 frames: the gated process noise opens the filter within a frame or two."""
    n = 120
    shift = 60.0 * np.clip((np.arange(n) - 60) / 8.0, 0, 1)
    clean = np.stack([BASE + np.array([500.0 + s, 300.0]) for s in shift])
    rng = np.random.default_rng(9)
    frames = [c + rng.normal(0, SIGMA, c.shape) for c in clean]
    out = _run(KalmanKpsStabilizer(), frames)
    err = np.linalg.norm(out - clean, axis=2).mean(1)
    assert err[60:80].max() < 8.0                                # one-sixth of the 60 px excursion
    assert err[90:].mean() < 0.6                                 # settled again


def test_the_result_does_not_depend_on_where_the_face_is_in_the_frame():
    """Translation invariance. Dividing ABSOLUTE coordinates by a fluctuating face size
    injects the jitter being removed (position 900 px / size 266 px, size noisy by 0.3%);
    the state stays in pixels and only the noise parameters scale with the face."""
    clean0, frames0 = _noisy_static(offset=(0.0, 0.0), seed=11)
    clean1, frames1 = _noisy_static(offset=(1500.0, 800.0), seed=11)
    a = _run(KalmanKpsStabilizer(), frames0)
    b = _run(KalmanKpsStabilizer(), frames1)
    assert np.abs((b - np.array([1500.0, 800.0])) - a).max() < 1e-3


def test_it_is_resolution_invariant():
    """The same RELATIVE noise at twice the size gets the same relative smoothing."""
    _, f1 = _noisy_static(scale=1.0, seed=2)
    _, f2 = _noisy_static(scale=2.0, seed=2)
    r1 = _d2(_run(KalmanKpsStabilizer(), f1)[30:]) / _d2(np.stack(f1)[30:])
    r2 = _d2(_run(KalmanKpsStabilizer(), f2)[30:]) / _d2(np.stack(f2)[30:])
    assert abs(r1 - r2) < 0.05


def test_two_faces_keep_separate_tracks():
    stab = KalmanKpsStabilizer()
    left, right = BASE + np.array([300.0, 300.0]), BASE + np.array([1200.0, 300.0])
    rng = np.random.default_rng(1)
    for t in range(60):
        a = stab.apply(left + rng.normal(0, SIGMA, left.shape), t)
        b = stab.apply(right + rng.normal(0, SIGMA, right.shape), t)
    assert len(stab.tracks) == 2
    assert np.abs(a - left).max() < 3.0 and np.abs(b - right).max() < 3.0


def test_reset_forgets_the_track_and_bad_input_passes_through():
    stab = KalmanKpsStabilizer()
    stab.apply(BASE + 500.0, 0)
    assert stab.tracks
    stab.reset()
    assert stab.tracks == []
    odd = np.zeros((3, 2))
    assert stab.apply(odd, 1).shape == (3, 2)


def test_a_dropped_frame_does_not_break_the_track():
    stab = KalmanKpsStabilizer()
    rng = np.random.default_rng(4)
    for t in list(range(0, 40)) + list(range(43, 80)):          # frames 40-42 missing
        out = stab.apply(BASE + 600.0 + rng.normal(0, SIGMA, BASE.shape), t)
    assert len(stab.tracks) == 1
    assert np.abs(out - (BASE + 600.0)).max() < 3.0


def test_warmup_frames_is_a_sane_whole_number():
    w = KalmanKpsStabilizer().warmup_frames()
    assert isinstance(w, int) and 3 <= w <= 60


# --- the tracked pre-pass: stabilize_method == 'kalman' must actually be consulted -------
def _tracked(method, landmarks=True):
    from insightface.app.common import Face

    from roop.procmgr_tracking import TrackingMixin

    class Options:
        stabilize_face = True
        stabilize_method = method
        stabilize_min_cutoff = 0.05
        stabilize_beta = 0.02
        stabilize_landmarks = landmarks

    class Mgr(TrackingMixin):
        def __init__(self):
            self.options = Options()
            self.input_face_datas = []
            self._track_pose_source_map = {}

    rng = np.random.default_rng(21)
    raw = {}
    emb = rng.normal(size=512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    lm_rel = rng.normal(0, 40, (106, 2))
    obs = {}
    for i in range(60):
        kps = (BASE + np.array([800.0, 500.0]) + rng.normal(0, SIGMA, BASE.shape)).astype(np.float32)
        lm = (lm_rel + np.array([800.0, 500.0]) + (kps.mean(0) - np.array([800.0, 500.0]))).astype(np.float32)
        bbox = np.array([650.0, 350.0, 950.0, 650.0], np.float32)
        face = Face(bbox=bbox, kps=kps, det_score=0.9, embedding=emb)
        face['landmark_2d_106'] = lm
        obs[i] = face
        raw[i] = {'kps': kps.copy(), 'landmark_2d_106': lm.copy()}     # the builder mutates the Faces
    out = Mgr()._build_temporal_faces([{'id': 0, 'obs': obs, 'emb_mean': emb, 'emb_n': 60}], gap_max=10)
    return raw, out


def test_the_tracked_prepass_uses_the_kalman_method_when_selected():
    raw, out = _tracked('kalman')
    k_raw = np.stack([raw[i]['kps'] for i in range(60)])
    k_out = np.stack([out[i][0]['kps'] for i in range(60)])
    # white noise here: the filter halves the second-difference energy (see the fixture tests for the
    # measured claim); what matters is that the builder consulted the method at all
    assert _d2(k_out[20:].reshape(-1, 10)) < 0.65 * _d2(k_raw[20:].reshape(-1, 10))
    # dense landmarks ride the same displacement as the keypoint centroid
    for i in (30, 45):
        shift_kps = out[i][0]['kps'].mean(0) - raw[i]['kps'].mean(0)
        shift_lm = out[i][0]['landmark_2d_106'] - raw[i]['landmark_2d_106']
        assert np.allclose(shift_lm, shift_kps, atol=1e-3)


def test_the_default_method_still_uses_the_coupled_smoother():
    """`kalman` must not leak into the default path: one_euro + stabilize_landmarks runs the
    coupled adaptive smoother, so its output differs from the Kalman option's."""
    _, kal = _tracked('kalman')
    _, default = _tracked('one_euro')
    a = np.stack([kal[i][0]['kps'] for i in range(20, 60)])
    b = np.stack([default[i][0]['kps'] for i in range(20, 60)])
    assert not np.allclose(a, b, atol=1e-4)
