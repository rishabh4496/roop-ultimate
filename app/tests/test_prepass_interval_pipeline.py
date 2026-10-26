"""`prepass_detect_interval` through the REAL pre-pass (`_precompute_temporal`), on real image frames.

The scheduler tests (test_prepass_interval.py) drive `TemporalFaceTracker` alone. These drive the whole of Stage 2:
decode loop -> tracker plan -> detector (stubbed, and COUNTED) -> association -> optical-flow chain between the
detections -> gap-fill -> `_temporal_faces`, with a textured face moving through synthetic frames so the flow has
something to follow and the truth is known exactly.

Two paths, because the original defect only existed on one of them: the direct path (one detector, results consumed
at once) and the pooled path (results consumed up to pool_workers + 2 frames after the frame was planned).
"""
import collections
import contextlib
import io
import os
import sys
from unittest import mock

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
pytestmark = pytest.mark.gpu            # imports the pre-pass module (insightface / the ML stack), like its siblings

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from insightface.app.common import Face                                  # noqa: E402

import roop.globals                                                      # noqa: E402
from roop.procmgr_tracking import TrackingMixin                          # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


N, H, W = 80, 240, 320
FACE_W, FACE_H = 84, 108


def _blur_noise(h, w, seed, sigma=2.0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), sigma)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _path(t):
    """Where the face centre is at frame t: a steady drift with a slow sway."""
    return 90.0 + 1.6 * t, 120.0 + 12.0 * np.sin(t / 14.0)


def _frames():
    bg = _blur_noise(H, W, 1, sigma=3.0)
    face = _blur_noise(FACE_H, FACE_W, 2, sigma=1.6)
    frames, truth = [], []
    for t in range(N):
        cx, cy = _path(t)
        x0, y0 = int(round(cx - FACE_W / 2)), int(round(cy - FACE_H / 2))
        g = bg.copy()
        g[y0:y0 + FACE_H, x0:x0 + FACE_W] = face
        frames.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        truth.append((x0, y0))
    return frames, truth


EMB = np.random.default_rng(7).normal(size=512).astype(np.float32)
EMB /= np.linalg.norm(EMB)


def _face_at(x0, y0):
    cx, cy = x0 + FACE_W / 2.0, y0 + FACE_H / 2.0
    kps = np.array([[cx - 20, cy - 18], [cx + 20, cy - 18], [cx, cy], [cx - 16, cy + 28], [cx + 16, cy + 28]],
                   np.float32)
    ang = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    lm = np.stack([cx + 34 * np.cos(ang), cy + 46 * np.sin(ang)], 1).astype(np.float32)
    return Face(bbox=np.array([x0, y0, x0 + FACE_W, y0 + FACE_H], np.float32), kps=kps, landmark_2d_106=lm,
                det_score=0.95, embedding=EMB.copy())


class Options:
    stabilize_face = False
    stabilize_method = 'one_euro'
    stabilize_min_cutoff = 0.05
    stabilize_beta = 0.02
    face_distance_threshold = 0.75
    selected_index = 0


class Captured:
    embedding = EMB


class Mgr(TrackingMixin):
    def __init__(self):
        self.options = Options()
        self.input_face_datas = []
        self._track_pose_source_map = {}
        self.target_face_datas = [Captured()]
        self.target_face_groups = [0]
        self.progress_gradio = None
        self._track_assignments = {}
        self._track_scanned = 0

    def _publish_live(self, frame):
        pass


def run_prepass(k, *, pooled, hints=None, monkeypatch):
    monkeypatch.setenv('ROOP_PREPASS_DETECT_INTERVAL', str(k))
    monkeypatch.delenv('ROOP_TEMPORAL_ROI_INTERVAL', raising=False)
    frames, truth = _frames()
    by_id = {id(f): i for i, f in enumerate(frames)}
    calls = collections.OrderedDict()          # frame -> 'full' | 'roi'

    def full(frame, *a, **kw):
        calls[by_id[id(frame)]] = 'full'
        return [_face_at(*truth[by_id[id(frame)]])]

    def roi(frame, crop, *a, **kw):
        calls.setdefault(by_id[id(frame)], 'roi')
        return [_face_at(*truth[by_id[id(frame)]])]

    mgr = Mgr()
    if hints is not None:
        mgr._active_ranges_stats = {'scene_cuts': list(hints)}
    patches = [mock.patch('roop.face_util.get_all_faces', full), mock.patch('roop.face_util.get_all_faces_in_roi', roi)]
    # Pinned either way: this machine's own config would otherwise decide (a 12 GB card pools its detectors, so the
    # "direct" path was not direct at all and every result was consumed after a lag).
    patches += [mock.patch('roop.procmgr_tracking.session_pool.detmask_pooling_enabled', lambda: bool(pooled)),
                mock.patch('roop.procmgr_tracking.session_pool.detmask_pool_size', lambda *a, **kw: 3 if pooled else 1)]
    was, roop.globals.processing = roop.globals.processing, True
    out = io.StringIO()
    try:
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(out))
            mgr._precompute_temporal(None, frames, 0, N, N)
    finally:
        roop.globals.processing = was
    return mgr, calls, truth, out.getvalue()


def _kps_error(mgr, truth):
    """Per frame: the centroid error of the cached face against the exact path, in pixels (None if no face)."""
    errs = {}
    for f, faces in mgr._temporal_faces.items():
        if not faces:
            continue
        cx, cy = truth[f][0] + FACE_W / 2.0, truth[f][1] + FACE_H / 2.0
        c = np.asarray(faces[0]['kps']).mean(0)
        # the stub's five keypoints average to (0, +4.0) below the box centre by construction
        # ((-18 - 18 + 0 + 28 + 28) / 5); compare like with like
        errs[f] = float(np.hypot(c[0] - cx, c[1] - (cy + 4.0)))
    return errs


@pytest.mark.parametrize('pooled', [False, True], ids=['direct', 'pooled'])
class TestIntervalThroughTheRealPrepass:
    def test_interval_one_detects_every_frame_and_invents_nothing(self, pooled, monkeypatch):
        mgr, calls, truth, _ = run_prepass(1, pooled=pooled, monkeypatch=monkeypatch)
        assert len(calls) == N
        assert all(not f.get('_interpolated') for fs in mgr._temporal_faces.values() for f in fs)
        assert mgr._prepass_schedule['interval'] == 1 and mgr._prepass_schedule['coast'] == 0

    def test_interval_four_detects_about_a_third_or_less_and_still_covers_every_frame(self, pooled, monkeypatch):
        mgr, calls, truth, text = run_prepass(4, pooled=pooled, monkeypatch=monkeypatch)
        assert len(calls) <= 0.36 * N, (pooled, len(calls), sorted(calls))
        assert mgr._prepass_schedule['interval'] == 4
        assert mgr._prepass_schedule['coast'] >= 0.6 * N
        assert set(mgr._temporal_faces) == set(range(N)), 'a frame lost its face'
        assert 'prepass_detect_interval 4' in text and 'detector call(s)' in text

    def test_the_first_gap_of_a_track_is_flow_bridged_too(self, pooled, monkeypatch):
        """A chain used to start only after a track's third observation: at K = 4 that is frame 8, so the first
        two gaps of every track were bridged by geometry alone. With one observation enough, frames 1-3 are
        placed by flow. Only a synchronous run HAS that gap: a pooled one detects frames 0..pool_workers+2 while
        its first result is still in flight (the startup burst), so the track is three observations old by then."""
        mgr, calls, truth, _ = run_prepass(4, pooled=pooled, monkeypatch=monkeypatch)
        if pooled:
            assert all(f in calls for f in range(0, 4)), 'the startup burst detects the first frames'
            return
        assert sorted(calls)[:4] == [0, 4, 8, 12]
        first = [mgr._temporal_faces[f][0] for f in (1, 2, 3)]
        assert all(f.get('_flow_bridged') for f in first), [bool(f.get('_flow_bridged')) for f in first]

    def test_the_frames_between_detections_are_placed_by_optical_flow_to_within_a_few_pixels(self, pooled, monkeypatch):
        mgr, calls, truth, _ = run_prepass(4, pooled=pooled, monkeypatch=monkeypatch)
        errs = _kps_error(mgr, truth)
        bridged = [f for f, faces in mgr._temporal_faces.items() if faces[0].get('_interpolated')]
        assert len(bridged) >= 0.55 * N
        flow = [f for f in bridged if mgr._temporal_faces[f][0].get('_flow_bridged')]
        assert len(flow) >= 0.6 * len(bridged), (len(flow), len(bridged))
        assert max(errs[f] for f in bridged) < 4.0, sorted((round(errs[f], 1), f) for f in bridged)[-5:]
        assert float(np.median([errs[f] for f in bridged])) < 1.5
        assert max(errs[f] for f in calls if f in errs) < 1.0           # a detected frame IS the detection

    def test_a_stage1_cut_hint_forces_a_full_detection_on_a_frame_that_would_have_coasted(self, pooled, monkeypatch):
        _, plain, _, _ = run_prepass(4, pooled=pooled, monkeypatch=monkeypatch)
        hinted_frame = next(f for f in range(30, 60) if f not in plain)           # a coast slot
        mgr, calls, _, text = run_prepass(4, pooled=pooled, hints=[hinted_frame], monkeypatch=monkeypatch)
        assert mgr._stage1_cut_hints == [hinted_frame]
        assert calls.get(hinted_frame) == 'full'
        # ... and so does the frame before it: the last frame of the shot needs an anchor inside its own shot
        assert calls.get(hinted_frame - 1) == 'full'
        assert mgr._prepass_schedule['stage1_cut_hints'] == 2
        assert '2 full detection(s) forced by a Stage 1 cut hint' in text
        # a hint is a SCHEDULING hint, not a shot boundary: gap-fill across it is still allowed
        assert hinted_frame not in (getattr(mgr, '_shot_boundaries', None) or set())

    def test_the_default_run_without_hints_reports_none(self, pooled, monkeypatch):
        mgr, calls, _, _ = run_prepass(4, pooled=pooled, monkeypatch=monkeypatch)
        assert mgr._stage1_cut_hints == [] and mgr._prepass_schedule['stage1_cut_hints'] == 0


def test_hints_are_rebased_to_the_scan_origin_and_junk_is_ignored(monkeypatch):
    frames, truth = _frames()
    mgr = Mgr()
    mgr._active_ranges_stats = {'scene_cuts': [100, 130, 110.5, 'x', None, 100, 90, -5]}
    by_id = {id(f): i for i, f in enumerate(frames)}
    with mock.patch('roop.face_util.get_all_faces', lambda fr, *a, **k: [_face_at(*truth[by_id[id(fr)]])]), \
            mock.patch('roop.face_util.get_all_faces_in_roi', lambda fr, c, *a, **k: [_face_at(*truth[by_id[id(fr)]])]):
        was, roop.globals.processing = roop.globals.processing, True
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mgr._precompute_temporal(None, frames, 100, 100 + N, N)
        finally:
            roop.globals.processing = was
    # absolute decoder frames 100 and 130 with a scan that starts at 100 -> 0 is dropped (not a cut), 30 stays;
    # 90 and -5 fall before the scan; floats and strings are not frame numbers
    assert mgr._stage1_cut_hints == [30]


def test_a_stage1_that_did_not_run_leaves_no_hints(monkeypatch):
    frames, truth = _frames()
    mgr = Mgr()
    by_id = {id(f): i for i, f in enumerate(frames)}
    with mock.patch('roop.face_util.get_all_faces', lambda fr, *a, **k: [_face_at(*truth[by_id[id(fr)]])]), \
            mock.patch('roop.face_util.get_all_faces_in_roi', lambda fr, c, *a, **k: [_face_at(*truth[by_id[id(fr)]])]):
        was, roop.globals.processing = roop.globals.processing, True
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mgr._precompute_temporal(None, frames, 0, N, N)
        finally:
            roop.globals.processing = was
    assert mgr._stage1_cut_hints == []


def test_the_batch_runner_forgets_a_previous_jobs_stage1_answer():
    text = open(os.path.join(APP, 'roop', 'procmgr_batch.py'), encoding='utf-8').read()
    assert 'self._active_ranges_stats = None' in text


# ── the tail: faces after a track's LAST detection ─────────────────────────────────────────────────
from types import SimpleNamespace                                         # noqa: E402

from roop.flow_coast import FlowResult                                    # noqa: E402


def _flow_result(frame, consistent=True):
    f = _face_at(100 + 2 * frame, 80)
    return FlowResult(consistent, f['landmark_2d_106'].copy(), f['kps'].copy(), f['bbox'].copy(),
                      np.eye(2, 3, dtype=np.float32), 100, 90, 0.9, 0.2, 'ok')


def _tail_mgr(k, cuts=(), others=None):
    mgr = Mgr()
    mgr._temporal_tracker = SimpleNamespace(detect_interval=k)
    mgr._shot_boundaries = set(cuts)
    return mgr


def _tail_track(last_real=3, flow_frames=range(4, 11), bad=(), missing=()):
    """`bad`: a flow result is STORED but not consistent; `missing`: no result at all for that frame."""
    obs = {i: _face_at(100 + 2 * i, 80) for i in range(0, last_real + 1)}
    flow = {i: _flow_result(i, consistent=(i not in bad)) for i in flow_frames if i not in missing}
    return {'id': 0, 'obs': obs, 'emb_mean': EMB.copy(), 'emb_n': len(obs), 'flow': flow}


class TestTail:
    def test_it_extends_at_most_k_minus_one_frames_and_flags_them(self):
        for k in (2, 4, 8):
            mgr = _tail_mgr(k)
            out = mgr._build_temporal_faces([_tail_track()], gap_max=10)
            tail = sorted(f for f in out if f > 3)
            assert tail == list(range(4, 4 + min(k - 1, 7))), (k, tail)
            face = out[4][0]
            assert face.get('_flow_tail') and face.get('_flow_coasted') and face.get('_interpolated')
            assert face['_track_id'] == 0
            assert mgr._flow_tail == len(tail)

    def test_interval_one_has_no_tail_so_nothing_changes_for_the_default(self):
        out = _tail_mgr(1)._build_temporal_faces([_tail_track()], gap_max=10)
        assert sorted(out) == [0, 1, 2, 3]
        out = Mgr()._build_temporal_faces([_tail_track()], gap_max=10)        # no tracker state at all
        assert sorted(out) == [0, 1, 2, 3]

    def test_it_stops_at_a_shot_cut(self):
        out = _tail_mgr(8, cuts={6})._build_temporal_faces([_tail_track()], gap_max=10)
        assert sorted(f for f in out if f > 3) == [4, 5]            # a cut at 6 starts a new shot

    def test_it_stops_where_the_flow_chain_stopped(self):
        out = _tail_mgr(8)._build_temporal_faces([_tail_track(missing={6})], gap_max=10)
        assert sorted(f for f in out if f > 3) == [4, 5]

    def test_it_stops_at_a_flow_step_that_was_stored_but_is_not_consistent(self):
        out = _tail_mgr(8)._build_temporal_faces([_tail_track(bad={6})], gap_max=10)
        assert sorted(f for f in out if f > 3) == [4, 5]

    def test_it_needs_flow_evidence_at_all(self):
        track = _tail_track()
        track['flow'] = {}
        assert sorted(_tail_mgr(8)._build_temporal_faces([track], gap_max=10)) == [0, 1, 2, 3]

    def test_it_does_not_paint_onto_another_tracks_real_face(self):
        other = {'id': 1, 'obs': {5: _face_at(100 + 2 * 5, 80)}, 'emb_mean': EMB.copy(), 'emb_n': 1}
        out = _tail_mgr(8)._build_temporal_faces([_tail_track(), other], gap_max=10)
        assert [f for f in out if f > 3 and any(x.get('_track_id') == 0 for x in out[f])] == [4]

    def test_the_68_landmarks_ride_the_keypoint_displacement(self):
        track = _tail_track()
        base = track['obs'][3]
        base['landmark_3d_68'] = np.zeros((68, 3), np.float32)
        out = _tail_mgr(4)._build_temporal_faces([track], gap_max=10)
        shift = out[4][0]['kps'].mean(0) - base['kps'].mean(0)
        assert np.allclose(out[4][0]['landmark_3d_68'][:, :2], shift, atol=1e-4)
        assert np.all(out[4][0]['landmark_3d_68'][:, 2] == 0)
