"""The Stage 2 smoothed-trajectory cache (`roop/trajectory_cache.py`): matrices, pose, file, checkpoint, route.

The matrices are solved in closed form for every face at once. A cache that held "nearly" the matrix the swap
computes would be a different face warped, so the first block pins them against the three things they stand for:
`skimage.SimilarityTransform.estimate` (the least squares `estimate_norm` runs), `face_util.estimate_norm` itself
(template, scale, SAR branch and all; to its own float32 noise, ~2e-5 px: it differs by that much between a float32
and a float64 call of the SAME keypoints) and `cv2.invertAffineTransform` (the paste-back inverse).

What the cache does NOT do is pinned too: nothing in the swap path reads its matrices (`test_stage3_does_not_read_it`),
because Stage 3's alignment depends on a crop size, template and SAR one 112-px matrix cannot stand in for.
"""
import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import trajectory_cache as tc                                    # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


# the ArcFace 112 template (insightface's), written out so the array tests need no model stack
ARC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def _similarity(scale, theta, tx, ty):
    c, s = scale * np.cos(theta), scale * np.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


def _apply(m, pts):
    return pts @ m[:, :2].T + m[:, 2]


def _kps_for(m_forward, noise=0.0, seed=0, n=1):
    """kps in the frame such that m_forward maps them onto the template (plus noise)."""
    rng = np.random.default_rng(seed)
    inv = cv2.invertAffineTransform(m_forward)
    out = []
    for _ in range(n):
        out.append(_apply(inv, ARC) + rng.normal(0, noise, ARC.shape))
    return np.stack(out)


# ── the matrices ─────────────────────────────────────────────────────────────────────────────────
class TestMatrices:
    def test_a_known_similarity_is_recovered_exactly(self):
        for scale, theta, tx, ty in [(0.4, 0.0, -20.0, 5.0), (0.25, 0.6, 7.0, -30.0), (1.7, -1.2, 100.0, 44.0)]:
            m = _similarity(scale, theta, tx, ty)
            got = tc.similarity_matrices(_kps_for(m), ARC)[0]
            assert np.abs(got - m).max() < 1e-9, (scale, theta)

    def test_noisy_points_give_what_skimage_gives(self):
        from skimage import transform as trans
        rng = np.random.default_rng(1)
        n = 400
        m = [_similarity(rng.uniform(0.15, 1.5), rng.uniform(-1.4, 1.4), rng.uniform(-80, 80), rng.uniform(-80, 80))
             for _ in range(n)]
        kps = np.stack([_kps_for(mm, noise=2.0, seed=i)[0] for i, mm in enumerate(m)])
        got = tc.similarity_matrices(kps, ARC)
        for i in range(n):
            t = trans.SimilarityTransform()
            t.estimate(kps[i], ARC)
            assert np.abs(got[i] - t.params[0:2, :]).max() < 1e-8, i

    def test_a_clip_with_non_square_pixels_is_fitted_in_display_space(self):
        from skimage import transform as trans
        sar = 16.0 / 15.0
        kps = _kps_for(_similarity(0.4, 0.2, 5, 6), noise=1.5, seed=3)[0]
        lmk = kps.copy()
        lmk[:, 0] *= sar
        t = trans.SimilarityTransform()
        t.estimate(lmk, ARC)
        want = t.params[0:2, :].copy()
        want[:, 0] *= sar
        got = tc.similarity_matrices(kps[None], ARC, sar=sar)[0]
        assert np.abs(got - want).max() < 1e-8

    def test_unusable_rows_come_back_nan_not_garbage(self):
        good = _kps_for(_similarity(0.4, 0.1, 0, 0))[0]
        bad_nan = good.copy()
        bad_nan[2, 0] = np.nan
        same = np.tile(good[0], (5, 1))                      # all five points coincide
        out = tc.similarity_matrices(np.stack([good, bad_nan, same]), ARC)
        assert np.isfinite(out[0]).all() and np.isnan(out[1]).all() and np.isnan(out[2]).all()

    def test_the_inverse_is_cv2s_and_composes_to_identity(self):
        rng = np.random.default_rng(2)
        m = np.stack([_similarity(rng.uniform(0.2, 1.5), rng.uniform(-3, 3), rng.uniform(-90, 90), rng.uniform(-90, 90))
                      for _ in range(200)])
        inv = tc.invert_matrices(m)
        for i in range(200):
            assert np.abs(inv[i] - cv2.invertAffineTransform(m[i])).max() < 1e-10
            comp = np.vstack([m[i], [0, 0, 1]]) @ np.vstack([inv[i], [0, 0, 1]])
            assert np.abs(comp - np.eye(3)).max() < 1e-10

    @pytest.mark.gpu
    def test_it_is_what_estimate_norm_returns(self):
        """The real function: template points, scale handling and the SAR branch included."""
        from roop import face_util
        import roop.globals
        rng = np.random.default_rng(4)
        kps = np.stack([_kps_for(_similarity(rng.uniform(0.2, 1.2), rng.uniform(-1, 1), rng.uniform(-50, 50),
                                              rng.uniform(-50, 50)), noise=1.5, seed=i)[0].astype(np.float32)
                        for i in range(300)])
        dst = tc.template_points()
        for sar in (1.0, 16.0 / 15.0, 64.0 / 45.0):
            was = getattr(roop.globals, 'video_sar', 1.0)
            roop.globals.video_sar = sar
            try:
                want = np.stack([face_util.estimate_norm(k, 112, 'arcface') for k in kps])
            finally:
                roop.globals.video_sar = was
            got = tc.similarity_matrices(kps, dst, sar=sar)
            assert np.abs(got - want).max() < 5e-5, sar              # estimate_norm's own float32 noise is 2e-5
            # and the real function in float64 agrees much more closely: the gap above is ITS rounding, not ours
            was = getattr(roop.globals, 'video_sar', 1.0)
            roop.globals.video_sar = sar
            try:
                want64 = np.stack([face_util.estimate_norm(k.astype(np.float64), 112, 'arcface') for k in kps])
            finally:
                roop.globals.video_sar = was
            assert np.abs(got - want64).max() < np.abs(got - want).max() + 1e-9
            assert np.abs(got - want64).max() < 1e-5, sar


# ── building ─────────────────────────────────────────────────────────────────────────────────────
def _face(track, frame, kps=None, flags=(), bbox=(0, 0, 10, 10)):
    d = {'_track_id': track, 'kps': np.asarray(kps if kps is not None else _kps_for(_similarity(0.4, 0.0, 0, 0))[0],
                                                np.float32),
         'bbox': np.asarray(bbox, np.float32)}
    for f in flags:
        d[f] = True
    return d


def _faces(n=30):
    out = {}
    for f in range(n):
        row = [_face(2, f, flags=('_interpolated',) if f % 7 == 0 else ()),
               _face(0, f, kps=_kps_for(_similarity(0.4, 0.01 * f, 3.0 * f, 0))[0],
                     flags=('_interpolated', '_flow_bridged') if f % 5 == 0 else ())]
        out[f] = row[::-1] if f % 2 else row                 # insertion order must not matter
    return out


@pytest.mark.gpu
class TestBuild:
    def test_rows_are_sorted_by_frame_then_track_with_absolute_frames(self):
        c = tc.TrajectoryCache.build(_faces(20), frame_start=100)
        assert len(c) == 40
        fr, tr = c.rows['frame'], c.rows['track']
        assert fr[0] == 100 and fr[-1] == 119
        assert np.all(np.diff(fr.astype(np.int64)) >= 0)
        for f in range(100, 120):
            assert list(tr[fr == f]) == [0, 2]
        assert c.meta['frame_start'] == 100 and c.meta['faces'] == 40 and c.meta['tracks'] == 2

    def test_flags_say_why_a_face_is_not_a_detection(self):
        c = tc.TrajectoryCache.build(_faces(20))
        by = {(int(r['frame']), int(r['track'])): int(r['flags']) for r in c.rows}
        assert by[(5, 0)] == tc.FLAG_INTERPOLATED | tc.FLAG_FLOW
        assert by[(7, 2)] == tc.FLAG_INTERPOLATED
        assert by[(3, 0)] == 0

    def test_matrices_are_the_smoothed_keypoints_matrices(self):
        c = tc.TrajectoryCache.build(_faces(10))
        for r in c.rows:
            want = tc.similarity_matrices(np.asarray(r['kps'], np.float64)[None], tc.template_points())[0]
            assert np.abs(r['M'] - want).max() < 1e-12
            comp = np.vstack([r['M'], [0, 0, 1]]) @ np.vstack([r['M_inv'], [0, 0, 1]])
            assert np.abs(comp - np.eye(3)).max() < 1e-9

    def test_the_pose_is_solved_from_the_stored_keypoints_in_head_pose_convention(self):
        from roop.head_pose import head_pose_from_kps
        c = tc.TrajectoryCache.build(_faces(10))
        for r in c.rows:
            hp = head_pose_from_kps(np.asarray(r['kps']))
            assert hp is not None
            assert np.allclose(r['pose'], (hp.yaw, hp.pitch, hp.roll), atol=1e-4)
        # a template-shaped face is frontal: yaw and pitch near zero, roll near zero
        straight = tc.TrajectoryCache.build({0: [_face(0, 0)]})
        yaw, pitch, roll = straight.rows['pose'][0]
        assert abs(roll) < 1.0 and abs(yaw) < 40.0 and abs(pitch) < 40.0

    def test_a_face_without_five_finite_keypoints_or_a_track_is_left_out(self):
        bad = {0: [_face(0, 0), {'kps': None, '_track_id': 1}, _face(None, 0),
                   _face(3, 0, kps=np.full((5, 2), np.nan)), {'_track_id': 4, 'kps': np.zeros((3, 2))}]}
        c = tc.TrajectoryCache.build(bad)
        assert len(c) == 1 and int(c.rows['track'][0]) == 0

    def test_an_empty_clip_builds_an_empty_cache(self):
        c = tc.TrajectoryCache.build({})
        assert len(c) == 0 and c.frame_range is None
        assert tc.pose_payload(c, 0)['faces'] == []


# ── the file ─────────────────────────────────────────────────────────────────────────────────────
class TestFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='traj-')
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = os.path.join(self.dir, 'p.trajectory.npy')
        rows = np.zeros(50, dtype=tc.ROW)
        rows['frame'] = np.arange(50) // 2 + 10
        rows['track'] = np.arange(50) % 2
        rows['kps'] = np.random.default_rng(0).uniform(0, 500, (50, 5, 2))
        rows['pose'] = np.random.default_rng(1).uniform(-40, 40, (50, 3))
        rows['M'] = np.random.default_rng(2).normal(size=(50, 2, 3))
        self.cache = tc.TrajectoryCache(rows, {'frame_start': 10, 'faces': 50})

    def test_save_open_is_lossless_and_memory_mapped(self):
        entry = self.cache.save(self.path)
        got = tc.TrajectoryCache.open(self.path, meta=entry)
        assert isinstance(got.rows, np.memmap) and got.rows.dtype == tc.ROW
        assert got.rows.tobytes() == self.cache.rows.tobytes()
        assert entry['sidecar']['size'] == os.path.getsize(self.path)
        import hashlib
        assert entry['sidecar']['sha256'] == hashlib.sha256(open(self.path, 'rb').read()).hexdigest()
        assert not got.rows.flags.writeable                      # a view of a checkpoint is read-only

    def test_the_write_is_atomic_and_leaves_no_temp_file(self):
        self.cache.save(self.path)
        self.cache.save(self.path)
        assert os.listdir(self.dir) == ['p.trajectory.npy']

    def test_a_file_of_another_shape_is_refused_not_misread(self):
        np.save(self.path, np.zeros(5, dtype=np.float32))
        with pytest.raises(ValueError):
            tc.TrajectoryCache.open(self.path)

    def test_a_pickled_file_cannot_execute_code(self):
        np.save(self.path, np.array([object()], dtype=object), allow_pickle=True)
        with pytest.raises(ValueError):
            tc.TrajectoryCache.open(self.path)

    def test_queries_at_a_frame_near_a_frame_and_a_series(self):
        c = self.cache
        assert len(c.at(10)) == 2 and len(c.at(34)) == 2 and len(c.at(35)) == 0
        assert len(c.at(0, absolute=False)) == 2                    # job-relative: frame_start is 10
        used, rows = c.nearest(40)
        assert used == 34 and len(rows) == 2                        # 6 frames past the end is within the window
        used, rows = c.nearest(500)
        assert len(rows) == 0
        s = c.series(0, max_points=5)
        assert len(s['frame']) == 5 and len(s['yaw']) == 5
        summ = c.summary()
        assert (summ['first_frame'], summ['last_frame'], summ['faces'], summ['track_ids']) == (10, 34, 50, [0, 1])

    def test_pose_payload_names_what_it_found_and_what_it_was_asked(self):
        p = tc.pose_payload(self.cache, 5)                          # job-relative frame 5 -> absolute 15
        assert p['available'] and p['frame'] == 15 and p['requested'] == 5 and p['frame_start'] == 10
        assert [f['track'] for f in p['faces']] == [0, 1]
        assert set(p['faces'][0]) >= {'yaw', 'pitch', 'roll', 'flags', 'bbox', 'scale'}
        assert tc.pose_payload(None, 0)['available'] is False


# ── the match against restored faces ───────────────────────────────────────────────────────────────
@pytest.mark.gpu
class TestWindow:
    def test_a_resumed_window_maps_the_matching_rows_and_nothing_else(self):
        faces = _faces(40)
        full = tc.TrajectoryCache.build(faces, frame_start=100)
        later = {f - 15: v for f, v in faces.items() if f >= 15}          # a resume that starts 15 frames in
        win = full.window_for(later, frame_start=115)
        assert win is not None and len(win) == len(full) - 30
        assert int(win.rows['frame'][0]) == 115 and win.meta['frame_start'] == 115

    def test_one_keypoint_that_differs_refuses_the_file(self):
        faces = _faces(20)
        full = tc.TrajectoryCache.build(faces)
        changed = {f: [dict(x) for x in v] for f, v in faces.items()}
        changed[9][0]['kps'] = changed[9][0]['kps'] + np.float32(0.01)
        assert full.window_for(changed) is None
        assert full.window_for(faces) is not None

    def test_a_missing_face_or_a_different_track_refuses_the_file(self):
        faces = _faces(20)
        full = tc.TrajectoryCache.build(faces)
        fewer = {f: v[:1] for f, v in faces.items()}
        assert full.window_for(fewer) is None
        retag = {f: [dict(x, _track_id=7) for x in v] for f, v in faces.items()}
        assert full.window_for(retag) is None


# ── the registry ─────────────────────────────────────────────────────────────────────────────────
def test_the_active_cache_is_set_cleared_and_not_shared_across_jobs():
    tc.clear_active()
    assert tc.active() is None
    c = tc.TrajectoryCache(np.zeros(0, dtype=tc.ROW))
    tc.set_active(c, 'proj-1')
    assert tc.active() is c and tc.active_project() == 'proj-1'
    tc.clear_active()
    assert tc.active() is None and tc.active_project() is None


# ── the project checkpoint ───────────────────────────────────────────────────────────────────────
from tests.test_stage_cache import RecordCase, make_pm                       # noqa: E402
import project_checkpoint as pc                                              # noqa: E402
import stage_checkpoint as sck                                               # noqa: E402
from roop import stage_cache as sc                                           # noqa: E402


@pytest.mark.gpu
class TestCheckpoint(RecordCase):
    def _stage2(self, pm=None):
        pm = pm or make_pm(30)
        entry = sc.write_stage2(pc.stage_sidecar_path(self.pid, 'stage2'), pm, frame_start=0, frame_count=30,
                                ranges_abs=None, plan=pm.stage2_frame_plan)
        pc.record_stage(self.pid, 'stage2', entry, live=self.live)
        return pm

    def _adapter(self):
        logs = []
        cache = sck.ProjectStageCache(self.pid, app_version='abc123',
                                      log=lambda *a, **k: logs.append(' '.join(map(str, a))))
        cache._live = self.live
        return cache, logs

    def _record_trajectory(self, pm):
        traj = tc.TrajectoryCache.build(pm._temporal_faces, frame_start=0)
        cache, logs = self._adapter()
        assert cache.save_trajectory(traj), logs
        return traj

    def test_it_is_recorded_beside_stage2_and_found_again(self):
        pm = self._stage2()
        self._record_trajectory(pm)
        found = pc.usable_stage_cache(self.pid, self.live)
        assert found['stage2'] is not None and found['trajectory'] is not None, found['reasons']
        assert os.path.exists(pc.stage_sidecar_path(self.pid, 'trajectory'))

    def test_a_resume_maps_it_only_when_it_matches_the_restored_faces(self):
        pm = self._stage2()
        traj = self._record_trajectory(pm)
        cache, logs = self._adapter()
        win = cache.restore_trajectory(pm._temporal_faces, frame_start=0)
        assert win is not None and len(win) == len(traj), logs
        assert isinstance(win.rows, np.memmap)
        # different faces under the same key: refused, with the reason printed
        other = make_pm(30, seed=99)
        cache2, logs2 = self._adapter()
        assert cache2.restore_trajectory(other._temporal_faces, frame_start=0) is None
        assert any('does not match' in line for line in logs2), logs2

    def test_a_new_stage2_drops_the_trajectory_derived_from_the_old_one(self):
        pm = self._stage2()
        self._record_trajectory(pm)
        self._stage2(make_pm(30, seed=5))                              # stage 2 recorded again
        found = pc.usable_stage_cache(self.pid, self.live)
        assert found['trajectory'] is None
        assert not os.path.exists(pc.stage_sidecar_path(self.pid, 'trajectory'))

    def test_a_new_stage1_drops_it_too(self):
        pm = self._stage2()
        self._record_trajectory(pm)
        pc.record_stage(self.pid, 'stage1', {'frame_start': 0, 'frame_count': 30, 'active_ranges': [[0, 29]],
                                             'mode': 'x', 'digest': 'd'}, live=self.live)
        assert pc.usable_stage_cache(self.pid, self.live)['trajectory'] is None
        assert not os.path.exists(pc.stage_sidecar_path(self.pid, 'trajectory'))

    def test_a_trajectory_without_a_usable_stage2_is_not_offered(self):
        pm = self._stage2()
        self._record_trajectory(pm)
        os.remove(pc.stage_sidecar_path(self.pid, 'stage2'))
        found = pc.usable_stage_cache(self.pid, self.live)
        assert found['stage2'] is None and found['trajectory'] is None

    def test_a_tampered_trajectory_file_is_refused(self):
        pm = self._stage2()
        self._record_trajectory(pm)
        path = pc.stage_sidecar_path(self.pid, 'trajectory')
        data = bytearray(open(path, 'rb').read())
        data[len(data) // 2] ^= 0x01
        open(path, 'wb').write(bytes(data))
        found = pc.usable_stage_cache(self.pid, self.live)
        assert found['trajectory'] is None
        assert any('trajectory cache file digest differs' in r for r in found['reasons']), found['reasons']

    def test_completing_the_project_deletes_it(self):
        pm = self._stage2()
        self._record_trajectory(pm)
        path = pc.stage_sidecar_path(self.pid, 'trajectory')
        assert os.path.exists(path)
        pc.update_state(self.pid, 'COMPLETED')
        assert not os.path.exists(path)

    def test_a_cache_that_cannot_be_written_is_reported_not_raised(self):
        cache, logs = self._adapter()
        assert cache.save_trajectory(None) is False
        assert cache.save_trajectory(tc.TrajectoryCache(np.zeros(0, dtype=tc.ROW))) is False


# ── the pipeline wiring and the route ────────────────────────────────────────────────────────────────
def _src(rel):
    return open(os.path.join(APP, rel), encoding='utf-8').read()


def test_the_prepass_seals_the_trajectory_and_the_batch_runner_records_it():
    tracking = _src('roop/procmgr_tracking.py')
    assert 'self._seal_trajectory(frame_start, frame_count)' in tracking
    batch = _src('roop/procmgr_batch.py')
    assert "self._seal_trajectory(frame_start, frame_count, restored=True)" in batch
    assert "_stage_cache.save_trajectory(getattr(self, '_trajectory', None))" in batch
    assert '_trajectory_cache.clear_active()' in batch                 # a new job drops the old job's heads


def test_stage3_does_not_read_it():
    """The honest limit: nothing on the swap path takes a matrix from this cache (see the module docstring)."""
    for rel in ('roop/ProcessMgr.py', 'roop/face_util.py', 'roop/processors/frame/face_swapper.py',
                'roop/processors/frame/onnx_swappers.py', 'roop/procmgr_masking.py'):
        assert 'trajectory_cache' not in _src(rel), rel


def test_the_route_is_registered_and_reads_the_active_cache():
    api = _src('api.py')
    assert 'import routes_trajectory as _routes_trajectory' in api
    assert 'app.include_router(_routes_trajectory.router)' in api
    route = _src('routes_trajectory.py')
    for path in ('/api/trajectory/summary', '/api/trajectory/pose', '/api/trajectory/series'):
        assert path in route


def test_the_route_answers_from_the_active_cache():
    pytest.importorskip('fastapi')
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routes_trajectory
    app = FastAPI()
    app.include_router(routes_trajectory.router)
    client = TestClient(app)
    tc.clear_active()
    assert client.get('/api/trajectory/summary').json()['available'] is False
    assert client.get('/api/trajectory/pose?frame=3').json()['available'] is False
    rows = np.zeros(6, dtype=tc.ROW)
    rows['frame'] = [10, 10, 11, 11, 12, 12]
    rows['track'] = [0, 1] * 3
    rows['pose'] = np.arange(18).reshape(6, 3)
    rows['M'][:, 0, 0] = 1.0
    tc.set_active(tc.TrajectoryCache(rows, {'frame_start': 10}), 'p1')
    try:
        s = client.get('/api/trajectory/summary').json()
        assert s['available'] and s['faces'] == 6 and s['project_id'] == 'p1' and s['track_ids'] == [0, 1]
        p = client.get('/api/trajectory/pose?frame=1').json()           # job-relative 1 -> absolute 11
        assert p['frame'] == 11 and [f['track'] for f in p['faces']] == [0, 1] and p['faces'][0]['yaw'] == 6.0
        assert client.get('/api/trajectory/pose?frame=11&absolute=true').json()['frame'] == 11
        ser = client.get('/api/trajectory/series?track=1&points=2').json()
        assert ser['available'] and len(ser['frame']) == 2
        assert client.get('/api/trajectory/pose?frame=-1').status_code == 422
    finally:
        tc.clear_active()
