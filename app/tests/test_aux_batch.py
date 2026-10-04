"""Batched GPU-crop aux models (roop/aux_batch.py) and their face_util seam.

Pinned here, without a GPU or the buffalo_l sessions:

  * the crops: `affine_crops` is `cv2.warpAffine` BIT FOR BIT -- the recogniser's 5-point
    transform and the landmark models' box transform, crops cut by or outside the frame,
    and a crop taken from a sub-region of the frame. Not "close": the 68-point regressor
    runs in FP16 and moves by about a pixel of crop space for ONE grey level of input;
  * insightface's post-processing, vectorised: `_trans_points` equals
    `face_align.trans_points`, `_post_landmark` equals `Landmark.get` after `session.run`;
  * the leader/follower coalescing: every caller's faces are filled in, concurrent callers
    share batches, a batch is capped at `max_batch`, an error reaches every caller in the
    batch, and a closed or incomplete engine RAISES (it must never be a silent no-op, which
    would read as faces that matched nobody);
  * the seam: outside `aux_batch_scope` `_apply_aux` is the original per-face loop; inside,
    a failing engine is counted and the per-face loop still produces the result.
"""
import os
import sys
import threading
import time
import types

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

torch = pytest.importorskip('torch')
ab = pytest.importorskip('roop.aux_batch')
fu = pytest.importorskip('roop.face_util')
from insightface.app.common import Face                       # noqa: E402
from insightface.utils import face_align                      # noqa: E402


def _image(h=720, w=1280, seed=0):
    rng = np.random.RandomState(seed)
    return cv2.GaussianBlur(rng.randint(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 2)


def _kps(rng, w, h):
    cx, cy = rng.uniform(-100, w + 100), rng.uniform(-100, h + 100)
    s = rng.uniform(40, 500)
    k = np.array([[cx - .3 * s, cy - .15 * s], [cx + .3 * s, cy - .15 * s], [cx, cy + .05 * s],
                  [cx - .2 * s, cy + .3 * s], [cx + .2 * s, cy + .3 * s]], np.float32)
    return k + rng.randn(5, 2).astype(np.float32) * s * 0.03, (cx, cy, s)


class TestCropGeometry:
    @pytest.mark.parametrize('kind,size', [('norm', 112), ('bbox', 192)])
    def test_equals_cv2_warp_affine_exactly(self, kind, size):
        img = np.random.RandomState(2).randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        rng = np.random.RandomState(3)
        roi = torch.from_numpy(img)
        for _ in range(40):
            kps, (cx, cy, s) = _kps(rng, 1280, 720)
            m = (ab.norm_matrix(kps, size) if kind == 'norm'
                 else ab.bbox_matrix(np.array([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2],
                                              np.float32), size))
            ref = cv2.warpAffine(img, m, (size, size), borderValue=0.0)
            out = ab.affine_crops(roi, (0, 0), m[None], size)[0].permute(1, 2, 0).numpy()
            assert np.array_equal(out, ref.astype(np.float32))

    def test_a_crop_from_a_sub_region_equals_the_crop_from_the_frame(self):
        img = _image(seed=1)
        rng = np.random.RandomState(4)
        kps, _ = _kps(rng, 1280, 720)
        kps = np.clip(kps, 300, 900).astype(np.float32)           # a face well inside the frame
        m = ab.norm_matrix(kps, 112)
        x0, y0, x1, y1 = ab.source_footprint(ab.invert_affine(m[None]), 112)
        ox, oy = int(x0) - 2, int(y0) - 2
        sub = np.ascontiguousarray(img[oy:int(y1) + 3, ox:int(x1) + 3])
        full = ab.affine_crops(torch.from_numpy(img), (0, 0), m[None], 112)
        part = ab.affine_crops(torch.from_numpy(sub), (ox, oy), m[None], 112)
        assert torch.equal(full, part)

    def test_a_batch_equals_its_faces_one_at_a_time(self):
        img = _image(seed=1)
        rng = np.random.RandomState(4)
        roi = torch.from_numpy(img)
        ms = np.stack([ab.norm_matrix(_kps(rng, 1280, 720)[0], 112) for _ in range(5)])
        both = ab.affine_crops(roi, (0, 0), ms, 112)
        for i in range(5):
            assert torch.equal(both[i:i + 1], ab.affine_crops(roi, (0, 0), ms[i:i + 1], 112))

    def test_bbox_matrix_equals_insightface_transform(self):
        box = np.array([100.5, 80.25, 260.75, 300.0], np.float32)
        w, h = box[2] - box[0], box[3] - box[1]
        center = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
        _c, m_ref = face_align.transform(np.zeros((400, 400, 3), np.uint8), center, 192,
                                         192 / (max(w, h) * 1.5), 0)
        np.testing.assert_allclose(ab.bbox_matrix(box, 192), m_ref, atol=1e-9)

    def test_invert_affine_equals_cv2(self):
        rng = np.random.RandomState(5)
        ms = np.stack([ab.norm_matrix(_kps(rng, 1280, 720)[0], 112) for _ in range(4)])
        ref = np.stack([cv2.invertAffineTransform(m) for m in ms])
        np.testing.assert_allclose(ab.invert_affine(ms), ref, atol=1e-9)

    def test_footprint_contains_every_tap(self):
        rng = np.random.RandomState(6)
        kps, _ = _kps(rng, 1280, 720)
        m = ab.norm_matrix(kps, 112)
        inv = ab.invert_affine(m[None])
        x0, y0, x1, y1 = ab.source_footprint(inv, 112)
        for u, v in ((0, 0), (111, 0), (0, 111), (111, 111), (55, 55)):
            sx, sy = inv[0] @ np.array([u, v, 1.0])
            assert x0 - 1e-9 <= sx <= x1 + 1e-9 and y0 - 1e-9 <= sy <= y1 + 1e-9


class TestPostProcessing:
    def test_trans_points_equal_insightface_2d_and_3d(self):
        rng = np.random.RandomState(7)
        m = ab.bbox_matrix([10, 20, 210, 260], 192)
        for dim in (2, 3):
            pts = (rng.randn(68, dim) * 40 + 96).astype(np.float32)
            ref = face_align.trans_points(pts, m)
            np.testing.assert_allclose(ab._trans_points(pts, m), ref, atol=1e-4)

    @pytest.mark.parametrize('lmk_num,dim,pose', [(106, 2, False), (68, 3, True)])
    def test_post_landmark_equals_landmark_get(self, lmk_num, dim, pose):
        from insightface.model_zoo.landmark import Landmark
        model = Landmark.__new__(Landmark)
        model.input_size, model.lmk_num, model.lmk_dim = (192, 192), lmk_num, dim
        model.taskname = 'landmark_%dd_%d' % (dim, lmk_num)
        model.require_pose = pose
        model.input_mean, model.input_std = 127.5, 128.0
        model.input_name, model.output_names = 'data', ['fc1']
        if pose:
            from insightface.data import get_object
            model.mean_lmk = get_object('meanshape_68.pkl')
        rng = np.random.RandomState(8)
        raw = (rng.rand(lmk_num * dim if not pose else 3309) * 2 - 1).astype(np.float32)

        class Session:
            def run(self, *_a, **_k):
                return [raw[None].copy()]
        model.session = Session()
        box = np.array([100, 120, 300, 340], np.float32)
        a = Face(bbox=box.copy(), kps=None, det_score=1.0)
        model.get(np.zeros((480, 640, 3), np.uint8), a)
        b = Face(bbox=box.copy(), kps=None, det_score=1.0)
        ab._post_landmark(model, raw.copy(), ab.bbox_matrix(b.bbox, 192), b)
        np.testing.assert_allclose(b[model.taskname], a[model.taskname], atol=1e-3)
        if pose:
            np.testing.assert_allclose(b['pose'], a['pose'], atol=1e-4)


class _FakeEngine(ab.AuxBatchEngine):
    """The coalescing machinery with `_process` replaced by a recorder."""

    def __init__(self, max_batch=16, delay=0.0, fail=False):
        self.max_batch = max_batch
        self._sessions = {'recognition': object()}
        self._cv = threading.Condition()
        self._busy = False
        self._queue = []
        self._stat_lock = threading.Lock()
        self.reset_stats()
        self.batches = []
        self.delay, self.fail = delay, fail

    def _process(self, batch):
        time.sleep(self.delay)
        self.batches.append([len(j.faces) for j in batch])
        if self.fail:
            raise RuntimeError('boom')
        for j in batch:
            for f in j.faces:
                f.embedding = np.ones(4, np.float32)


def _faces(n):
    return [Face(bbox=np.zeros(4, np.float32), kps=np.zeros((5, 2), np.float32), det_score=1.0)
            for _ in range(n)]


class TestCoalescing:
    def test_single_caller_fills_its_faces(self):
        eng = _FakeEngine()
        faces = _faces(3)
        eng.run(None, faces, ('recognition',))
        assert all(f.embedding is not None for f in faces) and eng.batches == [[3]]

    def test_concurrent_callers_share_batches_and_all_complete(self):
        eng = _FakeEngine(delay=0.05)
        sets = [_faces(2) for _ in range(8)]
        ts = [threading.Thread(target=eng.run, args=(None, s, ('recognition',))) for s in sets]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        assert all(not t.is_alive() for t in ts)
        assert all(f.embedding is not None for s in sets for f in s)
        assert sum(sum(b) for b in eng.batches) == 16
        assert len(eng.batches) < 8, 'eight waiting callers never shared an inference'

    def test_a_batch_is_capped_at_max_batch(self):
        eng = _FakeEngine(max_batch=4, delay=0.05)
        sets = [_faces(3) for _ in range(6)]
        ts = [threading.Thread(target=eng.run, args=(None, s, ('recognition',))) for s in sets]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        assert all(sum(b) <= 4 for b in eng.batches)
        assert all(f.embedding is not None for s in sets for f in s)

    def test_an_error_reaches_every_caller_of_that_batch(self):
        eng = _FakeEngine(delay=0.05, fail=True)
        errors = []

        def call():
            try:
                eng.run(None, _faces(1), ('recognition',))
            except RuntimeError as exc:
                errors.append(exc)
        ts = [threading.Thread(target=call) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        assert len(errors) == 4 and eng.stats['errors'] >= 1

    def test_a_missing_session_raises_instead_of_doing_nothing(self):
        eng = _FakeEngine()
        faces = _faces(1)
        with pytest.raises(RuntimeError, match='no session'):
            eng.run(None, faces, ('landmark_3d_68',))
        assert getattr(faces[0], 'embedding', None) is None

    def test_a_closed_engine_raises(self):
        eng = _FakeEngine()
        eng.close()
        with pytest.raises(RuntimeError):
            eng.run(None, _faces(1))


class _Model:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def get(self, frame, face):
        self.log.append((self.name, id(face)))
        face[self.name] = 1


class TestSeam:
    def _fa(self, log):
        return types.SimpleNamespace(models={'detection': _Model('detection', log),
                                             'recognition': _Model('recognition', log),
                                             'landmark_2d_106': _Model('landmark_2d_106', log)})

    def test_outside_a_scope_it_is_the_original_per_face_loop(self):
        log = []
        faces = _faces(2)
        fu._apply_aux(None, faces, self._fa(log))
        assert [n for n, _ in log] == ['recognition', 'landmark_2d_106'] * 2
        assert not fu.aux_batch_active()

    def test_a_failing_engine_falls_back_and_is_counted(self, monkeypatch, capsys):
        class Broken:
            def run(self, *_a, **_k):
                raise RuntimeError('engine down')
        monkeypatch.setitem(fu._AUX_SCOPE, 'on', True)
        monkeypatch.setitem(fu._AUX_SCOPE, 'engine', Broken())
        monkeypatch.setattr(fu, '_AUX_WARNED', set())
        log = []
        faces = _faces(2)
        fu._apply_aux(None, faces, self._fa(log))
        assert all(f.get('recognition') == 1 and f.get('landmark_2d_106') == 1 for f in faces)
        assert 'aux batch FAILED' in capsys.readouterr().out

    def test_a_working_engine_handles_its_tasks_and_the_loop_the_rest(self, monkeypatch):
        seen = []

        class Ok:
            def run(self, frame, faces, tasks):
                seen.append(tasks)
        monkeypatch.setitem(fu._AUX_SCOPE, 'on', True)
        monkeypatch.setitem(fu._AUX_SCOPE, 'engine', Ok())
        log = []
        fu._apply_aux(None, _faces(1), self._fa(log))
        assert seen == [('recognition', 'landmark_2d_106')]
        assert log == [], 'the per-face loop re-ran models the engine had handled'

    def test_swallow_keeps_the_callers_failure_tolerance(self):
        class Bad(_Model):
            def get(self, frame, face):
                raise ValueError('bad crop')
        fa = types.SimpleNamespace(models={'landmark_3d_68': Bad('landmark_3d_68', [])})
        fu._apply_aux(None, _faces(2), fa, swallow='test')       # must not raise
        with pytest.raises(ValueError):
            fu._apply_aux(None, _faces(1), fa)

    def test_the_default_is_off(self, monkeypatch):
        """Measured slower (docs/perf/aux_batch_2026-10-05.md): it must not be on unasked."""
        monkeypatch.delenv('ROOP_AUX_BATCH', raising=False)
        assert fu.aux_batch_wanted() is False
        with fu.aux_batch_scope() as info:
            assert not fu.aux_batch_active()
        assert info['engine'] is None

    def test_scope_off_is_inert(self, monkeypatch):
        monkeypatch.setenv('ROOP_AUX_BATCH', '0')
        with fu.aux_batch_scope() as info:
            assert not fu.aux_batch_active()
        assert info['engine'] is None

    def test_scope_on_without_a_real_analyser_builds_nothing(self, monkeypatch):
        monkeypatch.setenv('ROOP_AUX_BATCH', '1')
        with fu.aux_batch_scope() as info:
            assert fu.aux_batch_active()
        assert info['engine'] is None
        assert not fu.aux_batch_active()
