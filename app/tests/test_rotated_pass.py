"""The rotated rescues run the aux models only on faces they keep.

`_rescue_rotated` and `_detect_faces`'s partial-miss rescue used to detect on a rotated
frame WITH aux (recognition, 106- and 68-point landmarks) and discard the aux work of
every duplicate. `face_util._rotated_pass` detects without aux, tests duplicates on an
un-rotated COPY of the coordinates, runs the aux models on the ROTATED frame from the
ROTATED-space keypoints for the survivors only, and then un-rotates them.

What is pinned here:
  * the survivors are the same faces, with the same coordinates and the same aux
    results, as the verbatim old loop (reference implementation below), AND duplicates
    got no aux call;
  * the aux models see the rotated frame and the rotated keypoints;
  * a failing aux step drops the pass, as a failing old `_detect_faces_raw` did;
  * SCRFD's `unclamped=True` keeps `fa.get()`'s geometry, the default keeps clamping;
  * `_rescue_upscaled` still runs on retinaface_r50: a skip was proposed, measured
    (it gained 13 faces on 13 of 175 empty frames, mostly occluded kiss faces), and
    rejected -- see the note at the call site.
"""

import contextlib
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

fu = pytest.importorskip('roop.face_util')
from roop import baseline_probe as bp  # noqa: E402

W, H = 200, 120
FRAME = np.zeros((H, W, 3), dtype=np.uint8)
MARK = {'clockwise': 1, 'anticlockwise': 2, '180': 3}


class Face(dict):
    """Attribute-style, like insightface's Face."""
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def rotated_marker_frame(angle):
    shape = (W, H, 3) if angle in ('clockwise', 'anticlockwise') else (H, W, 3)
    return np.full(shape, MARK[angle], dtype=np.uint8)


def mkface(x0, y0, x1, y1):
    kps = np.array([[x0 + 0.3 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                    [x0 + 0.7 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                    [x0 + 0.5 * (x1 - x0), y0 + 0.55 * (y1 - y0)],
                    [x0 + 0.35 * (x1 - x0), y0 + 0.8 * (y1 - y0)],
                    [x0 + 0.65 * (x1 - x0), y0 + 0.8 * (y1 - y0)]], dtype=np.float32)
    return Face(bbox=np.array([x0, y0, x1, y1], dtype=np.float32), kps=kps, det_score=0.9)


class AuxModel:
    """Embedding = f(marker of the frame it saw, the keypoints it was given)."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def get(self, img, face):
        self.log.append((self.name, int(img[0, 0, 0]), face.kps.copy()))
        face['emb_' + self.name] = np.concatenate([[float(img[0, 0, 0])], face.kps.ravel()])
        return None


@pytest.fixture
def world(monkeypatch):
    """A fake detector: what each rotated frame 'contains', in ROTATED space."""
    import roop.globals as g
    log = []
    detections = {}                      # angle -> list of (x0,y0,x1,y1) in rotated space
    state = types.SimpleNamespace(log=log, detections=detections, fail_aux=False, raw_calls=[])
    models = {'detection': types.SimpleNamespace(), 'recognition': AuxModel('recognition', log),
              'landmark_2d_106': AuxModel('lm106', log)}

    def rot(angle):
        return lambda frame: rotated_marker_frame(angle)

    monkeypatch.setattr(fu, 'rotate_clockwise', rot('clockwise'))
    monkeypatch.setattr(fu, 'rotate_anticlockwise', rot('anticlockwise'))
    monkeypatch.setattr(fu, 'rotate_image_180', rot('180'))
    monkeypatch.setattr(g, 'detector_engine', 'scrfd', raising=False)

    def raw(frame, det_size=None, det_thresh=None, aux=True, unclamped=False):
        state.raw_calls.append({'aux': aux, 'unclamped': unclamped})
        angle = {v: k for k, v in MARK.items()}[int(frame[0, 0, 0])]
        faces = [mkface(*b) for b in detections.get(angle, [])]
        if aux:                          # what fa.get() did: aux on every face
            for f in faces:
                for name, m in models.items():
                    if name != 'detection':
                        m.get(frame, f)
        return faces

    @contextlib.contextmanager
    def lease():
        if state.fail_aux:
            class Boom:
                def get(self, img, face):
                    raise RuntimeError('aux failed')
            yield types.SimpleNamespace(models={'detection': None, 'recognition': Boom()})
        else:
            yield types.SimpleNamespace(models=models)

    monkeypatch.setattr(fu, '_detect_faces_raw', raw)
    monkeypatch.setattr(fu, 'lease_face_analyser', lease)
    return state


def legacy_pass(frame, angle, rotate, known, w, h):
    """The pre-change loop, verbatim in behaviour: detect WITH aux, un-rotate the real
    face, keep it unless it duplicates something already kept."""
    kept = []
    for f in fu._detect_faces_raw(rotate(frame)) or []:
        fu._unrotate_face_coords(f, w, h, angle)
        if not fu._is_face_duplicate(f, list(known) + kept):
            kept.append(f)
    return kept


def same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x.bbox, y.bbox)
        assert np.array_equal(x.kps, y.kps)
        assert np.array_equal(x['emb_recognition'], y['emb_recognition'])
        assert np.array_equal(x['emb_lm106'], y['emb_lm106'])


def unrot_box(angle, box):
    """Where a rotated-space box lands upright (via the module's own un-rotation)."""
    g = fu._Geometry(mkface(*box))
    fu._unrotate_face_coords(g, W, H, angle)
    return g.bbox


def test_survivors_and_their_aux_results_match_the_old_loop_and_duplicates_get_none(world):
    angle = '180'
    # In ROTATED space: A, a near-copy of A (a duplicate of it), and a distinct B.
    A, A2, B = (20, 20, 70, 80), (22, 21, 72, 81), (120, 30, 170, 95)
    world.detections[angle] = [A, A2, B]
    # The first pass already found a face where B lands once un-rotated, so B is a
    # duplicate of it, exactly as A2 is a duplicate of A.
    first_pass = [mkface(*unrot_box(angle, (118, 28, 168, 93)))]

    world.log.clear()
    old = legacy_pass(FRAME, angle, fu.rotate_image_180, first_pass, W, H)
    old_aux = len(world.log)
    world.log.clear()
    new = fu._rotated_pass(FRAME, angle, fu.rotate_image_180, first_pass, W, H)
    new_aux = len(world.log)

    same(old, new)
    assert len(new) == 1                                   # A2 duplicates A, B duplicates the known face
    # old: 3 faces x 2 aux models; new: 1 face x 2 aux models
    assert (old_aux, new_aux) == (6, 2)
    assert world.raw_calls[-1] == {'aux': False, 'unclamped': True}


def test_aux_models_see_the_rotated_frame_and_the_rotated_keypoints(world):
    world.detections['clockwise'] = [(30, 10, 90, 70)]
    out = fu._rotated_pass(FRAME, 'clockwise', fu.rotate_clockwise, [], W, H)
    assert len(out) == 1
    rotated_kps = mkface(30, 10, 90, 70).kps
    for name, marker, kps in world.log:
        assert marker == MARK['clockwise']                  # the rotated frame, not the plate
        assert np.array_equal(kps, rotated_kps)             # rotated-space keypoints at call time
    # ...and the face handed back is upright: its kps are no longer the rotated ones
    assert not np.array_equal(out[0].kps, rotated_kps)


def test_later_candidates_are_judged_against_earlier_survivors_of_the_same_pass(world):
    # three overlapping detections of ONE face: only the first survives, as before
    world.detections['anticlockwise'] = [(40, 20, 100, 80), (41, 21, 101, 81), (39, 19, 99, 79)]
    old = legacy_pass(FRAME, 'anticlockwise', fu.rotate_anticlockwise, [], W, H)
    new = fu._rotated_pass(FRAME, 'anticlockwise', fu.rotate_anticlockwise, [], W, H)
    same(old, new)
    assert len(new) == 1


def test_a_failing_aux_step_drops_the_whole_pass(world):
    world.detections['180'] = [(20, 20, 70, 80)]
    world.fail_aux = True
    with pytest.raises(RuntimeError):
        fu._rotated_pass(FRAME, '180', fu.rotate_image_180, [], W, H)


def test_hybrid_engine_drops_a_face_without_a_five_point_fit_before_the_duplicate_test(world, monkeypatch):
    import roop.globals as g
    monkeypatch.setattr(g, 'detector_engine', 'retinaface_r50', raising=False)
    world.detections['180'] = [(20, 20, 70, 80)]
    real = fu._detect_faces_raw

    def no_kps(frame, **kw):
        faces = real(frame, **kw)
        for f in faces:
            f.kps = None
        return faces

    monkeypatch.setattr(fu, '_detect_faces_raw', no_kps)
    assert fu._rotated_pass(FRAME, '180', fu.rotate_image_180, [], W, H) == []


def test_rescue_rotated_counts_what_it_skipped(world):
    world.detections['clockwise'] = [(30, 10, 90, 70), (31, 11, 91, 71)]
    bp.set_enabled(True)
    bp.reset()
    try:
        out = fu._rescue_rotated(FRAME, expected_count=None)
        assert len(out) == 1
        assert bp.total('rescue.rotated_pass.duplicates_aux_skipped') == 1
        assert bp.total('rescue.rotated_pass.survivors') == 1
    finally:
        bp.set_enabled(None)
        bp.reset()


def test_rescue_rotated_stops_at_expected_count_like_before(world):
    world.detections['clockwise'] = [(30, 10, 90, 70)]
    world.detections['anticlockwise'] = [(110, 10, 170, 70)]
    world.detections['180'] = [(60, 40, 120, 100)]
    out = fu._rescue_rotated(FRAME, expected_count=2)
    assert len(out) == 2
    assert [c['aux'] for c in world.raw_calls] == [False, False]      # 180 never tried


# ── SCRFD geometry ───────────────────────────────────────────────────────────

def _scrfd_fa(bboxes, kpss):
    det = types.SimpleNamespace(input_size=(512, 512), det_thresh=0.5,
                                detect=lambda frame, max_num=0, metric='default': (bboxes, kpss))
    return types.SimpleNamespace(models={}, det_model=det)


def test_unclamped_matches_insightface_geometry_and_default_still_clamps(monkeypatch):
    import roop.globals as g
    pytest.importorskip('insightface')
    monkeypatch.setattr(g, 'detector_engine', 'scrfd', raising=False)
    bboxes = np.array([[-12.0, -5.0, 90.0, 140.0, 0.9]], dtype=np.float32)       # off the 200x120 canvas
    kpss = np.array([[[-3.0, 20.0], [60.0, 22.0], [30.0, 50.0], [5.0, 90.0], [70.0, 95.0]]], dtype=np.float32)
    fa = _scrfd_fa(bboxes, kpss)

    @contextlib.contextmanager
    def lease():
        yield fa

    monkeypatch.setattr(fu, 'lease_face_analyser', lease)
    raw = fu._detect_faces_raw(FRAME, aux=False, unclamped=True)
    assert np.array_equal(raw[0].bbox, bboxes[0, 0:4])
    assert np.array_equal(raw[0].kps, kpss[0])
    assert float(raw[0].det_score) == pytest.approx(0.9)
    clamped = fu._detect_faces_raw(FRAME, aux=False)
    assert clamped[0].bbox[0] == 0.0 and clamped[0].bbox[1] == 0.0       # the existing behaviour, untouched
    assert clamped[0].bbox[3] == float(H)


# ── source pins ──────────────────────────────────────────────────────────────

def _fn_src(name):
    import ast
    src = open(os.path.join(APP, 'roop', 'face_util.py'), encoding='utf-8').read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(name)


def test_both_rescues_go_through_the_shared_pass_and_never_detect_with_aux_on_a_turn():
    for name in ('_rescue_rotated', '_detect_faces'):
        body = _fn_src(name)
        assert '_rotated_pass(' in body, name
        assert '_detect_faces_raw(rot(frame))' not in body, name
        assert '_detect_faces_raw(r_frame)' not in body, name


def test_upscale_rescue_still_runs_on_retinaface_r50(monkeypatch):
    """Decision record: skipping it on r50 was measured and rejected (it gained a
    face on 13 of 175 empty baseline frames). If this test is changed, re-run
    tests/probe_r50_upscale_gain.py first."""
    import roop.globals as g
    calls = []
    monkeypatch.setattr(g, 'detector_engine', 'retinaface_r50', raising=False)
    monkeypatch.setattr(g, 'rescue_small_faces', True, raising=False)
    monkeypatch.setattr(g, 'TARGET_FACE_GROUP', [], raising=False)
    monkeypatch.setattr(g, 'INPUT_FACESETS', [], raising=False)
    monkeypatch.setattr(fu, '_detect_faces_raw', lambda *a, **k: [])
    monkeypatch.setattr(fu, '_enrich_detected_faces', lambda frame, faces: faces)
    monkeypatch.setattr(fu, '_rescue_upscaled', lambda frame: calls.append('up') or [mkface(10, 10, 60, 60)])
    out = fu._detect_faces(FRAME)
    assert calls == ['up'] and len(out) == 1


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
