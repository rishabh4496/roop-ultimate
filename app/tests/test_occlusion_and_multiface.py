"""Offline interaction regressions: real Kalman, Hungarian and Farneback paths."""
import numpy as np
import cv2
import pytest

from roop.multi_face_tracker import MultiFaceTracker
from roop.occlusion_masker import OcclusionMasker
from roop.face_z_order import depth_order, subtract_foreground


def face(x, identity, score=.95, size=40):
    return dict(bbox=np.array([x, 10, x + size, 10 + size], np.float32),
                embedding=np.eye(2, dtype=np.float32)[identity], det_score=score)


def test_crossing_and_low_confidence_keep_ids_and_bindings():
    tracker = MultiFaceTracker(max_age=3)
    expected = None
    for frame in range(25):
        detections = [face(4 * frame, 0, .3 if 9 <= frame <= 15 else .95),
                      face(96 - 4 * frame, 1)]
        if frame % 2:
            detections.reverse()
        result = tracker.update(detections, frame)
        ids = {int(np.argmax(f['embedding'])): f['_track_id'] for f in result}
        if expected is None:
            expected = ids
            for person, tid in ids.items():
                tracker.bind(tid, f'person_{person}.fsz')
        assert ids == expected
        for person, tid in ids.items():
            assert tracker.bindings[tid] == f'person_{person}.fsz'
    with pytest.raises(ValueError):
        tracker.bind(expected[0], 'wrong.fsz')
    for frame in range(25, 30):
        tracker.update([], frame)
    assert not tracker.bindings


def test_low_detections_cannot_spawn_and_stale_frames_do_not_rewind():
    tracker = MultiFaceTracker()
    assert tracker.update([face(0, 0, .3)], 0) == []
    f = tracker.update([face(0, 0)], 1)[0]
    state = tracker.tracks[f['_track_id']].state.copy()
    assert tracker.update([face(90, 1)], 0) == []
    np.testing.assert_array_equal(tracker.tracks[f['_track_id']].state, state)


@pytest.mark.parametrize('colour', [(25, 25, 25), (75, 125, 185)])
def test_moving_object_mask_and_feature_protection(colour):
    masker = OcclusionMasker()
    clean = np.full((96, 96, 3), (100, 150, 200), np.uint8)
    labels = np.ones((96, 96), np.uint8)
    labels[28:34, 22:35] = 4
    clean[28:34, 22:35] = 10  # natural eye remains protected
    silhouette = np.ones((96, 96), np.float32)
    for i, x in enumerate(range(8, 57, 4)):
        target = clean.copy()
        target[45:85, x:x+15] = colour
        m = masker.compute(target, clean, silhouette, labels=labels,
                           track_id=7, frame_index=i)
        assert m[50:80, x+2:x+13].mean() > .6
        assert m[29:33, 23:34].max() == 0
        assert m[10:20, 65:85].mean() < .05
    assert masker.flow_calls == i


def test_semantic_exclusion_and_preserve_glasses():
    crop = np.full((64, 64, 3), 150, np.uint8)
    labels = np.ones((64, 64), np.uint8)
    labels[10:15] = 17
    labels[25:30] = 6
    masker = OcclusionMasker()
    mask = masker.compute(crop, crop, np.ones((64, 64)), labels=labels)
    assert mask[12, 32] == mask[27, 32] == 1
    mask = masker.compute(crop, crop, np.ones((64, 64)), labels=labels,
                          preserve_glasses=False)
    assert mask[27, 32] == 0


def test_z_order_subtracts_dilated_foreground_semantics():
    faces = [face(10, 0, size=35), face(25, 1, size=50)]
    faces[0]['_track_id'], faces[1]['_track_id'] = 4, 2
    assert depth_order(faces, (100, 100, 3)) == [0, 1]
    masks = [np.zeros((100, 100), np.float32) for _ in faces]
    masks[0][10:45, 10:45] = 1
    masks[1][10:60, 25:75] = 1
    result = subtract_foreground(faces, masks, (100, 100, 3))
    assert result[0][20, 30] == 0
    assert result[0][20, 23] == 0  # dilation
    assert result[0][20, 15] == 1
    assert result[1][20, 30] == 1
    assert depth_order(list(reversed(faces)), (100, 100, 3)) == [1, 0]


def test_flow_dropouts_cuts_and_unordered_workers():
    masker = OcclusionMasker()
    clean = np.full((64, 64, 3), (100, 150, 200), np.uint8)
    target = clean.copy()
    target[20:60, 20:35] = 10
    support = np.ones((64, 64), np.float32)
    labels = np.ones((64, 64), np.uint8)
    masker.compute(target, clean, support, labels=labels, track_id=1, frame_index=0)
    # The swap accidentally reproduces the object for a frame, removing the
    # residual signal. Flow must carry nonzero occlusion through that dropout.
    m = masker.compute(target, target, support, labels=labels, track_id=1, frame_index=1)
    assert .2 < m[30:50, 23:32].mean() < .5
    m = masker.compute(target, target, support, labels=labels, track_id=1, frame_index=5)
    assert m.max() == 0  # no propagation across missing frames
    clone = masker.clone_for_block()
    assert not clone._state
    masker.set_ordered(False)
    masker.compute(target, clean, support, labels=labels, track_id=2, frame_index=6)
    assert 2 not in masker._state


def test_render_prepass_crossing_keeps_two_archived_tracks(monkeypatch):
    # Stub only the detector; execute the real ordered pre-pass association,
    # track archive, captured-reference assignment and immutable source map.
    from tests.test_track_reid import _run_scan, _Face, _Mgr
    original_init = _Mgr.__init__
    def separate_references(self, captured):
        original_init(self, captured)
        self.target_face_groups = [0, 1]
    monkeypatch.setattr(_Mgr, '__init__', separate_references)
    script = []
    for i in range(25):
        row = [_Face(np.eye(2)[0], (100+4*i, 100), size=40),
               _Face(np.eye(2)[1], (196-4*i, 100), size=40)]
        for person, f in enumerate(row):
            f.det_score = .3 if person == 0 and 9 <= i <= 15 else .95
        script.append(row[::-1] if i % 2 else row)
    mgr, tracks = _run_scan(script, captured=tuple(np.eye(2)))
    assert len(tracks) == 2
    for t in tracks:
        assert len(t['obs']) == 25
        assert {int(np.argmax(f.embedding)) for f in t['obs'].values()} == {int(np.argmax(t['emb_mean']))}
        assert mgr._track_source_map[t['id']][0] == int(np.argmax(t['emb_mean']))
    with pytest.raises(TypeError):
        mgr._track_source_map[0] = (99, None)


def test_production_compositor_passes_semantic_z_exclusion_to_paste():
    from types import SimpleNamespace
    from roop.ProcessMgr import ProcessMgr
    from roop.face_parser import ParseResult
    mgr = ProcessMgr.__new__(ProcessMgr)
    faces = [SimpleNamespace(**face(10, 0, size=35)), SimpleNamespace(**face(25, 1, size=50))]
    faces[0]._track_id, faces[1]._track_id = 4, 2
    plate = np.zeros((100, 100, 3), np.uint8)
    mgr._semantic_parse = lambda crop: ParseResult(np.ones(crop.shape[:2], np.uint8), 'bisenet')
    painted = []
    def paste(src, f, frame, plate=None, region=None):
        mask = np.ones(frame.shape[:2], np.float32)
        if region is not None:
            region.trim_frame(mask)
        painted.append((src, mask))
        return frame
    mgr.process_face = paste
    mgr._composite_faces([(1, faces[1]), (0, faces[0])], plate, plate.copy())
    assert [src for src, _ in painted] == [0, 1]
    assert painted[0][1][20, 30] == 0
    assert painted[1][1][20, 30] == 1


def test_real_paste_preserves_object_pixels(monkeypatch):
    from tests.test_matte_roi import Stub
    from roop.face_parser import ParseResult
    import roop.globals
    monkeypatch.setenv('ROOP_GRAIN_MATCH', '0')
    monkeypatch.setattr(roop.globals, 'skin_lab_match', 0., raising=False)
    clean = np.full((96, 96, 3), (100, 150, 200), np.uint8)
    target = clean.copy()
    target[45:85, 40:55] = 15
    swap = np.full_like(clean, (120, 170, 220))
    masker = OcclusionMasker()
    stub = Stub()
    stub._temporal_engine = lambda name: masker
    result = stub.paste_upscale(
        swap, swap, np.array([[1., 0, 0], [0, 1., 0]]), target.copy(), 1,
        [0, 0, 0, 0, 0., 0.], target_crop=target,
        semantic_parse=ParseResult(np.ones((96, 96), np.uint8), 'bisenet'),
        track_id=3, frame_index=1)
    np.testing.assert_array_equal(result[50:75, 43:52], target[50:75, 43:52])
    assert np.mean(np.abs(result[40:55, 65:75].astype(float) - target[40:55, 65:75])) > 5
    assert 3 in masker._state  # the actual paste path consumed this stage


def test_tracking_threshold_reaches_detector_without_changing_settings(monkeypatch):
    from roop import face_util
    import roop.globals
    seen = []
    before = getattr(roop.globals, 'face_detector_threshold', None)
    monkeypatch.setattr(face_util, '_detect_faces_raw',
                        lambda frame, **kwargs: seen.append(kwargs) or [])
    # No detection/ML work is mocked above this raw detector boundary.
    face_util.get_all_faces(np.zeros((64, 64, 3), np.uint8), rescue=False, det_thresh=.20)
    assert seen == [{'det_thresh': .20}]
    assert getattr(roop.globals, 'face_detector_threshold', None) == before


def test_frozen_bindings_win_over_misleading_partial_frame_embeddings():
    from types import MappingProxyType, SimpleNamespace
    from roop.multi_face_tracker import locked_source
    from roop.selected_routing import compute_selected_assignment
    # Both crops now resemble the OTHER person, as happens at a crossing.
    faces = [face(30, 1, .3), face(35, 0, .3)]
    for i, f in enumerate(faces):
        f['_track_id'] = i
    bindings = MappingProxyType({0: (0, np.eye(2)[0]), 1: (1, np.eye(2)[1])})
    # The identity-lock branch reads this same function before its vetoes.
    assert [locked_source(bindings, f, {0, 1}) for f in faces] == [0, 1]
    assert locked_source(bindings, faces[0], {1}) is None
    result = compute_selected_assignment(
        faces, [SimpleNamespace(embedding=e) for e in np.eye(2)], [0, 1], {0, 1}, 0, 2, .75,
        identity_match=lambda refs, f: (0, 1-float(refs[0].embedding @ f['embedding'])),
        unreliable=lambda f: True,
        track_binding=lambda f: locked_source(bindings, f, {0, 1}),
        locked_bindings=True)
    assert result.pending == [(0, 0), (1, 1)]
    assert all(reason.startswith('swapped') for reason in result.reasons.values())


def test_stage_cache_preserves_authoritative_bindings(tmp_path):
    from types import MappingProxyType, SimpleNamespace
    from tests.test_stage_cache import make_pm
    from roop.stage_cache import write_stage2, restore_stage2
    from roop.stage2_plan import build_stage2_plan
    pm = make_pm(n_frames=4)
    pm._track_source_map = MappingProxyType(pm._track_source_map)
    path = str(tmp_path / 'tracks.npz')
    write_stage2(path, pm, frame_start=0, frame_count=4, ranges_abs=None,
                 plan=pm.stage2_frame_plan)
    restored = SimpleNamespace()
    restore_stage2(path, restored, new_start=0, new_count=4,
                   current_ranges_abs=None, build_plan=build_stage2_plan)
    assert isinstance(restored._track_source_map, MappingProxyType)
    with pytest.raises(TypeError):
        restored._track_source_map[1] = (99, None)


def test_prepass_reid_cannot_steal_a_track_already_matched_this_frame():
    from tests.test_track_reid import _run_scan, _Face, TARGET, NEAR, FAR, _at_distance
    stranger = _at_distance(TARGET, .65)
    script = [[_Face(TARGET, NEAR)] for _ in range(20)]
    script += [[_Face(stranger, FAR), _Face(TARGET, NEAR)] for _ in range(3)]
    _, tracks = _run_scan(script)
    assert len(tracks) == 2
    assert sorted(len(t['obs']) for t in tracks) == [3, 23]


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
