import cv2
import numpy as np
from contextlib import nullcontext
from importlib import import_module
from queue import Queue
from types import SimpleNamespace

from roop.multi_face_tracker import MultiFaceTracker
from roop.trajectory_cache import TrajectoryPlan
from roop.typing import Face
from roop.video_temporal import SourceSlots, VideoTemporalState


POINTS = np.array([[30, 28], [60, 28], [45, 46], [33, 67], [57, 67]], np.float32)
FRAME = np.full((120, 160, 3), 130, np.uint8)


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def face(x=0, embedding=(1., 0.), noise=0.):
    points = POINTS + (x, 0) + noise
    return Face(bbox=np.array([15 + x, 13, 75 + x, 85], np.float32),
                kps=points, embedding=np.array(embedding, np.float32), det_score=.95)


def test_geometry_reduces_jitter_and_follows_a_fast_step():
    state = VideoTemporalState()
    history = []
    jitter = np.array([[.4, -.3], [-.4, .3], [.25, -.25], [0., .2], [0., -.2]])
    for i in range(20):
        state.begin(FRAME, i)
        target = face(noise=jitter * (-1) ** i)
        state.geometry(target, 1)
        history.append(target.kps)
    assert np.abs(np.diff(history, axis=0)).mean() < .12
    state.begin(FRAME, 20)
    target = face(x=18)
    state.geometry(target, 1)
    assert np.max(np.abs(target.kps[:, 0] - POINTS[:, 0] - 18)) < .7
    assert np.max(np.abs(target.bbox - face(x=18).bbox)) < .7


def test_identity_and_source_slots_survive_crossing_and_a_miss():
    tracker = MultiFaceTracker(max_coast=3, min_hits_to_coast=2)
    slots = SourceSlots()
    for i, x in enumerate((0, 7, 15, 22, 30, 37)):
        detections = [face(x, (1, 0)), face(28 - x, (0, 1))]
        detections.sort(key=lambda f: f.bbox[0])
        tracked = tracker.update(detections, i)
        slots.update(tracked, i, 2)
        assert all(f['_temporal_source_slot'] == np.argmax(f.embedding) for f in tracked)
    remaining = tracker.update([face(-16, (0, 1))], 6)
    predictions = tracker.coast(6, frame_shape=FRAME.shape, occupied=remaining)
    assert len(predictions) == 1 and predictions[0]['_coast_age'] == 1
    slots.update(remaining + predictions, 6, 2)
    assert remaining[0]['_temporal_source_slot'] == 1
    assert predictions[0]['_temporal_source_slot'] == 0


def test_cut_clears_every_history_on_boundary_even_in_short_shots():
    state = VideoTemporalState()
    state.begin(FRAME, 0)
    state.geometry(face(), 1)
    state.patches[1] = object()
    state.selected_sources[1] = 0
    state.last_swapped_frame = FRAME
    assert state.begin(np.zeros_like(FRAME), 1)
    assert not state.tracks and not state.patches and not state.selected_sources
    assert state.last_swapped_frame is None
    assert state.begin(FRAME, 2)
    target = face(x=30)
    state.geometry(target, 1)
    np.testing.assert_array_equal(target.kps, face(x=30).kps)


def test_canonical_alpha_and_colour_reject_new_occlusion_and_lighting_change():
    state = VideoTemporalState()
    alpha = np.full(FRAME.shape[:2], .6, np.float32)
    state.begin(FRAME, 0)
    state.mask(1, alpha, FRAME)
    stats = tuple(np.full(3, value, np.float32) for value in (100, 20, 125, 25))
    state.color(1, 'lab', stats)
    state.begin(FRAME, 1)
    next_alpha = alpha + .05
    next_alpha[30:50, 35:50] = 0
    mask = state.mask(1, next_alpha, FRAME)
    assert .6 < mask[60, 60] < .64
    assert not mask[30:50, 35:50].any()
    next_stats = tuple(v + 2 for v in stats)
    smoothed = state.color(1, 'lab', next_stats)
    assert (smoothed[2] < next_stats[2]).all()
    state.begin(FRAME, 2)
    exposed = tuple(v + 30 for v in stats)
    np.testing.assert_array_equal(state.color(1, 'lab', exposed), exposed)


def test_texture_keeps_target_expression_and_damps_static_shimmer():
    state = VideoTemporalState()
    y, x = np.indices(FRAME.shape[:2])
    pattern = ((x + y) % 2 * 20 - 10).astype(np.int16)[:, :, None]
    a = (FRAME + pattern).astype(np.uint8)
    b = (FRAME - pattern).astype(np.uint8)
    guide = FRAME.copy()
    guide[62:88, 53:86] = 35
    b[62:88, 53:86] = 45
    state.begin(FRAME, 0)
    state.texture(1, 'restorer', a, FRAME, None)
    state.begin(FRAME, 1)
    result = state.texture(1, 'restorer', b, guide, None)
    assert cv2.absdiff(result[:50], a[:50]).mean() < .3 * cv2.absdiff(b[:50], a[:50]).mean()
    np.testing.assert_array_equal(result[62:88, 53:86], b[62:88, 53:86])


def test_overlap_suffix_agrees_with_contiguous_matte_history():
    full = VideoTemporalState()
    split = VideoTemporalState()
    for i in range(33):
        alpha = np.full(FRAME.shape[:2], .55 + .04 * (-1) ** i, np.float32)
        full.begin(FRAME, i)
        expected = full.mask(1, alpha, FRAME)
        if i >= 18:
            split.begin(FRAME, i - 30)
            actual = split.mask(1, alpha, FRAME)
        if i >= 30:
            np.testing.assert_allclose(expected, actual, atol=.001)


def test_video_first_face_dispatch_coasts_and_resets_before_detection(monkeypatch):
    module = import_module('roop.ProcessMgr')
    mgr = module.ProcessMgr(progress=None)
    mgr.options = SimpleNamespace(swap_mode='first', selected_index=0)
    mgr.input_face_datas = [SimpleNamespace(faces=[face()])]
    mgr._video_temporal_enabled = True
    mgr._reset_dispatch_tracker()
    detections = iter([[face()], [face(1)], [], []])
    monkeypatch.setattr(module, 'get_all_faces', lambda _frame: next(detections))
    monkeypatch.setattr(module, '_gpu_guard', lambda **kwargs: nullcontext())
    rescues = []
    landing = []
    monkeypatch.setattr(module, '_detect_face_in_roi', lambda _frame, box: rescues.append(box))

    def paint(_slot, target, output, **kwargs):
        landing.append(target)
        return output

    monkeypatch.setattr(mgr, 'process_face', paint)
    for i in range(3):
        assert mgr.swap_faces(FRAME, stabilize=True, frame_idx=i)[0] == 1
    assert [f.get('_coast_age', 0) for f in landing] == [0, 0, 1]
    assert len(rescues) == 1
    assert mgr.swap_faces(np.zeros_like(FRAME), stabilize=True, frame_idx=3)[0] == 0
    assert len(rescues) == 1
    assert not mgr._dispatch_face_tracker.tracks

    monkeypatch.setattr(module, 'get_first_face', lambda _frame: face())
    assert mgr.swap_faces(FRAME)[0] == 1
    assert mgr._video_temporal_state() is None


def test_segment_reader_keeps_preroll_indices_outside_output_range(monkeypatch):
    module = import_module('roop.ProcessMgr')
    mgr = module.ProcessMgr(progress=None)
    mgr._segment_warmup_count = 3
    mgr.frames_queue = [Queue()]
    monkeypatch.setattr(module.roop.globals, 'processing', True)
    monkeypatch.setattr(module, 'wait_while_paused', lambda: None)
    positions = []
    frames = iter(range(17, 23))
    cap = SimpleNamespace(set=lambda _prop, i: positions.append(i), read=lambda: (True, next(frames)))
    mgr.read_frames_thread(cap, frame_start=20, frame_end=23, num_threads=1)
    assert positions == [17]
    assert list(mgr.frames_queue[0].queue) == [(i, 20 + i) for i in range(-3, 3)] + [None]


def test_trajectory_seals_filtered_video_landmarks_without_filtering_again():
    frames = {i: [face(x)] for i, x in enumerate([0, 0, 32, 32])}
    for faces in frames.values():
        faces[0]['_track_id'] = 4
    plan = TrajectoryPlan.build(frames, fps=30., smooth_landmarks=False)
    for i, faces in frames.items():
        np.testing.assert_array_equal(plan.at(i)[0]['kps'], faces[0].kps)
