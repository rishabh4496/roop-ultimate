"""SAM 2 gated to the frames the earlier stages kept (Phase 7).

Two layers, both without the SAM 2 checkpoint (it is not downloaded here):
  * `roop/sam2_runs.py`: which frames, which seeds, where the memory bank must reset - pure functions
  * `Mask_SAM2.precompute_runs` / `ProcessMgr._precompute_sam2_runs` against a FAKE predictor that records every
    call, so the claims "only these frames are decoded", "a fresh bank per run", "seeded at the run's first frame
    with the plan's own boxes" and "masks land at absolute indices" are asserted on the real orchestration code
CPU only, no torch needed (the predictor is a stand-in; `torch` is imported by `_propagate`, so the test skips
where it is absent).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import sam2_runs  # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def rec(box):
    return SimpleNamespace(bbox=np.array(box, np.float32))


# ── which frames ─────────────────────────────────────────────────────────────────────────────────────────────
def test_consecutive_plan_frames_are_one_run_and_a_gap_over_one_starts_another():
    plan = {f: [rec([1, 2, 3, 4])] for f in (5, 6, 7, 9, 10, 20)}
    runs = sam2_runs.runs_from_plan(plan, 100)
    assert [(a, b) for a, b, _ in runs] == [(5, 7), (9, 10), (20, 20)]      # 7 -> 9 is a gap of 2: a_{i+1} - b_i > 1


def test_a_single_missing_frame_is_a_boundary():
    plan = {f: [rec([0, 0, 1, 1])] for f in (0, 1, 3, 4)}
    assert [(a, b) for a, b, _ in sam2_runs.runs_from_plan(plan, 10)] == [(0, 1), (3, 4)]


def test_each_run_is_seeded_with_the_plans_own_boxes_at_its_first_frame():
    plan = {10: [rec([1, 1, 5, 5]), rec([20, 20, 30, 30])], 11: [rec([2, 2, 6, 6])], 40: [rec([9, 9, 19, 19])]}
    runs = sam2_runs.runs_from_plan(plan, 100)
    assert len(runs[0][2]) == 2 and np.array_equal(runs[0][2][1], [20, 20, 30, 30])    # frame 10, not frame 11
    assert len(runs[1][2]) == 1 and np.array_equal(runs[1][2][0], [9, 9, 19, 19])
    assert all(b.dtype == np.float32 for _, _, bs in runs for b in bs)


def test_seeds_are_copies_not_views_of_the_plan():
    r = rec([1, 2, 3, 4])
    runs = sam2_runs.runs_from_plan({0: [r]}, 5)
    runs[0][2][0][0] = 99
    assert r.bbox[0] == 1


def test_frames_outside_the_clip_and_empty_records_are_dropped():
    plan = {-3: [rec([0, 0, 1, 1])], 2: [rec([0, 0, 1, 1])], 3: [], 50: [rec([0, 0, 1, 1])]}
    assert [(a, b) for a, b, _ in sam2_runs.runs_from_plan(plan, 10)] == [(2, 2)]


def test_stage1_ranges_become_unseeded_runs_clipped_to_the_clip():
    runs = sam2_runs.runs_from_ranges([(0, 4), (8, 30), (50, 60)], 12)
    assert runs == [(0, 4, None), (8, 11, None)]


def test_choose_prefers_the_plan_then_the_ranges_then_the_whole_clip():
    plan = {3: [rec([0, 0, 1, 1])], 4: [rec([0, 0, 1, 1])]}
    assert sam2_runs.choose(plan, [(0, 9)], 20)[1] == 'plan'
    assert sam2_runs.choose({}, [(0, 9)], 20)[1] == 'ranges'
    assert sam2_runs.choose(None, None, 20) == ([(0, 19, None)], 'clip')
    assert sam2_runs.choose({}, [], 20)[1] == 'clip'


def test_an_all_empty_plan_falls_through_to_the_ranges():
    assert sam2_runs.choose({1: []}, [(2, 5)], 10)[1] == 'ranges'


def test_coverage():
    runs = [(0, 9, None), (20, 29, None)]
    assert sam2_runs.coverage(runs, 100) == (20, 20.0)


# ── the orchestration, against a fake predictor ──────────────────────────────────────────────────────────────
class FakePredictor:
    image_size = 8

    def __init__(self):
        self.calls = []
        self.live_states = 0

    def init_state(self, video_path=None, offload_video_to_cpu=True, offload_state_to_cpu=True):
        self.live_states += 1
        self.calls.append(('init', None))
        return {'n': 0}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box):
        self.calls.append(('box', (frame_idx, obj_id, tuple(np.asarray(box).tolist()))))
        state['n'] = max(state['n'], 1)

    def propagate_in_video(self, state):
        import torch
        for i in range(state['len']):
            yield i, [0], torch.ones((1, 1, 4, 4))

    def reset_state(self, state):
        self.live_states -= 1
        self.calls.append(('reset', None))


@pytest.fixture
def sam2():
    torch = pytest.importorskip('torch')                                # noqa: F841
    from roop.processors import Mask_SAM2 as mod
    m = mod.Mask_SAM2()
    m.predictor = FakePredictor()
    m.device = 'cpu'
    # the frame buffer is a real preprocessing step (resize + normalise); replace it with a counter so the test
    # stays about WHICH frames are handed over, and let the fake propagate exactly what was buffered
    class Buf:
        def __init__(self, capacity=0):
            self.count = 0

        def add(self, fr):
            self.count += 1
    m.new_frame_buffer = lambda capacity=0: Buf(capacity)
    orig_init = m.predictor.init_state

    def init(video_path=None, **kw):
        state = orig_init(video_path=video_path, **kw)
        state['len'] = m._current_len
        return state

    m.predictor.init_state = init
    m._current_len = 0
    mod_in_memory = mod._in_memory_frames

    class Loader:
        def __init__(self, buf):
            m._current_len = buf.count

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    mod._in_memory_frames = Loader
    mod.SAM2FrameBuffer = Buf
    yield m
    mod._in_memory_frames = mod_in_memory


def test_only_the_runs_are_decoded_and_each_gets_a_fresh_bank(sam2):
    decoded = []

    def read_run(first, last):
        for f in range(first, last + 1):
            decoded.append(f)
            yield np.zeros((4, 4, 3), np.uint8)

    runs = [(5, 7, [np.array([1, 2, 3, 4], np.float32)]), (20, 21, [np.array([5, 6, 7, 8], np.float32),
                                                                   np.array([9, 9, 9, 9], np.float32)])]
    stats = sam2.precompute_runs(runs, read_run, (4, 4))
    assert decoded == [5, 6, 7, 20, 21]                                  # nothing else was read
    assert stats == {'runs': 2, 'frames': 5, 'resets': 2}
    kinds = [c[0] for c in sam2.predictor.calls]
    assert kinds == ['init', 'box', 'reset', 'init', 'box', 'box', 'reset']     # init/reset around EVERY run
    assert sam2.predictor.live_states == 0                               # no tracker state outlives its run
    assert sorted(sam2.precomputed) == [5, 6, 7, 20, 21]                 # absolute indices, never 0..n


def test_the_first_box_of_each_run_goes_in_at_the_runs_own_first_frame(sam2):
    runs = [(30, 31, [np.array([1, 1, 2, 2], np.float32)])]
    sam2.precompute_runs(runs, lambda a, b: (np.zeros((2, 2, 3), np.uint8) for _ in range(b - a + 1)), (2, 2))
    assert sam2.predictor.calls[1] == ('box', (0, 0, (1.0, 1.0, 2.0, 2.0)))      # frame_idx 0 OF THE RUN'S BUFFER


def test_a_run_whose_frames_cannot_be_read_is_skipped_without_a_bank(sam2):
    stats = sam2.precompute_runs([(5, 7, [np.zeros(4, np.float32)])], lambda a, b: iter(()), (4, 4))
    assert stats['runs'] == 0 and sam2.predictor.calls == [] and sam2.precomputed == {}


def test_a_run_without_seeds_is_skipped(sam2):
    stats = sam2.precompute_runs([(0, 3, [])], lambda a, b: iter(()), (4, 4))
    assert stats['runs'] == 0 and sam2.predictor.calls == []


def test_the_whole_clip_path_is_unchanged_and_also_releases_its_state(sam2):
    from roop.processors import Mask_SAM2 as mod
    buf = mod.SAM2FrameBuffer()
    for _ in range(3):
        buf.add(None)
    sam2.precompute(buf, [np.array([1, 2, 3, 4], np.float32)], (4, 4))
    assert sorted(sam2.precomputed) == [0, 1, 2]
    assert [c[0] for c in sam2.predictor.calls] == ['init', 'box', 'reset']


def test_get_crop_mask_knows_nothing_about_frames_outside_the_runs(sam2):
    sam2.precompute_runs([(5, 6, [np.array([0, 0, 1, 1], np.float32)])],
                         lambda a, b: (np.zeros((4, 4, 3), np.uint8) for _ in range(b - a + 1)), (4, 4))
    M = np.eye(2, 3, dtype=np.float32)
    outside = sam2.get_crop_mask(100, M, (4, 4))
    assert outside.shape == (4, 4) and float(outside.max()) == 0.0       # "swap the whole crop": no extra masking
    assert 5 in sam2.precomputed and 100 not in sam2.precomputed


# ── the wiring: where SAM 2 is called from (sites named, so a refactor cannot silently restore the full-clip pass) ──
def _src(rel):
    with open(os.path.join(APP, rel), encoding='utf-8') as fh:
        return fh.read()


def test_batch_runs_sam2_after_stage2_in_temporal_mode_and_before_it_otherwise():
    s = _src('roop/procmgr_batch.py')
    assert s.count('self._run_sam2_prepass(') == 3                       # early (non-temporal), after Stage 2, fallback
    assert 'self._precompute_sam2(' not in s                             # nothing calls the whole-clip pass directly
    assert 'not is_awebp and not self._temporal_mode' in s
    after_stage2 = s.index('_stage_cache.save_stage2(self')
    assert s.index('self._run_sam2_prepass(', after_stage2) > after_stage2
    assert s.index('self._run_sam2_prepass(', after_stage2) < s.index('_release_replayed_analysis', after_stage2)


def test_tracking_gates_through_runs_and_falls_back_to_the_whole_clip():
    s = _src('roop/procmgr_tracking.py')
    assert 'sam2_runs.choose(' in s and 'runs=None if source == \'clip\' else runs' in s
    assert 'if runs:\n            return self._precompute_sam2_runs(' in s.replace('\r\n', '\n')
