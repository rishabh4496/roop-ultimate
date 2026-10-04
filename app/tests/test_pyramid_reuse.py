"""MultiScaleFaceDetector.detect: reuse the single pass, and where the extra scales run.

  1. When the adaptive single pass triggers the pyramid, its result is the pyramid's
     scale-1.0 level and is reused: 3 detector inferences instead of 4, and the merged
     boxes/keypoints are EXACTLY what the old four-inference algorithm produced.
  2. The scale passes run in the calling thread when it is a pool worker (it holds a
     pooled FaceAnalysis lease), else on one persistent module-level executor -- not on
     a ThreadPoolExecutor built per call.
  3. should_trigger_pyramid's thresholds are untouched.

The reference below is the OLD algorithm rebuilt from the module's own public helpers
(pad -> single pass -> pyramid of every level -> rescale -> DIoU-NMS -> unpad), so the
exact-equality assertions do not depend on the code under test.
"""

import contextlib
import os
import queue
import sys
import threading

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals  # noqa: E402
from roop import face_detector as fd  # noqa: E402
from roop import baseline_probe as bp  # noqa: E402

H, W = 1080, 1920


def frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


class Detector:
    """A fake detect_fn whose boxes depend on the image it is handed, so every pyramid
    level returns something different and an order mistake changes the merge."""

    def __init__(self, big_face=True, kps=True):
        self.calls = []                     # (image id, (h, w), thread ident)
        self.big_face, self.kps = big_face, kps
        self.lock = threading.Lock()

    def __call__(self, img, ds, dt):
        with self.lock:
            self.calls.append((id(img), img.shape[:2], threading.get_ident()))
        h, w = img.shape[:2]
        fh = (0.8 if self.big_face else 0.2) * h
        x0, y0 = 0.30 * w, 0.10 * h + (h % 7)
        boxes = np.array([[x0, y0, x0 + 0.5 * fh, y0 + fh, 0.97 - (h % 5) * 0.01],
                          [0.05 * w + (w % 11), 0.6 * h, 0.05 * w + 60.0 + (w % 11), 0.6 * h + 70.0, 0.62]],
                         dtype=np.float32)
        kpss = (np.stack([np.full((5, 2), 1.0, np.float32) * (h % 13),
                          np.full((5, 2), 2.0, np.float32) * (w % 17)]) if self.kps else None)
        return boxes, kpss


def reference(fr, fn, scales=None, det_size=640, thresh=0.5, trigger_from_single=True):
    """The pre-change algorithm: single pass, then the pyramid with EVERY level run."""
    scales = scales or list(fd.DEFAULT_PYRAMID_SCALES)
    padded, offs = fd.apply_context_padding(fr, min_padding=fd.MIN_BORDER_PADDING_PX, mode='reflect')
    b, k = fn(padded, det_size, thresh)
    ub, uk = fd.remove_context_padding(b, k, offs)
    assert fd.should_trigger_pyramid(fr.shape[:2], initial_dets=ub), "fixture must trigger"
    cands, kcs = [], []
    for sc, img in fd.generate_scale_pyramid(padded, scales):
        bb, kk = fn(img, det_size, thresh)
        bo, ko = fd.rescale_detections(bb, kk, scale_factor=sc)
        cands.append(bo)
        kcs.append(ko if ko is not None and len(ko) == len(bo) else np.zeros((len(bo), 5, 2), np.float32))
    merged, mk = np.vstack(cands), (np.vstack(kcs) if kcs else None)
    nms = getattr(roop.globals, 'face_detector_nms', 0.40)        # the threshold detect() reads
    kept, kk, _ = fd.diou_nms(merged, kpss=mk, iou_thresh=nms, offset=1.0)
    return fd.remove_context_padding(kept, kk, offs)


@pytest.fixture(autouse=True)
def _clean():
    bp.set_enabled(True)
    bp.reset()
    fd.shutdown_scale_executor()
    yield
    fd.shutdown_scale_executor()
    bp.set_enabled(None)
    bp.reset()


def run(det, **kw):
    d = fd.MultiScaleFaceDetector(detect_fn=det)
    return d.detect(frame(), det_size=640, det_thresh=0.5, max_workers=kw.pop('max_workers', 2), **kw)


# ── 1. reuse ─────────────────────────────────────────────────────────────────

def test_triggered_pyramid_reuses_the_single_pass_and_matches_the_old_merge_exactly():
    new_det, ref_det = Detector(), Detector()
    got_b, got_k = run(new_det)
    ref_b, ref_k = reference(frame(), ref_det)
    assert np.array_equal(got_b, ref_b)
    assert np.array_equal(got_k, ref_k)
    assert len(ref_det.calls) == 4              # single pass + three levels
    assert len(new_det.calls) == 3              # single pass + the two levels it does not already have
    # the 1.0 level was NOT run again: the padded image is handed to the detector once
    pad = fd.apply_context_padding(frame())[1][0]
    padded_ids = [c[0] for c in new_det.calls if c[1] == (H + 2 * pad, W + 2 * pad)]
    assert len(padded_ids) == 1
    assert bp.total('pyramid.single_pass_reused') == 1
    assert bp.total('pyramid.detect_fn_calls') == 3


def test_the_reused_level_keeps_its_slot_in_the_merge():
    """Ties in DIoU-NMS are broken by candidate order, so level 1.0 must stay last."""
    new_det = Detector()
    got_b, _ = run(new_det)
    seq = [c[1] for c in new_det.calls]
    assert seq[0][0] > seq[1][0] and seq[0][0] > seq[2][0]       # padded single pass, then the smaller levels
    ref_b, _ = reference(frame(), Detector())
    assert np.array_equal(got_b, ref_b)


def test_a_single_pass_without_kps_is_reused_and_merged_like_before():
    got_b, got_k = run(Detector(kps=False))
    ref_b, ref_k = reference(frame(), Detector(kps=False))
    assert np.array_equal(got_b, ref_b)
    assert np.array_equal(got_k, ref_k)


def test_the_kept_single_pass_arrays_are_not_modified():
    seen = {}

    class Spy(Detector):
        def __call__(self, img, ds, dt):
            out = super().__call__(img, ds, dt)
            if 'first' not in seen:
                seen['first'] = (out[0].copy(), out[0])
            return out

    run(Spy())
    before, live = seen['first']
    assert np.array_equal(before, live)


def test_no_reuse_when_the_pyramid_has_no_scale_1():
    det = fd.MultiScaleFaceDetector(detect_fn=Detector(), default_scales=[0.5, 0.75])
    d = det.detect_fn
    det.detect(frame(), det_size=640, det_thresh=0.5, max_workers=2)
    assert len(d.calls) == 3                    # single pass + 2 levels, nothing to reuse
    assert bp.total('pyramid.single_pass_reused') == 0


def test_configured_scales_have_no_single_pass_and_run_every_level():
    det = Detector()
    fd.MultiScaleFaceDetector(detect_fn=det).detect(
        frame(), det_size=640, det_thresh=0.5, scales='0.5,1.0', max_workers=2)
    assert len(det.calls) == 2
    assert bp.total('pyramid.single_pass_reused') == 0


def test_estimated_height_trigger_has_no_single_pass_to_reuse():
    det = Detector(big_face=False)
    fd.MultiScaleFaceDetector(detect_fn=det).detect(
        frame(), det_size=640, det_thresh=0.5, estimated_face_height=600.0, max_workers=2)
    assert len(det.calls) == 3
    assert bp.total('pyramid.single_pass_reused') == 0


def test_a_non_triggering_frame_is_still_one_inference():
    det = Detector(big_face=False)
    run(det)
    assert len(det.calls) == 1


# ── 2. where the passes run ──────────────────────────────────────────────────

def test_a_pool_worker_runs_its_scale_passes_in_its_own_thread():
    det = Detector()
    out = {}

    def worker():
        fd.pool_worker_enter()
        try:
            out['res'] = run(det)
        finally:
            fd.pool_worker_exit()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert {c[2] for c in det.calls} == {t.ident}               # nothing ran on another thread
    assert bp.total('pyramid.mode.sequential_in_pool_worker') == 1
    assert bp.total('pyramid.mode.shared_executor') == 0
    assert fd._SCALE_EXECUTOR is None                           # and no executor was even built
    ref_b, _ = reference(frame(), Detector())
    assert np.array_equal(out['res'][0], ref_b)


def test_a_plain_caller_uses_one_persistent_executor_across_calls():
    det = Detector()
    base_threads = threading.active_count()
    run(det)
    first = fd._SCALE_EXECUTOR
    assert first is not None
    for _ in range(11):
        run(det)
    assert fd._SCALE_EXECUTOR is first                          # same object, not rebuilt per call
    # a ThreadPoolExecutor per call would leave 12 x 2 threads behind or churn them;
    # a persistent one is bounded by its width however many calls are made
    assert threading.active_count() - base_threads <= fd._SCALE_EXECUTOR_WIDTH
    assert bp.total('pyramid.mode.shared_executor') == 12
    # the scale passes really did leave the calling thread
    assert any(c[2] != threading.get_ident() for c in det.calls)


def test_the_executor_is_widened_never_shrunk():
    ex2 = fd._scale_executor(2)
    assert fd._scale_executor(1) is ex2
    ex4 = fd._scale_executor(4)
    assert ex4 is not ex2 and fd._SCALE_EXECUTOR_WIDTH == 4


def test_one_remaining_level_runs_sequentially_whatever_the_caller():
    det = fd.MultiScaleFaceDetector(detect_fn=Detector(), default_scales=[0.5, 1.0])
    det.detect(frame(), det_size=640, det_thresh=0.5, max_workers=2)
    assert len(det.detect_fn.calls) == 2                        # single pass + the 0.5 level
    assert bp.total('pyramid.mode.sequential') == 1
    assert fd._SCALE_EXECUTOR is None


def test_pool_worker_flag_is_per_thread_and_nests():
    assert not fd.in_pool_worker()
    fd.pool_worker_enter()
    fd.pool_worker_enter()
    seen = []
    t = threading.Thread(target=lambda: seen.append(fd.in_pool_worker()))
    t.start()
    t.join()
    assert fd.in_pool_worker() and seen == [False]
    fd.pool_worker_exit()
    assert fd.in_pool_worker()
    fd.pool_worker_exit()
    assert not fd.in_pool_worker()
    fd.pool_worker_exit()                                       # underflow stays at zero
    assert not fd.in_pool_worker()


# ── the lease marks exactly the threads that hold a pooled instance ─────────

def test_lease_face_analyser_marks_the_thread_only_while_a_pooled_instance_is_held(monkeypatch):
    fu = pytest.importorskip('roop.face_util')
    q = queue.Queue()
    q.put(object())
    monkeypatch.setattr(fu, '_ensure_face_analyser', lambda: None)
    monkeypatch.setattr(fu, 'analysis_pooled', lambda: True)
    monkeypatch.setattr(fu, '_ANALYSER_Q', q)
    assert not fd.in_pool_worker()
    with fu.lease_face_analyser():
        assert fd.in_pool_worker()
    assert not fd.in_pool_worker()
    with pytest.raises(RuntimeError):
        with fu.lease_face_analyser():
            raise RuntimeError('boom')
    assert not fd.in_pool_worker()                              # cleared on the error path too


def test_a_single_unpooled_analyser_does_not_mark_the_thread(monkeypatch):
    fu = pytest.importorskip('roop.face_util')
    monkeypatch.setattr(fu, '_ensure_face_analyser', lambda: None)
    monkeypatch.setattr(fu, 'analysis_pooled', lambda: False)
    monkeypatch.setattr(fu, 'FACE_ANALYSER', object())
    with fu.lease_face_analyser():
        assert not fd.in_pool_worker()


# ── 3. thresholds ────────────────────────────────────────────────────────────

def test_trigger_thresholds_are_unchanged():
    assert fd.CLOSEUP_HEIGHT_THRESHOLD == 500
    assert fd.CLOSEUP_COVERAGE_RATIO == 0.75
    assert fd.DEFAULT_PYRAMID_SCALES == [0.5, 0.75, 1.0]
    shape = (1080, 1920)
    assert not fd.should_trigger_pyramid(shape, initial_dets=np.zeros((0, 5)))
    assert fd.should_trigger_pyramid(shape, initial_dets=np.array([[0, 0, 100, 500, 0.9]]))
    assert not fd.should_trigger_pyramid(shape, initial_dets=np.array([[0, 0, 100, 499, 0.9]]))
    assert fd.should_trigger_pyramid(shape, initial_dets=np.array([[0, 0, 100, 810, 0.9]]))      # 0.75 * 1080
    assert fd.should_trigger_pyramid(shape, estimated_face_height=500)
    assert not fd.should_trigger_pyramid(shape, estimated_face_height=499)


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
