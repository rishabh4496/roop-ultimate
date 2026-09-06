"""The dispatch tracker is order-dependent, and it was shared by every worker.

`FaceTracker.update` predicts every track forward by `dt`, increments `missed`
on all of them, and clears it only on the tracks IT matched. `coast` then
invents a face for every track whose `missed` is above zero. All of that is
correct for ONE thread walking a clip in order and wrong for N threads sitting
on N different frames of it, which is what both production dispatch paths do:

  * parallel stabilization  -- worker k owns a contiguous block, so the N
                               workers are hundreds of frames apart;
  * round-robin             -- worker i gets frames i, i+N, i+2N.

`test_shared_tracker_invents_faces_on_empty_frames` drives one shared tracker
the way interleaved block workers do and asserts the symptom that was reported:
a swapped face pasted onto frames where the detector found nobody. The rest
assert the fix's contract -- each block and each round-robin worker gets its own
instance, and coasting is refused where frames are not consecutive.

These are mechanism tests. They say the tracker is driven correctly; they do not
say a rendered clip improved.
"""

import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.tracker import FaceTracker

FRAME_SHAPE = (400, 800, 3)


def _face(cx, cy=100.0, width=40.0, height=52.0):
    return {
        'bbox': np.asarray((cx - width / 2, cy - height / 2,
                            cx + width / 2, cy + height / 2), dtype=np.float32),
        'embedding': np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        'kps': np.asarray([[cx - 8, cy - 8], [cx + 8, cy - 8], [cx, cy],
                           [cx - 6, cy + 10], [cx + 6, cy + 10]], dtype=np.float32),
        'det_score': np.float32(0.9),
    }


# Four workers, 40-frame blocks, one chunk. The person is in blocks 0 and 3 and
# absent from blocks 1 and 2 -- the ordinary case of somebody leaving frame.
WORKERS, BLOCK, STEPS = 4, 40, 40


def _person_present(frame_index):
    return frame_index < BLOCK or frame_index >= 3 * BLOCK


def _run_interleaved(tracker_for):
    """Advance the workers one frame each per step, as threads do, and count
    the frames that got a swapped face while the detector found nobody."""
    phantoms = 0
    for step in range(STEPS):
        for worker in range(WORKERS):
            index = worker * BLOCK + step
            detections = ([_face(100.0 + 4.0 * (index % 60))]
                          if _person_present(index) else [])
            tracker = tracker_for(worker)
            faces = tracker.update(detections, index)
            coasted = tracker.coast(index, frame_shape=FRAME_SHAPE, occupied=faces)
            if coasted and not detections:
                phantoms += 1
    return phantoms


class SharedTrackerIsUnsafe(unittest.TestCase):
    def test_shared_tracker_invents_faces_on_empty_frames(self):
        shared = FaceTracker(max_age=30)
        phantoms = _run_interleaved(lambda _worker: shared)
        self.assertGreater(
            phantoms, 20,
            'the shared-tracker defect no longer reproduces; if the tracker '
            'gained its own ordering guard, this test should assert that '
            'instead of being deleted')
        # It also never terminates: `coasted_run` is reset by any other
        # worker's real match, so MAX_COAST_FRAMES stops bounding the run.
        self.assertEqual(shared.stats['coast_expired'], 0)

    def test_per_block_trackers_invent_nothing(self):
        trackers = [FaceTracker(max_age=30) for _ in range(WORKERS)]
        self.assertEqual(_run_interleaved(lambda worker: trackers[worker]), 0)


class CloneForBlock(unittest.TestCase):
    def test_clone_carries_configuration_and_no_state(self):
        source = FaceTracker(max_age=21, max_cost=0.71, process_noise=1.5,
                             measurement_noise=3.0, max_coast=7,
                             min_hits_to_coast=2, history=9, max_outside=0.25)
        for index in range(6):
            source.update([_face(100.0 + index)], index)
        clone = source.clone_for_block()
        self.assertIsNot(clone, source)
        for field in ('max_age', 'max_cost', 'process_noise', 'measurement_noise',
                      'max_coast', 'min_hits_to_coast', 'history', 'max_outside'):
            self.assertEqual(getattr(clone, field), getattr(source, field), field)
        self.assertEqual(clone.tracks, {})
        self.assertIsNone(clone._last_frame_index)
        # And the source keeps its own state.
        self.assertTrue(source.tracks)


class _Mgr:
    """The three attributes `_dispatch_tracker` reads, and nothing else."""

    from roop.ProcessMgr import ProcessMgr
    _dispatch_tracker = ProcessMgr._dispatch_tracker
    _reset_dispatch_tracker = ProcessMgr._reset_dispatch_tracker

    def __init__(self, ordered=True):
        self._tls = threading.local()
        self._dispatch_epoch = 0
        self._reset_dispatch_tracker()
        # After the reset: it deliberately restores ordered dispatch, and the
        # dispatch loop is what declares otherwise.
        self._dispatch_ordered = ordered


class DispatchTrackerSelection(unittest.TestCase):
    def test_sequential_run_uses_the_shared_instance(self):
        mgr = _Mgr(ordered=True)
        tracker, may_coast = mgr._dispatch_tracker()
        self.assertIs(tracker, mgr._dispatch_face_tracker)
        self.assertTrue(may_coast)

    def test_a_block_gets_its_own_instance_and_may_coast(self):
        mgr = _Mgr(ordered=True)
        mgr._tls.temporal_block = True
        tracker, may_coast = mgr._dispatch_tracker()
        self.assertIsNot(tracker, mgr._dispatch_face_tracker)
        self.assertTrue(may_coast)
        # Stable within the block.
        self.assertIs(mgr._dispatch_tracker()[0], tracker)

    def test_round_robin_gets_its_own_instance_and_may_not_coast(self):
        mgr = _Mgr(ordered=False)
        tracker, may_coast = mgr._dispatch_tracker()
        self.assertIsNot(tracker, mgr._dispatch_face_tracker)
        self.assertFalse(may_coast)

    def test_two_round_robin_workers_never_share(self):
        mgr = _Mgr(ordered=False)
        seen = {}

        def worker(name):
            seen[name] = mgr._dispatch_tracker()[0]

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(seen), 4)
        self.assertEqual(len({id(t) for t in seen.values()}), 4)
        self.assertNotIn(id(mgr._dispatch_face_tracker),
                         {id(t) for t in seen.values()})

    def test_a_new_run_invalidates_a_worker_thread_s_clone(self):
        """Worker threads outlive a run on the pooled paths; their thread-local
        clone must not survive into the next one."""
        mgr = _Mgr(ordered=False)
        first = mgr._dispatch_tracker()[0]
        mgr._reset_dispatch_tracker()
        self.assertIsNot(mgr._dispatch_tracker()[0], first)

    def test_reset_restores_ordered_dispatch(self):
        mgr = _Mgr(ordered=False)
        mgr._reset_dispatch_tracker()
        self.assertTrue(mgr._dispatch_ordered)


class CallSitesGoThroughTheAccessor(unittest.TestCase):
    """A source-level guard, because the defect is invisible at runtime.

    Reaching for `self._dispatch_face_tracker` inside a worker still returns a
    working tracker and still produces faces; only the positions are wrong, and
    only under concurrency. Nothing raises, the swap audit reads 100%, and the
    render exits 0. So the contract is asserted where it can be seen: the
    per-frame path may only reach the tracker through `_dispatch_tracker`.
    """

    @staticmethod
    def _source_of(name):
        import inspect

        from roop.ProcessMgr import ProcessMgr
        return inspect.getsource(getattr(ProcessMgr, name))

    def test_swap_faces_does_not_touch_the_shared_tracker(self):
        source = self._source_of('swap_faces')
        self.assertIn('self._dispatch_tracker()', source)
        self.assertNotIn('self._dispatch_face_tracker', source)

    def test_a_parallel_block_clones_the_tracker(self):
        source = self._source_of('_run_stab_parallel')
        self.assertIn('self._tls.dispatch_tracker', source)
        self.assertIn('clone_for_block()', source)

    def test_swap_faces_reads_the_roi_cache_per_worker(self):
        source = self._source_of('swap_faces')
        self.assertIn("self._tls.last_found_bboxes", source)
        self.assertNotIn('self.last_found_bboxes', source)


if __name__ == '__main__':
    unittest.main()
