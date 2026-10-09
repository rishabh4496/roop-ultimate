"""StabRawCache: the warm-up dedup's bookkeeping, and the eligibility rules that keep it off when it could be wrong.

The cache's job is to make two neighbouring stabilization blocks compute an overlap frame ONCE. What can go wrong is
quiet: a hit served for the wrong crop, an entry that outlives its second visit, a producer that fails and strands a
waiter, a waiter that deadlocks the pool. Each is pinned here without a GPU. Whether a render is bit-identical with
the cache on is tests/ab_stab_dedup.py's question, not this file's.
"""
import os
import sys
import threading
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import roop.globals                                                  # noqa: E402
from roop.stab_dedup import StabRawCache                             # noqa: E402


def entry(tag):
    return {'fake_frame': np.full((4, 4, 3), tag, np.uint8), 'enhanced_frame': np.full((8, 8, 3), tag, np.uint8),
            'scale_factor': 2, 'img_mask': np.full((8, 8), tag / 255.0, np.float32), 'swap_model_mask': None}


class ZoneTests(unittest.TestCase):
    def test_only_the_last_warmup_frames_of_a_block_with_a_successor_are_cached(self):
        c = StabRawCache(block=24, warmup=6, n_frames=96)
        zone = [g for g in range(96) if c.in_zone(g)]
        # blocks start at 0,24,48,72; a block's tail is read by the NEXT block's warm-up, the last block has none
        self.assertEqual(zone, list(range(18, 24)) + list(range(42, 48)) + list(range(66, 72)))

    def test_warmup_longer_than_a_block_disables_the_cache(self):
        c = StabRawCache(block=12, warmup=13, n_frames=200)
        self.assertFalse(any(c.in_zone(g) for g in range(200)))

    def test_no_warmup_no_zone(self):
        self.assertFalse(any(StabRawCache(24, 0, 100).in_zone(g) for g in range(100)))

    def test_warmup_equal_to_block_covers_the_whole_previous_block(self):
        c = StabRawCache(block=6, warmup=6, n_frames=24)
        self.assertEqual([g for g in range(24) if c.in_zone(g)], list(range(18)))

    def test_out_of_range_and_none(self):
        c = StabRawCache(24, 6, 48)
        self.assertFalse(c.in_zone(None))
        self.assertFalse(c.in_zone(-1))
        self.assertFalse(c.in_zone(48))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        roop.globals.processing = True
        self.c = StabRawCache(24, 6, 96)
        self.key = (20, 0, 0, 3, b'M')

    def test_first_visit_produces_second_is_served_then_it_is_gone(self):
        state, slot = self.c.acquire(self.key)
        self.assertEqual(state, 'produce')
        self.c.publish(slot, entry(7))
        state, got = self.c.acquire(self.key)
        self.assertEqual(state, 'hit')
        self.assertEqual(int(got['fake_frame'][0, 0, 0]), 7)
        self.assertEqual(self.c._live_bytes, 0)                    # freed on the second visit
        self.assertEqual(self.c.acquire(self.key)[0], 'produce')   # a third visit starts over

    def test_published_arrays_are_copies_the_producer_cannot_mutate_afterwards(self):
        state, slot = self.c.acquire(self.key)
        e = entry(9)
        self.c.publish(slot, e)
        e['img_mask'][:] = 0.0              # the producer's own filters write into its arrays
        e['fake_frame'][:] = 0
        _, got = self.c.acquire(self.key)
        self.assertEqual(int(got['fake_frame'][0, 0, 0]), 9)
        self.assertAlmostEqual(float(got['img_mask'][0, 0]), 9 / 255.0, places=6)

    def test_a_different_matrix_never_hits(self):
        _, slot = self.c.acquire((20, 0, 0, 3, b'M1'))
        self.c.publish(slot, entry(1))
        self.assertEqual(self.c.acquire((20, 0, 0, 3, b'M2'))[0], 'produce')

    def test_a_waiter_gets_the_result_instead_of_recomputing(self):
        _, slot = self.c.acquire(self.key)
        out = {}

        def waiter():
            out['r'] = self.c.acquire(self.key)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.15)
        self.assertTrue(t.is_alive())                                # blocked on the in-flight producer
        self.c.publish(slot, entry(5))
        t.join(5)
        self.assertEqual(out['r'][0], 'hit')
        self.assertEqual(self.c.stats['hit_waited'], 1)

    def test_a_producer_that_fails_releases_its_waiter_to_compute_for_itself(self):
        done = threading.Event()
        holder = {}

        def producer():
            holder['slot'] = self.c.acquire(self.key)
            done.set()
            time.sleep(0.2)
            self.c.abandon()                                         # frame ended without publishing

        pt = threading.Thread(target=producer)
        pt.start()
        done.wait(2)
        self.assertEqual(holder['slot'][0], 'produce')
        state, _ = self.c.acquire(self.key)
        pt.join(5)
        self.assertEqual(state, 'bypass')
        self.assertEqual(self.c.stats['failed'], 1)
        self.assertEqual(self.c.acquire(self.key)[0], 'produce')    # and the key is usable again

    def test_abandon_after_publish_is_a_no_op(self):
        _, slot = self.c.acquire(self.key)
        self.c.publish(slot, entry(2))
        self.c.abandon()
        self.assertEqual(self.c.acquire(self.key)[0], 'hit')

    def test_cancel_unblocks_a_waiter(self):
        _, slot = self.c.acquire(self.key)
        out = {}
        t = threading.Thread(target=lambda: out.setdefault('r', self.c.acquire(self.key)))
        t.start()
        time.sleep(0.1)
        roop.globals.processing = False
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(out['r'][0], 'bypass')

    def test_drop_before_frees_stale_entries_but_keeps_the_carried_tail(self):
        for gi in (10, 20, 23):
            _, slot = self.c.acquire((gi, 0, 0, 0, b'M'))
            self.c.publish(slot, entry(gi))
        self.c.drop_before(18)
        self.assertEqual(self.c.acquire((10, 0, 0, 0, b'M'))[0], 'produce')    # dropped
        self.assertEqual(self.c.acquire((20, 0, 0, 0, b'M'))[0], 'hit')
        self.assertEqual(self.c.acquire((23, 0, 0, 0, b'M'))[0], 'hit')

    def test_over_the_byte_cap_a_producer_does_not_publish_and_the_second_visitor_computes_itself(self):
        one = sum(v.nbytes for v in entry(1).values() if isinstance(v, np.ndarray))
        c = StabRawCache(24, 6, 96, max_bytes=one + one // 2)      # room for one entry, not two
        _, s1 = c.acquire((18, 0, 0, 0, b'M'))
        c.publish(s1, entry(1))
        _, s2 = c.acquire((19, 0, 0, 0, b'M'))
        c.publish(s2, entry(2))                                    # over the cap
        self.assertEqual(c.stats['capped'], 1)
        self.assertEqual(c.acquire((18, 0, 0, 0, b'M'))[0], 'hit')
        self.assertEqual(c.acquire((19, 0, 0, 0, b'M'))[0], 'produce')   # not held: starts over, never wrong
        self.assertLessEqual(c.peak_bytes, c.max_bytes)

    def test_a_waiter_is_released_when_the_producer_is_capped(self):
        c = StabRawCache(24, 6, 96, max_bytes=1)
        _, slot = c.acquire(self.key)
        out = {}
        t = threading.Thread(target=lambda: out.setdefault('r', c.acquire(self.key)))
        t.start()
        time.sleep(0.1)
        c.publish(slot, entry(3))
        t.join(5)
        self.assertEqual(out['r'][0], 'bypass')

    def test_peak_bytes_tracks_what_is_held(self):
        for gi in (18, 19):
            _, slot = self.c.acquire((gi, 0, 0, 0, b'M'))
            self.c.publish(slot, entry(1))
        one = sum(v.nbytes for v in entry(1).values() if isinstance(v, np.ndarray))
        self.assertEqual(self.c.peak_bytes, 2 * one)
        self.c.clear()
        self.assertEqual(self.c._live_bytes, 0)

    def test_concurrent_producers_and_consumers_never_lose_or_duplicate_an_entry(self):
        n = 200
        served = [0]
        produced = [0]
        lock = threading.Lock()

        def worker(order):
            for gi in order:
                state, obj = self.c.acquire((gi, 0, 0, 0, b'M'))
                if state == 'produce':
                    time.sleep(0.0005)
                    self.c.publish(obj, entry(gi % 250))
                    with lock:
                        produced[0] += 1
                elif state == 'hit':
                    with lock:
                        served[0] += 1
                    assert int(obj['fake_frame'][0, 0, 0]) == gi % 250

        a = list(range(n))
        ts = [threading.Thread(target=worker, args=(a,)), threading.Thread(target=worker, args=(a[::-1],))]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertFalse(any(t.is_alive() for t in ts))
        # every frame is visited exactly twice (once per thread): one produced it, the other was served
        self.assertEqual(produced[0] + served[0], 2 * n)
        self.assertEqual(self.c._live_bytes, 0)


class EligibilityTests(unittest.TestCase):
    """_stab_dedup_plan refuses every configuration where the raw stage could depend on a block's state."""

    def mixin(self, **over):
        from roop.procmgr_stabilization import StabilizationSchedulingMixin

        class P:
            def __init__(self, type_, name):
                self.type, self.processorname = type_, name

        class PM(StabilizationSchedulingMixin):
            pass

        pm = PM()
        pm._temporal_mode = True
        pm._temporal_faces = {}
        pm.kps_stabilizer = None
        pm._kps_stab_factory = None
        pm._landmark_smoother = type('L', (), {'enabled': False})()
        pm.processors = [P('swap', 'faceswap'), P('enhance', 'restore_ultra'), P('mask', 'mask_xseg')]
        for k, v in over.items():
            setattr(pm, k, v)
        return pm

    def plan(self, pm, block=24, wu=6, n=600):
        os.environ.pop('ROOP_STAB_DEDUP', None)
        return pm._stab_dedup_plan(block, wu, n)

    def test_the_shipped_configuration_is_eligible(self):
        cache, why = self.plan(self.mixin())
        self.assertIsNotNone(cache, why)

    def test_kill_switch(self):
        os.environ['ROOP_STAB_DEDUP'] = '0'
        try:
            self.assertIsNone(self.mixin()._stab_dedup_plan(24, 6, 600)[0])
        finally:
            os.environ.pop('ROOP_STAB_DEDUP', None)

    def test_no_temporal_detection_means_a_live_kps_filter_so_no_cache(self):
        self.assertIsNone(self.plan(self.mixin(_temporal_mode=False))[0])
        self.assertIsNone(self.plan(self.mixin(kps_stabilizer=object()))[0])
        self.assertIsNone(self.plan(self.mixin(_kps_stab_factory=lambda: None))[0])
        self.assertIsNone(self.plan(self.mixin(_landmark_smoother=type('L', (), {'enabled': True})()))[0])

    def test_ordered_temporal_engines_disable_it(self):
        for name in ('temporal_identity', 'temporal_occlusion', 'target_appearance', 'temporal_compositing',
                     'temporal_quality'):
            pm = self.mixin()
            setattr(pm, '_' + name, type('E', (), {'enabled': True})())
            self.assertIsNone(self.plan(pm)[0], name)

    def test_any_number_of_maskers_is_fine_because_they_all_run_on_every_visit(self):
        # The live app appends mask_occluder after mask_xseg ("[Occlusion] occlusion masking on").
        from types import SimpleNamespace as NS
        mk = lambda *pairs: [NS(type=t, processorname=n) for t, n in pairs]
        two = mk(('swap', 'faceswap'), ('enhance', 'restore_ultra'), ('mask', 'mask_xseg'), ('mask', 'mask_occluder'))
        self.assertIsNotNone(self.plan(self.mixin(processors=two))[0])
        self.assertIsNotNone(self.plan(self.mixin(processors=mk(('swap', 'faceswap'), ('mask', 'mask_xseg3'))))[0])
        self.assertIsNotNone(self.plan(self.mixin(processors=mk(('swap', 'faceswap'))))[0])    # no mask at all

    def test_an_enabled_enhance_gate_disables_it(self):
        pm = self.mixin()
        pm._enhance_gate = type('G', (), {'enabled': True})()
        self.assertIsNone(self.plan(pm)[0])
        pm._enhance_gate = type('G', (), {'enabled': False})()
        self.assertIsNotNone(self.plan(pm)[0])

    def test_unverified_enhancers_and_odd_chains_disable_it(self):
        from types import SimpleNamespace as NS
        mk = lambda *pairs: [NS(type=t, processorname=n) for t, n in pairs]
        for name in ('adaptive_enhancer', 'gpen', 'codeformer', 'ultramax'):
            self.assertIsNone(self.plan(self.mixin(processors=mk(('swap', 'faceswap'), ('enhance', name),
                                                                 ('mask', 'mask_xseg'))))[0], name)
        self.assertIsNotNone(self.plan(self.mixin(processors=mk(('swap', 'faceswap'), ('enhance', 'restoreformer++'),
                                                                ('mask', 'mask_xseg'))))[0])
        # a masker ahead of the restorer would put mask state upstream of the shared stage
        self.assertIsNone(self.plan(self.mixin(processors=mk(('swap', 'faceswap'), ('mask', 'mask_xseg'),
                                                             ('enhance', 'restore_ultra'))))[0])
        self.assertIsNone(self.plan(self.mixin(processors=mk(('swap', 'faceswap'), ('enhance', 'restore_ultra'),
                                                             ('enhance', 'restore_ultra'), ('mask', 'mask_xseg'))))[0])
        self.assertIsNone(self.plan(self.mixin(processors=mk(('enhance', 'restore_ultra'), ('swap', 'faceswap'),
                                                             ('mask', 'mask_xseg'))))[0])

    def test_geometry_guards(self):
        self.assertIsNone(self.plan(self.mixin(), wu=0)[0])
        self.assertIsNone(self.plan(self.mixin(), block=12, wu=13)[0])


class SeamIsWiredTests(unittest.TestCase):
    """The pieces that must stay together in ProcessMgr, pinned by source so a refactor cannot drop one quietly."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(APP, 'roop', 'ProcessMgr.py'), encoding='utf-8') as fh:
            cls.src = fh.read()
        cls.face = cls.src.split('    def process_face(', 1)[1].split('\n    def ', 1)[0]
        cls.par = cls.src.split('    def _run_stab_parallel', 1)[1].split('    def update_progress', 1)[0]

    def test_every_frame_releases_an_unpublished_slot(self):
        self.assertGreaterEqual(self.par.count('_dedup.abandon()'), 2)       # warm-up loop and output loop

    def test_cache_is_built_trimmed_reported_and_freed(self):
        for needle in ('self._stab_dedup_plan(', 'drop_before(', 'summary_line()', 'self._stab_raw.clear()'):
            self.assertIn(needle, self.par, needle)

    def test_publish_happens_before_any_mask_processor_runs(self):
        # process_mask applies THIS block's MaskStabilizer inside itself, so what it returns is already filtered:
        # the shared stage must end before the first call to it.
        i_branch = self.face.index("            elif p.type == 'mask':")
        i_pub = self.face.index('_raw_cache.publish(')
        i_first_call = self.face.index('self.process_mask(')
        self.assertGreater(i_pub, i_branch)
        self.assertLess(i_pub, i_first_call)

    def test_a_hit_skips_swap_and_enhance_but_still_runs_every_mask_processor(self):
        # one guard in the swap branch, one at the top of the enhance branch, none in the mask branch
        swap_guard = "if _raw_hit is not None:" + chr(10) + " " * 16 + "continue"
        enhance_guard = "if _raw_hit is not None:" + chr(10) + " " * 20 + "continue"
        self.assertEqual(self.face.count(swap_guard), 1)
        self.assertEqual(self.face.count(enhance_guard), 1)
        mask_branch = self.face.split("            elif p.type == 'mask':", 1)[1].split("            elif (p.type == 'enhance'", 1)[0]
        self.assertNotIn('_raw_hit', mask_branch)
        self.assertNotIn("_raw_hit['img_mask']", self.face)       # the mask is never shared

    def test_the_cached_entry_holds_no_filtered_value(self):
        entry = self.face.split('_raw_cache.publish(_raw_slot, {', 1)[1].split('})', 1)[0]
        self.assertNotIn('mask', entry.replace('swap_model_mask', ''))

    def test_key_carries_the_matrix_and_the_source(self):
        k = self.face.split('_raw_cache.acquire((', 1)[1].split('))', 1)[0]
        for needle in ('_raw_gi', 'face_index', 'selected_src_idx', 'M, dtype=np.float32'):
            self.assertIn(needle, k, needle)

    def test_rotated_frontalized_and_quality_faces_never_use_it(self):
        gate = self.face.split('_raw_cache = self._stab_raw', 1)[1].split('_raw_gi = ', 1)[0]
        for needle in ('applied_rotation_action is None', 'rotation_action is None', 'M_frontal is None',
                       '_quality_mgr'):
            self.assertIn(needle, gate, needle)


if __name__ == '__main__':
    unittest.main(verbosity=2)
