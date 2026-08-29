"""The GPU lock is split per stage. These are the invariants that keeps.

WHY THIS FILE EXISTS. TensorRT's only rule is that one execution context must
not be entered by two threads at once. A single global lock enforced far more
than that -- detect excluded mask, mask excluded swap, swap excluded enhance,
none of which share a context. On a card big enough for pools nobody noticed,
because `pooled=True` bypasses the lock at every stage. Below 7GB
`_auto_pool_defaults` turns both pools OFF deliberately (the VRAM is not there),
so every stage fell through to that one lock and the whole pipeline ran one
thread wide: measured 9.49 fps at 31.6% GPU utilisation against 20.84 fps at
44.6% once the lock was split, on identical VRAM (2346 MB). The unpooled path
now matches the pooled one (22.22 fps, 4100 MB) to within a few percent, so the
small card keeps its VRAM headroom AND its speed.

Getting the grouping WRONG is a corrupted CUDA context rather than a slow
render, so these tests are about which calls must continue to share a lock.
"""
import os
import re
import sys
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import roop.globals                                            # noqa: E402
from roop.procmgr_runtime import _gpu_guard, _gpu_lock         # noqa: E402


def src(*parts):
    with open(os.path.join(APP, *parts), encoding='utf-8') as f:
        return f.read()


class Mechanics(unittest.TestCase):

    def setUp(self):
        self._saved = roop.globals.execution_providers
        roop.globals.execution_providers = ['TensorrtExecutionProvider']

    def tearDown(self):
        roop.globals.execution_providers = self._saved

    def test_the_same_owner_gets_the_same_lock(self):
        self.assertIs(_gpu_guard(owner='analysis'), _gpu_guard(owner='analysis'))

    def test_different_owners_get_different_locks(self):
        seen = [_gpu_guard(owner=k) for k in
                ('analysis', 'swap', 'mask', 'enhance', 'expression')]
        self.assertEqual(len(set(id(x) for x in seen)), len(seen))

    def test_no_owner_keeps_the_old_global_lock(self):
        """Any site not explicitly classified must behave exactly as before."""
        self.assertIs(_gpu_guard(), _gpu_lock)

    def test_a_stage_lock_is_not_the_global_lock(self):
        """Or splitting them would have changed nothing."""
        self.assertIsNot(_gpu_guard(owner='swap'), _gpu_lock)

    def test_pooled_still_bypasses_every_lock(self):
        """A pooled stage leases its own context and must NOT also serialise."""
        for k in (None, 'analysis', 'swap'):
            with _gpu_guard(pooled=True, owner=k):
                pass
        self.assertFalse(_gpu_guard(owner='swap').locked())

    def test_a_non_tensorrt_provider_takes_no_lock_at_all(self):
        roop.globals.execution_providers = ['CUDAExecutionProvider']
        with _gpu_guard(owner='analysis'):
            with _gpu_guard(owner='analysis'):
                pass          # a real lock would deadlock here

    def test_the_lock_registry_is_built_thread_safely(self):
        """Workers reach a new owner concurrently on the first frame, and two
        threads that each mint their own Lock for the same key would not exclude
        each other at all."""
        from roop import procmgr_runtime as rt
        got, start = [], threading.Barrier(8)

        def grab():
            start.wait()
            got.append(rt._stage_lock('race-probe'))

        ts = [threading.Thread(target=grab) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(len(set(id(x) for x in got)), 1)


class EveryAnalyserCallSharesOneLock(unittest.TestCase):
    """detect, re-detect, the autorotate confirmation, verify and the tracking
    pre-pass all enter the SAME FaceAnalysis instance when pooling is off. They
    must therefore all name the same owner."""

    def setUp(self):
        self.pm = src('roop', 'ProcessMgr.py')
        self.tr = src('roop', 'procmgr_tracking.py')

    def test_no_analyser_guard_is_left_unowned(self):
        """`_gpu_guard(pooled=analysis_pooled())` with no owner takes the GLOBAL
        lock, which no longer excludes the other detections."""
        for name, code in (('ProcessMgr.py', self.pm),
                           ('procmgr_tracking.py', self.tr)):
            for m in re.finditer(r"_gpu_guard\(pooled=analysis_pooled\(\)([^)]*)\)", code):
                self.assertIn("owner='analysis'", m.group(1),
                              f"{name}: an analyser guard without owner='analysis'")

    def test_the_autorotate_redetect_is_guarded_at_all(self):
        """REGRESSION. `rotface = get_first_face(rotcutplate)` ran with no guard
        whatsoever. `lease_face_analyser` documents that the caller serialises
        when there is no pool, so with pooling off -- which is the default below
        7GB -- two workers could enter one TensorRT context at once. Invisible on
        a big card, because the lease hands each thread its own instance there."""
        i = self.pm.index('rotface = face_util.get_first_face_detector_only(rotcutplate)')
        window = self.pm[max(0, i - 400):i]
        self.assertIn("_gpu_guard(pooled=analysis_pooled(), owner='analysis')",
                      window)

    def test_verify_shares_the_analysis_lock(self):
        self.assertIn("_prof('verify'), _gpu_guard(pooled=analysis_pooled(), "
                      "owner='analysis')", self.pm)


class TheStagesThatMustNotShare(unittest.TestCase):
    """swap, mask and enhance own disjoint sessions -- verified by the fact that
    none of those processors imports the face analyser at all. If one ever does,
    it needs the 'analysis' lock too and this test is where that gets caught."""

    def test_no_swap_mask_or_enhance_processor_touches_the_analyser(self):
        import glob
        bad = []
        for pat in ('Mask_*.py', 'Enhance_*.py', 'FaceSwap*.py', 'Expression_*.py'):
            for f in glob.glob(os.path.join(APP, 'roop', 'processors', pat)):
                code = open(f, encoding='utf-8').read()
                if re.search(r'\b(get_all_faces|get_first_face|'
                             r'get_face_analyser|lease_face_analyser)\s*\(', code):
                    bad.append(os.path.basename(f))
        self.assertEqual(bad, [], f"these now detect and need owner='analysis' "
                                  f"on their stage guard: {bad}")

    def test_the_stage_owners_used_are_the_documented_set(self):
        used = set()
        for code in (src('roop', 'ProcessMgr.py'),
                     src('roop', 'procmgr_tracking.py')):
            # The call contains nested parens (`pooled=analysis_pooled()`), so
            # `[^)]*` stops at the FIRST ')' and finds only the owners whose call
            # has no nested call before them -- which is how this test first
            # reported {'swap', 'expression'} on a file carrying all five.
            used |= set(re.findall(
                r"_gpu_guard\((?:[^()]|\([^()]*\))*owner='([a-z]+)'", code))
        self.assertEqual(used, {'analysis', 'swap', 'mask', 'enhance', 'expression'})


class TheSmallCardPolicy(unittest.TestCase):

    def test_pools_are_off_below_7gb(self):
        """Deliberate and measured: the pooled config costs 4100 MB against
        2346 MB, and a 6GB card cannot pay that beside the desktop. It is now
        nearly free to leave them off -- 20.84 fps against the pooled 22.22."""
        from roop import session_pool
        # The EXPLICIT overrides have to be cleared, not just the VRAM set. This
        # asserts the AUTO tier. Several other
        # test modules set it at import time in this shared process, so without
        # this the test reads their value and fails on a policy that is fine.
        saved = {k: os.environ.pop(k, None) for k in
                 ('ROOP_TRT_POOL', 'ROOP_DETMASK_POOL')}
        os.environ['ROOP_VRAM_GB'] = '6'
        try:
            session_pool._pool_cache.clear()
            self.assertEqual(session_pool._auto_pool_defaults(), (0, 0))
            self.assertFalse(session_pool.pooling_enabled())
        finally:
            os.environ.pop('ROOP_VRAM_GB', None)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
            session_pool._pool_cache.clear()

    def test_an_explicit_override_is_clamped_on_the_small_card_tier(self):
        """A stale or benchmark-written pool value cannot bypass the 6GB
        single-context safety boundary after restart."""
        from roop import session_pool
        saved = {k: os.environ.pop(k, None) for k in
                 ('ROOP_TRT_POOL', 'ROOP_DETMASK_POOL')}
        os.environ['ROOP_VRAM_GB'] = '6'
        os.environ['ROOP_TRT_POOL'] = '2'
        try:
            session_pool._pool_cache.clear()
            self.assertEqual(session_pool.pool_size(), 0)
            self.assertFalse(session_pool.pooling_enabled())
        finally:
            for k in ('ROOP_VRAM_GB', 'ROOP_TRT_POOL'):
                os.environ.pop(k, None)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
            session_pool._pool_cache.clear()

    def test_the_vram_override_exists_so_the_small_tier_can_be_tested(self):
        """Every tier below 7GB is a decision about a card most of this project's
        measurement happens away from. A policy that can only be exercised by
        physically being on that machine does not get exercised."""
        from roop import session_pool
        os.environ['ROOP_VRAM_GB'] = '6'
        try:
            self.assertEqual(session_pool._detect_vram_gb(), 6.0)
        finally:
            os.environ.pop('ROOP_VRAM_GB', None)

    def test_a_junk_override_falls_back_to_real_detection(self):
        from roop import session_pool
        os.environ['ROOP_VRAM_GB'] = 'not-a-number'
        try:
            self.assertIsInstance(session_pool._detect_vram_gb(), float)
        finally:
            os.environ.pop('ROOP_VRAM_GB', None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
