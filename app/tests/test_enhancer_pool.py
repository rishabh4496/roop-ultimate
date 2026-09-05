"""The enhancer must be able to run on more than one thread at a time.

`_gpu_guard` exempts a processor from the global TensorRT lock only if that
processor owns a SessionPool. An enhancer without one therefore serialises the
single most expensive stage in a render — ~36% of wall clock — to one thread
while every other worker waits on the lock, and no `max_threads` setting can
lift that: with the enhancer at ~22 ms/face against ~57 ms of other per-face
work, throughput saturates at ~4 threads and then stops scaling entirely.

That is a silent failure. Everything still renders, correctly, just with the
GPU parked well below capacity — which is exactly how it went unnoticed.

These use a stubbed onnxruntime, so they assert the real leasing behaviour
rather than scanning source text for the shape of it. Source scanning would
pass on prose: this file's own docstring contains the words `self.pool` and
`lease`, and a grep-based guard that matched them here would report a pool that
does not exist.
"""

import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import session_pool                                    # noqa: E402
import roop.processors.Enhance_CodeFormer as CF                  # noqa: E402


class _FakeIOB:
    def __init__(self):
        self.bound = {}

    def bind_cpu_input(self, name, arr):
        self.bound[name] = arr

    def bind_output(self, name, dev):
        pass

    def copy_outputs_to_cpu(self):
        return [np.full((1, 3, 512, 512), 0.25, dtype=np.float32)]


class _FakeSession:
    """Records which of these are inside run_with_iobinding at once.

    Tracks DISTINCT sessions, not a plain count. Counting alone would make the
    concurrency test vacuous: the global lock this change removes lives in
    ProcessMgr's `_gpu_guard`, not in the processor, so with no pool at all four
    threads would still enter this stub together — on one shared session. What
    has to be proven is that each thread got its OWN TensorRT context, which is
    the thing `_gpu_guard` trusts when it waives the lock.
    """
    registry = []
    inflight = None
    peak_distinct = 0
    _lock = threading.Lock()

    def __init__(self, path, opts, providers):
        _FakeSession.registry.append(self)

    def io_binding(self):
        return _FakeIOB()

    def get_inputs(self):
        return [type("I", (), {"name": "x", "type": "tensor(float)"}),
                type("I", (), {"name": "w", "type": "tensor(double)"})]

    def get_outputs(self):
        return [type("O", (), {"name": "y"})]

    def run_with_iobinding(self, iob):
        with _FakeSession._lock:
            _FakeSession.inflight.add(id(self))
            _FakeSession.peak_distinct = max(_FakeSession.peak_distinct,
                                             len(_FakeSession.inflight))
        time.sleep(0.01)
        with _FakeSession._lock:
            _FakeSession.inflight.discard(id(self))

    @classmethod
    def reset(cls):
        cls.registry, cls.inflight, cls.peak_distinct = [], set(), 0


class EnhancerPoolCase(unittest.TestCase):
    """Swaps in a stub onnxruntime and restores the module-level pool cache, so
    one test can never leak a pool size into another.

    The process-wide TensorRT resource manager is reset for the same reason,
    and it is NOT optional. Every pool built here registers itself with that
    manager, and these stubbed sessions hold no real VRAM — so the manager sees
    its tracked budget grow by ~920MB per context while the card's free memory
    never moves, and charges the difference to the next test as
    `_resident_unobserved_mb`. After a few tests that phantom debt exceeds the
    card and later pools get admitted at 1 context, failing assertions about
    concurrency that pass perfectly well in isolation. Reset per test, so what
    a test measures is its own pool rather than the sum of the ones before it.
    """

    def setUp(self):
        self._ort = CF.onnxruntime
        self._resolve = CF.resolve_relative_path
        self._cache = dict(session_pool._pool_cache)
        self._manager = session_pool._resource_manager
        session_pool._resource_manager = session_pool.TensorRTResourceManager()
        CF.onnxruntime = type("ort", (), {
            "InferenceSession": _FakeSession,
            "SessionOptions": lambda: type("o", (), {
                "graph_optimization_level": None})(),
            "GraphOptimizationLevel": type("g", (), {"ORT_ENABLE_EXTENDED": 1}),
        })
        CF.resolve_relative_path = lambda p: p
        _FakeSession.reset()

    def tearDown(self):
        CF.onnxruntime = self._ort
        CF.resolve_relative_path = self._resolve
        session_pool._resource_manager = self._manager
        session_pool._pool_cache.clear()
        session_pool._pool_cache.update(self._cache)

    @staticmethod
    def _pools(n):
        session_pool._pool_cache.clear()
        session_pool._pool_cache.update({"trt": n, "detmask": n})

    def _make(self, fp16=False):
        p = CF.Enhance_CodeFormer()
        p.Initialize({"devicename": "cuda", "fp16": fp16})
        return p


class TestCodeFormerPool(EnhancerPoolCase):
    def test_builds_one_session_per_pool_slot(self):
        self._pools(4)
        p = self._make()
        self.assertEqual(len(_FakeSession.registry), 4)
        self.assertIsNotNone(p.pool)

    def test_gpu_guard_sees_the_pool(self):
        """This attribute is the whole mechanism: _gpu_guard checks
        `getattr(p, 'pool', None) is not None` to decide whether to hand this
        stage the global lock."""
        self._pools(4)
        self.assertIsNotNone(getattr(self._make(), "pool", None))
        self._pools(0)
        self.assertIsNone(getattr(self._make(), "pool", None))

    @staticmethod
    def _run_threads(p, n=4):
        frame = np.zeros((512, 512, 3), np.uint8)
        threads = [threading.Thread(target=lambda: p.Run(None, None, frame))
                   for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return _FakeSession.peak_distinct

    def test_threads_run_on_their_own_contexts(self):
        """The point of the change: four workers inside inference at once, each
        on a DIFFERENT session."""
        self._pools(4)
        self.assertEqual(self._run_threads(self._make()), 4)

    def test_that_assertion_is_not_vacuous(self):
        """Same four threads with pooling off must collapse onto ONE session.

        Without this, the test above passes whether or not a pool exists — the
        stub has no lock of its own, so four threads sharing a single session
        would still be 'concurrent'. Here they are concurrent on one context,
        which is precisely the state the global GPU lock exists to prevent.
        """
        self._pools(0)
        p = self._make()
        self.assertIsNone(p.pool)
        self.assertEqual(self._run_threads(p), 1)

    def test_pooled_output_matches_the_single_session_path(self):
        """Pooling is a concurrency change, not a numerical one."""
        frame = np.zeros((512, 512, 3), np.uint8)
        self._pools(4)
        pooled_frame, pooled_scale = self._make().Run(None, None, frame)
        self._pools(0)
        plain_frame, plain_scale = self._make().Run(None, None, frame)
        np.testing.assert_array_equal(pooled_frame, plain_frame)
        self.assertEqual(pooled_scale, plain_scale)

    def test_an_oom_building_extras_falls_back_rather_than_dying(self):
        """A big swapper, an expression pool and four restorer contexts all want
        the same 12GB. Losing that race must cost speed, not the render."""
        self._pools(4)
        calls = {"n": 0}

        def flaky(path, opts, providers):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("CUDA out of memory (simulated)")
            return _FakeSession(path, opts, providers)

        CF.onnxruntime.InferenceSession = flaky
        p = self._make()
        self.assertIsNone(p.pool, "OOM must leave the single-session path")
        frame = np.zeros((512, 512, 3), np.uint8)
        self.assertEqual(p.Run(None, None, frame)[0].shape, (512, 512, 3))

    def test_release_drops_the_pool_and_the_primary(self):
        self._pools(4)
        p = self._make()
        p.Release()
        self.assertIsNone(p.pool)
        self.assertIsNone(p.model_codeformer)

    def test_precision_switch_rebuilds_the_pool(self):
        """fp16 is a precision switch on the same graph, and it goes through
        Release/Initialize — which must not leave a pool of sessions built from
        the OTHER weights."""
        self._pools(4)
        p = self._make(fp16=True)
        first = list(_FakeSession.registry)
        p.Initialize({"devicename": "cuda", "fp16": False})
        self.assertIsNotNone(p.pool)
        self.assertEqual(len(_FakeSession.registry), len(first) + 4)


class TestRestoreFormerStillPools(EnhancerPoolCase):
    """The enhancer CodeFormer was compared against. If this one ever loses its
    pool the same ceiling comes back on the default recommendation.

    Asserted by RUNNING it, not by grepping for `self.pool.lease()`. That is
    how this test was written and it broke the moment the lease moved inside
    `enhance_common.exclusive` -- while the pool it was guarding was still
    there, still being leased, on every call. A guard that fails on a rename
    and passes on prose is checking the wrong thing; this file's own docstring
    says so.
    """

    def setUp(self):
        super().setUp()
        import roop.processors.Enhance_RestoreFormerPPlus as RF
        self.RF = RF
        self._rf_ort, self._rf_path = RF.onnxruntime, RF.resolve_relative_path
        RF.onnxruntime = CF.onnxruntime
        RF.resolve_relative_path = lambda q: q
        RF.Enhance_RestoreFormerPPlus.model_restoreformerpplus = None
        RF.Enhance_RestoreFormerPPlus.pool = None

    def tearDown(self):
        self.RF.onnxruntime, self.RF.resolve_relative_path = self._rf_ort, self._rf_path
        self.RF.Enhance_RestoreFormerPPlus.model_restoreformerpplus = None
        self.RF.Enhance_RestoreFormerPPlus.pool = None
        super().tearDown()

    def _make_rf(self):
        p = self.RF.Enhance_RestoreFormerPPlus()
        p.Initialize({"devicename": "cuda"})
        return p

    def test_builds_one_session_per_pool_slot(self):
        self._pools(4)
        p = self._make_rf()
        self.assertIsNotNone(p.pool)
        self.assertEqual(len(_FakeSession.registry), 4)

    def test_threads_run_on_their_own_contexts(self):
        self._pools(4)
        self.assertEqual(
            TestCodeFormerPool._run_threads(self._make_rf()), 4)


class TestSelfExcluding(EnhancerPoolCase):
    """The contract that lets ProcessMgr drop the enhance-stage lock.

    The stage lock was held across the WHOLE of Run(), and for the look-filter
    restorers the network is a small minority of that -- GPEN 256 Pro measured
    4.3 ms of GPU inside a 39.0 ms call, so ~89% of what was serialised never
    touched the GPU and the stage did not scale past one face at a time on any
    card without a pool. Dropping that lock is only safe while each processor
    excludes concurrent use of its OWN contexts, which is what
    `enhance_common.exclusive` does and what `self_excluding` promises.
    """

    ENHANCERS = ('Enhance_GPEN256Pro', 'Enhance_GPENRealistic', 'Enhance_UltraMax',
                 'Enhance_CodeFormer', 'Enhance_RestoreFormerPPlus',
                 'Enhance_GPEN', 'Enhance_GFPGAN')

    def test_every_converted_enhancer_declares_it(self):
        import importlib
        for name in self.ENHANCERS:
            mod = importlib.import_module(f'roop.processors.{name}')
            cls = getattr(mod, name)
            self.assertIs(getattr(cls, 'self_excluding', None), True,
                          f"{name} routes through exclusive() but does not "
                          f"declare self_excluding, so ProcessMgr still wraps "
                          f"its whole Run() in the enhance-stage lock")

    def test_unpooled_threads_do_not_share_a_context(self):
        """The guarantee the stage lock used to provide, now provided here.

        With pooling off there is ONE session, and before this change four
        worker threads entered it together -- safe only because the stage lock
        upstream kept them apart. With the lock gone the processor's own lock
        has to, so peak concurrency on the single context must be 1.
        """
        self._pools(0)
        p = self._make()
        self.assertIsNone(p.pool)
        self.assertEqual(TestCodeFormerPool._run_threads(p, n=4), 1)

    def test_the_stage_guard_reads_the_attribute(self):
        """`_gpu_guard(pooled=True)` is a no-op context; that is the whole
        mechanism. A processor declaring self_excluding must get one even with
        no pool, and one that declares nothing must still get the lock."""
        from roop.procmgr_runtime import _gpu_guard
        import roop.globals as g
        old = g.execution_providers
        g.execution_providers = ['TensorrtExecutionProvider']
        try:
            class _Declared:
                self_excluding = True
                pool = None

            class _Legacy:
                pool = None

            def guard_for(p):
                excl = getattr(p, 'self_excluding', None)
                if excl is None:
                    excl = getattr(p, 'pool', None) is not None
                return _gpu_guard(pooled=excl, owner='enhance')

            import contextlib
            self.assertIsInstance(guard_for(_Declared()), contextlib.nullcontext)
            self.assertNotIsInstance(guard_for(_Legacy()), contextlib.nullcontext)
        finally:
            g.execution_providers = old


if __name__ == "__main__":
    unittest.main()
