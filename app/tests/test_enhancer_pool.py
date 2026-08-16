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
    one test can never leak a pool size into another."""

    def setUp(self):
        self._ort = CF.onnxruntime
        self._resolve = CF.resolve_relative_path
        self._cache = dict(session_pool._pool_cache)
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
    pool the same ceiling comes back on the default recommendation."""

    def test_source_declares_a_pool(self):
        import inspect
        import roop.processors.Enhance_RestoreFormerPPlus as RF
        src = inspect.getsource(RF)
        self.assertIn("session_pool.SessionPool", src)
        self.assertIn("self.pool.lease()", src)


if __name__ == "__main__":
    unittest.main()
