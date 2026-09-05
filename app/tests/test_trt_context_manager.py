"""Phase 3 contracts for TensorRT context budgeting and safe pool lifecycle."""

import os
import sys
import threading
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import session_pool  # noqa: E402
from roop import bench  # noqa: E402


class TensorRTResourceManagerTests(unittest.TestCase):

    def test_phase4_sweep_includes_six_contexts(self):
        self.assertEqual(bench.PROFILES['full']['pool_levels'], (1, 2, 3, 4, 6))
        self.assertEqual(bench.PROFILES['stress']['pool_levels'], (1, 2, 3, 4, 6))
        self.assertEqual(session_pool._resource_spec('swapper:realswap').max_contexts, 6)

    def test_model_and_shape_are_part_of_the_budget(self):
        small = session_pool._resource_spec(
            'enhancer:gpen256', (1, 3, 256, 256)).slot_mb((1, 3, 256, 256))
        large = session_pool._resource_spec(
            'enhancer:gpen512', (1, 3, 512, 512)).slot_mb((1, 3, 512, 512))
        self.assertGreater(large, small)

    def test_auto_selection_uses_live_free_memory_and_safety_margin(self):
        manager = session_pool.TensorRTResourceManager()
        with mock.patch.object(session_pool, '_live_vram_mb', return_value=(9000, 12000)):
            self.assertEqual(manager.select_pool_size(
                4, 'enhancer:ultramax', (1, 3, 512, 512), explicit=False), 4)
        with mock.patch.object(session_pool, '_live_vram_mb', return_value=(1200, 12000)):
            self.assertEqual(manager.select_pool_size(
                4, 'enhancer:ultramax', (1, 3, 512, 512), explicit=False), 1)

    def test_explicit_selection_is_exempt_from_policy_caps(self):
        """Commit 0382a70's contract: no static tier ceiling, no measured knee.

        An operator who asks for more contexts than the automatic policy would
        pick gets them, provided the memory is there. A control that offers a
        value the backend quietly refuses to use is the defect that commit
        removed, and this asserts it stays removed.
        """
        manager = session_pool.TensorRTResourceManager()
        spec = session_pool._resource_spec('enhancer:ultramax', (1, 3, 512, 512))
        with mock.patch.object(session_pool, '_live_vram_mb', return_value=(11000, 12000)):
            # Above what the auto path would choose, and above any knee.
            with mock.patch.object(session_pool, '_matching_benchmark_knee', return_value=2):
                self.assertEqual(manager.select_pool_size(
                    4, 'enhancer:ultramax', (1, 3, 512, 512), explicit=True), 4)
                # ...and the auto path, same call, IS held to that knee.
                self.assertEqual(
                    session_pool.TensorRTResourceManager().select_pool_size(
                        4, 'enhancer:ultramax', (1, 3, 512, 512), explicit=False), 2)
        self.assertLessEqual(spec.slot_mb((1, 3, 512, 512)) * 4, 11000)

    def test_explicit_selection_is_still_bounded_by_physical_vram(self):
        """Explicit overrules POLICY, not physics.

        Building 4 x ~920MB of contexts with 100MB free does not run slower in
        a tunable way: the driver pages contexts over PCIe and the render
        wedges, reporting ~100% GPU "utilisation" at a third of the power limit
        with the memory bus near idle. It presents as a hang rather than an
        OOM, so no other guard catches it — which is exactly how a 12GB card
        reached 0.0GB free mid-render with every pool knob set explicitly.
        """
        manager = session_pool.TensorRTResourceManager()
        with mock.patch.object(session_pool, '_live_vram_mb', return_value=(100, 12000)):
            self.assertEqual(manager.select_pool_size(
                4, 'enhancer:ultramax', (1, 3, 512, 512), explicit=True), 1)

    def test_explicit_selection_survives_an_unreadable_vram_probe(self):
        """A missing reading says nothing about the card, so it must not be
        treated as pressure and used to overrule the operator. Only the
        automatic path falls back to one context there."""
        manager = session_pool.TensorRTResourceManager()
        with mock.patch.object(session_pool, '_live_vram_mb', return_value=(0, 0)):
            self.assertEqual(manager.select_pool_size(
                4, 'enhancer:ultramax', (1, 3, 512, 512), explicit=True), 4)
            self.assertEqual(manager.select_pool_size(
                4, 'enhancer:ultramax', (1, 3, 512, 512), explicit=False), 1)


class SessionPoolLifecycleTests(unittest.TestCase):

    def test_queue_is_bounded_and_warmup_covers_each_context(self):
        warmed = []
        pool = session_pool.SessionPool(
            lambda i: i, 2, model_key='test:swap',
            warmup_fn=lambda item, index: warmed.append((item, index)))
        self.assertEqual(pool._q.maxsize, 2)
        self.assertEqual(pool.warmup(), 2)
        self.assertEqual(warmed, [(0, 0), (1, 1)])
        pool.release()

    def test_leases_wait_for_pool_warmup_transition(self):
        started = threading.Event()
        finish = threading.Event()
        acquired = threading.Event()

        def warm(_item, _index):
            started.set()
            finish.wait(1.0)

        pool = session_pool.SessionPool(lambda i: i, 1,
                                        model_key='test:swap',
                                        warmup_fn=warm)
        thread = threading.Thread(target=pool.warmup)
        thread.start()
        self.assertTrue(started.wait(1.0))

        def waiter():
            with pool.lease():
                acquired.set()

        lease_thread = threading.Thread(target=waiter)
        lease_thread.start()
        time.sleep(0.03)
        self.assertFalse(acquired.is_set())
        finish.set()
        thread.join(1.0)
        lease_thread.join(1.0)
        self.assertTrue(acquired.is_set())
        pool.release()

    def test_admission_limit_bounds_new_leases(self):
        pool = session_pool.SessionPool(lambda i: i, 2, model_key='test:swap')
        pool.set_admission_limit(1)
        held = pool.lease()
        held.__enter__()
        acquired = threading.Event()

        def waiter():
            with pool.lease():
                acquired.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.03)
        self.assertFalse(acquired.is_set())
        held.__exit__(None, None, None)
        thread.join(1.0)
        self.assertTrue(acquired.is_set())
        pool.release()

    def test_release_waits_for_active_lease_before_clearing_items(self):
        pool = session_pool.SessionPool(lambda i: i, 1, model_key='test:swap')
        entered = pool.lease()
        entered.__enter__()
        released = threading.Event()

        def closer():
            pool.release()
            released.set()

        thread = threading.Thread(target=closer)
        thread.start()
        time.sleep(0.03)
        self.assertFalse(released.is_set())
        entered.__exit__(None, None, None)
        thread.join(1.0)
        self.assertTrue(released.is_set())
        self.assertEqual(pool.size, 0)

    def test_resize_refuses_active_contexts(self):
        pool = session_pool.SessionPool(lambda i: i, 1, model_key='test:swap')
        with pool.lease():
            with self.assertRaises(RuntimeError):
                pool.resize(2)
        pool.release()


if __name__ == '__main__':
    unittest.main()
