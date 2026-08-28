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

    def test_explicit_selection_is_not_overridden(self):
        manager = session_pool.TensorRTResourceManager()
        with mock.patch.object(session_pool, '_live_vram_mb', return_value=(100, 12000)):
            self.assertEqual(manager.select_pool_size(
                4, 'enhancer:ultramax', (1, 3, 512, 512), explicit=True), 4)


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
