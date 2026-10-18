"""ZeroCopyBufferPool: the ring, its events, its locks, its VRAM accounting, and ORT binding.

What is pinned, and the failure each test stands for
----------------------------------------------------

* ORDERING. ``lease`` makes the lessee's stream wait on the slot's event, so a pass
  that reuses a slot cannot write it while the previous pass's GPU consumer (warp /
  blend / copy) is still reading it. ``test_the_next_lessee_waits_for_the_previous_consumer``
  has a negative control that shows the stale read happens without the event.
* OWNERSHIP. Slot locks make two threads on one slot take turns; a nested lease on one
  ring raises instead of deadlocking.
* BINDING. Bytes handed to ORT through ``data_ptr()`` are dense (a strided source is
  gathered by ``load``) and the result equals the host path bit for bit, on a real
  session, from many threads at once.
* TEARDOWN. ``scope`` frees the rings and clears bindings on a normal exit, an error
  and a cancel.
"""
import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch                                                       # noqa: E402

from roop.optimized_processor import (                            # noqa: E402
    ZeroCopyBufferPool, get_zero_copy_pool, release_zero_copy_pools, zero_copy_enabled)

CUDA = torch.cuda.is_available()
R = 64          # a small ring resolution; the real ones (512/1024) differ only in size


def _affine_model(out_dim0):
    """y = 2*x + 1, input [1,3,R,R]; output batch axis static (1) or symbolic ('N')."""
    import onnx
    from onnx import TensorProto, helper
    two = helper.make_tensor('two', TensorProto.FLOAT, [], [2.0])
    one = helper.make_tensor('one', TensorProto.FLOAT, [], [1.0])
    nodes = [helper.make_node('Mul', ['x', 'two'], ['m']), helper.make_node('Add', ['m', 'one'], ['y'])]
    graph = helper.make_graph(
        nodes, 'affine',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, R, R])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [out_dim0, 3, R, R])],
        initializer=[two, one])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    return model.SerializeToString()


def _cuda_session(out_dim0=1):
    import onnxruntime as ort
    sess = ort.InferenceSession(_affine_model(out_dim0),
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    if sess.get_providers()[0] != 'CUDAExecutionProvider':
        raise unittest.SkipTest('CUDA execution provider not active')
    return sess


@unittest.skipUnless(CUDA, 'CUDA required')
class TestRing(unittest.TestCase):
    def setUp(self):
        self.pool = ZeroCopyBufferPool(0, slots=4)

    def tearDown(self):
        self.pool.release()

    def test_slot_is_frame_index_modulo_slots(self):
        seen = []
        for frame in (0, 1, 2, 3, 4, 9):
            with self.pool.lease(R, frame_idx=frame) as slot:
                seen.append(slot.index)
        self.assertEqual(seen, [0, 1, 2, 3, 0, 1])

    def test_without_a_frame_index_the_slots_rotate(self):
        seen = []
        for _ in range(8):
            with self.pool.lease(R) as slot:
                seen.append(slot.index)
        self.assertEqual(sorted(set(seen)), [0, 1, 2, 3])
        self.assertEqual(len(seen), 8)

    def test_buffers_are_contiguous_cuda_float32_of_the_ring_shape(self):
        with self.pool.lease(R, frame_idx=0) as slot:
            for t in (slot.input, slot.output):
                self.assertTrue(t.is_cuda and t.is_contiguous())
                self.assertEqual(t.dtype, torch.float32)
                self.assertEqual(tuple(t.shape), (1, 3, R, R))
        ring = {s.input.data_ptr() for i in range(4) for s in [self._slot(i)]}
        self.assertEqual(len(ring), 4, 'four distinct input buffers')

    def _slot(self, i):
        with self.pool.lease(R, frame_idx=i) as slot:
            return slot

    def test_load_gathers_a_strided_source(self):
        src = torch.arange(R * R * 3, dtype=torch.float32, device='cuda').reshape(R, R, 3)
        view = src.permute(2, 0, 1).flip(0).unsqueeze(0)          # non-contiguous
        self.assertFalse(view.is_contiguous())
        with self.pool.lease(R, frame_idx=0) as slot:
            slot.load(view)
            self.assertTrue(slot.input.is_contiguous())
            self.assertTrue(torch.equal(slot.input, view))

    def test_load_rejects_a_wrong_shape(self):
        with self.pool.lease(R, frame_idx=0) as slot:
            with self.assertRaises(ValueError):
                slot.load(torch.zeros(1, 3, R, R + 1, device='cuda'))

    def test_a_ring_is_built_only_for_the_resolution_used(self):
        with self.pool.lease(R, frame_idx=0):
            pass
        stats = self.pool.stats()
        self.assertEqual(list(stats['rings'].values()), [4])
        self.assertEqual(len(stats['rings']), 1)
        self.assertAlmostEqual(stats['allocated_mib'], 4 * 2 * 3 * R * R * 4 / 1024 ** 2, places=6)

    def test_nested_lease_on_one_ring_raises(self):
        with self.pool.lease(R, frame_idx=0):
            with self.assertRaises(RuntimeError):
                with self.pool.lease(R, frame_idx=1):
                    pass
        with self.pool.lease(R, frame_idx=1):      # and the first lease released cleanly
            pass


@unittest.skipUnless(CUDA, 'CUDA required')
class TestOrdering(unittest.TestCase):
    CYCLES = 400_000_000        # ~0.15 s of GPU time: far longer than any host-side gap below

    def test_the_next_lessee_waits_for_the_previous_consumer(self):
        pool = ZeroCopyBufferPool(0, slots=1)
        try:
            def first():
                with pool.lease(R, frame_idx=0) as slot:
                    slot.output.zero_()
                    torch.cuda._sleep(self.CYCLES)
                    slot.output.fill_(1.0)        # the consumer is still "reading" when this lease ends

            t = threading.Thread(target=first)
            t.start()
            t.join()
            with pool.lease(R, frame_idx=0) as slot:
                snapshot = slot.output.clone()
            torch.cuda.synchronize()
            self.assertEqual(float(snapshot.min()), 1.0, 'second lessee read before the first finished')
        finally:
            pool.release()

    def test_negative_control_no_event_means_a_stale_read(self):
        """Same scenario with raw tensors and no event: proves the test above can fail."""
        buf = torch.zeros(1, 3, R, R, device='cuda')
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(s1):
            torch.cuda._sleep(self.CYCLES)
            buf.fill_(1.0)
        with torch.cuda.stream(s2):
            snapshot = buf.clone()
        torch.cuda.synchronize()
        self.assertEqual(float(snapshot.max()), 0.0)

    def test_two_threads_on_one_slot_take_turns(self):
        pool = ZeroCopyBufferPool(0, slots=1)
        order, inside, gate = [], threading.Event(), threading.Event()
        try:
            def a():
                with pool.lease(R, frame_idx=0):
                    order.append('a-in')
                    inside.set()
                    gate.wait(5)
                    order.append('a-out')

            def b():
                inside.wait(5)
                with pool.lease(R, frame_idx=0):
                    order.append('b-in')

            ta, tb = threading.Thread(target=a), threading.Thread(target=b)
            ta.start(); tb.start()
            inside.wait(5)
            threading.Event().wait(0.2)           # b is now parked on the slot lock
            self.assertEqual(order, ['a-in'])
            gate.set()
            ta.join(5); tb.join(5)
            self.assertEqual(order, ['a-in', 'a-out', 'b-in'])
        finally:
            pool.release()


@unittest.skipUnless(CUDA, 'CUDA required')
class TestVramAndTeardown(unittest.TestCase):
    def test_a_ring_that_would_eat_the_margin_is_built_with_fewer_slots(self):
        pool = ZeroCopyBufferPool(0, slots=4, margin_mb=1000.0)
        per_slot_mb = 2 * 3 * R * R * 4 / 1024 ** 2
        real = torch.cuda.mem_get_info
        # free = margin + room for exactly two slots
        torch.cuda.mem_get_info = lambda d=0: (int((1000.0 + 2.5 * per_slot_mb) * 1024 ** 2), 1 << 34)
        try:
            with pool.lease(R, frame_idx=0):
                pass
        finally:
            torch.cuda.mem_get_info = real
            stats = pool.stats()
            pool.release()
        self.assertEqual(list(stats['rings'].values()), [2])
        self.assertEqual(list(stats['degraded'].values()), [2])

    def test_peak_survives_release_and_allocated_returns_to_zero(self):
        pool = ZeroCopyBufferPool(0)
        with pool.lease(R):
            pass
        peak = pool.stats()['peak_mib']
        pool.release()
        after = pool.stats()
        self.assertEqual(after['allocated_mib'], 0.0)
        self.assertEqual(after['rings'], {})
        self.assertEqual(after['peak_mib'], peak)
        self.assertGreater(peak, 0.0)

    def test_scope_releases_on_error_and_on_cancel(self):
        for exc in (RuntimeError, KeyboardInterrupt):
            pool = ZeroCopyBufferPool(0)
            cleared = []

            class Iob:
                def clear_binding_inputs(self): cleared.append('in')
                def clear_binding_outputs(self): cleared.append('out')

            with self.assertRaises(exc):
                with pool.scope(Iob()):
                    with pool.lease(R):
                        pass
                    raise exc()
            self.assertEqual(pool.stats()['rings'], {}, exc.__name__)
            self.assertEqual(cleared, ['in', 'out'], exc.__name__)
            with pool.lease(R):          # and it is reusable afterwards
                pass
            pool.release()

    def test_release_waits_for_an_inflight_lease(self):
        pool = ZeroCopyBufferPool(0, slots=1)
        inside, gate, done = threading.Event(), threading.Event(), threading.Event()

        def holder():
            with pool.lease(R, frame_idx=0):
                inside.set()
                gate.wait(5)

        t = threading.Thread(target=holder); t.start(); inside.wait(5)
        r = threading.Thread(target=lambda: (pool.release(), done.set())); r.start()
        threading.Event().wait(0.2)
        self.assertFalse(done.is_set(), 'release() freed a ring that is still leased')
        gate.set(); t.join(5); r.join(5)
        self.assertTrue(done.is_set())

    def test_a_lessee_parked_on_a_released_ring_gets_a_fresh_one(self):
        """release() nulls the slot tensors; a thread already waiting on that slot's
        lock must retry on a new ring, not crash on `None`."""
        pool = ZeroCopyBufferPool(0, slots=1)
        inside, gate = threading.Event(), threading.Event()
        got = {}

        def holder():
            with pool.lease(R, frame_idx=0):
                inside.set()
                gate.wait(5)

        def waiter():
            inside.wait(5)
            with pool.lease(R, frame_idx=0) as slot:
                got['input'] = slot.input

        t, w = threading.Thread(target=holder), threading.Thread(target=waiter)
        t.start(); w.start(); inside.wait(5)
        threading.Event().wait(0.2)                 # waiter is parked on the slot lock
        r = threading.Thread(target=pool.release); r.start()
        threading.Event().wait(0.2)                 # release() is parked behind the holder
        gate.set()
        for x in (t, w, r):
            x.join(10)
        self.assertFalse(any(x.is_alive() for x in (t, w, r)))
        self.assertIsNotNone(got.get('input'), 'the waiter must have leased a live slot')
        self.assertEqual(tuple(got['input'].shape), (1, 3, R, R))
        pool.release()

    def test_the_retry_path_itself(self):
        """Deterministic version of the above: a held reference to a released ring is skipped."""
        pool = ZeroCopyBufferPool(0, slots=1)
        with pool.lease(R, frame_idx=0):
            pass
        stale = next(iter(pool._rings.values()))
        real = pool._ring
        calls = []

        def once_stale(resolution, dtype, out_dtype):
            if not calls:
                calls.append(1)
                stale[0].input = None          # exactly what release() leaves behind
                return stale
            return real(resolution, dtype, out_dtype)

        pool._ring = once_stale
        pool._rings.clear()                    # and the pool no longer lists it
        with pool.lease(R, frame_idx=0) as slot:
            self.assertIsNotNone(slot.input)
            self.assertIsNot(slot, stale[0])
        self.assertEqual(len(calls), 1)
        pool.release()

    def test_module_helpers(self):
        a, b = get_zero_copy_pool(0), get_zero_copy_pool(0)
        self.assertIs(a, b)
        with a.lease(R):
            pass
        release_zero_copy_pools()
        self.assertEqual(a.stats()['rings'], {})
        old = os.environ.get('ROOP_ZEROCOPY_ENHANCE')
        try:
            os.environ['ROOP_ZEROCOPY_ENHANCE'] = '0'
            self.assertFalse(zero_copy_enabled())
            os.environ['ROOP_ZEROCOPY_ENHANCE'] = '1'
            self.assertTrue(zero_copy_enabled())
        finally:
            if old is None:
                os.environ.pop('ROOP_ZEROCOPY_ENHANCE', None)
            else:
                os.environ['ROOP_ZEROCOPY_ENHANCE'] = old


@unittest.skipUnless(CUDA, 'CUDA required')
class TestOrtBinding(unittest.TestCase):
    def setUp(self):
        self.pool = ZeroCopyBufferPool(0, slots=4)

    def tearDown(self):
        self.pool.release()

    def _host(self, sess, x):
        return sess.run(None, {'x': x})[0]

    def test_bound_input_and_output_equal_the_host_path_bit_for_bit(self):
        sess = _cuda_session(1)
        rng = np.random.RandomState(0)
        for i in range(6):
            x = rng.randn(1, 3, R, R).astype(np.float32)
            with self.pool.lease(R, frame_idx=i) as slot:
                slot.load(torch.from_numpy(x).cuda())
                out = slot.run(sess, sess.io_binding(), 'x', 'y', output_is_static=True)
                self.assertEqual(out.data_ptr(), slot.output.data_ptr(), 'ORT must write into the slot')
                got = out.cpu().numpy()
            self.assertTrue(np.array_equal(got, self._host(sess, x)))

    def test_a_strided_source_is_gathered_not_misread(self):
        sess = _cuda_session(1)
        rng = np.random.RandomState(1)
        hwc = torch.from_numpy(rng.rand(R, R, 3).astype(np.float32)).cuda()
        view = hwc.permute(2, 0, 1).flip(0).unsqueeze(0)
        want = self._host(sess, view.contiguous().cpu().numpy())
        with self.pool.lease(R, frame_idx=0) as slot:
            slot.load(view)
            got = slot.run(sess, sess.io_binding(), 'x', 'y').cpu().numpy()
        self.assertTrue(np.array_equal(got, want))
        # the trap `load` exists for: the raw pointer of the strided view reads other bytes
        iob = sess.io_binding()
        iob.bind_input('x', 'cuda', 0, np.float32, (1, 3, R, R), view.data_ptr())
        iob.bind_output('y', 'cuda', 0)
        sess.run_with_iobinding(iob)
        iob.synchronize_outputs()
        wrong = iob.copy_outputs_to_cpu()[0]
        self.assertFalse(np.array_equal(wrong, want), 'binding a strided view directly must be wrong')

    def test_a_zero_dim_extra_input_is_bound_with_shape_empty_tuple(self):
        """CodeFormer's fidelity `w` is a 0-D float64: bound as shape (), not (1,)."""
        from unittest.mock import MagicMock
        iob, sess = MagicMock(), MagicMock()
        w = torch.tensor(0.65, device='cuda', dtype=torch.float64)
        with self.pool.lease(R, frame_idx=0) as slot:
            slot.run(sess, iob, 'x', 'y', 0, output_is_static=True, extra_inputs={'w': w})
        names = [c.args[0] for c in iob.bind_input.call_args_list]
        self.assertEqual(names, ['x', 'w'])
        self.assertEqual(iob.bind_input.call_args_list[1].args[4], ())
        self.assertEqual(iob.bind_input.call_args_list[1].args[3], np.dtype(np.float64))
        self.assertEqual(iob.bind_input.call_args_list[0].args[5], slot.input.data_ptr())

    def test_a_dynamic_output_is_ort_owned_and_still_exact(self):
        sess = _cuda_session('N')
        x = np.random.RandomState(2).randn(1, 3, R, R).astype(np.float32)
        with self.pool.lease(R, frame_idx=0) as slot:
            slot.load(torch.from_numpy(x).cuda())
            out = slot.run(sess, sess.io_binding(), 'x', 'y', output_is_static=False)
            self.assertNotEqual(out.data_ptr(), slot.output.data_ptr(), 'dynamic output must not be Torch-owned')
            got = out.cpu().numpy()
        want = self._host(sess, x)
        self.assertTrue(np.array_equal(got, want),
                        f"shape {got.shape} vs {want.shape}, dtype {got.dtype}, "
                        f"max|d|={np.abs(got - want).max()}, nan={np.isnan(got).sum()}, "
                        f"got[0,0,0,:3]={got[0,0,0,:3]} want={want[0,0,0,:3]}")

    def test_the_ort_owned_copy_is_complete_before_torch_reads_it(self):
        """Regression: `cudaMemcpy` D2D does not block the host, and the ring's worker streams
        do not wait for the legacy default stream, so ~5% of calls (17/300 single-threaded)
        returned stale memory. Many short calls is how that shows."""
        sess = _cuda_session('N')
        iob = sess.io_binding()
        rng = np.random.RandomState(5)
        bad = 0
        for i in range(250):
            x = rng.randn(1, 3, R, R).astype(np.float32)
            with self.pool.lease(R, frame_idx=i) as slot:
                slot.load(torch.from_numpy(x).cuda())
                got = slot.run(sess, iob, 'x', 'y', output_is_static=False).cpu().numpy()
            bad += int(not np.array_equal(got, 2 * x + 1))
        self.assertEqual(bad, 0, f"{bad}/250 ORT-owned outputs were read before the copy finished")

    def test_many_threads_on_four_slots_never_see_each_others_pixels(self):
        threads, rounds, errors = 8, 25, []

        def worker(k):
            try:
                sess = _cuda_session(1)
                iob = sess.io_binding()
                rng = np.random.RandomState(100 + k)
                for i in range(rounds):
                    x = rng.randn(1, 3, R, R).astype(np.float32)
                    with self.pool.lease(R, frame_idx=k * rounds + i) as slot:
                        slot.load(torch.from_numpy(x).cuda())
                        got = slot.run(sess, iob, 'x', 'y').cpu().numpy()
                    if not np.array_equal(got, 2 * x + 1):
                        errors.append((k, i))
            except Exception as error:      # surface, never hang the join
                errors.append((k, repr(error)))

        ts = [threading.Thread(target=worker, args=(k,)) for k in range(threads)]
        for t in ts: t.start()
        for t in ts: t.join(120)
        self.assertFalse(any(t.is_alive() for t in ts), 'a worker hung')
        self.assertEqual(errors, [])
        self.assertEqual(self.pool.stats()['leases'], threads * rounds)


if __name__ == '__main__':
    unittest.main()
