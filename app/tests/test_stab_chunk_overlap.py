"""The stabilization pool is persistent and chunks overlap; output order must not move.

`_run_stab_parallel` used to spawn a thread per block-slot for every chunk and join them all before the next chunk
could start, so each chunk ended in a tail where workers that drew cheap blocks idled behind the unluckiest one
(py-spy on a live 720p render: GPU util 13-17% for ~1-2 s of every ~10 s, CPU 24%). It now keeps `width` threads for
the whole render over one FIFO block queue, with up to two chunks in flight. These drive the REAL function with a
fake `process_frame`:

  * frames reach the writer exactly once and strictly in order, overlapped or not;
  * with a straggler block in every chunk the overlapped run is clearly faster than the one-chunk-at-a-time run;
  * a pause that lands while two chunks are in flight neither deadlocks nor drops, duplicates or reorders a frame;
  * the pool's threads are gone when the render ends.
"""

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

import roop.globals as globals_mod  # noqa: E402
from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.procmgr_runtime import pause_controller  # noqa: E402


class _Writer:
    def __init__(self):
        self.frames = []
        self.lock = threading.Lock()

    def write_frame(self, frame):
        with self.lock:
            self.frames.append(frame)


def _manager(process_frame, writer):
    m = ProcessMgr.__new__(ProcessMgr)
    m._runtime_scheduler = None
    m._stab_chunk_queue_capacity = 1
    m._stab_warmup = 0
    m._runtime_stab_small = False
    m._stab_frame_bytes = 3
    m._kps_stab_factory = None
    m._enh_stab_factory = None
    m._mask_stab_factory = None
    m._tls = threading.local()
    m.output_to_file = True
    m.output_to_cam = False
    m.videowriter = writer
    m.streamwriter = None
    m._temporal_faces = None
    m._track_assignments = None
    m._precomputed_kps = None
    m._runtime_monitor = None
    m._runtime_adaptive = None
    m.process_frame = process_frame
    return m


def _run(manager, n, geometry, threads=4):
    previous = globals_mod.processing
    globals_mod.processing = True
    try:
        with mock.patch.object(ProcessMgr, "_stab_parallel_geometry", return_value=geometry):
            manager._run_stab_parallel("unused", list(range(n)), 0, n, n, threads, lambda: None)
    finally:
        globals_mod.processing = previous


def _pool_threads():
    return [t for t in threading.enumerate() if t.name.startswith("stab_proc") and t.is_alive()]


class OverlappedChunks(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        pause_controller.finish()

    def tearDown(self):
        self._env.stop()
        pause_controller.finish()

    def _timed(self, overlap, n=128, block=4, width=4, slow=0.08, fast=0.004):
        os.environ["ROOP_STAB_OVERLAP"] = "1" if overlap else "0"

        def process_frame(frame, frame_idx=None, output_pending=False):
            # The first block of every chunk is the straggler.
            time.sleep(slow if (frame_idx // block) % width == 0 else fast)
            return ("out", frame_idx)

        writer = _Writer()
        m = _manager(process_frame, writer)
        t0 = time.perf_counter()
        _run(m, n, (0, block, width, width))
        return time.perf_counter() - t0, writer.frames

    def test_every_frame_is_written_once_and_in_order(self):
        for overlap in (True, False):
            _, frames = self._timed(overlap, n=100)
            self.assertEqual(frames, [("out", i) for i in range(100)], f"overlap={overlap}")

    def test_a_straggler_block_no_longer_stalls_the_next_chunk(self):
        _timed_off, _ = self._timed(False)
        timed_on, _ = self._timed(True)
        self.assertLess(timed_on, _timed_off * 0.8,
                        f"overlap on {timed_on:.2f}s vs off {_timed_off:.2f}s")

    def test_warmup_frames_are_primed_across_the_chunk_boundary_and_discarded(self):
        os.environ["ROOP_STAB_OVERLAP"] = "1"
        seen = []
        lock = threading.Lock()

        def process_frame(frame, frame_idx=None, output_pending=False):
            with lock:
                seen.append(frame_idx)
            time.sleep(0.002)
            return ("out", frame_idx)

        writer = _Writer()
        m = _manager(process_frame, writer)
        # warm-up 2, block 6, width 3, 3 blocks a chunk -> chunks of 18 frames
        _run(m, 54, (2, 6, 3, 3), threads=3)
        self.assertEqual(writer.frames, [("out", i) for i in range(54)])
        # Every block but the very first re-runs its 2 warm-up frames (blocks 1..8 -> 16 extra visits).
        self.assertEqual(len(seen), 54 + 16)

    def test_the_pool_does_not_outlive_the_render(self):
        self._timed(True, n=40)
        deadline = time.time() + 5
        while _pool_threads() and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(_pool_threads(), [])

    def test_a_pause_with_two_chunks_in_flight_drops_nothing_and_reorders_nothing(self):
        os.environ["ROOP_STAB_OVERLAP"] = "1"
        visits = []
        lock = threading.Lock()

        def process_frame(frame, frame_idx=None, output_pending=False):
            with lock:
                visits.append(frame_idx)
            time.sleep(0.03)
            return ("out", frame_idx)

        writer = _Writer()
        m = _manager(process_frame, writer)

        def pause_then_resume():
            time.sleep(0.35)
            pause_controller.request()
            time.sleep(0.5)
            pause_controller.resume()

        t = threading.Thread(target=pause_then_resume, daemon=True)
        t.start()
        done = threading.Event()
        err = []

        def go():
            try:
                _run(m, 160, (0, 4, 4, 4))
            except BaseException as exc:   # noqa: BLE001 - surfaced below
                err.append(exc)
            finally:
                done.set()

        runner = threading.Thread(target=go, daemon=True)
        runner.start()
        self.assertTrue(done.wait(60), "render hung across a pause")
        t.join()
        self.assertEqual(err, [])
        # The writer skips indices it already wrote, so a retried chunk cannot duplicate; what must hold is
        # that nothing is missing and nothing is out of order.
        self.assertEqual(writer.frames, [("out", i) for i in range(160)])
        # Prove the interrupt path actually ran: some frames were computed, dropped and recomputed.
        self.assertGreater(len(visits), 160, "the pause never interrupted a chunk; the test proved nothing")


if __name__ == "__main__":
    unittest.main()
