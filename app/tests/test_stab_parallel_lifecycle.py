"""Regression tests for bounded stabilized-reader/writer shutdown."""

import os
import sys
import threading
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals as globals_mod  # noqa: E402
from roop.ProcessMgr import ProcessMgr  # noqa: E402


class _FailingFrames:
    def __getitem__(self, _key):
        class _Iterator:
            def __iter__(self):
                return self

            def __next__(self):
                raise RuntimeError("decode boom")

        return _Iterator()


class StabilizedLifecycle(unittest.TestCase):
    def test_normal_eof_flushes_the_final_chunk(self):
        manager = ProcessMgr.__new__(ProcessMgr)
        manager._runtime_scheduler = None
        manager._stab_chunk_queue_capacity = 1
        manager._stab_warmup = 0
        manager._runtime_stab_small = False
        manager._stab_frame_bytes = 3
        manager._kps_stab_factory = None
        manager._enh_stab_factory = None
        manager._mask_stab_factory = None
        manager._tls = threading.local()
        manager.output_to_file = False
        manager.output_to_cam = False
        manager.videowriter = None
        manager.streamwriter = None
        manager._temporal_faces = None
        manager._track_assignments = None
        manager._precomputed_kps = None
        manager._runtime_monitor = None
        manager._runtime_adaptive = None
        manager.process_frame = lambda frame, frame_idx=None, output_pending=False: frame

        previous = globals_mod.processing
        globals_mod.processing = True
        try:
            with mock.patch.object(ProcessMgr, "_stab_parallel_geometry",
                                   return_value=(0, 1, 1, 1)):
                manager._run_stab_parallel(
                    "unused", ["frame-0", "frame-1"], 0, 2, 2, 1, lambda: None)
        finally:
            globals_mod.processing = previous

        self.assertFalse(manager._parallel_stab)

    def test_reader_failure_does_not_leave_consumer_or_writer_blocked(self):
        manager = ProcessMgr.__new__(ProcessMgr)
        manager._runtime_scheduler = None
        manager._stab_chunk_queue_capacity = 1
        manager._stab_warmup = 0
        manager._runtime_stab_small = False
        manager._stab_frame_bytes = 3
        manager._kps_stab_factory = None
        manager._enh_stab_factory = None
        manager._mask_stab_factory = None
        manager._tls = threading.local()
        manager.output_to_file = False
        manager.output_to_cam = False
        manager.videowriter = None
        manager.streamwriter = None
        manager._temporal_faces = None
        manager._track_assignments = None
        manager._precomputed_kps = None
        manager._runtime_monitor = None
        manager._runtime_adaptive = None

        previous = globals_mod.processing
        globals_mod.processing = True
        try:
            with mock.patch.object(ProcessMgr, "_stab_parallel_geometry",
                                   return_value=(0, 1, 1, 1)):
                with self.assertRaisesRegex(RuntimeError, "decode boom"):
                    manager._run_stab_parallel(
                        "unused", _FailingFrames(), 0, 1, 1, 1, lambda: None)
        finally:
            globals_mod.processing = previous

        self.assertFalse(
            any(thread.name in ("stab_reader", "stab_writer")
                and thread.is_alive() for thread in threading.enumerate())
        )


if __name__ == "__main__":
    unittest.main()
