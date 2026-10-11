"""Targeted regression tests for sustained-throughput fixes.

These tests stay CPU-only. They verify the contracts that prevent repeated disk
work, stale temporal-state copying, excessive VRAM admission queries, and
missing job-boundary cleanup without requiring a model or GPU.
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import project_checkpoint as checkpoint  # noqa: E402
from roop.temporal_identity import TemporalIdentityStabilizer  # noqa: E402
import roop.session_pool as session_pool  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SustainedThroughputFixTest(unittest.TestCase):
    def test_checkpoint_identity_uses_cached_segment_digest(self):
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(b"segment-content")
            path = fh.name
        try:
            with mock.patch.object(checkpoint, "file_sha256", side_effect=AssertionError("rehash")):
                identity = checkpoint.file_identity(path, precomputed_sha256="cached-digest")
            self.assertEqual(identity["sha256"], "cached-digest")
        finally:
            os.remove(path)

    def test_vram_query_is_cached_within_ttl(self):
        calls = []
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda: (calls.append(1) or (4 * 1024**3, 8 * 1024**3)),
            )
        )
        old_cache = session_pool._live_vram_cache
        old_cached_t = session_pool._live_vram_cached_t
        try:
            session_pool._live_vram_cache = (0.0, 0.0)
            session_pool._live_vram_cached_t = 0.0
            with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                 mock.patch.object(session_pool.time, "perf_counter", side_effect=(10.0, 10.1)):
                first = session_pool._live_vram_mb(ttl=0.5)
                second = session_pool._live_vram_mb(ttl=0.5)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
        finally:
            session_pool._live_vram_cache = old_cache
            session_pool._live_vram_cached_t = old_cached_t

    def test_temporal_clone_drops_only_long_retired_tracks(self):
        engine = TemporalIdentityStabilizer(enabled=True)
        for track_id in range(65):
            state = engine._state(track_id)
            state.last_frame_index = 1000 if track_id == 64 else 0
        recent = engine._state(100)
        recent.last_frame_index = 900

        clone = engine.clone_for_block()

        self.assertIn(64, clone.states)
        self.assertIn(100, clone.states)
        self.assertNotIn(0, clone.states)
        self.assertEqual(len(clone.states), 2)

    def test_job_boundary_cleanup_is_wired(self):
        api_source = open(os.path.join(ROOT, "app", "api.py"), encoding="utf-8").read()
        core_source = open(os.path.join(ROOT, "app", "roop", "core.py"), encoding="utf-8").read()
        self.assertIn("from roop.core import release_resources", api_source)
        self.assertIn('"release_" + "recognition_engine"', core_source)
        self.assertIn("torch.cuda.empty_cache()", core_source)
        self.assertIn("_cuda_clearCublasWorkspaces", core_source)


if __name__ == "__main__":
    unittest.main()
