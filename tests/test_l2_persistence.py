"""app/core/l2_persistence.py: the carve-out API, the arena, and that nothing is left set behind.

These tests prove the calls are made correctly and cleaned up. They do NOT prove the cache keeps anything resident - that is
a measurement (tests/probe_l2_persistence.py, docs/perf/fused_postprocess/): the driver accepts a window it then ignores.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
from app.core import l2_persistence as l2  # noqa: E402

pytestmark = pytest.mark.gpu
_ok, _why = l2.available()
if not _ok:
    pytest.skip(f"L2 persistence unavailable: {_why}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _restore():
    yield
    l2.clear()


class TestQuery:
    def test_matches_torch_and_is_self_consistent(self):
        info = l2.query_l2()
        assert info.l2_bytes == torch.cuda.get_device_properties(0).L2_cache_size
        assert 0 <= info.max_persisting_bytes <= info.l2_bytes
        assert info.max_window_bytes >= info.max_persisting_bytes


class TestArena:
    def test_views_are_disjoint_aligned_and_inside_one_allocation(self):
        a = l2.PersistentArena(8 * l2.MiB)
        roi = a.alloc("roi", (2, 3, 128, 128), torch.float16)
        emb = a.alloc("emb", (2, 512), torch.float32)
        assert roi.is_contiguous() and tuple(roi.shape) == (2, 3, 128, 128) and emb.dtype == torch.float32
        lo, hi = a.base_ptr, a.base_ptr + a.nbytes
        for v in (roi, emb):
            assert lo <= v.data_ptr() < hi and v.data_ptr() % l2.ALIGN == 0
        assert roi.data_ptr() + roi.numel() * 2 <= emb.data_ptr()                       # no overlap
        roi.fill_(3.0)
        emb.fill_(7.0)
        assert float(roi.float().mean()) == 3.0 and float(emb.mean()) == 7.0            # writes do not bleed across views

    def test_overflow_is_refused(self):
        a = l2.PersistentArena(1 * l2.MiB)
        with pytest.raises(MemoryError):
            a.alloc("too_big", (2 * l2.MiB,), torch.uint8)


class TestWindow:
    def test_pin_grants_a_carve_out_and_covers_the_arena(self):
        info, a, s = l2.query_l2(), l2.PersistentArena(8 * l2.MiB), torch.cuda.Stream()
        a.alloc("roi", (4, 3, 256, 256), torch.float16)
        res = l2.pin_arena(s, a, 16 * l2.MiB, 1.0)
        assert res["applied"] and 0 < res["granted_persisting_bytes"] <= info.max_persisting_bytes
        assert res["window_bytes"] == a.used and res["hit_ratio"] == 1.0

    def test_a_window_larger_than_the_carve_out_scales_hit_ratio_down(self):
        info, s = l2.query_l2(), torch.cuda.Stream()
        a = l2.PersistentArena(32 * l2.MiB)
        a.alloc("big", (30 * l2.MiB,), torch.uint8)
        res = l2.pin_arena(s, a, 16 * l2.MiB, 1.0)
        assert res["window_bytes"] > res["granted_persisting_bytes"] and res["hit_ratio"] < 1.0
        assert res["hit_ratio"] == pytest.approx(res["granted_persisting_bytes"] / res["window_bytes"])

    def test_streaming_window_over_a_decode_ring(self):
        ring = torch.empty(4 * l2.MiB, dtype=torch.uint8, device="cuda")
        assert l2.mark_streaming(torch.cuda.Stream(), ring)["window_bytes"] == 4 * l2.MiB

    def test_clear_returns_the_carve_out_and_work_still_runs(self):
        s, a = torch.cuda.Stream(), l2.PersistentArena(4 * l2.MiB)
        v = a.alloc("x", (1024,), torch.float32)
        l2.pin_arena(s, a)
        with torch.cuda.stream(s):
            v.fill_(2.0)
        torch.cuda.synchronize()
        l2.clear(s)
        assert float(v.mean()) == 2.0
        from cuda.bindings import runtime as rt
        err, granted = rt.cudaDeviceGetLimit(rt.cudaLimit.cudaLimitPersistingL2CacheSize)
        assert int(err) == 0 and granted == 0


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
