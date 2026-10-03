"""`roop.core` must not turn on `cudnn.benchmark`.

It used to (`_configure_torch_cuda_acceleration`), and with it on:

* a render whose ROI size changes every frame re-autotunes cuDNN on every new shape: the
  compositing engine's CUDA path measured 17.7 ms/call off, 456 ms/call on (single thread) and
  485 ms/call at 8 threads, for ONE repeated shape no gain at all (17.4 vs 17.5 ms);
* the result depends on WHICH THREAD ran it. PyTorch's cuDNN benchmark cache is thread-local,
  so every worker autotunes alone and can pick another algorithm: with `roop.core` imported,
  430 of 960 concurrent `composite_roi` calls differed by one level on ~4 pixels from the
  single-threaded result. That was `tests/test_stage8_compositing.py::test_thread_safety`
  failing in every full run and passing alone (importing `roop.core` is what a full run adds).
"""
import concurrent.futures as futures
import os
import sys

import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

torch = pytest.importorskip("torch")


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def test_importing_core_leaves_cudnn_benchmark_off():
    import roop.core  # noqa: F401  (the import is the thing under test)
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_concurrent_cuda_compositing_matches_the_single_thread_result():
    import roop.core  # noqa: F401  (sets the global torch flags, as a render does)
    from roop.compositing_engine import CompositingQualityEngine
    engine = CompositingQualityEngine()
    rng = np.random.default_rng(100)
    p = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    t = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    m = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float32)
    expected = engine.composite_roi(p, t, m)
    for _ in range(8):                                  # one pass flaked; eight rarely agree by luck
        with futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = [f.result() for f in [pool.submit(engine.composite_roi, p, t, m) for _ in range(16)]]
        assert all(np.array_equal(r, expected) for r in results)
