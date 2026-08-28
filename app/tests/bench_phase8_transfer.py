"""Phase 8 CPU/GPU transfer and full-frame copy benchmark.

This is deliberately a small, dependency-light harness.  It measures the
operations that the production path actually uses at 1080p and 4K, plus the
guarded ``paste_upscale`` and writer paths.  It does not claim end-to-end FPS;
the full ``roop.bench --profile full`` run remains the end-to-end gate.

Run from ``app`` on each validation GPU:

    env\\Scripts\\python.exe tests\\bench_phase8_transfer.py
"""

import gc
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.procmgr_masking import MaskingMixin


class _MaskingBench(MaskingMixin):
    def __init__(self):
        self.options = SimpleNamespace(
            show_face_area_overlay=False,
            blend_ratio=1.0,
        )


class _Sink:
    def write(self, value):
        return len(value)


def _stats(values):
    return {
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(statistics.quantiles(values, n=20)[18], 3),
    }


def _timed(fn, rounds=32, warm=6):
    for _ in range(warm):
        fn()
    values = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000.0)
    return _stats(values)


def _frame_cases():
    return (
        (1920, 1080, 700, 250),
        (3840, 2160, 1500, 700),
    )


def measure_cpu_paths():
    bench = _MaskingBench()
    result = {}
    for width, height, x, y in _frame_cases():
        base = np.random.default_rng(8).integers(
            0, 256, (height, width, 3), dtype=np.uint8)
        fake = np.random.default_rng(9).integers(
            0, 256, (512, 512, 3), dtype=np.uint8)
        matrix = np.array([[1.0, 0.0, x], [0.0, 1.0, y]], dtype=np.float32)

        def frame_copy():
            base.copy()

        def retry_old():
            clockwise = np.rot90(base.copy(), 1, (1, 0))
            clockwise.copy()
            counter = np.rot90(base.copy())
            counter.copy()

        def retry_new():
            clockwise = np.rot90(base, 1, (1, 0))
            clockwise.copy()
            counter = np.rot90(base)
            counter.copy()

        def paste_inplace():
            target = base.copy()
            bench.paste_upscale(
                fake, fake, matrix, target, 1, [0, 0, 0, 0, 0, 0],
                inplace=True)

        def paste_legacy():
            target = base.copy()
            bench.paste_upscale(
                fake, fake, matrix, target, 1, [0, 0, 0, 0, 0, 0],
                inplace=False)

        def writer_bytes():
            _Sink().write(base.tobytes())

        def writer_view():
            _Sink().write(memoryview(base))

        result[f"{width}x{height}"] = {
            "bytes": int(base.nbytes),
            "frame_copy": _timed(frame_copy),
            "retry_old": _timed(retry_old),
            "retry_new": _timed(retry_new),
            "paste_legacy": _timed(paste_legacy),
            "paste_inplace": _timed(paste_inplace),
            "writer_tobytes": _timed(writer_bytes),
            "writer_memoryview": _timed(writer_view),
        }
        del base, fake
        gc.collect()
    return result


def measure_cuda_paths():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": type(exc).__name__}

    device = torch.device("cuda")
    result = {"available": True, "device": torch.cuda.get_device_name(0)}
    for width, height, _x, _y in _frame_cases():
        source = np.random.default_rng(11).random(
            (height, width, 3), dtype=np.float32)
        cpu = torch.from_numpy(source)

        def h2d():
            value = cpu.to(device)
            torch.cuda.synchronize()
            del value

        gpu = cpu.to(device)
        torch.cuda.synchronize()

        def d2h():
            value = gpu.cpu()
            torch.cuda.synchronize()
            del value

        row = {"h2d": _timed(h2d, rounds=12, warm=3),
               "d2h": _timed(d2h, rounds=12, warm=3),
               "bytes": int(source.nbytes)}
        try:
            pinned = torch.empty_like(cpu, pin_memory=True)

            def pinned_h2d():
                pinned.copy_(cpu)
                value = pinned.to(device, non_blocking=True)
                torch.cuda.synchronize()
                del value

            row["pinned_h2d_with_stage"] = _timed(
                pinned_h2d, rounds=12, warm=3)
        except Exception as exc:  # pinned allocation is platform dependent
            row["pinned_h2d_with_stage"] = {"unsupported": type(exc).__name__}
        result[f"{width}x{height}"] = row
        del gpu, cpu, source
        torch.cuda.empty_cache()
    return result


def main():
    print(json.dumps({"cpu": measure_cpu_paths(),
                      "cuda": measure_cuda_paths()}, indent=2))


if __name__ == "__main__":
    main()
