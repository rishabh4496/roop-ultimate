"""Measure conventional ORT execution against CUDA I/O binding.

Run from ``app`` after models have been downloaded:

    env/Scripts/python.exe benchmark_speed.py --model models/inswapper_128.onnx

The report deliberately labels the two arms by transport method rather than
claiming an end-to-end video gain: this isolates the per-face ONNX transfer
cost that RealSwap pays twice.  Use the same command on each physical GPU; it
never substitutes one device's measurements for another.
"""

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from roop.utilities import CudaOrtIOBinding


class GpuSampler:
    """Lightweight nvidia-smi sampler; unavailable systems report null metrics."""

    def __init__(self, period=0.10):
        self.period = float(period)
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name='benchmark_gpu_sampler', daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                raw = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'], text=True,
                    stderr=subprocess.DEVNULL, timeout=2).strip().splitlines()[0]
                util, memory = (float(part.strip()) for part in raw.split(',')[:2])
                self.samples.append((util, memory))
            except Exception:
                pass
            self.stop_event.wait(self.period)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if not self.samples:
            return {'gpu_utilization_avg_pct': None, 'gpu_utilization_peak_pct': None,
                    'gpu_memory_peak_mb': None}
        util, memory = zip(*self.samples)
        return {'gpu_utilization_avg_pct': round(statistics.fmean(util), 2),
                'gpu_utilization_peak_pct': round(max(util), 2),
                'gpu_memory_peak_mb': round(max(memory), 2)}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def measure(label, fn, iterations, warmup):
    for _ in range(warmup):
        fn()
    sampler = GpuSampler()
    sampler.start()
    timings = []
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - started) * 1000.0)
    finally:
        gpu = sampler.finish()
    elapsed_ms = sum(timings)
    result = {
        'label': label,
        'iterations': iterations,
        'fps': round(iterations * 1000.0 / elapsed_ms, 3),
        'latency_mean_ms': round(statistics.fmean(timings), 3),
        'latency_p50_ms': round(percentile(timings, 0.50), 3),
        'latency_p95_ms': round(percentile(timings, 0.95), 3),
    }
    result.update(gpu)
    return result


def build_feed(session, batch):
    feed = {}
    rng = np.random.default_rng(20260904)
    for meta in session.get_inputs():
        shape = []
        for axis, value in enumerate(meta.shape):
            shape.append(int(value) if isinstance(value, int) and value > 0 else (batch if axis == 0 else 1))
        feed[meta.name] = rng.standard_normal(shape, dtype=np.float32)
    return feed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/inswapper_128.onnx', type=Path)
    parser.add_argument('--iterations', default=120, type=int)
    parser.add_argument('--warmup', default=20, type=int)
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Use >1 only with a model/engine built for that batch shape.')
    parser.add_argument('--json', type=Path, help='Optional report path')
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f'model not found: {args.model}')

    available = ort.get_available_providers()
    providers = [name for name in ('TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                   'CPUExecutionProvider') if name in available]
    session = ort.InferenceSession(str(args.model), providers=providers)
    feed = build_feed(session, max(1, args.batch_size))
    binding = CudaOrtIOBinding(session)
    baseline = measure('standard session.run (before)', lambda: session.run(None, feed),
                       args.iterations, args.warmup)
    report = {
        'model': str(args.model), 'providers': session.get_providers(), 'batch_size': args.batch_size,
        'before': baseline,
    }
    if binding.enabled:
        after = measure('persistent CUDA I/O binding (after)', lambda: binding.run(feed),
                        args.iterations, args.warmup)
        report['after'] = after
        report['improvement_pct'] = round((after['fps'] / baseline['fps'] - 1.0) * 100.0, 2)
    else:
        report['after'] = None
        report['improvement_pct'] = None
        report['note'] = 'CUDA/TensorRT execution provider unavailable; I/O binding arm skipped.'
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
