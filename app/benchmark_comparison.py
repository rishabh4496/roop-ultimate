"""Benchmark the legacy transport against the optimized in-memory transport.

Examples from ``app``::

    python benchmark_comparison.py --video input.mp4 --frames 300
    python benchmark_comparison.py --model models/inswapper_128.onnx --batch-size 8

The video comparison isolates decode/pipe transport and therefore does not
pretend to be a face-swap quality benchmark.  The model comparison measures
the same ORT feed through ordinary ``session.run`` and the new adaptive CUDA
I/O-bound runner.  Both reports include latency, FPS, normalized CPU load, GPU
engine utilization, and whole-device VRAM from ``nvidia-smi`` when available.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from roop.optimized_processor import (
    FFmpegRawReader,
    OnnxBatchRunner,
    VideoSpec,
    create_onnx_session,
    probe_video,
)

try:
    import psutil
except ImportError:  # pragma: no cover - dependency is part of the app venv
    psutil = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - model benchmark can be skipped
    ort = None  # type: ignore[assignment]


@dataclass
class ResourceSample:
    """One resource sample from the process and the active GPU."""

    cpu_percent: float = 0.0
    gpu_percent: Optional[float] = None
    encoder_percent: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None


class ResourceSampler:
    """Low-overhead sampler that degrades cleanly when nvidia-smi is absent."""

    def __init__(self, period: float = 0.10, device_id: int = 0) -> None:
        self.period = max(0.02, float(period))
        self.device_id = max(0, int(device_id))
        self.samples: List[ResourceSample] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="benchmark-resource-sampler", daemon=True)
        self.process = psutil.Process(os.getpid()) if psutil is not None else None

    def _gpu_sample(self) -> ResourceSample:
        """Query utilization and memory in one nvidia-smi call."""

        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.encoder,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            f"--id={self.device_id}",
        ]
        try:
            raw = subprocess.check_output(
                command, text=True, stderr=subprocess.DEVNULL, timeout=2.0
            ).strip()
            values = [float(part.strip()) for part in raw.splitlines()[0].split(",")]
            return ResourceSample(
                gpu_percent=values[0],
                encoder_percent=values[1],
                memory_used_mb=values[2],
                memory_total_mb=values[3],
            )
        except (OSError, IndexError, ValueError, subprocess.SubprocessError):
            return ResourceSample()

    def _run(self) -> None:
        if self.process is not None:
            self.process.cpu_percent(None)
        while not self.stop_event.is_set():
            gpu = self._gpu_sample()
            if self.process is not None:
                cpu = float(self.process.cpu_percent(None))
            else:
                cpu = 0.0
            logical = float(psutil.cpu_count(logical=True) or 1) if psutil is not None else 1.0
            gpu.cpu_percent = cpu / logical
            self.samples.append(gpu)
            self.stop_event.wait(self.period)

    def start(self) -> None:
        """Start sampling."""

        self.thread.start()

    def finish(self) -> Dict[str, Optional[float]]:
        """Stop sampling and return mean/peak resource values."""

        self.stop_event.set()
        self.thread.join(timeout=3.0)
        if not self.samples:
            return {
                "cpu_load_avg_pct": None,
                "cpu_load_peak_pct": None,
                "gpu_utilization_avg_pct": None,
                "gpu_utilization_peak_pct": None,
                "encoder_utilization_avg_pct": None,
                "vram_used_peak_mb": None,
                "vram_total_mb": None,
            }

        def values(name: str) -> List[float]:
            return [
                float(value)
                for sample in self.samples
                for value in [getattr(sample, name)]
                if value is not None
            ]

        cpu = values("cpu_percent")
        gpu = values("gpu_percent")
        encoder = values("encoder_percent")
        used = values("memory_used_mb")
        total = values("memory_total_mb")
        return {
            "cpu_load_avg_pct": round(statistics.fmean(cpu), 2) if cpu else None,
            "cpu_load_peak_pct": round(max(cpu), 2) if cpu else None,
            "gpu_utilization_avg_pct": round(statistics.fmean(gpu), 2) if gpu else None,
            "gpu_utilization_peak_pct": round(max(gpu), 2) if gpu else None,
            "encoder_utilization_avg_pct": round(statistics.fmean(encoder), 2) if encoder else None,
            "vram_used_peak_mb": round(max(used), 2) if used else None,
            "vram_total_mb": round(max(total), 2) if total else None,
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return a deterministic nearest-rank percentile."""

    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return float(ordered[index])


def timed(
    label: str,
    function: Callable[[], int],
    warmup: int,
    iterations: int,
    device_id: int = 0,
) -> Dict[str, Any]:
    """Measure a callable and add resource telemetry."""

    for _ in range(max(0, int(warmup))):
        function()
    sampler = ResourceSampler(device_id=device_id)
    sampler.start()
    latencies: List[float] = []
    completed = 0
    try:
        for _ in range(max(1, int(iterations))):
            started = time.perf_counter()
            completed += int(function())
            latencies.append((time.perf_counter() - started) * 1000.0)
    finally:
        resources = sampler.finish()
    elapsed = sum(latencies) / 1000.0
    result: Dict[str, Any] = {
        "label": label,
        "iterations": len(latencies),
        "items": completed,
        "latency_mean_ms": round(statistics.fmean(latencies), 3),
        "latency_p50_ms": round(_percentile(latencies, 0.50), 3),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 3),
        "fps": round(completed / elapsed, 3) if elapsed > 0.0 else 0.0,
    }
    result.update(resources)
    return result


def _feed_for_session(session: Any, batch_size: int) -> Dict[str, np.ndarray]:
    """Create deterministic inputs at each model's declared dtype/shape."""

    generator = np.random.default_rng(20260914)
    feeds: Dict[str, np.ndarray] = {}
    for meta in session.get_inputs():
        shape: List[int] = []
        for axis, dimension in enumerate(meta.shape):
            if axis == 0:
                shape.append(max(1, int(batch_size)))
            elif isinstance(dimension, int) and dimension > 0:
                shape.append(int(dimension))
            else:
                shape.append(1)
        type_name = str(getattr(meta, "type", "tensor(float)"))
        dtype = {
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(uint8)": np.uint8,
        }.get(type_name, np.float32)
        if np.issubdtype(dtype, np.integer):
            value = generator.integers(0, 2, size=shape, dtype=dtype)
        else:
            value = generator.standard_normal(shape).astype(dtype)
        feeds[meta.name] = np.ascontiguousarray(value)
    return feeds


def compare_onnx(args: argparse.Namespace) -> Dict[str, Any]:
    """Compare ordinary ORT transport with adaptive CUDA I/O binding."""

    if ort is None:
        raise RuntimeError("onnxruntime is not installed")
    model = Path(args.model)
    if not model.is_file():
        raise FileNotFoundError(model)
    session = create_onnx_session(model)
    batch = max(1, int(args.batch_size))
    feeds = _feed_for_session(session, batch)
    try:
        session.run(None, feeds)
    except Exception:
        if batch != 1:
            batch = 1
            feeds = _feed_for_session(session, batch)
        session.run(None, feeds)
    runner = OnnxBatchRunner(
        session=session,
        requested_batch=batch,
        device_id=int(args.device_id),
    )
    before = timed(
        "legacy session.run",
        lambda: batch if session.run(None, feeds) else 0,
        args.warmup,
        args.iterations,
        device_id=int(args.device_id),
    )
    after = timed(
        "optimized I/O binding + governor",
        lambda: batch if runner.run(feeds, batch_size=batch) else 0,
        args.warmup,
        args.iterations,
        device_id=int(args.device_id),
    )
    return {
        "kind": "onnx",
        "model": str(model),
        "batch_size": batch,
        "providers": runner.active_providers,
        "before": before,
        "after": after,
        "speedup_factor": round(after["fps"] / before["fps"], 4)
        if before["fps"]
        else None,
        "governor": {
            "hard_cap_mb": runner.governor.hard_cap_mb,
            "batch_limit": runner.governor.batch_limit,
            "oom_events": runner.governor.oom_events,
        },
    }


def _legacy_video_count(path: str, frame_limit: int) -> int:
    """Read frames through the old OpenCV capture path."""

    capture = cv2.VideoCapture(path)
    count = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        while count < frame_limit:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            count += 1
        return count
    finally:
        capture.release()


def _optimized_video_count(path: str, spec: VideoSpec, frame_limit: int) -> int:
    """Read frames through the one-process raw FFmpeg path."""

    count = 0
    with FFmpegRawReader(path, spec=spec) as reader:
        while count < frame_limit:
            frame = reader.read()
            if frame is None:
                break
            count += 1
    return count


def compare_video(args: argparse.Namespace) -> Dict[str, Any]:
    """Compare decode throughput without claiming swap/compositor speed."""

    path = str(args.video)
    spec = probe_video(path)
    limit = max(1, int(args.frames))
    before = timed(
        "legacy cv2.VideoCapture decode",
        lambda: _legacy_video_count(path, limit),
        warmup=0,
        iterations=1,
        device_id=int(args.device_id),
    )
    after = timed(
        "optimized FFmpeg rawvideo pipe decode",
        lambda: _optimized_video_count(path, spec, limit),
        warmup=0,
        iterations=1,
        device_id=int(args.device_id),
    )
    return {
        "kind": "video_decode",
        "video": path,
        "video_spec": {
            "width": spec.width,
            "height": spec.height,
            "fps": spec.fps,
            "frame_count": spec.frame_count,
        },
        "note": "Transport-only comparison; face inference, compositing, and encode are not included.",
        "before": before,
        "after": after,
        "speedup_factor": round(after["fps"] / before["fps"], 4)
        if before["fps"]
        else None,
    }


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="ONNX model for transport benchmarking")
    parser.add_argument("--video", type=Path, help="Video for decode-pipeline benchmarking")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--json", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    """Run requested comparisons and print a machine-readable report."""

    args = parse_args()
    if args.model is None and args.video is None:
        raise SystemExit("provide --model, --video, or both")
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": os.name,
        "results": [],
    }
    if args.model is not None:
        report["results"].append(compare_onnx(args))
    if args.video is not None:
        report["results"].append(compare_video(args))
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
