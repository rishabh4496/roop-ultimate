"""Verify strict TensorRT execution, cache generation, and GPU throughput.

Run from ``app`` for a dynamic-batch export::

    python verify_trt_fps.py --model models/hififace_unofficial_256.onnx \
        --batch-size 8 --warmup 10 --iterations 100

The command intentionally refuses the repository's static batch-one
``inswapper_128.onnx`` export.  Use a dynamic-batch re-export for this test;
the legacy processor remains available for models that cannot be re-exported.
Only TensorRT engine/timing cache files are written.  Frames are never dumped
to disk by this benchmark.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from roop.optimized_processor import TrtOnnxBatchRunner
from roop.trt_session_builder import (
    Shape,
    TensorRTSessionConfig,
    build_tensorrt_session,
    derive_profile_shapes,
)

try:
    import torch
except ImportError as error:  # pragma: no cover - environment dependent
    raise SystemExit("verify_trt_fps.py requires the CUDA PyTorch package") from error

try:
    import psutil
except ImportError:  # pragma: no cover - optional telemetry
    psutil = None  # type: ignore[assignment]


def _parse_shape(value: str) -> Tuple[str, Shape]:
    """Parse ``input_name=1x3x256x256`` CLI syntax."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("shape must use name=1x... syntax")
    name, encoded = value.split("=", 1)
    try:
        shape = tuple(int(item) for item in encoded.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from error
    if not name or not shape or any(item <= 0 for item in shape):
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}")
    return name, shape


def _parse_output_shape(value: str) -> Tuple[str, Shape]:
    """Parse a dynamic output allocation override."""

    return _parse_shape(value)


def _torch_dtype(ort_type: str) -> Any:
    """Map ORT's tensor type spelling to a Torch dtype."""

    mapping = {
        "tensor(float)": torch.float32,
        "tensor(float16)": torch.float16,
        "tensor(double)": torch.float64,
        "tensor(int64)": torch.int64,
        "tensor(int32)": torch.int32,
        "tensor(int16)": torch.int16,
        "tensor(int8)": torch.int8,
        "tensor(uint8)": torch.uint8,
        "tensor(bool)": torch.bool,
    }
    try:
        return mapping[str(ort_type)]
    except KeyError as error:
        raise TypeError(f"unsupported input type {ort_type!r}") from error


def _device_memory(device_id: int) -> Tuple[int, int]:
    """Return free/total CUDA bytes using PyTorch's allocator view."""

    with torch.cuda.device(device_id):
        free, total = torch.cuda.mem_get_info()
    return int(free), int(total)


class GpuSampler:
    """Sample process CPU and whole-GPU telemetry without blocking inference."""

    def __init__(self, device_id: int, period: float = 0.10) -> None:
        self.device_id = max(0, int(device_id))
        self.period = max(0.02, float(period))
        self.samples: List[Dict[str, float]] = []
        self.stop_event = threading.Event()
        self.process = psutil.Process(os.getpid()) if psutil is not None else None
        self.thread = threading.Thread(
            target=self._run,
            name="verify-trt-resource-sampler",
            daemon=True,
        )

    def _nvidia_smi(self) -> Dict[str, float]:
        """Read GPU/encoder utilization and VRAM from nvidia-smi."""

        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.encoder,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            f"--id={self.device_id}",
        ]
        try:
            raw = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).strip()
            values = [float(value.strip()) for value in raw.splitlines()[0].split(",")]
            return {
                "gpu_utilization_pct": values[0],
                "encoder_utilization_pct": values[1],
                "vram_used_mb": values[2],
                "vram_total_mb": values[3],
            }
        except (IndexError, OSError, ValueError, subprocess.SubprocessError):
            return {}

    def _run(self) -> None:
        if self.process is not None:
            self.process.cpu_percent(None)
        logical = float(psutil.cpu_count(logical=True) or 1) if psutil is not None else 1.0
        while not self.stop_event.is_set():
            sample = self._nvidia_smi()
            if self.process is not None:
                sample["cpu_load_pct"] = float(self.process.cpu_percent(None)) / logical
            self.samples.append(sample)
            self.stop_event.wait(self.period)

    def __enter__(self) -> "GpuSampler":
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3.0)

    def summary(self) -> Dict[str, Optional[float]]:
        """Return mean utilization and peak memory values."""

        def values(key: str) -> List[float]:
            return [float(sample[key]) for sample in self.samples if key in sample]

        result: Dict[str, Optional[float]] = {}
        for key in ("cpu_load_pct", "gpu_utilization_pct", "encoder_utilization_pct"):
            current = values(key)
            result[f"{key}_avg"] = round(statistics.fmean(current), 2) if current else None
            result[f"{key}_peak"] = round(max(current), 2) if current else None
        memory = values("vram_used_mb")
        total = values("vram_total_mb")
        result["vram_used_peak_mb"] = round(max(memory), 2) if memory else None
        result["vram_total_mb"] = round(max(total), 2) if total else None
        return result


def _cache_snapshot(path: Path) -> Dict[str, int]:
    """Count TensorRT cache artifacts before and after the benchmark."""

    if not path.exists():
        return {"engine": 0, "profile": 0, "timing": 0}
    files = [item for item in path.iterdir() if item.is_file()]
    return {
        "engine": sum(item.suffix.lower() == ".engine" for item in files),
        "profile": sum(item.suffix.lower() == ".profile" for item in files),
        "timing": sum(item.suffix.lower() == ".timing" for item in files),
    }


def _make_feeds(
    session: Any,
    optimal_shapes: Mapping[str, Shape],
    batch_size: int,
    device_id: int,
) -> Dict[str, Any]:
    """Create deterministic CUDA inputs matching the model's declared types."""

    generator = torch.Generator(device=f"cuda:{device_id}")
    generator.manual_seed(20260914)
    feeds: Dict[str, Any] = {}
    for meta in session.get_inputs():
        declared = tuple(int(value) for value in optimal_shapes[meta.name])
        shape = (int(batch_size),) + declared[1:]
        dtype = _torch_dtype(getattr(meta, "type", "tensor(float)"))
        if dtype in {torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool}:
            tensor = torch.zeros(shape, dtype=dtype, device=f"cuda:{device_id}")
        else:
            tensor = torch.randn(
                shape,
                dtype=dtype,
                device=f"cuda:{device_id}",
                generator=generator,
            )
        feeds[meta.name] = tensor
    return feeds


def benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a strict session, warm it, and report latency/FPS/telemetry."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    batch_size = int(args.batch_size)
    if batch_size < 2:
        raise ValueError("--batch-size must be >= 2; batch-one is intentionally rejected")
    model = Path(args.model)
    if not model.is_file():
        raise FileNotFoundError(model)
    shape_overrides = dict(args.shape or [])
    output_shapes = dict(args.output_shape or [])
    base = TensorRTSessionConfig.from_environment(
        cache_path=args.cache_dir,
        device_id=args.device_id,
    )
    config = replace(
        base,
        min_batch=min(int(base.min_batch), batch_size),
        opt_batch=min(max(2, int(base.opt_batch or batch_size)), batch_size),
        max_batch=max(int(base.max_batch or batch_size), batch_size),
        shape_overrides=shape_overrides,
    )
    minimum, optimal, maximum = derive_profile_shapes(
        model,
        min_batch=config.min_batch,
        opt_batch=config.opt_batch or batch_size,
        max_batch=config.max_batch or batch_size,
        shape_overrides=shape_overrides,
        require_dynamic_batch=config.require_dynamic_batch,
        allow_static_batch_one=config.allow_static_batch_one,
        default_spatial_shape=config.default_spatial_shape,
    )
    cache_path = Path(config.cache_path)
    before = _cache_snapshot(cache_path)
    session = build_tensorrt_session(model, config=config)
    runner = TrtOnnxBatchRunner(
        model,
        config=config,
        session=session,
        output_shapes=output_shapes,
    )
    feeds = _make_feeds(session, optimal, batch_size, args.device_id)
    for _ in range(max(0, int(args.warmup))):
        runner.run_gpu(feeds, batch_size=batch_size, pad_to_batch=False)
    torch.cuda.synchronize(args.device_id)
    latencies: List[float] = []
    with GpuSampler(args.device_id) as sampler:
        for _ in range(max(1, int(args.iterations))):
            started = time.perf_counter()
            runner.run_gpu(feeds, batch_size=batch_size, pad_to_batch=False)
            torch.cuda.synchronize(args.device_id)
            latencies.append((time.perf_counter() - started) * 1000.0)
    after = _cache_snapshot(cache_path)
    elapsed = sum(latencies) / 1000.0
    items = len(latencies) * batch_size
    return {
        "model": str(model),
        "providers": runner.active_providers,
        "device_id": args.device_id,
        "batch_size": batch_size,
        "iterations": len(latencies),
        "latency_mean_ms": round(statistics.fmean(latencies), 3),
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 3),
        "latency_p95_ms": round(sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3),
        "fps": round(items / elapsed, 3) if elapsed > 0.0 else 0.0,
        "batch_fps": round(len(latencies) / elapsed, 3) if elapsed > 0.0 else 0.0,
        "cache_before": before,
        "cache_after": after,
        "cache_artifacts_created": {
            key: max(0, after[key] - before[key]) for key in after
        },
        "cuda_memory_free_mb": round(_device_memory(args.device_id)[0] / 1024**2, 2),
        "cuda_memory_total_mb": round(_device_memory(args.device_id)[1] / 1024**2, 2),
        "profiles": {
            "min": minimum,
            "opt": optimal,
            "max": maximum,
        },
        "telemetry": sampler.summary(),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="dynamic-batch ONNX model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cache-dir", default="models/trt_cache")
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        action="append",
        help="optimal dynamic input shape, e.g. target=8x3x256x256",
    )
    parser.add_argument(
        "--output-shape",
        type=_parse_output_shape,
        action="append",
        help="dynamic output allocation, e.g. output=8x3x256x256",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the verifier and return a shell status."""

    try:
        result = benchmark(parse_args(argv))
    except Exception as error:
        print(f"TRT verification FAILED: {error}")
        return 2
    import json

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
