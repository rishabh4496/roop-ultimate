# Baseline Report: Sustained Throughput and Null Control

**Date:** 2026-10-11  
**Branch:** `perf/sustained-throughput` from `b3189690fcd1fb3049d6f7ee283fe6a1fcb76327`  
**Hardware:** RTX 4070 Desktop, 12,282 MiB VRAM, driver 616.56, CUDA 12.8, 31.7 GB RAM  
**Runtime:** Python 3.10.20, PyTorch 2.7.0+cu128, ONNX Runtime 1.23.2, TensorRT 10.9.0.34

## Workload

`double/d4.mp4`, 1280x720 at 30 fps, frames 0..600, sources `harjot,ashna`, 20 threads, live `app/config.yaml`, TensorRT provider, Hyperswap, Restore Ultra, DFL XSeg, SCRFD, and stabilization enabled. `ROOP_STAB_CHUNK_MB=1536` was pinned.

## Null control

| Metric | Null arm 1 | Null arm 2 |
|---|---:|---:|
| Render time | 67.85 s | 61.70 s |
| Render FPS | 8.84 | 9.72 |
| Pipeline FPS | 12.77 | 14.02 |
| Faces/frame | 1.12 | 1.12 |
| Faces/s | 14.3 | 15.7 |
| Peak VRAM | 7,823 MB | 7,690 MB |
| Peak GPU temperature | 60 C | 59 C |
| Peak GPU utilization | 81% | 96% |
| Harjot swaps | 453/481 | 453/481 |
| Ashna swaps | 241/246 | 241/246 |
| Wrong facesets | 0 | 0 |

## Windowed pipeline FPS, null arm 2

| Frames | FPS | Faces/frame | Faces/s | Interpretation |
|---|---:|---:|---:|---|
| 1..100 | 100.18 | 0.00 | 0.0 | Empty-face and decode-dominated window |
| 101..200 | 9.65 | 1.31 | 12.6 | Two-person workload |
| 201..300 | 15.78 | 1.66 | 26.2 | Higher face work rate |
| 301..400 | 17.02 | 1.72 | 29.3 | Higher face work rate |
| 401..500 | 12.71 | 1.02 | 13.0 | Mixed workload |
| 501..600 | 21.32 | 1.00 | 21.3 | One-face workload recovery |

## Interpretation

The fixture reproduces a large apparent FPS fall after faces appear and a later recovery when face density falls. It therefore confirms that content complexity is a major factor. It does not by itself prove a monotonic performance leak. GPU temperature and clocks did not support thermal throttling.

The original sidecar process-RSS fields are excluded because it sampled only the benchmark runner PID rather than the complete process tree. The corrected sidecar in `tools/telemetry_sidecar.py` aggregates child processes for future runs. GPU and system telemetry from these arms remains usable.

Raw logs and CSVs are stored in this directory. The structured comparison is `benchmark_results.csv`.
