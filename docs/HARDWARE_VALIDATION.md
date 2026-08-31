# Hardware Validation Ledger

This file is the per-change validation ledger required by the dual-GPU policy.
The RTX 3060 and RTX 4070 are never combined into one generic benchmark. The
detailed historical matrix is in [`HARDWARE_VALIDATION_MATRIX.md`](HARDWARE_VALIDATION_MATRIX.md).

## Current session — 2026-08-31

| Field | Current session |
|---|---|
| Current GPU | NVIDIA GeForce RTX 4070, compute capability 8.9 |
| Total VRAM | 12,282 MiB |
| CUDA / PyTorch | 12.8 / 2.7.0+cu128 |
| Driver | 616.56 |
| GPU-sensitive change | **NO** — verification and documentation only |
| RTX 4070 test | **PASS** — existing baseline artifact and targeted tests re-verified; no application code changed |
| RTX 3060 test | **NOT TESTED** in this session |

The current 4070 probe had 11,106 MiB available at probe time. The locked
benchmark below was recorded earlier with driver 610.88; it is not relabeled as
a new run on driver 616.56.

## Required per-change table

| Commit | Change | RTX 3060 | RTX 4070 | Notes |
|---|---|---|---|---|
| `139e89125de032735a594b62f3e445f83548c691` | Phase 0 verification documentation | NOT TESTED (this session) | PASS (this session) | GPU-independent docs; no CUDA/provider/model code changed |
| `677385e49dddd9889be780d11fae52d8a07857fd` | Stable performance foundation under verification | PASS for recorded physical baseline; strict RSS gate FAIL | PASS for recorded locked baseline | Historical evidence is separate from this session’s test status; see matrix and progress document |
| `c2ba7224ab4be7edccaef0f09bd2f3dbb7140cca` | Phase 1 hardware/inference audit and 4070 benchmark | NOT TESTED (this session) | PASS (this session) | No application code changed; 3060 remains a separate required validation target |

## Recorded baseline evidence

| Metric | RTX 3060 | RTX 4070 |
|---|---:|---:|
| Baseline FPS | 4.53 mean (4.55 / 4.52; 4.33 superseded) | 9.62 |
| Peak / mean VRAM | 4,685 / 2,816 MB | 7,067 / 4,080.346 MB |
| Peak / mean RSS | 3.734 / 2.164 GB | 11.663 / 7.568 GB |
| Peak / mean CPU | 97.21% / 31.12% | 99.2% / 20.49% |
| Peak / mean GPU | 99.0% / 57.56% | 76.0% / 33.952% |
| Decode / encode | 451.13 / 314.14 FPS | 189.87 / 46.69 FPS |
| Quality / stability | 951 seen, 946 swapped, 0 wrong FaceSets; 600/600 | 856 seen, 850 swapped, 0 wrong FaceSets; 600/600 |

The 4070 baseline is the controlled `double/d4.mp4`, 1280×720, frames 0–600,
RealSwap + RealityUX + GPEN 256 Pro, TensorRT pools 2/2, 10 workers, and
hevc_nvenc output. The 3060 automatic run deliberately used its safe profile:
TensorRT disabled, no enhancer, XSeg-only mask, CPU decode, pools 0/0, and
guarded FP32 swap. Consequently these rows are not a like-for-like speed
comparison.

## Phase 1 RTX 4070 backend verification

Current ORT advertised `TensorrtExecutionProvider`, `CUDAExecutionProvider`,
and `CPUExecutionProvider`. Live resolution selected TRT→CUDA→CPU for auto and
TensorRT requests; CPU was the honest fallback for unavailable DirectML, ROCm,
and CoreML providers. The warm `roop.bench --profile quick --no-apply` pass
exited 0 in 177.7 s and measured 397.2 detector calls/s, 205.5 swap calls/s,
35.9 UltraMax calls/s, 227.9 XSeg calls/s, and 55.9 BiSeNet calls/s. Cold
engine-build evidence was 199 s for the swapper and 179 s for BiSeNet. The
first cold pass reported an XSeg invalid-throughput anomaly and exited 1; the
cached pass reproduced XSeg successfully. This is recorded as a benchmark
tooling limitation, not a reason to remove RealityUX.

The live cache-key probe confirmed that every requested TensorRT tuning knob
(workspace, partition iterations, builder level, auxiliary streams, and CUDA
graphs) changes the cache identity. LayerNorm FP32 fallback was also visible in
the TensorRT log. No default was changed and no dual-GPU validation claim is
made.

## Validation rule

For every future GPU-sensitive commit, append a row with the exact commit SHA,
date, GPU name, compute capability, VRAM, driver, CUDA, TensorRT, ONNX Runtime,
provider, precision, workload, benchmark values, failures, and fixes. A change
is **DUAL-GPU VALIDATED** only after physical PASS results are recorded for
both devices. A commit alone is never hardware validation.
