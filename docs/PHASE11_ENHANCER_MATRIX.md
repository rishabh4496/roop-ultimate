# Phase 11 enhancer performance matrix

This is the complete matrix schema for the 29 source-discovered enhancement,
restoration, colorization, and super-resolution paths. The benchmark stores
one copy per hardware profile; rows are never averaged across GPUs. `pending`
means that no honest measurement for that exact path/profile is available yet.
The current host is RTX 4070-capable; RTX 3060 rows remain pending until that
physical device is attached. The selected-path benchmark can fill its row in
`benchmark_results.enhancer_matrix`; all other rows remain pending rather than
inheriting another model's result.

| Enhancer | Backend | Precision | Input | Output | Batch | Contexts | Streams | FPS | Latency | VRAM | CPU | Notes |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|
| CodeFormer | ONNX Runtime | FP32 / mixed candidate | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | `Enhance_CodeFormer`; finite guard |
| CodeFormer FP16 | ONNX Runtime | FP16 graph / FP32 post | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | separate FP16 graph; finite guard |
| DMDNet | PyTorch | FP32 | 512x512 + landmarks | 512x512→input | ref-dependent | 1 | torch default | pending | pending | pending | pending | `Enhance_DMDNet`; specialized face metadata required |
| GFPGAN | ONNX Runtime | forced FP32 TRT | 512x512 | 512x512→input | 1 | 1 | provider | pending | pending | pending | pending | forced-FP32 and collapse guard retained |
| GPEN 256 | ONNX Runtime | provider policy | 256x256 | 256x256→input | 1 | 1 | provider | pending | pending | pending | pending | `Enhance_GPEN`; independent 256 measurement |
| GPEN 512 | ONNX Runtime | provider policy | 512x512 | 512x512→input | 1 | 1 | provider | pending | pending | pending | pending | classic GPEN-BFR-512 |
| GPEN 1024 | ONNX Runtime | FP32 TRT required | 1024x1024 | 1024x1024→input | 1 | 1 | provider | pending | pending | pending | pending | FP32 fallback preserved |
| GPEN 2048 | ONNX Runtime | FP32 TRT required | 2048x2048 | 2048x2048→input | 1 | 1 | provider | pending | pending | pending | pending | FP32 fallback preserved |
| GPEN 256 Pro | ONNX Runtime + torch post | provider policy / FP32 post | 256x256 | 256 or 512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | GPU and CPU texture paths measured separately |
| GPEN Realistic 256 | ONNX Runtime | provider policy | 256x256 | 256x256→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | fast/soft tier |
| GPEN Realistic 512 | ONNX Runtime | provider policy | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | sharper paste-resolution tier |
| RestoreFormer++ | ONNX Runtime | provider policy | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | per-slot binding/context |
| UltraMax | ONNX Runtime + torch post | CodeFormer FP16 / FP32 post | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | lean host path; texture restore off by default |
| KEEP (sidecar) | isolated HTTP sidecar | sidecar-defined | PNG face | sidecar image | 1 | sidecar-defined | sidecar | pending | pending | pending | pending | optional sidecar; pass-through on failure |
| Real-ESRGAN x2 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x2 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | TRT opt-in only |
| Real-ESRGAN x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | TRT opt-in only |
| Real-ESRGAN Anime x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | anime 6B export |
| UltraSharp x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | TensorRT not forced |
| LSiDIR x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | model-specific tile benchmark required |
| Clear Reality x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | model-specific tile benchmark required |
| SPAN x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | batch-1 baseline until A/B |
| Compact ESRGAN x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | SRVGGNetCompact export |
| NOMOS 8K x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | model-specific tile benchmark required |
| DeOldify artistic | ONNX Runtime | provider policy | 256x256 grayscale | source LAB merge | 1 | 1 | provider | pending | pending | pending | pending | adjacent colorization path |
| DeOldify stable | ONNX Runtime | provider policy | 256x256 grayscale | source LAB merge | 1 | 1 | provider | pending | pending | pending | pending | adjacent colorization path |
| LANCZOS x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | classical resampling |
| FSR x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | Lanczos + CAS |
| SPLINE x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | classical resampling |
| SINC x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | classical resampling |

## Quality and acceptance fields

Each completed row must additionally record pixel max/mean difference against
its batch-1 or reference output, PSNR/SSIM where meaningful, identity metric if
available, output range, non-finite count, collapse decision, visible artifact
review, and stability over a sustained run. The acceptance classification is
one of: **A beneficial on both**, **B RTX 3060-specific**, **C RTX 4070-specific**,
**D neutral**, **E regression on one GPU**, or **F unsafe/rejected**.

The current implementation already records the guards and provider policy in
`docs/PHASE11_ENHANCER_INVENTORY.md`; missing performance/quality cells are
deliberately not filled from a different enhancer or GPU.

## RTX 4070 benchmark table

This is the first physical Phase 11 pass on the available RTX 4070 (Ada,
SM 8.9, 11.994 GB total VRAM). These are independent short warmed runs on a
synthetic 256-gradient face/frame, so they are implementation-path evidence,
not a claim about a complete video workload. VRAM, CPU utilization, quality
metrics, and sustained stability remain pending where they were not sampled.
The face runs used one context and the frame runs used CUDA, tile 64, and
tile batch 1. TensorRT was not forced onto frame models.

| Enhancer | Backend | Precision | Input | Output | Batch | Contexts | Streams | FPS | Latency | VRAM | CPU | Notes |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| CodeFormer | TensorRT via ONNX Runtime | FP32 | 512x512 | 512x512 | 1 | 1 | provider | 27.53 | 36.32 ms | pending | pending | 4070 full `Run`; finite guard |
| CodeFormer FP16 | TensorRT via ONNX Runtime | FP16 graph | 512x512 | 512x512 | 1 | 1 | provider | 25.80 | 38.76 ms | pending | pending | 4070 full `Run`; slower in this pass, not promoted |
| DMDNet | PyTorch | FP32 | 512x512 + landmarks | 512x512 | pending | 1 | torch | pending | pending | pending | pending | requires real landmark/reference metadata |
| GFPGAN | CUDA via ONNX Runtime | FP32 | 512x512 | 512x512 | 1 | 1 | provider | 22.15 | 45.16 ms | pending | pending | forced-FP32 behavior and collapse guard retained |
| GPEN 256 | CUDA via ONNX Runtime | FP32 | 256x256 | 256x256 | 1 | 1 | provider | 12.27 | 81.52 ms | pending | pending | 4070 full `Run` |
| GPEN 512 | CUDA via ONNX Runtime | FP32 | 512x512 | 512x512 | 1 | 1 | provider | 12.65 | 79.05 ms | pending | pending | 4070 full `Run` |
| GPEN 1024 | CUDA via ONNX Runtime | FP32 | 1024x1024 | 1024x1024 | 1 | 1 | provider | 5.99 | 166.83 ms | pending | pending | FP32 fallback preserved |
| GPEN 2048 | CUDA via ONNX Runtime | FP32 | 2048x2048 | 2048x2048 | 1 | 1 | provider | 2.47 | 404.76 ms | pending | pending | FP32 fallback preserved |
| GPEN 256 Pro | CUDA via ONNX Runtime | FP32 post | 256x256 | 512x512 | 1 | 1 | provider | 79.81 | 12.53 ms | pending | pending | full path; isolated 4070 post GPU 582.67 FPS vs CPU 64.05 FPS |
| GPEN Realistic 256 | CUDA via ONNX Runtime | FP32 | 256x256 | 256x256 | 1 | 1 | provider | 89.00 | 11.24 ms | pending | pending | 4070 full `Run` |
| GPEN Realistic 512 | CUDA via ONNX Runtime | FP32 | 512x512 | 512x512 | 1 | 1 | provider | 12.93 | 77.32 ms | pending | pending | 4070 full `Run` |
| RestoreFormer++ | CUDA via ONNX Runtime | FP32 | 512x512 | 512x512 | 1 | 1 | provider | pending | pending | pending | pending | per-slot binding path; run not completed in this pass |
| UltraMax | TensorRT via ONNX Runtime | FP16 graph / FP32 post | 512x512 | 512x512 | 1 | 1 | provider | 33.00 | 30.32 ms | pending | pending | selected inference-stage result; lean texture-off path retained |
| KEEP (sidecar) | HTTP sidecar | sidecar-defined | PNG face | sidecar image | pending | sidecar | sidecar | pending | pending | pending | pending | optional sidecar unavailable for this pass |
| Real-ESRGAN x2 | CUDA via ONNX Runtime | FP32 | tiled 128x128 | 256x256 | 1 | 1 | provider | 4.52 | 221.42 ms | pending | pending | full `RunThreadSafe`; tile 64 |
| Real-ESRGAN x4 | CUDA via ONNX Runtime | FP32 | tiled 128x128 | 512x512 | 1 | 1 | provider | 3.42 | 292.67 ms | pending | pending | full `RunThreadSafe`; tile 64 |
| Real-ESRGAN Anime x4 | CUDA via ONNX Runtime | FP32 | tiled 128x128 | 512x512 | 1 | 1 | provider | 11.12 | 89.93 ms | pending | pending | full `RunThreadSafe`; tile 64 |
| UltraSharp x4 | CUDA via ONNX Runtime | FP32 | tiled 128x128 | 512x512 | 1 | 1 | provider | 2.41 | 415.28 ms | pending | pending | TRT not forced; full `RunThreadSafe` |
| LSiDIR x4 | CUDA via ONNX Runtime | FP32 | tiled 128x128 | 512x512 | 1 | 1 | provider | 3.43 | 291.56 ms | pending | pending | TRT not forced; full `RunThreadSafe` |
| Clear Reality x4 | CUDA via ONNX Runtime | FP32 | tiled 128x128 | 512x512 | 1 | 1 | provider | 62.40 | 16.03 ms | pending | pending | TRT not forced; full `RunThreadSafe` |
| SPAN x4 | CUDA via ONNX Runtime | FP32 | tiled 128x128 | 512x512 | 1 | 1 | provider | 61.42 | 16.28 ms | pending | pending | TRT not forced; full `RunThreadSafe` |
| Compact ESRGAN x4 | CUDA via ONNX Runtime | FP32 | tiled 128x128 | 512x512 | 1 | 1 | provider | 50.81 | 19.68 ms | pending | pending | TRT not forced; full `RunThreadSafe` |
| NOMOS 8K x4 | CUDA via ONNX Runtime | FP32 | tiled 128x128 | 512x512 | 1 | 1 | provider | 3.33 | 300.67 ms | pending | pending | TRT not forced; full `RunThreadSafe` |
| DeOldify artistic | CUDA via ONNX Runtime | provider policy | 256x256 | source LAB merge | 1 | 1 | provider | pending | pending | pending | pending | adjacent colorizer, not run in this pass |
| DeOldify stable | CUDA via ONNX Runtime | provider policy | 256x256 | source LAB merge | 1 | 1 | provider | pending | pending | pending | pending | adjacent colorizer, not run in this pass |
| LANCZOS x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | no physical benchmark in this pass |
| FSR x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | no physical benchmark in this pass |
| SPLINE x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | no physical benchmark in this pass |
| SINC x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | no physical benchmark in this pass |

## RTX 3060 benchmark table

No physical RTX 3060 was attached for this Phase 11 pass. Every cell below is
therefore explicitly pending; no RTX 4070 result is copied into this table.
The required follow-up is the same warmed full-path benchmark on the detected
RTX 3060, recording available/peak VRAM, host CPU, quality, and stability for
each row. The existing 3060 memory-safety evidence remains valid and separate.

| Enhancer | Backend | Precision | Input | Output | Batch | Contexts | Streams | FPS | Latency | VRAM | CPU | Notes |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| CodeFormer | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| CodeFormer FP16 | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| DMDNet | pending | pending | 512x512 + landmarks | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| GFPGAN | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| GPEN 256 | pending | pending | 256x256 | 256x256 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| GPEN 512 | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| GPEN 1024 | pending | pending | 1024x1024 | 1024x1024 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending; retain FP32 fallback |
| GPEN 2048 | pending | pending | 2048x2048 | 2048x2048 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending; retain FP32 fallback |
| GPEN 256 Pro | pending | pending | 256x256 | 512x512 | pending | pending | pending | pending | pending | pending | pending | benchmark neural, GPU post, CPU post separately |
| GPEN Realistic 256 | pending | pending | 256x256 | 256x256 | pending | pending | pending | pending | pending | pending | pending | compare against 512 on same workload |
| GPEN Realistic 512 | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | compare against 256 on same workload |
| RestoreFormer++ | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| UltraMax | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | preserve lean texture-off path until measured |
| KEEP (sidecar) | pending | pending | PNG face | sidecar image | pending | pending | pending | pending | pending | pending | pending | sidecar validation pending |
| Real-ESRGAN x2 | pending | pending | tiled dynamic | x2 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| Real-ESRGAN x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| Real-ESRGAN Anime x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| UltraSharp x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | do not force TRT without evidence |
| LSiDIR x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| Clear Reality x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| SPAN x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| Compact ESRGAN x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| NOMOS 8K x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| DeOldify artistic | pending | pending | 256x256 grayscale | source LAB merge | pending | pending | pending | pending | pending | pending | pending | adjacent colorizer validation pending |
| DeOldify stable | pending | pending | 256x256 grayscale | source LAB merge | pending | pending | pending | pending | pending | pending | pending | adjacent colorizer validation pending |
| LANCZOS x2 | pending | uint8 | source frame | x2 | serial | none | pending | pending | 0 | pending | 3060 CPU validation pending |
| FSR x2 | pending | uint8 | source frame | x2 | serial | none | pending | pending | 0 | pending | 3060 CPU validation pending |
| SPLINE x2 | pending | uint8 | source frame | x2 | serial | none | pending | pending | 0 | pending | 3060 CPU validation pending |
| SINC x2 | pending | uint8 | source frame | x2 | serial | none | pending | pending | 0 | pending | 3060 CPU validation pending |

## Current optimization classifications

- CodeFormer FP16: **D. neutral/not promoted on the available target** for
  this pass; its 4070 full-path result was slower than FP32, and dual-target
  evidence is still required.
- GPEN 1024 and 2048 FP16 forcing: **F. unsafe/rejected**; the existing FP32
  fallback remains active.
- Frame-model TensorRT forcing: **F. unsafe/rejected by default** until a
  model-specific A/B proves it safe and faster on both targets.
- UltraMax texture restoration: **D. neutral/inactive**; the measured lean
  path is retained and texture restore is not re-enabled merely because code
  exists.
- All other changes in this phase remain **pending dual-target acceptance**;
  a 4070-only improvement is not considered universal.

## GPEN 256 Pro isolated post-stage quality

On the available RTX 4070, the isolated 512px texture/sharpen stage measured
1.72 ms (582.67 FPS) on the GPU implementation and 15.61 ms (64.05 FPS) on
the CPU implementation. The GPU implementation is therefore the faster
post-stage on this target; it is not a universal promotion until the same A/B
is repeated on the RTX 3060. On a synthetic gradient, GPU versus CPU output
was max absolute difference 1/255, mean absolute difference 0.492/255,
PSNR 51.21 dB, and SSIM 0.9961. Both paths retain the existing finite,
collapse, and fallback guards. Neural-versus-post profiling on the 4070
indicates the CPU post-stage is the dominant bottleneck when selected; with
GPU post enabled, the complete measured path is 12.53 ms and no quality
promotion is made from this synthetic comparison alone.

## End-to-end acceptance summary: RTX 4070

The following fields are required for the final controlled video benchmark.
The available 4070 quick run supplies only the selected-path values shown;
fields marked pending were not sampled in that run and are not inferred from
enhancer-only timings.

| Baseline FPS | Final FPS | Improvement | Peak VRAM | Average VRAM | CPU utilization | GPU utilization | Decode throughput | Inference throughput | Enhancement throughput | Encode throughput | Latency | Stability | Output quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 73.96 standard-only (not a controlled baseline) | 31.63 enhanced quick run (not comparable) | pending controlled pair | pending | pending | pending | pending | pending Phase 11 fixture | 33.00 calls/s UltraMax selected stage | 33.00 calls/s UltraMax selected stage | pending | 30.32 ms selected stage | quick run completed; sustained pending | guards passed; metric suite pending |

## End-to-end acceptance summary: RTX 3060

No physical RTX 3060 was available for Phase 11. The existing small-card
memory/stability evidence is retained in the state files, but it is not a
substitute for this enhancer benchmark.

| Baseline FPS | Final FPS | Improvement | Peak VRAM | Average VRAM | CPU utilization | GPU utilization | Decode throughput | Inference throughput | Enhancement throughput | Encode throughput | Latency | Stability | Output quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pending physical 3060 fixture | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
