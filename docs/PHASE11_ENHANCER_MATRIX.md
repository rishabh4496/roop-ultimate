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

Re-measured 2026-08-29 by `app/tests/bench_phase11_enhancers.py`, which
supersedes the first pass for every face path. Run it to reproduce:

    env/Scripts/python.exe tests/bench_phase11_enhancers.py --json app/output/phase11_4070_rows.json

Device: RTX 4070, Ada, SM 8.9, 12282 MiB, driver 610.88, CUDA 12.8,
TensorRT 10.9.0.34, ONNX Runtime 1.23.2. Provider, swap model and pool size are
read live from `config.yaml` (`tensorrt`, `realswap`, pool 2) rather than from
CLI defaults, and the pipeline is brought up through `angle_bench.init_pipeline`
so TensorRT's DLLs are on PATH -- without that, ONNX Runtime falls back to CPU
without saying so. Input is a real aligned 256 face crop (s1.mp4 frame 300),
which is what `realswap` hands the enhancer; the first pass used a synthetic
gradient, on which every texture and clarity operator measures as doing nothing.
Three timed rounds of 30 calls after 8 warm calls; each path is
Initialize -> warm -> timed -> Release so peak VRAM stays bounded and rows do
not contend.

| Enhancer | Backend | Precision | Input | Output | Batch | Contexts | Streams | FPS | Latency | VRAM | CPU | Notes |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| CodeFormer | TensorRT via ONNX Runtime | FP32 | 256 crop -> 512 | 512x512 | 1 | 2 | provider | 28.99 | 34.50 +- 0.15 ms | 1060 MB | pending | full `Run`; finite guard |
| CodeFormer FP16 | TensorRT via ONNX Runtime | FP16 graph | 256 crop -> 512 | 512x512 | 1 | 2 | provider | 26.79 | 37.33 +- 0.04 ms | 861 MB | pending | full `Run`; FP32 is FASTER here, so FP16 stays unpromoted |
| DMDNet | PyTorch | FP32 | 512x512 + landmarks | 512x512 | pending | 1 | torch | pending | pending | pending | pending | requires real landmark/reference metadata; not run |
| GFPGAN | TensorRT via ONNX Runtime | forced FP32 | 256 crop -> 512 | 512x512 | 1 | 1 | provider | 24.00 | 41.66 +- 0.56 ms | 773 MB | pending | forced-FP32 and collapse guard retained; collapse check passed |
| GPEN 256 | TensorRT via ONNX Runtime | provider policy | 256x256 | 256x256 | 1 | 1 | provider | 210.65 | 4.75 +- 0.07 ms | 269 MB | pending | pastes at scale 1 -- cheapest, and barely above the unenhanced input |
| GPEN 512 | TensorRT via ONNX Runtime | provider policy | 256 crop -> 512 | 512x512 | 1 | 1 | provider | 33.72 | 29.66 +- 0.32 ms | 1357 MB | pending | full `Run` |
| GPEN 1024 | TensorRT via ONNX Runtime | forced FP32 | 256 crop -> 1024 | 1024x1024 | 1 | 1 | provider | 10.68 | 93.62 +- 0.35 ms | 1520 MB | pending | FP32 fallback preserved (FP16 overflows to black) |
| GPEN 2048 | TensorRT via ONNX Runtime | forced FP32 | 256 crop -> 2048 | 2048x2048 | 1 | 1 | provider | 3.99 | 250.43 +- 0.55 ms | 2984 MB | pending | FP32 fallback preserved |
| GPEN 256 Pro | TensorRT + torch CUDA post | provider policy | 256x256 | 512x512 | 1 | 2 | provider | 143.69 | 6.96 +- 0.45 ms | 738 MB | pending | **4.6x faster than the 32.2 ms recorded on this card 2026-08-25** -- the torch-CUDA post path (`_gpu_filter_core`) landed since |
| GPEN Realistic | TensorRT via ONNX Runtime | provider policy | 256 crop -> 512 | 512x512 | 1 | 2 | provider | 36.03 | 27.75 +- 0.02 ms | 2356 MB | pending | default tier is 512 (`ROOP_GPENR_SIZE`); matches the 27.5 ms recorded 2026-08-24 |
| RestoreFormer++ | TensorRT via ONNX Runtime | provider policy | 256 crop -> 512 | 512x512 | 1 | 2 | provider | 29.77 | 33.60 +- 0.08 ms | 1324 MB | pending | was pending in the first pass; now measured |
| UltraMax | TensorRT + CPU post | FP16 graph / FP32 post | 256 crop -> 512 | 512x512 | 1 | 2 | provider | 11.53 | 86.75 +- 0.78 ms | 855 MB | pending | **2.9x its own recorded cost**; see the split below |
| KEEP (sidecar) | HTTP sidecar | sidecar-defined | PNG face | sidecar image | pending | sidecar | sidecar | pending | pending | pending | pending | optional sidecar unavailable for this pass |

### The first pass's face rows were wrong, and how that is known

Nine of the eleven paths above reproduce this repository's own independently
recorded measurements on this same card (CLAUDE.md, 2026-08-24): GFPGAN 41.66 vs
41.7, CodeFormer FP16 37.33 vs 37.9, GPEN Realistic 27.75 vs 27.5, GPEN 256 4.75
vs 5.3. Against that agreement, the first pass's GPEN family is 2.6x to 14x
pessimistic:

| path | first pass | re-measured | this repo's 08-24 record |
|---|---:|---:|---|
| GPEN 256 | 81.52 ms | **4.75 ms** | 5.3 ms |
| GPEN 512 | 79.05 ms | **29.66 ms** | -- |
| GPEN Realistic 512 | 77.32 ms | **27.75 ms** | 27.5 ms |
| GPEN 1024 | 166.83 ms | **93.62 ms** | -- |
| GPEN 2048 | 404.76 ms | **250.43 ms** | -- |

Two internal contradictions in the first pass point the same way without needing
the outside record at all. Its GPEN 256 (81.52 ms) and its GPEN 512 (79.05 ms)
land within 3% of each other, which a 4x change in pixels cannot do on real GPU
execution; and its GPEN 256 (81.52 ms) and its GPEN Realistic 256 (11.24 ms) run
the same 256 network 7x apart. Its GPEN rows are also labelled
`CUDA via ONNX Runtime` while its CodeFormer and UltraMax rows are labelled
`TensorRT`, so those rows did not measure the provider `config.yaml` selects.

**The frame super-resolution rows from that same pass are therefore not carried
forward as measured.** They were produced by the same harness in the same run,
were never committed, and cannot be re-run. They are listed below as pending
re-measurement rather than quoted, because a pass that got its face rows wrong
by up to 14x has not earned trust on its frame rows.

| Enhancer | Backend | first-pass figure (NOT carried forward) | status |
|---|---|---:|---|
| Real-ESRGAN x2 | CUDA via ONNX Runtime | 221.42 ms | pending re-measurement |
| Real-ESRGAN x4 | CUDA via ONNX Runtime | 292.67 ms | pending re-measurement |
| Real-ESRGAN Anime x4 | CUDA via ONNX Runtime | 89.93 ms | pending re-measurement |
| UltraSharp x4 | CUDA via ONNX Runtime | 415.28 ms | pending re-measurement |
| LSiDIR x4 | CUDA via ONNX Runtime | 291.56 ms | pending re-measurement |
| Clear Reality x4 | CUDA via ONNX Runtime | 16.03 ms | pending re-measurement |
| SPAN x4 | CUDA via ONNX Runtime | 16.28 ms | pending re-measurement |
| Compact ESRGAN x4 | CUDA via ONNX Runtime | 19.68 ms | pending re-measurement |
| NOMOS 8K x4 | CUDA via ONNX Runtime | 300.67 ms | pending re-measurement |
| DeOldify artistic / stable | CUDA via ONNX Runtime | not run | pending |
| LANCZOS / FSR / SPLINE / SINC x2 | CPU/ffmpeg | not run | pending |

### UltraMax: 57% of its cost is CPU eye post-processing added 2026-08-27

UltraMax was recorded at 28.68 ms on 2026-08-23 and 30.6 ms on 2026-08-24, and
was then 1.21x FASTER than the `Codeformer (fp16)` network it runs inside. It now
measures 86.75 ms, which is 2.32x SLOWER than that same network (37.33 ms).

The cause is code, not measurement. Six commits on 2026-08-27 (`07e6cd5`,
`142a285`, `45550aa`, `3965958`, `bbb8465`, `54d252b`) added
`_protect_swapped_eyes` and `_rebalance_eye_detail`, plus an unconditional
`_STRUCTURE_SHARPEN` unsharp -- a stack of full-frame 512 `cv2.GaussianBlur` /
`cvtColor` / float32 work on the host, per face. The two eye operators are gated
on `ROOP_ULTRAMAX_CHROMA`, which splits the cost exactly:

| UltraMax arm | ms/face | vs the network alone |
|---|---:|---|
| default (eye operators ON) | 86.81 +- 0.56 | +49.5 ms |
| `ROOP_ULTRAMAX_CHROMA=1.0` (eye operators SKIPPED) | 37.27 +- 0.11 | 37.33 ms `Codeformer (fp16)` |

So `_protect_swapped_eyes` + `_rebalance_eye_detail` cost **49.5 ms/face** --
57% of the processor's total, and 1.33x the entire neural network. With them
skipped UltraMax returns to being the CodeFormer FP16 network almost exactly, as
its own module docstring describes.

This is not called a defect here: it is deliberate quality work, and none of the
numbers above measure whether it improves the picture. It is called out because
it is **host** cost, and the acceptance classification depends on a host the
RTX 3060 does not have. This machine has 24 physical / 32 logical cores; the
RTX 3060 Laptop target has 14 physical / 20 logical, and already runs one worker
under the sub-7 GB policy. 49.5 ms/face of CPU work does not scale with the GPU
and cannot be assumed neutral there.

**Classification: pending.** It is not D (neutral) and it is not C
(RTX 4070-specific) until the 3060 measures it. The candidate remedy -- porting
the two operators to the torch-CUDA path that GPEN 256 Pro already uses, which is
what took that processor from 32.2 ms to 6.96 ms -- is recorded as a lead, not
applied, because it would change the rendered picture and has no quality
evidence behind it yet.

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
- UltraMax periocular post-processing (`_protect_swapped_eyes`,
  `_rebalance_eye_detail`, added 2026-08-27): **PENDING dual-target
  classification, and it cannot be D.** Re-measured 2026-08-29 it costs
  49.5 ms/face on the host -- 57% of the processor and 1.33x its own neural
  network -- taking UltraMax from 1.21x FASTER than `Codeformer (fp16)` to
  2.32x slower. Host cost does not scale with the GPU, and the RTX 3060 Laptop
  target has 14 physical cores against this machine's 24 and already runs one
  worker under the sub-7 GB policy. Lead, not applied: GPEN 256 Pro's
  torch-CUDA post path took that processor from 32.2 ms to 6.96 ms on this
  card, and these two operators are the same class of work.
- GPEN 256 Pro torch-CUDA post path: **C. RTX 4070-specific until the 3060
  repeats it**, but a large one -- 32.2 ms (2026-08-25, this card) to 6.96 ms
  measured 2026-08-29, still pasting at 512.
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
GPU post enabled, the complete measured path is **6.96 +- 0.45 ms**
(re-measured 2026-08-29 on a real 256 face crop through the production
TensorRT provider; the 12.53 ms previously recorded here came from the
superseded first pass). No quality promotion is made from the synthetic
gradient comparison alone.

## End-to-end acceptance summary: RTX 4070

The following fields are required for the final controlled video benchmark.
The available 4070 quick run supplies only the selected-path values shown;
fields marked pending were not sampled in that run and are not inferred from
enhancer-only timings.

| Baseline FPS | Final FPS | Improvement | Peak VRAM | Average VRAM | CPU utilization | GPU utilization | Decode throughput | Inference throughput | Enhancement throughput | Encode throughput | Latency | Stability | Output quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 73.96 standard-only (not a controlled baseline) | 31.63 enhanced quick run (not comparable) | pending controlled pair | pending | pending | pending | pending | pending Phase 11 fixture | 11.53 calls/s UltraMax selected stage | 11.53 calls/s UltraMax selected stage | pending | 86.75 +- 0.78 ms selected stage | quick run completed; sustained pending | guards passed; metric suite pending |

## End-to-end acceptance summary: RTX 3060

No physical RTX 3060 was available for Phase 11. The existing small-card
memory/stability evidence is retained in the state files, but it is not a
substitute for this enhancer benchmark.

| Baseline FPS | Final FPS | Improvement | Peak VRAM | Average VRAM | CPU utilization | GPU utilization | Decode throughput | Inference throughput | Enhancement throughput | Encode throughput | Latency | Stability | Output quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pending physical 3060 fixture | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
