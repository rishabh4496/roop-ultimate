# Dual-GPU hardware validation matrix

This is the acceptance record for the two first-class NVIDIA targets. Results
are never averaged across GPUs. Runtime identity is detected from the active
stack and persisted with `hardware_profile_key`; free VRAM is telemetry and is
not used as a profile identity.

## Current validation state

- RTX 4070: physically available in earlier sessions (`nvidia-smi` reported
  12,282 MiB total and driver 610.88). NOT present in the 2026-08-29 (later)
  session recorded below.
- RTX 3060: **physically present and detected** as of 2026-08-29 (later
  session). Its acceptance rows nevertheless remain `pending` — not for want of
  hardware, but because the locked fixture is absent from that machine. See
  "RTX 3060 physical session" below. No 4070 value is copied into this table.

## RTX 3060 physical session — 2026-08-29 (later)

The laptop was probed rather than assumed. Detected profile:

| Field | Detected |
|---|---|
| GPU / architecture / CC | NVIDIA GeForce RTX 3060 Laptop GPU, Ampere, **8.6** |
| VRAM total / available | 6.0 GB / 4.586 GB at probe |
| Driver / CUDA | 616.56 / 12.8 |
| TensorRT / ONNX Runtime | 10.9.0.34 / 1.23.2 |
| Tensor Core modes | bf16, fp16 |
| FP16 / BF16 | supported |
| INT8 / FP8 | **not exposed** on this stack |
| NVDEC / NVENC | available — `av1/h264/hevc/vp9_cuvid`, `av1/h264/hevc_nvenc` |
| CPU | i7-12700H, 14 physical / 20 logical, 6 P + 8 E, affinity supported |
| CPU topology source | `windows-cpu-set-efficiency-class` (a real OS report, not inferred) |
| RAM | 15.797 GB |

Note the CPU row against the 4070's: Windows exposed **no** P/E topology on the
4070, which is why Gate D was deferred there. It *is* exposed here, so the 3060
is the target on which Gate D's CPU-distribution matrix can actually be run.

### RESOLVED: the fixture was replicated, and the baseline is MEASURED

The operator replicated the clip tree to `C:\pinokio\roop-keep\` mid-session.
`double/d4.mp4` fingerprints as **1280x720, 13305 frames**, matching the locked
identity exactly, so the run below is on the real Phase 2 workload. The
resolver prefers the hyphen root, so no flag or config edit was needed.

Command actually run, exactly as documented:

```bash
cd app
env/Scripts/python.exe tests/baseline_controlled.py --tag phase2_3060 --target "RTX 3060"
```

| Metric | RTX 3060 (measured) | RTX 4070 (locked) |
|---|---:|---:|
| End-to-end FPS | **4.33** | 9.62 |
| Frames / wall clock | 600 / 345.81 s | 600 / — |
| Mean frame latency | 413.16 ms | 233.34 ms |
| Decode FPS | 451.13 | 189.87 |
| Encode FPS | 314.14 (`hevc_nvenc`) | 46.69 (`hevc_nvenc`) |
| Peak / mean VRAM | 4,685 MB / 2,816 MB | 7,067 MB / 4,080 MB |
| Peak / mean RSS | **3.734 GB** / 2.164 GB | 11.663 GB / 7.568 GB |
| Peak / mean GPU util | 99.0% / 57.56% | 76.0% / 33.95% |
| Peak / mean CPU util | 97.21% / 31.12% | 99.2% / 20.49% |
| Peak P-core / E-core | 97.40% / 97.73% | 99.19% / 99.21% (inferred) |
| Peak power | 125.07 W | 118.09 W |
| Faces seen / swapped | 951 / 946 | 856 / 850 |
| **Wrong faceset** | **0** (644 attributed) | **0** (642 attributed) |
| Stability | 600/600, exit 0 | 600/600, exit 0 |

**This pair is NOT a like-for-like speed comparison, and the record now says so
in the artifact itself** (`comparable_to_locked_baseline: false`). See the stack
table below: the 3060 ran with no enhancer, no TensorRT, a degraded mask and CPU
decode — it is doing materially *less* work and is still 2.2x slower. Its GPU
sat at 99% peak / 57.6% mean against the 4070's 76% / 34%, so this target is
genuinely GPU-bound where the workstation was stage/CPU-bound.

Quality is not degraded by any of that: **zero wrong-faceset applications**
across 644 attributed swaps, matching the 4070.

Two observations worth carrying forward:

- **The strict `<2.5 GB` RSS gate still fails: peak 3.734 GB.** That is on the
  720p locked fixture and is *higher* than the 2.62–2.79 GB previously recorded
  on smaller clips, so the gate remains blocked and the earlier figures were not
  measured on this workload.
- **22.2% of frames (191 of 859) had no face detected at all.** The session logs
  list a "15% no-face rate" as an open item that could not be reproduced for
  want of the source clip. It reproduces here, on d4, at 22.2%.

### PREVIOUS BLOCKER (resolved above): the locked fixture was not on this machine

`PERFORMANCE_BASELINE.md` locks the baseline to `double/d4.mp4` at
**1280x720**. The laptop holds a clip also named `d4.mp4` — but it is
**854x480, 8310 frames**, i.e. the clip the session logs call `duo/d4.mp4`.
They are different videos sharing a filename.

A 40-frame smoke render on the local clip completed cleanly (rc 0, 3.36 fps,
peak RSS 2.965 GB, peak VRAM 3336 MB, peak GPU 97%), so the pipeline and the
whole harness path are working on this target. The number is nonetheless **not
a Phase 2 baseline row**: a smaller frame at a different face scale is a
different workload, and comparing it to 9.62 fps would be meaningless.

`tests/fixtures.py` now fingerprints the resolved clip and
`baseline_controlled.py` refuses to mark a mismatched run comparable
(`comparable_to_locked_baseline: false`), so this cannot be filed by accident.

**To close the row:** copy the 1280x720 `double/d4.mp4` to the laptop under
`<PINOKIO_HOME>/roop keep/` (or set `ROOP_CLIP_ROOT`), then run

```bash
cd app
env/Scripts/python.exe tests/baseline_controlled.py --tag phase2_3060 --target "RTX 3060"
```

### The 3060 runs a materially different stack, by design

The sub-7GB policy adapts the pipeline before it starts. These are
hardware-adaptive decisions, not defects, but they mean the two targets' rows
are **not like-for-like** and must never be presented as one comparison:

| Stage | RTX 4070 baseline | RTX 3060, automatic |
|---|---|---|
| Provider | TensorRT | **CUDA/CPU** (TRT disabled by the laptop RSS policy) |
| Enhancer | GPEN 256 Pro | **None** (stripped by the RSS gate) |
| Mask | RealityUX (XSeg + BiSeNet) | **XSeg only**, BiSeNet parser skipped |
| Decode | — | **CPU** (NVDEC → CPU by the RSS policy) |
| Pools | TRT 2 / detmask 2 | **0 / 0**, detector 1 |
| Swap precision | — | guarded **FP32** |

Any 3060 acceptance row must state which of these were in force.

## RTX 4070

The existing locked controlled baseline is recorded in
[`PERFORMANCE_BASELINE.md`](../PERFORMANCE_BASELINE.md). It is the available
4070 evidence, not a new measurement from this change.

| Metric | Result |
|---|---:|
| Baseline FPS | 9.62 |
| Final FPS | not applicable; no new default was promoted |
| Improvement | not applicable |
| Peak VRAM | 7,067 MB |
| Average VRAM | 4,080 MB |
| CPU utilization | 99.2% peak / 20.49% mean |
| GPU utilization | 76.0% peak / 33.95% mean |
| Decode throughput | 189.87 frames/s |
| Inference throughput | 28.6 faces/s |
| Enhancement throughput | 60.9 faces/s |
| Encode throughput | 46.69 frames/s, `hevc_nvenc` |
| Latency | 233.34 ms/frame worker time |
| Stability | 600/600 frames encoded, exit 0 |
| Output quality | 856 faces seen, 850 swapped, 0 wrong faceset applications |

## RTX 3060

Measured 2026-08-29 on the physical laptop against the locked fixture. Read
with the adaptive-downgrade table below: this row is the machine's real
automatic behaviour, not the 4070's stack running slower.

| Metric | Result |
|---|---:|
| Baseline FPS | **4.33** |
| Final FPS | not applicable; no new default was promoted |
| Improvement | not applicable |
| Peak VRAM | 4,685 MB |
| Average VRAM | 2,816 MB |
| CPU utilization | 97.21% peak / 31.12% mean |
| GPU utilization | 99.0% peak / 57.56% mean |
| Decode throughput | 451.13 frames/s (CPU decode) |
| Inference throughput | 947 swap calls at 127.56 ms/call |
| Enhancement throughput | not applicable — enhancer disabled by the RSS gate |
| Encode throughput | 314.14 frames/s, `hevc_nvenc` |
| Latency | 413.16 ms/frame worker time |
| Stability | 600/600 frames encoded, exit 0 |
| Output quality | 951 faces seen, 946 swapped, **0 wrong faceset** |
| Peak RSS | 3.734 GB — **strict `<2.5 GB` gate still FAILS** |

## Exact follow-up benchmark

Run the same workload and software configuration on each physical target;
only the explicit report label changes. No configuration file rewrite is
required:

```bash
cd app
python -m roop.bench --profile full --target "RTX 3060" --no-apply
python -m roop.bench --profile full --target "RTX 4070" --no-apply
```

For the end-to-end acceptance run, use the controlled fixture and distinct
tags:

```bash
cd app
python tests/baseline_controlled.py --tag dual_gpu_3060
python tests/baseline_controlled.py --tag dual_gpu_4070
```

Each result must retain the detected GPU name, architecture, compute
capability, total/available VRAM, CUDA, TensorRT, ONNX Runtime, driver, Tensor
Core/precision, NVDEC, NVENC, model identity, input/output resolution,
enhancer, batch, and workload characteristics. An optimization is accepted
globally only after both target rows have measured work/quality/stability
results; otherwise it is target-specific, neutral, regression, unsafe, or
pending.

The report assembler verifies that the target label matches the detected GPU
identity and marks rows with missing final metrics as `measured_partial`.
Therefore a mislabeled run or a stage-only measurement cannot become a complete
RTX 3060/RTX 4070 acceptance result.

## Phase 12 end-to-end matrix state

The reproducible post-inference matrix is implemented at
`app/tests/phase12_benchmark.py`. It measures stabilization OFF/ON, mask
OFF/ON, color processing OFF/ON, and a postprocess-heavy enhancer using the
real decode-to-encode wall clock. It writes separate tables for each target.

The current environment has completed the RTX 4070 render matrix in the
application environment. The RTX 3060 is physically unavailable here, so its
rows remain pending rather than being inferred from the 4070.

### RTX 4070 Phase 12 results

All rows use the same 600-frame `d4.mp4` fixture, TensorRT, and detected
hardware profile. Inference/enhancement throughput is faces per second;
decode/encode throughput is frames per second. Visual quality remains pending
manual review, while the automated identity checks passed for every row.

| Configuration | Baseline FPS | Final FPS | Improvement | Peak/avg VRAM MB | CPU mean % | GPU mean % | Decode FPS | Inference FPS | Enhance FPS | Encode FPS | Latency ms | Stability | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 16.06 | 16.06 | 0.00% | 4992 / 3319 | 20.493 | 32.879 | 229.89 | 27.80 | — | 67.26 | 156.72 | pass | pending visual review |
| stabilization ON | 16.06 | 13.88 | -13.57% | 4967 / 3359 | 19.871 | 32.464 | 133.04 | 27.67 | — | 43.83 | 135.50 | pass | pending visual review |
| mask ON | 16.06 | 14.90 | -7.22% | 5588 / 3437 | 20.646 | 33.177 | 184.05 | 24.49 | — | 62.96 | 223.24 | pass | pending visual review |
| color ON | 16.06 | 15.96 | -0.62% | 4968 / 3334 | 20.519 | 32.190 | 215.83 | 23.93 | — | 67.80 | 163.05 | pass | pending visual review |
| postprocess heavy | 16.06 | 7.55 | -52.99% | 6679 / 4058 | 22.529 | 37.617 | 158.31 | 19.66 | 6.07 | 43.20 | 456.67 | pass | pending visual review |

### RTX 3060 Phase 12 results

| Configuration | Baseline FPS | Final FPS | Improvement | Peak/avg VRAM | CPU | GPU | Decode | Inference | Enhance | Encode | Latency | Stability | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stabilization OFF / mask OFF / color OFF | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| stabilization ON | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| mask ON | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| color ON | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| postprocess heavy | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

The Phase 12 changes are therefore not classified as universally successful
yet: cross-target acceptance is **pending** until the RTX 3060 table is
measured. The heavy postprocess configuration is a measured RTX 4070
regression, not a global optimization claim.

Run these exact commands in the app environment when each machine is available:

```bash
cd app
python tests/phase12_benchmark.py --target "RTX 3060"
python tests/phase12_benchmark.py --target "RTX 4070"
```

## Phase 13 encoder/output matrix state

The reproducible codec and segment-lifecycle matrix is implemented at
`app/tests/phase13_benchmark.py`. It runs the real decode -> inference ->
enhancement -> encode path, records writer finalization separately, and keeps
the requested codec authoritative. Segment sizes are explicit benchmark arms;
automatic mode derives a duration-based chunk from the detected source FPS.
The single-segment path promotes the encoded part directly, while multi-part
outputs retain the manifest and concat path required for resume and crash
recovery.

The current environment completed the RTX 4070 matrix with 120 frames of the
controlled `d4.mp4` fixture. `libx264`, `h264_nvenc`, and `hevc_nvenc` were
available. Automated identity checks found zero wrong-face applications and
all outputs had 120/120 frames; visual quality still requires manual review.
The baseline FPS is the same 7.72 FPS render arm for each row.

### RTX 4070 Phase 13 results

| Codec | Segment frames | Final FPS | Improvement | Encode write/finalize s | Encode share | Encode FPS | Rotations | Peak/avg VRAM MB | CPU mean % | GPU mean % | Latency ms | Stability | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| libx264 | 50 | 7.72 | 0.00% | 1.49 / 0.62 | 1.82% | 80.54 | 3 | 5140 / 3308 | 28.325 | 29.921 | 410.38 | pass | pending visual review; wrong faceset=0 |
| libx264 | 120 | 7.96 | 3.11% | 0.81 / 0.00 | 0.71% | 148.15 | 1 | 5165 / 3324 | 28.755 | 29.630 | 390.44 | pass | pending visual review; wrong faceset=0 |
| h264_nvenc | 50 | 8.08 | 4.66% | 0.78 / 0.14 | 0.81% | 153.85 | 3 | 5523 / 3333 | 28.347 | 29.838 | 393.54 | pass | pending visual review; wrong faceset=0 |
| h264_nvenc | 120 | 8.16 | 5.70% | 0.38 / 0.00 | 0.34% | 315.79 | 1 | 5546 / 3338 | 27.927 | 29.660 | 394.81 | pass | pending visual review; wrong faceset=0 |
| hevc_nvenc | 50 | 8.35 | 8.16% | 0.77 / 0.10 | 0.78% | 155.84 | 3 | 5378 / 3324 | 28.314 | 29.118 | 387.87 | pass | pending visual review; wrong faceset=0 |
| hevc_nvenc | 120 | 9.03 | 16.97% | 0.34 / 0.00 | 0.31% | 352.94 | 1 | 5176 / 3209 | 26.545 | 27.890 | 345.19 | pass | pending visual review; wrong faceset=0 |

These are true end-to-end results, not isolated FFmpeg timings. Encoding is
not the limiting stage for this workload on the 4070: even the most expensive
50-frame arm used 2.11 seconds of writer time over a 115.72-second run. The
120-frame rotation reduced encoder lifecycle overhead and improved end-to-end
FPS in every tested codec. This is classified as beneficial on the RTX 4070,
but not yet universally accepted until the RTX 3060 is measured.

An additional synthetic writer lifecycle check encoded 300 frames with
`libx264` and measured 2.827 s at 60-frame chunks, 0.533 s at 150-frame
chunks, and 0.456 s at one 300-frame segment. This supports avoiding overly
frequent rotation while keeping the explicit chunk override for users who
need a smaller crash-loss window.

### RTX 3060 Phase 13 results

The RTX 3060 is unavailable in the current environment. No 4070 values are
copied into this table.

| Codec / segment arm | Baseline FPS | Final FPS | Improvement | Peak/avg VRAM | CPU | GPU | Decode | Inference | Enhancement | Encode | Latency | Stability | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| libx264 / 50 | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| libx264 / 120 | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| h264_nvenc / 50 | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| h264_nvenc / 120 | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hevc_nvenc / 50 | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hevc_nvenc / 120 | pending physical validation | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Run the exact pending-target benchmark when the laptop is available:

```bash
cd app
python tests/phase13_benchmark.py --target "RTX 3060" --codecs libx264,h264_nvenc,hevc_nvenc --segment-sizes 50,120
```

## Phase 14 runtime autotuner state

Phase 14 adds `RuntimeAutotuner` and the explicit retune driver
`app/tests/phase14_autotune.py`. The search is bounded to 12 short,
end-to-end candidates and staged across backend/precision, TensorRT
concurrency, batch, CPU threading, queue/buffer, and encoder choices. Its
score is end-to-end FPS with VRAM, RAM, instability, quality-regression, and
startup penalties. Cached profiles include the detected software stack,
hardware identity, model/workload characteristics, and selected configuration.

The search itself has not been claimed as a physical performance result on
either target yet; the existing Phase 13 4070 codec results remain evidence
for the encoder stage only. Unit coverage exercises both hardware tiers and
rejects faster-but-unstable or quality-regressing candidates.

### RTX 3060 Phase 14

| Result | Status |
|---|---|
| Runtime profile / candidate search | pending physical validation |
| Selected configuration / best FPS / baseline FPS | pending |
| VRAM / RAM / CPU / GPU | pending |

### RTX 4070 Phase 14

| Result | Status |
|---|---|
| Runtime profile / candidate search | pending physical autotune run |
| Selected configuration / best FPS / baseline FPS | pending |
| VRAM / RAM / CPU / GPU | pending |

Run the exact bounded retune on each available target; these commands do not
rewrite the saved configuration, and the 3060 command must not be run against
the 4070 as a substitute:

```bash
cd app
python tests/phase14_autotune.py --target "RTX 3060" --force
python tests/phase14_autotune.py --target "RTX 4070" --force
```

## Phase 15 runtime monitoring and adaptive control

Phase 15 adds opt-in rolling telemetry and a safe-boundary controller. Normal
operation keeps resource sampling and diagnostic logging disabled. The
controller requires three consecutive windows before acting and applies a
cooldown; it never destroys active TensorRT contexts or interrupts in-flight
inference. Hardware-dependent values are read from the active runtime profile,
and P/E utilization requires explicit logical-index topology data when the OS
does not expose it directly.

No physical Phase 15 result is claimed here yet. The exact benchmark must run
the same representative end-to-end workload separately on each target with
`ROOP_RUNTIME_MONITOR=1 ROOP_RUNTIME_DIAGNOSTICS=1`, and may add
`ROOP_RUNTIME_ADAPTIVE=1` only for the adaptive arm.

### RTX 3060 Phase 15

| Metric | Status |
|---|---|
| End-to-end/stage FPS and latency | pending physical validation |
| CPU/P-core/E-core/GPU utilization | pending physical validation |
| VRAM/RAM, queues, worker utilization | pending physical validation |
| Bottleneck classification and adaptive stability | pending physical validation |

### RTX 4070 Phase 15

| Metric | Status |
|---|---|
| End-to-end/stage FPS and latency | pending physical validation |
| CPU/P-core/E-core/GPU utilization | pending physical validation |
| VRAM/RAM, queues, worker utilization | pending physical validation |
| Bottleneck classification and adaptive stability | pending physical validation |

Use separate reports and profile keys for the two GPUs; do not combine their
FPS or resource values into an average.

## Phase 16 final integrated validation

Phase 16 is a regression/acceptance pass over the integrated runtime. It uses
the immutable controlled workload definition from
`app/tests/baseline_controlled.py`, and accepts an optimization only from
end-to-end render results. Component or per-face measurements are not used as
success claims. The RTX 4070 was physically present for this pass; the RTX
3060 was not physically available and has no substituted values below.
The immutable official reference remains the 600-frame RTX 4070 result in
`PERFORMANCE_BASELINE.md` (9.62 FPS, GPEN 256 Pro, RealityUX, stabilization
ON, HEVC NVENC). The shorter Phase 16 rows below are paired before/after
acceptance arms with their own same-workload baseline; they do not overwrite
or silently re-label that official reference.

The detected RTX 4070 profile was: NVIDIA GeForce RTX 4070, Ada Lovelace,
compute capability 8.9, 11.994 GB total VRAM, 9.527 GB available at probe,
driver 610.88, CUDA 12.8, TensorRT 10.9.0.34, ONNX Runtime 1.23.2, FP16/
BF16/INT8 available, FP8 not exposed, NVDEC and NVENC available. The host
reported 24 physical / 32 logical CPU threads and 31.691 GB RAM. P/E rows in
the telemetry are explicitly marked as inferred from logical indices, not an
OS topology report.

### RTX 4070: final before/after matrix

The primary 720p multi-face matrix used `d4.mp4`, frames 0–119, TensorRT,
`realswap`, tracking and the detected runtime settings. Baseline is the
same-run `None` enhancer / `None` mask / stabilization OFF arm at 8.50 FPS.
Stage FPS is frames per second for decode/encode and faces per second for
swap/enhancement. Every arm completed 120/120 frames with zero wrong-faceset
applications and exit code 0.

| Arm | Baseline FPS | Final FPS | Improvement | Peak/avg VRAM MB | Peak RAM GB | CPU mean % | GPU mean % | Decode FPS | Swap/inference FPS | Enhance FPS | Encode FPS | Latency ms | Stability | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 8.50 | 8.50 | 0.00% | 4923 / 3029 | 24.589 | 25.952 | 28.536 | 64.86 | 15.75 | — | 631.58 | 445.69 | pass | 120/120; 0 wrong faceset; spot-check pass |
| stabilization ON | 8.50 | 7.06 | -16.94% | 5178 / 3258 | 24.668 | 27.759 | 30.225 | 292.68 | 15.75 | — | 428.57 | 223.52 | pass | 120/120; 0 wrong faceset; visual review pending |
| mask ON | 8.50 | 7.01 | -17.53% | 6001 / 3410 | 24.441 | 28.023 | 32.714 | 59.11 | 12.51 | — | 571.43 | 670.79 | pass | 120/120; 0 wrong faceset; visual review pending |
| color ON | 8.50 | 8.23 | -3.18% | 5266 / 3361 | 24.221 | 27.960 | 31.949 | 66.30 | 12.58 | — | 631.58 | 486.52 | pass | 120/120; 0 wrong faceset; visual review pending |
| postprocess heavy (UltraMax + RealityUX + stabilization + RCT) | 8.50 | 3.58 | -57.88% | 6765 / 3772 | 25.237 | 27.257 | 34.726 | 285.71 | 17.06 | 6.74 | 400.00 | 723.56 | pass | 120/120; 0 wrong faceset; visual review pending |

The Phase 12 acceptance classification on this target is: stabilization,
mask, color, and heavy postprocessing are **regressions on this workload**
when enabled globally; heavy postprocessing is not accepted as a speed
optimization. The 4070 is CPU/stage limited for the baseline (GPU mean
28.536%, frame-total 57.2% of measured stage time), not encode limited.

After the runtime-policy fix, a 30-frame integrated guard arm completed with
the live monitor enabled: the profile selected 8 workers, queue depth 3, swap
and tile batches 2, face concurrency 3, four in-flight frames, and a 144-frame
stabilization chunk. The monitor reported 3.233 end-to-end FPS, input/output
queue averages of 9/2, 50% worker utilization, 30.86% VRAM pressure, 68.75%
RAM utilization, and `synchronization-bound`. This was diagnostics-only (no
adaptive changes were enabled); the external sampler supplied 27.932% mean
GPU utilization and 24.590% mean CPU utilization for the corresponding run.
The monitor's own CPU/GPU fields were unavailable on this host, so they are
not substituted with estimates.

### RTX 4070: enhancer and feature-toggle coverage

The final integrated render arms directly exercised no enhancer and UltraMax
(the postprocess-heavy arm). The other discovered enhancers have existing
compatibility/per-face evidence, but that is not an end-to-end acceptance
claim. Their final integrated status is:

| Coverage | Status |
|---|---|
| None | completed above |
| UltraMax | completed above; 3.58 FPS in the heavy postprocess arm |
| GPEN, GPEN 256, GPEN 256 Pro, GPEN Realistic, CodeFormer, GFPGAN, RestoreFormer++, frame upscalers, other discovered enhancers | pending dedicated end-to-end render on this target |
| temporal detection OFF/ON, tracking OFF/ON, NVDEC OFF/ON, FP32/FP16/mixed paired on one workload | partial: tracking/temporal ON and FP32/mixed/FP16 coverage completed; OFF/ON feature pairs pending dedicated rerun |

### RTX 4070: codec/output before/after matrix

This paired matrix uses the same 120-frame workload and a 120-frame segment.
The explicit codec remains authoritative. The libx264 row is the codec-matrix
baseline; encoding itself is not the end-to-end bottleneck here.

| Codec | Baseline FPS | Final FPS | Improvement | Peak/avg VRAM MB | CPU mean % | GPU mean % | Decode FPS | Swap FPS | Encode FPS | Encode time s | Latency ms | Stability | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| libx264 | 8.26 | 8.26 | 0.00% | 5139 / 3293 | 27.754 | 32.561 | 65.22 | 14.79 | 413.79 | 0.29 | 479.37 | pass | 120/120; 0 wrong faceset; visual review pending |
| h264_nvenc | 8.26 | 8.27 | +0.12% | 5561 / 3401 | 28.568 | 32.111 | 59.11 | 14.79 | 315.79 | 0.38 | 472.61 | pass | 120/120; 0 wrong faceset; visual review pending |
| hevc_nvenc | 8.26 | 8.42 | +1.94% | 5591 / 3314 | 27.915 | 32.653 | 68.18 | 13.90 | 324.32 | 0.37 | 462.09 | pass | 120/120; 0 wrong faceset; visual review pending |

The result is classified **beneficial on the RTX 4070 only so far** for the
HEVC encoder arm; no universal encoder claim is made until the RTX 3060 runs
the same matrix. Segment rotation count was one for each 120-frame arm.

### RTX 4070: resolution and precision acceptance

These are true end-to-end compatibility runs with cadence taken from the
input. The 1080p source is 25 FPS with AAC; the 4K smoke source is 24 FPS and
has no audio stream. They are not cross-resolution FPS comparisons.

| Input / faces | Precision | Frames | Final FPS | Peak VRAM GB | CPU % | GPU % | Duration / resolution / audio | Identity / texture / channel | Status |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| 1920x1080 / one source | FP32 | 418 | 6.326 | 6.021 | 4.7 | 40.0 | 16.720 s / 1920x1080 / AAC retained | pass / pass / pass | pass |
| 1920x1080 / one source | mixed | 418 | 7.588 | 4.975 | 6.0 | 30.2 | 16.720 s / 1920x1080 / AAC retained | pass / pass / pass | pass |
| 4096x2160 / one source smoke | FP16 | 60 | 0.713 | 5.953 | 8.0 | 20.2 | 2.500 s / 4096x2160 / source has no audio | pass / pass / pass | pass |

The harness cadence fix used by these runs is in
`app/tests/angle_video.py`; it prevents a fixed 30 FPS test entry from
creating false duration failures on 24/25 FPS inputs.

### RTX 3060: final integrated validation

The RTX 3060 was unavailable in this environment. No result, FPS, VRAM,
quality, or stability value is inferred from the RTX 4070. The complete
Phase 16 target remains **pending physical validation**:

| Matrix | Status |
|---|---|
| 720p multi-face baseline / stabilization / mask / color / heavy | pending physical validation |
| Codec: libx264 / h264_nvenc / hevc_nvenc, 120-frame segment | pending physical validation |
| 1080p FP32 / mixed and 4K FP16 smoke | pending physical validation |
| Enhancer matrix: none, GPEN variants, UltraMax, CodeFormer, GFPGAN, RestoreFormer++, frame upscalers, other discovered enhancers | pending end-to-end physical validation |
| temporal detection / tracking / NVDEC OFF-ON / NVENC OFF-ON | pending physical validation |
| CPU/P/E/GPU/VRAM/RAM/queues/transfers/synchronization | pending physical validation |
| frame order, duration, audio, masks, stabilization, black/NaN/dropped/duplicate/deadlock/leak/corruption checks | pending physical validation |

Run on the physical RTX 3060 without rewriting configuration, using the
isolated application environment:

```bash
cd app
env/Scripts/python.exe tests/phase12_benchmark.py --target "RTX 3060" --video <720p-d4> --start 0 --end 120 --out output/phase16_validation/phase12_3060
env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 3060" --video <720p-d4> --start 0 --end 120 --codecs libx264,h264_nvenc,hevc_nvenc --segment-sizes 120 --out output/phase16_validation/phase13_3060
env/Scripts/python.exe tests/compat_one.py --precision fp32 --provider tensorrt --mask-engine None --enhancer None --clip <1080p> --source harjot --out output/phase16_validation/resolution_3060/1080_fp32
env/Scripts/python.exe tests/compat_one.py --precision mixed --provider tensorrt --mask-engine None --enhancer None --clip <1080p> --source harjot --out output/phase16_validation/resolution_3060/1080_mixed
```

Repeat the accepted enhancer, feature-toggle, and codec arms with the same
input files and record each result under the RTX 3060 hardware profile key.
Only after those rows pass end-to-end quality and stability checks may an arm
be classified **beneficial on both**.

## Gate D — CPU optimization matrix

**This gate is measurable on the RTX 3060 host and was not on the 4070.**
Windows exposed no P/E topology on the workstation, so Gate D was deferred
there. The laptop reports `windows-cpu-set-efficiency-class` — a real OS
report — with 6 P-cores (12 logical) and 8 E-cores on an i7-12700H. Running it
here also satisfies the plan's own requirement not to hardcode i9-14900K
behaviour: this is a different hybrid CPU.

Measured 2026-08-30, `d4.mp4` frames 0..120, 4 candidates:

| Candidate | Worker threads | FPS | Peak VRAM | Mean CPU | Mean GPU |
|---|---:|---:|---:|---:|---:|
| auto | 8 | 3.24 | 4,617 MB | 38.44% | 50.67% |
| p_only | 12 | 3.82 | 5,433 MB | 38.71% | 47.91% |
| p_priority_e (E limit 2) | 14 | 3.82 | 5,186 MB | 37.38% | 46.44% |
| p_plus_e | 20 | **3.96** | 5,297 MB | 37.87% | 46.26% |

### The matrix does NOT show that P/E distribution is worth 22%

**The arms vary two things at once.** Each distribution policy also selects its
own worker count — 8, 12, 14, 20 — and the FPS ordering tracks the thread count
monotonically. "p_plus_e is fastest" is therefore not separable from "20 workers
beats 8 workers" on this evidence, and the two middle arms are *identical* at
3.82 despite differing worker counts, which looks like saturation rather than a
distribution effect.

`p_only` -> `p_plus_e` is **+3.7%**, which is exactly this project's documented
run-to-run variance on identical settings. Single, non-counterbalanced arms
cannot resolve a difference that size. **No distribution policy is promoted on
this result.**

### What the gate DID establish: the automatic thread count is too low

`auto` is the only arm that used the machine's saved `max_threads: 8`, and it
is the slowest by 18-22%. That constant was never measured on this hardware:
`_threads_basis` records `v3|14|8`, and the session log for 2026-08-25 states
plainly that the `<7GB` tier knee of 8 "was measured on a **4070 with pools
forced to 0/0**, never on real 6GB silicon" and still "owes a measurement".

### The owed thread-knee measurement, counterbalanced — the knee of 8 is VINDICATED

Run 8 / 20 / 20 / 8 on `d4.mp4` frames 0..120 so the first-arm effect cancels.
Worker count is the ONLY variable; no CPU-distribution or thread-pinning
environment was set.

| Worker threads | rep a | rep b | mean | peak RSS |
|---|---:|---:|---:|---:|
| 8 (the saved `auto` value) | 3.41 | 3.43 | **3.42** | 2.818 GB |
| 20 (every logical core) | 3.40 | 3.38 | **3.39** | 2.823 GB |

**-0.9%: neutral**, far inside the 3.7% run-to-run floor, and the direction even
favours 8. Raising the worker count on this 6 GB / 14-core laptop buys nothing.

This closes the measurement the 2026-08-25 session recorded as owed. The `<7GB`
tier knee of **8 is correct on real 6 GB silicon**, not merely inherited from
the 4070 — and `max_threads: 8` should stay. It also means the Gate D spread
above is NOT explained by worker count, because the two hypotheses were tested
independently and both failed:

| Hypothesis for Gate D's 3.24 -> 3.96 | Verdict |
|---|---|
| P/E distribution policy | not separable — each arm also changed the thread count |
| Worker thread count | **rejected**: 8 vs 20 counterbalanced is -0.9% |

### Isolating the distribution policy — the effect is REAL, +19.6%

Worker count fixed at 20 and `ROOP_RUNTIME_{ORT_INTRA,ORT_INTER,CV,FFMPEG}_THREADS=1`
applied to both arms, so the CPU distribution policy is the only variable.
Counterbalanced auto / p_plus_e / p_plus_e / auto.

| Distribution | rep a | rep b | mean | mean GPU |
|---|---:|---:|---:|---:|
| auto | 3.15 | 3.13 | **3.14** | 45.7% |
| p_plus_e | 3.81 | 3.70 | **3.76** | 44.5% |

**+19.6%**, with no overlap between the arms — both `p_plus_e` runs beat both
`auto` runs. This is well outside the 3.7% floor and is a genuine Gate D win.

Note also that `auto` **with** pinning (3.14) is slower than `auto` **without**
it (3.39 from the thread A/B): the thread pinning is not free on its own, and
only pays off in combination with the P/E-aware distribution. This is why the
original 4-arm matrix could not be read — it moved three things at once.

### Acceptance: the candidate against what the user actually runs

Counterbalanced prod / cand / cand / prod, `d4.mp4` frames 0..120.

| Configuration | rep a | rep b | mean |
|---|---:|---:|---:|
| production default (auto, 8 workers, no pinning) | 2.97 | 3.17 | **3.07** |
| Gate D candidate (p_plus_e, 20 workers, pinned) | 3.63 | 3.67 | **3.65** |

**+18.9%**, no overlap, reproducing the isolation run's +19.6% in an independent
experiment.

### METHODOLOGICAL WARNING for this target: ~15% cross-run drift

The same configuration (auto / 8 / unpinned) measured **3.41, 3.43, 2.97, 3.17**
across experiment sets — a 15% spread, four times the 4070's documented 3.7%.
This is a thermally-constrained laptop and its absolute numbers wander between
sets.

**Only counterbalanced comparisons within a single set are trustworthy here.**
Two arms from different sets must never be compared, and any 3060 claim under
~15% that is not counterbalanced is noise. The +19% survives because both
experiments were internally counterbalanced with no overlap between arms.

### Gate D disposition: SHIPPED, hardware-adaptive

Per the project rule "ship the fix, not the flag", the win is now the automatic
default rather than an env var. `auto` previously skipped the P/E branch
entirely, so **no hybrid CPU ever got P/E-aware scheduling by default**.

`auto` now resolves to `p_plus_e` when — and only when — the OS actually
reports which logical processors are efficiency cores.
`_REAL_CPU_TOPOLOGY_SOURCES` allowlists `windows-cpu-set-efficiency-class`,
`linux-cpu-capacity-logical` and the explicit `environment-indices` override.
Every "could not tell" source is excluded, because guessing which indices are
E-cores would pin workers to the wrong processors — worse than not pinning.

An explicit `ROOP_CPU_DISTRIBUTION` or setting still wins outright.

**Classification: B — RTX 3060-SPECIFIC.** The RTX 4070 workstation reports no
hybrid topology at all (Gate D was deferred there for exactly that reason), so
its behaviour is provably unchanged; a regression test asserts `auto` stays
`auto` on a non-hybrid report. This must be re-validated on the 4070 only if
Windows begins exposing hybrid topology on that host.

Confirmed on the shipped path: at startup `run.py` now prints
`[CPU] affinity distribution=p_plus_e logical=20
source=windows-cpu-set-efficiency-class` and publishes
`ROOP_CPU_DISTRIBUTION=p_plus_e` with affinity applied, before the model
pipeline loads.

The final measurement, counterbalanced at the shipped worker count with no
thread pinning, is the acceptance evidence for the change:

| Distribution @ 8 workers | rep a | rep b | mean |
|---|---:|---:|---:|
| auto (old behaviour) | 3.15 | 3.23 | **3.19** |
| p_plus_e (now automatic) | 3.81 | 3.77 | **3.79** |

**+18.8%.** Worker count is unchanged at 8, so `max_threads` and its
`_threads_auto` provenance are untouched.

### GATE A FINDING: the bench harness measured a different program than production

Verifying the shipped default exposed benchmark contamination that predates
this session and affects **every** number produced by these harnesses.

`run.py` builds a hardware-only profile at startup and calls
`RuntimeOptimizer.apply_environment`, publishing worker / queue / pool hints
and applying CPU affinity *before any thread pool exists*. `two_face_video.py`
— which every phase harness shells out to — never did this. `ProcessMgr`
re-applies the environment much later, at which point the CPU-affinity decision
no longer changes the run.

The symptom was precise: the shipped default benched at **3.19 fps, exactly the
`auto` mean**, while the identical configuration supplied through the process
environment benched at **3.79**. The harness was under-reporting the shipped
product by 18.8% because the setting arrived too late to take effect.

Fixed by `_apply_startup_runtime_environment()` in `two_face_video.py`, which
reproduces `run.py`'s startup pass. Explicit caller environment still wins —
`apply_environment` only fills variables that are absent — so counterbalanced
A/Bs keep control of the key under test.

**Consequence for the record: the RTX 3060 Phase 2 baseline of 4.33 fps was
measured with the uncorrected harness and under-reports production.** It is
re-run below rather than adjusted.

**Caveat on absolute values:** these 120-frame arms run ~3.4 fps against the
600-frame locked baseline's 4.33 fps. Processing FPS excludes startup (it is
parsed from the encoder's own line), but a short window amortises in-render
warm-up less. 120-frame arms are comparable to each other, not to the baseline.

## Gate C: future-architecture readiness

The runtime profiler is capability-driven. It records the detected GPU name,
architecture, compute capability, total and available VRAM, driver, CUDA,
TensorRT, ONNX Runtime, Tensor Core math modes, FP16, BF16, INT8, FP8, NVDEC,
and NVENC capabilities. Unknown compute capabilities remain usable as an
explicit `SM major.minor` architecture identity; they are not classified as
Ada, Ampere, Rubin, or any other family by name.

Precision selection requires the detected hardware capability, a usable
TensorRT/provider path, model-specific policy evidence, and quality validation.
BF16 is currently an explicit provider path; INT8 and FP8 are recorded when
the installed TensorRT builder exposes them but remain rejected until a real
calibrated provider path and measured quality/throughput result are added.
No precision is selected merely because a future GPU exposes a feature.

The runtime profile key includes hardware/software identity, model revision,
precision, workload shape, and runtime schedule. TensorRT engine parents also
include a canonical fingerprint of effective workspace, partition, context,
builder, auxiliary-stream, CUDA-Graph, and precision settings. TensorRT's
graph hash remains responsible for the model graph and concrete shapes; the
profile cache independently records those shapes and workload characteristics.
This prevents an RTX 4070 result from becoming a generic RTX 3060 or future
architecture result.

### Gate C defect found on the RTX 3060: the engine cache had no driver identity

The claim above that engine caches "distinguish ... driver ..." was **false in
the built artifact**. The live cache directory on this machine read:

    mixed_NVIDIA_GeForce_RTX_3060_Laptop_GPU_sm0806_cuda12.8_drvunknown_trt10.9.0.34_ort1.23.2_lnfp32_seq_heur_b3_a-1_g0

`drvunknown`. `backend_manager.cache_namespace` resolved the driver solely
through `torch._C._cuda_getDriverVersion`, above a comment asserting that probe
"is available on the supported CUDA builds". It is not present on torch
2.7.0+cu128, so the value silently stayed `"unknown"` on every machine using
this build — including the 4070.

TensorRT engines are driver-sensitive and are not guaranteed portable across
driver upgrades, so this is a genuine **cache invalidation error**: a stale
engine could be reused after a driver change with nothing to detect it. Gate A
lists "cache invalidation errors" as a class to hunt; this is one, and it was
inside the mechanism Gate C relies on.

Fixed by falling back to `nvidia-smi` when the torch probe is absent; the key
now records `drv616.56` on this host. Note this changes the namespace, so the
first run after the fix rebuilds engines once per precision — correct, since
engines built under an unidentified driver were never safe to reuse.

The pre-existing test asserted only `'_drv' in ns`, which passes for the literal
string `drvunknown`, so it had been providing false assurance. It now asserts
the driver actually resolves.

### Tested versus future-ready

| Capability/target | Status |
|---|---|
| RTX 4070 Ada, SM 8.9, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2 | detected and physically tested in the separate 4070 rows above |
| RTX 3060 Ampere | mandatory physical validation remains pending in this session; no values are inferred |
| Unknown/future compute capability identity and cache isolation | logically covered by runtime tests; no future GPU benchmark claimed |
| Rubin-class hardware | not available and not tested; no Rubin optimization or compatibility claim |
| FP16/BF16/INT8/FP8 selection | capability and policy gates implemented; only modes with a validated provider/model result may be promoted |
| NVDEC/NVENC | detected from the installed FFmpeg stack; target-specific throughput remains the measured matrix above or pending |

Future TensorRT versions may add fields to the builder configuration mapping.
The cache fingerprint accepts those fields without a GPU-family allowlist, so
new runtime options are isolated rather than silently inheriting a legacy
engine. Adding a new precision still requires an actual provider path and
quality/performance validation.
