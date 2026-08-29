# Dual-GPU hardware validation matrix

This is the acceptance record for the two first-class NVIDIA targets. Results
are never averaged across GPUs. Runtime identity is detected from the active
stack and persisted with `hardware_profile_key`; free VRAM is telemetry and is
not used as a profile identity.

## Current validation state

- RTX 4070: physically available in the current environment (`nvidia-smi`
  reported 12,282 MiB total and driver 610.88).
- RTX 3060: unavailable in the current environment. Its measurements remain
  `pending`; no 4070 value is copied into this table.

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

| Metric | Result |
|---|---:|
| Baseline FPS | pending physical fixture |
| Final FPS | pending |
| Improvement | pending |
| Peak VRAM | pending |
| Average VRAM | pending |
| CPU utilization | pending |
| GPU utilization | pending |
| Decode throughput | pending |
| Inference throughput | pending |
| Enhancement throughput | pending |
| Encode throughput | pending |
| Latency | pending |
| Stability | pending |
| Output quality | pending |

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
