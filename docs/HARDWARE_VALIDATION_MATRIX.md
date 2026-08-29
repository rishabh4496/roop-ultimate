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
