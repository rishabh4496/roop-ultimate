# ROOP-ULTIMATE PERFORMANCE BASELINE

## IMPORTANT

This file is the immutable reference for pre-optimization performance.

**Do not overwrite the original baseline.**

If new baseline measurements are needed, append them under a dated section.

---

# HARDWARE

- CPU: Intel Core i9-14900K
- GPU: NVIDIA RTX 4070
- RAM: 32 GB

Optional secondary validation:
- GPU: NVIDIA RTX 3060

---

# USER-REPORTED BASELINE

**Maximum observed FPS:** approximately 20 FPS

This number must be reproduced under a controlled, documented workload before it is treated as the official benchmark.

---

# OFFICIAL BASELINE — LOCKED 2026-08-29 (RTX 4070)

Measured by `app/tests/baseline_controlled.py`. This replaces the
Phase 2 TODO block. The immutable user-reported baseline above is
untouched; note it has still never been reproduced under a controlled
workload, so it is not the comparison point -- this block is.

## Software

- OS: Windows 11 Pro 10.0.26200
- NVIDIA driver: 610.88
- CUDA: 12.8
- TensorRT: 10.9.0.34
- ONNX Runtime: 1.23.2
- Python: 3.10.20
- FFmpeg: 8.1.2
- OpenCV: 4.9.0

## Workload

- Input video: `G:/pinokio/roop-keep/double/d4.mp4` frames 0..600
- Resolution: 1280x720
- Input FPS: 30
- Codec: h264
- Number of faces: 2 people, 856 faces detected across the window
- Face detector: retinaface_r50 at det_size 512
- Face swap model: realswap (swap-model mask strength 25)
- Enhancement: GPEN 256 Pro
- Mask engine: RealityUX
- Stabilization: face + mask + enhancer all ON, one_euro (production config)
- Tracking: on
- Worker threads: 10; TRT pool 2, detmask pool 2
- Output resolution: 1280x720
- Output codec: hevc_nvenc at quality 10
- Other options: merger clarity 0.4, upscale_after_swap off

## Measurements

- End-to-end FPS: **9.62** (600 frames in 62.34 s of processing)
- Average frame latency: 233.34 ms per frame of worker thread time
- P95 latency: not measured (the stage probe records totals and call counts,
  not a per-call distribution; adding one would need a per-call histogram)
- Decode FPS: 189.87
- Encode FPS: 46.69
- CPU utilization: peak 99.2%, mean 20.49% across all 32 logical cores
- P-core utilization: peak 99.194% (8 P-cores / 16 logical, INFERRED from core
  counts, not an OS topology report -- Windows does not expose hybrid topology
  to psutil)
- E-core utilization: peak 99.206% (16 E-cores)
- GPU utilization: peak 76.0%, mean 33.952%
- GPU memory used: peak 7067.0 MB, mean 4080.346 MB
- GPU memory free: 5215 MB at peak (12282 MB total)
- System RAM used: peak 27.825 GB
- Peak RAM: process + descendants peak RSS 11.663 GB, mean 7.568 GB
- GPU power: peak 118.09 W
- CPU<->GPU transfer time: not measured. ORT owns its provider transfers and
  fences them internally; the app has no application-managed H2D/D2H to time.
  Phase 8 recorded this same limitation.
- Synchronization time: not measured, for the same reason.
- Queue depth: 3 (runtime default for >1 worker; clamped by the in-flight budget)
- Swap audit: 856 faces seen, 850 swapped, **0 wrong
  faceset applied** across 642 attributed swaps

## Reproduction command

```text
app\env\Scripts\python.exe app\tests\baseline_controlled.py --tag phase2_4070
```

Everything else -- provider, swap model, thread count, pool sizes, stabilizer
settings -- is read from that machine's own `config.yaml`, so the identical
command produces the RTX 3060 row without editing anything.

---

# BASELINE RULES

Use the same representative workload when comparing optimization phases.

Where possible:
- use the same input video,
- same models,
- same output settings,
- same quality settings,
- same driver/software versions.

A benchmark must be long enough to avoid measuring only startup/warmup behavior.

Report both:
- warm-up behavior
- steady-state throughput.

---

# BASELINE ACCEPTANCE

The official baseline becomes locked only after Phase 2 produces a reproducible benchmark.

Once locked, do not edit the original values.

---

# CONTROLLED RTX 4070 REFERENCE — 2026-08-28

This dated record supplements the original Phase 2 TODO block; it does not
overwrite the immutable user-reported baseline above. It is a warm/cached
runtime reference for the current Phase 0–6 implementation, not a replacement
for the eventual dual-GPU final baseline.

## Software and hardware

- GPU: NVIDIA GeForce RTX 4070, compute capability 8.9, 11.99 GiB / 12282 MiB
- Driver: 610.88
- CPU: Intel Core i9-14900K, 24 physical / 32 logical cores
- RAM: 32 GB
- CUDA / PyTorch: CUDA 12.8 / torch 2.7.0+cu128
- TensorRT / ONNX Runtime: 10.9.0.34 / 1.23.2
- Python / OpenCV / FFmpeg: 3.10.20 / 4.9.0 / 8.1.2

## Real-video workload

- Input: `G:/pinokio/roop-keep/double/d1.mp4`
- Resolution / source FPS / frames: 1280×720 / 25 FPS / 141 frames
- Provider: TensorRT → CUDA → CPU
- Swap model: RealSwap, mixed guarded precision
- Enhancer: GPEN 256 Pro
- Mask: RealityUX
- Stabilization: enabled, strength 0.6
- Tracking: enabled
- Worker threads: 6
- Swap-model mask strength: 25
- Merger clarity: 0.4
- Output: 141 frames, H.264 validation output

## Measurements

- End-to-end processing: 34.44 s; **4.09 FPS**
- Peak reported process RSS during processing: approximately **10.47 GB**
- Observed `nvidia-smi` sample: 3.36–3.74 GiB used, 8.27–8.65 GiB free,
  29–75% GPU utilization, 45–111 W; this is sampled telemetry, not a peak
- Swap audit: 359 faces detected; 346 swapped (96.4%); 13 shared-crop
  refusals; output rows recorded under the generated Phase 4 validation output
- Auto-capture warning: the two people overlap in every scanned frame
  (separation 0.107), so identity quality is not a clean acceptance workload

## Reproduction command

```text
app\\env\\Scripts\\python.exe app\\tests\\two_face_video.py --tag phase4_corrected_4070_d1 --video G:/pinokio/roop-keep/double/d1.mp4 --sources harjot,ashna --start 0 --end 141 --capture -1 --capture-budget 30 --provider tensorrt --swap-model realswap --enhancer "GPEN 256 Pro" --mask-engine RealityUX --stabilize-mask 1 --stabilize-mask-strength 0.6 --tracking 1 --threads 6 --swap-model-mask-strength 25 --merger-clarity 0.4 --out app/output/phase4_corrected_4070_d1
```

The RTX 3060 Laptop row remains **PENDING**. Its prior strict Phase 4
measurement was approximately 2.82–2.83 GB RSS and failed the required
`<2.5 GB` gate; no RTX 4070 result is copied to that target.

## Gate E supplemental 4070 comparison - 2026-08-30

This is a supplemental scheduler A/B and does not alter the immutable locked
baseline above. Both arms used the locked d4 1280x720 workload, frames 0-600,
the production models/settings, and the same detected RTX 4070 software stack.

| Target / arm | FPS | Peak / avg VRAM (MB) | Peak / avg RSS (GB) | Avg CPU / GPU (%) | Decode / swap / enhance / encode (FPS) | Latency (ms) | Quality |
|---|---:|---:|---:|---:|---:|---:|---|
| RTX 4070 pre-scheduler | 11.00 | 6548 / 3640.584 | 11.738 / 7.411 | 21.675 / 33.067 | 260.87 / 22.12 / 41.58 / 618.56 | 295.50 | 0 wrong-faceset |
| RTX 4070 unified scheduler | 11.11 | 6581 / 3637.348 | 11.790 / 7.399 | 21.546 / 33.292 | 301.51 / 22.92 / 42.06 / 576.92 | 282.33 | 0 wrong-faceset |
| RTX 3060 | **PENDING** | **PENDING** | **PENDING** | **PENDING** | **PENDING** | **PENDING** | physical validation unavailable |

The measured 4070 change is +1.0% end-to-end FPS with stable resource use and
no quality regression. It is not a universal dual-GPU acceptance until the
physical RTX 3060 pair is run; the 3060 row remains independently pending.
