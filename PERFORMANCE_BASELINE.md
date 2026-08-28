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

# OFFICIAL BASELINE — TO BE FILLED IN PHASE 2

## Software

- OS:
- NVIDIA driver:
- CUDA:
- TensorRT:
- ONNX Runtime:
- Python:
- FFmpeg:
- OpenCV:

## Workload

- Input video:
- Resolution:
- Input FPS:
- Codec:
- Number of faces:
- Face detector:
- Face swap model:
- Enhancement:
- Stabilization:
- Output resolution:
- Output codec:
- Other options:

## Measurements

- End-to-end FPS:
- Average frame latency:
- P95 latency:
- Decode FPS:
- Encode FPS:
- CPU utilization:
- P-core utilization:
- E-core utilization:
- GPU utilization:
- GPU memory used:
- GPU memory free:
- System RAM used:
- Peak RAM:
- CPU↔GPU transfer time:
- Synchronization time:
- Queue depth:

## Reproduction command

```text
TODO: record exact command/settings used for the official baseline.
```

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
