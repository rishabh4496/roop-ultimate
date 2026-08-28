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
