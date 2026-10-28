# STAGE 0 — PERFORMANCE + QUALITY REPRODUCIBLE BASELINE REPORT

**Timestamp:** `2026-09-30T15:58:49.468921+00:00`  
**Classification:** **GPU-BOUND (High GPU compute/bandwidth saturation; GPU stage optimization directly scales throughput)**  

## 1. Environment & Hardware

- **GPU:** NVIDIA GeForce RTX 4070 (12282 MiB VRAM, Compute 8.9)
- **CPU:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (32 logical threads, 24 physical cores)
- **System RAM:** 31.7 GB (32452 MiB)
- **NVIDIA Driver:** 616.56

## 2. Pipeline Configuration

- **Swap Model:** `hyperswap`
- **Enhancer:** `None`
- **Mask Engine:** `None`
- **Provider:** `cuda`
- **Detector Engine / Size:** SCRFD / `640`
- **Execution Threads:** `20`

## 3. Overall Performance Summary

| Metric | Measured Value |
|---|---:|
| **Overall Throughput (FPS)** | **1.08 FPS** |
| **Total Frames Processed** | 390 frames across 13 scenarios |
| **Total Wall Clock Time** | 362.74 s |
| **Mean GPU Utilization** | 62.8% (Peak: 100.0%) |
| **Mean CPU Utilization** | 4.9% (Peak: 94.9%) |
| **Peak VRAM Allocated** | **12136.1 MiB** |
| **Peak Process RSS (RAM)** | **20515.0 MiB** |
| **GPU Synchronization Stalls** | 224.97 ms total |

## 4. Bottleneck Ranking (Per-Stage Latency Budget)

| Rank | Stage | Mean Latency (ms) | Share of Budget (%) | Primary Bound |
|:---:|:---|---:|---:|:---|
| 1 | `swap` | 442.33 ms | 56.5% | GPU Compute / TensorRT |
| 2 | `segmentation_mask` | 258.93 ms | 33.1% | GPU Memory / Memory Bandwidth |
| 3 | `detector` | 38.74 ms | 5.0% | GPU Compute / TensorRT |
| 4 | `blending` | 19.92 ms | 2.5% | GPU Memory / Memory Bandwidth |
| 5 | `verify` | 13.05 ms | 1.7% | Host CPU / I/O |
| 6 | `lighting` | 4.42 ms | 0.6% | Host CPU / I/O |
| 7 | `encode` | 2.72 ms | 0.3% | Host CPU / I/O |
| 8 | `decode` | 0.99 ms | 0.1% | Host CPU / I/O |
| 9 | `landmark` | 0.96 ms | 0.1% | Host CPU / I/O |
| 10 | `faceset_lookup` | 0.31 ms | 0.0% | Host CPU / I/O |

## 5. Quality Failure Ranking

| Rank | Failure Mode | Failure Count | Severity | Target Mitigation |
|:---:|:---|---:|:---|:---|
| 1 | Missed Faces | 31 | **HIGH** | Adaptive pyramid & temporal tracking |
| 2 | Face Detection Failures | 27 | **HIGH** | Adaptive pyramid & temporal tracking |
| 3 | Profile-Angle Failures | 25 | **MEDIUM** | 3-point pose warp & roll compensation |
| 4 | Incorrect Face Assignments | 17 | **LOW** | Hungarian track identity lock |
| 5 | Occlusion Failures | 0 | **MEDIUM** | Temporal occlusion smoothing |

## 6. Scenario-by-Scenario Matrix (All 13 Categories)

| # | Scenario | Resolution | Frames | Steady FPS | Frame Lat (ms) | GPU (%) | Peak VRAM (MB) | Missed Faces | Identity Sim |
|:---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Frontal Face | 1280x720 | 30 | **11.62** | 190.92 | 33.1% | 5027 | 0 | 0.985 |
| 2 | 45-Degree Face | 1280x720 | 30 | **11.62** | 97.27 | 32.8% | 6259 | 0 | 0.907 |
| 3 | Extreme Profile | 1280x720 | 30 | **4.36** | 216.86 | 35.0% | 7742 | 25 | 0.467 |
| 4 | Small Face | 1280x720 | 30 | **11.36** | 98.81 | 37.4% | 9093 | 0 | 0.767 |
| 5 | Multiple People | 1920x1080 | 30 | **3.44** | 305.81 | 33.6% | 10558 | 0 | 0.986 |
| 6 | Face Entering/Leaving Frame | 1280x720 | 30 | **7.48** | 151.34 | 35.3% | 11937 | 0 | 0.982 |
| 7 | Partial Occlusion | 1280x720 | 30 | **2.89** | 350.89 | 68.3% | 12110 | 0 | 0.803 |
| 8 | Hands/Object Crossing Face | 1280x720 | 30 | **1.21** | 832.85 | 87.7% | 12115 | 2 | 0.901 |
| 9 | Dark Scene | 1280x720 | 30 | **0.80** | 1184.54 | 92.0% | 12120 | 0 | 0.908 |
| 10 | High-Motion Scene | 1280x720 | 30 | **0.67** | 1497.20 | 94.8% | 12111 | 0 | 0.967 |
| 11 | Multiple Faces Interacting | 1280x720 | 30 | **0.33** | 3055.87 | 93.3% | 12110 | 4 | 0.424 |
| 12 | 1080p Resolution | 1920x1080 | 30 | **0.50** | 2003.88 | 93.9% | 12111 | 0 | 0.987 |
| 13 | 4K Resolution | 3840x2160 | 30 | **0.64** | 1662.44 | 78.8% | 12136 | 0 | 0.987 |

## 7. Recommended Optimization Order

- **1. Face Swapper TensorRT I/O Binding & Batch Inference: Swap model dominates frame latency share. Persistent GPU buffers & batching will eliminate CPU<->GPU copies.**
- **2. Occlusion / Segmentation Mask Acceleration: RealityUX/XSeg is second highest stage latency. Move pre- and post-processing onto GPU streams.**
- **3. Face Detector Pyramidal Optimization: SCRFD detection costs ~15-20% of frame time. Temporal skipping and adaptive scale pyramid will save compute.**
- **4. Restoration Model TRT Mixed-Precision Optimization: GPEN/UltraMax restoration should utilize locked FP16 TRT engine kernels.**
- **5. Zero-Copy Blending & Warp Composite: Move OpenCV affine warp and color transfer entirely into PyTorch/CUDA tensor kernels.**
