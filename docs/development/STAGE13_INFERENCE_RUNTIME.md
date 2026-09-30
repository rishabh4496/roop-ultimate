# Stage 13: Inference Runtime Optimization

## 1. Overview & Architecture

Stage 13 establishes a comprehensive model inference optimization, validation, and benchmarking harness across the Roop Ultimate pipeline. It audits models across all functional pipeline families, evaluates five distinct runtime execution modes, evaluates advanced execution features, enforces the empirical small-model threshold rule, tailors engine compilation to our dual-device hardware environment, and guarantees automatic invalidation and rebuild of stale TensorRT engines.

```
                      ┌────────────────────────────────────────┐
                      │        ONNX Model Architecture         │
                      └───────────────────┬────────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
         [Small / Light Model]                         [Compute-Heavy Model]
      (<45MB, Landmarks, Embedders)                  (Swappers, Restorers, Masks)
                   │                                             │
      ┌────────────▼────────────┐                   ┌────────────▼────────────┐
      │  CUDA Execution Provider│                   │ TensorRT Execution EP   │
      │  - Low dispatch latency │                   │ - High FLOPS saturation │
      │  - Zero TRT context     │                   │ - FP16 / MIXED kernels  │
      │    overhead penalty     │                   │ - LayerNorm FP32 guard  │
      └─────────────────────────┘                   └─────────────────────────┘
```

---

## 2. Functional Model Categories Audited

Every model is classified into one of six functional categories:

| Category | Model Families | Typical Input Shapes | Batch Behavior | Recommended Mode |
|---|---|---|---|---|
| **`DETECTOR`** | SCRFD-10G, SCRFD-2.5G, RetinaFace, YOLOFace | `[1, 3, 512, 512]` to `[1, 3, 640, 640]` | Static / Batch-1 | TensorRT FP16 / CUDA |
| **`RECOGNIZER`** | ArcFace w600k, Buffalo_l | `[1, 3, 112, 112]` | Small batch | **CUDA EP** (Avoids TRT overhead) |
| **`SWAPPER`** | Hyperswap 256, Inswapper 128, RealSwap | `[B, 3, 256, 256]`, `[B, 512]` | Dynamic `[1..16]` | **TensorRT MIXED** |
| **`RESTORER`** | GPEN-512, GPEN-256 Pro, CodeFormer, GFPGAN | `[B, 3, 512, 512]` / `[B, 3, 256, 256]` | Dynamic `[1..4]` | **TensorRT MIXED** (LayerNorm FP32) |
| **`MASK`** | XSeg, BiSeNet, Occluder | `[1, 256, 256, 3]` / `[1, 3, 512, 512]` | Static / Batch-1 | TensorRT FP16 / CUDA |
| **`LANDMARK`** | 2DFAN4, Landmark 106, PIPNet | `[1, 3, 192, 192]` / `[1, 3, 256, 256]` | Static / Batch-1 | **CUDA EP** (Overhead dominance) |

---

## 3. The 5 Evaluated Execution Modes

For every model, the benchmarker measures:

1. **`CUDA` (CUDAExecutionProvider)**: Standard CUDA kernel execution with CuDNN exhaustive convolution search and memory arena caching. Fast kernel launch and near-zero setup overhead.
2. **`TensorRT_FP32` (TensorrtExecutionProvider)**: Strict FP32 precision engine without FP16 kernel conversion.
3. **`TensorRT_FP16` (TensorrtExecutionProvider)**: Pure FP16 kernel execution. Fast compute throughput on Tensor Cores.
4. **`TensorRT_MIXED` (TensorrtExecutionProvider)**: FP16 accelerated compute paired with FP32 fallback for sensitive nodes (`trt_layer_norm_fp32_fallback=True`). Prevents NaN activation collapse and rainbow discoloration on restorers and swappers.
5. **`CPU` (CPUExecutionProvider)**: Multi-threaded host CPU fallback.

---

## 4. Empirical Small-Model Rule

> **"Do not assume TensorRT is faster for every small model."**

For lightweight models (e.g. 106-point landmark detector at ~1.2MB, or ArcFace 112x112 at ~40MB):
- **TensorRT Overhead**: Context switching, dynamic I/O binding buffer resolution, and stream synchronization impose **1.5 ms – 2.5 ms** of baseline overhead per invocation.
- **CUDA EP Overhead**: Direct CUDA kernel launches execute in **0.8 ms – 1.2 ms** with virtually zero binding overhead.
- **Decision Logic**: When estimated or measured CUDA latency is below the dispatch threshold or outperforms TensorRT, `evaluate_small_model_runtime` enforces `ExecutionMode.CUDA`, reserving TensorRT exclusively for compute-heavy stages (Swapping and Restoration).

---

## 5. Dual-Hardware Engine Configurations

| Parameter | Main Device (RTX 4070 Desktop) | Secondary Device (RTX 3060 Laptop) |
|---|---|---|
| **Architecture** | `sm_89` (Ada Lovelace) | `sm_86` (Ampere) |
| **Total VRAM** | 12.0 GB | 6.0 GB |
| **TRT Workspace** | **4096 MB (4 GiB)** | **1536 MB (1.5 GiB cap)** |
| **Builder Level** | Level 3 (Exhaustive tactics) | Level 2 (Heuristics enabled, lower RAM spike) |
| **CUDA Graphs** | **Allowed** (Static shape models) | **Disabled** (RSS safety strictly $<2.5$ GB) |
| **Context Pool** | Multi-context allowed (Pool: 2) | Single context (Pool: 0, global guard lock) |
| **Max Batch Size** | Up to 16 | Up to 4 |

---

## 6. Engine Compatibility Signature & Automatic Rebuild

Serialized `.engine` files are architecture-dependent and driver-specific. To eliminate silent crashes, corrupt reads, and segfaults, each engine is stored alongside a `.engine.meta.json` file carrying an `EngineCompatibilitySignature`:

```json
{
  "gpu_arch": "sm_89",
  "trt_version": "10.0.1",
  "cuda_version": "12.4",
  "model_sha256": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
  "precision": "fp16",
  "shape_profile": {
    "min": "1x3x256x256",
    "opt": "4x3x256x256",
    "max": "8x3x256x256"
  },
  "builder_config_hash": "a1b2c3d4e5f6"
}
```

### Invalidation Triggers
An engine is immediately flagged as **invalid** if:
1. GPU architecture changes (`sm_89` $\neq$ `sm_86`).
2. TensorRT or CUDA driver/toolkit version changes.
3. ONNX model file content hash changes (model updated or replaced).
4. Target precision changes (`fp16` vs `mixed` vs `fp32`).
5. Shape profile bounds mismatch.
6. Engine file is 0 bytes or corrupt.

### Automatic Rebuild
When an engine is flagged invalid:
- `EngineCacheManager` purges the stale `.engine` and `.meta.json`.
- A sequential rebuild is acquired under `_rebuild_lock` (preventing concurrent VRAM allocation spikes).
- A clean engine is compiled according to the hardware profile, and the new signature is atomically committed.
