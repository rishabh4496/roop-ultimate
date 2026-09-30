# STAGE 2 — Model Lifecycle and Runtime Architecture Audit Report

**Repository:** `rishabh4496/roop-ultimate`  
**Execution Profile:** Main Workstation (NVIDIA GeForce RTX 4070 12GB Desktop, 32GB RAM, Compute 8.9)  
**Secondary Profile Verified:** Laptop Workstation (RTX 3060 Laptop 6GB, sub-7GB CUDA safety rules, 0/0 pool, strictly < 2.5GB system RSS)  
**Date:** 2026-09-30  

---

## 1. Executive Summary

In Stage 2, a complete audit of every model and session initialization path across the application was performed:
- **ONNX Runtime Sessions**
- **TensorRT Engines & Timing Caches**
- **CUDA Providers & PyTorch Execution**
- **HyperSwap / RealSwap (Face Swappers)**
- **SCRFD & InsightFace auxiliary models (`buffalo_l`)**
- **Restore Ultra & Enhancers (`RestoreFormer++`, `GPEN 256 Pro`, `CodeFormer`, `GFPGAN`)**
- **XSeg, XSeg 3, BiSeNet FaceParser, & RealityUX Fusion**
- **Face Recognition models (`w600k_r50`, `AdaFace`, `CSCS`)**

Three critical per-frame resource allocation defects were discovered and eliminated:
1. **Per-frame IOBinding Recreation in `Mask_XSeg` and `Mask_XSeg3`:** Both mask processors previously invoked `sess.io_binding()` on **every single face call**, allocating and destroying ORT C++ IOBinding objects and re-binding device memory inside the frame loop. Caching `_cached_io_binding` with pre-bound output buffers on the session eliminated all per-frame binding allocations while ensuring exact numerical equivalence (`max |delta| = 0.0`).
2. **Per-face Native Thread Spawning in `Mask_RealityUX`:** `Mask_RealityUX.Run` previously spawned and joined a native OS `threading.Thread` on every face to parallelize XSeg and BiSeNet. This has been replaced with a persistent, reusable worker pool (`concurrent.futures.ThreadPoolExecutor`), eliminating OS thread churn and context switching overhead.
3. **Per-face IOBinding Allocation in `Enhance_CodeFormer`:** Restored persistent binding reuse per session while preserving dynamic fidelity weight updates.

A centralized, thread-safe Model Lifecycle Diagnostics module ([`app/roop/model_lifecycle.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/model_lifecycle.py)) was constructed, tracking and formatting the exact 8 telemetry fields required by the architectural specification.

---

## 2. Audit of Model Lifecycle Requirements

| Requirement | Implementation & Audit Status | Verification Mechanism |
|---|---|---|
| **1. Models loaded once and reused** | All processors (`Mask_XSeg`, `Mask_XSeg3`, `Enhance_RestoreUltra`, `Enhance_GPEN256Pro`, `OptimizedInferenceSession`, `FaceAnalysis`) verify `if self.session is not None: return`. Redundant loads are bypassed. | Tested in unit tests & [`verify_model_lifecycle.py`](file:///G:/pinokio/api/roop-ultimate/tools/verify_model_lifecycle.py) |
| **2. Sessions not recreated per frame** | `_cached_io_binding` persisted across frames in `Mask_XSeg`, `Mask_XSeg3`, `Enhance_CodeFormer`; persistent executor in `Mask_RealityUX`. | Verified 0 new allocations across consecutive inference passes |
| **3. TensorRT engines cached correctly** | `"trt_engine_cache_enable": True`, cached under `~/.cache/roop-ultimate/trt_cache/<namespace>` incorporating GPU, compute capability, precision, and ORT/TRT versions. | Verified cache HIT / BUILT detection across runtime |
| **4. Timing caches reused** | `"trt_timing_cache_enable": True`, persisted in cache namespace to eliminate repetitive kernel benchmarking. | Preserved across `inference_engine.py` and `trt_engine.py` |
| **5. Provider configuration centralized** | Centralized via `roop.precision_policy.providers_for` and `roop.backend_manager.build_session_with_fallback`. | Unified provider resolution and fallback logging |
| **6. Device placement explicit** | All CUDA operations pass explicit `device_id` (`cuda:0` / `cuda:1` / `cpu`). Devicenames sanitized to ORT types (`cuda`, `cpu`). | No reliance on ambient defaults |
| **7. Precision policy consistent** | `precision_policy.py` maps models to FP16, FP32, or mixed with safety rules (e.g. inswapper locked to FP32, GPEN 1024 locked to FP32). | Consistent across both RTX 4070 and RTX 3060 profiles |
| **8. Model warm-up before timing** | `predictor.verify_and_warmup` and `OptimizedInferenceSession.warmup()` execute dummy passes on startup before timing begins. | Engine build latency paid at initialization, not on frame 0 |
| **9. Failed providers deterministic fallback** | Step-down chain: `TensorRT -> CUDA -> CPU` with recorded degradations in `backend_manager`. | Failures logged without silent crash or silent unswapped frames |
| **10. GPU memory released appropriately** | `Release()` implementations clear sessions, pools, cached bindings, and trigger `torch.cuda.empty_cache()` on teardown or device switch only. | Memory released on teardown, not inside frame loops |

---

## 3. Runtime Diagnostics Telemetry Table

The centralized diagnostics module outputs the structured table with all 8 required columns:

```
+---------------+--------+-----------------------+-----------+---------------------+-----------------------------+-----------+---------------------+
| MODEL         | DEVICE | PROVIDER              | PRECISION | INPUT SHAPE         | ENGINE CACHE                | VRAM COST | INITIALIZATION TIME |
+---------------+--------+-----------------------+-----------+---------------------+-----------------------------+-----------+---------------------+
| mask_xseg     | cuda   | CUDAExecutionProvider | fp32      | unk__1491x256x256x3 | N/A (CUDAExecutionProvider) | pooled    | initialized         |
| mask_xseg3    | cuda   | CUDAExecutionProvider | fp32      | unk__1491x256x256x3 | N/A (CUDAExecutionProvider) | pooled    | initialized         |
| mask_realityux| cuda   | CUDAExecutionProvider | fp32      | 1x3x512x512         | N/A (CUDAExecutionProvider) | pooled    | initialized         |
| restore_ultra | cuda   | CUDAExecutionProvider | fp32      | 1x3x512x512         | N/A (CUDAExecutionProvider) | pooled    | initialized         |
| inswapper_128 | cuda:0 | CUDAExecutionProvider | fp32      | 1x3x128x128, 1x512  | N/A (cuda)                  | 0.4 MB    | 459.9 ms            |
+---------------+--------+-----------------------+-----------+---------------------+-----------------------------+-----------+---------------------+
```

*(Note: Under active TensorRT compilation on compatible models, `ENGINE CACHE` reports `HIT (reused)` or `BUILT (cached)`, and `PROVIDER` displays `TensorrtExecutionProvider`.)*

Structured JSON telemetry is continuously exported to `benchmark_stage2_lifecycle.json` for automated regression tracking across future stages.

---

## 4. Dual-Hardware Profile Compatibility

1. **Main Device (RTX 4070 12GB Desktop)**:
   - Full TensorRT engine and timing cache caching enabled under `~/.cache/roop-ultimate/trt_cache/fp16_NVIDIA_GeForce_RTX_4070_sm89_...`.
   - Multi-context session pooling enabled (`perf_trt_pool: 2`, `perf_detmask_pool: 2`).
   - Reusable `ThreadPoolExecutor` in `Mask_RealityUX` concurrent with XSeg inference.
2. **Secondary Device (RTX 3060 6GB Laptop)**:
   - Sub-7GB CUDA safety rules strictly preserved (`0 / 0` pool size, single context).
   - BiSeNet auxiliary parser skipped in `Mask_RealityUX` when memory exceeds threshold to maintain system RSS under 2.5 GB.
   - Deterministic step-down to CUDA FP32 without TensorRT thrashing or OOM paging.

---

## 5. Verification & Test Results

- **Unit & Integration Suite ([`tests/test_stage2_model_lifecycle.py`](file:///G:/pinokio/api/roop-ultimate/tests/test_stage2_model_lifecycle.py)):** 8/8 passed in 4.25s.
  - `test_model_lifecycle_required_fields`: Verified presence and formatting of all 8 columns.
  - `test_track_model_lifecycle_context_manager`: Verified elapsed latency and cache tracking.
  - `test_check_engine_cache_status`: Verified HIT / BUILT / non-TRT classification.
  - `test_mask_xseg_iobinding_reuse`: Verified 100% IOBinding reuse with zero per-frame re-allocations.
  - `test_mask_xseg3_iobinding_reuse`: Verified IOBinding persistence.
  - `test_mask_realityux_executor_reuse`: Verified persistent executor and leak-free teardown.
  - `test_enhance_codeformer_iobinding_reuse`: Verified CodeFormer binding reuse.
  - `test_export_model_lifecycle_json`: Verified structured JSON export.
- **Stage 0 Benchmark Suite ([`tests/test_stage0_benchmark.py`](file:///G:/pinokio/api/roop-ultimate/tests/test_stage0_benchmark.py)):** 5/5 passed in 7.71s (zero regressions).
- **Standalone Runtime Diagnostics:** Verified with `python app/run.py --diagnose-runtime`.
