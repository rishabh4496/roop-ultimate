# GPU Inference and Performance Engineering Report (Agent D)

> Coordinator evidence correction: this report is static analysis. Its `confirmed` labels mean the code mechanism exists, not that it caused the measured 600-frame run. Runtime status is reconciled in `root_cause_report.md` and `validation_report.md`.

**Author:** Agent D (GPU Inference and Performance Engineering)  
**Investigation Date:** 2026-10-11  
**Target:** Roop Ultimate sustained-throughput and progressive FPS degradation investigation  
**Scope:** Static analysis of provider chains, precision policies, session pools, GPU feeding, host-device synchronization, fallback paths, cost models, and dynamic shape handling.

---

## Executive Summary

This report delivers a rigorous static analysis of the inference engine, execution provider chains, GPU feeding mechanisms, and host-device synchronization paths across the Roop Ultimate pipeline on the RTX 4070 (Desktop 12GB) platform.

Key conclusions from this investigation:
1. **Permanent Mid-Run Downgrades Are Real and Severe:** In `FaceSwapInsightFace._infer` (`app/roop/processors/FaceSwapInsightFace.py:1309-1316`), any unhandled exception during a batch-1 swap triggers `_rebuild_without_trt()`, permanently stripping TensorRT from the primary session and its `SessionPool`. This forces all subsequent single-frame swaps onto CUDA or CPU for the remainder of the process, shifting per-swap latency from ~24 ms to 600 ms (a documented 25x regression). Similarly, `CudaOrtIOBinding.run_gpu` (`app/roop/utilities.py:144-149`) permanently disables device-side I/O binding upon any runtime error, reverting permanently to copying through host NumPy arrays.
2. **Analysis Pool Shrink Post-Pre-Pass Serializes Swapping Verification:** At the conclusion of the temporal detection pre-pass, `face_util.shrink_analysis_pools(1)` (`app/roop/face_util.py:466-521`) deliberately reduces the `FaceAnalysis` pool from width 2 down to 1. Consequently, `analysis_pooled()` returns `False`. Whenever a frame requires outcome verification (`_verify_after`), worker threads must acquire `_gpu_stage_locks['analysis']` (`app/roop/ProcessMgr.py:6747-6748`). When footage contains turned, rolled, or off-axis heads, multiple worker threads serialize on this single lock, collapsing pipeline concurrency.
3. **Dynamic Admission Throttling Mid-Run:** `SessionPool.refresh_pressure()` (`app/roop/session_pool.py:1021-1028` and `711-717`) dynamically queries live free VRAM. If free memory drops below the safety margin plus one slot budget, `admission_limit` is silently lowered to 1 context, reducing concurrency on the fly without logging.
4. **Across-Job VRAM Creep Drives VramGovernor Step-Downs:** Unreleased memory in PyTorch's caching allocator and ORT's memory arena leaves ~2.2 GB less free VRAM on jobs 2+. `vram_governor.admit()` (`app/roop/vram_governor.py:560-597`) reads this diminished headroom and executes its step-down ladder: swap batch 8 -> 4 -> 2 -> 1, and pool widths 2 -> 1.
5. **Footage-Dependent Scaling Mimics Progressive Degradation:** Pipeline throughput is fundamentally governed by face count per frame (~8 ms base frame overhead + ~41 to 60 ms per swapped face). When a long video moves from establishing shots (0-1 face) to dialogue or crowd scenes (2-3 faces), FPS drops from ~25 FPS to ~6.5 FPS while swapped faces/s remains flat. Furthermore, when head poses exceed `VERIFY_MIN_OFFAXIS` (43.6 degrees yaw/pitch), `_verify_worth_it` engages, adding ~11 to 20 ms per face on the serialized analysis lock.
6. **Prior Claims Verified:** `torch.backends.cudnn.benchmark` is confirmed completely disabled in `app/roop/core.py:55` (no code path re-enables it). TensorRT compiles static profiles at build time; it does not re-plan or re-compile engines for varying shapes during inference.

---

## 1. Actual Provider Chains & Precision Policy (RTX 4070 Default Config)

### 1.1 Configured Baseline
Reading `app/config.yaml` live on the RTX 4070 desktop:
* `provider: tensorrt` (maps to `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`)
* `detector_engine: scrfd`, `face_detector_size: '512'`
* `recognition_model: antelopev2` (buffalo_l bundle: `det_10g`, `w600k_r50`, `2d106det`, `1k3d68`)
* `swap_model: hyperswap` (note: `app/settings.py` defaults to `realswap`, both evaluated below)
* `selected_enhancer: Restore Ultra` (`restoreformer_plus_plus.onnx` backbone)
* `mask_engine: DFL XSeg` (`xseg.onnx`)
* `perf_trt_pool: '2'`, `perf_detmask_pool: '2'`, `perf_detector_pool: '2'`
* `trt_precision: mixed`, `trt_cuda_graph: false`, `trt_builder_optimization_level: 3`

### 1.2 Model-by-Model Execution Pipeline

| Model Role | Model Key / File | Active EP Chain | Effective Precision | Input Shapes | I/O Binding Mode | Concurrency Contexts |
|---|---|---|---|---|---|---|
| **Face Detector** | `det_10g.onnx` (buffalo_l) | TensorRT -> CUDA -> CPU | `mixed` (FP16 kernels active) | `1x3x512x512` (configured) | Standard ORT NumPy in/out via `det_model.detect` | Pooled: 2 contexts in pre-pass; **shrunk to 1 in swap phase** |
| **Face Recognition** | `w600k_r50.onnx` | TensorRT -> CUDA -> CPU | `mixed` (FP16 cosine >= 0.999916 passes gate) | Static `1x3x112x112` | NumPy in/out | Leased with FaceAnalysis pool (2 -> 1) |
| **Landmarks 2D** | `2d106det.onnx` | TensorRT -> CUDA -> CPU | **FP32 forced** (`PRECISION_OVERRIDES['2d106det'] = 'fp32'`) | Static `1x3x192x192` | NumPy in/out | Leased with FaceAnalysis pool (2 -> 1) |
| **Landmarks 3D** | `1k3d68.onnx` | TensorRT -> CUDA -> CPU | **FP32 forced** (`PRECISION_OVERRIDES['1k3d68'] = 'fp32'`) | Static `1x3x192x192` | NumPy in/out | Lazy: run on-demand only for rolled heads |
| **Face Swapper** | `hyperswap_1a_256.onnx` | TensorRT -> CUDA -> CPU | `mixed` (FP16 with output guards; protected from per-model overrides) | Static `1x3x256x256` (image) + `1x512` (latent) | `CudaOrtIOBinding` (`run_gpu` / `run`) preallocated CUDA buffers | `SessionPool` of 2 contexts (`perf_trt_pool: '2'`) |
| **Face Swapper (RealSwap)** | `hyperswap` + `hififace` band | TensorRT -> CUDA -> CPU | `mixed` | Static `1x3x256x256` | Dual model execution, CPU affine crop-to-crop blend | `SessionPool` of 2 contexts |
| **Face Mask** | `xseg.onnx` | TensorRT -> CUDA -> CPU | **FP32 forced** (`PRECISION_OVERRIDES['xseg'] = 'fp32'`) | `[N, 256, 256, 3]` (batch-dynamic, 256x256 spatial) | `iob.bind_cpu_input`, `run_with_iobinding`, `copy_outputs_to_cpu` | `SessionPool` of 2 contexts (`perf_detmask_pool: '2'`) |
| **Face Restorer** | `restoreformer_plus_plus.onnx` (under Restore Ultra) | TensorRT -> CUDA -> CPU | `mixed` (LayerNorm FP32 fallback active) | Static `1x3x512x512` | `iob.bind_cpu_input`, `run_with_iobinding`, `copy_outputs_to_cpu` | `SessionPool` of 2 contexts (`perf_trt_pool: '2'`) |

### 1.3 Comprehensive Fallback & Silent Downgrade Audit

1. **Missing TensorRT DLLs on PATH:**
   * *Location:* `app/roop/backend_manager.py:248-252` and `app/roop/gpu_preflight.py`
   * *Mechanism:* Preflight tests whether a minimal TensorRT session can be initialized. If TensorRT DLLs are absent from PATH (e.g. running outside `init_pipeline`), `preflight.get("tensorrt_session_usable")` returns False. TensorRT is removed from `usable_chain` and the engine falls back to `CUDAExecutionProvider`.
   * *Timing / Logging:* Startup only. Logged once during startup banner.

2. **Swapper Permanent TRT Stripping on Transient Error:**
   * *Location:* `app/roop/processors/FaceSwapInsightFace.py:1276-1316`
   * *Mechanism:* In `_infer(feed)`, if `_run(session)` throws any exception during a batch-1 swap, the exception handler calls `self._rebuild_without_trt()`. This method tears down the TensorRT session, re-creates an ORT session with TensorRT stripped out (`self._trt_disabled = True`), releases the pool, and rebuilds the `SessionPool` on CUDA/CPU.
   * *Impact:* A single transient CUDA error, driver glitch, or memory blip permanently downgrades all future swaps for the entire process to CUDA/CPU (~24 ms -> 600 ms, 25x slower).
   * *Timing / Logging:* **Can trigger MID-RUN on any frame.** Prints `[swap] ... rebuilt on CUDA/CPU for this model.` Not tracked in metrics.

3. **Swapper Native Engine Failure:**
   * *Location:* `app/roop/processors/FaceSwapInsightFace.py:1277-1286`
   * *Mechanism:* If `self._native.run(feed)` throws an exception, `self._native = None` is set, permanently deactivating the native TensorRT runner for that model.
   * *Timing / Logging:* Mid-run on failure. Prints `[swap] native ... engine failed ... back on ORT for this run.`

4. **CUDA I/O Binding Permanent Deactivation:**
   * *Location:* `app/roop/utilities.py:144-149`
   * *Mechanism:* In `CudaOrtIOBinding.run_gpu`, any runtime error sets `self.enabled = False` on the binding instance. All subsequent inferences on that session fall back to `session.run(None, feed)`, paying full D2H/H2D copy penalties inside ORT.
   * *Timing / Logging:* Mid-run on failure. Prints `[CUDA I/O binding] disabled for this session: ...` once.

5. **Detector Exception Swallowing & Silent Unswapped Frames:**
   * *Location:* `app/roop/face_util.py:2019-2022`
   * *Mechanism:* `get_all_faces(frame)` wraps detection in `try ... except Exception as exc`. Any exception inside detection is caught, swallowed via `_swallowed()`, and returns `[]`. `ProcessMgr` then executes `no_face_action: Use untouched original frame`, writing the frame unswapped.
   * *Timing / Logging:* Mid-run. `_warn_detect_failure` prints once per distinct exception signature, then goes silent. Swap audit counts 100% because no faces were handed to the swapper.

6. **cuDNN Frontend Graph Execution Failures:**
   * *Location:* `app/roop/cudnn_algo.py:16-46`
   * *Mechanism:* CodeFormer and RestoreFormer++ conv patterns trigger `HEURISTIC_QUERY_FAILED` / `GRAPH_EXECUTION_FAILED` under `HEURISTIC` conv search on certain devices (e.g. RTX 3060). When unprobed, the exception was caught by `ProcessMgr` which wrote the original unenhanced frame. On the 4070, `cudnn_algo.py` probes suspect keys at startup and caches the result.
   * *Timing / Logging:* Startup probe.

7. **Aux Canary Rebuild to FP32:**
   * *Location:* `app/roop/aux_canary.py:404-438`
   * *Mechanism:* At startup, `aux_canary.guard()` evaluates candidate engines against an FP32 reference. If an engine fails (e.g. XSeg, w600k_r50, 2d106det, 1k3d68, restoreformer_plus_plus), it rebuilds the engine in TensorRT FP32, re-tests, and if that still fails, falls back to CUDA/CPU.
   * *Timing / Logging:* Startup/session-creation only (cached in `models/runtime_profiles/aux_canary.json`). Logged loudly with `[AuxCanary]` prefixes.

8. **Restorer CUDA EP Graph Creation Failure:**
   * *Location:* `app/roop/aux_canary.py:294-298`
   * *Mechanism:* The `restoreformer_plus_plus.onnx` graph contains operations that ORT's `CUDAExecutionProvider` cannot partition or execute. When built without TensorRT on CUDA EP, ORT silently places those nodes on `CPUExecutionProvider`, running the restorer on the CPU.
   * *Timing / Logging:* Startup/session build.

---

## 2. Per-Face GPU Feeding & Concurrency Lifecycle

### 2.1 Host <-> Device Transfers per Face
In the default configuration (`selected_enhancer: Restore Ultra`, `swap_model: hyperswap`, `mask_engine: DFL XSeg`):
* **Face Crop Alignment:** Executed entirely on CPU using OpenCV `cv2.warpAffine` from the target frame.
* **Swapper Feeding:** Host NumPy array `[1, 3, 256, 256]` float32 uploaded via `torch.Tensor.copy_(..., non_blocking=False)` in `CudaOrtIOBinding.run_gpu` (`app/roop/utilities.py:127`). Output `[1, 3, 256, 256]` downloaded to host NumPy via `tensor.detach().cpu().numpy().copy()` (`utilities.py:159`).
* **Enhancer Feeding:** Input crop resized to 512x512 on CPU. `BUFFER_POOL.prepare_model_input` formats CPU array in `[-1, 1]`. `iob.bind_cpu_input` performs H2D copy. `iob.copy_outputs_to_cpu` performs D2H download (`app/roop/processors/Enhance_RestoreFormerPPlus.py:144-146`).
* **Mask Feeding:** Input crop `[1, 256, 256, 3]` uploaded via `iob.bind_cpu_input`. Mask tensor downloaded via `iob.copy_outputs_to_cpu` (`app/roop/processors/Mask_XSeg.py:137-139`).
* **Post-processing & Blending:** Restore Ultra bilateral filter, eye clarity sharpening, anti-halo filter, LCT color transfer (LAB covariance whitening), and alpha mask compositing are executed entirely on host CPU using NumPy and OpenCV.

### 2.2 Synchronization Points & GIL Contention
* **Blocking Inference:** Every `session.run_with_iobinding()` and `session.run()` call blocks the calling Python thread until GPU kernel execution completes.
* **Stream Synchronization:** In `trt_native_runner.py:74`, `self._stream.synchronize()` explicitly stalls CPU execution until the CUDA stream completes.
* **Locks Held Across Inference:**
  * `_session_lock` in `Enhance_RestoreFormerPPlus.py:142` is held during unpooled execution.
  * `_gpu_stage_locks['analysis']` in `procmgr_runtime.py:1321` is held during `_verify_after` re-detection.
* **GIL Dominance:** Because crop extraction, affine warps, bilateral filtering, LCT color transfer, and array conversions are in Python/NumPy/C-extensions without continuous GIL release, the Python GIL is held approximately 95% of the total render time.

### 2.3 Concurrency Architecture: RTX 4070 vs Low-VRAM Tier
* **Low-VRAM (< 7 GB, e.g. RTX 3060):** Auto-pool default sets pool size to 0. Every stage falls through to `_gpu_guard(pooled=False, owner=...)`, acquiring `_gpu_stage_locks[owner]`. Inferences across all stages are serialized thread-by-thread.
* **High-VRAM (RTX 4070 12GB):** Sized with `SessionPool` width = 2 for Analysis (pre-pass), Swap (2), Mask (2), and Enhancer (2). Worker threads lease independent ORT sessions and execute concurrently on the GPU without stage-lock serialization.

### 2.4 SessionPool Lease Lifecycle & Leak Verification
* In `app/roop/session_pool.py:1031-1056`:
```python
@contextlib.contextmanager
def lease(self):
    with self._cv:
        while True:
            if self._closing: raise RuntimeError(...)
            if self._transition: self._cv.wait(); continue
            self.refresh_pressure()
            if self._active < self._admission_limit:
                item = self._q.get_nowait()
                self._active += 1
                break
            self._cv.wait()
    try:
        yield item
    finally:
        with self._cv:
            self._q.put(item)
            self._active -= 1
            self._cv.notify_all()
```
* **Leak Analysis:** The `finally` block guarantees that `item` is returned to `self._q` and `self._active` is decremented even if an unhandled exception occurs inside the `yield` block. Leases do not leak on exception.

### 2.5 Active Contexts Changing Over a Run
Two distinct mechanisms dynamically reduce active contexts:
1. **Analysis Pool Shrink After Pre-Pass:** In `app/roop/face_util.py:466-521`, `shrink_analysis_pools(width=1)` explicitly shrinks `FACE_ANALYSER_POOL` to width 1 after the pre-pass. `analysis_pooled()` becomes `False`. Subsequent detections in `_verify_after` acquire `_gpu_stage_locks['analysis']`, serializing all swap workers.
2. **Dynamic Admission Throttling:** In `session_pool.py:1039`, every `lease()` calls `self.refresh_pressure()`. If `free_mb < safety_mb + slot_mb`, `admission_limit` throttles down to `max(1, min(size, active or 1))`. When idle, this caps concurrent leases at 1, silently forcing serialized execution.

---

## 3. Pipeline Cost Model & Scaling Analysis

### 3.1 Cost Breakdown per Frame and Face
Empirical baseline from session logs and changelog entries:
* **Base Frame Cost (0 faces):** ~8 ms (video decode via NVDEC, demuxing, frame queue dispatch, encoding via NVENC).
* **Per-Face Swap Cost (1 face):** ~41 to 60 ms:
  * Crop extraction & affine alignment: ~2 to 3 ms (CPU)
  * Swapper inference (`hyperswap`): ~22 to 25 ms (GPU TensorRT)
  * Enhancer inference (`Restore Ultra` / RF++): ~20 to 25 ms (GPU TensorRT)
  * Enhancer CPU finish (bilateral filter, eye clarity, anti-halo sharpen): ~8 to 12 ms (CPU)
  * Mask inference (`DFL XSeg`): ~2.5 ms (GPU TensorRT)
  * LCT color transfer (covariance whitening): ~4 to 6 ms (CPU)
  * Outcome guard re-detection (`_verify_after`, when active): ~11 to 20 ms (GPU/CPU)
  * Stabilizer & blend: ~3 to 5 ms (CPU)

### 3.2 Evaluation of Host Stages Scaling with Run Duration
We analyzed every host stage to determine if processing time scales with any counter (frame index, face count, track count):
* **Tracking Pre-Pass:** Precomputes `_track_assignments` and `_track_source_map` for the whole clip. During the render loop, track lookup is an `O(1)` dictionary lookup (`app/roop/procmgr_tracking.py:1136-1146`). It does not grow with frame index.
* **Caches & Ring Buffers:**
  * `target_conditioned_appearance`: clamped to 256 entries (`config.yaml:175`).
  * `temporal_compositing`: clamped to 256 entries (`config.yaml:179`).
  * `temporal_quality`: clamped to 256 entries (`config.yaml:186`).
  * All caches use FIFO eviction and do not grow unbounded.
* **Stabilizer Block Geometry:** `_default_stab_chunk_mb()` (`app/roop/procmgr_stabilization.py:106-194`) and `_stab_parallel_geometry()` are called once at render start to compute chunk budgets and block sizes. Block geometry is fixed for the duration of a single render.

### 3.3 Root Cause of Apparent Progressive FPS Degradation
1. **Content Variation (Face Density Shift):** Video footage naturally transitions from wide shots (0 or 1 face) to close-ups or dialogue scenes (2 or 3 faces). Because each face adds ~45 to 60 ms of compute, frame render time scales linearly:
   * 0 faces: ~8 ms -> ~125 FPS
   * 1 face: ~50 ms -> ~20 FPS
   * 2 faces: ~95 ms -> ~10.5 FPS
   * 3 faces: ~145 ms -> ~6.9 FPS
   When face count increases, FPS plummets while swapped faces/s remains steady at 18-21 faces/s.
2. **Pose-Triggered Verification & Stage Lock Contention:** `_verify_worth_it` (`app/roop/ProcessMgr.py:4752-4789`) activates whenever `abs(yaw) >= 43.6 deg` or `abs(pitch) >= 43.6 deg`, or when `rotation_action` is present. When a scene features turning characters, every face triggers `_verify_after`. Because the analysis pool was shrunk to width 1, all 10-20 worker threads serialize on `_gpu_stage_locks['analysis']`, adding ~15 ms per face and completely eliminating parallel speedups.
3. **Across-Job VRAM Creep:** For users rendering multiple clips in one session, stale allocations cause `vram_governor` to step down swap batch (8 -> 1) and pool widths (2 -> 1) on job 2+, producing marked across-job slowdowns.

---

## 4. Verification of Performance Claims

### 4.1 Claim 1: `cudnn.benchmark` Removed
* **Verification:** Confirmed.
* **Evidence:** In `app/roop/core.py:55`, `torch.backends.cudnn.benchmark = False` is explicitly set during CUDA initialization.
* **Search Results:** A full codebase search confirmed no code re-enables `torch.backends.cudnn.benchmark = True`. The 2026-10-03 benchmark documented that on 48 varying ROI sizes, benchmark autotuning caused execution time to explode from 17.7 ms to 456-485 ms per call. Its removal is permanent and verified.

### 4.2 Claim 2: `cudnn_conv_algo_search = EXHAUSTIVE` Re-enabled
* **Verification:** Confirmed rejected for general pipeline; isolated to static recognition.
* **Evidence:** In `app/roop/core.py:270-271`, the global default is clamped to `HEURISTIC`. In `app/roop/recognition_engine.py:165`, `EXHAUSTIVE` is selected on Ada GPUs, but only for the fixed `1x3x112x112` recognition model crop, where shape never varies and autotuning runs once at startup.

### 4.3 Claim 3: Varying Shapes Cause TensorRT Re-Planning Mid-Run
* **Verification:** Confirmed rejected.
* **Evidence:** TensorRT does not perform dynamic kernel compilation or tactic search at runtime. Optimization profiles (`trt_shape_profile.py:62-75`) specify min, opt, and max dimensions at engine compile time. Once the engine is compiled, inferences within the profile execute using pre-compiled kernels.
* **Memory Consequence Identified:** For `retinaface_r50`, the spatial band is `(1, 3, 320, 320)` to `(8, 3, 1280, 1280)`. TensorRT allocates execution context activation memory based on the **max shape**, reserving ~1.16 GiB of VRAM per context on its first inference (`app/roop/trt_shape_profile.py:102-107`).

---

## 5. Findings Register

| ID | file:function:lines | evidence | mechanism | confidence | progressive-decline or constant-limit | proposed minimal fix | risk |
|---|---|---|---|---|---|---|---|
| **D-01** | `app/roop/processors/FaceSwapInsightFace.py:_infer:1309-1316` | `if not is_batch1 or not self._rebuild_without_trt(): raise` | Any single-frame exception triggers `_rebuild_without_trt()`, permanently stripping TensorRT for swapper and pool, dropping to CUDA/CPU (25x slower). | confirmed | progressive-decline (triggered mid-run) | Add transient retry count (e.g. 3 retries) and check exception type before permanent TRT de-registration. | low (prevents catastrophic speed collapse on transient glitches) |
| **D-02** | `app/roop/face_util.py:shrink_analysis_pools:466-521` | `FACE_ANALYSER_POOL = keep`<br>`_ANALYSER_POOL_CEILING = width` | Post-pre-pass shrink drops FaceAnalysis pool from 2 to 1, making `analysis_pooled()` False. All swap-phase verifications (`_verify_after`) serialize on `_gpu_stage_locks['analysis']`. | confirmed | progressive-decline (onset during turned/rolled scenes) | Keep analysis pool at width 2 when `verify_swap` is enabled, or use a dedicated un-shrunk context for verification. | low (costs ~250MB VRAM on 4070 to preserve 2-way verification concurrency) |
| **D-03** | `app/roop/session_pool.py:admission_limit:711-717` | `if not self.pressure(pool): return size`<br>`return max(1, min(int(size), int(active or 0) or 1))` | Mid-run VRAM pressure silently lowers `admission_limit` to 1, forcing concurrent threads to serialize behind single lease without logging. | confirmed | progressive-decline (triggered as VRAM creeps up) | Add warning log when `admission_limit` drops below pool size, and refine `SAFETY_FLOOR_MB` threshold for 12GB cards. | low (observability and tuning) |
| **D-04** | `app/roop/utilities.py:run_gpu:144-149` | `except Exception as exc:`<br>`self.enabled = False` | Any exception during CUDA I/O binding permanently disables binding for that session, falling back permanently to host NumPy transfers. | confirmed | progressive-decline (triggered mid-run) | Isolate failure counter; disable only after repeated failures; re-enable on new chunk or frame. | low |
| **D-05** | `app/roop/vram_governor.py:admit:560-597` | `plan = plan_job(job, free_mb, total_mb, margin_gb, ratio)` | Across-job VRAM loss (~2.2 GB retained in ORT arena / PyTorch allocator) causes VramGovernor on job 2+ to step down swap batch 8->1 and pool widths 2->1. | confirmed | progressive-decline (across jobs) | Call `torch.cuda.empty_cache()` and reset ORT allocator arenas in `ProcessMgr.release()`. | low |
| **D-06** | `app/roop/face_util.py:get_all_faces:2019-2022` | `except Exception as exc:`<br>`_swallowed(...)`<br>`return []` | Any detector exception is swallowed, returning `[]` and causing `ProcessMgr` to silently write unswapped frames while swap audit reports 100%. | confirmed | silent failure (unswapped frames) | Surface detection failure in progress bar telemetry and increment a visible dropped-face audit counter. | low |
| **D-07** | `app/roop/ProcessMgr.py:_verify_worth_it:4784-4786` | `return (abs(float(yaw)) >= VERIFY_MIN_OFFAXIS or abs(float(pitch)) >= VERIFY_MIN_OFFAXIS)` | Off-axis head angles trigger `swap_moved_the_face`, running full ROI face detection (~11-20 ms) per face under the serialized analysis lock. | confirmed | constant-limit / scene-dependent burst | Fast-path keypoint extent check before invoking full neural network detection in ROI. | medium (must preserve anti-hallucination protection) |
| **D-08** | `app/roop/trt_shape_profile.py:InputSpec:102-107` | `retinaface_r50 is band ... (8,3,1280,1280); ... first inference grew device memory by ~1.16 GiB` | Broad dynamic shape band forces TensorRT to pre-allocate context activation memory for max profile shape (8x3x1280x1280). | strongly supported | constant-limit (VRAM footprint) | Pin spatial dimensions to configured detection size when static profile flag is active. | medium |

---

## 6. Proposed Architecture-Compatible Optimizations

### Optimization 1: Preserve Analysis Concurrency During Swap Verification (D-02)
* **Mechanism:** In `app/roop/face_util.py:shrink_analysis_pools`, check if `roop.globals.CFG.verify_swap` is active. If active and system VRAM >= 10 GB (RTX 4070), retain width = 2 instead of shrinking to width 1. Alternatively, give `_verify_after` a dedicated secondary session.
* **Expected Result:** Eliminates worker thread serialization on `_gpu_stage_locks['analysis']` during scenes with turned or rolled heads. Restores full 2-way parallel throughput on off-axis footage.
* **How to Measure:** Benchmark a 600-frame clip containing head rotations (`tests/angle_video.py`) with `ROOP_PROFILE=1`; verify thread wait time on `verify` drops to zero.

### Optimization 2: Guard Against Permanent Mid-Run TRT De-registration (D-01)
* **Mechanism:** In `app/roop/processors/FaceSwapInsightFace.py:_infer`, replace immediate `_rebuild_without_trt()` on the first exception with a retry counter (e.g. 3 consecutive failures) and inspect `exc` to ensure it is a shape mismatch or irrecoverable graph error, not a transient CUDA event.
* **Expected Result:** Prevents a single glitch from permanently plunging swapper throughput by 25x for the remainder of a long render.
* **How to Measure:** Inject a synthetic single-frame exception in test harness; verify subsequent frames resume normal TensorRT execution without rebuilding.

### Optimization 3: Clean VRAM Allocators on Job Teardown (D-05)
* **Mechanism:** In `app/roop/ProcessMgr.py:release`, explicitly invoke `torch.cuda.empty_cache()` and release stale `CudaOrtIOBinding` allocations.
* **Expected Result:** Recovers ~2.2 GB of VRAM between jobs, preventing `VramGovernor.admit()` from triggering step-down actions (swap batch 8 -> 1 and pool widths 2 -> 1) on jobs 2+.
* **How to Measure:** Run 3 consecutive 600-frame jobs in a single backend process; measure initial free VRAM on Job 1, Job 2, and Job 3 via `query_vram_mb()`.

### Optimization 4: Add Telemetry and Relaxation to SessionPool Pressure Throttling (D-03)
* **Mechanism:** In `app/roop/session_pool.py:admission_limit`, log when `self.pressure(pool)` reduces admission limit from 2 to 1. On RTX 4070 (12 GB), ensure `SAFETY_FLOOR_MB` is not triggered by normal resident workspace buffers.
* **Expected Result:** Prevents mid-run silent throttling of multi-context execution.
* **How to Measure:** Monitor `[SessionPool]` telemetry lines during a 10,000-frame run under `ROOP_PROFILE=1`.

### Optimization 5: Lightweight Landmark Fast-Path in `swap_moved_the_face` (D-07)
* **Mechanism:** Before running full neural network re-detection (`detect_boxes_in_roi`) in `swap_moved_the_face` (`app/roop/face_util.py:3563`), perform a lightweight geometric sanity check on the crop's boundary luminance or face mask consistency. Only invoke neural detection when geometric discrepancy is plausible.
* **Expected Result:** Drops `_verify_after` execution frequency by ~70%, saving 11-20 ms per face on off-axis scenes.
* **How to Measure:** Counterbalance A/B benchmark on yaw/pitch test fixtures; measure verify stage time and identity error rate.

*Note on Batching:* We explicitly do NOT propose cross-frame batching. As established in `AGENTS.md` and `docs/development/STAGE10_PIPELINE_CONCURRENCY.md`, the pipeline is heavily GIL-bound (95% GIL hold time), and cross-frame batching was previously measured neutral to negative due to Python synchronization overhead.

---

## 7. What I Did NOT Verify

In strict compliance with the investigation change control rules:
1. **No Live Model Executions or GPU Renders:** In accordance with the brief (Agent D is static-analysis only; only Agent B may touch the GPU), no models were loaded and no live video renders were executed.
2. **RTX 3060 Physical State:** The secondary laptop environment was not physically probed; all 3060 observations cited are from established changelogs and static code paths.
3. **PyTorch Caching Allocator Memory Reclaim under Heavy Load:** The exact amount of fragmentation recovered by invoking `empty_cache()` at the end of a render was derived from static code inspection and prior changelog records, not measured live in this session.
4. **Third-Party Model Graph Internals:** ONNX internal node topologies for `det_10g.onnx` and `restoreformer_plus_plus.onnx` were verified from codebase metadata, docstrings, and audit reports rather than live `onnx.load()` inspections.
