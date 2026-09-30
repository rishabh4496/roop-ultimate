# Stage 14: Advanced Settings Redesign & Presets Engine

## 1. Overview & Architecture

Stage 14 establishes a comprehensive settings classification, auditing, conflict resolution, and dynamic presets engine in [`app/roop/advanced_settings_manager.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/advanced_settings_manager.py).

The subsystem addresses four historical failure modes in user configuration:
1. **Settings that do nothing**: e.g., configuring `cpu_ort_inter_threads` when running on CUDA/TensorRT execution providers.
2. **Duplicate settings**: e.g., `perf_detmask_pool` vs `perf_detector_pool`.
3. **Conflicting settings**: e.g., setting `force_cpu=True` while simultaneously requesting `provider="tensorrt"`, or setting `perf_batch_swap="off"` with `perf_batch_max > 1`.
4. **Unsafe combinations**: e.g., enabling `trt_cuda_graph=True` or `perf_trt_pool > 0` on 6GB VRAM laptop GPUs (RTX 3060), risking driver lockups and breaking the strict `<2.5 GB` RSS cap.

---

## 2. The 8 Setting Groups

Every advanced setting is formally classified into one of eight functional groups:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                 Advanced Settings Architecture                  │
  └──────┬────────────┬───────────┬───────────┬───────────┬─────────┘
         │            │           │           │           │
     [QUALITY]  [PERFORMANCE]   [GPU]      [VIDEO]   [DETECTION]
     - Quality    - Threads   - Provider   - NVDEC     - Engine
     - Face Scale - Batching  - Precision  - NVENC     - Threshold
     - Clarity    - RAM limit - Trt Pool   - Codec     - NMS
     - Sharpen                - Safety     - Presets   - Temporal
                              - Pinned                 - Pyramid
         │            │           │
    [TRACKING]  [RESTORATION]  [EXPERT]
    - Step      - Enhancer     - Builder Opt
    - Stitch    - Align        - Aux Streams
    - Demarcate - Expression   - CUDA Graphs
    - Verify    - Gaze / Blink - CV Threads
    - Remeasure - Lighting     - Priority
```

### Complete Group Schema

1. **`QUALITY`**: Visual output fidelity, CRF compression factor, aligned face scale factor, clarity filters, and sharpening.
2. **`PERFORMANCE`**: Concurrency worker threads, auto-thread scaling rules, cross-frame swap batching, and RAM limits.
3. **`GPU`**: Execution providers (TensorRT vs CUDA), numerical precision policies (FP16/MIXED), TRT context pools, VRAM safety margins, CUDA arena strategies, pinned host buffers, and GPU affine warping.
4. **`VIDEO`**: Hardware video decoding (NVDEC), hardware video encoding (NVENC), software encoder presets, container formats, and 10-bit HDR pipelines.
5. **`DETECTION`**: Detection engines (SCRFD-10G, SCRFD-2.5G, RetinaFace, YOLOFace), confidence cutoffs, overlap NMS thresholds, scale pyramids, temporal tracking, and low-light face rescue.
6. **`TRACKING`**: Detection stride step, track stitching, interacting-face demarcation, post-swap outcome verification, upright face remeasurement, and identity confidence matching.
7. **`RESTORATION`**: Super-resolution models (GPEN-512, GPEN-256 Pro, CodeFormer, GFPGAN), pre-alignment, expression strength, eye-gaze follow, blink synchronization, and lighting harmonization.
8. **`EXPERT`**: TensorRT builder optimization levels, auxiliary CUDA streams, CUDA graphs, internal OpenCV thread pools, expression pools, and OS process priority.

---

## 3. Mandatory Metadata Requirements

Every catalogued setting enforces strict structural metadata:
- **`description`**: Human-readable explanation of what the knob does.
- **`default`**: Safe shipping baseline default.
- **`valid_range`**: Permissible continuous min/max bounds or discrete option sets.
- **`hardware_impact`**: Exact resource consumption across VRAM, system RAM, CPU cores, and PCIe bus bandwidth.
- **`quality_impact`**: Concrete visual impact on image sharpness, SSIM, identity preservation, and artifact mitigation.
- **`performance_impact`**: Measured effect on processing FPS, inference latency, and render times.

---

## 4. The 5 Presets

| Preset | Target Objective | Detector | Swapper Batch | Enhancer | Video Quality | Codec Preset |
|---|---|---|---|---|---|---|
| **`FAST`** | Maximize FPS / throughput | `scrfd_2.5g` (det_size=512) | 16 (FP16) | `none` | CRF 22 | `p3` (Fast NVENC) |
| **`BALANCED`** | Optimal sweet spot | `retinaface_r50` (det_size=512) | 8 (MIXED) | `gpen_256` | CRF 18 | `p5` (Balanced) |
| **`QUALITY`** | High fidelity studio output | `scrfd_10g` (det_size=640) | 4 (MIXED) | `gpen_512` | CRF 16 | `p6` (Slow / High Quality) |
| **`ULTRA`** | Maximum perfection | `scrfd_10g` + Pyramid (det_size=640) | 4 (MIXED) | `gpen_512` + CodeFormer FP16 | CRF 14 | `p7` (Highest Quality) |
| **`AUTO`** | Dynamic context resolver | *Dynamic* | *Dynamic* | *Dynamic* | *Dynamic* | *Dynamic* |

---

## 5. Dynamic AUTO Resolver

The `AUTO` preset dynamically calculates optimal parameters based on 8 live factors:

1. **GPU Model & Architecture**:
   - `RTX 4070 Desktop (sm_89)`: Enables dual TRT context pool (`perf_trt_pool=2`), high worker thread count (up to 20), large batch max (16), and CUDA graphs for static models.
   - `RTX 3060 Laptop (sm_86)`: Clamps TRT context pool to `0` (single context under global lock), disables CUDA graphs, caps batch size to `4` (or `2` on 4K), and enforces tight VRAM safety margin of `1.0 GB` to keep system RSS $< 2.5\text{ GB}$.
   - `CPU Fallback`: Reverts provider to `cpu`, disables NVDEC/NVENC, turns off batch swapping, and scales worker threads to physical CPU cores.
2. **VRAM Capacity & Free Headroom**: Adjusts swap batch sizes and restorer resolution dynamically.
3. **System RAM**: Limits concurrency to prevent Windows swapfile thrashing.
4. **Target Face Resolution**: Automatically selects `gpen_512` for 512px renders, `gpen_256` for 256px, and disables enhancer for 128px.
5. **Face Count**:
   - $\ge 2$ faces: Activates `face_demarcate='on'`, `track_stitch='on'`, and locks `temporal_step=1` to prevent identity cross-bleeding.
   - 1 face: Relaxes demarcation and verification to reduce computational overhead.
6. **Enhancer Selection**: Adjusts VRAM safety margins and thread pools when heavy restorers (GPEN-512, CodeFormer) are loaded.
7. **Detector Selection**: SCRFD-10G chosen for complex or crowded scenes; RetinaFace for standard shots.
8. **Video Resolution**:
   - 4K ($3840\times 2160$): Selects `scrfd_10g`, clamps batch size to prevent VRAM exhaustion, and tunes NVENC to `p4` for sustainable throughput.
   - 1080p: Default standard profile.
