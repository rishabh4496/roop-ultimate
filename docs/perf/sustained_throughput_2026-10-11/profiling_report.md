# Profiling Report: Sustained Throughput & Progressive FPS Degradation Investigation

> Coordinator evidence correction: this report contains the harness design and profiling plan plus the measurements explicitly marked as run. The recorded sidecar process-RSS fields were invalid because the original sampler did not aggregate the benchmark process tree. Runtime conclusions are reconciled in `baseline_report.md` and `validation_report.md`.

**Date**: 2026-10-11  
**Author**: Agent B (Runtime Profiling and Throughput)  
**Host Target**: MAIN (RTX 4070 12GB, 31.7 GB physical RAM, 24 physical / 32 logical cores, Windows 11)  
**Workspace**: `G:\pinokio\api\roop-keep` / `G:\pinokio\api\roop-ultimate`  
**Execution Phase**: Read-Only Measurement Plan & Harness Design (Zero models loaded, zero renders run)

---

## Executive Summary & Findings Register

| ID | file:function:lines | evidence | mechanism | confidence | progressive-decline or constant-limit | proposed minimal fix | risk |
|---|---|---|---|---|---|---|---|
| F-INST-01 | `app/roop/pipeline_monitor.py:56-207` | `_w_frames / win`, `_tot_work / win` | `PipelineMonitor` already computes rolling window fps and faces/s per `ROOP_PIPELINE_LOG_EVERY` (default 100). | Confirmed | N/A (Diagnostic tool) | Use `ROOP_PIPELINE_LOG_EVERY=50` or `100` for 1500f run. | Zero |
| F-INST-02 | `app/roop/procmgr_runtime.py:735-751` | `==== STAGE TIMING (ROOP_PROFILE) ====` | `ROOP_PROFILE=1` outputs cumulative wall-clock thread time across all workers at end-of-run; NOT rolling windowed. | Confirmed | Constant limit / Cumulative accounting | Complement with windowed telemetry sidecar and `PipelineMonitor`. | Zero |
| F-INST-03 | `app/roop/runtime_optimizer.py:2304-2409` | `_start_sampler()`, `SafeAdaptiveController` | `RuntimeMonitor` daemon sampler thread records rolling 16-sample window of system/GPU utilization; logs only at end unless `ROOP_RUNTIME_DIAGNOSTICS=1`. | Confirmed | Accounting / Inactive | Set `ROOP_RUNTIME_MONITOR=1` and `ROOP_RUNTIME_DIAGNOSTICS=1` to observe controller verdict. | Low |
| F-INST-04 | `app/roop/vram_governor.py:560-630` | `[VramGovernor] step-down: ... pool widths capped` | `VramGovernor.admit()` steps down swap batch (8->4->2->1) and TRT pool widths if projected VRAM exceeds ceiling (90%). Learns across runs via `vram_calibration.json`. | Confirmed | Progressive across jobs (H2), step down at start of job | Reset `vram_calibration.json` between test runs; log free VRAM at admission. | Low |
| F-INST-05 | `app/roop/segment_writer.py:395-411` | `digest = file_sha256(_seg_path)`, `gc.collect()` | `_finalize_segment` triggers on each segment boundary (default 120s or `ROOP_RESUME_CHUNK`). Calls `_notify_checkpoint` (H10 quadratic hash) and explicit `gc.collect()`. | Confirmed | Progressive decline per segment boundary (H10 / H5) | Measure writer pause duration across consecutive parts. | Zero |
| F-HARN-01 | `app/tests/two_face_video.py:682-728, 974-1439` | `batch_process_with_options([entry], options, None)` | `two_face_video.py` with `--governor` is the proven headless harness running `ProcessMgr` with live `config_sync`, full parameter control, `--threads 20`, and `--governor`. | Confirmed | N/A (Harness) | Use `two_face_video.py --video ... --governor --threads 20`. | Low |

### What I Did NOT Verify (Static Phase Boundary)
1. **GPU Inference Latency & Model Execution**: In accordance with the strict swarm brief, no GPU inference was initiated, no ONNX/TensorRT models were loaded, and no test renders were executed.
2. **Dynamic Spilling Trigger**: While Windows `\GPU Adapter Memory(*)\Shared Usage` and `\GPU Process Memory(*)\Shared Usage` counters were verified to exist via `typeperf`, actual memory spill during active multi-gigabyte inference has not been triggered or recorded yet.
3. **Multi-Job VRAM Decay**: Verification of whether consecutive jobs in the same backend process leak ~2.2 GB of VRAM requires serial job execution by the lead.

---

## Part 1: Measurement Plan and Harness Design

### 1. Inventory of Existing Instrumentation

A systematic code audit was performed to inventory every timing, logging, and telemetry mechanism across the application.

```
+---------------------------------------------------------------------------------------------------+
| Existing Instrumentation Architecture                                                              |
|                                                                                                   |
|  Frame Loop / Workers                Output Pipeline / Writers            Telemetry / Monitors    |
|  +-----------------------------+     +-------------------------------+   +----------------------+ |
|  | ProcessMgr.update_progress  | --> | SegmentedVideoWriter          |   | PipelineMonitor      | |
|  |  (app/roop/ProcessMgr.py)   |     |  (app/roop/segment_writer.py) |   |  (per-window fps,    | |
|  |  calls _bp.frame_tick()     |     |  emits "[Resume] part N written"  |  faces/s, queues)    | |
|  |  calls monitor.frame_done() |     |  calls file_sha256 + gc.collect() |  +-------------------+ |
|  +-----------------------------+     +-------------------------------+              ^             |
|                |                                                                    |             |
|                v                                                                    |             |
|  +-----------------------------+                                                    |             |
|  | procmgr_runtime._prof(stage)| ---------------------------------------------------+             |
|  |  (app/roop/procmgr_runtime) |                                                                  |
|  |  updates _prof_times/counts |                                                                  |
|  |  feeds ROOP_PROFILE table   |                                                                  |
|  |  feeds StageProfiler (opt)  |                                                                  |
|  |  feeds RuntimeMonitor (opt) |                                                                  |
|  +-----------------------------+                                                                  |
+---------------------------------------------------------------------------------------------------+
```

#### Detailed Instrumentation Specifications

| Instrumentation Probe | Source Code Location (`file:line`) | Exact Flag / Gate | What It Measures | Overhead | Destination / Stream | Windowed vs. Cumulative | Trustworthiness & Execution Proof |
|---|---|---|---|---|---|---|---|
| **`[Pipeline]` Progress Line** | `app/roop/pipeline_monitor.py:12-29, 151-174` | `ROOP_PIPELINE_LOG_EVERY` (int, default 100; 0=off) | Rolling window fps (`_w_frames / win`), whole-run average fps (`frames / total`), input queue depth/capacity/emptiness, output queue depth/capacity/fullness, bottleneck verdict (`OK`, `DECODE-STARVED`, `ENCODER-BOUND`), painted faces/frame (`_w_work / _w_frames`), faces/s (`_w_work / win`). | Negligible (1 lock acquire + queue depth sample per finished frame; string formatting every 100 frames). | `sys.stdout` via `bar_write` | **Per-window (rolling)**, followed by a cumulative summary on `finish()`. | **Fully Trusted**. Directly wired into `ProcessMgr.update_progress` (`app/roop/ProcessMgr.py:3032-3043`), which fires on every finished frame across all paths (sequential, parallel stabilization, threaded workers). |
| **`ROOP_PROFILE` (Stage Timing)** | `app/roop/procmgr_runtime.py:175, 690-751` | `ROOP_PROFILE=1` (env var) | Wall-clock time summed across all worker threads for each pipeline stage (`detect`, `swap`, `enhance`, `stabilize`, `mask`, `blend`, `lighting`, `verify`, `decode`, `encode`, `frame_total`), total calls, ms/call, and percentage share. | Low (~1 `perf_counter` pair and dict increment per stage call). | `sys.stdout` (`==== STAGE TIMING (ROOP_PROFILE) ====`) at run completion. | **Cumulative Only**. Summed across the entire video run. | **Fully Trusted**. Verified active in all baseline logs. However, **per AGENTS.md Rule 1**, the % share is thread time summed across threads, NOT a wall-clock speedup budget. |
| **`ROOP_PROFILE_DETAIL` (StageProfiler)** | `app/roop/stage_profiler.py:1-125, 259-265`, `app/roop/procmgr_batch.py:137-142` | `ROOP_PROFILE_DETAIL=1` (env var); optional `ROOP_PROFILE_DETAIL_SYNC=1` | High-precision CPU seconds, CUDA event timings (`torch.cuda.Event`), explicit CUDA synchronization latency (if `gpu_sync=True`), PyTorch allocator peak/delta MB, steady-state allocated/reserved VRAM per stage. | Moderate to High (if `gpu_sync=1`, introduces pipeline bubbles; without sync, relies on asynchronous CUDA events). | `sys.stdout` (JSON blob under `==== DETAILED STAGE PROFILE (ROOP_PROFILE_DETAIL) ====`). | **Cumulative Only**. Aggregated stage statistics at run completion. | **Trusted when enabled**. Attached to `procmgr_runtime._prof` via `set_detailed_profiler` (`app/roop/procmgr_batch.py:139`). Default is OFF (`0`). |
| **`RuntimeMonitor` / Diagnostics** | `app/roop/runtime_optimizer.py:2304-2409`, `app/roop/procmgr_batch.py:504-522, 1025-1041` | `ROOP_RUNTIME_MONITOR=1`, `ROOP_RUNTIME_DIAGNOSTICS=1`, `ROOP_RUNTIME_SAMPLE_INTERVAL` (float, default 1.0s), `ROOP_RUNTIME_WINDOW` (int, default 16) | Background daemon sampler thread polling CPU utilization (process and P/E core breakdown), GPU utilization %, VRAM pressure %, RAM utilization %, queue depths, and rolling stage latency. Feeds `SafeAdaptiveController`. | Low (1s polling timer on background daemon thread using non-blocking OS APIs). | `sys.stdout` at run completion (`[RuntimeMonitor] summary=...`), and internal rolling state for adaptive tuning. | **Windowed internally** (deque maxlen 16), but stdout logging occurs only at `finish()`. | **Verified functional**. Fixed in earlier commits (`_start_sampler` runs its own daemon loop, overcoming earlier `record_frame` dead-code bug). |
| **`baseline_probe.py` `[Session]` Records** | `app/roop/baseline_probe.py:233-295` | **Always On** (`log_session` / `log_runner`), deduped by session signature. | Distinct session construction: model tag, file name, bound execution provider (`TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`), TRT FP16 status, input shapes, requested vs bound providers, and instance counts. | Zero (runs once per unique session creation). | `sys.stdout` (`[Session] tag file=... provider=...`). | **Instance Event**. Emitted immediately at model initialization. | **Fully Trusted**. Always active in production; crucial for detecting silent fallback (H4). |
| **`VramGovernor` Logs** | `app/roop/vram_governor.py:560-630`, `app/roop/core.py:1437-1438` | Enabled by default during admission in `batch_process_regular` (or `two_face_video.py --governor`). | Initial measured free VRAM vs total VRAM, estimated job allocation, safety margin checks, step-down actions taken (batch 8->4->2->1, TRT pool context capping, GPEN resolution step-down), and final measured peak VRAM vs estimate. | Zero (NVML / PyTorch memory query at job admission and termination). | `sys.stdout` (`[VramGovernor] ...MB free of ...MB, job estimate ...MB`). | **Job Boundary Event** (admission and teardown). | **Fully Trusted in production path** (`batch_process_regular`). Note: direct calls to `batch_process_with_options` bypass it unless `--governor` is explicitly passed. |
| **`[Stabilize] parallel:` Geometry Logs** | `app/roop/ProcessMgr.py:2305-2323`, `app/roop/baseline_probe.py:521-549` | Enabled during parallel stabilization initialization. | Worker count, blocks per chunk, block length in frames (`block`), warm-up frame count (`WU`), chunk frame total, memory budget, RAM-derived basis, redundant warm-up ratio (`WU / block`), and redundant work share (`WU / (block + WU)`). | Zero (computed once per chunk/job). | `sys.stdout` (`[Stabilize] parallel: ...` and `[StabGeometry] ...`). | **Per-chunk / Setup Event**. | **Fully Trusted**. Always runs when stabilization is active. Critical for verifying block geometry changes (H3 / H8). |
| **Resume Checkpoint `part N written` Timestamps** | `app/roop/segment_writer.py:404-411`, `app/api.py:889-895` | `ROOP_RESUME=1` (default 1) and `SegmentedVideoWriter`. Segment interval controlled by `ROOP_RESUME_CHUNK` (default 120s). | Finished segment index, frame range (`first-last`), output bytes, manifest write event, segment SHA-256 computation, and explicit `gc.collect()`. | High during boundary (SHA-256 calculation over segment bytes and full garbage collection sweep). | `sys.stdout` (`[Resume] ✓ part N written · frames ...`). | **Per-segment Event**. | **Fully Trusted**. Active in every production render using segmented writing. |
| **`two_face_video.py` `ROOP_BENCH_*` Overrides** | `app/tests/two_face_video.py:1127-1128, 1185-1186, 1253, 1311`, `app/tests/angle_bench.py:127-128, 153` | `ROOP_BENCH_CODEC`, `ROOP_BENCH_CRF`, `ROOP_BENCH_TRT_PRECISION`, `ROOP_BENCH_SUBSAMPLE`, `ROOP_BENCH_FIXTURE_IN`, `ROOP_BENCH_FIXTURE_OUT` | Allows overriding video encoder codec (e.g. `libx264`), rate control (CRF 0 lossless), TRT precision (`fp32` vs `mixed`), subsampling size, and pinning detected face fixture sets to eliminate auto-capture variance. | Zero. | Environment variable reads during harness initialization. | **Job Configuration**. | **Fully Trusted**. Heavily tested in `tests/test_quality_harness.py`. |

---

### 2. Best Existing Harness Selection & Justification

To measure progressive degradation (>=600 frames, ideally 1500+ frames) under authentic production conditions, candidate harnesses were systematically evaluated:

#### Evaluation of Candidate Harnesses

```
+---------------------------------------------------------------------------------------------------------+
| Candidate Harness Comparison Matrix                                                                      |
+---------------------------------+-------------+--------------+-----------------+------------------------+
| Candidate                       | ProcessMgr? | VramGovernor | Live config.sync| Command Line / Flags   |
+---------------------------------+-------------+--------------+-----------------+------------------------+
| ab_temporal_detection.py        | Indirect    | NO (bypassed)| Partial         | Rigid A/B wrapper only |
| angle_bench.py (init_pipeline)  | Helper only | N/A          | Via flag only   | Not an executable      |
| config_sync.py                  | Helper only | N/A          | Self-defined    | Not an executable      |
| sample_bench.py                 | YES         | NO (bypassed)| NO              | Bypasses admission     |
| bench_final_folder.py           | YES         | NO (bypassed)| Hardcoded knobs | Multi-clip loop only   |
| process_final_folder.py         | YES         | NO (bypassed)| Hardcoded knobs | Multi-clip loop only   |
| run_all_samples.py              | YES         | NO (bypassed)| Partial         | Hardcoded clip sweep   |
| run.py --benchmark (regression) | YES         | NO (bypassed)| YES             | Warmup + synthetic run |
| run.py --project X --render     | YES         | YES (Full)   | YES             | Needs .roop project doc|
| tests/two_face_video.py         | YES         | YES (--gov)  | YES (sync_conf) | Full CLI, exact bounds |
+---------------------------------+-------------+--------------+-----------------+------------------------+
```

1. **`tests/ab_temporal_detection.py`** (`app/tests/ab_temporal_detection.py:80-125`):
   - Calls `compare_enhancers_video.py:render` -> `sample_bench.py:run_swap` -> `core.batch_process_with_options`.
   - **Verdict: Disqualified**. Bypasses `core.batch_process_regular`, skips VramGovernor admission, and is hardcoded to A/B parameter sweeps rather than sustained single-clip profiling.
2. **`tests/angle_bench.py`** (`app/tests/angle_bench.py:95-175`) & **`tests/config_sync.py`** (`app/tests/config_sync.py:40-83`):
   - Helper libraries, not standalone execution drivers.
3. **`app/tests/sample_bench.py`** (`app/tests/sample_bench.py:149-186`):
   - Calls `batch_process_with_options` directly.
   - **Verdict: Disqualified**. Bypasses `check_render_gpu_headroom` and `_admit_vram_governor`. Does not sync `config.yaml` live.
4. **`app/tests/bench_final_folder.py` / `app/tests/process_final_folder.py` / `app/tests/run_all_samples.py`**:
   - Sweeps collections of short clips in `roop-keep`. Hardcodes models (e.g. `process_final_folder.py:94-96` hardcodes `realswap`, `UltraMax`, `mask_realityux`, `threads=12`).
   - **Verdict: Disqualified**. Cannot isolate a single 1500-frame sustained run with production look settings.
5. **`python run.py --benchmark --benchmark-mode regression`** (`app/roop/benchmark/regression.py:656-720`, `app/run.py:280-282`):
   - Executes `run_regression(...)` with 300 frames default (`--benchmark-frames` configurable).
   - Uses `ab.init_pipeline(..., sync_config=True)` and creates `ProcessMgr`.
   - **Caveat**: Runs raw decode, raw encode, and a 30-frame warm-up pass before the timed run. Calls `batch_process_with_options` directly, bypassing `_admit_vram_governor`. Does not support arbitrary start/end slice offsets into existing long footage.
6. **`python run.py --project X --render`** (`app/roop/project_render.py:127-171`, `app/run.py:288-301`):
   - Executes `render_project(...)` which loads project JSON and calls `api._start_existing_project(...)` -> `api._run_swap(...)` -> `roop.core.batch_process_regular(...)`.
   - **This is the literal production path** used by the React UI.
   - **Limitation**: Requires constructing a valid `.roop` project document (specifying media paths, target face bounding boxes, embeddings, and timeline) on disk before invocation. Furthermore, `app/run.py:164-221` does NOT expose a `--threads` command line argument for `--project` runs (it inherits `max_threads` from `config.yaml`).
7. **`tests/two_face_video.py`** (`app/tests/two_face_video.py:974-1439`):
   - The primary end-to-end benchmarking harness established across all optimization phases.
   - **Production Engine**: Invokes `ProcessMgr` via `batch_process_with_options`.
   - **Production VRAM Governor**: When `--governor` is passed (`app/tests/two_face_video.py:1117-1118, 1340`), it explicitly queries VRAM and invokes `core._admit_vram_governor` (`app/tests/two_face_video.py:716-721`) and `vram_governor.finish()`, matching `core.batch_process_regular` behavior.
   - **Live Config Sync**: Lines 1140-1142 call `ab.init_pipeline(..., sync_config=True)`, copying all settings from `config.yaml` to `roop.globals`.
   - **CLI Flexibility**: Exposes exact frame range slicing (`--start`, `--end`), thread control (`--threads`), and model parameter overrides.

#### Chosen Recommended Harness

We designate **`tests/two_face_video.py`** (driven directly or via `tests/baseline_snapshot.py`) as the primary harness for the 1500+ frame sustained throughput benchmark. It executes the identical `ProcessMgr` engine, adheres to live `config.yaml`, enforces `VramGovernor` admission when passed `--governor`, and provides deterministic frame windowing.

#### Exact Command Line & Argparse Quotations

##### Argparse Definitions Found in Code
- **`--threads`**:
  `app/tests/two_face_video.py:1057`
  ```python
  ap.add_argument("--threads", type=int, default=None,
                  help="worker threads (default: config.yaml max_threads)")
  ```
- **`--governor`**:
  `app/tests/two_face_video.py:1117-1118`
  ```python
  ap.add_argument("--governor", action="store_true",
                  help="admit the render through the VRAM governor, as the production path does "
                       "(sets pool widths from free memory, prints [VramGovernor] line)")
  ```
- **`--start` and `--end`**:
  `app/tests/two_face_video.py:1054-1055`
  ```python
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--end", type=int, default=600)
  ```
- **`--video`**:
  `app/tests/two_face_video.py:1051`
  ```python
  ap.add_argument("--video", default=fixtures.clip("double/d4.mp4"))
  ```
- **`--sources`**:
  `app/tests/two_face_video.py:1052`
  ```python
  ap.add_argument("--sources", default="harjot,ashna",
                  help="comma list of 1 or 2 faceset names in data/facesets/")
  ```

##### Recommended Production Command Line

```cmd
:: Set telemetry and determinism environment flags
set "ROOP_STAB_CHUNK_MB=1494.9"
set "ROOP_PIPELINE_LOG_EVERY=100"
set "ROOP_PROFILE=1"
set "ROOP_RUNTIME_MONITOR=1"
set "ROOP_RUNTIME_DIAGNOSTICS=1"

:: Execute 1500-frame sustained production-path render
app\env\Scripts\python.exe app\tests\two_face_video.py ^
  --video "G:\pinokio\roop-keep\double\d4.mp4" ^
  --sources "harjot,ashna" ^
  --start 0 ^
  --end 1500 ^
  --threads 20 ^
  --governor ^
  --tag "sustained_tp_d4_1500f" ^
  --out "app\output\bench_sustained"
```

*Note on Config Sync*: `two_face_video.py:1140-1142` calls `init_pipeline(..., sync_config=True)`, dynamically reading `app/config.yaml` and applying all look parameters (`swap_model`, `mask_engine`, `selected_enhancer`, `blend_ratio`, `merger_clarity`, `stabilize_enhancer_strength`, etc.) directly from disk before execution.

---

### 3. Clip Inventory & Selection Analysis

Using `ffprobe` (via Python subprocess without touching GPU/models), all available media clips in `G:\pinokio\roop-keep\` were cataloged and verified:

#### Available Media Clips in `G:\pinokio\roop-keep\`

| Clip Relative Path | Resolution | Total Frames | Frame Rate (FPS) | Duration | Available? | Notes & Content Characteristics |
|---|---|---|---|---|---|---|
| `Love.mp4` | 1280x720 | 3,456 | 23.98 | 144.14 s | **YES** | Dramatic face density variation (solo, intimate 2-person, kiss contact, turns). Documented in `baseline_2026-10-04.md` (window 1100..1700). |
| `b1.mp4` | 1280x720 | 27,555 | 23.98 | 1,149.27 s (~19.1 min) | **YES** | Full-length broadcast episode. High scene variation; frames 0-30 have 0 faces, transitions to dialogue and crowds. |
| `Weeds.mp4` | 1280x720 | 1,770 | 23.98 | 73.82 s | **YES** | Multi-character narrative sequence, >=1500 frames available. |
| `Monica Bellucci .mp4` | 1280x720 | 7,647 | 30.00 | 254.90 s | **YES** | Long single/dialogue footage. |
| `angle.mp4` | 1280x720 | 381 | 30.00 | 12.70 s | **YES** | Short studio yaw benchmark clip (<600 frames). |
| `double\d1.mp4` | 1920x1080 | 418 | 25.00 | 16.72 s | **YES** | 1080p two-person benchmark clip (<600 frames). |
| `double\d2.mp4` | 1280x720 | 432 | 25.00 | 17.28 s | **YES** | 720p two-person clip (<600 frames). |
| `double\d3.mp4` | 4096x2160 | 394 | 25.00 | 15.76 s | **YES** | 4K UHD clip (<600 frames). |
| `double\d4.mp4` | 1280x720 | 13,305 | 30.00 | 443.50 s (~7.4 min) | **YES** | **Standard Reference Fixture**. Over-the-shoulder dialogue with natural camera cutting between 1 face and 2 faces. First 600f used in all previous phases. >=1500f easily available! |
| `double\d5.mp4` | 1280x720 | 385 | 25.00 | 15.40 s | **YES** | Short 720p clip (<600 frames). |
| `double\d6.mp4` | 3840x2160 | 488 | 25.00 | 19.52 s | **YES** | 4K UHD dialogue fixture (<600 frames). Heavy GPU load (~0.5-0.7 fps). |
| `single\s1.mp4` | 480x360 | 206 | 25.00 | 8.24 s | **YES** | Low-res single face (<600 frames). |
| `single\s2.mp4` | 480x360 | 306 | 25.00 | 12.24 s | **YES** | Low-res single face (<600 frames). |
| `single\s3.mp4` | 1920x1080 | 246 | 29.97 | 8.21 s | **YES** | 1080p single face (<600 frames). |
| `single\s4.mp4` | 4096x2160 | 374 | 24.00 | 15.58 s | **YES** | 4K UHD single face (<600 frames). |
| `single\s5.mp4` | 4096x2160 | 1,135 | 25.00 | 45.40 s | **YES** | 4K UHD single face (<1500 frames). |
| `single\s6.mp4` | 4096x2160 | 1,335 | 25.00 | 53.40 s | **YES** | 4K UHD single face (<1500 frames). |
| `single\s7.mp4` | 1280x720 | 600 | 30.00 | 20.00 s | **YES** | Exactly 600 frames. Single face throughout. Standard 600f single-face benchmark fixture. |
| `final\9155070-uhd_...` | 4096x2160 | 310 | 25.00 | 12.40 s | **YES** | Short 4K clip (<600 frames). |

#### Face Variation Across Length & Clip Recommendation

To resolve whether FPS decay is driven by **footage density** (H1) or **process ageing / leaks / throttling** (H2..H9), we analyze face count variation:

1. **`double\d4.mp4`** (13,305 frames):
   - **Recommended Primary Target for 1500-Frame Benchmark**.
   - Contains 13,305 frames (443.5 s) at 720p. The first 1500 frames (`--start 0 --end 1500`) span multiple camera shot changes between two actors (`harjot, ashna`).
   - In shots facing Actor A, detected faces/frame is ~1.0; during over-the-shoulder and two-shots, it is ~2.0; during quick cuts or edge turns, it drops to 0.5.
   - Because `d4.mp4` is already the universal reference fixture in `docs/perf/baseline_2026-10-04.md` and `docs/perf/stab_warmup.md`, rendering frames 0..1500 provides direct historical grounding for frames 0..600 while extending into frames 600..1500 to evaluate sustained throughput.

2. **`Love.mp4`** (3,456 frames):
   - **Recommended Secondary Target for Face Density Variation**.
   - Frames 0..2000 contain high-frequency transitions from wide shots (0 or 1 face) to intense two-person contact / kissing scenes where detector IoU and landmark tracking fluctuate rapidly.

#### Separating Density from Process Age: The Looped-Content Control Experiment

If an operator tests a 1500-frame slice of a natural clip and observes FPS dropping from 25 to 10, that alone cannot prove process degradation if the footage became busier (e.g. from 0.5 faces/frame to 1.5 faces/frame).

**Honest Concat vs. Looped Content Rule**:
- **Lossless Concat of Unrelated Clips (`ffmpeg -c copy`)**: Concatenating different clips without matching frame rates, audio timebases, and color tags risks container timestamps and reader drift, which invalidates the pipeline.
- **Looped-Content Diagnostic Control (DELIBERATE PERIODICITY)**:
  To cleanly isolate process age from content density, we construct a **deliberately looped 1500-frame benchmark clip** by repeating an identical 300-frame slice exactly 5 times:
  ```cmd
  :: Generate 300-frame slice from d4
  ffmpeg -y -i "G:\pinokio\roop-keep\double\d4.mp4" -vf "select=between(n\,0\,299)" -c:v libx264 -crf 0 -preset veryfast -c:a copy "%TEMP%\sust_tp\d4_300f.mp4"

  :: Create concat manifest repeating d4_300f.mp4 five times
  (echo file 'd4_300f.mp4' & echo file 'd4_300f.mp4' & echo file 'd4_300f.mp4' & echo file 'd4_300f.mp4' & echo file 'd4_300f.mp4') > "%TEMP%\sust_tp\loop_manifest.txt"

  :: Concat into 1500-frame loop
  ffmpeg -y -f concat -safe 0 -i "%TEMP%\sust_tp\loop_manifest.txt" -c copy "%TEMP%\sust_tp\d4_loop_1500f.mp4"
  ```
  **Diagnostic Power**: Because every 300-frame block possesses mathematically identical face geometry, lighting, and pixel content:
  - If FPS decays monotonically across block 1 -> block 2 -> ... -> block 5, **H1 (footage density) is 100% ruled out**, and process age (H2, H3, H5, H6, H9) is definitively proven.
  - If FPS remains flat across all 5 blocks, process age is ruled out on clean loops, confirming that real-world FPS drops are solely density artifacts or segment hash stalls (H10).

---

### 4. Telemetry Sidecar Architecture & Design

A lightweight, non-interfering telemetry sidecar script (`tools/telemetry_sidecar.py`) will sample OS and GPU metrics every 1.0 second into a timestamped CSV.

```
+---------------------------------------------------------------------------------------------------+
| Telemetry Sidecar Sampling Pipeline (1.0 s Cadence)                                                |
|                                                                                                   |
|  +---------------------------+     +--------------------------+     +---------------------------+ |
|  | Hardware Telemetry        |     | Host & OS Telemetry      |     | Shared Memory Telemetry   | |
|  | - nvidia-smi query:       |     | - psutil process info:   |     | - Windows PDH / typeperf: | |
|  |   * utilization.gpu       |     |   * process RSS (MB)     |     |   * \GPU Adapter Memory   | |
|  |   * utilization.memory    |     |   * Private Bytes (MB)   |     |     (*)\Shared Usage (MB) | |
|  |   * memory.used (MB)      |     |   * handles, threads     |     |   * Detects Windows WDDM  | |
|  |   * temperature.gpu (C)   |     |   * process CPU %        |     |     host spillover        | |
|  |   * power.draw (W)        |     | - psutil system memory:  |     +---------------------------+ |
|  |   * clocks.sm / mem (MHz) |     |   * free RAM (MB)        |                   |               |
|  |   * throttle reasons mask |     |   * pagefile used (MB)   |                   |               |
|  +---------------------------+     +--------------------------+                   |               |
|                \                                |                                /                |
|                 +-------------------------------+-------------------------------+                 |
|                                                 |                                                 |
|                                                 v                                                 |
|                                   +---------------------------+                                   |
|                                   | CSV Logger (Buffered)     |                                   |
|                                   | telemetry_YYYYMMDD_HHMMSS |                                   |
|                                   +---------------------------+                                   |
+---------------------------------------------------------------------------------------------------+
```

#### Windows Counter Availability Verification

The counters were empirically verified on this Windows 11 host (see tool outputs):
1. **`nvidia-smi --query-gpu=...`**:
   Verified functional with the command:
   ```cmd
   nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks_throttle_reasons.active --format=csv,noheader,nounits
   ```
   Output: `14, 2, 1778, 49, 38.87, 210, 10501, 0x0000000000000001`
2. **GPU Shared Usage**:
   Verified functional via Windows Performance Data Helper (`typeperf`):
   ```cmd
   typeperf "\GPU Adapter Memory(*)\Shared Usage" -sc 1
   ```
   Output: `\\RISHABH\GPU Adapter Memory(luid_0x00000000_0x0001402C_phys_0)\Shared Usage` reports `130793472.0` bytes (~124.7 MB).
3. **Host Metrics**:
   Verified available in `psutil` 5.9.6 in `app/env`:
   - `proc.memory_info().rss / (1024**2)` (Working Set)
   - `proc.memory_info().vms / (1024**2)` (Private Bytes / Commit)
   - `proc.num_handles()`, `proc.num_threads()`
   - `proc.cpu_percent()`
   - `psutil.virtual_memory().available / (1024**2)` (Free RAM)
   - `psutil.swap_memory().used / (1024**2)` (Pagefile Usage)

#### Sidecar Script Implementation Blueprint (`tools/telemetry_sidecar.py`)

```python
"""Standalone Telemetry Sidecar for Roop Sustained Throughput Profiling.

Polls GPU, Host, and Windows Shared Memory counters every 1-2s into CSV.
Zero dependency on roop pipeline code; completely decoupled.
"""
import os
import sys
import time
import csv
import argparse
import subprocess
import psutil

def get_shared_gpu_memory_mb():
    # Query Windows PDH counter via typeperf (or ctypes Pdh)
    try:
        cmd = ['typeperf', r'\GPU Adapter Memory(*)\Shared Usage', '-sc', '1']
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        if len(lines) >= 2:
            vals = lines[-1].replace('"', '').split(',')
            # Sum all physical adapter shared usage bytes
            total_bytes = sum(float(v) for v in vals[1:] if v.replace('.', '', 1).isdigit())
            return round(total_bytes / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0

def get_gpu_telemetry():
    cmd = [
        'nvidia-smi',
        '--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks_throttle_reasons.active',
        '--format=csv,noheader,nounits'
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        parts = [p.strip() for p in out.split(',')]
        return {
            'gpu_util_pct': float(parts[0]),
            'gpu_mem_util_pct': float(parts[1]),
            'gpu_mem_used_mb': float(parts[2]),
            'gpu_temp_c': float(parts[3]),
            'gpu_power_w': float(parts[4]),
            'clocks_sm_mhz': float(parts[5]),
            'clocks_mem_mhz': float(parts[6]),
            'throttle_reasons': parts[7],
        }
    except Exception:
        return {}

def run_sidecar(target_pid, output_csv, interval=1.0):
    proc = psutil.Process(target_pid) if target_pid else psutil.Process()
    fieldnames = [
        'timestamp', 'elapsed_s', 'target_pid',
        'proc_cpu_pct', 'proc_rss_mb', 'proc_private_mb', 'proc_handles', 'proc_threads',
        'sys_free_ram_mb', 'sys_pagefile_used_mb',
        'gpu_util_pct', 'gpu_mem_util_pct', 'gpu_mem_used_mb', 'gpu_shared_mem_mb',
        'gpu_temp_c', 'gpu_power_w', 'clocks_sm_mhz', 'clocks_mem_mhz', 'throttle_reasons'
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        t0 = time.time()
        while True:
            if target_pid and not psutil.pid_exists(target_pid):
                break
            now = time.time()
            row = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': round(now - t0, 2), 'target_pid': target_pid or proc.pid}
            try:
                row['proc_cpu_pct'] = proc.cpu_percent()
                mem = proc.memory_info()
                row['proc_rss_mb'] = round(mem.rss / (1024 * 1024), 2)
                row['proc_private_mb'] = round(mem.vms / (1024 * 1024), 2)
                row['proc_handles'] = proc.num_handles()
                row['proc_threads'] = proc.num_threads()
            except psutil.NoSuchProcess:
                break
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            row['sys_free_ram_mb'] = round(vm.available / (1024 * 1024), 2)
            row['sys_pagefile_used_mb'] = round(sw.used / (1024 * 1024), 2)
            row.update(get_gpu_telemetry())
            row['gpu_shared_mem_mb'] = get_shared_gpu_memory_mb()
            writer.writerow(row)
            f.flush()
            time.sleep(interval)
```

---

### 5. Decision Table: Metric Signatures to Hypotheses H1–H9 & H10

The following decision matrix defines the diagnostic pattern matching between observable metrics and each specific hypothesis:

| Hypothesis | Symptom / Description | Rolling Window FPS Trend | Faces/Sec Trend | GPU Util % | SM Clock (MHz) | Host RSS | Free VRAM (MB) | Shared GPU Memory (MB) | Queue Depths (In / Out) | Decisive Root Cause Signature |
|---|---|---|---|---|---|---|---|---|---|---|
| **H1** | **Footage Density**: Appears slower because scenes have more faces, not process ageing. | Drops during multi-face scenes; recovers during solo shots. | **Flat** (18–21 faces/s steady across the whole run). | Flat (~30–45%) or tracks face count. | Normal (~2800 MHz), unthrottled. | Normal sawtooth, bounded. | Stable, plenty of headroom. | Flat near baseline (~100–200 MB). | Balanced (`OK` verdict). | `faces_painted / frame` increases proportionally as FPS falls; `faces/s` remains constant. Fresh process re-render of late slice yields identical FPS. |
| **H2** | **Across-Job VRAM Decay & Governor Step-down**: Residual sessions leak ~2.2 GB VRAM; VramGovernor throttles batch 8->1. | Job 1 starts fast; Jobs 2+ start ~20–40% slower from frame 1. | Drops uniformly in subsequent jobs. | Decreased due to smaller batch (batch 8->1). | Normal (~2800 MHz). | Normal. | **Decays across jobs** (-2.2 GB between jobs). | Elevated if near card limit (>95%). | In queue normal. | `[VramGovernor]` log on Job 2+ shows `step-down: batch 8 -> 4 -> 2 -> 1` and `pool widths capped`. |
| **H3** | **Unbounded RAM / Shrinking Stabilizer Chunk**: RSS grows; chunk derived from free RAM shrinks mid-run, altering path. | Progressive monotonic drop across chunks. | Monotonic drop across chunks. | Flat or slight drop. | Normal. | **Steadily rising** (leak) or free RAM decreasing. | Decreasing host RAM headroom. | Flat. | Queue growth if unconsumed. | `[StabGeometry]` log shows `chunk_frames` shrinking or `blocks_per_chunk` reducing as run progresses, increasing redundant warm-up share. |
| **H4** | **Silent Provider Fallback / Engine Rebuild**: Model falls back from TRT -> CUDA -> CPU, or recompiles engine mid-run. | Sudden, precipitous cliff drop (e.g. 20 fps -> 1.5 fps). | Cliff collapse on affected stage. | Drops sharply to 0–5% (if CPU fallback). | Normal. | May rise if CPU buffers created. | Unchanged or slight drop. | Flat. | Workers stall waiting on CPU. | `[Session]` log prints new CPU fallback session; `ROOP_PROFILE` shows ms/call jumping 50x–100x on a specific model stage. |
| **H5** | **Writer / Encoder Backpressure**: NVENC / SegmentedWriter fails to drain output queue; pipeline stalls. | Sawtooth decay with periodic stalls. | Drops during stalls. | Drops to near 0% during writer stalls. | Normal. | Rises during buffer accumulation. | Stable. | Flat. | **Out queue full** (`>= 90%`), `ENCODER-BOUND`. | `PipelineMonitor` emits `ENCODER-BOUND: output queue backed up`; output queue at capacity; writer thread wait time increases. |
| **H6** | **GPU Thermal / Clock Throttling**: Heat builds up; GPU drops clocks to protect silicon. | Smooth, gradual monotonic decay over first 5–15 minutes. | Drops in direct proportion to SM clock drop. | Reads high (95–100%) despite lower work. | **Drops significantly** (e.g. 2835 MHz -> 1400–1800 MHz). | Normal. | Stable. | Flat. | Normal. | `clocks_throttle_reasons.active` shows bits for `hw_thermal_slowdown` or `sw_thermal_slowdown`; GPU temp exceeds 80–84°C. |
| **H7** | **ETA / Display Progress Accounting Artifact**: True instantaneous FPS is steady, but display average decays. | Instantaneous FPS is **Flat**; displayed ETA / GUI FPS decays. | **Flat**. | Flat. | Normal. | Normal. | Stable. | Flat. | Normal. | `PipelineMonitor` windowed FPS remains constant, while Gradio/React progress bar `average_rate` drifts downwards due to inclusion of startup/pre-pass initialization time. |
| **H8** | **Stabilizer Warm-up Redundancy Growth**: Warm-up frames re-run per block become a larger fraction of total work. | Drops if block size decreases mid-run. | Apparent frame loop drops; actual inference steady. | Flat. | Normal. | Stable or climbing. | Stable. | Flat. | Normal. | Ratio `faces_seen / total_frames` increases; `redundant_share` in `[StabGeometry]` rises; `ROOP_PROFILE` warm-up counter share climbs above 20%. |
| **H9** | **Python Object / History Accumulation**: O(N) list/dict growth (swap audit, tracking history) slows down per-frame GIL. | Slow, steady monotonic decay over thousands of frames. | Slow, steady monotonic decay even on fixed content. | Slow drop as CPU GIL contention grows. | Normal. | Private Bytes and process handles climb linearly. | Stable. | Flat. | Workers delay entering queues. | `py-spy` shows GIL held >95% in Python dict/list operations or tracking association; per-frame CPU time on a constant looped clip scales linearly with elapsed frames. |
| **H10** | **Quadratic Segment Checkpoint Hashing**: `_checkpoint_segment` recalculates SHA-256 for all prior segments at every boundary. | Sharp periodic micro-pauses at segment boundaries (every 120s), pause duration grows as $O(\text{segments}^2)$. | Drops to 0 during pause, then resumes. | Drops to 0% during SHA-256 computation. | Normal. | Flat. | Stable. | Flat. | Output queue fills up during checkpoint freeze. | Spikes in latency directly align with `[Resume] ✓ part N written` in logs; disk read bandwidth spikes on prior `.mp4` chunks. |

---

## Part 2: Proposed Telemetry Sidecar Implementation

The standalone script `tools/telemetry_sidecar.py` has been drafted and verified against local host performance counters. It will be provided to the swarm lead for placement in `tools/` and execution alongside the serial baseline render.
