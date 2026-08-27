# Optimization Progress

Updated: 2026-08-28

## Current phase

- Last completed phase: Phase 2, committed as `d298fbf` (`Add centralized runtime optimizer`).
- Current phase: Phase 3, implemented in the working tree and awaiting final review/commit.
- Next phase: Phase 4, not started.
- Exact next action: review and commit the Phase 3 changes after the final verification results below; then validate the same workload on the RTX 3060 tier before considering Phase 4.

The expected project documents were missing when this session began. This file, `OPTIMIZATION_PLAN.md`, `SESSION_HANDOFF.md`, and `PERFORMANCE_BASELINE.md` were created from repository evidence so future sessions have a durable source of truth.

## Phase 0–2 repository evidence

- `1cae1de`: benchmark and UltraMax checkpoint.
- `83c980a`: TensorRT tuning and validated settings.
- `c8a24bf`: CPU scheduling for GPU pipeline feeding.
- `d298fbf`: centralized runtime optimizer in `app/roop/runtime_optimizer.py`, integrated through `ProcessMgr.py` and `run.py`, with runtime optimizer tests.

Earlier work also includes quality regression guards, physical hardware validation, the six-GB tier validation, transfer-path audit, GPEN 256 Pro coverage, and mixed TensorRT telemetry.

## Phase 3 changes in the working tree

### Files modified

- `app/roop/session_pool.py`
- `app/roop/backend_manager.py`
- `app/roop/bench.py`
- `app/roop/face_util.py`
- `app/roop/processors/FaceSwapInsightFace.py`
- `app/roop/processors/Enhance_CodeFormer.py`
- `app/roop/processors/Enhance_GPEN256Pro.py`
- `app/roop/processors/Enhance_GPENRealistic.py`
- `app/roop/processors/Enhance_RestoreFormerPPlus.py`
- `app/roop/processors/Enhance_UltraMax.py`
- `app/roop/processors/Expression_LivePortrait.py`
- `app/roop/processors/Mask_FaceParser.py`
- `app/roop/processors/Mask_FastSAM.py`
- `app/roop/processors/Mask_MobileSAM.py`
- `app/roop/processors/Mask_Occluder.py`
- `app/roop/processors/Mask_XSeg.py`
- `app/roop/processors/Mask_XSeg3.py`
- `app/roop/retinaface.py`
- `app/roop/yoloface.py`
- `app/roop/yunet.py`
- `app/tests/test_trt_context_manager.py` (new)
- `app/tests/test_backend_manager.py`

The untracked `facesets/` directory is user data and was not touched.

### Functions and architecture changed

`session_pool.py` now contains `ModelResourceSpec`, `_resource_spec`, `_live_vram_mb`, `TensorRTResourceManager`, `resource_manager`, the model-aware pool-size helpers, and the expanded `SessionPool` lifecycle (`lease`, `warmup`, `resize`, `release`, pressure refresh, admission limits, active-lease accounting, and bounded waiting). Model call sites now pass model keys and input shapes so resource estimates are tied to actual workloads. Face-swap pool replacement releases the previous pool safely before fallback.

`bench.py` now measures all context levels 1/2/3/4, reports context knees, and validates the selected pool without early stopping after the first plateau. Existing provider, engine-cache, and precision paths were preserved.

Final Phase 3 hardening also bounded the `SessionPool` queue, serialized warmup/resize transitions against new leases, made teardown wait for those transitions, and extended the TensorRT cache namespace with CUDA, driver, and TensorRT runtime versions.

## Tests already completed

- Phase 3 targeted tests: all passed, including the eight TensorRT context-manager tests and backend cache-identity coverage.
- Combined targeted regression suite: `180` tests passed.
- Full suite: `1344` tests run, `1342` passed, `2` known pre-existing failures, `1` skipped.
- Python compilation of all changed Python modules passed.

Known full-suite failures are unrelated to this Phase 3 working-tree change:

1. `test_bench_perf_env.TestBenchPerfEnvMatchesApp.test_direct_set_flags_match`: benchmark does not mirror three existing direct settings (`ROOP_TRT_BUILDER_OPT_LEVEL`, `ROOP_TRT_AUX_STREAMS`, `ROOP_CV_THREADS`).
2. `test_ui_settings_catalog.SettingsCatalog.test_panel_settings_are_all_findable_from_the_palette`: four existing Phase 2 settings are absent from the palette catalog.

## Benchmark evidence

The post-change full benchmark completed on the configured RTX 4070 workload with TensorRT mixed precision and seven stages. It measured the following stage curves in calls/sec for contexts 1/2/3/4:

| Stage | 1 | 2 | 3 | 4 | Selected knee |
|---|---:|---:|---:|---:|---:|
| RetinaFace detector | 336.6 | 405.4 | 423.2 | 414.1 | 3 |
| Recognition | 807.9 | 942.5 | 836.4 | 720.6 | 2 |
| Landmarks | 1347.2 | 1560.6 | 1420.9 | 1250.9 | 2 |
| Realswap | 188.1 | 244.7 | 252.9 | 250.5 | 2 by knee rule; raw best 3 |
| UltraMax | 36.3 | 41.3 | 38.6 | 39.8 | 2 |
| XSeg | 162.4 | 422.1 | 432.3 | 402.0 | 2 |
| BiSeNet | 76.8 | 92.4 | 90.7 | 94.1 | 2 |

Recommended pools were `trt_pool=2`, `detmask_pool=2`, `detector_pool=3`, and `expr_pool=2`. The current config explicitly sets the first three pools to `2`, so `--no-apply` correctly left user settings unchanged.

The composite heavy workload measured `21.67 FPS`. Widening `trt_pool` from 2 to 4 measured `20.15 FPS` (about 7% slower), so the wider pool was rejected. Standard CPU-thread throughput peaked at 16 workers (`66.73` synthetic FPS), enhanced at 8 workers (`27.50`), and heavy at 6 workers (`21.67`); no automatic config change was applied.

The same run measured TensorRT versus CUDA for detector `3.93x`, recognition `2.84x`, landmarks `1.21x`, realswap `2.21x`, XSeg `4.54x`, and BiSeNet `1.12x`. UltraMax CUDA was excluded because the existing ORT/CUDA fallback failed during LayerNorm/cuDNN frontend setup.

Batch swap improved from `177` to `702` tiles/sec. I/O probes measured CV2 decode `263.4 FPS`, NVDEC `206.0 FPS`, and encode peaks of x265 medium `41.9`, x265 faster `91.6`, x264 medium `91.3`, x264 faster `140.7`, H.264 NVENC `122.5`, and HEVC NVENC `148.8` FPS on the synthetic probe.

The pre-change benchmark used the older 1/2/4/8 sweep and was stopped during a cold XSeg build after partial results; it has no complete comparable end-to-end FPS. Therefore no end-to-end before/after improvement claim is made.

## Remaining issues and regressions

- Phase 3 has not yet been committed.
- A real-video before/after baseline with decode-to-encode FPS, latency, CPU, VRAM, RAM, and queue depth is still required.
- RTX 3060 validation of the new model-aware admission logic is still required.
- The existing UltraMax CUDA fallback failure remains.
- The two full-suite failures listed above remain.
- Current explicit settings (`perf_trt_pool=2`, `perf_detmask_pool=2`, `perf_detector_pool=2`) are authoritative; auto model-specific sizing will not replace them.
- No claim has been made that all contexts are concurrently resident or that CUDA Graphs/NVDEC/NVENC integration is complete; those are later evidence-driven work.
