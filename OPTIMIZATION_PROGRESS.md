# Optimization Progress

Updated: 2026-08-28

## Current phase

- Last completed phase: Phase 3 implementation, committed as `c439e43` (`Complete Phase 3 TensorRT autotuning`); physical RTX 3060 validation remains outstanding.
- Current phase: Phase 4, TensorRT engine/context optimization implemented and validated on the physical RTX 4070.
- Next phase: complete the Phase 4 hardware gate on a physical RTX 3060 Laptop and repeat the model/workload matrix there.
- Exact next action: run the same real-video and context-count measurements on the physical RTX 3060 Laptop; do not begin Phase 5.
- Phase 5 request: deferred; no precision policy or precision benchmark changes were started because Phase 4 remains incomplete on the required second hardware profile.

The expected project documents were missing when this session began. This file, `OPTIMIZATION_PLAN.md`, `SESSION_HANDOFF.md`, and `PERFORMANCE_BASELINE.md` were created from repository evidence so future sessions have a durable source of truth.

## Phase 0–2 repository evidence

- `1cae1de`: benchmark and UltraMax checkpoint.
- `83c980a`: TensorRT tuning and validated settings.
- `c8a24bf`: CPU scheduling for GPU pipeline feeding.
- `d298fbf`: centralized runtime optimizer in `app/roop/runtime_optimizer.py`, integrated through `ProcessMgr.py` and `run.py`, with runtime optimizer tests.

Earlier work also includes quality regression guards, physical hardware validation, the six-GB tier validation, transfer-path audit, GPEN 256 Pro coverage, and mixed TensorRT telemetry.

## Phase 3 implementation committed in `c439e43`

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

`bench.py` now measures context levels 1/2/3/4/6, records repeated-run spread/stability, reports model-specific context knees, and validates the selected pool without early stopping after the first plateau. Existing provider, engine-cache, and precision paths were preserved.

Final Phase 3 hardening also bounded the `SessionPool` queue, serialized warmup/resize transitions against new leases, made teardown wait for those transitions, and extended the TensorRT cache namespace with CUDA, driver, and TensorRT runtime versions.

## Verification

- Phase 3 targeted tests: all passed, including the eight TensorRT context-manager tests and backend cache-identity coverage.
- Current focused regression suite: `152` passed, `12` subtests passed, `1` warning.
- Combined targeted regression suite: `180` tests passed.
- Full suite: `1343` tests run, `1341` passed, `2` known pre-existing failures, `1` skipped; `589` subtests passed.
- Python compilation of all changed Python modules passed.

Known full-suite failures are unrelated to the Phase 3 implementation:

1. `test_bench_perf_env.TestBenchPerfEnvMatchesApp.test_direct_set_flags_match`: benchmark does not mirror three existing direct settings (`ROOP_TRT_BUILDER_OPT_LEVEL`, `ROOP_TRT_AUX_STREAMS`, `ROOP_CV_THREADS`).
2. `test_ui_settings_catalog.SettingsCatalog.test_panel_settings_are_all_findable_from_the_palette`: four existing Phase 2 settings are absent from the palette catalog.

## Phase 4 TensorRT engine/context audit and validation

The audit covered `core.py`, `session_pool.py`, all processor call sites, `face_util.py`, and `ProcessMgr.py`. Existing TensorRT engine/timing caches, precision and hardware namespaces, cache validation, context safety, session leasing, VRAM warnings, IO-binding locks, and model-specific FP32 fallbacks were retained. The current `core.py` build settings remain evidence-backed: dynamic model shapes are not replaced with a generic profile, the existing workspace policy is retained, and builder/tactic knobs remain part of the tuning/cache identity. Engine loading and warmup continue through the existing ORT/session-pool paths, with each pooled slot owning an independent context and binding.

Phase 4 adds the evidence loop rather than guessing a global context count. `session_pool.py` permits benchmark candidates through six contexts, but automatic selection is capped by the matching model/shape knee, live free VRAM, resident pools, per-context memory, workspace, batch size, and safety margin. A persisted knee is accepted only when GPU identity, VRAM capacity, provider, TensorRT precision, selected model, input shape, and TensorRT tuning identity match. Explicit numeric environment/config pool values remain authoritative and are never silently replaced. `bench.py` measures 1/2/3/4/6 contexts with repeated-run stability and persists the full curve, latency, free-VRAM observations, and selected knee.

The full RTX 4070 TensorRT mixed-precision benchmark measured the following stable model curves in calls/sec:

| Stage | 1 | 2 | 3 | 4 | 6 | Selected knee |
|---|---:|---:|---:|---:|---:|---:|
| RetinaFace detector | 310.3 | 359.9 | 384.9 | 414.4 | 453.8 | 6 |
| Recognition | 747.6 | 888.5 | 794.9 | 703.4 | 598.0 | 2 |
| Landmarks | 1232.9 | 1610.6 | 1460.4 | 1306.0 | 1068.3 | 2 |
| Realswap | 176.7 | 230.4 | 243.7 | 245.0 | 241.1 | 3 |
| UltraMax | 34.6 | 37.7 | 36.2 | 36.0 | 35.8 | 2 |
| RealityUX XSeg | 337.8 | 465.7 | 438.5 | 400.4 | 318.4 | 2 |
| RealityUX BiSeNet | 73.2 | 92.7 | 92.0 | 94.5 | 91.6 | 2 |

Measured model knees were `detector=6`, `recognition/landmarks/masks/enhancer=2`, and `swapper=3`. The aggregate recommendation was `trt_pool=3`, `detmask_pool=2`, `detector_pool=6`, and `expr_pool=2`; the current explicit `2/2/2` pool settings were preserved.

The composite heavy workload measured `20.55 FPS` at the aggregate knee (`trt_pool=3`). Widening only that pool to `6` measured `17.96 FPS`, so the higher-utilization configuration was rejected. Standard CPU-thread throughput peaked at 16 workers (`63.85` synthetic FPS), enhanced at 6 (`26.48`), and heavy at 6 (`20.55`); no automatic config change was applied.

The same run measured TensorRT versus CUDA for detector `3.63x`, recognition `2.70x`, landmarks `1.17x`, realswap `2.07x`, XSeg `9.58x`, and BiSeNet `1.08x`. UltraMax CUDA was excluded because the existing ORT/CUDA fallback failed during LayerNorm/cuDNN frontend setup.

Batch swap improved from `171.4` to `683.5` tiles/sec. I/O probes measured CV2 decode `263.0 FPS`, NVDEC `199.9 FPS`, and HEVC NVENC p5 `143.9 FPS` on the synthetic probe.

The pre-change benchmark used the older 1/2/4/8 sweep and was stopped during a cold XSeg build after partial results; it has no complete comparable end-to-end FPS. Therefore no end-to-end before/after improvement claim is made.

## Real-video Phase 4 validation

The same available 8-second, 1280x720/30 fps fixture (`240` frames) was run with the explicit TensorRT mixed/Realswap/RealityUX/UltraMax configuration and `2/2/2` detector/detmask/TRT pools. The output completed successfully with `238/238` face-bearing frames composited and `238` enhanced. The harness measured `46.9 s`, or `5.12` end-to-end FPS and approximately `195.4 ms/frame` wall latency. External whole-card samples recorded `7,984 MB` peak VRAM, `4,764 MB` average sampled VRAM, and `28.3%` average / `55%` peak GPU utilization. CPU utilization was not captured reliably by this wrapper and is intentionally not estimated. The fixture did not match the selected `harjot` source, so this is throughput/resource evidence only, not a quality result.

The earlier Phase 3 run below remains historical evidence and is not used as a comparable before/after claim.

## Historical real-video Phase 3 validation

The existing `tests/sample_bench.py` harness was run against the available ignored 8-second, 1280x720/30 fps fixture (`240` frames), using the live config: TensorRT mixed precision, Realswap, RealityUX, UltraMax, NVDEC on, HEVC NVENC p5, batch swap on, `12` worker threads. The completed RTX 4070 run produced:

- `394.4 s` measured by the harness (`0.61` end-to-end fps); ffmpeg processing was `376.41 s` (`0.64` fps).
- Runtime profile: `workers=12`, `queue=3`; runtime process memory reported by the pipeline was about `9.52 GiB`.
- External samples: system RAM peak `28.26 GiB`, whole-card VRAM peak `8.03 GiB`, GPU utilization `34.0%` average / `89.0%` peak.
- The output completed successfully and logged `238/238` face-bearing frames. The fixture did not match the selected `harjot` source: `237/238` faces were refused by the identity gate, so this run is resource/throughput evidence only, not a quality result.

The existing `tests/ab_small_card_pools.py` harness completed a counterbalanced real-clip A/B run with `ROOP_VRAM_GB=6`, `8` threads, and UltraMax. The simulated 6 GB policy (TRT/detmask pools `0/0`) averaged `11.61` fps; forced pools `2/2` averaged `13.54` fps and used up to `4625 MB` observed process-card rise on the physical 4070. All `238` faces were enhanced in both arms. This validates policy selection and pressure behavior only; it is not a physical RTX 3060 measurement.

## Remaining issues and regressions

- A physical RTX 3060 Laptop validation with decode-to-encode FPS, latency, CPU, VRAM, RAM, queue depth, and worker counts is still required.
- CPU utilization for the current RTX 4070 real-video run was not captured reliably and is intentionally marked unavailable rather than inferred.
- Phase 4 must remain open until the physical RTX 3060 repeats the model-specific 1/2/3/4/6 context sweep and real-video workload.
- A comparable pre-Phase-3 real-video baseline is unavailable; no before/after improvement claim is made.
- The existing UltraMax CUDA fallback failure remains.
- The two full-suite failures listed above remain.
- Current explicit settings (`perf_trt_pool=2`, `perf_detmask_pool=2`, `perf_detector_pool=2`) are authoritative; auto model-specific sizing will not replace them.
- No claim has been made that all contexts are concurrently resident or that CUDA Graphs/NVDEC/NVENC integration is complete; those are later evidence-driven work.
