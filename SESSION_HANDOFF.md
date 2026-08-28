# Session Handoff

Updated: 2026-08-28

## State

Phase 0, Phase 1, and Phase 2 are complete in repository history. Phase 2 ends at `d298fbf`. Phase 3 implementation is committed at `c439e43`; its physical RTX 3060 validation is still outstanding. Phase 4 is implemented and validated on the physical RTX 4070, but remains open for the second hardware profile.

The Phase 5 precision-optimization request was intentionally deferred. The standing phase gate requires completing Phase 4 on both hardware profiles first; no model precision policy, precision benchmark, or FP32 guard change was started.

The four optimization documents requested by the workflow did not exist at session start and were created during this session from the repository state. Treat the repository and these documents as the source of truth for the next session.

## Phase 3 implementation in `c439e43`

The working tree adds model-specific TensorRT resource specifications, shape/batch-aware slot estimates, live free-VRAM checks, safety margins, resident-pool accounting, pressure-based admission limits, bounded session-pool queues/waits, active-lease tracking, race-safe release/resize/warmup transitions, and explicit-setting provenance. Model call sites pass resource keys and input shapes across swapper, enhancers, masks, detectors, expression, and face analysis. Phase 4 extends the benchmark to 1/2/3/4/6 contexts, records stability, and uses validated model-specific knees for automatic selection. The TensorRT cache namespace includes CUDA, driver, and TensorRT versions in addition to GPU/SM/ORT/precision/tuning identity.

Do not reset or discard these changes. Do not touch the untracked `facesets/` user-data directory.

## Verification

- Phase 3 targeted and cache-identity suites passed; the current focused regression suite passed `152` tests plus `12` subtests.
- Full suite: `1343` tests, `1341` passed, `2` known unrelated failures, `1` skipped; `589` subtests passed.
- `py_compile` passed for all changed Python modules.
- `git diff --check` passed; line-ending warnings are only Git’s LF/CRLF notices.

Known unrelated failures:

- benchmark/app direct environment flag parity;
- four settings missing from the UI palette catalog.

## Benchmark handoff

The RTX 4070 full TensorRT mixed benchmark measured model-specific knees: detector `6`, recognition/landmarks/masks/enhancer `2`, and swapper `3`. Curves were measured independently at contexts `1/2/3/4/6`; the aggregate heavy composite reached `20.55 FPS` at `trt_pool=3`, while widening it to `6` fell to `17.96 FPS`. The higher-utilization setting was rejected. Automatic knee reuse now validates GPU/VRAM, provider, precision, selected model, input shape, and TensorRT tuning identity; explicit numeric pool settings remain authoritative.

The real-video RTX 4070 Phase 4 run used the available 8-second, 1280x720/30 fps, 240-frame fixture with explicit `2/2/2` detector/detmask/TRT pools. It completed in `46.9 s` (`5.12` end-to-end FPS; `195.4 ms/frame`), with `7,984 MB` peak and `4,764 MB` sampled-average whole-card VRAM, plus `28.3%` average / `55%` peak GPU utilization. CPU utilization was not captured reliably. The output composited and enhanced `238/238` face-bearing frames, but the selected source did not match the fixture, so this is throughput/resource evidence only.

The completed real-video RTX 4070 validation used the available 8-second, 1280x720/30 fps, 240-frame fixture and the live TensorRT mixed/UltraMax configuration. The harness measured `394.4 s` (`0.61` end-to-end fps), with ffmpeg processing at `376.41 s` (`0.64` fps). The runtime emitted `workers=12`, `queue=3`, and about `9.52 GiB` runtime memory; external monitoring saw `28.26 GiB` system-RAM peak, `8.03 GiB` whole-card VRAM peak, and `34.0%` average / `89.0%` peak GPU utilization. The output completed, but the selected source did not match the fixture (`237/238` face-bearing frames refused), so this is throughput/resource evidence only.

The counterbalanced `ab_small_card_pools.py` run completed in simulated `ROOP_VRAM_GB=6` mode: pools `0/0` averaged `11.61` fps and forced `2/2` averaged `13.54` fps; all `238` faces were enhanced. Because both arms ran on the physical 4070 with only the policy tier overridden, this is not physical 3060 evidence.

## Immediate next action

Run the same reproducible real-video benchmark and independent 1/2/3/4/6 model context sweep on a physical RTX 3060 Laptop profile, including decode, detection, swap, enhancement, stabilization, compositing, encoding, end-to-end FPS/latency, CPU, VRAM, RAM, queue depth, and worker counts. Keep the 6 GB policy at pools `0/0` unless measured evidence supports a safe change. Do not start Phase 5.
