# Session Handoff

Updated: 2026-08-28

## State

Phase 0, Phase 1, and Phase 2 are complete in repository history. Phase 2 ends at `d298fbf`. Phase 3 is implemented but uncommitted in the current working tree. Phase 4 has not started.

The four optimization documents requested by the workflow did not exist at session start and were created during this session from the repository state. Treat the repository and these documents as the source of truth for the next session.

## Phase 3 implementation to review

The working tree adds model-specific TensorRT resource specifications, shape/batch-aware slot estimates, live free-VRAM checks, safety margins, resident-pool accounting, pressure-based admission limits, bounded session-pool queues/waits, active-lease tracking, race-safe release/resize/warmup transitions, and explicit-setting provenance. Model call sites pass resource keys and input shapes across swapper, enhancers, masks, detectors, expression, and face analysis. The benchmark now tests all four context counts and reports context knees. The TensorRT cache namespace now includes CUDA, driver, and TensorRT versions in addition to GPU/SM/ORT/precision/tuning identity.

Do not reset or discard these changes. Do not touch the untracked `facesets/` user-data directory.

## Verification

- Phase 3 targeted and cache-identity suites passed (`13` tests in the final focused verification; the earlier combined Phase 3 suite passed `180`).
- Full suite: `1344` tests, `2` known unrelated failures, `1` skipped.
- `py_compile` passed for all changed Python modules.
- `git diff --check` passed; line-ending warnings are only Git’s LF/CRLF notices.

Known unrelated failures:

- benchmark/app direct environment flag parity;
- four settings missing from the UI palette catalog.

## Benchmark handoff

RTX 4070 full synthetic benchmark completed with heavy composite `21.67 FPS`. Context knees were detector 3 and most other stages 2; increasing the TRT pool from 2 to 4 reduced composite throughput to `20.15 FPS`. No real-video end-to-end before/after claim is available because the pre-Phase-3 run stopped during cold XSeg initialization.

## Immediate next action

Run the final post-patch targeted/full checks, review the complete diff, then commit Phase 3. After that, execute a reproducible real-video benchmark on the RTX 4070 and the RTX 3060 Laptop profile, including decode, detection, swap, enhancement, stabilization, compositing, encoding, end-to-end FPS/latency, CPU, VRAM, RAM, queue depth, and worker counts. Only after that evidence should Phase 4 be planned.
