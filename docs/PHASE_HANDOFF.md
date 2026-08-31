# Phase Handoff - Phase 5 Pose-Aware Source Selection

Date: 2026-08-31

## Current state

Phase 5 is implemented and validated in commit `9e11d83` on baseline commit
`079fb7a`. FaceSet V2 source metadata is now
used by an opt-in pose-aware selector. Existing V1 loading/selection,
temporal tracking, detector/ROI recovery, 3D reconstruction, swapper,
enhancer, provider, and hardware paths remain available.

## Implementation

- `app/roop/pose_source_selector.py` estimates target pose classically and
  scores V2 sources by pose, quality/identity, expression, lighting,
  proportions, scale, and hysteresis.
- `app/roop/procmgr_tracking.py` annotates replayed faces after established
  smoothing and roll resolution, preserving track ordering and IDs.
- `app/roop/ProcessMgr.py` uses the selector only when the existing
  `use_source_bank` option is enabled and gates image-source 3D correction on
  the selector’s confidence/reason decision.
- `app/roop/face_3d_recon.py` bounds yaw/pitch affine compensation, rejects
  unstable transforms, and only mirrors opposite strong off-axis views.
- `docs/POSE_AWARE_SOURCE_SELECTION.md` documents the record, scoring,
  fallback, configuration, and compatibility contract.

## Configuration and compatibility

`use_source_bank` remains opt-in. `ROOP_POSE_SOURCE_SWITCH_MARGIN` controls
source hysteresis and defaults to `0.035`. No user defaults or `.fsz` files
were rewritten. Preserve TensorRT/CUDA, FP16/FP32/mixed precision, detector
alternatives/pools, RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic,
UltraMax, all enhancers, 3D reconstruction, CPU/DirectML/AMD/Apple fallbacks,
RTX 4070 pool settings, RTX 3060 single-context guard, and laptop custom look
settings.

## Evidence

- Phase 5 tests: **15 passed**.
- Existing tracking regression: **38 passed**.
- Full suite: **1545 passed, 1 skipped, 0 failures** in **44.960 s**.
- Selector benchmark: **7,396.64 selections/s** and **205,510.14 warp
  plans/s** over 10,000 synthetic iterations and 7 sources.
- Python compilation, 45 tracked JavaScript syntax checks, and
  `git diff --check` passed.
- No real-photo identity/temporal quality measurement or physical RTX 3060
  end-to-end benchmark was performed; do not treat the microbenchmark as
  video FPS or a quality claim.

## Next session instructions

1. Start with `git status`, recent commits, `logs` (latest first), this file,
   `docs/OPTIMIZATION_PROGRESS.md`, and the Phase 5 design document.
2. Keep `G:\pinokio\prototype\system\examples\mochi\start.js` as the
   launcher reference. No launcher changes are needed; if one is proposed,
   reread `PINOKIO.md` and preserve the captured `input.event[1]` →
   `local.set` URL pattern.
3. Build an order-balanced real-photo evaluation set covering yaw 0/30/45/60/
   75/profile, upward/downward pitch, roll, and inversion.
4. Measure V2 source-choice accuracy, identity similarity, ID switches,
   landmark/pose jitter, detail/expression/lighting quality, end-to-end FPS,
   VRAM, and RSS with source selection and 3D fallback separately enabled on
   RTX 4070 and RTX 3060.
5. Review sparse pose bins, profile perspective, inversion, sudden pose
   changes, and frontal quality before considering any default change or new
   neural model.
6. Before finalizing, rerun targeted/full tests, available real benchmarks,
   Python/launcher syntax checks, `git diff --check`, and update both state
   documents with exact results and commit SHA.

## Do not break

Preserve output ordering, audio/segment behavior, `.fsz` compatibility,
wrong-FaceSet/finite/output-integrity guards, existing detector pools and ROI
helper, all provider/hardware fallback paths, and the dual-device memory/look
contracts. Do not enable unrelated expensive models by default.
