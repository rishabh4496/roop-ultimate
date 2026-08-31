# Phase Handoff - Phase 8 Target Expression Preservation

Date: 2026-08-31

## Current state

Phase 8 is implemented as an opt-in, model-free target-expression continuity
layer and remains open for real-video quality validation. It is based on
`ea7b2969cd2d0a110808e41fb533dbd9c4e72cb1`; no new commit was created and
working changes remain uncommitted. `ROOP_TEMPORAL_EXPRESSION` is disabled by
default. Phase 6 identity and Phase 7 occlusion remain opt-in independently.

## Implementation

`app/roop/temporal_expression.py` measures the target's existing 106-point
landmarks and maintains bounded state per track: left/right eye openness,
independent blink/wink state, mouth openness/MAR, eyebrow movement, jaw
movement, and confidence. Adaptive filtering and eye hysteresis suppress
detector chatter while allowing large real expression transitions through.

The ordered tracker writes expression measurements and event strengths onto
each target face. The swap worker can restore only target eye ellipses and the
target mouth polygon during an event. It never temporally blends the whole
face. Manual eye/mouth restoration retains precedence, and usable lip-sync
retains mouth precedence. No model or `.fsz` migration was added.

Enable deliberately with `ROOP_TEMPORAL_EXPRESSION=1`. Controls and benchmark
usage are documented in [`PHASE8_EXPRESSION.md`](PHASE8_EXPRESSION.md).

## Changed files for Phase 8

- `app/roop/temporal_expression.py`
- `app/roop/ProcessMgr.py`
- `app/roop/procmgr_tracking.py`
- `app/roop/procmgr_masking.py`
- `app/tests/test_temporal_expression.py`
- `app/tests/phase8_expression_bench.py`
- `README.md`
- `docs/PHASE8_EXPRESSION.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

Earlier uncommitted Phase 6/7 files remain preserved; do not reset or rewrite
them.

## Evidence

- Targeted Phase 8 contracts: **8 passed, 0 failures**.
- Changed Python modules and benchmark/tests compile successfully.
- Full regression: **1575 passed, 1 skipped, 0 failures** in **72.947 s**.
- `phase8_expression_bench.py --scenario all` was executed and correctly
  returned **pending** because no real paired target/output video was supplied.
- No real expression accuracy, temporal-difference, visual-review, end-to-end
  FPS, VRAM, RSS, CPU/GPU utilization, or quality improvement is claimed.

## Next session instructions

1. Obtain paired original target/rendered output videos for all eleven cases:
   `slow_blink`, `fast_blink`, `asymmetric_blink`, `wink`, `half_open_eyes`,
   `talking`, `smiling`, `mouth_wide_open`, `teeth_visible`, `frowning`, and
   `fast_transitions`.
2. Render every case with `ROOP_TEMPORAL_EXPRESSION=0` and `=1`, then run
   `app\\env\\Scripts\\python.exe app/tests/phase8_expression_bench.py` with
   each pair. Record per-eye/mouth correlation, MAE, range retention,
   frame-delta agreement, detection coverage, FPS, seconds/frame, VRAM, RSS,
   CPU/GPU utilization, and manual findings.
3. Review eyelid contours, slow/fast/asymmetric blinks, wink, half-open eyes,
   speech, smile/frown, teeth, wide mouth, fast transitions, pose changes,
   occlusion, and identity texture. Tune thresholds only from measured clips.
4. Validate both hardware profiles without changing policy: RTX 4070 pool 2
   / 12 GB behavior; RTX 3060 pool 0, single context, global guard, and custom
   look values `blend_ratio 0.85`, `face_mask_blend 25`, `merger_sharpen 0.55`,
   `stabilize_enhancer_strength 0.6`.
5. Do not enable the new flag by default or add a heavier expression model
   until the real quality/performance trade-off is recorded.

## Pinokio guardrails

No launcher files were changed. If a launcher change becomes necessary, keep
`G:\\pinokio\\prototype\\system\\examples\\mochi\\start.js` as the
reference, reread `G:\\pinokio\\prototype\\PINOKIO.md`, preserve the
captured `input.event[1]` to `local.set` URL pattern, and check `logs` before
debugging.

## Do not break

Preserve RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic, UltraMax, all
existing enhancers, TensorRT, FP16/FP32/mixed precision, source-bank and 3D
paths, detector alternatives, V1/V2 `.fsz` compatibility, provider fallbacks,
temporal identity/occlusion tracking, face overlap ownership, RTX 4070 pool
settings, RTX 3060 single-context guard, and laptop look settings.
