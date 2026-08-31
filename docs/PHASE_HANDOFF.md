# Phase Handoff - Phase 9 Real-Video Validation Of The Temporal Stack

Date: 2026-08-31
Device: physical RTX 4070, driver 616.56, TensorRT 10.9.0.34, ORT 1.23.2

## Read this first

**A regression in `1c0efd7` had disabled the swap on the shipped default path,
and it is fixed in this phase.** If you are on another machine, `git pull`
before rendering anything. Any render made between `1c0efd7` and this commit
produced almost no swaps while returning 0, reporting `100.0%` in the swap
audit, and running *faster* than the correct pipeline. Discard those outputs
and any timing taken from them.

## Current state

| item | state |
|---|---|
| Phases 0-5 | VERIFIED, untouched |
| Phase 6 (pose/source harness) | implemented, committed `1c0efd7` |
| Phase 6B `ROOP_TEMPORAL_IDENTITY` | implemented; **measured, hold** |
| Phase 7 `ROOP_TEMPORAL_OCCLUSION` | implemented; **measured, hold** |
| Phase 8 `ROOP_TEMPORAL_EXPRESSION` | implemented; **measured, recommended for promotion, still default off** |
| Phase 9 | this document |

All three temporal flags remain **disabled by default**. No saved user
configuration, `.fsz` format, model, provider policy, pool setting or look
value was changed in this phase.

The Phase 6/6B/7/8 sections of `OPTIMIZATION_PROGRESS.md` state "no new commit
was created ... remains uncommitted". That is stale - all of it is in `1c0efd7`.
They also record their benchmarks as `pending` for want of video fixtures;
the fixtures exist under `G:\pinokio\roop-keep\` and always did.

## What changed in Phase 9

- `app/roop/procmgr_tracking.py` - in `_build_temporal_faces`, the per-frame
  `out.setdefault(i, []).append(f)` and the `f['_track_id'] = t['id']` stamp
  were dedented out of their `for i, f in merged.items()` loop, so the append
  ran once per TRACK. Restored to the loop body; the `_track_id` stamp is
  restored to unconditional (it had been gated behind the opt-in Phase 5-8
  flags, which returned the source binding to the centroid fallback).
- `app/tests/test_temporal_faces_replay.py` - new. 5 coverage contracts over
  `_build_temporal_faces`, which nothing covered.
- `app/tests/phase8_expression_bench.py` - initializes the detector through
  `angle_bench.init_pipeline` before calling `get_all_faces`, which otherwise
  swallows detector exceptions and grades 0 frames while reporting
  `insufficient_detections`.
- `docs/OPTIMIZATION_PROGRESS.md`, `docs/PHASE_HANDOFF.md`.

No launcher file was touched.

## Evidence

Locked fixture: `double/d4.mp4`, 1280x720, frames 0..600, capture frame 4930,
sources `harjot,gargee`; realswap / GPEN 256 Pro / RealityUX / hevc_nvenc /
tensorrt / 12 threads, straight from `config.yaml`.

- **Null control (2 identical arms):** 12.91 and 12.87 fps, **0.3% spread**;
  819 faces seen, 809 swapped, 0 wrong-faceset in both.
- **Before the fix:** 5 faces seen, 3 swapped, **19.0 fps** - 47% "faster" for
  doing nothing.
- **`ROOP_TEMPORAL_EXPRESSION`, counterbalanced ABBA:** -0.77% throughput
  (inside the set's 1.4% spread, not resolvable); face counts identical in all
  four arms; mouth correlation 0.9085 vs 0.8547 and mouth MAE 0.1342 vs 0.1671
  (**-19.7%**) with complete separation between conditions; eye channels not
  resolvable (within-condition spread ~3x the delta).
- **`ROOP_TEMPORAL_IDENTITY` 4.34 fps, `ROOP_TEMPORAL_OCCLUSION` 5.18 fps,
  `threads=1` control with no flag 4.79 fps**, all three at 679/670 faces.
  The cost is the forced single-worker downgrade (12.90 -> 4.79, **-62.9%**),
  not the features.
- **Suite: 1580 tests, 1 skipped, 0 new failures**, 43.1 s. The 2
  `test_nvdec_reader` errors are the pre-existing ffmpeg-spawn environment
  failures. The new guard was verified to **fail 4 of 5** on the reverted code.

Not claimed: any 3060 number, any second clip, any manual visual review, any
identity-similarity or temporal-flicker measurement, and any Phase 6B/7 quality
result - their benches were not run, because a 2.7x throughput collapse decides
those features before quality does.

## Next session: exact starting point

Work the items in this order.

1. **Remove the `threads = 1` forcing at `app/roop/ProcessMgr.py:1671-1682.**
   This is the highest-value item in the phase and it is a design task, not a
   tuning one. The requirement is an ordered *output history*, which does not
   need an ordered *swap*: `_run_stab_parallel` already runs parallel workers
   behind one ordered `_writer`, and that is the shape to copy. Success is
   `ROOP_TEMPORAL_IDENTITY=1` at 12-thread throughput. Re-measure with
   `tests/baseline_controlled.py --env ROOP_TEMPORAL_IDENTITY=1`, counter-
   balanced, against a fresh null control.
   Note while in there: the `_label` ternary in that block can never yield
   `'TemporalExpression'`, because the expression flag is not in the guard.

2. **Close the expression promotion.** Repeat the ABBA design on two or three
   further clips - the four `G:\pinokio\roop-keep\expression\` clips are the
   right material - and report the **eye** channels, which are the only thing
   holding it. If they stay inside the noise while the mouth effect reproduces,
   set the default on. Command shape:

       env/Scripts/python.exe tests/baseline_controlled.py --tag <tag> \
           --out output/phase9 --env ROOP_TEMPORAL_EXPRESSION=1
       env/Scripts/python.exe tests/phase8_expression_bench.py \
           --target-video <out>/work/clip.mp4 \
           --output-video <out>/work/clip_*.mp4 --json <report>.json

3. **Investigate the 17% face loss under `threads=1`** (819 -> 679 with no flag
   set). Found in this phase, not investigated. It matters beyond the temporal
   stack: every path that downgrades to one worker - the 16 GB stabilizer-budget
   collapse on the 3060 among them - pays it.

4. **Cross-target: run all of the above on the RTX 3060.** Nothing in Phase 9
   was measured there. Preserve its single-context guard, pool 0/0 tier and its
   deliberate look values (`blend_ratio 0.85`, `face_mask_blend 25`,
   `merger_sharpen 0.55`, `stabilize_enhancer_strength 0.6`). Run the null
   control there first - that card's noise floor is a property of the window,
   ~3.3% at 600 frames and ~6% at 60.

5. Inherited and untouched: Phase 3's RSS gate still fails on the 3060 at
   3.73 GB; interacting faces remains characterized but unsolved; DMDNet is
   guarded but its original failing population was never reproduced.

## Standing rules that earned their keep again this session

- **Run a null control before any A/B.** It is what made 0.77% legible as noise
  and 19.7% legible as signal.
- **Read the face count beside the fps.** The regression presented as a 47%
  speedup. A configuration that goes faster by swapping fewer faces has not got
  faster.
- **The swap audit counts INTENT, not outcome.** It read 100% throughout.
- **A regression test must be shown to fail on the broken code**, or it is not
  evidence.
- **Prove a code path executes before believing "no effect".** The expression
  engine was structurally unable to run before the dedent fix.

## Do not break

RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic, UltraMax and every other
enhancer; TensorRT and the FP16/FP32/mixed precision policy; source-bank and 3D
paths; detector alternatives and `det_size` handling; V1/V2 `.fsz`
compatibility; provider fallbacks; face-overlap ownership; the RTX 4070 pool
settings; the RTX 3060 single-context guard and its laptop look values.
