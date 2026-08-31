# Phase Handoff - Phase 10 Temporal Engines On The Parallel-Block Path

Date: 2026-08-31
Device: physical RTX 4070, driver 616.56, TensorRT 10.9.0.34, ORT 1.23.2

## Read this first

Two things from this session change how you should read any benchmark here.

1. **A regression in `1c0efd7` had disabled the swap on the shipped default
   path** (Phase 9). Fixed in `da30500`. Any render made between those two
   commits swapped almost nothing while returning 0, reporting `100.0%` in the
   swap audit, and running *faster* than the correct pipeline. Discard them.
2. **This machine drifted 2.9x during Phase 10, on the unchanged default
   configuration.** The null control read 12.91, then 4.5, then 8.58 fps with
   nothing changed. Host RAM available moved 14.1 -> 4.5 -> 15.0 GB, and
   `_default_stab_chunk_mb` is `available * 0.40 / 6`, so the block geometry --
   and therefore which code path a run takes -- depends on free memory at start.
   **Run a null control per measurement window, not per session, and record
   `faces_seen` beside fps.**

## Current state

| item | state |
|---|---|
| Phases 0-5 | VERIFIED, untouched |
| Phase 6 (pose/source harness) | implemented, committed |
| Phase 6B `ROOP_TEMPORAL_IDENTITY` | opt-in; **now runs at width**, quality still unvalidated |
| Phase 7 `ROOP_TEMPORAL_OCCLUSION` | opt-in; falls back to sequential by design, quality unvalidated |
| Phase 8 `ROOP_TEMPORAL_EXPRESSION` | opt-in; measured, recommended for promotion, still default off |
| Phase 9 | the dedent fix + first real temporal measurements |
| Phase 10 | this document |

All three temporal flags remain **disabled by default**. No saved user
configuration, `.fsz` format, model, provider policy, pool setting or look value
was changed.

## What changed in Phase 10

`ProcessMgr.py:1671` used to set `threads = 1` whenever identity or occlusion was
on. Phase 9 measured that at -62.9%, and proved with a `threads=1` no-flag
control that the pinning was the entire cost.

Ordered is not the same as serial. `_run_stab_parallel` already gives each worker
a contiguous block, in frame order, with its own filter instances and a warm-up
it discards. These engines now ride that path:

- `warmup_frames(eps)` on both engines, derived from their own recurrences
  (identity 15 frames, occlusion 44), so `_stab_warmup_frames` picks them up.
- `clone_for_block()` on both. Identity carries the pre-pass-derived identity /
  pose / source state (read-only during the swap) and clears only the three
  fields the swap phase mutates. Occlusion carries nothing.
- `ProcessMgr._temporal_engine(name)` -- every mutating site reads through it.
- `set_ordered` now asks "is this worker seeing frames in order", true on the
  sequential loop and inside a block. It used to be `not _parallel_stab`, which
  made the occlusion engine return `disabled` for every frame of a parallel run.
- `_stab_min_block_multiple` (1 by default = no-op; 3 for these engines) floors
  the block at 3x the warm-up, because at a 1:1 ratio the priming costs more than
  the extra workers return.

## Evidence

Locked fixture, counterbalanced ABBA, every arm's path verified from
`faces_seen` (679 = one sequential pass, >750 = parallel re-processing warm-up):

| arm | position | fps | path | wrong faceset |
|---|---:|---:|---|---:|
| NEW | 1 | 8.55 | parallel | 0 |
| OLD | 2 | 4.41 | sequential | 0 |
| OLD | 3 | 4.43 | sequential | 0 |
| NEW | 4 | 7.07 | parallel | 0 |

**NEW 7.81, OLD 4.42, +76.7%.** OLD arms agree to 0.5%; the worst NEW arm beats
the best OLD arm by 60%.

Output equivalence, sequential vs parallel, 600 frames: mean absolute difference
**0.35% of full scale**, max 1.40/255, and at the 45-frame block boundaries
**0.857 vs 0.883 elsewhere (ratio 0.97)** -- boundary frames are not worse than
the rest, so the warm-up is doing its job.

The occlusion decision, measured in one early window: parallel at a 1:1
block/warm-up ratio was **3.75 fps against 5.18 pinned to one worker**, so the
floor now routes it to sequential instead. That is the old behaviour, not a
regression, and not a win.

Suite: **1596 tests, 1 skipped, 0 new failures**; 16 new contracts in
`tests/test_temporal_parallel_blocks.py`.

Not claimed: any 3060 number, any quality validation of either flag, any
cross-window fps comparison, and any optimality for the 3x floor.

## Next session: exact starting point

1. **Cut identity's per-face cost.** This is now the limiter and it is the only
   direction Gate E left open (remove work per face, not redistribute it). With
   the warm-up pinned to the baseline's 6 frames to isolate scheduling, identity
   ran **8.34 fps against a 12.90 baseline -- -35% that is the engine itself**.
   `blend_output` does two full-crop `GaussianBlur(0, 0, 4.0)` passes plus a
   resize per face per frame; `stabilize_mask` runs per face per frame. Profile
   those two first.
2. **Shorten occlusion's warm-up so it can go parallel at all.**
   `ROOP_OCCLUSION_ENTER_ALPHA` 0.90 -> 0.75 takes the warm-up from 44 to 17
   frames, which fits a 3x block in this budget. Measure the QUALITY cost of the
   faster object-matte release first -- that alpha exists to stop an occluder
   popping back in.
3. **Sweep the 3x floor.** It caps overhead at 33% and produced the better
   outcome in both measured cases, but 2x and 4x were not tried.
4. **Validate the quality of both flags on real footage.** Phase 9's disposition
   stands and is untouched by this phase; identity is now cheap enough to test
   properly.
5. **Cross-target: the 3060.** Less RAM means the sequential fallback fires more
   often -- the safe direction, but unverified. Run the null control there first;
   that card's noise floor is ~3.3% at 600 frames and ~6% at 60.
6. Inherited: Phase 3's RSS gate still fails on the 3060 at 3.73 GB; interacting
   faces remains characterized but unsolved.

## Standing rules that earned their keep this session

- **Null control per measurement window.** A 2.9x drift on the unchanged default
  made two arms look like a regression and a null result respectively.
- **Record the path, not just the number.** `faces_seen` distinguishes a
  sequential fallback from a slowdown for free on this fixture; without it they
  are identical.
- **Read the face count beside the fps.** Phase 9's regression presented as +47%.
- **A regression test must be shown to fail on the broken code.**
- **Prove a code path executes before believing "no effect".**

## Do not break

RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic, UltraMax and every other
enhancer; TensorRT and the FP16/FP32/mixed precision policy; source-bank and 3D
paths; detector alternatives and `det_size` handling; V1/V2 `.fsz`
compatibility; provider fallbacks; face-overlap ownership; the default
(no-flag) stabilization geometry, which `_stab_min_block_multiple = 1` keeps
bit-identical; the RTX 4070 pool settings; the RTX 3060 single-context guard and
its laptop look values.
