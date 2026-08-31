# Phase Handoff - Phase 11 Temporal Identity Per-Face Cost

Date: 2026-09-01
Device: physical RTX 4070, driver 616.56, TensorRT 10.9.0.34, ORT 1.23.2

## Read this first

Three things from this session change how you should read any benchmark here.

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

3. **The locked 1280x720 `double/d4.mp4` fixture is outside this repository.**
   It is available at `G:/pinokio/roop-keep/double/d4.mp4` on the 4070 host and
   was used for three Phase 11 feature-level renders. The 3060 still has no
   Phase 11 run.

## Current state

| item | state |
|---|---|
| Phases 0-5 | VERIFIED, untouched |
| Phase 6 (pose/source harness) | implemented, committed |
| Phase 6B `ROOP_TEMPORAL_IDENTITY` | opt-in; **now runs at width**, quality still unvalidated |
| Phase 7 `ROOP_TEMPORAL_OCCLUSION` | opt-in; falls back to sequential by design, quality unvalidated |
| Phase 8 `ROOP_TEMPORAL_EXPRESSION` | opt-in; measured, recommended for promotion, still default off |
| Phase 9 | the dedent fix + first real temporal measurements |
| Phase 10 | parallel-block execution for identity/occlusion; measured |
| Phase 11 | identity per-face cost reduction; **OPEN / INCOMPLETE** |

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

## Phase 11 implementation and evidence

`blend_output` now uses a 128px working crop for its low-frequency correction,
controlled by `ROOP_TEMPORAL_IDENTITY_LOWPASS_SIZE`. `0` is the old
full-resolution reference path. `stabilize_mask` avoids redundant validation
copies while retaining state ownership and an independent return buffer.

Files changed:

- `app/roop/temporal_identity.py`
- `app/tests/test_temporal_identity.py`
- `app/tests/bench_temporal_identity_cost.py`
- `docs/ENV_FLAGS.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

Three counterbalanced 1200-call pairs at 256x256 measured:

| path | blend calls/s | mask calls/s |
|---|---:|---:|
| full-resolution reference (`0`) | 747.9 | 1283.6 |
| reduced working crop (`128`) | 1277.4 | 1290.6 |
| change | +70.8% | +0.5% |

The mask result is neutral within host noise and is recorded as an allocation
reduction, not a speed promotion. The reduced identity path remained finite,
uint8, dimensionally valid, detail-preserving, and under the synthetic MAE
bound against the reference. This is not real-footage quality validation.

Validation: targeted temporal set **38 passed, 1 warning**; full suite **1605
passed, 1 skipped, 595 subtests passed, 2 existing warnings**; Python
compilation and `git diff --check` passed. Three physical RTX 4070
feature-level renders also completed 600/600 with zero wrong FaceSets; the RTX
3060 feature run and retained-output visual review remain pending.

The 128px path remains **opt-in experimental**. The approximation is not
byte-identical to the reference; `lowpass_size=0` remains available for
diagnostics. No temporal flag is promoted to default.

## Complete-phase checklist audit — NOT COMPLETE

| Requirement | Evidence | Status | Missing |
|---|---|---|---|
| IMPLEMENT | Bounded low-pass identity path and mask allocation reduction | PASS | None in scoped implementation |
| TEST | 38 targeted passes; full suite passes | PASS | None in unit coverage |
| BENCHMARK | Reproducible component A/B plus three locked-fixture 4070 renders | PARTIAL | Physical RTX 3060 run and comparable cross-arm attribution |
| REGRESSION TEST | 1605 passed, 1 skipped, 595 subtests; all 4070 arms had 0 wrong FaceSets | PARTIAL | Retained-output manual visual regression review |
| DOCUMENT | `ENV_FLAGS.md` and `OPTIMIZATION_PROGRESS.md` updated | PASS | None |
| HANDOFF | This file records commands, constraints, and next starting point | PASS | It intentionally hands off the missing validation |

Phase 11 must not be marked complete until the partial benchmark and regression
items are closed with a physical RTX 3060 feature run, comparable attribution,
and retained-output visual review. The current synthetic benchmark and the
automated wrong-FaceSet check are not substitutes.

## Requested Phase 9 handoff — identity-specific detail preservation

Status: **OPEN / INCOMPLETE**.

Implemented starting point:

- `app/roop/faceset_v2.py` creates and aggregates signed canonical
  high-frequency residuals with persistence confidence.
- `app/roop/identity_detail.py` decodes, template-warps, exposure-scales,
  masks, smooths, and composites the representation.
- `app/roop/ProcessMgr.py` invokes it after enhancer, post-enhance colour,
  merger, manual mask, and temporal low-band stages.
- `app/roop/temporal_identity.py` owns bounded per-track detail history and
  clears it on source changes.
- `identity_detail_strength` is 0 by default and is available through config,
  preview, and swap API payloads. Existing target-texture
  `detail_transfer_strength` is intentionally separate.

Validation already completed:

- Focused command → **28 passed, 1 warning**:
  `app/env/Scripts/python.exe -m pytest app/tests/test_identity_detail.py app/tests/test_faceset_v2.py app/tests/test_temporal_identity.py -q`.
- Component benchmark → 0.839016 synthetic retention correlation; 43.3257%
  temporal-delta reduction; 290.71 restorations/s.
- 4070 controlled V1-backed smoke, GPEN 256 Pro / RealityUX / TensorRT /
  RealSwap, locked `double/d4.mp4`, strength 0.35 → **120/120**, return code
  0, no identity-detail runtime errors. It did not exercise V2 metadata because
  the locked `harjot/gargee` archives are legacy V1.

Exact next-phase starting point:

1. Create or obtain a V2 copy of the locked source archives without changing
   source identities or the target fixture; confirm `FaceSet.format_version ==
   2` and `identity_detail_for()` returns a valid residual for every selected
   source.
2. Extend the retained-output harness so the V2-backed run keeps its output
   video and records per-frame detail metrics. Run off / strength 0.35 / a
   confidence-reduced arm on the 4070 with identical capture, enhancer, mask,
   provider, codec, stabilizers, and frame range. Compare retention, temporal
   delta, wrong FaceSets, FPS, RSS, and VRAM only when paths are comparable.
3. Manually review mole, freckles, scar, wrinkle, and microtexture regions over
   frontal, turned, low-resolution, motion-blurred, and dark frames, including
   occluders and expression changes. Confirm omission beats flicker when
   confidence drops.
4. Run the same V2-backed component and real-footage matrix on the physical
   RTX 3060 while preserving `blend_ratio=0.85`, `face_mask_blend=25`,
   `merger_sharpen=0.55`, and `stabilize_enhancer_strength=0.6`; keep its
   single-context/global GPU guard and 1536 MB hard cap.
5. Test GPEN, UltraMax, and at least one additional restorer with identity
   detail enabled; verify post-restorer ordering and visual retention. Do not
   promote the feature or change its default until these checks pass.

Do not restart Phase 11 temporal optimization from scratch. Its 4070 evidence,
open 3060 validation, and retained-output visual-review requirements remain
unchanged below.

## Next session: exact starting point

1. On the 4070, rerun an order-balanced off/reference (`0`)/128px set with
   retained output videos or a quality-review harness. Record FPS,
   `faces_seen`, wrong FaceSets, output finiteness/order, peak RSS, and peak
   VRAM; do not attribute raw FPS when face counts diverge.
2. Manually review annotated occluder, eyes, mouth, hair, and difficult-pose
   frames from those retained outputs; synthetic MAE is not enough for
   promotion.
3. Run the component and real-video checks on the physical RTX 3060 while
   preserving `blend_ratio=0.85`, `face_mask_blend=25`,
   `merger_sharpen=0.55`, and `stabilize_enhancer_strength=0.6`. Do not copy
   4070 results or caches.
4. Resume the Phase 10 follow-up: measure the quality cost of lowering
   `ROOP_OCCLUSION_ENTER_ALPHA` from 0.90 toward 0.75 before considering its
   3x warm-up block, then sweep the 2x/3x/4x minimum block floor in one stable
   measurement window.
5. Decide whether the identity experiment can be promoted or must remain
   opt-in only after those results. Interacting-face behavior and the inherited
   3060 RSS gate remain open.

## Requested Phase 10 handoff - target-conditioned lighting and color realism

Status: **OPEN / INCOMPLETE**.

Implemented starting point:

- `app/roop/appearance_conditioning.py` owns robust target illumination/chroma
  analysis, NORMAL/DARK/VERY_DARK classification, low-light restoration and
  sharpening factors, restorer protection, and the bounded per-track EMA.
- `app/roop/procmgr_color.py` extends the existing color-transfer path with
  target-conditioned low-frequency spatial illumination, exposure/highlight
  quantile anchors, local contrast, and bounded skin-region chroma. It does not
  create a wholesale texture paste.
- `app/roop/ProcessMgr.py` analyzes the aligned target crop once per face,
  reuses its stabilized result for both color passes, protects dark output from
  GPEN/UltraMax/other restorers, and passes the tier into merger sharpening and
  clarity. The appearance engine is cloned/reset with the existing ordered
  contiguous-block stabilizer lifecycle.
- `app/settings.py`, `app/api.py`, and the React Face Swap controls expose the
  feature. It is opt-in and leaves the existing custom 4070/3060 look defaults
  unchanged when off.

Validation already completed:

- Focused tests: **49 passed, 1 warning**.
- Full suite: **1618 passed, 1 skipped, 598 subtests passed, 2 warnings**.
- Component benchmark: **23.3602 ms/call** analysis; stable-light temporal
  colour delta reduction **82.2298%** (0.02481532 to 0.00440974) over the
  daylight/indoor/tungsten/fluorescent/sunset/blue/mixed/night/street/low-
  exposure/backlighting fixture set.
- 4070 real integration smoke: **120/120 frames**, **294/294 faces**,
  **0 wrong FaceSets**, **3.67 fps**, approximately **10.01 GB peak RSS**,
  target appearance enabled with RealSwap/RealityUX/GPEN 256 Pro/TensorRT.
  Legacy V1 sources mean this does not assess V2 identity detail.

Exact next-phase starting point:

1. Keep the current code and run the component benchmark on the physical RTX
   3060 laptop with its single-context/global GPU guard, 1536 MB stabilization
   cap, adaptive block sizing, and preserved look values:
   `blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`,
   `stabilize_enhancer_strength=0.6`. Record analysis cost, RSS, VRAM, and any
   runtime fallback; do not copy 4070 results or caches.
2. Prepare or obtain a locked V2 source archive and verify
   `FaceSet.format_version == 2` and `identity_detail_for()` before combining
   this phase with requested Phase 9 detail preservation.
3. Run retained-output real-footage arms for daylight, indoor/tungsten,
   fluorescent, sunset, blue/mixed, night/street-light, low exposure, and
   backlighting. Compare feature off/on with identical source, target, face
   detections, enhancer, mask, codec, and frame range; measure luminance/chroma
   error, spatial shadow retention, frame-to-frame color delta, wrong FaceSets,
   output finiteness/order, FPS, RSS, and VRAM.
4. Manually review retained frames for partial shadows, colored night casts,
   highlight rolloff, low-resolution/motion-blurred faces, occlusions, and
   expression changes. Include GPEN 256 Pro, GPEN Realistic, UltraMax, and at
   least one additional restorer; confirm none lifts VERY_DARK faces or destroys
   target-conditioned lighting.
5. If any real scene changes tier within a shot, verify the tier-change
   hysteresis/EMA admits the transition without warm-neutral-blue oscillation.
   Only after this matrix passes should the default be reconsidered; until then
   keep `target_conditioned_appearance` opt-in.

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

## Requested Phase 11 handoff - adaptive enhancer orchestration

Status: **OPEN / INCOMPLETE**.

The exact implementation starting point is the new opt-in `Adaptive` selection
in `app/roop/core.py`, backed by `app/roop/adaptive_enhancer.py` and the bridge
`app/roop/processors/AdaptiveEnhancer.py`. `ProcessMgr` publishes pose, mask
occlusion, target appearance, and identity-detail-required signals before the
wrapper runs. The wrapper calls at most one existing candidate per face and
keeps the existing manual branch untouched. Candidate model code is lazy and
bounded by hardware/profile policy; `VERY_DARK`, extreme pose, low confidence,
heavy occlusion, and unstable tracks prefer omission.

Validation commands:

```text
app/env/Scripts/python.exe -m pytest app/tests/test_adaptive_enhancer.py app/tests/test_phase11_inventory.py app/tests/test_runtime_optimizer.py app/tests/test_settings_wiring.py -q
app/env/Scripts/python.exe app/tests/bench_adaptive_enhancer_video.py --help
app/env/Scripts/python.exe app/tests/bench_adaptive_enhancer_video.py --clip <locked-clip> --source <faceset> --enhancers Adaptive,GPEN 256 Pro,GPEN Realistic,UltraMax --adaptive-profile BALANCED
```

The video harness records runtime/FPS, process RSS, peak VRAM when available,
plate-relative output quality, temporal consistency, detected identity
similarity, and high-frequency/detail retention. Compare identical source,
target, detector, mask, swapper, codec, and frame range for each arm. The
existing `bench_phase11_enhancers.py` remains the isolated model benchmark.

Observed on 2026-09-01: the locked 4070 two-face smoke using
`double/d4.mp4`, RealSwap, RealityUX, TensorRT, Adaptive/BALANCED, and 12
workers produced 120/120 output frames, 120/120 swaps for each of two tracked
people, 240 face rows, and 0 wrong-FaceSet applications. The full video matrix
attempt entered the renderer but stalled after CUDA stream-906 and the
existing optional RealSwap secondary-network fallback warnings; it was stopped
and must be repeated with the stream/fallback condition resolved. No runtime,
quality, or memory value from that stalled attempt is accepted as a benchmark.

Required gates still open:

1. The final full regression suite is complete: **1641 passed, 1 skipped, 599
   subtests passed, 2 existing warnings**.
2. Review the retained-output 4070 Adaptive smoke for high-quality, moderate,
   dark, extreme-angle, occluded, blurred, and identity-detail frames.
3. Repeat the video matrix on the physical RTX 3060 with one context/global GPU
   guard, 1536 MB stabilization cap, RSS under 2.5 GB, and preserved look values
   (`blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`,
   `stabilize_enhancer_strength=0.6`).
4. Use locked V2 FaceSets and verify identity-detail retention is measured before
   and after each restoration family; confirm restorers do not erase it.
5. Manually inspect for flicker, artificial sharp points, lighting mismatch,
   hallucinated dark-scene detail, wrong FaceSets, and expression/occlusion
   failures before changing any default.

Exact next-phase starting point: run the full suite, then the 4070 retained
Adaptive video smoke and the `bench_adaptive_enhancer_video.py` matrix with a
locked V2 source archive; record its output JSON in the Phase 11 matrix. Keep
`selected_enhancer` on its current manual default until the 3060 and visual
gates above pass.

## Requested Phase 12 handoff - temporal compositing and natural blending

Status: **OPEN / INCOMPLETE**.

The exact implementation starting point is the existing final paste authority
`app/roop/procmgr_masking.py::MaskingMixin.paste_upscale`, now wired to
`app/roop/temporal_compositing.py::TemporalCompositeController`. The controller
is opt-in via `temporal_compositing`; when disabled, the previous linear ROI
blend remains the active path. When enabled, it receives the already-trimmed
model/ownership/occlusion matte, target-face pose/confidence, target appearance
tier, and frame index. It stabilizes a compact canonical matte per track,
attenuates semantic outer boundaries, preserves target spatial lighting through
low-frequency local adaptation, and protects generated high-frequency identity
detail. Existing enhancers and identity-detail ordering are unchanged.

Validation already completed:

- Full suite: **1648 passed, 1 skipped, 599 subtests passed, 2 warnings**.
- Component matrix: `app/tests/bench_temporal_compositing.py` completed
  frontal, lateral, profile, hair, glasses, hand occlusion, dark, and bright
  conditions; report at `app/output/phase12_temporal_compositing.json`.
- Matching 4070 real smoke: `double/d4.mp4`, 4 frames, OFF **0.47 FPS / 326.27
  ms / 4704 MB VRAM / 8.659 GB RSS**, ON **0.49 FPS / 318.28 ms / 4675 MB VRAM /
  8.544 GB RSS**, both return code 0, 2 face swaps, 0 wrong FaceSets. This is
  safety/cost evidence only because detector coverage was 1/4 frames.
- Poisson/gradient-domain candidate: available through OpenCV but not selected
  for production because it is CPU/full-patch work with no measured benefit
  over the bounded two-band ROI method.

Required gates still open:

1. Run the real Phase 12 matrix with locked annotated footage covering frontal,
   lateral, profile, hair, glasses, hand occlusion, dark, and bright scenes;
   record seam/edge quality, lighting transition, identity-detail retention,
   temporal shimmer, FPS, RSS, and VRAM for OFF/ON with identical inputs.
2. Repeat on the physical RTX 3060 laptop using the single-context/global GPU
   guard and 1536 MB stabilization cap. Preserve
   `blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`, and
   `stabilize_enhancer_strength=0.6`; do not transfer 4070 results or caches.
3. Review retained frames for jaw/cheek/forehead boundaries, hair, shadows,
   skin transitions, glasses and hands, bright/dark scenes, and hard/soft/hard
   edge oscillation. Include GPEN 256 Pro, GPEN Realistic, UltraMax, and
   identity-detail-enabled output to verify downstream restorers do not erase
   preserved details.
4. Compare the eight-condition output against the existing mask/ownership and
   target-conditioned appearance gates; if a real scene reveals over-feathering,
   target texture leakage, or an artificial sharp point, fix the same paste path
   and rerun the focused/full tests before changing defaults.

Exact next-phase starting point: run
`app/env/Scripts/python.exe app/tests/phase12_benchmark.py --target "RTX 3060"`
on the physical laptop with the locked annotated clip, then run the retained
4070 OFF/ON scene matrix and visual review. Keep `temporal_compositing=false`
as the default until both hardware and visual gates pass.

## Requested Phase 13 handoff - temporal artifact detection and selective correction

Status: **OPEN / INCOMPLETE**.

The exact implementation starting point is `app/roop/temporal_quality.py`,
called from the existing `ProcessMgr.process_face` path after alignment/source
selection and after enhancement/detail processing. It is opt-in through
`temporal_quality_control` and can be enabled for a controlled run with
`ROOP_TEMPORAL_QC=1`; `temporal_quality_logging` or `ROOP_TEMPORAL_QC_LOG=1`
adds records containing anomaly type, track, frame index, confidence, and
correction applied. Keep it disabled for legacy/default comparisons until the
gates below pass.

Validation already completed:

- `app/env/Scripts/python.exe -m pytest app/tests/test_temporal_quality.py -q`:
  **10 passed**.
- Combined Phase 13/11/12 focused integration set: **61 passed, 1 warning**.
- `app/tests/bench_temporal_quality.py --iterations 2000`: normal event rate
  **0.0**, one edge-triggered event per persistent anomaly case, approximately
  **0.654–0.678 ms** per inspect/record cycle.
- Full repository suite: **1658 passed, 1 skipped, 599 subtests passed, 2
  warnings**. The warnings are the existing Albumentations update notice and
  the existing NaN-to-uint8 enhancer guard fixture.
- Disabled controller remains a no-op; high-motion eye/jaw discontinuities are
  detected but do not reuse a prior transform and therefore preserve motion.

Required gates still open:

1. Run the full repository suite after Phase 13 and record the exact result;
   inspect for regressions in adaptive enhancer, V2 identity detail, target
   appearance, temporal compositing, source-bank selection, and mask ownership.
2. Run a locked real-video A/B with identical source, detections, swapper,
   enhancer, mask, codec, frame range, and workers. Record anomaly counts by
   type, correction counts, corrected-frame latency, FPS, RSS, VRAM, output
   finiteness/order, wrong-FaceSet count, identity similarity, detail retention,
   and temporal consistency. Review retained frames for drift, popping,
   brightness/color jumps, geometry jumps, hallucinated detail, detail loss,
   eye/jaw discontinuity, and flicker.
3. Repeat the relevant run on the physical RTX 3060 laptop with one context and
   the global GPU guard, 1536 MB stabilization cap, RSS under 2.5 GB, and the
   preserved look values (`blend_ratio=0.85`, `face_mask_blend=25`,
   `merger_sharpen=0.55`, `stabilize_enhancer_strength=0.6`). Do not transfer
   4070 caches or results.
4. Review low-resolution, motion-blurred, dark, occluded, lateral/profile, and
   high-motion expression frames. Confirm that correction is event-driven,
   real motion is retained, no generic blur is introduced, and GPEN 256 Pro,
   GPEN Realistic, UltraMax, RealityUX, and RealSwap do not erase preserved V2
   identity detail or reintroduce the detected artifact.
5. If real footage shows false positives or missed artifacts, adjust the same
   controller thresholds/correction gate, add a failing regression fixture,
   rerun the full suite and benchmark, then update this handoff. Do not make
   QC the default until these gates pass.

Exact next-phase starting point: run
`app/env/Scripts/python.exe -m pytest -q` from the repository root, then run
`app/env/Scripts/python.exe app/tests/bench_temporal_quality.py --iterations 2000`
and the existing locked video harness with
`--temporal-quality-control --temporal-quality-logging`. Start with
`temporal_quality_control=false` for the control arm, use the same Phase 11/12
scene matrix, and retain the existing RTX 4070/RTX 3060 hardware gates.

## Requested Phase 14 handoff — end-to-end GPU performance optimization

Phase 14 added an opt-in detailed profiler to the existing _prof hook and a
machine-readable comparator:

- app/roop/stage_profiler.py
- app/tests/phase14_bottleneck_report.py
- app/tests/test_stage_profiler.py
- app/tests/test_phase14_profile.py
- updates to app/roop/procmgr_runtime.py, ProcessMgr.py,
  procmgr_tracking.py, procmgr_masking.py, baseline_controlled.py, and
  bench_phase8_transfer.py

Run the active-path profile with:

app/env/Scripts/python.exe app/tests/baseline_controlled.py --tag phase14_detail --start 0 --end 60 --out app/output/phase14_profile --env ROOP_PROFILE_DETAIL=1

Use ROOP_PROFILE_DETAIL_SYNC=1 only for a short diagnostic arm. Compare
identical current-code arms with:

app/env/Scripts/python.exe app/tests/phase14_bottleneck_report.py --before <roi-off-json> --after <roi-on-json> --out <report-json>

The measured change is ROI-only blending (ROOP_BLEND_ROI_WARP=1, rollback
0). The 4070 paired detailed arms were 86.35 s / 2.75 FPS with ROI off and
82.07 s / 2.87 FPS with ROI on; blend wall time was 11.10 s versus 9.12 s
across the same 144 face calls, with zero wrong-FaceSet applications in both.
The result is promising but not yet a universal speed claim because the
standalone warp microbenchmark showed only ~0.9% median improvement and the
paired video result is one short arm. Synchronized 20-frame diagnostics
confirmed non-zero fence costs and must not be used as a throughput result.

### Phase 14 completion audit

| Requirement | Status | Evidence / remaining gap |
|---|---|---|
| IMPLEMENT | PASS | Existing stage hook, canonical matrix, opt-in profiler, measured ROI blend path, rollback flag |
| TEST | PASS | Phase 14 profiler/profile tests pass in the application environment; ROI output-equivalence test passes |
| BENCHMARK | PARTIAL | 4070 event-only/synchronized arms and transfer benchmark recorded; repeated arms, detail/expression/occlusion, and 3060 remain |
| REGRESSION TEST | PARTIAL | Focused tests pass; run and record the complete post-Phase-14 suite and longer preview/batch soak |
| DOCUMENT | PASS | ENV flags, progress, benchmark evidence, limitations, and this handoff |
| HANDOFF | PASS | Exact commands and unresolved gates below |

### Exact next-phase starting point

This is a resume point for **Phase 14**, not permission to begin a new phase:

1. Run app/env/Scripts/python.exe -m pytest -q after the final Phase 14
   edits and record the exact result.
2. Run at least three paired ROI-off/ROI-on arms on the locked 4070 workload
   with identical clips, models, workers, encoder, and environment; compare
   FPS, CPU/GPU/sync, sampled full-card peak/steady VRAM, RSS, output
   finiteness/order, wrong-FaceSet count, and retained visual quality.
3. Run the same detailed profile with the real V2 detail, expression, and
   occlusion options enabled so those rows are measured rather than left
   not_observed; use the phase-specific harnesses where each option has a
   locked fixture.
4. Repeat the accepted arm on the physical RTX 3060 with one GPU context,
   global GPU guard, 1536 MB stabilization cap, RSS below 2.5 GB, and
   blend_ratio=0.85, face_mask_blend=25, merger_sharpen=0.55,
   stabilize_enhancer_strength=0.6.
5. Review daylight/indoor/dark/occluded/lateral/profile and motion-blurred
   retained frames, including enhancer paths RealityUX, RealSwap, GPEN 256,
   GPEN Realistic, and UltraMax, before marking Phase 14 complete.

Do not change concurrency, TensorRT session ownership, provider transfer
policy, or enhancer defaults from the current evidence alone. Do not start
Phase 15 until these Phase 14 gaps are closed and the line-by-line checklist
in OPTIMIZATION_PROGRESS.md is updated from PARTIAL to PASS.

## Requested Phase 15 handoff — cross-hardware and regression validation

Status: **OPEN / INCOMPLETE**.

The audit implementation is intentionally observational:

- `app/roop/regression_audit.py` owns the backend/precision/workflow/lifecycle manifest and provider/cache classification.
- `app/tests/phase15_regression_audit.py` generates the report without building, deleting, or reusing TensorRT engines.
- `app/tests/test_phase15_regression_audit.py` protects unavailable-provider, enhancer-selector, coverage, and stale-cache behavior.

The current report is at `app/output/phase15_validation/audit.json`. It records the RTX 4070 host's TensorRT/CUDA/CPU providers as `available_not_validated`, ROCm/DirectML/CoreML as `unavailable`, 190 `not_run` workload/lifecycle rows, 19 stale-driver candidates, and 9 unscoped cache candidates. Do not delete those cache directories as part of this phase; review/rebuild them under the active driver-qualified namespace instead.

Validation already completed for this implementation:

- `app/env/Scripts/python.exe -m pytest -q app/tests/test_phase15_regression_audit.py` — **6 passed**.
- Combined focused provider/precision/hardware/inventory/Phase 14 set — **64 passed, 1 warning**.
- The CLI audit completed successfully and wrote the JSON report above.
- The retained-output integrity sweep over ten existing Phase 12/13 videos
  passed **10/10** (600 frames each; zero black, uniform, NaN, or duplicate
  frames). This catches encoder/output corruption only and does not replace
  the missing cross-hardware workload evidence.

Required gates before marking Phase 15 complete:

1. Run the full repository suite from the repository root and record its exact count after the final Phase 15 edits: `app/env/Scripts/python.exe -m pytest -q`.
2. On the RTX 4070, run fresh-process precision arms using the same locked clip/source/models/codec and record PASS/FAIL plus identity, detail, finiteness, FPS, VRAM, RSS, provider, and precision: `app/env/Scripts/python.exe app/tests/precision_matrix.py --clip G:/pinokio/roop-keep/double/d4.mp4 --source harjot --out app/output/phase15_validation/rtx4070_precision`. Use fresh processes for every provider/precision; never switch precision in a live ORT/TensorRT process.
3. Run the complete enhancer/profile video matrix on the same locked inputs with `bench_adaptive_enhancer_video.py`, including every label accepted by `compare_enhancers_video.VALID_ENHANCERS` and FAST, BALANCED, REALISTIC, and MAX QUALITY. Grade outputs independently with `app/tests/phase16_integrity.py`; a successful process is not sufficient.
4. Exercise image swap, multi-face swap, new and legacy faceset loading, faceset creation, preview, batch, and a long-video run. Record startup, shutdown, model release, memory release, repeated jobs, provider changes, precision changes, output ordering/finiteness, and anomaly logs. Re-run the audit report after each provider/device switch so cache namespaces are compared to the active runtime.
5. Repeat the supported provider rows on the physical RTX 3060 laptop with the single-context/global GPU guard, 1536 MB stabilization cap, RSS under 2.5 GB, and preserved look values: `blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`, `stabilize_enhancer_strength=0.6`. Do not transfer RTX 4070 engines or results. A missing physical device remains `pending`, not pass.
6. Run the AMD ROCm/DirectML and Apple CoreML subsets on their real supported hosts. If a provider is not applicable or cannot be installed, record `unavailable` with the host/runtime reason; do not mark it passed by CPU fallback. Verify CPU fallback separately.
7. Inspect regressions for stale engine reuse, provider/device mismatch, corrupted state, ONNX/TensorRT context races, memory leaks, preview/batch interference, identity/detail loss, brightness/color jumps, and temporal artifacts. Add a failing fixture for each defect found, then rerun the full suite and relevant benchmark.

Exact next-phase starting point: remain in Phase 15 and run the full suite (step 1), then the fresh-process RTX 4070 precision matrix (step 2). After all Phase 15 rows are evidenced and the checklist in `docs/OPTIMIZATION_PROGRESS.md` is PASS, the next phase may start with an output-integrity review using `app/tests/phase16_integrity.py` on retained artifacts. No Phase 16 completion or default change should be inferred from this audit-only implementation.
## Requested Phase 16 handoff — final production quality gate

Status: **OPEN / INCOMPLETE**. This is the final gate, so there is no later optimization phase to infer or start from an incomplete report.

Added without changing runtime processing behavior:

- `app/roop/final_quality_gate.py` — standardized 17-category clip manifest, four quality modes, five named component arms, every registered enhancer, metric schema, `.fsz` checks, and evidence-only winner selection.
- `app/tests/phase16_final_quality_gate.py` — report generator.
- `app/tests/test_phase16_final_quality_gate.py` — manifest, evidence, enhancer-preservation, winner, and legacy/V2 compatibility contracts.
- `docs/FINAL_ARCHITECTURE.md` — complete pipeline/resource-ownership architecture.

Generate the current report with:

`app/env/Scripts/python.exe app/tests/phase16_final_quality_gate.py --out app/output/phase16_validation/final_report.json`

Current report state: 17 categories, 0 ready clips, 425 benchmark rows, 0 complete runs, no selected winners, and `OPEN_INCOMPLETE`. The report also lists the Phase 1–15 audit status. The 425 rows are 17 clips × (FAST/BALANCED/REALISTIC/MAX QUALITY + RealityUX + RealSwap + GPEN 256 Pro + GPEN Realistic + UltraMax + 16 registered enhancer labels).

Final post-edit validation: **1,676 passed, 1 skipped, 599 subtests passed, 2 warnings**. The warnings are the existing Albumentations update notice and the existing NaN-to-uint8 enhancer-guard fixture warning. `git diff --check` passed.

Required final-production gates:

1. Supply real annotated clips for all 17 categories. Do not use a filename or process return code as quality evidence.
2. Run every row with fresh processes where provider or precision changes are involved. Record total time, FPS, frame latency, peak VRAM, CPU/GPU utilization, detection, swap, enhancer, blending, dropped, and fallback frames.
3. Record identity, temporal flicker, expression, eye state, pose, occlusion, boundaries, color, low-light realism, and identity-detail retention from retained output and annotations.
4. Independently inspect steep/inverted, lateral, fast motion, blinking/speaking, dark/night/mixed lighting, foreign-object, hands, glasses/hair, crossing faces, low resolution, and motion blur.
5. Run old and new `.fsz` archives through the compatibility check and existing FaceSet tests. Confirm V2 metadata/checksums and legacy PNG loading on representative production files.
6. Re-run all previous-phase tests and physical gates: RTX 4070/3060 constraints, CPU fallback, AMD ROCm/DirectML where supported, Apple CoreML where supported, every enhancer/mode, provider fallback, preview, batch, long video, startup/shutdown, release, repeated jobs, switching providers/GPUs/precision, and cache namespace validation.
7. Use `app/tests/phase16_integrity.py` on every retained output and investigate black, uniform, NaN, duplicate, unreadable, frame-count, duration, or audio regressions.
8. Populate evidence JSON and rerun the report. Only when every required row is complete and independently reviewed may the winner fields be used: fastest, best balanced, best quality, best night, best difficult angle, and best multi-face.

Exact continuation point: collect and register the 17 annotated clips, then execute the 425-row matrix through the existing render/benchmark harnesses and feed complete evidence into `phase16_final_quality_gate.py`. Until the report shows all rows complete and the progress checklist is PASS, the optimization program remains open and no defaults should be promoted.
