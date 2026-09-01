# Final validation matrix

Master record for the final end-to-end validation, repair and regression
campaign. One row per test. A row is PASS only when a run on this repository's
current code produced the evidence named in its own cell.

> **Two campaigns, two machines, one document.** Everything from
> `## Hardware targets` to `## Open / not attempted` is the **RTX 3060 Laptop**
> campaign, run on `C:\pinokio\`. The **RTX 4070** campaign is the section
> `## RTX 4070 campaign (2026-09-01)` at the end of this file, run on
> `G:\pinokio\`. Neither upgrades the other's `NOT TESTED` cells: the two
> machines are physically separate and approval rule 7 applies in both
> directions.

## How to read this document

Three rules govern every cell, and each was paid for by a defect in this
project's history.

1. **A green unit test is not a PASS for a subsystem.** The suite was 1018
   green through four separately-broken enhancers, and 1575 green through a
   regression that had disabled the swap on the shipped default path.
2. **Evidence must show the code path EXECUTED.** The swap audit counts
   INTENT, not outcome: it reported `swapped 100.0%` while four enhancers
   failed on 60 of 60 frames, and while a dedent bug left almost every frame
   unswapped. Every feature row below names its own execution proof (a
   `ROOP_PROFILE` stage with a call count, a resolved-state echo, or a metric
   with separation), never the audit alone.
3. **A number is quoted only with the window that produced it.** Resolution is
   measured per window, not assumed, and cross-window FPS is never compared.
   This campaign watched the same machine deliver 5.28 and 2.95 FPS on the
   *identical* configuration 24 minutes apart.

`NOT TESTED` is a truthful state. It is never upgraded to PASS by inference
from the other GPU.

## Hardware targets

| target | present in this campaign | notes |
|---|---|---|
| RTX 3060 Laptop 6GB, driver 616.56 | **YES** - every run below is on it | `C:\pinokio\`, 17 GB host RAM, 14P/20L CPU, CUDA 12.8, ORT 1.23.2 |
| RTX 4070 12GB | **NO - physically absent from this host** | lives on the other machine (`G:\pinokio\`). Every 4070 cell is `NOT TESTED`; prior 4070 values are cited as history, never as this campaign's evidence |

Under the campaign's own approval rule 7, a decision needing the other physical
machine is recorded and deferred, not guessed.

### Adaptive downgrades ACTIVE on this target

These change what several rows can possibly prove:

| stage | shipped behaviour on this card |
|---|---|
| provider | TensorRT disabled by the sub-7GB RSS policy; **CUDA/CPU used** |
| enhancer | `GPEN 256 Pro` -> **None** (sub-7GB RSS gate) |
| mask engine | RealityUX **degraded to XSeg only**; BiSeNet parser skipped |
| decode | NVDEC -> **CPU** |

**The shipped default on this card runs no enhancer at all.** Any enhancer row
here must force `ROOP_SMALL_CARD_ENHANCER=keep` and is then measuring a
non-default path for this device.

## Measurement resolution - established per window

| window | arms | result |
|---|---|---|
| null control, 600 frames (04:30) | 3 | 5.44 / 5.31 / 5.20 FPS -> **4.51% spread** |
| identity-detail set (04:45) | 4 | the two OFF arms alone spread **8.2%** |
| appearance window A (05:04) | 3 | OFF 5.28, GPU util ~63% |
| appearance window B (05:28) | 3 | OFF 2.95 / 3.08, GPU util ~37% - **the machine itself was ~45% slower** |
| decomposition window C (05:54) | 3 | util moved 37.6 -> 45.0 -> 55.0% across three CONSECUTIVE arms |
| peak host RSS, 3 null arms | 3 | 3.452 / 3.478 / 3.465 GB -> **0.75%** |
| **quality metrics, 3 null arms** | 3 | `own` spread **0.0007**, margin spread **0.0010** |

**An effect smaller than its own set's spread is reported as NOT RESOLVABLE.**
The quality floor is far tighter than the throughput floor, so identity effects
of ~0.001 are resolvable where a 5% FPS effect is not.

**THE THROUGHPUT CONCLUSION OF THIS CAMPAIGN.** Mean GPU utilization tracks FPS
across all 16 renders and varied 36-63% over the session, moving *within* a
single three-arm set. **Wall-clock A/B on this host was not viable today at any
effect size this campaign attempted.** Record GPU utilization beside every arm;
without it, a drifting host is indistinguishable from the setting under test.
Deterministic evidence - block geometry printed by the run, stage call counts,
`rows.csv` contents, quality metrics - stayed perfectly stable across the same
drift, and is what the conclusions below rest on.

## Status legend

`PASS` evidence produced here - `FAIL` reproduced defect - `FIXED` repaired and
retested here - `NOT TESTED` no run - `BLOCKED` needs absent hardware/media -
`N/R` not resolvable at this window's floor

---

## Phase 0 - repository and build integrity

| # | test | level | expected | actual | status | 3060 | 4070 |
|---|---|---|---|---|---|---|---|
| 0.1 | imports, config, model resolution | INTEGRATION | init reaches inference | CUDA/CPU up, buffalo_l resolved, det-size 512 | PASS | PASS | NOT TESTED |
| 0.2 | full unit suite | UNIT | no errors | **1666 tests, 0 errors, 0 failures, 1 skipped** (was 1664 with 3 errors) | **PASS** | PASS | NOT TESTED |
| 0.3 | repeated runs, one session | INTEGRATION | no stale state | 16 consecutive renders, all rc 0 | PASS | PASS | NOT TESTED |
| 0.4 | resource release between jobs | INTEGRATION | RSS bounded | peak RSS 3.34-3.57 GB across 16 renders, no monotonic growth | PASS | PASS | NOT TESTED |
| 0.5 | determinism of repeated identical runs | REGRESSION | same decisions | **pure swap geometry byte-identical across all 13 arms** (frame, person, bbox, faceset) | PASS | PASS | NOT TESTED |

## Phase 1 - inference backends

| # | test | level | actual | status | 3060 | 4070 |
|---|---|---|---|---|---|---|
| 1.1 | provider admission on sub-7GB | HARDWARE | `TensorRT disabled ... using CUDA/CPU` announced every run | PASS | PASS | NOT TESTED |
| 1.2 | TensorRT engine build/cache | HARDWARE | **not exercisable** - TRT not admitted on this card | BLOCKED | BLOCKED | NOT TESTED |
| 1.3 | FP16 / FP32 / mixed precision | HARDWARE | precision cannot reach an engine without TRT admission | BLOCKED | BLOCKED | NOT TESTED |
| 1.4 | CPU fallback | INTEGRATION | CPU EP present and used every run | PASS | PASS | NOT TESTED |
| 1.5 | pool sizing on 6GB | HARDWARE | `ROOP_TRT_POOL=0, ROOP_DETMASK_POOL=0` | PASS | PASS | NOT TESTED |

1.2/1.3 are BLOCKED, not FAIL: the refusal is correct policy for this VRAM
tier, and the campaign forbids degrading one GPU to suit the other.

## Phase 3 - face detection

| # | test | level | actual | status | 3060 | 4070 |
|---|---|---|---|---|---|---|
| 3.1 | **detector failures are visible** | UNIT | swallowed exceptions now warn once per signature; empty-list contract unchanged | **FIXED** | PASS | N/A (code) |
| 3.2 | detection stability across runs | REGRESSION | identical boxes in all 13 arms | PASS | PASS | NOT TESTED |
| 3.3 | hard angles / touching faces / NMS | VIDEO | **not run** - needs the dedicated angle+contact harnesses | NOT TESTED | - | NOT TESTED |

## Phase 5 / 13 - FaceSet creation, loading, identity detail

| # | test | level | actual | status | 3060 | 4070 |
|---|---|---|---|---|---|---|
| 5.1 | legacy V1 `.fsz` loads and swaps | INTEGRATION | `harjot`/`gargee` (5 root PNGs, no metadata) work in every arm | PASS | PASS | NOT TESTED |
| 5.2 | V2 build from V1, identities unchanged | INTEGRATION | `harjot_v2`/`gargee_v2`: version 2, 5/5 sources, `identity_detail_ok=5`, 5 distinct pose bins, quality 0.81-0.87 | PASS | PASS | NOT TESTED |
| 5.3 | **a valid V2 archive can still be inert** | UNIT | `migrate_legacy_fsz` yields version 2 with EMPTY `identity_details` | **FIXED** | PASS | N/A |
| 5.4 | identity detail executes on real footage | VIDEO | `identity_detail` stage **799 calls @ 9.55 ms** - first real V2 exercise anywhere | PASS | PASS | NOT TESTED |
| 5.5 | identity detail throughput cost | PERFORMANCE | -5.2%, inside the set's own 8.2% floor - and see the throughput caveat above, which applies to every FPS row in this document | **N/R** | N/R | NOT TESTED |
| 5.6 | identity detail memory cost | PERFORMANCE | +~100 MB peak RSS; VRAM within noise | PASS | PASS | NOT TESTED |
| 5.7 | identity detail component metrics | UNIT | retention **0.839016**, delta reduction **43.3257%** - identical to the 4070's recorded values; 225 vs 291 rest/s | PASS | PASS | history |
| 5.8 | identity detail identity effect | QUALITY | `own` 0.3976 -> 0.3971 (delta 0.0006, **inside** the 0.0007 floor) - no measurable gain, no harm | N/R | N/R | NOT TESTED |
| 5.9 | V2 vs V1 identity | QUALITY | V1 `own` 0.3579 / margin 0.4515; V2 `own` 0.3977 / margin 0.4313 - but **the two are graded against different reference vectors**, see below | **CONFOUNDED - NOT A RESULT** | - | NOT TESTED |
| 5.10 | corrupt/incomplete archive handling | UNIT | not run | NOT TESTED | - | NOT TESTED |
| 5.11 | moles/freckles survive on real footage | QUALITY | needs retained-output review; harness retains only `rows.csv` | NOT TESTED | - | NOT TESTED |

### 5.9 - WITHDRAWN AS A RESULT, kept as a lead and as a harness defect

This was drafted as the campaign's headline finding and it does not survive
its own check. Recording the whole thing, because the trap is reusable.

**The observation is real and reproducible.** Loading the same five source PNGs
as V1 versus V2 moves `own` from 0.3579 to 0.3977 and margin from 0.4515 to
0.4313 - ~57x and ~20x the measured quality floor, three arms per side, each
side internally tight to <0.001.

**The attribution is not.** `two_face_video.faceset_mean(fs)` builds the
grading reference by averaging `fs.faces[*].embedding`. `AverageEmbeddings()`
**mutates `faces[0].embedding` in place** to the mean of all five - and returns
early for `format_version >= 2`. So:

    V1 reference = mean([mean(e0..e4), e1, e2, e3, e4])
    V2 reference = mean([e0, e1, e2, e3, e4])

The two arms are graded against **different vectors**. The measured 0.040
therefore conflates a genuine runtime difference (what the swapper is handed)
with a change in the measuring stick (what it is scored against), and the split
between them is unknown.

**Two separate things are worth chasing, and they are not the same thing:**

1. *Runtime:* does V2 lose V1's multi-reference averaging with nothing
   compensating? `AverageEmbeddings` returning early for V2 is deliberate ("V2
   deliberately keeps each pose-specific face embedding intact"), on the premise
   that pose-aware selection compensates. That premise is untested here. The V2
   archives themselves are well-formed - 5 distinct pose bins, quality
   0.81-0.87, a quality-weighted identity embedding present - so this is about
   the runtime path, not the archive.
2. *Harness:* `faceset_mean` is not format-neutral, so **no V1-vs-V2 quality
   comparison made with this harness is valid**. That is a harness defect in its
   own right and it silently affects any future migration study.

**To settle it:** grade both arms against ONE fixed reference (e.g. the plain
mean of the five raw embeddings, computed before either code path mutates
anything), or read `embeddings_backup`, which V1 preserves precisely so the
original `faces[0]` can be recovered. Until then this row is a lead, not a
result, and must not be cited as a V2 regression.

Nothing in this campaign changed `AverageEmbeddings`, the V2 loader, or any
`.fsz` on disk.

## Phase 9/11 - temporal identity

| # | test | level | actual | status | 3060 | 4070 |
|---|---|---|---|---|---|---|
| 9.1 | Phase 11 low-pass blend cost | PERFORMANCE | **534.3 -> 902.4 calls/s, +68.9%**, counterbalanced 3 passes each way | PASS | PASS | history: +70.8% |
| 9.2 | Phase 11 mask path | PERFORMANCE | -1.7%, neutral | PASS | PASS | history: +0.5% |
| 9.3 | temporal identity quality on footage | QUALITY | not run | NOT TESTED | - | NOT TESTED |

Phase 11's 3060 benchmark cell, previously PARTIAL for want of a physical run,
is now closed at component level and reproduces the 4070 within 2 points.

## Phase 14 - lighting / colour / target-conditioned appearance

| # | test | level | actual | status | 3060 | 4070 |
|---|---|---|---|---|---|---|
| 14.1 | appearance component, 11 lighting scenes | UNIT | colour delta reduction **82.2298%**, identical to the 4070's value; **34.11 ms/call** vs 23.36 there | PASS | PASS | history |
| 14.2 | appearance executes on real footage | VIDEO | `lighting` stage **920 -> 1666 calls**, 9.06s -> 18.14s | PASS | PASS | NOT TESTED |
| 14.3 | appearance throughput cost | PERFORMANCE | three windows gave **-11.6%, -18.1%, +68%**. Not measurable on this host today | **NOT RESOLVABLE** | N/R | NOT TESTED |
| 14.4 | appearance changes stabilizer geometry | PERFORMANCE | **6-wide -> 3-wide, read from the run's own log in every ON arm** - deterministic, not inferred from timing | **CONFIRMED** | PASS | NOT TESTED |
| 14.5 | appearance identity effect | QUALITY | `own` 0.3977 -> 0.4194, margin 0.4313 -> 0.4212. ~30x and ~10x the floor | **measured cost** | - | NOT TESTED |
| 14.6 | dark/night real-footage arms | QUALITY | not run | NOT TESTED | - | NOT TESTED |

### 14.3 / 14.4 - what is confirmed, and what is withdrawn

**CONFIRMED, from the run's own log rather than from a stopwatch.**
`_stab_warmup_frames` takes the SLOWEST active stabilizer. The appearance EMA's
`alpha = 0.30` needs 13 warm-up frames against the baseline's 7; the 3x
minimum-block floor turns that into 39-frame blocks; the host-RAM chunk budget
(~332 MB, itself derived from free RAM) then fits only three:

    OFF: 6 workers, 6 blocks x 16f, warm-up 7f
    ON : warm-up 13f needs 39f blocks; budget fits 3, so 3-wide instead of 6

This appears in **every** ON arm and is deterministic. Two corollaries, both
independently verified:

* `faces_seen` 936 -> 843 is **not lost detections** - it is fewer re-processed
  warm-up frames. Proof: `rows.csv` holds **683 rows over 477 frames in every
  arm**, pure swap geometry is byte-identical, and forcing a *narrower* budget
  (`ROOP_STAB_CHUNK_MB=127`) pushed `faces_seen` UP to 991, exactly as more
  blocks predicts. `faces_seen` is a block-count proxy, not a detection count.
* The remedy the log already names (`ROOP_STAB_CHUNK_MB`) trades host RAM, the
  scarce resource on this 16 GB machine.

**WITHDRAWN: the wall-clock cost.** Three windows disagree in direction:

| window | OFF | ON | reading |
|---|---|---|---|
| A (05:04) | 5.28 | 4.67 / 4.66 | -11.6% |
| B (05:28, counterbalanced) | 2.95 / 3.08 | 2.47 | -18.1% |
| C (05:54, decomposition) | 3.05 / 2.70 | **5.13** | **+68%** |

Mean GPU utilization tracks FPS almost exactly across all nine arms (36-63%),
and it moved *within* set C - 37.6%, 45.0%, 55.0% on three consecutive arms. So
throughput here is dominated by how much GPU the host granted at that moment,
not by the setting under test. `nvidia-smi` showed no active throttle after the
fact; this is a laptop under sustained load over a ~100-minute session.

**The honest result is that the cost is not measurable on this host today.**
The mechanism above is real and worth acting on; the number is not. Anyone
quoting -11.6% or -18.1% from this campaign would be quoting the machine's mood.
Settling it needs many more counterbalanced pairs inside a single stable window,
with GPU utilization recorded beside every arm - or the 4070.

## Cross-cutting regression results

| # | check | actual | status |
|---|---|---|---|
| R.1 | swap geometry across all 13 arms | byte-identical (frame, person, bbox, faceset) | PASS |
| R.2 | wrong-faceset applications | **0 in every arm** | PASS |
| R.3 | output completeness | 600/600 frames every arm, rc 0 | PASS |
| R.4 | run-to-run pixel determinism | `touched`/`own`/`other` drift slightly; geometry does not. GPU float non-determinism, quantified as the quality floor above | DOCUMENTED |
| R.5 | `ROOP_BLEND_ROI_WARP` vs legacy | **bit-identical** - asserted for the first time in this environment | PASS |

## Defects found and fixed in this campaign

| # | defect | why it mattered | evidence | status |
|---|---|---|---|---|
| D.1 | `face_util.get_all_faces` swallowed EVERY detector exception silently | the exact cause of two prior investigations: yoloface returning 0 faces at "329 fps", and a bench grading 0/600 frames and blaming the footage. A run finishes rc 0, valid output, audit 100% | 6 contracts in `test_detector_failure_visibility.py`; return contract deliberately unchanged | **FIXED** |
| D.2 | 35 harnesses hard-coded `G:/pinokio/...` | the other machine's drive. On this target every default was unreachable, and folder sweeps silently swept nothing | all migrated to `fixtures.clip()/clip_dir()`; 8/8 fixture paths resolve here; guarded by `test_fixture_paths.py` | **FIXED** |
| D.3 | `test_phase14_profile` imported `pytest`, absent from `app/env` | the module never ran, including the ONLY check that `ROOP_BLEND_ROI_WARP` is bit-identical. A collection error reads as an environment complaint, not an untested optimization | converted to unittest; 3 tests pass | **FIXED** |
| D.4 | `test_nvdec_reader` x2 carried as "pre-existing environment errors" | not environmental. The tests skipped the ffmpeg resolution the app performs; ffmpeg lives under `PINOKIO_HOME/bin`, not on PATH | resolves like the app, skips honestly if truly absent; 4/4 pass | **FIXED** |
| D.5 | `baseline_controlled` could not express the Phase 10 arm | the documented 3060 appearance arm would have silently measured the baseline twice | `--target-appearance-mode`; 4 of 6 new contracts verified to FAIL on the unpatched harness | **FIXED** |
| D.6 | no arm recorded the feature state the CHILD resolved | an inert arm is indistinguishable from a null result - the Phase 15 controller failure mode | `feature_*` readback from the child's own echo | **FIXED** |
| D.7 | `build_faceset_v2` needed to refuse detectionless V2 | see 5.3 | `test_build_faceset_v2.py` | **FIXED** |
| D.8 | `import fixtures` placed inside `if HERE not in sys.path:` | **introduced by D.2's fix.** Python puts a script's own dir on `sys.path`, so the branch never runs and every migrated harness raised NameError. Syntax checks pass happily | guard asserts module-level REACHABILITY, not presence; harness re-verified by an actual run | **FIXED** |
| D.9 | `two_face_video.faceset_mean` is **not format-neutral** | it averages `faces[*].embedding`, and `AverageEmbeddings()` mutates `faces[0]` in place for V1 but returns early for V2. So a V1 arm and a V2 arm are scored against different reference vectors, and any V1-vs-V2 quality comparison through this harness is invalid | see 5.9; caught by checking the metric's definition before publishing the finding it produced | **FOUND, NOT FIXED** |

## Process errors made during this campaign

Recorded because they cost real arms and the lesson generalizes.

1. **The source tree was edited while benchmarks were rendering.** Three
   appearance arms died at ~4 seconds each. A measurement campaign must freeze
   the tree; documentation edits are safe, `app/` edits are not.
2. **A static check was trusted to prove a runtime property.** D.8 passed a
   syntax check and an ordering check, and still broke every harness. The
   guard was rewritten to assert reachability, and the harness was then proven
   by running it.

## Open / not attempted

| item | why |
|---|---|
| Everything on the RTX 4070 | different physical machine; approval rule 7 |
| TensorRT, FP16, FP32, mixed precision | correctly refused on a 6GB card |
| Enhancer matrix (GPEN 256 Pro / Realistic / UltraMax / RealityUX / RealSwap) at default | the sub-7GB gate strips the enhancer; needs `ROOP_SMALL_CARD_ENHANCER=keep` and is then a non-default path |
| Foreign-object occlusion, interacting faces, expression/blink, night scenes, compound scenarios A-H | not run; fixtures exist (`double/`, `expression/`, `single/`, `final/`, `3d model/`) |
| Retained-output visual review | the controlled harness keeps only `rows.csv` |
| Long-run soak / leak test | not run |
| V2 identity regression (5.9) on a second clip and on the 4070 | needed before generalizing |

---

# RTX 4070 campaign (2026-09-01)

Run on the MAIN device. Separate physical machine from the 3060 campaign above;
nothing here upgrades a 3060 cell and nothing there upgrades a 4070 cell.

## Target and stack

| item | value |
|---|---|
| GPU / driver | RTX 4070 12GB, driver 616.56, compute 8.9 |
| runtime | TensorRT 10.9.0.34, ORT 1.23.2, torch 2.7.0+cu128, cv2 4.9.0 |
| host | 24P/32L CPU, 31.7 GB RAM, Windows 11, `G:\pinokio\` |
| live config | `realswap` / `RealityUX` / `UltraMax` / `tensorrt` / `hevc_nvenc` / 12 threads / `retinaface_r50` at 512 |
| notable | `target_conditioned_appearance: true` is LIVE here. It is `False` in `settings.py` and in `roop/globals.py`, so on this machine it is not an experiment -- it is the shipped path |
| fixture | `double/d4.mp4` 1280x720, sources `harjot,gargee`, capture frame 4930 (pinned) |

**No adaptive downgrade is active on this card.** Unlike the 3060, TensorRT is
admitted, the enhancer is not stripped, RealityUX runs both engines and NVDEC is
used, so every enhancer row below is measuring the default path.

## The instrument this campaign added

**A pixel difference is not evidence that a feature ran**, and until now nothing
here knew where that line sat. `tests/measure_output_noise_floor.py` renders one
unchanged configuration N times and reports the spread. On this target, on the
production stack, `double/d4.mp4` frames 0..60, three independent pairs:

| pair | mean abs diff | max | frames differing |
|---|---:|---:|---:|
| run 0 vs 1 | 0.7135/255 | 22 | 60 of 60 |
| run 0 vs 2 | 0.7160/255 | 20 | 60 of 60 |
| run 1 vs 2 | 0.7131/255 | 21 | 60 of 60 |

**Floor: mean 0.714/255, max 22/255**, and the three pairs agree to 0.4%, which
makes it an instrument rather than an anecdote. It is measured through the same
lossy encode as every A/B, so the two are directly comparable; it is not a claim
about the pipeline's internal divergence, which is smaller.

It is not caused by the obvious suspects:

| suspect | test | result |
|---|---|---|
| worker scheduling / RAM-derived block geometry | threads 12 -> 1 | 0.7469 -- unchanged |
| TensorRT tactic selection | tensorrt -> cuda | 0.8921 -- present on both providers |
| set/dict iteration order | `PYTHONHASHSEED=0` | 0.7804 -- unchanged |

Frame 0 already differs at one worker, so it is not temporal-state divergence
either. Detected bounding boxes are identical across runs while the identity
cosines in `rows.csv` are not, so the divergence begins at or after the swap.
What remains is non-deterministic GPU reduction order. This corroborates the
3060 campaign's R.4 independently, and quantifies it.

## Phase 0 - repository and build integrity (4070)

| # | test | evidence | 4070 |
|---|---|---|---|
| 0.1 | imports, config, models, providers | torch CUDA True, ORT `['Tensorrt','CUDA','CPU']`, buffalo_l resolved, det-size 512, TRT engines built and cached | PASS |
| 0.2 | full unit suite | **1666 -> 1691 tests, 0 failures, 0 errors, 1 skipped** (25 added by this campaign) | PASS |
| 0.3 | repeated runs in one session | 30+ consecutive renders, every one rc 0 | PASS |
| 0.4 | host RSS bounded over a long render | 5,979-frame `double/d3.mp4` render, 289 samples: min 10.48, **peak 15.26 GB**, quarter means **14.68 / 14.78 / 14.79 / 13.07** -- flat then falling, no monotonic growth | PASS |
| 0.5 | determinism of repeated identical runs | detection geometry identical; pixels differ at the floor above | PASS (documented) |

## Phase 15 - enhancer regression, all 14 selectable paths (4070)

`tests/enhancer_regression_sweep.py`, 30 frames, 60 swapped faces per arm,
production stack. **The acceptance test is a `ROOP_PROFILE` `enhance` stage call
count equal to the swapped-face count** -- never a return code, never the swap
audit, and never a pixel delta. The audit counts faces it was HANDED, and a
pixel delta below the floor above means nothing.

| enhancer | enhance calls | ms/face | fps | verdict |
|---|---:|---:|---:|---|
| None | 0 / 60 | - | 1.87 | PASS (no stage, correctly) |
| GFPGAN | 60 / 60 | 48.9 | 1.15 | PASS |
| Codeformer | 60 / 60 | 43.9 | 1.13 | PASS |
| Codeformer (fp16) | 60 / 60 | 68.4 | 1.12 | PASS |
| DMDNet | 60 / 60 | 157.4 | 0.93 | PASS |
| GPEN 256 | 60 / 60 | 5.2 | 1.77 | PASS |
| GPEN 256 Pro | 60 / 60 | 14.8 | 1.24 | PASS |
| GPEN Realistic | 60 / 60 | 38.0 | 1.24 | PASS |
| GPEN | 60 / 60 | 38.5 | 1.26 | PASS |
| GPEN 1024 | 60 / 60 | 142.2 | 0.55 | PASS |
| GPEN 2048 | 60 / 60 | 370.8 | 0.14 | PASS |
| UltraMax | 60 / 60 | 132.0 | 0.98 | PASS |
| Restoreformer++ | 60 / 60 | 39.8 | 1.23 | PASS |
| **Adaptive** | 60 / 60 | **0.0** | **1.95** | **NO-OP - see D4070.3** |

Zero wrong-faceset applications in every arm. **DMDNet works on this card**, so
the inherited "DMDNet is broken" is 3060-specific.

The fps column is a 30-frame window and includes model init; it ranks the arms
against each other inside one window and must not be quoted as throughput.
`ms/face` is the stage's own per-call figure and is the comparable number.

## Phase 8 - single-image face swap (4070)

"Image face swap works" is an acceptance criterion neither campaign had a run
behind: every end-to-end harness here renders VIDEO, so the still path
(`roop.core.live_swap` -- no tracker, no temporal engines, no stabilizer
geometry, no encoder) was covered only by unit tests.
`tests/image_swap_smoke.py` closes it. Production stack, `single/s1.mp4`,
`harjot`:

| frame | face-region delta | identity to source, before -> after |
|---:|---:|---|
| 200 | 6.44/255 | 0.0596 -> **0.6665** |
| 600 | 6.44/255 | 0.0437 -> **0.6712** |
| 1000 | 6.05/255 | 0.0591 -> **0.6941** |

Graded on identity rather than pixels on purpose: two renders of one unchanged
configuration differ on every frame here, so "the output changed" proves
nothing. An embedding moving from ~0.05 to ~0.67 toward the source cannot be
produced by a filter, a colour transfer or an enhancer.

**The harness carries a `--control` arm that skips the swap and must FAIL.** Run
here: region delta 0.00/255, identity 0.0596 -> 0.0596, both assertions fired.
Without that, a rubber stamp and a real check are indistinguishable.

## Phase 11/19 - two interacting swapped faces, and long-run stability (4070)

One full-length render of `double/d3.mp4` -- **5,979 frames, 15,684 detected
faces, 1109 s at 5.39 fps** -- production stack (RealSwap / RealityUX /
UltraMax / TensorRT / hevc_nvenc / 12 workers), sources `harjot,gargee`. This is
the interacting-faces workload and the long-run soak in one pass.

### What the pipeline decided

| audit reason | faces | share |
|---|---:|---:|
| swapped (identity lock) | 13,585 | 89.3% |
| swapped (per-frame match) | 510 | 3.4% |
| **track matched but has no source** | **1,516** | **10.0%** |
| discarded: the swap put the face somewhere it was not | 73 | 0.5% |

Of the 14,095 faces actually swapped, **4,663 (33.1%) had INTERPOLATED
landmarks** -- nobody detected them; box and keypoints were filled in between
neighbours. Interpolated faces bypass the identity gates, so a third of this
clip's swaps are not gated the way the other two thirds are.

### Wrong faceset

| person | frames | swapped | wrong faceset | re-measured as the other person |
|---|---:|---:|---:|---|
| harjot (box 0) | 5,975 | 5,622 (94.1%) | **13 of 1,340 attributed swaps** | 193 of 2,765 gradable |
| gargee (box 1) | 5,774 | 5,267 (91.2%) | **4 of 1,612 attributed swaps** | 37 of 1,880 gradable |

**17 wrong-faceset applications of 2,952 attributable swaps (0.58%)**, from the
pipeline's own decision log rather than from re-detecting the output.

**This is a NEW baseline, not a comparison.** The 2026-08-23 duo audit that
recorded 10 wrong-faceset applications on "d3" ran `duo/d3.mp4` (854x480,
3,597 frames). This is `double/d3.mp4` (5,979 frames), and `duo/` does not exist
on this machine at all. Two different clips share the filename -- the same trap
that once put the wrong 854x480 `d4` into a run that otherwise looked valid --
so 17 must not be read as a regression from 10.

The "re-measured as the other person" column is reported and deliberately NOT
treated as a defect count: it re-detects the finished output, which on exactly
the contact frames where a two-faceset bug would live hits the shared
recognition-crop problem and reports each person as the other regardless of what
the swap did. The pipeline's own decision is the sound instrument here.

### Long-run stability

Host RSS across 289 samples of that render: min 10.48 GB, **peak 15.26 GB**,
quarter means **14.68 / 14.78 / 14.79 / 13.07 GB**. Flat and then falling, on a
31.7 GB host. No monotonic growth over 5,979 frames, and the process exited
cleanly (rc 0) with a complete output file.

### What remains open here

`track matched but has no source` at 10.0% is the largest single refusal class
on this clip and is untouched by this campaign. It is the coverage gap the
project already characterises as intake rather than gating -- a track whose
best frames never entered the source bank binds to nothing, and the fix lives
upstream in capture, not in the assignment threshold.

## Defects found and fixed in the 4070 campaign

| # | defect | why it mattered | evidence | status |
|---|---|---|---|---|
| D4070.1 | **the end-to-end harness did not render the user's config** | `two_face_video.py` -- which `baseline_controlled.py` and the whole Phase/Gate campaign run through -- inherited `angle_bench.init_pipeline`'s "state everything explicitly" semantics. 28 keys diverged from `config.yaml`. Never set by any harness: `target_conditioned_appearance` False vs **True**, `detail_transfer_strength` 0.0 vs **0.4**, `color_match_after_enhance` False vs **True**, `codeformer_fidelity` 0.5 vs 0.55, `parser_regions` None vs the five configured regions. A/B arms stay valid (both sides equally off); absolute values and any quality grading were never production | the harness echo printed `target_appearance=False` on a run that passed no flag; after the fix, `True`. `tests/test_bench_config_parity.py` verified to FAIL on the pre-fix state, on 28 keys | **FIXED** |
| D4070.2 | **identity detail restored nothing and said nothing** | `--identity-detail-strength 0.35` on the shipped V1 facesets: no `identity_detail` stage in the profile at all, rc 0, audit 100%, and a pixel delta (0.766) inside the noise floor (0.714). All three instruments read clean | one-shot report naming the cause (V1 archive / V2 without residual / no FaceSet), verified on hardware: one message on a 20-frame render. 6 tests | **FIXED** |
| D4070.3 | **the adaptive enhancer restored nothing on 60 of 60 faces** | and presented as the FASTEST arm of the sweep (1.95 fps against 1.87 for `None`), with 60 enhance calls -- the wrapper is called and returns immediately when it selects `none`. The recorded 4070 Adaptive smoke reports 120/120 frames and 0 wrong-FaceSets and never noticed | reason counts and the quality band are now reported: `{'high-quality-face-minimal-enhancement': 60}` over **min 0.7665 / p50 0.7994 / max 0.8188** against BALANCED's 0.68 cut. **No threshold changed** -- see below. 9 tests | **FIXED (visibility)** |
| D4070.4 | `faceset_mean` was not format-neutral (the 3060 campaign's D.9) | `AverageEmbeddings()` mutates `faces[0]` in place for V1 and returns early for V2, so V1 and V2 arms were graded against different reference vectors: 0.427 max abs apart on unit-scale embeddings | reconstructs face 0 from `embeddings_backup`; the guard asserts both formats equal the plain mean, and a second test anchors the divergence so the fix cannot be mistaken for a no-op. Verified to fail on the old definition | **FIXED** |
| D4070.5 | adaptive fallback printed per face | two lines per face on a failing render: 120,000 on a 60,000-frame two-face clip. The project's own bounded-reporting rule, not applied here | once per distinct cause, counted, totalled at Release | **FIXED** |
| D4070.6 | an absent `quality` entered the adaptive band as 0.0 | would make a high-scoring population read as low-scoring, inverting the conclusion the band exists to support | found by a test written for the reporting contract, not by inspection | **FIXED** |

### D4070.3 in full - why no threshold was re-tuned

The selector is not broken. On `double/d6.mp4` the same BALANCED profile chose
`gpen_realistic` for 18 of 73 faces over a quality band of **0.42-0.47**, the
other 55 refused by `extreme-angle-geometry-first` -- and that gate reads the
real solved pose, not the crude keypoint fallback, because `ProcessMgr`
publishes `_adaptive_yaw`/`_adaptive_pitch`.

So the finding is bounded and specific: **on good footage every profile refuses,
because the population sits far above every cut** (d4's 0.7665 minimum is above
even MAX QUALITY's 0.76), and on hard footage it engages. That is the stated
policy working.

What this campaign measured is **the distribution the gate reads**. It did NOT
measure whether restoration would have improved those faces, and that second
half is exactly what four gate changes in this project's history lacked before
being implemented and reverted. Re-tuning without it would be the fifth.

## Cross-cutting notes (4070)

| # | item | result |
|---|---|---|
| X.1 | target-conditioned appearance executes | mean 1.24/255 against a 0.71 floor, max 71 against 22. Real, not noise |
| X.2 | temporal compositing executes | mean 1.82/255, max 110. Real |
| X.3 | temporal quality control executes | mean 0.92/255 (inside the floor) but max **171** against 22 -- localized corrections, which is what an event-driven controller should look like |
| X.4 | identity detail at 0.35 | mean 0.766 against a 0.714 floor -- **not resolvable**, and the profile confirmed it never ran. See D4070.2 |
| X.5 | `lighting` stage cost | 180 calls per 60 faces = 3 distinct sites (appearance analysis, pre-enhance colour transfer, post-enhance colour match) at 23.8 ms/call, 7.6% of thread time. Not redundant work; three real stages sharing one label |

## Open on the 4070

| item | why |
|---|---|
| 12 settings with no UI | `identity_detail_strength`, `temporal_compositing_*` (7) and `temporal_quality_*` (4) are in `settings.py` and `api.py` but in no React control, so they are reachable only by editing `config.yaml` |
| whether Adaptive's cuts should move | needs a quality comparison on real footage, not a distribution |
| foreign-object occlusion, expression/blink, night scenes, compound scenarios A-H | not run in this campaign |
| retained-output visual review | the harness keeps `rows.csv` and the video, but no human review was performed |
| everything on the RTX 3060 | different physical machine; approval rule 7 |
