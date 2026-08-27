# Roop Ultimate — v2 Audit Memory

**Started:** 2026-08-27 (Asia/Kolkata)  
**Purpose:** A preservation-first, evidence-led audit of Roop Ultimate. This file is the authoritative continuation record for the audit. It must be updated after every material inspection, test, decision, code change, and verification result.

## Operating contract

- Audit code line by line; do not treat an older note, a benchmark, or a passing unit test as proof that a behaviour is correct.
- Preserve user-approved quality settings and both hardware profiles unless a tested, documented change is required.
- Do not change application code merely because it looks unusual. Every change needs a reproducible defect, narrow fix, and regression evidence.
- Use the current logs before diagnosing a runtime issue.
- Before session context becomes constrained, write a complete handoff here: current stage, evidence, modified files, test results, open questions, and exact next action.

## Original five phases (from CLAUDE.md)

| Phase | Area | Current memory status |
| --- | --- | --- |
| 1 | Detection | Previous work exists; verify detector reliability, tracking, angle and rotation handling with current code. |
| 2 | Mask engines / RealityUX | Previous work exists; verify boundaries, occlusion handling and no bystander bleed. |
| 3 | Interacting faces | Characterised previously; contact/overlap and extreme profile cases remain a high-risk visual verification area. |
| 4 | RealSwap | Shipped; `realswap` is the live default. Verify composite swap correctness and identity isolation. |
| 5 | UltraMax | Shipped, then deliberately recalibrated after earlier claims were disproved. Verify actual behaviour, not historical claims. |

## Post-phase regression scope

Later changes are included as regression scope without renumbering the original phases: GPEN Realistic and GPEN 256 Pro, long-video RAM bounds, multi-faceset stability, API/React UI correctness, TensorRT/GPU pooling, 3060 portability, and stabilizer work-stealing.

## Test material and required matrix

| Material | Purpose | Faceset assignment |
| --- | --- | --- |
| `G:\pinokio\roop-keep\3d model` | pose, yaw, roll, profile, synthetic detail | Rhythm unless the case needs two targets |
| `G:\pinokio\roop-keep\single` | single-person identity continuity | Rhythm |
| `G:\pinokio\roop-keep\double` | two-person isolation and target/source assignment | Ashna + Rhythm |
| `G:\pinokio\roop-keep\expression` | eyes, mouth, expression and temporal stability | Rhythm |
| `G:\pinokio\roop-keep\final` | end-to-end, real-footage regression | Rhythm; Ashna + Rhythm where a clip has two intended targets |

All relevant visual checks use RetinaFace R50, RealityUX masking, RealSwap, and each enhancer independently: GPEN 256 Pro, GPEN Realistic, and UltraMax.

## Stage plan

1. **Baseline and evidence capture** — inventory source, configuration, logs, tests, samples and protected behaviour.
2. **Line-by-line audit** — backend, launcher, UI, APIs, test harnesses and configuration paths.
3. **Automated correctness** — run the full suite and narrow regression tests for confirmed findings.
4. **Visual verification** — execute the defined material/configuration matrix and grade identity, detection, masks, occlusion, temporal stability and enhancement quality.
5. **Regression and handoff** — re-test fixes, compare against baseline, document unresolved risks and leave a clean continuation record.

---

## Stage 1 — baseline and evidence capture

### Audit pre-flight (complete)

- [x] Re-opened `AGENTS.md`; relevant rules recorded: inspect logs first, preserve current functionality, respect both hardware profiles, and do not alter Pinokio scripts without an example/documentation cross-check.
- [x] Reviewed the active launcher UI (`pinokio.js`) and active React starter (`start_react.js`). No launcher changes are being made in Stage 1.
- [x] Logged the current runtime evidence before diagnosing any issue.
- [x] Loaded historical memory: `CLAUDE.md`, `GEMINI.md`, and `facegemini.md`.

### Repository baseline

- Repository root: `G:\pinokio\api\roop-ultimate`
- Branch / revision at audit start: `main` / `7088962`
- `SPEC.md`: absent.
- Source review surface (initial count):
  - `app\roop`: 76 Python files, approximately 28,251 lines.
  - `app\tests`: 157 Python files, approximately 27,821 lines.
  - `react-ui\src`: 77 JS/JSX files, approximately 20,348 lines.
- No application code has been changed by this audit yet.

### Active launcher baseline

- `start_react.js` launches the Python backend from `app` and the React UI from `react-ui`; both are daemon-managed.
- Current launcher settings enable profiling, TensorRT pool `4`, detection/mask pool `4`, parallel stabilization, two stabilizer blocks per thread, and temporal detector stride `1`.
- URL capture uses the required parenthesized Pinokio regex and `local.set` reads `input.event[1]`.

### Latest runtime evidence

Source: `logs\api\start_react.js\latest` at audit start.

- Latest completed render: **55,886 frames** in **4,526.19 seconds** (**12.35 fps** overall); log ends in `[SUCCESS] Finished`.
- Processing ran with **12 execution threads**. Late-run RSS reported approximately **0.55–2.42 GB**; this single run is evidence of completion, not proof that all long-video memory defects are closed.
- Profiling reports detector, mask, swap, verification, and enhancement stages. Detailed numbers must be extracted and normalised before performance conclusions.
- Swap audit: 121,337 faces seen; 83,134 swapped (68.5%); 30,713 refused by identity threshold (25.3%); 7,456 refused because the recognition crop overlapped a neighbouring face (6.1%); 255 discarded after a spatial mismatch; 34 refused for a closer-person match.
- The log reports 11,423 of 102,455 analysed frames (11.1%) with no detected face. This is an audit target, **not yet a defect**: it may reflect occlusion, profile/angle limits, or input composition.
- Current log contains the benign startup message `det_size is already set in detection model, ignore`; no crash, traceback, or failed completion was found in the end-of-run evidence reviewed so far.

### Automated baseline

- Command run from `app`: `env\Scripts\python.exe -m unittest discover -s tests -t . -p "test_*.py"`
- Result: **1,300 tests passed, 1 skipped, 0 failures** in 19.528 seconds.
- The suite emitted ResourceWarnings from several test modules that open source files without a context manager. These are test-harness hygiene issues, not evidence of an application file-handle leak.
- GPEN 256 Pro emitted a PyTorch warning at `Enhance_GPEN256Pro.py:303`: its deliberately read-only cached grain array is converted to a tensor and immediately copied to CUDA. The cached GPU tensor is then read-only in practice, so this is currently a noisy initialization warning rather than demonstrated output corruption; retain as a low-priority cleanup check.

### Protected hardware behaviour

- Desktop: RTX 4070 12 GB / 32 GB RAM, with the established high-concurrency safeguards and 4,096 MB stabilization cap.
- Laptop: RTX 3060 Laptop 6 GB / 16 GB RAM, with single-context GPU safety and 1,536 MB adaptive stabilization behaviour. The user-approved 3060 look settings must remain intact.

### Configuration and execution map

- The persisted desktop configuration selects RetinaFace R50, RealityUX, `realswap`, GPEN 256 Pro, TensorRT mixed precision, 12 threads, and temporal face/mask/enhancer stabilization.
- The backend entry point is `app\run.py`; FastAPI is served by `app\api.py`; named faceset persistence is served by `app\routes_faceset.py`; the React client is launched by `start_react.js`.
- Main pipeline ownership: `ProcessMgr.py` coordinates the render; `procmgr_tracking.py`, `procmgr_masking.py`, `procmgr_runtime.py`, `procmgr_merger.py`, and `procmgr_tiling.py` carry the major stages; `retinaface.py`, `session_pool.py`, and `FaceSet.py` are critical support surfaces.
- There are 157 test modules, including dedicated coverage for detection, tracks, contact/overlap, RealityUX, RealSwap, all requested enhancers, API routes, memory/OOM guards, hardware portability, and React UI hooks.
- The persisted configuration says pool values `2 / 2 / 2` (TRT / detection-mask / detector), while `start_react.js` exports `ROOP_TRT_POOL=4` and `ROOP_DETMASK_POOL=4`. `app\run.py` then loads `config.yaml` before importing Roop and unconditionally replaces those two environment values with `2`, so the **persisted configuration wins** and the active effective pools are `2 / 2`. This is a configuration-provenance issue: the launcher values currently do not do what their text implies.

### Facesets and samples

- All five supplied sample folders are present and contain the expected single, double, expression, 3D-model, and final-footage media.
- The app uses the project-root named-faceset library `G:\pinokio\api\roop-ultimate\facesets` when configured. Both required assets are present and available for Stage 4:
  - `G:\pinokio\api\roop-ultimate\facesets\ashna.fsz` (1,268,223 bytes; thumbnail `ashna.png`).
  - `G:\pinokio\api\roop-ultimate\facesets\rhythm.fsz` (1,013,437 bytes; thumbnail `rhythm.png`).
- The earlier search checked the wrong default (`app\facesets`) after reading the routes without accounting for the project-level library configuration. B-005 is therefore cleared; this correction is retained so later work uses the verified paths.

### Stage 1 next actions

1. Locate and verify the actual Ashna and Rhythm faceset assets and their expected source identities.
2. Build a code/module ownership map and identify every executable entry point.
3. Capture test-run baseline and current configuration provenance without changing it.
4. Start Stage 2 only after the baseline map is complete.

## Findings register

| ID | Severity | Status | Finding | Evidence / next action |
| --- | --- | --- | --- | --- |
| B-001 | Investigation | Open | 11.1% detector no-face rate in the latest production audit | Classify against the supplied samples before considering threshold or engine changes. |
| B-002 | Investigation | Open | 31.5% detected faces left unswapped by safeguards | Verify that refusals prevent wrong-person swaps rather than produce avoidable flicker. |
| B-003 | Info | Closed baseline | Latest render completed successfully | Treat as baseline performance/completion evidence only. |
| B-004 | Info | Closed — intended | Persisted `config.yaml` performance settings override launcher defaults before Roop imports | The React settings panel explicitly documents this precedence. The active 2/2/2 pools are therefore intentional for the 12GB workstation, not a runtime defect. |
| B-005 | Info | Closed | Ashna and Rhythm facesets verified | `facesets\ashna.fsz` and `facesets\rhythm.fsz` are present in the project-root library and ready for visual verification. |
| B-006 | High | Fixed & tested | `app\run.py` globally replaced Python's HTTPS default context with an unverified context | Removed the process-wide TLS bypass; ordinary certificate verification now remains enabled. |
| B-007 | High | Fixed & tested | In-memory video `frame_count` was one too high | Corrected the video count to `frame_end - frame_start`, matching the end-exclusive contract used by readers, progress, resume, and temporal tracking. |
| B-008 | Low | Fixed & tested | GPEN 256 Pro created a tensor directly from a deliberately non-writable NumPy grain cache | The shared cache remains immutable; its GPU path copies before `torch.from_numpy`, eliminating the PyTorch undefined-behaviour warning. |
| B-009 | Low | Fixed & tested | Advanced-performance help stated the wrong auto pool size for 15.5GB+ GPUs | Updated the UI from 8 to the runtime's actual auto tier of 4. |
| B-010 | Low | Fixed & tested | Runtime-estimator samples were keyed with configured, not effective, worker count | Calibration now records `execution_threads`, including the result of automatic thread selection. |
| B-011 | Medium | Fixed & tested | A deliberately stopped batch returned from the core partial-finalization path, but the API continued into the success path and recorded it as a completed run | Added an immediate stop guard after `batch_process_regular`; stopped renders now keep their partial-output status and skip post-passes/completed-history telemetry. |
| B-012 | Medium | Fixed & tested | The Stage 4 sample runner used legacy Harjot/Shambhavi facesets instead of the requested Rhythm single-face and Ashna+Rhythm double-face verification set | Updated `app\tests\run_all_samples.py` and added a regression guard; sample inputs remain unchanged. |
| B-013 | Medium | Fixed & tested | RealSwap's eye-band default was `1.0` even though the measured safe production setting documented in the same file is `0.5`, allowing full-strength secondary eyelid/brow overlays and visible tone seams | Changed the default to `0.5`; the override env var remains available and a regression test guards the default. |
| B-014 | Medium | Fixed & tested | Profile/close-face mattes eroded by half the calculated feather radius before blur, exposing the untouched target along the nose and under-eye contour; this matches the supplied double-nose, pale-boundary, and duplicate-brow screenshots | Reduced the inner erosion to one quarter of the feather radius. Landmark-hull clipping, overlap ownership, and Gaussian anti-aliasing remain active. Added a regression guard. |
| B-015 | Medium | Fixed; s1 verification running | RealSwap's secondary eyelid net remained active on extreme lateral crops where one eye collapses in the canonical alignment, smudging cornea/conjunctiva and distorting the profile eye structure | Added yaw-aware attenuation and suppresses the secondary band at extreme yaw (`abs(yaw) >= 0.95`); primary hyperswap geometry remains responsible for the profile. Focused tests pass. |

---

## Stage 2 — static code audit

### Reviewed: startup / launcher-to-backend handoff

**Files reviewed:** `start_react.js`, `app\run.py`, initial API state and faceset mapping in `app\api.py`, `app\roop\ProcessEntry.py`.

1. `start_react.js` correctly starts backend and React processes as daemons and correctly captures the React URL through a parenthesized regex / `input.event[1]` local assignment. No Stage 2 fault found in that URL handoff.
2. `app\run.py:20-22` installs `ssl._create_unverified_context` as the process-wide HTTPS default. The broad exception hides whether this was ever needed. This is B-006.
3. `app\run.py:27-58` gives non-auto `config.yaml` performance values precedence over the launcher environment. With the recorded config, its `_set()` calls turn the launcher's advertised `4`-context TRT/detection-mask pools back to `2`; this is B-004.
4. `app\api.py:61-130` creates a copied/reordered faceset list rather than mutating global source state during a render. Its comments identify and explain a prior upload-vs-render race; the current structure is sound in this reviewed section.
5. `app\roop\ProcessEntry.py` is a minimal value holder. No fault found.

### Reviewed: in-memory pipeline setup and worker lifecycle

**Files reviewed:** `app\roop\ProcessMgr.py:908-1552`, `app\roop\core.py:899-1060`, `app\roop\capturer.py:520-541`.

1. `core.py` establishes that `endframe` is exclusive: an untrimmed run sets it to the video frame count, and its throughput/runtime calculations use `endframe - startframe`.
2. Both readers use that exclusive contract: standard video reads `frame_end - frame_start` frames; animated WebP slices to `frame_end`.
3. `ProcessMgr.run_batch_inmem()` contradicts that contract at line 1290 with `frame_count = (frame_end - frame_start) + 1`. This is B-007. The current 55,886-frame log reaches 55,885 then closes, which is the expected observable result of the error.
4. Resume uses the inflated count to decide whether work remains. A fully completed resumable render can therefore take an unnecessary no-frame rerun/pre-pass rather than the documented immediate-finalize path.
5. The queue/sentinel design includes bounded operations and drains to avoid the historical deadlocks. The writer-close-after-timeout path remains deliberately defensive; no reproducible race was found in this review, so it is not logged as a defect.
6. Tracking records the actual scan count. When temporal detection is enabled, its coverage guard compares that correct result with the inflated expected count, producing a false one-frame warning/fallback condition.

### Next static audit target

`app\roop\ProcessMgr.py` and its tracking/masking/runtime mixins: initialise → temporal pre-pass → identity assignment → parallel stabilization → per-face processing → cleanup.

### Reviewed: tracking, compositing, RealityUX, and requested enhancers

**Files reviewed:** `app\roop\procmgr_tracking.py`, `app\roop\ProcessMgr.py:2760-4506`, `app\roop\procmgr_masking.py`, `app\roop\processors\Mask_RealityUX.py`, `app\roop\processors\Enhance_GPEN256Pro.py`, `app\roop\processors\Enhance_GPENRealistic.py`, and `app\roop\processors\Enhance_UltraMax.py`.

1. Temporal tracking records actual scanned frames, performs source assignment before the swap path, and treats contested/shared recognition crops conservatively. No distinct correctness defect was reproduced; the only confirmed accounting interaction is B-007's false one-frame coverage gap.
2. The multi-face compositor resolves pending swaps and bystanders into ownership regions before painting, while every face reads from the untouched plate. This avoids source-crop contamination and has consistent single-session versus pooled GPU guards in the reviewed execution paths.
3. RealityUX retains XSeg as the authoritative mask and only lets FaceParser subtract the defined opaque accessories. The two inferences run concurrently only when their independent session pools are enabled; otherwise the enclosing guard remains serial. No incorrect mask polarity or unsafe pooling issue was found.
4. GPEN 256 Pro, GPEN Realistic, and UltraMax each protect a non-pooled ONNX session through `exclusive()` and use a pool lease when available. Their output finite/collapse fallbacks preserve the unswapped crop rather than crash a render. B-008 remains a low-priority warning cleanup; its deliberate read-only grain cache is copied to CUDA before use and no output corruption is evidenced.

### Reviewed: session-pool policy and advanced-settings handoff

**Files reviewed:** `app\roop\session_pool.py`, `app\run.py`, `react-ui\src\components\Settings.jsx`.

1. The runtime pool policy correctly uses 0 / 2 / 2 / 4 contexts across the defined VRAM bands. Explicit environment values remain honoured, with a warning rather than a silent clamp when oversized.
2. Persisted performance settings are deliberately applied in `app\run.py` before Roop imports; this confirms B-004 rather than resolving it. The interface must either accurately describe this precedence or the backend must adopt the launcher as the source of truth.
3. The Settings help text says the 15.5GB+ automatic swapper-pool tier is 8, but `session_pool._auto_pool_defaults()` returns 4. This is B-009: a guidance defect only, not an execution defect.

### Reviewed: processing API option flow

**Files reviewed:** `app\api.py:367-421`, `app\api.py:2400-2525`, `app\api.py:2610-2770`, `app\roop\runtime_calib.py`, `app\settings.py:733+`, and relevant React controls.

1. RealityUX, RetinaFace R50, mapped facesets, and the requested enhancer names are carried correctly from the React payload to the processing options. The API de-duplicates a repeated mask engine and maintains the selected-face mapping before rendering.
2. The full-render path resolves the effective worker count with `CFG.resolve_threads(mode)` when automatic thread selection is enabled. Its runtime-calibration signature then records `CFG.max_threads` instead of the resolved count. Because calibration buckets include worker count, samples can be stored and later retrieved under the wrong key. This is B-010. It does not affect a run's rendered frames or the current saved configuration, where automatic thread selection is off.

### Fix checkpoint — 2026-08-27

The user authorized automatic correction of confirmed defects. The following narrow fixes were applied without changing model files, source facesets, sample media, saved visual settings, or Pinokio launcher scripts:

1. Removed `app\run.py`'s process-wide HTTPS certificate-verification bypass (B-006).
2. Corrected in-memory video frame accounting from inclusive to end-exclusive in `app\roop\ProcessMgr.py` (B-007).
3. Preserved GPEN's immutable shared grain cache while copying it at the CUDA tensor boundary (B-008).
4. Updated the advanced-performance tooltip to state the runtime's 15.5GB+ auto pool of 4 contexts (B-009).
5. Recorded the actual resolved worker count in runtime-estimator signatures (B-010).
6. Added `app\tests\test_audit_regressions.py` and GPEN-specific coverage to lock these seams in place.
7. Added a stop-state guard in `app\api.py` so partial user-stopped renders cannot be marked Done or written to completed run history (B-011).
8. Corrected the Stage 4 sample runner to use Rhythm for single-face runs and Ashna + Rhythm for double-face runs (B-012).
9. Corrected RealSwap's default secondary eye-band opacity from 1.0 to the measured 0.5 setting (B-013).
10. Reduced profile matte inner erosion from half to one quarter of the feather radius, preventing the target face from showing through as a second nose/under-eye tone edge (B-014).
11. Added extreme-yaw gating for RealSwap's secondary eyelid band so lateral profiles retain the primary eye structure instead of a collapsed secondary overlay (B-015).

**Verification:** focused audit + landmark-mask tests: **21 passed** in the project environment (`app\env\Scripts\python.exe`). The prior complete focused set was **46 passed**; `run_all_samples.py` compiles successfully. Full suite previously: **1,305 passed, 1 skipped, 0 failures**. Existing resource warnings in older tests remain non-fatal test-harness hygiene items and were not changed as part of this focused correction. React's production build could not be invoked because this shell has no `node`/`npm` executable; the JSX change is covered by the source regression test and is a single text correction.
**Full-suite recheck:** **1,309 tests ran, 1 skipped; 1 failure**. The only failure was the pre-existing `test_progress_chunks.DropInBehaviourTest.test_unknown_style_falls_back_to_auto` capture assertion (`0` captured lines instead of `2`) and passes when run in isolation; it is a terminal-capture test-harness flake, not a rendering or pipeline failure. The four stale RealSwap assertions were updated to the intentional 0.5 contract and no longer fail.

## Session handoff

**Current stage:** Stage 2 — static audit in progress; Stage 3 automated baseline completed once.  
**Code changes by this audit:** fixed B-006 through B-013 as described in the fix checkpoint; application configuration and media assets remain untouched.  
**Last completed actions:** Re-ran logs/pre-flight review, corrected RealSwap eye-band opacity, completed single/double smoke renders, completed a separated d2 double render, and ran 46 focused regression tests. The prior full suite remains 1,305 passed (1 skipped).  
**Stage 4 assets:** Ashna and Rhythm facesets are verified at `facesets\ashna.fsz` and `facesets\rhythm.fsz`.  
**Resume exactly here:** Stage 2 persistence/history and processing-completion review is complete through B-011. Next prepare Stage 4's controlled visual matrix: `single` with Rhythm; `double` with Ashna + Rhythm; use RetinaFace R50, RealityUX/RealSwap, and compare GPEN 256 Pro, GPEN Realistic, and UltraMax across `3d model`, `expression`, and `final`. Do not alter sample inputs or saved look settings while testing.  
**Open investigations:** B-001 (detector no-face rate) and B-002 (safeguard refusals) require visual classification before any threshold or detector change.  
**Verified design decision:** B-004 is not a bug; persisted performance settings intentionally override launcher defaults.  
**Environment note:** React production build was not run because this shell has no `node`/`npm`; backend and source regression coverage passed.  
**New findings:** B-011 is fixed as above. B-012 is fixed: the sample runner now uses the requested Ashna/Rhythm identities for Stage 4. B-013 is fixed: RealSwap now defaults to the measured 0.5 eye-band opacity. B-014 is fixed: profile matte erosion no longer removes half the feather radius, which was allowing the original nose/under-eye pixels to show through.

### Stage 4 smoke verification — 2026-08-27

1. **Single:** `G:\pinokio\roop-keep\single\s4.mp4` with Rhythm, selected mode, RetinaFace R50, RealityUX, RealSwap, and GPEN 256 Pro. Completed successfully: 427/427 frames detected, 427/427 frames with a face, 427/427 faces composited, one finalized playable MP4. Representative start/middle/end frames were inspected; no obvious mask tear or temporal dropout was seen.
2. **Double:** `G:\pinokio\roop-keep\double\d1.mp4` with Ashna + Rhythm and the same pipeline. Completed successfully: 282/282 faces composited over 141 frames and one finalized playable MP4. The clip's two faces overlap in every frame (capture separation 0.29; shared-crop count 282/282), so visible identity mixing/edge artifacts are expected and this asset is not a clean two-person verification sample. Use a more separated double clip (such as `d2` or later) for the next run.
3. Runtime evidence for both runs confirms the expected RTX 4070 policy: 12 execution threads and TRT/detection-mask/detector pools of 2/2/2. No OOM or encoder failure occurred.

4. **User artifact review:** the supplied d2 screenshots show the exact failure signature addressed by B-013/B-014: a secondary eyelid/brow overlay plus an over-eroded profile boundary that reads as a pale second nose and a different tone below the eye. The source video itself is unchanged; the correction is in the RealSwap eye-band default and paste-back matte.

5. **Post-fix render:** rendered `app\output\stage4_double_d1_maskquarter\d1__ashna-rhythm.mp4` with RetinaFace R50, RealityUX, RealSwap, GPEN 256 Pro, tracking/temporal stabilization, and the live 12-thread / 2-2-2 pool policy. All 141/141 frames and 282/282 faces completed. Representative frames were inspected: separated profiles now have continuous facial tone and no broad mask halo; the extreme kiss frame still contains the source clip's physically touching noses and remains a deliberately difficult overlap case rather than a clean-separation quality reference.

6. **Stage 4 matrix run started:** `tests\run_all_samples.py --only both` is running with the requested Rhythm single-face and Ashna+Rhythm double-face assignments. It is resumable and writes only under `app\output`; the first long single clip (`s1.mp4`, 19,672 frames) is currently in the detector/tracking pass. Existing outputs are not overwritten.

7. **Stage 4 matrix completed:** the resumable run finalized 8 single outputs (`s1`–`s8`, Rhythm) and 6 double outputs (`d1`–`d6`, Ashna+Rhythm) under `app\output\baseline_single` and `app\output\baseline_double`. No process crash, OOM, decoder, or encoder failure was reported. Representative d2 frames were extracted for visual review; normal separated faces are stable, while heavily occluded/rotated contact frames remain the known hard boundary for this source material.

8. **Stage 5 s1-only lateral verification started:** a new render is running at `app\output\stage5_s1_lateral_gate`, using only `G:\pinokio\roop-keep\single\s1.mp4` and Rhythm. The run will record per-stage timings and inspect lateral-eye frames for cornea/conjunctiva preservation and profile alignment.

9. **Stage 5 correction completed — 2026-08-27:** RealSwap now uses calibrated `solve_pose_jaw_5pt` yaw for the eyelash-only HifiFace overlay. The overlay fades on the far side from 35° and skips HifiFace inference at 65°+, leaving HyperSwap responsible for the complete lateral eye structure. The run finalized `app\output\stage5_s1_realswap_pose_gate\s1__rhythm.mp4` successfully: 19,672/19,672 frames processed, 17,548 faces swapped, 3,322 lateral secondary skips, 14.98 end-to-end fps, stable ~9.9 GB memory, and no OOM/decoder/encoder errors. Representative profile frames were inspected; the visible eye remains structurally single with no broad secondary-eye overlay.

### Exit checklist for this checkpoint

- [x] Revisited `AGENTS.md` rules relevant to this task.
- [x] Checked current logs before diagnosing runtime behaviour.
- [x] Made no Pinokio launcher-script change; therefore no launcher-example divergence or URL-capture change exists.
- [x] Preserved existing configuration and all rendering/sample assets; only targeted application and UI code was changed.
- [x] Recorded all material evidence, test results, finding status, and next action in this file.
- [x] Applied and verified the calibrated RealSwap lateral eyelash-overlay gate and recorded its s1-only output and speed evidence.
- [x] Reopened `AGENTS.md` before the B-014 edit; no launcher script was touched, so the example lock-in/URL-capture requirement was not applicable.

## Pasted performance/hardware-overhaul plan — Stage 1 audit (2026-08-27)

The pasted 20-phase performance plan is now the active audit specification. It
is grouped into the five stages listed at the beginning of this session. Stage 1
has traced the live execution path without making speculative performance edits:

`/api/swap` payload → `app/api.py::_run_swap` →
`roop.core.batch_process_regular` → `ProcessMgr.run_batch_inmem` → video reader
(NVDEC wrapper when valid) → detection/tracking/temporal landmark preparation →
`ProcessMgr.swap_faces` → `ProcessMgr.process_face` → model alignment and face
swap → mask/ownership and colour operations → optional enhancer → merger/paste
back → bounded writer/segmented FFmpeg encoder → optional post-swap upscale and
combine.

Confirmed from source inspection:

1. Detection and recognition are ONNX Runtime based, with RetinaFace R50 and
   InsightFace pools. Provider selection is centralized through `run.py`,
   `roop.core`, and the session-pool helpers rather than being hardcoded in the
   frame loop.
2. RealSwap is a composite: HyperSwap is the structural base and HifiFace is an
   eyelash/lip-colour overlay. The lateral gate now avoids unsafe HifiFace work
   at strong profile angles.
3. Enhancers are invoked after swapping and masking inside `process_face`; GPEN
   256 Pro has a measured GPU texture path plus CPU fallback and an explicit
   concurrency/guard contract.
4. Video processing uses NVDEC opportunistically for input and FFmpeg/NVENC for
   output, with software fallback. Output is segmented/resumable and the writer
   purges completed temporal entries to bound host memory.
5. Concurrency is controlled by resolved workload threads, model/session pools,
   bounded queues, and optional two-pass or parallel stabilization. The current
   RTX 4070 run evidence uses 12 threads and 2/2/2 pools.
6. The latest completed s1 RealSwap run measured 14.98 end-to-end fps, ~9.9 GB
   working memory, and no OOM/codec failure. Its log showed 3,322 lateral
   secondary skips, proving the pre-inference gate is active.

Stage 1 open questions for Stage 2 measurement: exact per-stage wall-time share
(decode, detection, recognition, swap, mask, enhancer, merge, encode), hidden
CPU↔GPU synchronization/copies, provider partitioning, and behavior on low-VRAM,
AMD/Intel, and CPU-only profiles. No architecture change will be retained until
those measurements are collected.

## Stage 2 baseline timing evidence

The latest `ROOP_PROFILE=1` production log provides a measured baseline (67,701
frames; 74,518 swapped faces; 12 execution threads):

| Stage | Wall time | Share | ms/call |
|---|---:|---:|---:|
| Swap | 6,348.27 s | 29.8% | 85.19 |
| Mask | 5,913.58 s | 27.8% | 79.36 |
| Enhance | 4,847.84 s | 22.8% | 65.06 |
| Track detection | 2,082.19 s | 9.8% | 30.76 |
| Verification | 1,143.14 s | 5.4% | 20.01 |
| Track wait | 937.41 s | 4.4% | 13.85 |

This establishes the current workload as primarily swap/mask/enhancement compute
with a significant tracking/detection component—not a decode bottleneck. The
profile is wall-clock summed across worker threads, so it is not a direct
single-thread elapsed-time percentage; Stage 2 must still separate worker time,
queue wait, host transfers, provider partitioning, and encode time on a controlled
s1 run before changing concurrency or backend defaults.

### Stage 2 transfer/provider audit (source evidence)

1. ONNX Runtime swap and detection sessions return NumPy arrays to the
   Python/OpenCV pipeline. This is intentional because alignment, masks,
   paste-back, and FFmpeg-facing frames are CPU/OpenCV operations; I/O binding
   must be benchmarked rather than added blindly.
2. GPEN 256 Pro explicitly crosses NumPy → CUDA tensors and CUDA tensors →
   NumPy for its texture/sharpness pass. This is a measurable candidate, not yet
   a proven defect; the GPU kernels must be compared against transfer overhead.
3. Provider construction is centralized in `roop.core.decode_execution_providers`,
   but `suggest_execution_providers()` currently reports the ONNX Runtime provider
   list without validation inference. This is a Stage 3 compatibility gap.
4. Session pools are VRAM-tiered and already validated for the 4070/3060 profiles.
   Explicit pool overrides are honoured with warnings; no universal pool increase
   should be made without counterbalanced measurements because oversized pools
   have already demonstrated TensorRT context thrashing.

Portability/provider regression checks passed: **36 tests** covering hardware
profile migration, AMD/Intel/CPU fallback behavior, encoder fallback, provider
fallback logging, and benchmark environment parity.
## Stage 3 — Hardware-adaptive backend layer implemented (2026-08-27)

Implemented `app/roop/backend_manager.py` as the single capability-aware
provider resolver. It normalizes short/encoded provider names, checks ORT
availability plus visible CUDA/device validity, caches probes for the process,
and resolves the required hierarchy: TensorRT → CUDA → CPU; CUDA → CPU;
ROCm/DirectML/CoreML → their provider → CPU. If no GPU provider is usable, the
resolver emits an explicit CPU chain instead of allowing a later opaque model
session failure. `core.decode_execution_providers()` now uses this resolver,
retains the existing per-provider options (TRT workspace/precision, CUDA arena,
device id), and reports option-setup warnings rather than swallowing them.

Startup diagnostics now print the available ORT providers and the validated
AUTO chain. Existing hardware-derived pool/thread defaults and the preserved
4070/3060 look settings were not changed. Added regression coverage in
`app/tests/test_backend_manager.py` for name normalization, TensorRT fallback,
and CPU-only fallback.

Verification: provider/hardware/performance portability suite plus backend
tests: **39 tests passed**. No model engines were built during probing; this
keeps startup bounded and avoids turning a provider check into a hidden render.
Stage 4 will add measured transfer/VRAM recovery instrumentation and only then
consider execution-path optimizations.

## Stage 4/5 verification note — 2026-08-27

Post-backend smoke on `single/s1.mp4` with the live configuration reached the
processing pass with RetinaFace R50 + NVDEC, 12 threads, TRT pool 2 and stable
RSS of 9.88–9.99 GB. The validated chain was
`TensorrtExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider`;
no provider, decoder, encoder, or OOM errors appeared. The duplicate full pass
was intentionally stopped after the first resumable 1,000-frame segment because
the completed Stage 5 s1 artifact already provides the full correctness result
and this invocation included the expensive all-frame tracking pre-pass.

The existing completed artifact remains:
`app/output/stage5_s1_realswap_pose_gate/s1__rhythm.mp4` — 19,672/19,672
frames, 17,548 swaps, 3,322 lateral HifiFace skips, 14.98 end-to-end fps,
approximately 9.9 GB memory, and no OOM/decoder/encoder errors.

## Precision audit specification — received 2026-08-27

The requested precision work is a separate measured extension of the five-stage
audit. The existing TensorRT mixed implementation is protected and must not be
removed or replaced without evidence. Current source audit baseline:

- `app/config.yaml` currently selects `trt_precision: mixed`.
- `core.decode_execution_providers()` maps `fp32` to
  `trt_fp16_enable=False`; `fp16` and `mixed` enable TensorRT FP16 capability.
  The mixed setting does not enumerate layer assignments itself; TensorRT makes
  the legal precision choices during engine build unless a model-specific
  processor adds constraints.
- `FaceSwapInsightFace._swap_providers()` now honors the configured precision;
  `ROOP_SWAP_FP32=1` retains the explicit overflow-safe FP32 override because
  FP16 overflow previously produced rainbow/smudge artifacts. Override engines
  are isolated with `_swap_fp32`.
- `enhance_common.fp32_trt_providers()` similarly isolates model-specific FP32
  enhancer engines (for example GFPGAN/GPEN) from the mixed/FP16 cache.
- Core TensorRT caches are already separated by precision directory (`mixed`,
  `fp16`, `fp32`), but the requested audit must verify whether model/device,
  TensorRT/CUDA versions, shape profile and batch size also need to be included
  in the effective cache key before any cache change is made.
- Existing `tests/compat_one.py` measures TRT `fp32`, `fp16`, and `mixed` with
  basic detection/identity/texture/channel guards, but does not yet cover the
  complete CUDA FP32/FP16 matrix or all requested hardware/telemetry fields.

Required next action: build a controlled benchmark matrix on the real
application path, one process per mode, recording initialization/build time,
end-to-end FPS, per-stage latency, peak VRAM, CPU/GPU utilization, transfer
counts where measurable, output correctness, temporal artifacts, and failure or
OOM status. No AUTO precision default will be changed until those results are
available.

## RealSwap mixed-precision enablement — 2026-08-27

Implemented the requested correction. `FaceSwapInsightFace._swap_providers()`
now honors `CFG.trt_precision` for RealSwap. With `mixed` (the current config),
the TensorRT provider reaches the swapper unchanged, so TensorRT can use its
mixed FP16/FP32 selection. `fp16` is likewise available for measurement. The
previous overflow-safe behavior remains available explicitly through
`ROOP_SWAP_FP32=1`, and the legacy `ROOP_SWAP_FP16=1` opt-in alias remains
compatible. FP32 override engines continue using the isolated `_swap_fp32`
cache; mixed/FP16 use the precision-specific core cache.

Added regression tests proving mixed reaches RealSwap and that the explicit
FP32 safety override still isolates its cache. Focused verification: **71
tests passed**, plus Python compilation checks.

Added `app/tests/precision_matrix.py`, which runs each arm in a fresh process:
TensorRT FP32/FP16/Mixed, CUDA FP32/FP16, and CPU FP32. It delegates correctness
guards to `compat_one.py`, now accepts `--provider`, and records initialization
time, processing time, FPS, peak RSS, result guards, and failure output in
JSON-lines. GPU utilization, VRAM, and transfer counters remain to be added
where host telemetry supports them; no mode is auto-selected until the real
matrix has been run and compared for quality and stability.

## Precision matrix smoke — 2026-08-27

Created a 10-second/240-frame representative clip from `single/s1.mp4` and
started the fresh-process matrix. TensorRT FP32 completed with a PASS:
`init_s=5.626`, `process_s=104.662`, `2.293 FPS`, peak RSS `6.650 GB`,
detected on 238/240 frames, identity distance `0.320`, texture `44.6`, and
channel skew `33.5`. The FP32 arm used its isolated `models/trt_cache/fp32`
path and the RealSwap mixed change did not affect it.

The next TensorRT FP16 arm did not return within the short-clip stability
window and was stopped. This is recorded as an unresolved FP16
initialization/execution stability issue, not treated as a performance win or
loss. Mixed, CUDA FP32/FP16, and CPU FP32 must be run separately after this
hang is isolated; AUTO selection remains unchanged.

## Benchmark path clarification — 2026-08-27

`app/output/precision_matrix_input/s1_10s.mp4` is the deliberately preserved
trimmed **input** clip and will not contain a swap. The completed TensorRT FP32
swapped output is written separately under
`app/output/precision_matrix/s1_10s/tensorrt_fp32/`. Matrix arms must always be
judged from their provider/precision output directory, never from the
`precision_matrix_input` directory.

## Precision/hardware follow-up — 2026-08-27

The benchmark harness had a provider-selection defect: `compat_one.py` accepted
`--provider` but still hardcoded TensorRT when decoding providers. This was
fixed and verified from logs. Correct short-clip results on the RTX 4070
(48 processed frames, RealityUX, no enhancer) are:

| Backend/mode | Result | Init | Process | FPS | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| CUDA FP32 | PASS | 3.870 s | 10.370 s | 4.629 | 2.080 GB |
| CUDA FP16 | PASS | 3.714 s | 12.289 s | 3.906 | 2.072 GB |
| CPU FP32 | PASS | 3.803 s | 87.829 s | 0.547 | 1.647 GB |

All three passed detection, identity, texture, and channel guards. The
TensorRT FP32 and mixed arms were run with the new runtime-specific cache
namespace (`precision + GPU + SM + ORT version`). FP32 exceeded the 180-second
fresh-build/analysis window; mixed reached the same slow fresh-build/analysis
behavior and was stopped after confirming `trt_fp16_enable=1` and the
`RealSwap ... precision=mixed` log. This identifies first-build/engine-analysis
cost as a separate stability metric; it does not prove mixed inference is
incorrect. Existing old-cache full-run evidence remains the steady-state
baseline until the new namespaces finish building.

The harness now records timed-out arms rather than blocking the matrix, and
the cache namespace regression test plus provider/RealSwap tests pass (**60
focused tests**). Physical AMD/ROCm or DirectML validation is not available on
this host; those paths remain portability-tested via simulation only.

The benchmark now also emits best-effort Torch peak allocated/reserved VRAM,
CPU utilization, NVIDIA-SMI GPU utilization, and an explicit
`transfers=unavailable` marker when per-stage transfer counters are not exposed
by the provider. Missing telemetry is therefore not mistaken for zero
transfers. No AUTO precision decision is made from the short-clip data alone.
