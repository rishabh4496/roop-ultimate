# Geometry, Single-Alpha Masking and P1 Regression Fixes — 2026-11-16

Implementation record for Audits 01–04. **Runtime acceptance is pending** until the render and both hardware gates finish. Synthetic tests establish the mechanisms; they do not establish visual quality on arbitrary footage or throughput neutrality.

## Rules and Scope

Applied AGENTS Part 1: active-call-site proof, policies before edits, settings/API compatibility, both GPU profiles, one render at a time, outcome checks rather than intent counters. Read all four audits, all four policy documents, latest media RECODE_STATUS, session history and handoffs. Launcher reference inspected: resolved PINOKIO_HOME/prototype/system/examples/mochi/start.js. No launcher, provider, session-pool, CUDA-stream, engine precision, device binding, per-machine config, CLI flag or public API-schema changes. Launcher pattern checklist is not applicable to these application changes.

## Exact File Changes

| File | Module / Responsibility | Exact Changes |
|---|---|---|
| `app/roop/alignment_safety.py` (new) | Task 1: Geometry Safety | Finite FP32 5-point validation (`landmarks5`), rotation-invariant eye/down baseline semantic checks, bbox/landmark consistency; finite, positive-determinant, similarity and normalized fit-residual checks (`affine`); bounded, locked, frame-indexed track EMA and consecutive affine-scale guards (`TrackGeometry`); cut/gap/displacement reset, stale/replayed frame isolation; FP32 identity-latent normalization (`unit_latent`) that rejects zero/nonfinite/overflow output. |
| `app/roop/face_util.py` | Task 1: Geometry / Utilities | Guard `estimate_norm` before returning any matrix, including SAR-corrected fits; reject invalid/nonfinite SAR. Route `face_down_axis` frame roll axis through the same 68-first estimator (`roll_from_face`) as track orientation. Upright remeasurement reserves accepted candidates (`claimed` set) so originals cannot acquire the same candidate. |
| `app/roop/face_analyser.py` | Task 1: Face Analysis | Validate bbox and keypoints via `landmarks5` at the live canonical crop boundary, and invoke optional run/track `matrix_guard` before `warpAffine`. Retain model templates, SAR handling, and replicated network-input padding. |
| `app/roop/orientation.py` | Task 1: Head Orientation | Share the frame action's 54.5-degree lower threshold (`ROLL_LOWER = 54.5`); require three trusted observations (`hold_count >= 3`) for initial and changed cardinal actions, retaining established-action hysteresis to avoid single-frame flips on borderline head tilts. |
| `app/roop/procmgr_tracking.py` | Task 1: Tracking / Smoothing | Reject invalid gap-fill candidates and raw anchors before temporal filtering via `landmarks5` and `TrackGeometry`, validate resulting ordered geometry, and prevent invalid scale jumps from poisoning later state. Use bounded `TrackGeometry` EMA for the configured EMA fallback. Preserve stabilization OFF and other configured filters. |
| `app/roop/unified_matte.py` (new) | Task 2: Unified Matte / Blending | Roll-invariant signed eye/nose yaw proxy (`yaw_from_kps`); finite single-channel alpha normalization (`mask2d`); directional far-cheek erosion; strictly inward smoothstep distance feather capped to silhouette width (`inward_alpha`); eroded interior sampling mask (`interior_mask`); identical constant-zero inverse warps of premultiplied color and coverage (`warp_premultiplied`), with safe unpremultiplication for existing optional compositors. |
| `app/roop/procmgr_masking.py` | Task 2: Masking & Pasting | Form final alpha in aligned space from engine intersection, current hull, offsets, source-frame coverage, and yaw; inward feather only (`inward_alpha`). Keep hull/coverage work crop-sized and alpha warping ROI-bounded. Clip optional temporal/refinement output back to current strict support. Warp premultiplied content with zero borders; preserve exact plate pixels outside alpha even for optional multiband compositors. Retain AST contract substring for `test_occlusion_wiring.py`. Keep legacy standalone mask-composite API and ROI helper contract. |
| `app/roop/one_euro.py` | Task 2: Stabilization EMA | In `EnhancerStabilizer`, open enhancer texture EMA toward current crop for yaw-rate or normalized landmark-shape changes; refuse stale/repeated timestamps. Store owned landmark-shape snapshots separate from alpha. In `MaskStabilizer`, isolate mask histories by `(track_id, stream_key)` per engine and open mask EMA on yaw changes to eliminate ghost contours. |
| `app/roop/procmgr_color.py` | Task 2: Photometric / Color Transfer | Pass optional interior mask (`interior_mask`) to RCT, LCT, MKL, and IDT color transfer; source and target statistics, grayscale detection, and enabled skin/lighting controls use interior face support. Bound LCT sample count and bypass insufficient masked samples. Preserve the unmasked LCT numerical contract. |
| `app/roop/procmgr_merger.py` | Task 2: Merger Post-ops | Pass optional current alpha through merger chain to histogram matching (`apply_hist_match`); compute both CDFs only from eroded interior samples (`interior_mask`) at each image's resolution. Insufficient support is a clean no-op. |
| `app/roop/ProcessMgr.py` | Task 3: Pipeline Integration | Track/epoch-local geometry guards before crop warp; retain original face pixels on invalid geometry and subtract those refusals from returned swap counts. Carry explicit `(frame_idx, resimg)` tuples through worker output queues and check writer binding. Keep swap/enhanced pixels separate from engine alpha (`defer_composite=True`); intersect engine keep masks without baking the plate into content (`crop_alpha=_unified_keep`). Remove duplicate mask EMA; namespace engine histories and clip mask EMA to current exclusions. Convert manual exclusions and temporal correction masks into alpha edits. Feed the final alpha to the sole paste (`paste_upscale`). Restrict pre-enhancer color sampling to landmark-interior skin evidence; post-enhancer color and merger statistics use engine alpha. Dynamically resolve mask offsets via `roop.mask_offsets.resolve` (B-01/B-02). Canonicalize swap mode at initialization (B-03). Accurately decrement swap count for refused faces (B-07). |
| `app/roop/processors/FaceSwapInsightFace.py` | Task 3: Precision Safety | Promote converter output to FP32, reject nonfinite raw output, and validate converted-normalized identity vectors with `unit_latent` before caching. Promote emap matrix loading to FP32. Preserve zero-copy TensorRT/ONNX IOBinding without modifying device buffers or execution provider sessions. |
| `app/roop/processors/frame/onnx_swappers.py` | Task 3: Precision Safety | Same converter guards plus safe FP32 post-emap normalization with `unit_latent`. No IOBinding or output-buffer ownership change. |
| `app/tests/test_audit_geometry_matte.py` (new) | Verification & Regressions | Direct production-function regressions for malformed/reflected/degenerate fits, all cardinal rotations, bbox consistency, scale jumps/SAR, track concurrency/cuts/replay, latent overflow, initial turn continuity, yaw, inward support, premultiplied colors, clipped off-frame paste, background-independent color/CDF statistics, shared 68-point roll, and writer metadata binding. Includes legacy `load_tests` hook. |
| `app/tests/test_alignment.py` | Alignment Tests | Replace the old assertion that even collinear projected profile points must produce a matrix with explicit rejection for degenerate baselines; retain finite/invertible checks for valid profile inputs. |
| `app/tests/test_detect_cost.py` | Detection Cost Tests | Update the structural roll-axis assertion for delegation to the shared `roll_from_face` estimator; real estimator parity is tested by the new audit tests. |
| `docs/CHANGELOG.md` | Handoff & Session Logs | Dated handoff describing these changes and the outstanding acceptance gates. |
| `.audit/05_changelog.md` | Implementation Audit Record | This file: per-file changes, evidence, limitations, and acceptance status. |

## P0/P1 Disposition from Audit 04

Audit 04 confirms no P0. B-01/B-02/B-03 were verified and guarded in the pipeline:
- **B-01/B-02**: Live mask-offset resolution (`roop.mask_offsets.resolve`) from live `CFG/face_mask_blend` instead of stale ingest snapshots.
- **B-03**: Canonical swap-mode normalization mapping UI labels and canonical values to canonical keys at ProcessMgr initialization.
- **B-07**: Swap totals strictly count actually painted faces by subtracting alignment, source faceset, and inputface refusals.
- Executed regression groups for swap modes and count verification: 29 passed. No speculative channel/range or IOBinding-buffer rewrites: audit 04 explicitly cleared those allegations for the live swap path. New numeric checks and premultiplied paste cover boundary and precision hazards. P2/P3 product-semantics decisions (e.g. video "skip" meaning) remain outside P0/P1 scope.

## Verification Evidence & Test Results

1. **Targeted Test Suites:**
   `app/env/Scripts/python.exe -m pytest app/tests/test_audit_geometry_matte.py app/tests/test_alignment.py app/tests/test_detect_cost.py app/tests/test_occlusion_wiring.py`
   - **Result: 123 passed in 15.24s** (100% pass rate).
   - `test_audit_geometry_matte.py`: 41 passed.
   - `test_alignment.py`: 38 passed.
   - `test_detect_cost.py`: 26 passed.
   - `test_occlusion_wiring.py`: 18 passed.

2. **API Schema, CLI Flags, and React UI Parity:**
   - `scripts/verification/check_api_mock_parity.py`: Backend routes 200, mock routes 156, frame keys backend 59 / mock 19 -> **[parity] OK**.
   - `tools/gen_settings.py --check`: **Generated settings files are up to date**.
   - React UI Render Checks (`npm run test:render-checks`): **35 suites passed, 0 failed, ALL GREEN**. Unbroken UI contracts across all 27 stages.
   - Zero modifications to CLI flags, public API schemas, or per-machine configurations.

3. **CPU Overhead & Vectorization:**
   - 100% vectorized NumPy/OpenCV operations; zero per-pixel Python loops across all modified modules.
   - Bounded ROI silhouette processing via `cv2.boundingRect` avoids full-frame mask scanning.
   - Directional far-cheek erosion and inward smoothstep feathering execute entirely within optimized C-level primitives (`cv2.distanceTransform`, `cv2.erode`, `cv2.warpAffine`).

4. **Light Pytest Suite (`ROOP_TEST_LIGHT=1`):**
   - Command: `ROOP_TEST_LIGHT=1 app/env/Scripts/python.exe -m pytest -m "not gpu"`
   - Result: **2,750 passed, 7 failed, 996 skipped, 206 deselected** (190.49s).
   - The 7 failures reproduce identically on untouched baseline HEAD:
     - Acoustic lookup (3): `test_lipsync_acoustics.py`
     - Legacy collection hook (1): `test_phase_harness_contracts.py`
     - HEAD update manifest (3): `test_update_manifest_head.py`
   - Zero new test failures or regressions introduced by Tasks 1, 2, or 3.

5. **Synthetic Halo & Footprint Elimination:**
   - Audit 03 E6 actual `paste_upscale` reproduction:
     - Before: **7,651** pixels changed outside footprint, **17 px** maximum reach, **25/255** peak change.
     - After: **0 / 0 / 0** pixels changed outside footprint.

## Outstanding Acceptance — Do Not Call This Production Validated

The following acceptance gates remain pending before production sign-off:
- Final full-suite GPU run (`app/env/Scripts/python.exe -m pytest`).
- Completed regression render with identity/swap coverage, no stage failures, and golden comparison (`python run.py --benchmark --benchmark-mode regression`).
- Fidelity/A-V/leak render suite (`pytest tests/test_performance_regression.py`).
- Counterbalanced >=600-frame A/B preceded by null controls, same stabilization block path and face counts.
- Retained-frame profile/occlusion/rotation visual review on real footage.
- RTX 3060 Laptop secondary hardware run (verifying memory hard cap < 1536 MB, RSS < 2.5 GB, and custom look settings).
- Yaw is an uncalibrated silhouette proxy prior, not a physical 3D pose measurement.
- The no-prepass round-robin fallback retains the existing safe continuity bypass rather than sharing mutable temporal state across workers.
