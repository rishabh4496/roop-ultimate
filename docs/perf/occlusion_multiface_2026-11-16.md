# Bug #3 / #4: offline occlusion and multi-face tracking

Implementation, 2026-11-16; final verification 2026-11-17. Acceptance remains pending real interaction footage
and the physical RTX 3060. No speedup or general visual-quality claim is made.

## Execution paths

- `multi_face_tracker.MultiFaceTracker`: original MIT two-stage ByteTrack-style
  implementation over the existing application's Kalman state. SciPy Hungarian
  assignment uses 0.55 IoU distance + 0.45 cosine distance on clean >=0.55 detections;
  the existing known shared-crop safeguard uses geometry for contaminated high-score crops;
  [0.20, 0.55) observations rescue unmatched tracks through IoU only, never
  spawn tracks or update identity EMA. Clean appearance gating and the existing
  guarded cross-shot re-ID remain. Stale/out-of-order observations do not rewind
  state. Bindings reject reassignment and expire with tracks.
- `TrackingMixin._precompute_tracks`: the actual ordered detection consumer
  now advances this tracker, linking its live IDs to the archived tracklets.
  A regression reproduced fallback re-ID stealing a track already matched
  to a later detection in the same frame. All Hungarian matches are now
  reserved before re-ID, and obsolete byte-to-archive aliases are removed.
  `_assign_track_sources` still resolves captured references and explicit user
  pins; its result is frozen in `_track_source_map`, read by the existing exact
  track-ID source-selection path. Both selected-face consumers honor these
  frozen bindings before instant appearance gates, so a corrupted crossing
  crop cannot reselect or steal its source. Legacy mutable maps keep their
  existing appearance-gated behavior. Synthetic/coasted geometry is excluded
  from authoritative bindings; current user selection remains authoritative.
  The detector receives a per-call 0.20 floor,
  using its leased analyser; the saved user threshold/config is unchanged.
- `MaskingMixin.paste_upscale`: computes M_occluder on the original aligned
  target under the actual silhouette, multiplies alpha by 1-M_occluder, then
  retains the existing strict-support clamp after temporal matte refinement.
  Semantic hair/cloth/hat and optional glasses exclusion combines with robust
  trimmed LAB Mahalanobis residual, target-vs-swap structural evidence, connected
  component cleaning and protected semantic eyes/brows/nose/mouth/lips.
  Without labels only support-boundary-connected residuals are accepted.
- `OcclusionMasker`: backward Farneback flow advects the previous probability
  map into current coordinates. Current evidence wins; a 0.35 historical tail
  covers a single dropout. Gaps, scene cuts, unordered workers and incompatible
  crop shapes cannot propagate history. CPU analysis is capped at 192 pixels
  per side; history is capped at 32 tracks. Ordered stabilization blocks clone
  their own masker. Scene flush and job initialization/release reset it.
- `ProcessMgr._composite_faces` / `face_z_order.semantic_depth_regions`: stable
  track-ID depth ties, 0.5 relative box area + 0.3 normalized chin Y + 0.2 landmark
  visibility. Quantization resists small detection noise. IoU >0.05 activates
  dilated foreground semantic subtraction; foreground ownership is not clipped
  against a background face. Unswapped bystanders also claim foreground pixels.
  Masks use pair ROIs, with the existing hull fallback when BiSeNet is absent.

`app/api.py` needs no new endpoint or setting: preview and render already use
ProcessMgr and existing reference/manual pin assignment. No launcher, precision,
provider policy, venv or saved look configuration is changed. No new dependency,
model, network call, or copied upstream tracker code is introduced. New modules
carry SPDX MIT; existing NumPy/SciPy are BSD-3-Clause and OpenCV is Apache-2.0.

## Validation record

`app/tests/test_occlusion_and_multiface.py`: 14 passed (included in the final focused run). It exercises moving dark and skin-toned
rectangles, protected eyes and glasses preference, true Farneback propagation
through a missing residual, gap/unordered state isolation, crossing/weak-score
association, immutable/expired bindings, actual ordered pre-pass archive identity,
actual compositor mask delivery, actual paste pixels and detector threshold
forwarding. Only detector outputs/semantic labels stand in for model inference;
association, Kalman, flow, mask analysis and paste math run for real.

The first integration put residual analysis before the paste silhouette and
failed the 600-frame outcome gate (13/600 changed despite 600/600 audit intent).
Moving it into paste_upscale under the real silhouette fixed that failure:
the next 600-frame render changed 597/600 frames and recorded a new baseline.
The initial repeat PASSed: 596/600 changed, face SSIM mean/min
0.9925/0.9873, face PSNR 52.1861 dB, frame PSNR 66.2287 dB.
`interaction_occlusion` executed 649 times (warm-up block overlap included).
Evidence: `benchmarks/baselines/benchmark_occlusion_multiface_20261116.json`.
The configured Restore Ultra enhancer had **zero timed enhance calls** on
this small-face fixture (existing minimum-face-size gate), so this run does
not validate an active enhancer. Both mask and swap stages ran.
This compares against the new 600-frame baseline, not the historical 300-frame
golden, and is not an old-code A/B or a speed claim. The two passing arms have
the same pinned 12-frame stabilization blocks (`ROOP_STAB_CHUNK_MB=256`). Do not use the failed integration run as a baseline.

Focused tracking/assignment/overlap checks: 155 tests + 5 subtests passed before
the per-call detector-floor addition. The subsequent detector/ROI set has one
failure, `test_small_card_prepass::RescueSwitch::test_roi_default_is_untouched`
(expected 4 raw calls, actual 1). It fails identically with the five unchanged
production modules exported from HEAD into an ignored control directory.

The final focused set (new cases, tracking, selection, holds, identity locks,
checkpoint codec and overlap compositor) passed 265 tests after all code changes.

Stage 2 checkpoints preserve the frozen-binding marker on restore. The
pre-pass feature fingerprint now includes `association_policy=bytetrack-v1`,
so earlier cached association results cannot silently substitute for this path.

No claim that heuristic residuals can distinguish every beard/skin-coloured
object or repair hallucinated detector landmarks. Synthetic labels do not prove
real BiSeNet occluder segmentation. Real hands, hair, cups, microphones, glasses
and interacting-face footage, counterbalanced >=600-frame A/B with swap rate
and faces_seen, fidelity/A-V/leak tests and RTX 3060 memory validation remain
open. Compiled face_engine engines were not prebuilt before the full run; engine/GPU failures remain in its result.

Validation subprocesses block external DNS and TCP via an ignored sitecustomize
under app/output/offline_guard; HF/Transformers offline modes are also enabled.
No Git push is performed because the task explicitly forbids external networking.

## Final verification, 2026-11-17

- Final code: 265 focused tests passed; 600-frame regression PASS, 596/600 visibly changed, face SSIM
  mean/min 0.9920/0.9853, face PSNR 48.837 dB, frame PSNR 63.8595 dB.
  The occlusion analysis stage ran 649 times. Evidence:
  `benchmarks/baselines/benchmark_occlusion_multiface_20261117.json`.
- Full run: 7,966 passed, 80 failed, 30 errors, 31 skipped, 11 xfailed,
  4 deselected, 3,603.76 seconds. This run started before the final binding,
  cache and same-frame re-ID corrections; those were verified separately in
  the post-correction focused run and the final 600-frame render. The full
  suite is **not green**. Raw log: `app/output/occlusion_full_suite.txt`.
- Its existing 30-frame model-backed interaction fixture found/swapped 24/24
  detected targets behind the bystander and 30/30 targets in front, with zero
  bystander swaps. Front bystander pixel MSE was 0; behind MSE max 0.3034.
  The deeply covered target was still undetected on 6 frames. The skin-colour
  gate now passes (dE max 0.4615), producing an XPASS(strict) against its old
  expected-failure marker. No gate or expected-failure marker was weakened.
- The fixture's flicker ratchet FAILS (2.045 versus <=1.6). A separate matched
  bar-only control, exporting seven original production modules from commit
  `eb5d231`, reproduced the failure: HEAD 1.92975; final code 1.99959. Both
  use 30 frames, realswap/RealityUX/no enhancer, TensorRT, 20 threads and
  `ROOP_STAB_CHUNK_MB=256`. This is a failure reproduction on a short synthetic
  fixture, not a counterbalanced acceptance A/B or evidence of neutrality.
  Data: `benchmarks/baselines/benchmark_occlusion_bar_control_20261117.json`.
  Residual flicker remains open; Bug #3 is **not fully closed**.

