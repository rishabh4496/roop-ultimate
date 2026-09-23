# Detection and interacting-face traceability matrix

Date: 2026-09-23

This matrix maps every explicit scenario in the request and every changed public behavior to a concrete check and its observed result. It deliberately separates detector evidence from proof that a selected source was actually applied in a complete video render.

## User requirements

| Requirement | Concrete check | Observed result | Status |
|---|---|---|---|
| Straight/upright faces | Live CUDA multi-angle sweep on `double/d1.mp4`, frame 5, 0 degrees | 1 face detected | PASS for detector path |
| Upside-down faces | Same sweep, 180 degrees | 2 faces detected | PASS for detector path |
| Rotating/cardinal angles | Same sweep at 90 and 270 degrees | 90 degrees: 2 faces. 270 degrees: 1 face | PARTIAL. The 270-degree case remains a miss |
| Faces at different angles in one frame | Rotated-rescue unit `test_rescue_rotated_accumulates_across_different_orientations` plus synthetic combined upright/inverted frame | Unit path accumulated 2 faces. Synthetic combined frame detected 1 face | PARTIAL. The real detector composite limitation remains unresolved |
| Hands, ice creams, bananas, and similar foreground objects | `tests/occlusion_ground_truth.py --occluder texture --frames 20`, documented in `app/docs/PHASE2_OCCLUSION.md` | 7 informative frames. Synthetic texture protection was neutral within noise: worst 18% off versus 20% on, median 40% versus 42%. Clean swap retention was 98% | LIMITED. This is a synthetic textured occluder, not real hands, food, or microphones. Real-object footage was not run |
| Half faces and missing keypoints | Box-only `crop_contamination` regression and boundary/close-up detector tests | Box-only overlap returned contamination above 0.70. Multi-scale odd-dimension and close-up tests passed | PARTIAL. No dedicated real half-face clip was accepted |
| Two faces touching or kissing | Contact-junction sweep and `face_contact` tests | Phantom junction index `[2]` was removed, both parents remained, three real faces remained, distant small face remained, and profile crop contamination was `[0.454, 0.429]` | PASS for geometry and junction handling |
| Kissing on cheeks, lips, or forehead | Contact geometry and overlap demarcation tests with interacting profiles, overlapping boxes, and facial hulls | Duplicate candidates were rejected, distinct touching faces were preserved, contested overlap was partitioned, and minimum sequential coverage was 1.0000 | PASS for geometry. No complete real-video visual acceptance was established |
| Two facesets interacting with each other | `app/tests/ab_face_count.py` is the concrete two-faceset full-video harness. The real-identity selected-face compositor harness, public selected-face acceptance, bounded chunk attempts, and live full-video render were also attempted | The GPU identity harness passes for selecting A, selecting B, and explicitly selecting nobody. It proves source identity and bystander preservation on a three-face real-faceset canvas. The public acceptance captured faces but changed 0 pixels for either detected person. The 418-frame production render timed out after 600 seconds before output creation. A manual 20-frame chunk could not form two targets because its single capture frame contained one face, and the legacy per-person capture remained stalled until stopped. The earlier 10-frame smoke changed pixels but did not prove source identity application | PARTIAL. Selected compositor routing is fixed and verified, but complete applied-source acceptance on the real source video remains blocked by the live render/capture path |
| No change to face-angle formulas | `TestStrictAngleFormulasUntouched`, golden angle tests, and source diff over the implementation commits | Protected signatures and golden calculations passed. No protected angle implementation changed | PASS |

## Changed public behavior

| Changed behavior | Concrete check | Observed result |
|---|---|---|
| `parse_scale_pyramid` rejects invalid, non-finite, out-of-range, and excessive scales | `test_parse_scale_pyramid_bounds_and_robustness` | Invalid values fell back safely, valid values were bounded to five levels, and standard aliases remained compatible |
| `generate_scale_pyramid` preserves effective anisotropic resize ratios | `test_scale_pyramid_odd_dimensions_exact_rescaling` | Odd 101x151 input produced 50x76 output and exact x/y inverse scaling |
| `rescale_detections` keeps boxes and keypoints aligned | Same odd-dimension test plus `test_multiscale_kpss_pairing_synchronization` | Coordinates matched expected values and boxes/keypoints stayed equal length |
| Explicit single-scale detector mode does not silently use default pyramid levels | `test_multiscale_detector_explicit_single_scale_does_not_use_default_scales` | Detection function was called exactly once for `[1.0]` and `off` |
| Rotated rescue accumulates detections and survives one failed orientation | `test_rescue_rotated_accumulates_across_different_orientations` and `test_rescue_rotated_per_orientation_try_except` | Two distinct orientation detections accumulated. A simulated first-orientation exception did not abort the next orientation |
| Touching-face duplicate discrimination | `_is_face_duplicate` tests and live contact sweep | Concentric duplicates were suppressed while distinct touching faces with separated centers or landmarks were preserved |
| CLAHE rescue enriches from the clean source frame | `test_clahe_rescue_enrichment_uses_clean_original_frame` | Auxiliary recognition received the untouched value-42 frame and populated an embedding |
| Missing-keypoint crop contamination fallback | `test_contamination_falls_back_to_box_overlap_when_quad_is_none` | Heavy box overlap returned contamination above 0.70 rather than zero |
| Junction suppression | `face_contact` unit suite and live sweep | Only the junction was removed. Parent faces and valid three-face layouts survived |
| Connected-component overlap demarcation | `TestFaceOverlapConnectedComponents`, `face_overlap` suite, and live sweep | Disjoint pair ROIs stayed localized, contested coverage reached 1.0000, and bystanders were protected |
| Mask occluder input metadata | `test_mask_occluder_model_input_shape_metadata` | The public pool shape is `(1, 256, 256, 3)` and the obsolete `(1, 3, 512, 512)` shape is absent |
| Detector fallback observability | `test_no_new_silent_broad_handlers`, fallback reporter tests, and exact suite | New broad handlers are observable. Repeated fallback counts aggregate once per site, and strict mode re-raises |
| Pool and detector startup logging | Focused live initialization logs and the exact suite | Pool messages render with ASCII separators and report the active provider and instance count without changing detection behavior |
| `get_all_faces` compatibility with one-argument detector mocks | Full exact regression gate, including the compatibility regression | The compatibility fallback passes without changing the production empty-list contract |
| Explicit selected-face state is not widened to legacy all-captured fallback | `app/tests/test_target_selection_contract.py` plus the GPU `integration_selected_face_regression.py` harness using `akansha`, `anshita`, `anushree`, and `ashna` facesets | The prior harness exposed an explicit `selection_mode=none` being widened to person A. ProcessOptions now tracks constructor and post-construction selection assignment. Focused contract tests pass, and the GPU identity harness reports `VERDICT: PASS`: A and B selections swap only the selected person, no selection routes `{}`, and all identities remain unchanged |
| Real pipeline output production | 10-frame smoke artifact check | 10 input frames produced 10 output frames and all 10 changed pixels, but source identity and selected-face application were not proven |

## Regression and integration gates

- Exact required gate after the selected-face fix: `app\\env\\Scripts\\python.exe -m pytest -q` -> **3288 passed, 6 skipped, 2 xfailed, 8 warnings, 1009 subtests passed** in 500.08 seconds.
- Post-fix focused selected-routing gate: **33 passed** in 3.70 seconds.
- Post-fix GPU selected-face identity harness: **VERDICT: PASS** in 114.95 seconds. It used four available facesets and verified selected A, selected B, explicit no-selection, preview/render parity, and per-candidate route logging.
- Served API startup reached `API_READY` and `UI_READY` on `http://127.0.0.1:8001` with TensorRT active. The public acceptance rerun reached the backend but skipped before upload because its shared source fixture was unavailable; the local `app/facesets` identities used by the GPU harness are present. The temporary backend was stopped after this check.
- Focused changed-behavior set: **95 passed, 3 skipped** in 10.43 seconds.
- Prior light gate: **3233 passed, 6 skipped, 2 xfailed, 54 deselected**.
- The selected-face fix and traceability updates are committed and pushed on `origin/main`; the repository was clean before the final validation attempts.

## Acceptance boundary

The implementation and detector/geometry regressions are validated. Complete applied-source acceptance is not. The required next acceptance artifact is a continuous full-source-video render with per-frame and per-face swap counts, source-identity or visual audit, and a nonzero applied-source result. A short output that merely differs from its input is insufficient.
