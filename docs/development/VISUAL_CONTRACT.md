# Visual-Quality Contract — Stage 2A Audit

Audit date: 2026-09-01  
Repository: `main` at `fd40c31438e8e03b77e3e2abaaad5266b3f61049`  
Scope: read-only audit of the current visual and output pipeline. No application
code was changed for this gate.

## Status vocabulary

- `CURRENT`: present in the repository, regardless of whether it is enabled by default.
- `WORKING`: implementation and relevant automated/source checks agree. This does not mean final retained-output visual acceptance.
- `PARTIAL`: works only under stated conditions, falls back, or is unavailable on one execution path.
- `BROKEN`: repository evidence shows a known failure for the stated scope.
- `MISSING`: no implementation was found for the requested capability.
- `UNVERIFIED`: implementation exists, but the required visual, hardware, or end-to-end evidence is absent.

`WORKING` below means implementation-level evidence. The Phase 16 report remains
`OPEN_INCOMPLETE` with 17 clips, 425 rows, zero ready clips, and zero complete
runs; therefore no row is a final visual-quality promotion.

## CURRENT IMPLEMENTATION

### End-to-end visual path

| Stage | Current implementation and exact evidence | Controls / limits | Status |
|---|---|---|---|
| Input and frame acquisition | `/api/swap` starts `_run_swap` in `app/api.py:2651-2673`. `core.batch_process` chooses extracted-frame processing or `ProcessMgr.run_batch_inmem` in `app/roop/core.py:1011-1079`. Video encoder availability is probed before the long analysis pass in `core.py:980-994`. | Images, video, GIF, and animated WebP have separate branches in `core.py:1075-1079` and `1095-1106`. | `CURRENT / WORKING` at control-flow level; full media-matrix visual acceptance is `UNVERIFIED`. |
| Detection | `ProcessMgr.swap_faces` uses cached temporal faces when covered, otherwise `get_first_face` / `get_all_faces`, with ROI rescue and partial-miss rescue in `app/roop/ProcessMgr.py:3417-3493`. Detection resolution, threshold, NMS, detector engine, landmark refinement, and small-face rescue are applied by `app/api.py:2488-2503` and `2707-2765`. | Detector-specific resolution behavior is explicit in `react-ui/src/components/FaceSwap.jsx:1980-1988`; the label states YOLOFace/SCRFD exports ignore the resolution selector. | `CURRENT / WORKING` by source and settings tests; real difficult-scene recall is `UNVERIFIED`. |
| Alignment / pose | `process_face` selects the swap model’s output size/template and calls `align_crop` in `app/roop/ProcessMgr.py:4629-4646`. It computes corrected 5-point pose with `solve_pose_5pt` and jaw-blind angles in `ProcessMgr.py:4677-4745`. Autorotation re-detects and re-expresses ownership regions in `ProcessMgr.py:4500-4627`. | Swap templates are model-specific. Optional enhancer re-alignment is controlled by `enhancer_align` and implemented in `ProcessMgr.py:5220-5278`. | `CURRENT / WORKING` for implemented paths; angle-specific retained-output review is `UNVERIFIED`. |
| Source selection / face processing | FaceSet V1/V2 source selection is in `ProcessMgr.py:4809-4874`; V2 pose/lighting/expression selection is conditional on `use_source_bank`. Image-source-only 3D reconstruction is gated in `ProcessMgr.py:4922-5010`. Frontalization and inverse defrontalization are in `ProcessMgr.py:5012-5042` and `5188-5195`. | V1/no-bank/missing metadata falls back to the first source. 3D reconstruction does not affect embedding-based swappers; this is explicit in `ProcessMgr.py:4922-4939`. | `CURRENT / PARTIAL`: fallback behavior is explicit and tested, but source-bank/V2 quality is not fully graded. |
| Swap model | `FaceSwapInsightFace.SWAP_MODELS` defines actual models, sizes, templates, normalization, embedding mode, and optional secondary model in `app/roop/processors/FaceSwapInsightFace.py:44-342`. `Initialize` loads the selected graph and publishes the model contract in `:557-712`. `Run` performs primary inference and optional RealSwap secondary eyelid-band composition in `:1722-1789`. | RealSwap is a two-network composite; its measured 4070 cost and batching limitation are documented in `:1722-1761` and `:720-735`. | `CURRENT / WORKING` for model dispatch/fallback; comparative visual ranking is `UNVERIFIED` except for recorded historical rows. |
| Enhancement | `ProcessMgr.process_face` runs the selected enhancer after swapping and mask setup in `ProcessMgr.py:5215-5318`; optional enhancer alignment is opt-in. Adaptive selection is delegated to existing restorers and receives quality/pose/occlusion metrics at `ProcessMgr.py:5204-5214`. | Outputs are checked for non-finite values by enhancer implementations. GFPGAN also checks finite-but-collapsed output in `app/roop/processors/Enhance_GFPGAN.py:98-126`. | `CURRENT / WORKING` for guards/dispatch; model-by-model visual quality is `UNVERIFIED`. |
| Mask construction | `MaskingMixin.process_mask` supports dense engines, SAM2 precomputed masks, optional non-frontal routing, occluder threshold/blur, undersized-mask recovery, mask stabilization, temporal identity masks, per-face ownership, and temporal occlusion observation in `app/roop/procmgr_masking.py:853-1197`. | A second mask engine is mapped and applied through `app/api.py:380-416`, `ProcessMgr.py:5196-5203`. `mask_engine_2` can restore more original pixels but does not narrow the swap by design. | `CURRENT / WORKING` by mask and wiring tests; difficult occlusion visual acceptance is `UNVERIFIED`. |
| Mask geometry / ownership | `paste_upscale` builds the ellipse, intersects the landmark hull, feathers, applies optional model mask, temporal matte, overlap ownership, and then blends in `procmgr_masking.py:380-480`. Multi-face ownership and far-to-near paint order are in `ProcessMgr.py:4288-4348`. | Model masks are emitted only by graphs whose output shape proves a one-channel output; see `FaceSwapInsightFace.py:700-711` and `_stash_masks` at `:919-940`. | `CURRENT / WORKING` on normal and batched-with-attribution paths; model-mask behavior on cross-frame batching is `PARTIAL`. |
| Blend / paste-back | `paste_upscale` uses float32 alpha compositing, optional ROI-only warps, optional temporal multiband composition, and bounded uint8 output in `app/roop/procmgr_masking.py:491-573`. The legacy path remains available when ROI mode is disabled or optional temporal composition fails. | `blend_ratio` mixes enhanced output with the base swap before the final matte at `procmgr_masking.py:523-550`; `output_face_scale` changes the paste matrix at `:384-390`. | `CURRENT / WORKING` at implementation level; natural-boundary visual scoring is `UNVERIFIED`. |
| Color / illumination | `ColorTransferMixin.apply_color_transfer` supports `none`, `rct`, `lct`, `mkl`, and `idt`, with grayscale protection and optional target-conditioned appearance in `app/roop/procmgr_color.py:59-111`. Exact transforms are `_color_transfer_rct/lct/mkl/idt` at `:113-322`. | First color pass is after swap at `ProcessMgr.py:5179-5186`; optional re-match after enhancement is `:5385-5409`. Target-conditioned lighting uses low-frequency luminance/chroma rather than target texture in `appearance_conditioning.py:135-206`. | `CURRENT / WORKING` for modes/wiring; scene-level color judgment is `UNVERIFIED`. |
| Sharpening / detail | Signed merger sharpening, clarity, histogram match, motion blur, grain match, and degrade are applied after enhancement/color in `ProcessMgr.py:5412-5427`; implementation and neutral no-op contract are in `app/roop/procmgr_merger.py:82-130` and `:135-237`. Detail transfer is an edge-stopped high-pass operation in `procmgr_color.py:14-57`, invoked at `ProcessMgr.py:5367-5383`. | UltraMax’s former refinement filter is retained but disabled by default; `Enhance_UltraMax.py:36-87` records the measured no-gain/worse-flicker result. `merger_clarity` is implemented but has no normal React control. | `CURRENT / WORKING` for enabled operations and neutral behavior; visual benefit/regression is `UNVERIFIED`. |
| Mouth / eyes / expression | Original mouth restoration and lip-sync are mutually arbitrated at `ProcessMgr.py:5785-5815` and `5866-5896`. Eye restoration and expression-region restoration use separate geometry and pose gates at `:5816-5865`. | React controls and conditional fields are in `FaceSwap.jsx:2286-2358`. Expression restoration depends on its restorer/model path and is not a universal free CPU operation. | `CURRENT / PARTIAL`: code and unit coverage exist; profile/extreme-angle visual acceptance is `UNVERIFIED`. |
| Temporal stability | Keypoint stabilizers are selected in `ProcessMgr.py:869-894`; frame-level temporal detection/precomputation and ordered execution are in `ProcessMgr.py:3397-3428` and `app/roop/procmgr_tracking.py`. Enhancer and mask stabilizers run at `ProcessMgr.py:5320-5329` and `procmgr_masking.py:1118-1154`. Temporal identity, appearance, compositing, occlusion, expression, and quality control are instantiated in `ProcessMgr.py:582-606` and used at `:5537-5566` and `:5625-5718`. | Stabilization is skipped for rotated crop space where geometry is inconsistent. Temporal identity/compositing/QC remain opt-in and are not normal React controls. | `CURRENT / PARTIAL`: state transitions and bounded behavior pass automated tests; visual temporal acceptance is `UNVERIFIED`. |
| Encoding | `FFMPEG_VideoWriter._build_cmd` writes raw BGR24 frames, selects NVENC/software codec parameters, uses CRF/CQ, bounds threads, and normally applies a colorspace filter in `app/roop/ffmpeg_writer.py:198-293`. Hardware encoder startup can fall back to software before the first frame in `:313-345`. | Output pixel format is `yuv420p` at `:286-287`. The odd-dimension branch uses a scale filter and does not also append the default colorspace filter (`:270-281`). | `CURRENT / PARTIAL`: normal encoding/fallback is implemented; odd-dimension color behavior requires validation. |
| Recovery / output finalization | `SegmentedVideoWriter` rotates playable segments, writes a manifest, resumes only matching identities, and promotes/concatenates output in `app/roop/segment_writer.py:120-395`. `core.py:1081-1141` finalizes stopped output, restores audio, converts GIFs, and cleans temporary frames. | Deliberate stop still finalizes a playable partial output. AI upscale and interpolation run after swap, with swap resources released first, in `app/api.py:2867-2919`. | `CURRENT / WORKING` at source/control level; long-run crash/stop output evidence is `UNVERIFIED` in this session. |

## Verified visual feature matrix

| Feature | Current behavior and evidence | UI exposure | Status |
|---|---|---|---|
| Model-specific alignment/normalization/output size | Driven by `SWAP_MODELS` and published processor attributes, not one hardcoded 128px path (`FaceSwapInsightFace.py:44-56`, `:690-712`; `ProcessMgr.py:4630-4646`). | Swap model and subsample controls at `FaceSwap.jsx:2008-2022`. | `CURRENT / WORKING` |
| RealSwap primary/secondary composition | Hyperswap is base and HifiFace contributes the eyelid band; secondary mask is drained and primary mask remains authoritative (`FaceSwapInsightFace.py:1722-1789`, `:1632-1710`). | `realswap` appears in the swap model list/default path. | `CURRENT / WORKING`; visual quality `UNVERIFIED` |
| Target-conditioned lighting | Low-frequency luminance/spatial lighting and bounded chroma are applied when enabled (`procmgr_color.py:135-206`); EMA configuration is created in `ProcessMgr.py:591-603`. | Toggle, strength, temporal alpha at `FaceSwap.jsx:2030-2033`. | `CURRENT / WORKING` implementation; visual acceptance `UNVERIFIED` |
| RCT/LCT/MKL/IDT color modes | All four transforms have code; IDT is intentionally expensive and deterministic via seeded rotation (`procmgr_color.py:94-110`, `:261-322`). | Color selector at `FaceSwap.jsx:2029`. | `CURRENT / WORKING` implementation; comparative quality `UNVERIFIED` |
| Post-enhancer color re-match | Runs only when an enhancer output exists and uses half target-conditioned strength for the second pass (`ProcessMgr.py:5396-5407`). | Toggle at `FaceSwap.jsx:2351-2355`. | `CURRENT / WORKING` |
| Enhancer alignment | Rewarps a swap crop toward the enhancer `model_template`, fills the wider ring from the plate, and warps back (`ProcessMgr.py:5220-5308`). | Toggle at `FaceSwap.jsx:2343-2350`. | `CURRENT / WORKING` implementation; tradeoff `UNVERIFIED` |
| GFPGAN FP32 safety | TensorRT providers are forced to FP32 and finite/collapse guards return the resized pre-enhancer crop (`Enhance_GFPGAN.py:50-69`, `:98-126`). | GFPGAN is selectable. | `CURRENT / WORKING`; 4070 evidence exists, 3060 visual behavior `UNVERIFIED` |
| GPEN precision split | GPEN 1024/2048 force FP32; smaller models use `providers_for` (`Enhance_GPEN.py:90-109`). | GPEN models are selectable where offered. | `CURRENT / WORKING` implementation; end-to-end quality `UNVERIFIED` |
| CodeFormer / UltraMax precision | CodeFormer selects FP16 or FP32 graph through shared policy (`Enhance_CodeFormer.py:65-95`). UltraMax uses CodeFormer FP16 with FP32 post-processing (`Enhance_UltraMax.py:1-29`, `enhancer_inventory.py:26`). | CodeFormer fidelity appears only for CodeFormer labels; UltraMax is selectable. | `CURRENT / WORKING` implementation; visual ranking `UNVERIFIED` |
| Signed sharpen/soften | `apply_sharpen` supports positive sharpening and negative softening with scale-aware radius (`procmgr_merger.py:224-237`). | `Sharpen / soften` at `FaceSwap.jsx:2051`. | `CURRENT / WORKING` |
| Motion blur / grain / degrade / histogram | All four read the original aligned crop and run in the merger chain (`procmgr_merger.py:84-130`, `:191-237` and subsequent methods). | Four controls at `FaceSwap.jsx:2050-2054`. | `CURRENT / WORKING` implementation; footage benefit `UNVERIFIED` |
| Clarity | Implemented as a merger operation and used by `ProcessMgr.py:5419-5425`; neutral default is 0. | No normal React control; payload/config/backend support exists. | `CURRENT / PARTIAL` |
| Face mask edge blend | `blur_area` derives scale-aware blur/erosion from mask extent (`procmgr_masking.py:698-727`), and paste-back applies it after ownership trims (`:423-469`). | `Face mask edge blend` at `FaceSwap.jsx:2078`. | `CURRENT / WORKING` |
| Second mask engine / parser regions | Engine composition is mapped in `api.py:380-416`; Face Parser consumes region selection/grow settings in `app/roop/processors/Mask_FaceParser.py:67` and API helper `api.py:2454-2465`. | Mask engine, second engine, parser controls at `FaceSwap.jsx:2058-2070`. | `CURRENT / WORKING` implementation; difficult-object quality `UNVERIFIED` |
| Swapper-provided face mask | Model graph output is detected by shape and applied by `paste_upscale` when strength > 0 (`FaceSwapInsightFace.py:700-711`; `ProcessMgr.py:5741-5755`). | Slider at `FaceSwap.jsx:2008`. | `CURRENT / PARTIAL`: unavailable on cross-frame batcher path (`ProcessMgr.py:5072-5077`) and not real-footage judged. |
| Manual include/exclude mask | Canonical and legacy frame-space masks are parsed/applied after merger operations in `ProcessMgr.py:5429-5535`; payload reaches run and preview via `FaceSwap.jsx:399-400`, `:654-655`, and `api.py:2830-2838`. | Preview brush mask. | `CURRENT / WORKING` implementation; full-video review `UNVERIFIED` |
| Restore mouth / eyes | Original regions are composited after paste-back with pose-dependent fading (`ProcessMgr.py:5807-5854`). | React controls at `FaceSwap.jsx:2286-2327`. | `CURRENT / WORKING` implementation; extreme-angle quality `UNVERIFIED` |
| Expression restore | Temporal expression planning and regional eye/mouth restore are implemented (`ProcessMgr.py:5797-5865`; `temporal_expression.py`). | Strength/region reach preview and run payloads. | `CURRENT / PARTIAL`: requires its restorer and is not equivalent to simple target-region copy |
| Jaw reshape / face scale | Jaw reshape is a post-composite geometric warp (`ProcessMgr.py:5898-5919`); face scale changes paste matrix (`procmgr_masking.py:384-390`). | Controls at `FaceSwap.jsx:2049`, `:2093-2096`. | `CURRENT / WORKING` implementation; large-change quality `UNVERIFIED` |
| Temporal detection / keypoint stabilization | Temporal pre-pass, gap filling, identity locking, and One Euro/EMA options are implemented (`ProcessMgr.py:3417-3447`; `procmgr_tracking.py`; `ProcessMgr.py:869-894`). | Video controls at `FaceSwap.jsx:2100-2110`, `:2272-2284`. | `CURRENT / WORKING` implementation; retained-output flicker acceptance `UNVERIFIED` |
| Temporal compositing | Per-track 64x64 mask state, adaptive feather/multiband plan, and low/high-frequency separation are implemented (`temporal_compositing.py:119-220`; `procmgr_masking.py:426-484`, `:554-559`). | No normal React control. | `CURRENT / PARTIAL` — backend/config only and opt-in |
| Temporal quality control | Event-driven observation/correction is implemented and invoked before/after visual stages (`ProcessMgr.py:4879-4919`, `:5625-5718`; `temporal_quality.py:1-350`). | No normal React control. | `CURRENT / PARTIAL` — backend/config only and opt-in |
| Identity-specific detail | V2 detail is restored late, after merger and temporal low-band output, with visibility/temporal guards (`ProcessMgr.py:5568-5623`; `identity_detail.py`). | No normal React control. | `CURRENT / PARTIAL`: V1/no metadata is an explicit no-op; real V2 retention is not fully measured |
| Output upscale / interpolation | AI/classical upscale and RIFE/minterpolate run as post-passes in `app/post_swap.py` and `api.py:2867-2919`; resources are released before heavy upscale. | Controls at `FaceSwap.jsx:2022-2028`. | `CURRENT / WORKING` control path; visual output comparison `UNVERIFIED` |
| Codec, colorspace, audio | Raw BGR24 → codec → yuv420p, hardware-to-software fallback, audio mux, GIF conversion, and segmented finalization are implemented (`ffmpeg_writer.py:198-293`, `:313-345`; `core.py:1095-1141`). | Output controls are in Settings/FaceSwap; codec is configuration-driven. | `CURRENT / PARTIAL`: ordinary path implemented; odd-dimension color and long-run compatibility need validation |

## UI/backend wiring findings

### React controls confirmed to reach the implementation

`FaceSwap.jsx:387-401` builds the single `/api/swap` body from the settings
object. The preview builder at `FaceSwap.jsx:593-657` explicitly includes visual
controls, and `app/api.py:2312-2465` centralizes corresponding preview/run
helpers. `app/tests/test_settings_wiring.py` checks payload consumption,
preview/run agreement, neutral merger defaults, and helper invocation. The
supported-environment run for this audit passed those checks.

The visible controls with verified request-to-backend wiring are: swap model,
detector and thresholds, alignment refinement, swap mask strength, small-face
rescue, swapping steps, enhancer/adaptive profile, face distance, subsample size,
color mode, target appearance, blend ratio, detail transfer, merger operations,
mask engines/parser settings, manual masks, face scaling, jaw reshape, mouth/eye
restore, expression restore, enhancer alignment, post-enhance color match,
lipsync, temporal detection, keypoint/enhancer/mask stabilization, output upscale,
and interpolation. Wiring does not constitute visual acceptance.

### Visible controls with conditional or partial effect

| Control | Verified limitation | Status |
|---|---|---|
| Swapper’s own face mask strength | Only HifiFace/HyperSwap-like graphs emit the recognized mask. The cross-frame batcher deliberately sets `_swap_masks=None` because it cannot attribute returned masks to requests (`ProcessMgr.py:5072-5077`). | `PARTIAL` |
| Multi-angle source bank | Requires multiple source faces, pose records, and preferably FaceSet V2 metadata; otherwise selection falls back (`ProcessMgr.py:4816-4852`). | `PARTIAL` |
| 3D source pose matching | Only image-source swappers use the reconstructed crop; embedding-based swappers intentionally leave it unused (`ProcessMgr.py:4922-4939`). | `PARTIAL` by design |
| Frontalization | Only triggers beyond the configured pose threshold and can fail back to the unfrontalized crop (`ProcessMgr.py:5018-5042`). | `PARTIAL` |
| Enhancer alignment | Only applies when the selected enhancer publishes a different template and affine construction succeeds (`ProcessMgr.py:5235-5278`). | `PARTIAL` |
| Temporal controls | Video-only stateful behavior is not represented by a single-frame preview; `test_settings_wiring.py` classifies these as run-only rather than a preview omission. | `CURRENT / WORKING` wiring; visual effect `UNVERIFIED` |
| Identity detail | A nonzero setting without V2 detail data reports unavailable and leaves the crop unchanged (`ProcessMgr.py:5584-5592`; `identity_detail.py:56-60`). | `PARTIAL` |

### Backend/config controls not exposed in normal React UI

Verified backend/config fields with no normal React control were found for:

- `identity_detail_strength` (`app/settings.py:659`, `app/api.py:2499`, `:2760`);
- `temporal_compositing` and its six numeric controls (`settings.py:662-668`,
  `api.py:2398-2418`);
- `temporal_quality_control`, logging, history, and cache (`settings.py:670-673`,
  `api.py:2421-2436`);
- `target_conditioned_appearance_cache_size` (`settings.py:646`, `api.py:2371-2387`);
- `merger_clarity` (`settings.py` merger defaults/save fields, `api.py:2312-2332`,
  `procmgr_merger.py:94-129`).

These are not called broken: they are available through configuration, recipes,
or backend payloads where applicable, but are not presented as ready controls in
the normal React panel.

No React visual control was classified as outright `BROKEN` by the source/wiring
tests run in this audit. A control’s conditional no-op or fallback is recorded as
`PARTIAL` above. The repository does contain known model/target failures below.

### Explicit broken and missing states

- `BROKEN` for the historical RTX 3060 DMDNet validation scope: the recorded
  target run failed with `TypeError: 'NoneType' object is not subscriptable`;
  the current source now guards the missing landmark at
  `app/roop/processors/Enhance_DMDNet.py:139-149`, so this is a failed
  validation record, not proof that the current guard still fails.
- `MISSING` for a completed retained-output visual acceptance/promotion record:
  `app/output/phase16_validation/final_report.json` remains
  `OPEN_INCOMPLETE` with zero ready or complete runs. The visual pipeline is
  implemented, but a repository-level visual release decision is absent.
- `UNVERIFIED` remains the appropriate status where source and tests show a
  path exists but no retained-output or required-hardware evidence proves its
  visual result.

## Quality improvements already implemented

The following are present in the current code and are not future-state claims:

1. Model-specific alignment, normalization, output sizes, embedding modes, and RealSwap secondary composition (`FaceSwapInsightFace.py:44-342`, `:690-735`).
2. Corrected 5-point pose use for mouth/eye gates and mask routing (`ProcessMgr.py:4677-4745`).
3. Model-mask shape validation and mask preservation through regular and batched swap calls (`FaceSwapInsightFace.py:700-711`, `:919-956`).
4. Occluder threshold/blur, undersized-mask recovery, mask reuse, temporal mask smoothing, and per-face ownership (`procmgr_masking.py:1107-1197`).
5. Far-to-near multi-face paint ordering and overlap trimming (`ProcessMgr.py:4288-4348`).
6. Target-conditioned appearance with low-frequency lighting, bounded chroma, dark-scene tiers, and temporal EMA (`procmgr_color.py:135-206`, `appearance_conditioning.py:161-216`).
7. Enhancer alignment, non-finite guards, GFPGAN collapse guard, and model-specific precision routing (`ProcessMgr.py:5220-5318`, `Enhance_GFPGAN.py:50-126`, `precision_policy.py:108-152`).
8. ROI paste-back, multiband temporal composition, and bounded float32/uint8 conversion (`procmgr_masking.py:491-573`, `temporal_compositing.py:119-220`).
9. Merger post-ops with neutral-value no-op behavior, signed sharpening, and plate-derived motion/grain/degrade references (`procmgr_merger.py:11-34`, `:82-237`).
10. Source identity detail restoration is late in the crop pipeline and respects visibility/temporal state (`ProcessMgr.py:5568-5623`).
11. Segmented, resumable encoding and deliberate-stop finalization (`segment_writer.py:120-395`, `core.py:1081-1141`).

## Precision-sensitive visual operations

| Operation/model | Verified policy | Status / risk |
|---|---|---|
| GFPGAN | Forced FP32 TensorRT providers; repository measurement records finite FP16 collapse to near-grey and the guard catches residual collapse (`Enhance_GFPGAN.py:50-69`, `:113-126`). | `WORKING` guard; 4070 evidence, 3060 unverified |
| GPEN 1024/2048 | Forced FP32 TensorRT providers because FP16 overflow is documented in `Enhance_GPEN.py:99-109`. | `WORKING` policy; end-to-end visual result unverified |
| GPEN smaller / CodeFormer / RestoreFormer++ | Routed through `precision_policy.providers_for`; policy matrix distinguishes safe, candidate, unsafe, and unknown families (`precision_policy.py:108-152`, `:328-405`). | `CURRENT / PARTIAL`: policy explicit, but not every model/precision pair has physical evidence |
| Face swapping | Policy identifies raw FP16 as unsafe and mixed as a candidate; `_swap_providers` applies it (`precision_policy.py:144-147`; `FaceSwapInsightFace.py:443-482`). | `CURRENT / WORKING` routing; output-quality validation is unverified for every model |
| CPU color/mask/merger/blend | These paths use OpenCV/numpy float32 and return bounded uint8; examples are `procmgr_color.py:116-133`, `procmgr_masking.py:1222-1235`, and `procmgr_merger.py:224-237`. | `WORKING` implementation; CPU memory/throughput is separate from visual quality |
| Encoding | Raw BGR24 is converted by FFmpeg to `yuv420p`; codec-specific CRF/CQ logic is separate from inference precision (`ffmpeg_writer.py:213-245`, `:286-287`). | `CURRENT / PARTIAL`; odd-dimension colorspace behavior is unvalidated |

## Regressions and potentially dangerous conditions

These are repository-backed risks, not speculative optimization requests:

- The existing Phase 16 report is incomplete, so no final visual promotion is justified. This is recorded in `app/output/phase16_validation/final_report.json` and `docs/development/MASTER_PLAN.md:19-29`.
- The latest repository handoff records a 4070 two-face smoke with 120/120 output frames, 240 face rows, and zero wrong FaceSet applications, but also records a stalled full matrix after CUDA stream/RealSwap fallback warnings. The stalled attempt is explicitly not accepted as a benchmark (`docs/PHASE_HANDOFF.md:459-466`).
- The physical host used for this audit is an RTX 4070. No RTX 3060 is installed; historical repository records remain the only 3060 evidence. Those records show the strict `<2.5 GB` RSS gate failing at 3.73 GB and a DMDNet `NoneType` error (`docs/SECOND_GPU_VALIDATION.md:121-129`).
- Cross-frame batching drops swapper-provided masks by contract, so enabling the visible swap-mask slider cannot affect those frames (`ProcessMgr.py:5072-5077`).
- Resume identity does not include all writer options: `SegmentedVideoWriter` stores `preset`, `bitrate`, `threads`, `ffmpeg_params`, and `colorspace` in `_writer_options` at `segment_writer.py:145-150`, but the resume identity only records codec/CRF and dimensions at `:153-166`. Resuming after changing one of those options could mix segments with different encoding behavior. This is an output-safety risk and is not fixed in Stage 2A.
- `FFMPEG_VideoWriter._build_cmd` uses `elif` for the colorspace filter after the odd-dimension scale branch (`ffmpeg_writer.py:270-281`). Odd-sized inputs are therefore not verified to receive the same explicit colorspace conversion as even-sized inputs.
- Temporal compositing, temporal QC, and V2 identity detail are implemented but opt-in/config-driven and not exposed in the normal React panel. Treating their presence as visual acceptance would overstate readiness.

## Desired future state

This gate does not implement future behavior. The eventual visual gate must:

1. complete the 17-clip/425-row Phase 16 matrix with retained outputs;
2. review identity, boundary, color, low-light, occlusion, expression, detail, and temporal consistency on retained frames;
3. repeat comparable evidence on both required hardware targets;
4. decide explicitly whether backend-only visual controls should remain hidden or receive validated React controls; and
5. close the encoder-resume and odd-dimension colorspace validation risks before treating output recovery/color handling as fully verified.

## UNVERIFIED / UNKNOWN

- No current report proves a globally best enhancer, swapper, mask engine, color mode, or sharpening profile across required footage.
- No physical RTX 3060 visual run was possible in this session.
- No current retained-output review proves that temporal identity, temporal compositing, temporal QC, expression restore, identity detail, or the swapper mask slider improves visual quality without regressions across all required scenes.
- No repository evidence establishes that the odd-dimension FFmpeg colorspace branch is visually equivalent to the even-dimension path.
- No repository evidence establishes that a resumed render after changing colorspace/preset/bitrate/extra FFmpeg parameters is safe.
- The current local configuration values are not evidence of the separately recorded laptop look (`blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`, `stabilize_enhancer_strength=0.6`); those remain acceptance constraints from hardware records, not observed local output.

## Verification performed for this audit

- `app/env/Scripts/python.exe -m pytest` on focused visual, mask, enhancer, temporal, settings-wiring, and UI-default suites: **220 passed, 2 warnings, 3 subtests passed**.
- `app/env/Scripts/python.exe -m pytest -q` for the full repository suite:
  **1730 passed, 1 skipped, 4 warnings, 599 subtests passed in 55.33s**.
- Host probe: **NVIDIA GeForce RTX 4070**, 12282 MiB reported by `nvidia-smi`, driver 616.56; PyTorch CUDA available; ONNX Runtime reported TensorRT, CUDA, and CPU execution providers.
- No physical RTX 3060 validation was performed in this session.

## Source basis

Primary implementation: `app/roop/ProcessMgr.py`, `app/roop/procmgr_masking.py`,
`app/roop/procmgr_color.py`, `app/roop/procmgr_merger.py`,
`app/roop/procmgr_tracking.py`, `app/roop/processors/FaceSwapInsightFace.py`,
enhancer processors, `app/roop/precision_policy.py`,
`app/roop/temporal_compositing.py`, `app/roop/temporal_quality.py`,
`app/roop/identity_detail.py`, `app/roop/ffmpeg_writer.py`,
`app/roop/segment_writer.py`, `app/roop/core.py`, `app/post_swap.py`, and
`app/api.py`.

UI and contracts: `react-ui/src/components/FaceSwap.jsx`,
`react-ui/src/components/Settings.jsx`, `react-ui/src/components/settingsCatalog.js`,
`app/settings.py`, `app/tests/test_settings_wiring.py`,
`app/tests/test_merger_postops.py`, `app/tests/test_temporal_compositing.py`,
`app/tests/test_temporal_identity.py`, `app/tests/test_temporal_occlusion.py`,
`app/tests/test_enhancer_fp16_collapse.py`,
`app/output/phase16_validation/final_report.json`, `docs/PHASE_HANDOFF.md`, and
`docs/SECOND_GPU_VALIDATION.md`.
