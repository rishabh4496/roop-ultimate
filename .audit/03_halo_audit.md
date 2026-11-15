# Bug #2 — Double face halos on side angles (high yaw / profile)

**Scope.** Static call-path audit of the live render path, plus six read-only measurements that call this repository's own functions (`blur_area`, `paste_upscale`, `_color_transfer_lct`, `apply_sharpen`, `EnhancerStabilizer`). No source file was changed. `.audit/01_pipeline_map.md` and `.audit/02_disorientation_audit.md` were read first.

**Not done.** No real clip was available, so nothing here is a frame-level reproduction. Every number below comes from synthetic input fed to the real code. Each finding is tagged:

- **[MEASURED]**: the repo's real function was run on synthetic input and the number is quoted.
- **[CODE]**: read from a named `file:line`, not executed end to end.
- **[UNVERIFIED]**: plausible, but no evidence yet. Under `AGENTS.md` ("Never mark a stage … on a reading") it is neither shipped nor ruled out.

**Rules applied.** `AGENTS.md` Part 1: classify unverified behaviour honestly, treat a negative result as a deliverable, and `config.yaml` is per-machine, so do not "realign" the two machines' look settings. Part 2 (Pinokio launcher rules) does not apply.

**Live config read** (`app/config.yaml`, gitignored, this RTX 4070 machine): `swap_model: hyperswap`, `selected_enhancer: Restore Ultra`, `mask_engine: RealityUX`, `subsample_upscale: 256px`, `face_mask_blend: 25`, `blend_ratio: 1`, `color_match_after_enhance: true`, `stabilize_enhancer: true` at strength `0.6`, `stabilize_mask: true` at `0.5`, `merger_sharpen: 1`, `merger_clarity: 1`, `merger_hist_match: 0.3`, `merger_grain_match: 0.2`, `swap_model_mask_strength: 0`, `temporal_compositing: false`, `auto_bypass_restorer_native_swap: true`.

---

## 1. Verdict in one table

| # | Your hypothesis | Verdict | One-line reason |
|---|---|---|---|
| 1 | Swapper pastes with Mask A, then the enhancer re-crops wider and pastes again with Mask B | **Refuted as worded. A different two-matte structure exists and is the real cause.** | The swapper never pastes into the frame, and the enhancer works in crop space. There is one inverse warp and one paste. The two concentric edges come from a crop-space matte baked into pixels (A) plus a separate frame-space ellipse∩hull feather (B), then smeared further by temporal blending. §3, §4 |
| 2 | Square crop overflows past the jaw into background; the paste mask is a static box/ellipse | **Partly confirmed.** | The paste matte includes a static inscribed ellipse and an outward-dilated, outward-feathered hull. The crop does contain more background at high yaw. The crop does not "overflow", though. The matte tail escapes the crop footprint (§3.F4). |
| 3a | Occluder/parser mask has coordinate or interpolation mismatch (`INTER_NEAREST` vs `LINEAR`) | **Refuted.** [CODE] | The mask is computed on the same aligned crop it is applied to. No `INTER_NEAREST` on the main mask path. |
| 3b | Erosion and blur are fixed constants, not scaled to resolution and yaw | **Partly confirmed.** [CODE] | The feather scales with matte extent, not with yaw. Several mask-engine constants are fixed in label pixels. |
| 3c | Feather blurs outward past the eroded boundary | **Confirmed.** [MEASURED] | Erosion is a quarter of the feather radius, and the hull is dilated outward first. §3.F2 |
| 4a | Colour transfer uses whole-crop statistics, shifting skin luminance | **Confirmed as code, conditional as a defect.** [MEASURED] | Harmless when the swap output keeps the plate's background. Large when it does not. §3.F5 |
| 4b | `warpAffine` border mode causes bright/dark rings | **Narrower than stated.** [MEASURED] | `BORDER_REPLICATE` is correct for the aligned-crop input. It is wrong at the paste, where the matte tail reaches outside the crop. §3.F4 |

**Ranked causes** (evidence strength first, then fit to "side angles, double ring"):

| Rank | Cause | Evidence | Why it favours profiles |
|---|---|---|---|
| 1 | Crop-space temporal smoothing is yaw-blind and runs on the already-masked crop → stacked ghost contours | [MEASURED] mechanism; [CODE] it runs in video renders | A yaw turn barely moves the 5-point centroid, so the stabiliser stays heavy exactly while the contour moves fastest in crop space |
| 2 | Paste feather: outward hull dilation + outward Gaussian tail, 2× wider at the live `face_mask_blend: 25` than at the calibrated 12 | [MEASURED] on the real `blur_area` | At high yaw the crop is mostly background, so the band lies over background |
| 3 | The "restore original" pixels are not original at paste time (double resample + whole-crop post-ops after the mask is baked) | [MEASURED] resample loss; [CODE] post-op order | Larger cheek/background area inside the pasted band |
| 4 | Matte tail escapes the crop footprint into `BORDER_REPLICATE` streaks | [MEASURED] | Present at all poses; most visible where background is plain and the crop is tight |
| 5 | Whole-crop colour statistics when swap background ≠ plate background | [MEASURED] synthetic; real swap background [UNVERIFIED] | Background share of the crop grows with yaw |
| 6 | Two segmenters (XSeg + injected occluder) disagreeing at the silhouette; 106-point hull reliability on profiles | [UNVERIFIED] | Profile is where both are weakest |

---

## 2. The live sequence (what actually runs per face)

Processor order is the dict order built in `roop/core.py:960-1005` (`faceswap`, then the enhancer, then the mask stage "inserted last", per the comment at `core.py:975-976`). The mask engine `mask_occluder` is appended after RealityUX by default (`roop/occlusion_mask.py:103-121`, default on).

`ProcessMgr.process_face` (`roop/ProcessMgr.py:5583-7500`):

| Step | Site | What happens | Space |
|---|---|---|---|
| 1 | `:5736` `canonicalize_face_alignment` (`face_analyser.py:695-735`) | Five-point similarity `M`, `warpAffine(INTER_CUBIC, BORDER_REPLICATE)` into a `subsample_size` crop (256) | frame → crop |
| 2 | `:6307-6421` | Swap net runs on the crop. Output `fake_frame`, crop space, still includes whatever the net produced outside the face | crop |
| 3 | `:6456-6461` | **Colour transfer 1**: `apply_color_transfer(fake_frame, aligned_img)`, whole crop | crop |
| 4 | `:6584-6763` | **Enhancer** (Restore Ultra) on the pre-mask swap crop. `force_align=True` + template `ffhq_512` ≠ `arcface` → `_realign` path: swap crop warped into FFHQ framing, ring filled **from the plate** (`:6629-6673`), restored, warped back into swap-crop space (`:6726-6737`) | crop → FFHQ crop → crop |
| 5 | `:6478-6549` | **Mask stage.** `process_mask` per engine (RealityUX, then occluder). `_img_mask` is baked into the pixels: `result = target·(1−mask) + aligned_img·mask` for **both** `fake_frame` and `enhanced_frame`, using the **same** mask (`_composite_mask`, `procmgr_masking.py:1744-1782`) | crop |
| 6 | `:6776-6782` | `EnhancerStabilizer.apply(enhanced_frame)`: temporal blend of the **already-masked** crop | crop |
| 7 | `:6797-6804` | `HighFrequencyFlowStabilizer`: flow-warped, high band only. Switched on when `stabilize_hf_texture` (live: true, weight 0.2) and an enhancer ran (`ProcessMgr.py:1271-1280`). Unlike step 6 it removes head motion first, so it is not the ghost source | crop |
| 8 | `:6877-6890` | **Colour transfer 2** (`color_match_after_enhance: true`) on the masked crop | crop |
| 9 | `:6916-6924` | `apply_merger_post`: histogram match, clarity, sharpen, grain on the masked crop | crop |
| 10 | `:7219-7224` | `fake_frame` resized to 512 (cubic) | crop |
| 11 | `:7283-7288` → `paste_upscale` (`procmgr_masking.py:629-919`) | **The only paste.** Builds matte **B** = inscribed ellipse ∩ landmark hull, blurred (`blur_area`), warps the crop with `IM`, `BORDER_REPLICATE`, alpha-blends against the plate. Default is plain linear alpha (`:897`); the multiband path needs `temporal_compositing` (off) | crop → frame |
| 12 | `:7330+` | Mouth, eye, and expression restores in frame space (all off or faded at yaw in the live config) | frame |

`.audit/01_pipeline_map.md` Stage 8 describes Laplacian multiband compositing and Stage 5/7 describe `generate_composite_mask` and mask-weighted Reinhard. Those are not on this path (§7).

---

## 3. Findings

### F1 — Crop-space temporal smoothing is yaw-blind and runs on the masked crop (rank 1) [MEASURED + CODE]

**Where.** `EnhancerStabilizer.apply` (`roop/one_euro.py:564-616`), called at `ProcessMgr.py:6776-6782` *after* the mask is baked (step 5 → 6). `MaskStabilizer` (`one_euro.py:619+`, built at `ProcessMgr.py:1197-1251`) blends the mask the same way.

**Mechanism.**
- The blend factor comes from `motion = centroid displacement of the 5 keypoints, in frame space / face size` (`:597`). That is translation only. A yaw turn in place keeps the centroid nearly fixed, so `motion ≈ 0`.
- With `stabilize_enhancer_strength: 0.6` the base cutoff is `0.4·(1−0.6)+0.02 = 0.18`, giving `alpha = r/(r+1)` with `r = 2π·0.18` → **a ≈ 0.53**: about 47% of the previous crop is kept. The floor is 0.15 (`:599`).
- To reach `a ≥ 0.9` the centroid would have to move ≥ 15.6% of the keypoint span per frame.
- The previous crop is a **composited** crop: swap skin inside, restored original outside, joined by the previous frame's contour. In crop space the contour moves during a turn even though the keypoints are registered. Blending two such crops leaves the old contour behind as a ghost.
- `MaskStabilizer` is built with `motion_beta=0` on purpose (`ProcessMgr.py:1200-1216`). The comment's reasoning, that the aligned crop "already cancels head translation/rotation/scale", holds for in-plane motion but not for out-of-plane yaw. The mask's contour moves in the crop during a turn, and the EMA (strength 0.5 → `a ≈ 0.58`) lags it.

**Measured** (`.audit/03_halo_experiments/e5_stabilizer_yaw_ghost.py`, the real `EnhancerStabilizer`, 512 crop, face half-width 200→85 px over 9 frames, keypoint centroid held fixed):

| `stabilize_enhancer_strength` | max \|stabilised − current\| on a 100-level skin/background step | contour edges visible along the centre row |
|---|---|---|
| 0.6 (live) | **46 levels**, every turn frame | up to 4-5 stacked contours (e.g. frame 8: edges at x≈353, 369, 388, 408) |
| 0.25 (`settings.py` default) | 33 levels | up to 3 stacked contours |

The trailing contours are the previous frames' silhouettes, each persisting with weight ≈ (1−a)ⁿ. **Caveat:** the simulation fixes the centroid, which is the worst case. A real turn moves the centroid somewhat and opens `a` partially.

**Why this fits the report.**
- It exists only when the enhancer is on (`enhanced_frame is not None`, `:6777-6778`) and only in video renders. `_stab_active` is set True on the render paths (`procmgr_batch.py:364,376,660`; `ProcessMgr.py:2422`) and reset False in `initialize` (`:1286`).
- A single-frame preview should therefore **not** show it. That is a free discriminator (§6).

**Corroboration in the repo:** `docs/SESSION_LOGS.md` (2026-08-22 follow-up) lowered `stabilize_enhancer_strength` 0.5 → 0.25 "to prevent over-smoothing". The live config is back at 0.6.

### F2 — Paste feather reaches outward, and the live value is 2× the calibrated one (rank 2) [MEASURED]

**Where.** `paste_upscale` builds the matte at `procmgr_masking.py:641-681`; `create_landmark_mask` dilates the hull outward (`:1157-1161`, `expand_px = int(√(face_h·face_w)·blend/400)`); `blur_area` erodes by only a quarter of the feather radius and then Gaussian-blurs symmetrically (`:1109-1125`, `blend_px = mask_size·blend/200`, `erosion_px = max(1, blend_px//4)`).

**Measured** with the real `blur_area` on a disc standing in for hull∩ellipse (`e1_*.py`; disc diameter D ≈ face size S; 0.98/0.02 = alpha crossings):

| D (px) | blend | 0.98 point (px inside matte edge) | 0.5 point | 0.02 point (px outside matte edge) | band 0.98→0.02 |
|---|---|---|---|---|---|
| 200 | 12 | 11 | −3 | +5 | 16 |
| 200 | **25** | 22 | −6 | +10 | **32** |
| 300 | 12 | 16 | −4 | +8 | 24 |
| 300 | **25** | 33 | −9 | +14 | **47** |
| 450 | 12 | 23 | −6 | +11 | 34 |
| 450 | **25** | 49 | −14 | +21 | **70** |

Add the hull dilation: S = 300 → 9 px at blend 12, **18 px** at blend 25. Measured from the landmark hull edge, the 300 px face's transition band is **−7 … +17 px at blend 12** and **−15 … +32 px at blend 25**.

**Reading.** The feather is not inward-only. Alpha is already ≈ 0.5 at 9 px *inside* the matte edge, but the matte edge itself sits 18 px outside the hull, and the tail runs another 14 px. At the live setting the paste band is about twice as wide as at the calibrated one, and it lies over the original cheek or background. The code comment at `:1112-1119` shows erosion was cut from `blend_px//2` to `//4` to stop profiles from losing 15-20 px of contour ("second nose"). The cost moved to the outer edge.

**History.** `docs/SESSION_LOGS.md` 2026-08-22 (lines 12-30 and 150-163) records this exact symptom, "concentric double-edge seam along jawline and perimeter", fixed by `face_mask_blend: 12` and `swap_model_mask_strength: 0`. The same log lists `face_mask_blend: 30` as "the exact halo the session claimed to fix". The live config is 25. The `settings.py` default (code, shared by both machines) is 12 (`settings.py:1150`). Per `AGENTS.md` the live `config.yaml` is per-machine and is deliberately not realigned, so this is a flag for the owner rather than a change to make.

### F3 — "Restored original" pixels are not original at paste time (rank 3) [MEASURED + CODE]

Two separate mechanisms.

**(a) Resample chain.** Where the mask says "keep original", the crop holds `aligned_img` (cubic warp into 256), upsampled to 512 by `cv2.resize(frame, …)` with the **default linear** interpolation (`procmgr_masking.py:1765`), then warped back with cubic (`:799-802`). Measured on a 1/f synthetic texture (`e1_*.py`, real `cv2` chain, same interpolations):

| crop footprint in the frame (px) | crop px per frame px | MAE vs plate (levels) | high-frequency energy kept |
|---|---|---|---|
| 160 | 1.60 | 2.9 | 66% |
| 256 | 1.00 | 4.1 | 47% |
| 384 | 0.67 | 6.8 | 29% |
| 512 | 0.50 | 8.4 | 21% |
| 768 | 0.33 | 10.2 | 9.5% |
| 1024 | 0.25 | 11.3 | 4.7% |

Frontal IOD is 23.3% of the aligned crop (measured in the `_mask_crop_box` docstring, `procmgr_masking.py:1195-1199`), so a face ≈ 250 px wide has a footprint of roughly 500 px, i.e. ~21% HF retained in the synthetic test. Real footage already has camera blur, so absolute losses will be smaller; the direction is the point. Inside the 47 px feather band the pasted copy is softer than the plate it is blended over.

**(b) Post-ops after the mask is baked.** These all treat the crop as one image and touch the restored-original pixels too: `EnhancerStabilizer` (`:6776`), colour transfer 2 (`:6877`), `apply_merger_post` hist-match / clarity / sharpen / grain (`:6916`, `procmgr_merger.py:180+`). None takes the mask.

**Refuted sub-claims (negative results, kept on record):**
- Colour transfer 2 changes the untouched background by only 0.5-0.6 LAB levels when the swap background matches the plate (`e1_*.py`, E3).
- `apply_sharpen` rim at the face contour is small: ≤ 0.1× the step height (2.8 levels on a 30-level step; 5.6 on 60; clamp 18 never binds) (`e3b_e4_*.py`, E4). Not a visible halo source on its own.

### F4 — The matte tail escapes the crop footprint into `BORDER_REPLICATE` (rank 4) [MEASURED]

`paste_upscale` warps the crop with `borderMode=cv2.BORDER_REPLICATE` (`:799-830`), while the matte is warped with `borderValue=0` (`:606`, `:671`) and then Gaussian-blurred outward. Where the ellipse (inscribed in the crop) touches the footprint edge, the blurred tail has non-zero alpha **outside** the footprint, where the pasted colour is a streak of the crop's edge pixels.

**Measured** with the real `paste_upscale` (`e6_*.py`: flat grey plate, 420 px footprint, blend 25, no landmark hull; crop edge deliberately a different colour): **7,651 plate pixels changed outside the footprint, reach 17 px beyond it on both axes, peak change 25 levels.** With a hull present the leak is limited to where hull∩ellipse meets the footprint edge, mainly the forehead/top, where `landmark_hull`'s forehead extension (`:488`, `0.6·face_h` above the brow) pushes the hull to the crop top. Contrast in a real frame will be lower than the synthetic worst case.

`BORDER_REPLICATE` is **correct** for the aligned-crop input warp (`face_analyser.py:726,730`), which feeds a network and must stay in-distribution. It is wrong for the output paste.

### F5 — Colour-transfer statistics span the whole crop (rank 5, conditional) [MEASURED + UNVERIFIED]

`_color_transfer_lct` (`procmgr_color.py:495-553`) samples every 16th pixel of the **whole** source and target crop (`:524-526`); `_color_transfer_rct` (`:398-418`), `mkl`, `idt` do too. `_skin_region_mask` (`:41-74`) and `_soft_mask` use a static centred ellipse prior (`rx = 0.43·w`), independent of pose.

**Measured** (`e1_*.py` E3, `e3b_e4_*.py` E3b; real `_color_transfer_lct`; skin 16 LAB-L levels brighter in the swap than in the plate; "face share" is the elliptical skin share of the 256 crop):

| swap-net background | plate background | pose (face share) | residual skin ΔL after LCT |
|---|---|---|---|
| = plate's | any | frontal 52% / yaw~45 34% / profile 21% | −0.2 / 0.0 / +0.1 (**fixed**, all poses) |
| mid-grey fill | dark | frontal / profile | −3.0 / −4.5 |
| mid-grey fill | bright | frontal / profile | **+27.3 / +44.8** (a/b also −9…−15) |

So the dilution I first suspected does not happen when the swapper keeps the background. It happens when the swap output's background differs from the plate's, and then it scales with the background share, which rises with yaw. `paste_upscale` itself documents that "the swap model put grey fill pixels" outside the face on angled faces (`:675-676`). **What hyperswap/RealSwap actually emits outside the face is [UNVERIFIED]**; measuring it is one of the §6 steps. If it is faithful, F5 is a non-issue.

### F6 — Segmenter and hull reliability on profiles (rank 6) [UNVERIFIED]

- **Two contours.** The occluder is appended after RealityUX by default and the product of keeps is taken (`ProcessMgr.py:1344-1371`, comment at `:1346-1350`). The occluder threshold is hard (`> 0.35`, `procmgr_masking.py:1590-1593`, edge ≈ 5 frame px via `_edge_blur_kernel`), XSeg's ramp is soft. On profiles the two silhouettes may disagree by several px, and the gap shows as a band of original skin just inside the swap, a second ring. `ROOP_OCCLUSION_MASK=0` isolates it.
- **Hull.** `landmark_hull` (`:430-510`) is a convex hull over the 106-point set. A profile silhouette is not convex, so it can bridge the notch between brow and nose and the gap between nose and chin with background. The hidden-side contour points are predictions. Neither effect is measured.
- **Hull lag vs mask.** The hull is drawn from `landmark_2d_106` after `AdaptiveLandmarkSmoother` (enabled by `stabilize_face` + `stabilize_landmarks`, `ProcessMgr.py:1268-1270`, applied at `:3811-3827`), while XSeg's contour is computed fresh on the current crop. During a fast turn the two can disagree by the smoother's lag. Not measured.
- **Static pose-blind ellipses.** `_face_floor_ellipse` (`:188-221`), axis-aligned and centred between eye and mouth midpoints, drives `_recover_undersized_mask`, which can only pull toward swap (`:255`). It is gated to a gross failure (`_MASK_RECOVER_TRIGGER 0.5`), so it matters only when the model under-segments.
- **XSeg detail worth recording.** `Mask_XSeg.Run` zeroes everything under 0.1 (`Mask_XSeg.py` `result[result < 0.1] = 0`), which leaves a ~0.1 step at the mask edge. `RealityUX.Run` blurs accessory labels with a fixed `sigmaX=3` in label pixels (`Mask_RealityUX.py:342`) while its other kernels scale by `h/_REF_MASK_PX`. Minor.

### Refuted / not a cause (kept so they are not re-attempted)

| Claim | Evidence |
|---|---|
| Mask coordinate mismatch between the mask and the crop | `process_mask` receives the same `aligned_img` the crop was cut from (`ProcessMgr.py:6502-6511`); masks are resized with `INTER_LINEAR` (`procmgr_masking.py:1758`, `Mask_RealityUX.py:346`). `INTER_NEAREST` appears only on the glasses/mouth labels (`Mask_RealityUX.py:451`) and the realign coverage mask (`ProcessMgr.py:6661`, followed by a σ=2 blur). |
| Adaptive/profile alignment shifting features | `canonicalize_face_alignment` is a plain five-point similarity at every yaw (`face_analyser.py:695-735`, docstring explains why). The profile solvers in `adaptive_alignment_matrix` are not on this path. |
| Swap model's own mask clashing with RealityUX | `swap_model_mask_strength: 0` (config line 197). |
| Laplacian multiband compositing creating rings | Not executed: `_composite_plan` is None unless `temporal_compositing` is on (it is off) and `ROOP_COMPOSITING_ENGINE` defaults to 0. |
| `output_face_scale` growing the paste | Default 0.0, `_scale_paste` is a no-op (`:1022-1023`). |

---

## 4. Direct answers to the four numbered asks

**1. Dual-pass paste-back ("double halo").**
- There is no second paste. The enhancer runs on the swap crop in crop space (`ProcessMgr.py:6703-6714`) and returns to swap-crop space (`:6726-6737`). `paste_upscale` is called once (`:7283-7288`).
- The "wider box" exists only for Restore Ultra (`force_align`, FFHQ template). It is wider by design, filled from the **plate** (`:6653-6670`), and discarded on the way back, because the swap crop sits inside the FFHQ field. It does not produce a second edge.
- Both `fake_frame` and `enhanced_frame` are composited through the **same** `_img_mask` (`:6545-6549`), so there is no A-vs-B mismatch between them. With `blend_ratio: 1` the paste uses only `enhanced_frame`.
- The two concentric edges come from the crop-space matte A (baked in) vs the frame-space matte B (ellipse∩hull, feathered), plus temporal ghosting of the baked result (F1, F2, F3).

**2. Square-crop overflow on side angles.**
- The `_mask_crop_box` docstring (`procmgr_masking.py:1195-1199`) reports the interocular distance as a share of the aligned crop: frontal 23.3%, yaw 40 18.6%, yaw 60 12.2%. So the face narrows inside the crop as the head turns, and the crop fills with background. [CODE: figures quoted from that docstring, not re-measured.] My explanation for why (the eye-to-mouth span, which does not foreshorten with yaw, dominates the least-squares similarity scale) is an inference from how `estimate_norm` fits five points; I did not test it.
- The paste matte B is a static inscribed ellipse ∩ a dilated convex hull, not the skin/occluder contour (`procmgr_masking.py:641-681`). Where hull and true silhouette differ, B pastes original-pixel copies (F3) with the F2 band.

**3. Mask engine defects.** Coordinate alignment: refuted. Fixed-vs-dynamic kernels: the paste feather scales with matte extent but not with yaw; XSeg threshold and RealityUX accessory blur are fixed. Outward feather: confirmed (F2).

**4. Colour and border bleed.** Whole-crop statistics: confirmed, harmless unless swap and plate backgrounds differ (F5). Border mode: correct at input, wrong at paste (F4).

---

## 5. Unified single-pass, aligned masking — fix design

**Design principle.** Today the "keep original" decision is made twice, in two spaces, with a lossy bake between them. Make it once.

### 5.1 Target data flow

```
swap crop ──► enhance ──► colour / merger / stabilise  ──┐   (content only: no original pixels in it)
                                                         ├─► ONE paste: warp [content·α , α] with BORDER_CONSTANT 0
A_final (product of keeps, ownership, roto, manual) ─►  α │   out = plate·(1−α_f) + premult_content_f
                                                         ┘
```

1. **One alpha field, built once, never baked into pixels.**
   - `α_crop = 1 − A_final` at the output resolution, where `A_final` is the existing product of keeps (RealityUX × occluder × ownership × roto-inclusion × manual).
   - Delete the `_composite_mask(_img_mask, aligned_img, …)` calls at `ProcessMgr.py:6545-6549` and `:7172-7175` as bakes. `_img_mask` is kept and carried to the end.
2. **Content stays pure.**
   - Run colour transfer 1 and 2, `apply_merger_post`, `EnhancerStabilizer` and `HighFrequencyFlowStabilizer` on swap/enhanced content only. The stabiliser then blends face interior; the contour comes from the *current* frame's α, so F1 cannot leave ghost silhouettes.
   - Colour statistics become α-weighted (soft weights from `α_crop`, replacing the static `_skin_region_mask`/`_soft_mask` ellipse): kills F5. Hist-match uses the same weights.
3. **Paste once, with premultiplied alpha.**
   - In `paste_upscale`, warp `content·α` and `α` together with `borderMode=BORDER_CONSTANT, borderValue=0`, using the same `IM` and the existing ROI canvas (`_matte_roi`, `:586-627`). The plate is read directly for everything outside α, so nothing outside the swap is resampled (kills F3a) and nothing outside the footprint is touched (kills F4).
   - Output: `out = plate·(1 − α_f) + warp(content·α)`.
4. **One feather, inward only.**
   - Replace `ellipse ∩ dilated hull → erode(blend_px//4) → Gaussian(blend_px)` with: `α' = α · smoothstep(d / w)`, where `d` is the inside distance to the boundary of `A_final > 0.5` (the `soft_distance_matte` path in `temporal_smoother.py` already exists for 'distance' edge mode).
   - Width `w = clamp(k · S_frame_px · g(yaw), w_min, w_max)`, in frame pixels, so a profile (narrow matte) gets a proportionally narrow band. The default is calibrated to the profile that was already measured halo-free (`face_mask_blend: 12` equivalent).
   - The hull and the ellipse become **clips only**: `α'' = α' · clip` where `clip` is the hull eroded or dilated by ≤ 2 px (never the 6% outward dilation). This keeps the protection they were added for (swap-net grey fill, over-segmenting XSeg) without a second feather.
5. **Temporal handling that can see yaw.**
   - Measure motion in crop space, not frame space: `Δα` contour displacement (or `Δyaw` from `solve_pose_5pt`) opens the blend. Optionally warp the previous α along the existing flow code (`mask_flow_warp`) before the EMA.
   - Stabilise **content** and **α** separately; never stabilise a composite.

### 5.2 Smaller, independent steps that can ship before the full redesign

| Step | Change | Fixes | Risk |
|---|---|---|---|
| S1 | In `paste_upscale`, multiply the matte by a footprint coverage mask `W = warpAffine(ones, IM, border 0)` (or switch the paste warp to `BORDER_CONSTANT` plus premultiplied alpha) | F4 | Bit-identical inside the footprint; only the outside leak changes. Safest first step |
| S2 | Gate `EnhancerStabilizer`/`MaskStabilizer` blend on yaw rate or crop-space contour change; open `a` toward 1 when it is large | F1 | Reduces anti-flicker on turning faces by design; counterbalanced A/B needed |
| S3 | Reorder the stage: compute `_img_mask`, run all content ops, bake **last** (just before the resize to 512 at `:7219`) | most of F1, F3b | Moves one block; needs the A/B and the performance regression suite |
| S4 | Inner-only feather + 2 px hull clip | F2 | Changes look at every pose; calibration against 12 required |
| S5 | α-weighted colour statistics | F5 | Needs F5's real-footage check first |
| S6 | Owner decision: live `face_mask_blend` 25 vs calibrated 12 and `stabilize_enhancer_strength` 0.6 vs 0.25 (per-machine config, `AGENTS.md`) | F1, F2 mitigation, no code | Do not "realign" without the owner; the numbers above are the evidence |

### 5.3 Implementation constraints from `AGENTS.md` and the repo's own history

- New logic goes in a **flat** module (`roop/unified_matte.py`), **not** a `roop/core/` package (that shadowed `core.py` and broke `import roop.core`, Stage 7 incident).
- Any new setting: `app/settings.py` is the single source (`_load()`, `save()`, `UI_SETTINGS` if exposed, `ENV_SETTINGS`), then `python tools/gen_settings.py` and commit both generated files. Grep that something reads it.
- CPU-side work only, ROI-bounded; no change to precision or CUDA policy. Distance transform at crop resolution is small but must be timed. "The pipeline is GPU-bound": a change moves the clock only by removing GPU work, so state it as neutral and prove it.
- Ship the fix, not the flag: land behind `ROOP_UNIFIED_MATTE`, then flip the default after the A/B and remove the flag.
- Benchmarking rules: 600 frames minimum, counterbalanced A/B, null control first, `--threads 20`, pin `ROOP_STAB_CHUNK_MB` (render pixels depend on free RAM), record `faces_seen`. The pixel noise floor is 0.7142/255 mean, 22/255 max.
- Add a synthetic yaw-sweep test that counts contour edges along face normals and the pixel energy in the ring outside the silhouette, plus geometry tests: `α == 0` outside the footprint, `α ≤ A`-derived, monotone inner feather, plate pixels bit-exact where `α == 0`. Per `AGENTS.md`, a test that stubs the very thing being claimed does not count.

---

## 6. What to measure on a real clip before coding

None of this has been run on footage. These use existing toggles, so no code changes are needed.

| Toggle / probe | If the double halo disappears or shrinks, it points to |
|---|---|
| Preview the problem frame via `/api/preview`, then render the same frame in a video run | F1 (stabilisers are inactive in single-frame preview) |
| `stabilize_enhancer: false`, then `stabilize_mask: false` | F1 (separately: content vs mask) |
| `selected_enhancer: none` | Anything enhancer-gated (F1, F3b); a halo that survives is F2/F4/F6 |
| `face_mask_blend: 12` | F2 (band should halve per §3.F2) |
| `ROOP_OCCLUSION_MASK=0` | F6 (two-segmenter disagreement) |
| `ROOP_MASK_RECOVER=0` | F6 (floor-ellipse widening) |
| `color_match_after_enhance: false`, `merger_*: 0` | F3b / F5 |
| Dump per frame: `A_xseg`, `A_occluder`, product, `B` matte in frame space, hull polygon, `yaw`, `a` from `EnhancerStabilizer` | Locates which edge is which in the ring |
| Dump the raw `fake_frame` background vs `aligned_img` on a yaw ≥ 45 frame | F5 (does the swap net keep the background?) |
| Overlay `landmark_2d_106` hull on a profile frame | F6 (hull bridging / hidden-side points) |

`ROOP_DEBUG_DUMP_RANGE` (`ProcessMgr.py:7289-7298`) is the only existing dump hook, and it writes a hard-coded `result[280:600, 350:650]` window, so the per-frame dumps above need a small harness rather than that hook.

---

## 7. Corrections to `.audit/01_pipeline_map.md`

| Map says | Reality |
|---|---|
| Stage 8: Laplacian multiband pyramid composite | Default is plain linear alpha (`procmgr_masking.py:897`). Multiband only runs with `temporal_compositing` on (live: off) |
| Stage 5: composite mask = Box × Hull × Occluder × Parser × ModelMask (`mask_engine.generate_composite_mask`) | Only the legacy `onnx_swappers` frame path calls it (`onnx_swappers.py:487-515`). The live path is RealityUX + injected occluder via `process_mask`, then the separate ellipse∩hull matte in `paste_upscale` |
| Stage 7: mask-weighted Reinhard (`color_matcher.py`) | Only the same legacy path. The live path is whole-crop LAB statistics (`procmgr_color.py`) |
| §3.3: `face_mask_blur 0.05`, `face_mask_erosion 0` govern the matte | The live feather is `blur_area(mask_offsets[4])` (`face_mask_blend`); neither named setting is read on this path |

---

## 8. Reproduction and limits

- Scripts (read-only, import the repo): `.audit/03_halo_experiments/` — `e1_e2_e3_feather_resample_lct.py`, `e3b_e4_greyfill_sharpen_rim.py`, `e5_stabilizer_yaw_ghost.py`, `e6_paste_matte_footprint_leak.py`. Run each with `app/env/Scripts/python.exe` (set `NO_ALBUMENTATIONS_UPDATE=1`).
- All inputs are synthetic (discs, flat colours, 1/f noise). They establish that the mechanisms exist and how they scale. They do not establish how large the effect is on real footage.
- `E5` fixes the keypoint centroid (worst case for F1). `E1`'s disc is a stand-in for hull∩ellipse. `E2` loses less on real, already-soft footage.
- The full suite was not run (the prior audit notes it stalls in `face_engine/tests/test_cuda_streams.py`). No test was added or changed.
- The two machines' configs differ deliberately; every config number above is the RTX 4070 machine's.
