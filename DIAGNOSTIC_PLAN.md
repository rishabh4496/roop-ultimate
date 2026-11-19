# Roop Ultimate: Architectural & Root-Cause Diagnostic Plan

**Date:** 2026-11-19  
**Repository:** `rishabh4496/roop-ultimate`  
**Audit Scope:** Read-Only Architectural & Root-Cause Pipeline Audit across `app/`, `react-ui/`, tests, and configs.

---

## Executive Summary & System Overview

This diagnostic plan establishes a rigorous, ground-truth architectural trace of `roop-ultimate`. It identifies the structural root causes behind side-angle degradation (>30° yaw/pitch), identity washout, double noses, hard mask seams, temporal flickering, and CPU-GPU performance bottlenecks. The investigation encompasses the entire pipeline lifecycle—from FFmpeg hardware-accelerated decode to NVENC stream composition—across both supported hardware profiles:
1. **Main Device (Desktop Workstation):** NVIDIA RTX 4070 Desktop (12 GB VRAM, 32 GB RAM, 24 physical cores).
2. **Secondary Device (Laptop Workstation):** NVIDIA RTX 3060 Laptop (6 GB VRAM, 16 GB RAM, mobile TDP).

---

## 1. Detection & Side-Angle Pipeline

### 1.1 Target Face Detection Architecture & Engine Dispatch
- **Central Dispatcher:** [`app/roop/face_util.py:1048–1140`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L1048-L1140) (`_detect_faces_raw`).
  - Reads studio configuration setting `detector_engine` (`scrfd`, `retinaface_r50`, `retinaface`, `retinaface_r50_gpu`, `yoloface`, `yunet`, `gnkps`).
  - Clamps detection resolution canvas to 640x640 (`eff_size = 640`, [`face_util.py:1089–1090`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L1089-L1090)) to maintain anchor stride stability.
  - Leases thread-safe inference sessions from `_ANALYSER_Q` via `lease_face_analyser` ([`face_util.py:653–685`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L653-L685)), avoiding concurrency locks while enforcing GPU VRAM limits.
  - Employs a hybrid detector pattern ([`face_util.py:871–1047`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L871-L1047), `_hybrid_detector_faces`): third-party engines extract `bbox` + 5 `kps`, while InsightFace `buffalo_l` auxiliary models (`w600k_r50.onnx`, `2d106det.onnx`, `1k3d68.onnx`) provide embeddings, gender, age, and 68/106 dense landmarks.
- **Detector Implementations:**
  - **SCRFD** ([`app/roop/face_util.py:1122–1131`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L1122-L1131)): Anchor-free distance regression model (`det_10g.onnx`). With `aux=False`, calls `fa.det_model.detect(frame, max_num=0)` directly, skipping ~39% per-frame overhead when embeddings/dense landmarks are not required.
  - **RetinaFace R50** ([`app/roop/retinaface.py:40–145`](file:///G:/pinokio/api/roop-ultimate/app/roop/retinaface.py#L40-L145)): ResNet-50 backbone (`retinaface_r50.onnx`). Decoded via `RetinaFace3Output` with prior boxes `_R50_MIN_SIZES = ((16, 32), (64, 128), (256, 512))`, `_R50_STEPS = (8, 16, 32)`, and variance `(0.1, 0.2)` (lines 45–48). Preprocessing applies BGR mean subtraction `(104.0, 117.0, 123.0)` without variance scaling.
  - **RetinaFace 10G** ([`app/roop/retinaface.py:18–24`](file:///G:/pinokio/api/roop-ultimate/app/roop/retinaface.py#L18-L24)): Anchor-free variant routed through InsightFace distance-regression decoder.
  - **RetinaFace GPU Engine** ([`app/roop/retinaface_gpu_engine.py:45–180`](file:///G:/pinokio/api/roop-ultimate/app/roop/retinaface_gpu_engine.py#L45-L180)): Direct TensorRT/CUDA GPU tensor execution avoiding host round-trips.
  - **YOLOFace** ([`app/roop/yoloface.py:25–145`](file:///G:/pinokio/api/roop-ultimate/app/roop/yoloface.py#L25-L145)): YOLOv8-nano (`yoloface_8n.onnx`). Emits `(1, 20, N)` tensor (4 bbox + 1 score + 15 landmark points) in a 640 canvas normalized with `(canvas - 127.5) / 128.0`.
  - **YuNet** ([`app/roop/yunet.py:30–110`](file:///G:/pinokio/api/roop-ultimate/app/roop/yunet.py#L30-L110)): Lightweight OpenCV `FaceDetectorYN` (`face_detection_yunet_2023mar.onnx`), pooled via `_ensure_pool` (lines 38–70).
- **Multi-Scale Pyramid & Context Padding:**
  - Context padding ([`app/roop/face_detector.py:196–234`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_detector.py#L196-L234), `apply_context_padding`): Reflective border padding (`cv2.BORDER_REFLECT_101`, min 64px) preventing anchor edge truncation.
  - Scale pyramid ([`app/roop/face_detector.py:328–358`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_detector.py#L328-L358), `generate_scale_pyramid`): Multi-scale levels `[0.5, 0.75, 1.0]` triggered for macro close-ups (>500px height or >75% frame coverage; `should_trigger_pyramid`, lines 394–437).
  - Distance-IoU NMS ([`app/roop/face_detector.py:145–194`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_detector.py#L145-L194), `diou_nms`): Merges multi-scale candidate detections using $DIoU = IoU - \frac{d^2}{c^2}$.
- **Detection Rescue Cascade:**
  - [`app/roop/face_util.py:1142–1250`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L1142-L1250): On 0-detection frames, triggers sequential fallbacks:
    1. `_rescue_downscaled` (line 1159): 320x320 pass with threshold lowered by 0.15 for macro close-ups.
    2. `_rescue_upscaled` (line 1142): 2x bilinear upscale for tiny/distant faces.
    3. `_rescue_padded` (line 1191): 25% border padding for boundary-clipped faces.
    4. `_rescue_clahe` (line 1238): CLAHE contrast equalization for backlit/dark frames.
    5. `_rescue_rotated` (line 1374): Cardinal rotations (90°, 180°, 270°) for rolled faces.

### 1.2 Computation of 5-Point ArcFace Landmarks and Yaw / Pitch / Roll
- **5-Point Landmark Extraction & Refinement:**
  - Primary detector regression outputs raw `kps` $(5, 2)$: left eye, right eye, nose tip, left mouth corner, right mouth corner.
  - 68-Point Refinement ([`app/roop/face_util.py:687–715`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L687-L715), `_refine_kps_from_68`): Replaces detector keypoints with dlib-standard 3D-68 landmarks (`1k3d68.onnx`): eye centers from `pts[36:42]` and `pts[42:48]`, nose tip from `pts[30]`, mouth corners from `pts[48]` and `pts[54]`.
  - Recognition Crop Stashing ([`app/roop/face_util.py:720–744`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L720-L744), `_stash_recognition_crops`): Pre-refinement 112 crop stashed under `_rec_crop_arcface_112_v2` for AdaFace/w600k embedding calibration.
  - ArcFace Templates: Canonical `arcface_dst` defined at [`face_util.py:2706–2715`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L2706-L2715) (112x112 baseline) and `arcface_112_v2` (lines 2735–2742). Scaled to 128, 256, 512 via `swap_template_points` ([`face_util.py:3450–3486`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L3450-L3486)).
- **Head Pose Estimation (Yaw, Pitch, Roll):**
  - **Weak-Perspective 5-Point Joint Solver** ([`app/roop/face_util.py:2935–2984`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L2935-L2984), `solve_pose_5pt`):
    - Solves $x = s \cdot R_{2 \times 3} \cdot X$ against reference 3D head `_reference_5pt()` (lines 2830–2840, derived from `_REF3D_68`).
    - Uses precomputed pseudo-inverse: $A = \text{\_REF5\_PINV} \cdot x$ (line 2976).
    - `_decompose_projection(A)` (lines 2986–3030): Recovers orthonormal rotation axes via Gram-Schmidt ($r_1, r_2, r_3 = r_1 \times r_2$).
    - Angles in degrees:
      $$\text{pitch} = \arcsin(\text{clip}(-r_{3,y}, -1, 1))$$
      $$\text{yaw} = -\arctan2(r_{3,x}, r_{3,z})$$
      $$\text{roll} = \arctan2(r_{1,y}, r_{2,y})$$
  - **Jaw-Decoupled 5-Point Solver** ([`app/roop/face_util.py:3050–3330`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L3050-L3330), `solve_pose_jaw_5pt`):
    - Decouples head pose from mouth opening drop along `_jaw_axis_5pt()` over 5 Gauss-Newton steps (`_JAW_NEWTON_STEPS = 5`), preventing open mouths from falsely registering as pitch.
  - **Perspective-n-Point (PnP) 3D Solver** ([`app/roop/face_analyser.py:138–234`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_analyser.py#L138-L234), `estimate_head_pose_pnp`):
    - Solves 2D-to-3D correspondence using `CANONICAL_FACE_3D_5` (lines 63–69) or `_REF3D_68` (line 174) with camera intrinsic matrix ($\text{focal} = 1.2 \times \max(H, W)$) via `cv2.solvePnP(flags=SOLVEPNP_SQPNP/EPNP)`.
  - **Pose Conventions & Adapters** ([`app/roop/head_pose.py:1–190`](file:///G:/pinokio/api/roop-ultimate/app/roop/head_pose.py#L1-L190)):
    - Enforces uniform sign convention: $+yaw = \text{nose to viewer's right}$, $+pitch = \text{tilted down}$, $+roll = \text{clockwise}$.

### 1.3 Multi-Angle Source Facesets (`.fsz`): Creation, Selection & Averaging
- **Archive Format & Packaging:**
  - [`app/roop/faceset_builder.py:20–150`](file:///G:/pinokio/api/roop-ultimate/app/roop/faceset_builder.py#L20-L150) (`build_faceset`): Bundles `.fsz` ZIP archive storing reference PNGs (`reference_000.png`...) and `faceset.json` (V2 format containing SHA-256 digests, detected geometry `yaw`/`pitch`/`roll`, 3x3 `pose_bins`, and identity vectors).
- **Loading & Cache Management:**
  - [`app/roop/faceset_manager.py:179–250`](file:///G:/pinokio/api/roop-ultimate/app/roop/faceset_manager.py#L179-L250) (`load_and_validate_faceset`): Reads archive with SHA-256 verification. If the active model space differs from stored space (`active_model_id != stored_id`), decodes PNG members and recalculates embeddings using the active embedder.
- **Averaging vs. Discrete Embeddings:**
  - V1 Archives ([`app/roop/FaceSet.py:51–68`](file:///G:/pinokio/api/roop-ultimate/app/roop/FaceSet.py#L51-L68), `AverageEmbeddings`): Computes arithmetic mean across all face embeddings: `first_face.embedding = np.mean(embeddings, axis=0)`. This blurs pose-specific identity vectors into an ambiguous average.
  - V2 Archives: Averaging is explicitly bypassed (`if self.format_version >= 2: return`), preserving discrete angle-specific vectors.
- **Angle Selection & Spherical Linear Interpolation (SLERP):**
  - [`app/roop/faceset_v2.py:1294–1410`](file:///G:/pinokio/api/roop-ultimate/app/roop/faceset_v2.py#L1294-L1410) (`get_pose_matched_embedding`):
    - Queries source bank using target $(\text{yaw}, \text{pitch})$.
    - Computes angular Euclidean distance $d_i = \text{hypot}(\text{yaw}_i - \text{yaw}_{\text{tgt}}, \text{pitch}_i - \text{pitch}_{\text{tgt}})$.
    - Selects top-3 closest angle embeddings.
    - Computes normalized inverse-distance weights $w_i = \frac{1}{d_i + 1e-5} / \sum w$.
    - Performs hierarchical GPU Spherical Linear Interpolation (`slerp_vectors`, lines 1404–1410).
  - [`app/roop/pose_source_selector.py:347–398`](file:///G:/pinokio/api/roop-ultimate/app/roop/pose_source_selector.py#L347-L398) (`select_pose_aware_source`):
    - Cost scoring in `score_source_entry` (lines 306–335): Pose cost ($0.52 \Delta\text{yaw} + 0.28 \Delta\text{pitch} + 0.20 \Delta\text{roll}$) weighted at 50–68%, blended with quality, expression, lighting, and facial proportion metrics.
    - Hysteresis switching latch (`switch_margin = 0.035` / `ROOP_POSE_SOURCE_SWITCH_MARGIN`, lines 356–366) preventing frame-to-frame source reference chatter.

### 1.4 Root Causes: Why Swaps Degrade or Drop Out Past 30° Yaw / Pitch
1. **Unweighted Umeyama Similarity Collapse ($IOD \to 0$):**
   - [`app/roop/face_util.py:3489–3535`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L3489-L3535) (`estimate_norm`) & [`app/roop/face_analyser.py:695–745`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_analyser.py#L695-L745) (`canonicalize_face_alignment`): Fits a 4-DoF similarity transform ($s \cdot R, t$) to canonical frontal `arcface_dst`.
   - Past 30° yaw, 3D perspective compresses inter-ocular distance ($IOD$). Fitting unweighted 5-point coordinates to a frontal layout severely distorts the similarity matrix, inducing extreme isotropic scale drift or skewed crop orientation.
2. **Outcome Guard Veto Dropouts (`VERIFY_MIN_OFFAXIS` & `swap_moved_the_face`):**
   - [`app/roop/procmgr_runtime.py:989–991`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_runtime.py#L989-L991): `VERIFY_MIN_OFFAXIS = 30.0` activates post-swap outcome re-detection specifically for faces $\ge 30^\circ$ off-axis.
   - [`app/roop/face_util.py:3745–3840`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L3745-L3840) (`swap_moved_the_face`): Re-detects the swapped face inside the ROI. Because profile IOD collapses, `scale = max(interocular, extent * 0.70)` triggers; if `moved > tol` (0.5) and `keypoint_shape_change > shape_tol` (0.3 for away heads; `procmgr_runtime.py:1018–1020`), the swap is deemed invalid and **reverted to the original unswapped frame**, causing swap dropouts/blinking.
3. **Swapper Latent Domain Breakdown:**
   - InsightFace InSwapper 128 / SimSwap / Ghost generators are trained primarily on near-frontal distributions ($|\text{yaw}| \le 30^\circ$). At higher angles, the generator synthesizes frontal features into an oblique crop, producing melted jawlines and double noses.
4. **Mask Router Boundary Transition Flips:**
   - [`app/roop/nonfrontal.py:45–58`](file:///G:/pinokio/api/roop-ultimate/app/roop/nonfrontal.py#L45-L58) (`NONFRONTAL_MARGIN = 0.15`, `_TGT_PITCH_MAX = 30.0`, `_OFFAXIS_MAX = 50.0`): When `nonfrontal_score > 1.0`, [`app/roop/procmgr_masking.py:1680`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1680) switches from `canonical-crop` masking to `unwarped-box` masking. Near 30°, noisy keypoints cause mask path flip-flopping and edge tearing.
5. **Roll/Profile Ambiguity Pocket:**
   - [`app/roop/orientation.py:1–45`](file:///G:/pinokio/api/roop-ultimate/app/roop/orientation.py#L1-L45): On faces turned to profile and rolled past 90°, the detector assigns degenerate keypoints (eye-mouth midline inverts to negative despite 0.99 confidence), leading to upside-down warps unless latched along the track.

---

## 2. Swapping, Masking & Restoration Pipeline

### 2.1 Exact Frame-to-Face Execution Order
The complete frame-to-face lifecycle executes through [`app/roop/ProcessMgr.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py) (inheriting mixins `MaskingMixin` in `procmgr_masking.py`, `ColorTransferMixin` in `procmgr_color.py`, `MergerMixin` in `procmgr_merger.py`).

```
[Frame Ingestion] (ProcessMgr.process_frame, L3511)
  │
  ├─► [Face Tracking & Association] (swap_faces, L3970)
  │     └─► Watchdog Zero-Swap Check (L3571)
  │
  ├─► [Depth Sorting & Region Demarcation] (_composite_faces, L5385)
  │     └─► build_face_regions / semantic_depth_regions (L5423-5425)
  │
  └─► For each face (process_face, L5779):
        1. Geometric Alignment: M matrix via canonicalize_face_alignment (L5930-6001)
        2. Pose Estimation: solve_pose_5pt / solve_pose_jaw_5pt (L6069-6119)
        3. Source Routing: select_pose_aware_source & latent blend (L6164-6489)
        4. Temporal Smart-Skip: Optical flow warp bypass check (L6563-6601)
        5. Swapper Net: InSwapper / RealSwap RunBatch (L6602-6756)
             └─► Watchdog Tensor Check: NaN / Inf / Zero-Black audit (L6729)
        6. Pre-Restoration Color Match: apply_color_transfer on skin interior (L6772-6784)
        7. Mask Engines & Occluder: RealityUX / XSeg / Mask_Occluder (L6787-6880)
             ├─► Unified Keep Matte update: np.minimum(_unified_keep, 1.0 - mask) (L6840)
             └─► Watchdog Mask Geometry Check: Collapse (0px) or >70% jump (L6857)
        8. Restorer / Enhancer Net: GPEN / UltraMax / CodeFormer (L6879-7083)
             ├─► Template Re-alignment: ArcFace -> FFHQ 512 (L6924-6965)
             ├─► Watchdog Tensor Check (L7032)
             └─► Inverse Affine Transform back to swap space (L7045-7056)
        9. Anti-Flicker Temporal Restorer Smoothing: Farneback blend (L7090-7156)
       10. Expression / High-Frequency Skin Detail Injection (L7158-7215)
       11. Post-Enhance Tone / Color Grading (L7218-7243)
       12. DFL Merger Post-Ops: Clarity, Sharpen, Grain Match (L7260-7318)
       13. Consensus Identity Detail Map Restoration (L7445-7494)
       14. Paste-Back / Upscale Blending: paste_upscale (L7663-7668)
             ├─► Landmark hull + BiSeNet + inward_alpha distance transform (L775-849)
             ├─► Rim-only boundary illumination match (L1064-1076)
             └─► Multi-band Laplacian blend into out_img (L1097-1143)
       15. Post-Composite Restores: Mouth, eyes, lip-sync, jaw reshape (L7681-7890)
       16. Outcome Verification: _verify_after (L7956-7959)
```

### 2.2 Root Causes: Washed-Out Identity
1. **Low-Resolution Pasteback & Unenhanced Fallback:**
   - [`app/roop/ProcessMgr.py:7593–7598`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L7593-L7598): When `enhanced_frame is None` (or skipped), the swapper crop (`inswapper` 128x128 or `hyperswap` 256x256) is bilinearly upsampled to 512 via `cv2.resize(fake_frame, (512, 512), interpolation=cv2.INTER_CUBIC)`. Pasting a 128px cubic-upsampled crop onto 1080p/4K plates results in a soft, washed-out face.
   - [`app/roop/processors/Enhance_GPENRealistic.py:13–25, 89–100`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_GPENRealistic.py#L13-L25): Emits 512x512 (`scale_factor = 2`). If `ROOP_GPENR_SIZE=256` or native `GPEN 256` is selected, it outputs 256x256 (`scale_factor = 1`), halving effective facial resolution reaching the canvas.
   - [`app/roop/smart_restorer_bypass.py:25–75`](file:///G:/pinokio/api/roop-ultimate/app/roop/smart_restorer_bypass.py#L25-L75) & [`ProcessMgr.py:7005–7015`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L7005-L7015): At face sizes between 256px and 384px, `bypass_weight` ramps up. At $\ge 1.0$, the restorer is completely bypassed with a simple Difference of Gaussians (DoG) filter (`apply_dog_unsharp_mask`), producing flat, unenhanced skin with washed-out facial likeness.
2. **Color Correction Drift & Contrast Degradation:**
   - [`app/roop/procmgr_color.py:424–450`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_color.py#L424-L450) (`_color_transfer_rct`): Reinhard LAB transfer adjusts the L (luminance) channel without bounds: `source = (source - source_mean) * std_scale + target_mean`. If target is backlit, over-exposed, or hazy, the source face's contrast is completely flattened and washed out.
   - [`app/roop/procmgr_color.py:445–447`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_color.py#L445-L447): A/B chrominance scaling is bounded $[0.80, 1.20]$, but if target footage has low color saturation, the source face loses its natural warmth.
   - [`app/roop/procmgr_masking.py:1028–1043`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1028-L1043) (`skin_lab_match`): In `paste_upscale`, forces the swapped skin to match the underlying target plate's skin tone via `_bl.skin_lab_match`. At `_lab = 1.0` (default), it recolors the swapped face towards the plate's complexion, stripping away the source identity's distinctive skin pigmentation.
   - **Restorer Chrominance Bleaching:**
     - [`app/roop/processors/Enhance_UltraMax.py:287–288`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_UltraMax.py#L287-L288): CodeFormer codebook reconstruction drifts pale/desaturated. Without `luma_only_recolour`, it produces a ghost-like, chalky appearance.
     - [`app/roop/processors/Enhance_GPENRealistic.py:7–12`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_GPENRealistic.py#L7-L12): Native GPEN drifts chroma by +2.96 in LAB, flushing cheeks magenta.
   - [`app/roop/ProcessMgr.py:7228–7243`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L7228-L7243): `color_match_after_enhance` reapplies color transfer to `enhanced_frame`, compounding color degradation twice.
3. **Blend Modes & Excessive Inward Alpha Feathering:**
   - [`app/roop/procmgr_masking.py:1017–1023`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1017-L1023): If `blend_ratio` is $< 1.0$ (e.g. 0.85 on the RTX 3060 profile), `cv2.addWeighted(upsk_face, _blend_ratio, base, 1. - _blend_ratio, 0.)` blends 15% of the raw, unenhanced swap into the final face, muting sharp eye and skin features.
   - [`app/roop/procmgr_masking.py:1118–1128`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1118-L1128): In `paste_upscale`, `alpha_feather` mode executes linear interpolation: `roi_matte * roi_paste + (1.0 - roi_matte) * roi_target`. Over wide transition boundaries, target face pixels blend with swapped face pixels, producing a translucent, washed-out hybrid identity.
   - [`app/roop/unified_matte.py:57–68`](file:///G:/pinokio/api/roop-ultimate/app/roop/unified_matte.py#L57-L68) (`inward_alpha`): High `mask_feather` (default 20–25%) applies distance transform erosion deep into the facial interior ($w = feather \cdot span$). The boundary of the face reverts to the target, shrinking the visible swap area to an inner oval.
   - [`app/roop/ProcessMgr.py:7520`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L7520): In temporal QC recovery, `cv2.addWeighted(enhanced_frame, 0.55, fake_frame, 0.45, 0.0)` knocks down restorer strength by 45%.

### 2.3 Root Causes: Double Noses & Artifact Ghosting
1. **RealSwap Eyelid Band Bleed Across the Nose Bridge:**
   - [`app/roop/processors/FaceSwapInsightFace.py:1726–1737`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/FaceSwapInsightFace.py#L1726-L1737):
     - Eye mask outer radius: `_OUTER_X = 0.44`, `_OUTER_Y = 0.30`.
     - Eye aperture radius: `_EYE_APERTURE_X = 0.225`, `_EYE_APERTURE_Y = 0.066`.
     - In ArcFace template space, the eye centers are at $\pm 0.5 \times sep$. An outer horizontal radius of $0.44$ extends to $0.5 - 0.44 = 0.06$ from the center of the nose bridge!
     - If `_EYE_FEATHER` (0.05) softens the mask or `_OUTER_X` exceeds 0.44, the two eyelid ellipses overlap across the nose bridge. This composites HifiFace's nose bridge over Hyperswap's nose bridge, creating a **visible double nose bridge / ghosted nasal contour**.
   - [`app/roop/processors/FaceSwapInsightFace.py:1782–1816`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/FaceSwapInsightFace.py#L1782-L1816) (`_BASE_MIX`): If set to $> 0.0$ (historically 0.15), it mixes HifiFace globally across all pixels. Because HifiFace has an independent 3D shape branch with different nose/lip positions than Hyperswap, mixing them globally generates a doubled nose tip, doubled lips, and dual eyelid creases.
2. **Restorer Template Re-Alignment Distortion (ArcFace $\leftrightarrow$ FFHQ Roundtrip):**
   - [`app/roop/ProcessMgr.py:6924–6965, 7045–7056`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L6924-L6965):
     - Swapper operates on ArcFace template (`swap_template`, 112/256px); restorers operate on FFHQ (`ffhq_512`). ArcFace eyes sit 31px higher and face is 17% larger than FFHQ.
     - Re-alignment forward warp: `_A = _compose_affine(_M_e, _invert_affine(M))` warps swap crop to FFHQ (line 6932).
     - Re-alignment inverse warp: `_B = cv2.invertAffineTransform(_A)` with `_B[:, 2] *= float(scale_factor)` warps enhanced frame back to swap space (lines 7050–7056).
     - If keypoints have 1–2px inter-frame noise, subpixel rounding differences between $M$, $M_e$, $A$, and $B$ cause the restored nose tip and nostril wings to misalign with `fake_frame` by 1–3 pixels. When blended via `frequency_blend` or multi-band composite, this manifests as **doubled nostrils, dual nose bridges, and ghosted lip lines**.
3. **Target Mouth and Eyes Restoration at Angles (EPnP / Yaw Inversion):**
   - [`app/roop/ProcessMgr.py:6052–6068`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L6052-L6068): When pose was derived from `landmark_3d_68` via EPnP, missing models left yaw/pitch at $0.0^\circ$ (or inverted at $180^\circ - yaw$).
   - [`app/roop/procmgr_masking.py:2235–2241`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L2235-L2241) (`apply_eyes_area`) and lines 2333–2337 (`apply_mouth_area`):
     - Both methods use an angular fade: `if max_angle > 25.0: mask *= max(0.0, min(1.0, (38.0 - max_angle) / 13.0))`.
     - When head pose estimation fails or reads $0.0^\circ$ on a $60^\circ$ profile, the fade never triggers. The target plate's frontal mouth is pasted onto the side of the turned cheek, and target eyes are pasted onto the swapped nose bridge, causing **glaring double noses and duplicated lips**.
4. **Crop vs. Full-Frame Matte Warp Inconsistencies:**
   - [`app/roop/procmgr_masking.py:753–761, 896–898`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L753-L761): $M_{scale}$ is inverted to $IM$.
   - `_scale_paste` (line 761): Adjusts $IM$ by `output_face_scale`. If `upsk_face` is warped with translated $IM$ while `img_matte` is computed from landmark hulls before scaling, the matte boundary does not align with facial features, cutting off one nostril while duplicating the other.
   - [`app/roop/procmgr_masking.py:1726–1789`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1726-L1789): The unwarped bounding-box path (`crop_box`) extracts an unrotated box from `orig_frame`, infers the mask, pastes into `full_frame_mask`, and warps into canonical crop space via $M$. This dual-resampling roundtrip shears mask contours relative to the affine-aligned face crop, producing shifted nose boundaries.

### 2.4 Root Causes: Hard Mask Seams
1. **Alpha Feathering Inward Falloff & Elliptical Cutoffs:**
   - [`app/roop/procmgr_masking.py:775–784`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L775-L784): Default mask offsets construct an inner ellipse: `cx = (left + right) // 2`, `ax = (right - left) // 2`. If `mask_offsets[0..3]` are set to 0, the ellipse touches the boundary of the 512x512 crop box.
   - [`app/roop/procmgr_masking.py:887`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L887): Enforces hard boundary truncation: `alpha[[0, -1], :] = 0.; alpha[:, [0, -1]] = 0.`. When affine $IM$ warps this cropped alpha into frame space, the straight box edge projects into a visible diagonal line across the forehead or neck.
   - [`app/roop/unified_matte.py:65–68`](file:///G:/pinokio/api/roop-ultimate/app/roop/unified_matte.py#L65-L68): In `inward_alpha`, feathering is capped by `FEATHER_CAP = 0.18 * span`. On high yaw angles ($>10^\circ$), lateral inset `dist = np.maximum(dist - inset, 0.)` steeply cuts off the far cheek with a sharp transition.
2. **Hard Bounding Box Edge Clamping in Bounded ROI Warp:**
   - [`app/roop/procmgr_masking.py:1004–1013`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1004-L1013): Bounded ROI warp clips to `y0, y1 = max(0, nz_y.min() - 2), min(H, nz_y.max() + 3)`. If `nz_y` touches the frame border or non-zero matte values extend beyond padded bounds, a hard 1px seam step is painted at $(x_0, y_0, x_1, y_1)$.
3. **Hard Binary Thresholding in Mask Processors:**
   - [`app/roop/procmgr_masking.py:1823–1826`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1823-L1826) (`Mask_Occluder`):
     - `binary_mask = (img_mask > 0.35).astype(np.float32)`
     - `img_mask = cv2.GaussianBlur(binary_mask, (k, k), 0) if k > 1 else binary_mask`
     - Hard step at 0.35. If kernel $k \le 1$, the mask is strictly binary $\{0, 1\}$. When pasted, this produces jagged, aliased stair-stepping seams around fingers, microphones, or hair strands.
   - [`app/roop/processors/Mask_Clip2Seg.py:98–100`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Mask_Clip2Seg.py#L98-L100): Hard step function `clip_mask[clip_mask > 0.5] = 1.0; clip_mask[clip_mask <= 0.5] = 0.0`.
   - [`app/roop/processors/Mask_RealityUX.py:416–421`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Mask_RealityUX.py#L416-L421): In foreign object occlusion, `occ_raw = (xseg_mask > 0.40).astype(np.uint8) * 255` followed by `cv2.erode(occ_raw, morph_k, iterations=1)`. Hard erosion breaks thin occluders (glasses rims, hair) into discontinuous islands with hard perimeter seams.
4. **Seam Illumination Mismatch:**
   - [`app/roop/procmgr_masking.py:1055–1076`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1055-L1076): If `boundary_illumination_strength` is $0.0$ (default), any difference in exposure between the restored face crop and the plate creates a distinct brightness line at the boundary of the alpha ramp, visible even when the alpha feather is mathematically smooth.

### 2.5 Root Causes: Occluder vs. Face-Mask Conflicts
1. **Polarity Conventions Across Engines:**
   - **Roop Ultimate Standard Convention:**
     $$\text{Mask} = 1.0 \implies \text{Restore Original Plate}; \quad \text{Mask} = 0.0 \implies \text{Keep Swapped Face}$$
   - **Engine Polarity Inversions:**
     - `Mask_Occluder.py:109–112`: Raw `face_occluder.onnx` outputs $1.0$ on the *visible face*. It is explicitly inverted via `result = 1.0 - result`.
     - `Mask_XSeg.py:172–176`: Raw `xseg.onnx` outputs $1.0$ on face skin. Inverted via `result = 1.0 - result`.
     - `Mask_FaceParser.py:26`: Classes 1, 2, 3, 4, 5, 10, 11, 12, 13 are face ($0.0$); all others are $1.0$.
2. **Boolean Mask Arithmetic & Fusion Logic:**
   - [`app/roop/occlusion_mask.py:20–29`](file:///G:/pinokio/api/roop-ultimate/app/roop/occlusion_mask.py#L20-L29): The processor chain applies masks sequentially:
     $$\text{Result}_1 = \text{Plate} \cdot m_1 + \text{Swap} \cdot (1 - m_1)$$
     $$\text{Result}_2 = \text{Plate} \cdot m_2 + \text{Result}_1 \cdot (1 - m_2)$$
     $$\text{Keep}_{\text{final}} = (1 - m_1)(1 - m_2) \iff \text{Mask}_{\text{final}} = 1 - (1 - m_1)(1 - m_2) = \max(m_1, m_2)$$
   - [`app/roop/ProcessMgr.py:6840–6844`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L6840-L6844): Computes `_unified_keep = np.minimum(_unified_keep, 1.0 - mask2d(_img_mask))`. Taking `minimum` of keep masks is mathematically identical to taking `maximum` of exclusion masks ($\text{Exclusion} = \max(M_{\text{face}}, M_{\text{occluder}})$).
3. **Priority Inversions & Suppression Gates in RealityUX:**
   - [`app/roop/processors/Mask_RealityUX.py:334–385`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Mask_RealityUX.py#L334-L385):
     - BiSeNet non-face accessories (`is_accessory`: ears, hair, cloth, hat) are gated by XSeg:
       `accessory_allowed = np.clip((xseg_mask - 0.05) / 0.20, 0.0, 1.0)`
       `non_face = np.clip(is_accessory * accessory_allowed, 0.0, 1.0)`
     - **The Priority Inversion Bug:** `accessory_allowed` is zero when `xseg_mask < 0.05`. If XSeg falsely classifies an occluding hand, headphone band, or stray hair strand across the forehead as "face skin" ($xseg < 0.05$), BiSeNet is **completely suppressed** and prohibited from excluding it. XSeg overrides BiSeNet even when BiSeNet has 100% confidence.
   - **Ungated Glasses Exception** ([`Mask_RealityUX.py:387–410`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Mask_RealityUX.py#L387-L410)):
     - BiSeNet class 6 (glasses) bypasses `accessory_allowed` and is merged via `combined = np.maximum(combined, glasses)`.
     - If `glasses_frame_mask` has false positive detections on deep eye sockets or shadows, it overrides XSeg and forces original pixels back onto the swapped face.
   - **Threshold Incoherency Across Engines:**
     - `Mask_Occluder` threshold: $0.35$ ([`procmgr_masking.py:1823`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1823)).
     - `Mask_XSeg` threshold: $0.10$ ([`Mask_XSeg.py:174`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Mask_XSeg.py#L174)).
     - `RealityUX` XSeg occlusion threshold: $0.40$ ([`Mask_RealityUX.py:416`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Mask_RealityUX.py#L416)).
     - An occlusion signal at confidence $0.38$ is recognized as an occluder by `Mask_Occluder`, ignored by `RealityUX`, and treated as face boundary by `Mask_XSeg`, causing flickering edge chatter when switching between engines.
   - `_stamp_occlusion_state` ([`procmgr_masking.py:1905`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1905) & [`ProcessMgr.py:4755, 7912–7918`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L4755)):
     - Stamped from `mask_occluder` or `mask_xseg3` if $>8\%$ of landmarks read $>0.35$.
     - When `occlusion_state == 'partial'`, it previously discarded the entire face swap (lines 7899–7911), causing full-frame flickering whenever an object touched a face. It is now kept for diagnostics, but still forces lower identity lock detection thresholds (line 4768).

---

## 3. Flickering Root Causes

### 3.1 Stateless vs. Stateful Frame Processing Architecture
The pipeline exhibits a bifurcated execution model where underlying neural networks are largely stateless per inference call, but high-level tracking, stabilization, and appearance layers are deeply stateful across frames:
- **Worker Concurrency & Queue Dispatch Models** ([`app/roop/procmgr_batch.py:813–840, 1128–1220`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_batch.py#L813-L840)):
  1. *Unified Frame Pipeline* (`_run_unified_scheduler`, lines 1150–1167): Enforces single-worker inference (`_inference_workers = 1`) with continuous stabilizer FIFO (`_streaming_stabilization`). Frames flow sequentially through a bounded queue (depth 2–4).
  2. *Parallel Block Stabilization* (`_run_stab_parallel` in [`app/roop/ProcessMgr.py:2428–2920`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L2428-L2920)): Splits video into contiguous frame chunks (`CHUNK = blocks_per_chunk * block`). Each worker thread processes a contiguous sub-block (`_process_block`, lines 3130–3220) and primes thread-local filter clones (`self._tls.kps`, `self._tls.enh`, `self._tls.mask`, `self._tls.hf_stabilizer`, `self._tls.temporal_enhancer`) across $WU$ warm-up frames (`_stab_warmup_frames` in [`app/roop/procmgr_stabilization.py:25–91`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_stabilization.py#L25-L91)) before emitting real output frames.
  3. *Unordered Strided Multi-threading Fallback* (`procmgr_batch.py:1180–1211`): In standard multi-threaded execution without parallel block stabilization, `read_frames_thread` distributes frames round-robin: worker $i$ receives frames $i, i+N, i+2N$. Workers **do not see consecutive frames**.
- **Per-Worker Tracking Isolation** ([`app/roop/ProcessMgr.py:75–136`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L75-L136)):
  Because the Kalman tracker (`FaceTracker`, [`app/roop/tracker.py:25`](file:///G:/pinokio/api/roop-ultimate/app/roop/tracker.py#L25)) is causal and order-dependent, sharing it across strided workers causes $N$-fold velocity extrapolation. `get_thread_tracker` (lines 106–136) assigns each worker a thread-local clone keyed by `_dispatch_epoch`, and explicitly disables Kalman coasting across the stride (`_may_coast = False` in [`ProcessMgr.py:4184–4187`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L4184-L4187)).
- **Sequential Pre-Pass Cache Architecture** ([`app/roop/procmgr_batch.py:234–264`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_batch.py#L234-L264), [`app/roop/procmgr_tracking.py:2571–2770`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_tracking.py#L2571-L2770)):
  When `temporal_detection` / `_temporal_mode` is enabled, the pipeline executes a single-threaded sequential pre-pass (`_precompute_temporal`) before any swap worker starts. It scans the video in order, builds full-clip face tracks, gap-fills misses, pre-smooths keypoints/landmarks, and seals the result into `self._temporal_faces = {frame_idx: [Face, ...]}`. Main processing workers in Stage 3 then consume this read-only cache in parallel without re-detecting.

### 3.2 Non-Deterministic & Unsmoothed Per-Frame Signals
1. **Bounding Box and Landmark Jitter:**
   - Detectors (`RetinaFace`, `YOLO`, `SCRFD`, `YuNet` in [`app/roop/face_analyser.py:180–270`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_analyser.py#L180-L270)) perform per-frame NMS and bounding box regression with zero temporal memory. Bounding boxes and 5-point facial keypoints ($kps$) jitter by 1–5 pixels per frame from sensor noise and sub-pixel quantization.
   - In [`app/roop/ProcessMgr.py:5930–6002`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L5930-L6002), `canonicalize_face_alignment` ([`app/roop/face_util.py:195–260`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L195-L260)) calculates the affine similarity transform matrix $M$ directly from the 5 keypoints via `estimate_norm`. Sub-pixel $kps$ jitter directly modulates rotation, translation, and scale of $M$, causing the aligned crop to vibrate.
   - In [`app/roop/procmgr_tracking.py:3424–3450`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_tracking.py#L3424-L3450), independent `OneEuroFilter` instances historically filtered 5-point $kps$ (driving alignment $M$) and 106-point $lm106$ (driving the paste mask hull). Because cutoffs depend on velocity ($min\_cutoff + \beta \cdot |\hat{dx}|$), the two representations respond at different rates, drifting apart by 1–2 pixels during movement and opening a crawling gap at the jawline.
   - Fallthrough without smoothing: When `temporal_detection` is disabled or fails ([`ProcessMgr.py:4131–4162`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L4131-L4162)), `get_all_faces(frame)` runs per-frame with un-smoothed keypoints.
2. **Target Face-ID Re-indexing & Clustering Jumps:**
   - In [`app/roop/face_analyser.py:276`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_analyser.py#L276), `get_all_faces` sorts detected faces by bounding box x-coordinate (`x.bbox[0]`). In `all_input` mode ([`app/roop/ProcessMgr.py:4240–4253`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L4240-L4253)), face $i$ is assigned strictly to source $i$. When two subjects cross or move laterally, their x-order inverts, causing an instantaneous identity flip.
   - In `selected` / `selected_multi` mode without identity lock ([`ProcessMgr.py:4830–4920`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L4830-L4920)), faces are matched per frame using raw ArcFace/AdaFace embedding cosine distance (`_ada.identity_distance`). Off-axis yaw (>30°), motion blur, or partial occlusion spikes cosine distance from ~0.35 to >0.80. If distance exceeds `id_threshold`, the face is refused (`refused: over match threshold`), causing single-frame swap dropouts. When multiple persons are present, the distance can fall closer to person B than person A, flipping target assignment.
   - In [`ProcessMgr.py:4568–4680`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L4568-L4680), the identity lock's source veto evaluates the *current frame's* noisy embedding (`_dist_to_source(face, src_index)`). If $d_{own} > \text{VETO\_DIST}$ (0.85) on a profile or blurred frame, the veto fires (`VETO_FAR_FROM_OWN`), dropping the face to per-frame matching at the tighter threshold. When per-frame matching fails, the swap blinks off for that frame and returns on the next, generating violent on/off identity strobing.
   - In [`app/roop/procmgr_tracking.py:1945–2100`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_tracking.py#L1945-L2100) (`_assign_track_sources`), touching or close faces contaminate recognition crops (`_emb_contam`, [`app/roop/face_contact.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_contact.py)). If both tracks fail the assignment gate, neither receives a source (`track matched but has no source`, up to 42.7% on stress clips).
3. **Occluder Mask Oscillation & Flickering Edge Contours:**
   - Hard thresholding in `Mask_Occluder` ([`app/roop/procmgr_masking.py:1821–1826`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1821-L1826)): The occluder engine output is binarized with a hard threshold `(img_mask > 0.35)`. Continuous probabilities near 0.35 toggle between 0 and 1 from minor frame-to-frame noise, producing severe mask edge buzzing and contour popping across frames.
   - Asymmetric maximum in mask stabilization ([`procmgr_masking.py:1886`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1886)): Temporal mask smoothing enforces `img_mask = np.maximum(img_mask, _current_mask)`. Any transient noise spike in `_current_mask` instantly pulls the matte high, and `np.maximum` prevents temporal decay from smoothing it down, creating persistent edge flicker.
   - Non-frontal mask routing chatter ([`procmgr_masking.py:1657–1675`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1657-L1675)): Toggling between canonical crop space and unwarped bounding-box space (`is_non_frontal`) introduces an abrupt projection shift that flickers unless strictly latched by `NonFrontalRouter`.
4. **Per-Frame Color-Transfer Swings:**
   - Memoryless statistical estimation ([`app/roop/procmgr_color.py:424–450`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_color.py#L424-L450)): Reinhard Color Transfer computes mean and std independently per frame (`cv2.meanStdDev(source, mask=mask8)`). Similarly, LCT (line 532), MKL (line 600), and IDT (line 632) recompute color mapping from scratch on every frame.
   - Spatial skin sampling variation: Head rotation or expression shifts move skin highlights, shadows, and hair boundaries in and out of `mask8`, shifting `target_mean` and `target_std` frame by frame. This causes rapid luminance and chrominance strobing across the swapped face.
   - Disconnected temporal smoothing: [`app/roop/temporal_color.py:1–60`](file:///G:/pinokio/api/roop-ultimate/app/roop/temporal_color.py#L1-L60) records that a naive EMA on LAB color statistics was tested (`tests/probe_color_ema.py`) and left disconnected because lagging behind the target plate increased pumping by 9–20% ($dE/\text{frame}$ 0.138 $\to$ 0.165). Unless `target_conditioned_appearance` (`TargetAppearanceStabilizer` in [`app/roop/appearance_conditioning.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/appearance_conditioning.py)) is explicitly enabled, color transfer remains completely memoryless.
5. **Generative Restorer Shimmer:**
   - Generator hallucination sensitivity: Generative restorers (GPEN in [`app/roop/processors/Enhance_GPEN.py:163`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_GPEN.py#L163), CodeFormer in [`Enhance_CodeFormer.py:160`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_CodeFormer.py#L160), GFPGAN in [`Enhance_GFPGAN.py:100`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_GFPGAN.py#L100)) possess deep generative priors. Sub-pixel alignment shifts or interpolation artifacts alter intermediate latent features, synthesizing entirely different micro-textures (skin pores, eyelashes, fine hair, teeth reflections) on each frame.
   - Discrete codebook jumps: In CodeFormer, continuous latent vectors crossing Voronoi cell boundaries in the discrete VQ codebook produce discontinuous jumps in high-frequency texture.
   - Enhancer realignment noise ([`app/roop/ProcessMgr.py:6924–6965`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L6924-L6965)): `_realign` resamples the crop into the enhancer's 512px FFHQ template, composites plate context around the border, runs the net, and warps back via inverse affine bicubic interpolation, introducing double-resampling noise.
   - Stabilizer limitations:
     - `one_euro.EnhancerStabilizer` ([`app/roop/one_euro.py:207`](file:///G:/pinokio/api/roop-ultimate/app/roop/one_euro.py#L207)) blends whole crops with a scalar weight without optical flow; to avoid ghosting during motion, it must turn off, exposing raw shimmer exactly when the head moves.
     - `TemporalEnhancerBlend` ([`app/roop/enhancer_stabilizer.py:145–280`](file:///G:/pinokio/api/roop-ultimate/app/roop/enhancer_stabilizer.py#L145-L280)) registers outputs via Farneback optical flow on a 64px grid, but resets history on any occlusion $> 0.35$ or motion step (`SHOT_RESIDUAL > 24.0`), releasing bursts of raw generative flicker.

---

## 4. Performance Bottlenecks

### 4.1 Frame Pipeline: Video Decode to Video Encode
- **In-Memory Pipe & Host Buffering:**
  - Route: [`app/roop/procmgr_batch.py:447–454`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_batch.py#L447-L454) $\to$ [`app/roop/video_stream.py:920–976`](file:///G:/pinokio/api/roop-ultimate/app/roop/video_stream.py#L920-L976) (`open_video_capture`) $\to$ [`app/roop/video_stream.py:123–200`](file:///G:/pinokio/api/roop-ultimate/app/roop/video_stream.py#L123-L200) (`NVHardwareVideoReader`).
  - FFmpeg decodes video via NVDEC (`-hwaccel cuda`), but dumps uncompressed `rawvideo bgr24` to stdout (`pipe:1`), copying frames from GPU memory to CPU host RAM.
  - [`app/roop/video_stream.py:347–357`](file:///G:/pinokio/api/roop-ultimate/app/roop/video_stream.py#L347-L357) reads stdout pipe into `np.empty()` in CPU RAM.
  - [`app/roop/ProcessMgr.py:2562–2618`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L2562-L2618) (`prefetch_q`) and lines 3108–3118 (`combined`) buffer entire chunks (e.g. 192 frames @ 1080p = ~1.2 GB host RAM).
  - Audio extraction: [`app/roop/util_ffmpeg.py:126–170`](file:///G:/pinokio/api/roop-ultimate/app/roop/util_ffmpeg.py#L126-L170) (`extract_audio_wav`) & [`app/roop/core.py:1946`](file:///G:/pinokio/api/roop-ultimate/app/roop/core.py#L1946) (`finalize_render`).

### 4.2 Synchronous CPU-GPU Memory Copies (Ping-Pong Transfers)
The frame processing pipeline exhibits severe PCIe ping-pong bus traffic between GPU VRAM and CPU host RAM:
```
1. GPU NVDEC Decode
   │ [D2H Copy via stdout pipe:1]
   ▼
2. CPU Host RAM: NumPy frame array (video_stream.py:347)
   │ CPU OpenCV: canonicalize_face_alignment affine crop (ProcessMgr.py:5930-6001)
   │ [H2D Copy: utilities.py:127 or FaceSwapInsightFace.py:1509]
   ▼
3. GPU Swapper: InSwapper / RealSwap ONNX / TensorRT inference
   │ [D2H Copy: utilities.py:159 or CudaOrtIOBinding.get_outputs]
   ▼
4. CPU Host RAM: fake_frame NumPy array (ProcessMgr.py:6723)
   │ [H2D Copy: Mask_RealityUX.py:270 or Mask_Occluder.py:94]
   ▼
5. GPU Mask Engines: BiSeNet / XSeg / Face Occluder inference
   │ [D2H Copy: Mask_RealityUX.py:310]
   ▼
6. CPU Host RAM: img_mask NumPy array (ProcessMgr.py:6840)
   │ CPU OpenCV: FFHQ re-alignment warp (ProcessMgr.py:6932)
   │ [H2D Copy: Enhance_GPEN256Pro.py:683 or Enhance_CodeFormer.py:251]
   ▼
7. GPU Restorer: GPEN / CodeFormer / GFPGAN inference
   │ [D2H Copy: Enhance_GPEN256Pro.py:685]
   ▼
8. CPU Host RAM: enhanced_frame NumPy array (ProcessMgr.py:7045)
   │ CPU OpenCV: inverse affine warp, color transfer, paste_upscale Laplacian blend (ProcessMgr.py:7664)
   │ [H2D Copy via stdin pipe:0 (video_stream.py:787)]
   ▼
9. GPU NVENC Encode: Final frame encoding to H.264/HEVC bitstream
```
Every intermediate stage re-copies memory with `np.ascontiguousarray()`, `cv2.cvtColor()`, and `torch.from_numpy().cpu().numpy().copy()`, bottlenecking throughput on PCIe bus bandwidth and CPU cache evictions.

### 4.3 Missing TensorRT / CUDA IOBinding & Buffer Reuse
- [`app/roop/utilities.py:115`](file:///G:/pinokio/api/roop-ultimate/app/roop/utilities.py#L115): `CudaOrtIOBinding` instantiates a **new** `session.io_binding()` on every single call, binding inputs and outputs freshly per inference rather than pre-allocating persistent device pointers.
- Detectors ([`retinaface.py:210`](file:///G:/pinokio/api/roop-ultimate/app/roop/retinaface.py#L210), [`yoloface.py:180`](file:///G:/pinokio/api/roop-ultimate/app/roop/yoloface.py#L180)): Missing `IOBinding` entirely; execute bare `session.run()` with fresh `cv2.dnn.blobFromImage` CPU allocations.
- Restorers:
  - [`Enhance_GPEN256Pro.py:682–685`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_GPEN256Pro.py#L682-L685) uses `bind_cpu_input` and `copy_outputs_to_cpu`.
  - [`Enhance_CodeFormer.py:251–252`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_CodeFormer.py#L251-L252) copies outputs to CPU.
  - `Enhance_TileDiffusion.py` and `Frame_Upscale.py:228` invoke bare `session.run()`.
- RealSwap secondary swapper ([`FaceSwapInsightFace.py:2364–2367`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/FaceSwapInsightFace.py#L2364-L2367)): Explicitly bypasses `IOBinding` for `session.run()`.

### 4.4 Per-Frame Model / Session Re-Init & Cache Misses
- [`app/roop/ProcessMgr.py:3138–3168`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L3138-L3168): Every chunk block re-clears thread-local storage (`self._tls`) and re-clones 9 temporal engines and thread trackers.
- Redundant re-detections:
  - Rotated face fallback ([`ProcessMgr.py:5883`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L5883)).
  - 3D posed crop re-detection ([`ProcessMgr.py:6398`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L6398)).
  - Post-swap verification ([`ProcessMgr.py:7955`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L7955), `_verify_after` re-detects the swapped face inside the full frame).
- Detection rescue cascade ([`face_util.py:1943–2012`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L1943-L2012)): Executes up to 5 full-frame detection passes (`_rescue_upscaled`, `downscaled`, `padded`, `rotated`, `clahe`) whenever a single face is missed.

### 4.5 GIL Contention
- All worker threads are standard OS threads (`threading.Thread`, [`ProcessMgr.py:2899`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L2899)) in a single Python process. `worker_pool.py` `ProcessPoolExecutor` is not wired into `ProcessMgr`.
- The Global Interpreter Lock (GIL) serializes concurrent execution of OpenCV affine warps, landmark math, and tracking logic. Unpooled models serialize on `_gpu_guard()` ([`ProcessMgr.py:66`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L66)).
- Gate E empirical findings ([`PERFORMANCE_BASELINE.md`](file:///G:/pinokio/api/roop-ultimate/PERFORMANCE_BASELINE.md)): Pipeline is GPU-bound (scaling from 4 to 20 threads yielded only +0.7% FPS); multiprocessing IPC for 1080p uncompressed frames would add severe memory copy overhead.

### 4.6 Redundant Full-Frame Detection vs. ROI Tracking
- Pre-pass ([`procmgr_tracking.py:2571`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_tracking.py#L2571)) caches faces in `self._temporal_faces`.
- When disabled or cache-missed ([`ProcessMgr.py:4131`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L4131)), the engine runs full-frame `get_all_faces(frame)`, followed by ROI rescue (line 4136), retry_rotated (line 3641), and post-swap verify (line 7955).

### 4.7 Unbatched Multi-Face Inference
- [`app/roop/ProcessMgr.py:5431–5436`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L5431-L5436):
  ```python
  for k in order:
      self.process_face(...)
  ```
  Iterates faces in a Python sequential `for`-loop.
- In multi-face footage (e.g. 2–4 faces per frame), Swapper, Mask, and Enhancer run sequentially with `batch_size = 1`, causing linear latency scaling $\mathcal{O}(N)$.
- Cross-frame `swap_batcher.py` drops model masks and introduces queue latency.

### 4.8 Disk I/O Stalls
- **Disk Mode** ([`app/roop/core.py:1820–1896`](file:///G:/pinokio/api/roop-ultimate/app/roop/core.py#L1820-L1896)): Writes extracted PNG frames to disk via `ffmpeg.extract_frames`, duplicates to `_orig`, reads back via `cv2.imdecode`, and re-encodes via FFmpeg reading PNG files from disk.
- **In-Memory Pipe Mode** ([`app/roop/procmgr_batch.py:44–1235`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_batch.py#L44-L1235)): Uses stdin/stdout pipes; `finalize_render` ([`core.py:1946`](file:///G:/pinokio/api/roop-ultimate/app/roop/core.py#L1946)) still writes an intermediate `__temp.mp4` before final audio muxing.

---

## 5. Constraints Map, Standalone Invariants & UI

### 5.1 Offline-Only Requirements & Local Caching
- **Network Boundary Contract:** Defined in [`docs/development/NETWORK_CONTRACT.md:1–85`](file:///G:/pinokio/api/roop-ultimate/docs/development/NETWORK_CONTRACT.md#L1-L85). All video and frame processing is 100% offline and local. Loopback API (`127.0.0.1:${apiPort}`) and Vite dev server ([`react-ui/vite.config.js:35`](file:///G:/pinokio/api/roop-ultimate/react-ui/vite.config.js#L35)) are local-only application services.
- **Model Offline Caching & Directory Structure:**
  - Main directory: `app/models/` with dedicated subdirectories:
    - `models/CLIP/`: Azure CDN models ([`app/clip/clip.py:28–40`](file:///G:/pinokio/api/roop-ultimate/app/clip/clip.py#L28-L40)).
    - `models/CodeFormer/`: CodeFormer FP32 & FP16 ONNX ([`app/roop/core.py:717–724`](file:///G:/pinokio/api/roop-ultimate/app/roop/core.py#L717-L724)).
    - `models/Frame/`: Upscalers and frame interpolation ([`app/roop/core.py:725–739`](file:///G:/pinokio/api/roop-ultimate/app/roop/core.py#L725-L739)).
    - `models/musetalk_hf_cache/`: Offline cache for MuseTalk ([`app/roop/processors/Lipsync_MuseTalk.py:88–120`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Lipsync_MuseTalk.py#L88-L120)).
    - `~/.cache/clip/`: Local fallback cache for CLIP ([`app/clip/clip.py:165–175`](file:///G:/pinokio/api/roop-ultimate/app/clip/clip.py#L165-L175)).
- **Resumable Atomic Model Integrity:**
  - [`app/roop/model_integrity.py:1–120`](file:///G:/pinokio/api/roop-ultimate/app/roop/model_integrity.py#L1-L120): Hashes models against `app/model_manifest.json` (schema 1, 19 models with file, family, size, sha256, required, urls).
  - Caches validated file digests in `models/.integrity.json`.
  - Downloads use HTTP Range resume to `<file>.part`, check SHA-256 and size before atomic `os.replace`.
  - Corrupt local files are moved aside to `.corrupt_<timestamp>` and restored if download fails.
- **Network Probing & Missing Model Policy:**
  - [`app/roop/utilities.py:956–997`](file:///G:/pinokio/api/roop-ultimate/app/roop/utilities.py#L956-L997) (`is_online`): Probes target host on port 443 with 30s cache. Never called per-frame.
  - [`app/roop/utilities.py:1000–1087`](file:///G:/pinokio/api/roop-ultimate/app/roop/utilities.py#L1000-L1087) (`conditional_download` & `_handle_missing_model`):
    - If model exists locally, skips network entirely (lines 1035–1036).
    - If absent and offline: `required=False` (startup pre-warm in `core.py:705–742`) logs `[OFFLINE]` warning and continues; `required=True` (feature selection) raises `RuntimeError("Model '<name>' is not available locally...")`.
- **Feature-Scoped Offline Loaders:**
  - [`app/roop/processors/Lipsync_MuseTalk.py:90–135`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Lipsync_MuseTalk.py#L90-L135): `cached_or_download` calls `loader(*args, local_files_only=True)` against `models/musetalk_hf_cache`; contacts huggingface.co only on cache miss.
  - [`app/clip/clip.py:42–108`](file:///G:/pinokio/api/roop-ultimate/app/clip/clip.py#L42-L108): `_download` validates local SHA-256 before network probe.
  - [`app/roop/processors/Enhance_KEEP.py:38–70`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_KEEP.py#L38-L70): Calls local loopback HTTP sidecar on `127.0.0.1` (`app/sidecar_keep/.venv`), not a remote service.
  - `app/routes_diagnostics.py` & `app/update_manager.py`: `GET /api/update/check` is user-triggered only, cached 60s, never runs on boot; returns `UNVERIFIED` with local git identity when offline.
- **UI Offline Guarantee:**
  - [`react-ui/src/api.js:5`](file:///G:/pinokio/api/roop-ultimate/react-ui/src/api.js#L5): `export const API = window.location.origin;` (proxied to `127.0.0.1`).
  - `react-ui/index.html`: Zero external CDN fonts, scripts, or styles.
  - [`app/tests/test_offline_operation.py:1–95`](file:///G:/pinokio/api/roop-ultimate/app/tests/test_offline_operation.py#L1-L95): Verifies HUD reports local engine state without simulated internet latency, verifies cache-first load and atomic part downloads.

### 5.2 Checkpoint / Resume Schema & State Persistence
- **Project Checkpoint Schema:**
  - [`app/project_checkpoint.py:40–46`](file:///G:/pinokio/api/roop-ultimate/app/project_checkpoint.py#L40-L46): `PROJECT_SCHEMA_VERSION = 1`, `COMPATIBILITY = {"project_schema": 1, "processing_contract": "segmented-video-v1"}`.
  - Saved in `app/projects/<project_id>.json`.
  - States: `STATES = ("PAUSED", "INTERRUPTED", "RECOVERABLE", "FAILED", "COMPLETED", "PROCESSING")`.
  - Atomic persistence: `_atomic_write` ([`app/project_checkpoint.py:83–131`](file:///G:/pinokio/api/roop-ultimate/app/project_checkpoint.py#L83-L131)) dumps JSON to `.checkpoint-*.tmp`, flushes/fsyncs, and retries `os.replace` 8 times with exponential backoff under `_RECORD_LOCK` to prevent WinError 5/32 collision.
  - Validations & recovery: `validate` (lines 758–815) validates inputs (size, mtime, SHA-256), provider compatibility, and segment sequence. `relink_input` (lines 816–860) relinks moved media.
- **Stage 1 & Stage 2 Cache Sidecars:**
  - [`app/stage_checkpoint.py:1–65`](file:///G:/pinokio/api/roop-ultimate/app/stage_checkpoint.py#L1-L65) and [`app/roop/stage_cache.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/stage_cache.py): Cache Stage 1 (sparse scan) and Stage 2 (pre-pass) outputs into project sidecars `app/projects/<project_id>_stage1.json` and `_stage2.json` keyed by input media and non-volatile env vars (`VOLATILE_ENV_PREFIXES` ignored).
- **Server-Side Queue Schema & Resumption:**
  - [`app/routes_queue.py:94–102`](file:///G:/pinokio/api/roop-ultimate/app/routes_queue.py#L94-L102): `QUEUE_FILE = "app/queue.json"`, `SCHEMA_VERSION = 2`.
  - Survives app restart. On boot, `_normalize_loaded_job` (lines 205–225) converts all in-flight states (`PREPARING`, `PROCESSING`, `PAUSE_REQUESTED`, `PAUSED`) into `RECOVERABLE`.
  - `queue_resume` (lines 1142–1185) & `resume_project_job` (lines 1199–1225): validate project checkpoint and resume from `safe_frame`.
- **Cooperative Pause & Output-Safe Boundary:**
  - [`app/roop/procmgr_runtime.py:42–173`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_runtime.py#L42-L173) (`PauseController`): Implements cooperative pause.
  - `pause_scope` / `wait_while_paused` (lines 1488–1540, 2077–2086): Block worker at safe checkpoint, preserving model sessions, memory pools, and in-flight queues.
- **Sub-Segment Write-Ahead Journal & Segment Rotation:**
  - [`app/roop/journal.py:35–70`](file:///G:/pinokio/api/roop-ultimate/app/roop/journal.py#L35-L70): `JOURNAL_SCHEMA_VERSION = 1`. Writes `<job_dir>/journal.json` atomically after every completed segment and every 100 frames (`DEFAULT_UNSEGMENTED_INTERVAL`). Auto-recovers using ffprobe to discard truncated tail files and resume at exact next frame.
  - [`app/roop/segment_writer.py:1–85`](file:///G:/pinokio/api/roop-ultimate/app/roop/segment_writer.py#L1-L85): `SegmentedVideoWriter` rotates video into `seg_XXXX.mp4` every `ROOP_RESUME_CHUNK` (default 1000 frames) with `<output>.manifest.json`. On finish or stop, segments are losslessly merged via FFmpeg concat demuxer (`-c copy`).

### 5.3 Standalone Install Invariants & Package Dependencies
- **Physical Self-Containment Checks** ([`app/tests/test_standalone_install.py:1–215`](file:///G:/pinokio/api/roop-ultimate/app/tests/test_standalone_install.py#L1-L215)):
  - **Invariant 1** (`LOCAL_DIRS = ('env', 'models', 'facesets')`): Must be physical, local directories in `app/`.
  - **Invariant 2** (Reparse point check, lines 33–61): Must NOT be NTFS junctions or symlinks (`_is_reparse_point` checks `FILE_ATTRIBUTE_REPARSE_POINT = 0x400` via `os.stat`).
  - **Invariant 3** (Virtual environment ownership, lines 63–78): `sys.prefix` must resolve to `app/env`.
  - **Invariant 4** (Activate script alignment, lines 79–124): `Scripts/activate.bat` (or `bin/activate`) must record `VIRTUAL_ENV` equal to real `app/env`. Fixed by `tools/repair_venv_paths.py`.
  - **Invariant 5** (Git exclusion, lines 125–149): `app/env`, `app/models`, `app/facesets` (~49 GB) must be ignored in `.gitignore` (tested via `git check-ignore`).
  - **Invariant 6** (Zero upstream coupling, lines 150–213): Scans tracked files for forbidden needles `('roop-unleashed', 'roop_unleashed', 'C0untFloyd', 's0md3v')`. Allowed only in `NOTICE.md`, `docs/SESSION_LOGS.md`, rule pointers, and `test_standalone_install.py`.
- **Package Contracts & Fail-Closed Provisioning:**
  - `app/requirements.txt`: Pins core app dependencies (`numpy==1.26.4`, `insightface==0.7.3`, `diffusers==0.30.2`, `PyNvVideoCodec>=2.2,<3`, `triton-windows>=3.3,<3.4`, etc.).
  - [`app/provision_runtime.py:1–80`](file:///G:/pinokio/api/roop-ultimate/app/provision_runtime.py#L1-L80): Detects system/GPU architecture and provisions matching `onnxruntime` (`onnxruntime-gpu`, `directml`, `rocm`, `silicon`, `cpu`) and `tensorrt-cu12` packages fail-closed.
  - `react-ui/package.json`: Private standalone React app (Node `^20.19.0 || >=22.12.0`, React 19.2, Vite 8.1, Tailwind 4.3, Zustand 5.0).

### 5.4 Hardware Profile Invariants: RTX 4070 Desktop vs. RTX 3060 Laptop
| Metric / Policy | RTX 4070 Desktop (Main) | RTX 3060 Laptop (Secondary) |
|---|---|---|
| **GPU / VRAM** | RTX 4070 12.0 GB (PCIe 4.0 x16, 200W TDP) | RTX 3060 Laptop 6.0 GB (Mobile TDP) |
| **System RAM** | 32 GB (31.7 GB physical) | 16 GB (15.8 GB physical) |
| **CPU Topology** | 24 Cores / 32 Threads (i9-14900K) | Mobile Multi-core |
| **VRAM Tier** | `rtx_4070_desktop` (`vram_total_gb >= 11.5`) | `rtx_3060_laptop` (`vram_total_gb < 7`) |
| **Session Pools** | `perf_trt_pool: 2`, `perf_detmask_pool: 2`, `perf_detector_pool: 2` | `0 / 0` (Single context + global `_gpu_guard` lock) |
| **Execution Provider** | TensorRT FP16 / mixed precision (+20% speedup) | CUDAExecutionProvider (TensorRT **inert / not admitted**) |
| **VRAM Budget** | ~9.07 GB used, ~3.0 GB unfragmented headroom | Strict $< 2.5\text{ GB}$ system RSS ceiling |
| **Stabilization** | `hard_cap: 4096 MB`, 12-thread parallel execution | `hard_cap: 1536 MB`, `adaptive_block = max(2*wu, 16)` |
| **Look Settings** | Default calibrated pipeline | **Hand-tuned custom settings must not be overwritten:** `blend_ratio: 0.85`, `face_mask_blend: 25`, `merger_sharpen: 0.55`, `stabilize_enhancer_strength: 0.6` |

---

## 6. Ordered Checklist of Exact Files to Touch (Stages 1–6)

### Stage 1: Pipeline Decoupling, Telemetry & Invariants
- [ ] [`app/roop/core.py:1438–1553, 1708–1805`](file:///G:/pinokio/api/roop-ultimate/app/roop/core.py#L1438-L1553) — Enforce strict frame-index ordering; propagate temporal flags without silent configuration overrides.
- [ ] [`app/roop/procmgr_batch.py:44–350, 810–840, 1125–1220`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_batch.py#L44-L350) — Restructure queue dispatch; eliminate strided round-robin execution for stateful pipelines; guarantee ordered sequential blocks.
- [ ] [`app/roop/procmgr_stabilization.py:25–110, 247–290`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_stabilization.py#L25-L110) — Tune `_stab_warmup_frames` and block geometry calculations to guarantee $\le 1\%$ boundary seed residual.
- [ ] [`app/roop/unified_runtime_scheduler.py:80–220`](file:///G:/pinokio/api/roop-ultimate/app/roop/unified_runtime_scheduler.py#L80-L220) — Ensure temporal filter state persistence without thread-migration corruption.
- [ ] [`app/roop/stage_profiler.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/stage_profiler.py) & [`app/roop/stage_telemetry.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/stage_telemetry.py) — Wire microsecond telemetry hooks for detector, swapper, mask, restorer, and blend stages.
- [ ] [`app/roop/watchdog.py:262–324, 375–440`](file:///G:/pinokio/api/roop-ultimate/app/roop/watchdog.py#L262-L324) — Add telemetry logging for tensor checks and mask geometry inspection overhead.
- [ ] [`app/settings.py:35–45, 60–70`](file:///G:/pinokio/api/roop-ultimate/app/settings.py#L35-L45) — Register Stage 1 settings, sync with `tools/gen_settings.py`, and update schema/catalog artifacts.

### Stage 2: Session Pooling & Memory Architecture
- [ ] [`app/roop/session_pool.py:25–180`](file:///G:/pinokio/api/roop-ultimate/app/roop/session_pool.py#L25-L180) — Configure session pooling for `swapper:realswap`, `mask:realityux`, and `enhancer:gpenrealistic`.
- [ ] [`app/roop/face_util.py:280–370, 653–685`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L280-L370) — Thread-safe detector instance leasing (`_ANALYSER_Q`) and release mechanisms.
- [ ] [`app/roop/retinaface.py:350–410`](file:///G:/pinokio/api/roop-ultimate/app/roop/retinaface.py#L350-L410), [`app/roop/yoloface.py:50–85`](file:///G:/pinokio/api/roop-ultimate/app/roop/yoloface.py#L50-L85), [`app/roop/yunet.py:38–70`](file:///G:/pinokio/api/roop-ultimate/app/roop/yunet.py#L38-L70) — Thread-safe per-engine instance pools.
- [ ] [`app/roop/processors/FaceSwapInsightFace.py:75–115, 1025–1040`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/FaceSwapInsightFace.py#L75-L115) — Dual-session residency management for RealSwap (Hyperswap primary + HifiFace secondary) under 6 GB VRAM profile.
- [ ] [`app/roop/processors/Mask_RealityUX.py:270–289`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Mask_RealityUX.py#L270-L289) — Manage thread pool executor (`_executor`) and VRAM lifecycle for concurrent XSeg + BiSeNet execution.
- [ ] [`app/roop/processors/Enhance_GPENRealistic.py:240–266`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_GPENRealistic.py#L240-L266) & [`Enhance_RestoreFormerPPlus.py:45–95`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_RestoreFormerPPlus.py#L45-L95) — Input tensor caching and zero-copy CUDA memory buffer reclamation.
- [ ] [`app/roop/faceset_manager.py:179–250`](file:///G:/pinokio/api/roop-ultimate/app/roop/faceset_manager.py#L179-L250) — Model switching and re-embedding caches.

### Stage 3: Detection Continuity, Multi-Scale Recall & Tracking
- [ ] [`app/roop/face_detector.py:145–270, 328–438, 499–740`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_detector.py#L145-L270) — `MultiScaleFaceDetector`, `apply_context_padding`, and `diou_nms` optimization.
- [ ] [`app/roop/adaptive_detector.py:35–150`](file:///G:/pinokio/api/roop-ultimate/app/roop/adaptive_detector.py#L35-L150) — Motion thresholds, profile yaw gating ($\ge 45^\circ$), and ROI rescue passes.
- [ ] [`app/roop/orientation.py:45–160`](file:///G:/pinokio/api/roop-ultimate/app/roop/orientation.py#L45-L160) — Head roll bounds (`ROLL_LOWER = 54.5`, `ROLL_UPPER = 135.0`), track roll prediction, and coasting.
- [ ] [`app/roop/procmgr_tracking.py:279–450, 1945–2100, 3176–3450`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_tracking.py#L279-L450) —
  - `_build_temporal_faces` (lines 3176–3450): Coupled 5-point keypoint + 106-point landmark smoothing; optical flow gap bridging (`_flow_bridge_geometry`).
  - `_assign_track_sources` (lines 1945–2100): Relax `assign_max` rejection thresholds; stabilize multi-person track assignments.
- [ ] [`app/roop/tracker.py:25–180`](file:///G:/pinokio/api/roop-ultimate/app/roop/tracker.py#L25-L180) & [`app/roop/multi_face_tracker.py:40–160`](file:///G:/pinokio/api/roop-ultimate/app/roop/multi_face_tracker.py#L40-L160) — Kalman motion prediction tuning; Hungarian cost matrix balancing to prevent track ID reassignment.
- [ ] [`app/roop/one_euro.py:20–150`](file:///G:/pinokio/api/roop-ultimate/app/roop/one_euro.py#L20-L150) — Cutoff and beta parameter optimization for low-jitter, zero-lag landmark filtering.

### Stage 4: Geometric Transformation, Landmark Consistency & Side-Angle Stability
- [ ] [`app/roop/face_analyser.py:72–234, 695–760`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_analyser.py#L72-L234) — `estimate_head_pose_pnp`, `compute_canonical_roll_angle`, `canonicalize_face_alignment`, and `weighted_umeyama_alignment`.
- [ ] [`app/roop/face_util.py:687–715, 2935–3330, 3450–3535`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_util.py#L687-L715) — `_refine_kps_from_68`, `solve_pose_5pt`, `solve_pose_jaw_5pt`, and `estimate_norm`.
- [ ] [`app/roop/geometric_alignment.py:60–250`](file:///G:/pinokio/api/roop-ultimate/app/roop/geometric_alignment.py#L60-L250) — `profile_stable_anchor_alignment`, 3-point anchor solver, and matrix condition checks.
- [ ] [`app/roop/head_pose.py:60–190`](file:///G:/pinokio/api/roop-ultimate/app/roop/head_pose.py#L60-L190) — `HeadPose` normalization and `head_pose_from_kps`.
- [ ] [`app/roop/pose_template.py:20–130`](file:///G:/pinokio/api/roop-ultimate/app/roop/pose_template.py#L20-L130) — Angle-projected affine pose templates.
- [ ] [`app/roop/alignment_safety.py:10–150`](file:///G:/pinokio/api/roop-ultimate/app/roop/alignment_safety.py#L10-L150) — 5-point validation and `TrackGeometry` numerical stability.
- [ ] [`app/roop/face_frontalize.py:25–120`](file:///G:/pinokio/api/roop-ultimate/app/roop/face_frontalize.py#L25-L120) — Coordinate space preservation in `frontalize_crop` and `defrontalize_crop`.
- [ ] [`app/roop/ProcessMgr.py:5930–6007, 6924–6982, 7045–7056`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L5930-L6007) — Re-alignment pipeline between ArcFace swapper crop and FFHQ restorer template; fix 1–2px subpixel nostril/lip shifts.
- [ ] [`app/roop/processors/FaceSwapInsightFace.py:1720–1745`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/FaceSwapInsightFace.py#L1720-L1745) — Adjust RealSwap eyelid mask geometry (`_OUTER_X = 0.44`, `_OUTER_Y = 0.30`) to eliminate eyelid band bleed across the nose bridge.

### Stage 5: Swapping Engines, High-Likeness Compositing & Multi-Angle Source Selection
- [ ] [`app/roop/FaceSet.py:51–120, 164–370`](file:///G:/pinokio/api/roop-ultimate/app/roop/FaceSet.py#L51-L120) — `AverageEmbeddings`, `canonical_poses`, `select_reference_index_canonical`, and `get_pose_matched_embedding`.
- [ ] [`app/roop/faceset_v2.py:287–328, 1294–1450`](file:///G:/pinokio/api/roop-ultimate/app/roop/faceset_v2.py#L287-L328) — `pose_matrix_cell`, GPU SLERP interpolation (`slerp_vectors`), and `select_reference_index`.
- [ ] [`app/roop/pose_source_selector.py:180–398`](file:///G:/pinokio/api/roop-ultimate/app/roop/pose_source_selector.py#L180-L398) — `select_pose_aware_source`, `score_source_entry`, and hysteresis latch.
- [ ] [`app/roop/angle_portfolio.py:40–150`](file:///G:/pinokio/api/roop-ultimate/app/roop/angle_portfolio.py#L40-L150) — 9-bin yaw/pitch angle portfolio.
- [ ] [`app/roop/processors/FaceSwapInsightFace.py:525–536, 1526–1850`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/FaceSwapInsightFace.py#L525-L536) — RealSwap dual-net model tuning, `_mix_outputs`, `_eye_region_mask`, enforce `_BASE_MIX == 0.0`, and smooth yaw fade curves (`_LATERAL_FADE_DEG = 35`, `_LATERAL_SKIP_DEG = 65`).
- [ ] [`app/roop/procmgr_tiling.py:20–150`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_tiling.py#L20-L150) — Pixel boost tiling (`implode_pixel_boost`, `explode_pixel_boost`) and tile mask reassembly (`_explode_mask`).
- [ ] [`app/roop/swap_identity.py`](file:///G:/pinokio/api/roop-ultimate/app/roop/swap_identity.py) & [`app/roop/appearance_conditioning.py:30–120`](file:///G:/pinokio/api/roop-ultimate/app/roop/appearance_conditioning.py#L30-L120) — ArcFace 512-d recognition embedding normalization; target lighting decoupling.
- [ ] [`app/roop/ProcessMgr.py:4000–4350, 4568–4680`](file:///G:/pinokio/api/roop-ultimate/app/roop/ProcessMgr.py#L4000-L4350) — Stabilize face candidate sorting by track ID; replace memoryless single-frame source vetoes (`VETO_FAR_FROM_OWN`) with temporal hysteresis.
- [ ] [`app/roop/procmgr_runtime.py:960–1030`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_runtime.py#L960-L1030) — Tune outcome guard parameters: `VERIFY_SWAP`, `VERIFY_MIN_OFFAXIS = 30.0`, `VERIFY_AWAY_YAW = 88.0`, `VERIFY_AWAY_MOVED_TOL = 0.5`, `VERIFY_AWAY_SHAPE_TOL = 0.3`.

### Stage 6: Restorers, Mask Fusion, Seamless Blending & UI Harmonization
- [ ] [`app/roop/processors/Enhance_GPENRealistic.py:275–302, 393–474`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_GPENRealistic.py#L275-L302) — Zero-copy and luminance-only color preservation (`_keep_source_colour`).
- [ ] [`app/roop/processors/Enhance_UltraMax.py:19–30`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_UltraMax.py#L19-L30) — Lean host path gather LUT and saturating scale.
- [ ] [`app/roop/processors/Enhance_CodeFormer.py:204–281`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_CodeFormer.py#L204-L281) & [`Enhance_GFPGAN.py:50–65`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Enhance_GFPGAN.py#L50-L65) — FP32 TensorRT precision enforcement to prevent grey-frame collapse.
- [ ] [`app/roop/enhancer_stabilizer.py:145–280`](file:///G:/pinokio/api/roop-ultimate/app/roop/enhancer_stabilizer.py#L145-L280) — Enhance `TemporalEnhancerBlend` with higher-resolution dense optical flow (DIS/Farneback); widen photometric deadband to prevent shimmer bursts.
- [ ] [`app/roop/temporal_smoother.py:340–450`](file:///G:/pinokio/api/roop-ultimate/app/roop/temporal_smoother.py#L340-L450) — `HighFrequencyFlowStabilizer` multi-band frequency separation and motion-compensated high-frequency smoothing.
- [ ] [`app/roop/processors/Mask_RealityUX.py:367–432`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Mask_RealityUX.py#L367-L432) — Fix `accessory_allowed` suppression gate so BiSeNet can exclude occlusions even when XSeg misses them; smoothstep edge guided filtering.
- [ ] [`app/roop/processors/Mask_Occluder.py:94–114`](file:///G:/pinokio/api/roop-ultimate/app/roop/processors/Mask_Occluder.py#L94-L114) & [`procmgr_masking.py:1823`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1823) — Replace hard `> 0.35` binary thresholding with continuous distance transform / bilateral guided filter.
- [ ] [`app/roop/nonfrontal.py:45–140`](file:///G:/pinokio/api/roop-ultimate/app/roop/nonfrontal.py#L45-L140) & [`procmgr_masking.py:1655–1790`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L1655-L1790) — `NonFrontalRouter`, `nonfrontal_score`, and crop-box boundary stabilization.
- [ ] [`app/roop/procmgr_color.py:338–450, 532–590`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_color.py#L338-L450) — Temporal smoothing on LAB target mean and standard deviation $(\mu_L, \mu_A, \mu_B, \sigma_L, \sigma_A, \sigma_B)$ in `_color_transfer_rct` and `_color_transfer_lct`.
- [ ] [`app/roop/temporal_color.py:1–180`](file:///G:/pinokio/api/roop-ultimate/app/roop/temporal_color.py#L1-L180) — Scene-cut-aware temporal color affine filtering (`lct_affine` / `rct_affine`) without tone pumping.
- [ ] [`app/roop/appearance_conditioning.py:50–220`](file:///G:/pinokio/api/roop-ultimate/app/roop/appearance_conditioning.py#L50-L220) — Quantile-anchored temporal luminance and chrominance stabilization in `TargetAppearanceStabilizer`.
- [ ] [`app/roop/procmgr_merger.py:181–234`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_merger.py#L181-L234) — Clarity, sharpen, and histogram match ordering in `apply_merger_post`.
- [ ] [`app/roop/procmgr_masking.py:747–1150`](file:///G:/pinokio/api/roop-ultimate/app/roop/procmgr_masking.py#L747-L1150) (`paste_upscale`) — Bounded ROI translation, multi-band Laplacian blending, rim-only illumination matching (`boundary_illumination_match`), and inward alpha feathering (`inward_alpha`).
- [ ] [`app/roop/unified_matte.py:45–69`](file:///G:/pinokio/api/roop-ultimate/app/roop/unified_matte.py#L45-L69) — Smoothstep distance transform feathering.
- [ ] [`app/roop/blender.py:10–120`](file:///G:/pinokio/api/roop-ultimate/app/roop/blender.py#L10-L120) — Multi-band Laplacian pyramid blending and skin Lab matching.
- [ ] [`app/utilities.py:115–160`](file:///G:/pinokio/api/roop-ultimate/app/utilities.py#L115-L160) — Persistent `CudaOrtIOBinding` buffer reuse to eliminate per-inference allocation overhead.
- [ ] `react-ui/` surface controls:
  - [`react-ui/src/components/Settings.jsx`](file:///G:/pinokio/api/roop-ultimate/react-ui/src/components/Settings.jsx) & [`settingsTabs.js`](file:///G:/pinokio/api/roop-ultimate/react-ui/src/components/settingsTabs.js)
  - [`react-ui/src/components/faceswap/PipelineControls.jsx`](file:///G:/pinokio/api/roop-ultimate/react-ui/src/components/faceswap/PipelineControls.jsx)
  - [`react-ui/src/components/faceswap/FacePoseGizmo.jsx`](file:///G:/pinokio/api/roop-ultimate/react-ui/src/components/faceswap/FacePoseGizmo.jsx)
  - [`react-ui/src/components/faceswap/EnhancerPipelineStack.jsx`](file:///G:/pinokio/api/roop-ultimate/react-ui/src/components/faceswap/EnhancerPipelineStack.jsx)
  - [`react-ui/src/components/faceswap/ColorGradingPanel.jsx`](file:///G:/pinokio/api/roop-ultimate/react-ui/src/components/faceswap/ColorGradingPanel.jsx)
  - [`react-ui/src/components/LiveTelemetry.jsx`](file:///G:/pinokio/api/roop-ultimate/react-ui/src/components/LiveTelemetry.jsx)
