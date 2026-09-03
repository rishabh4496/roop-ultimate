// Baked-in defaults for every setting editable inside the Face Swap tab.
// Snapshot of the user's preferred configuration (reconciled 2026-08-22 against
// app/config.yaml and app/settings.py). The "Reset defaults" button in
// FaceSwap.jsx merges this over the live settings and persists it to the
// backend CFG.
//
// This file is ONLY read by "Reset defaults" (useUserDefaults.js) and as a
// lower-priority base in BatchSwap.jsx — live settings win there. It is NOT
// what a render uses. Changing a value here alone changes nothing about what
// runs: app/config.yaml is the live state and app/settings.py is what a fresh
// install gets, so all three have to move together or the stack silently
// disagrees with itself (it did, on 34 keys, until 2026-08-22).
// Deliberately excludes global/Settings-tab keys (provider, threads, theme,
// output codec/quality, server options, perf knobs) so a reset never touches
// anything outside this tab.
export const FACESWAP_DEFAULTS = {
  // Swap settings
  swap_model: 'realswap',
  face_detection_mode: 'Selected face',
  detector_engine: 'retinaface_r50',
  face_detector_size: '640',
  default_det_size: true,
  face_detector_threshold: 0.5,
  face_detector_nms: 0.3,
  detector_scale_pyramid: 'auto',
  refine_landmarks: true,
  // Angled-face alignment sits in the same alignment group as the two settings
  // either side of it and was the last Face Swap control missing here, so
  // "Reset defaults" restored its neighbours and silently left this one at
  // whatever it was — the same gap expression_restore_strength had below.
  // 'off'. All three angle layers below key on a 5-point pose solve that fits ONE
  // reference head, and nose protrusion carries most of the yaw signal — so a
  // prominent-nosed person turning 30° is read as 45° and gets a crop, a trim and
  // a fade meant for a pose they are not in. Kept selectable; see
  // The other two layers of the same angle structure.
  // Only hififace / hyperswap emit a mask; ignored by every other swapper.
  swap_model_mask_strength: 0,
  rescue_small_faces: true,
  num_swap_steps: 1,
  selected_enhancer: 'UltraMax',
  adaptive_enhancer_profile: 'BALANCED',
  codeformer_fidelity: 0.55,
  max_face_distance: 0.75,
  subsample_upscale: '256px',
  upscale_after_swap: false,
  upscale_model_after: 'fsr_x2',
  interp_after_swap: 'off',
  color_transfer_mode: 'lct',
  // Target-conditioned appearance is opt-in and preserves source identity.
  target_conditioned_appearance: false,
  target_conditioned_appearance_strength: 0.75,
  target_conditioned_appearance_temporal_alpha: 0.30,
  blend_ratio: 1,

  // Masking parameters
  mask_engine: 'RealityUX',
  // A second occlusion engine, composed as a union with the first. 'None' is
  // the previous behaviour; the pairing to reach for is XSeg + Face Occluder.
  mask_engine_2: 'None',
  mask_clip_text: 'cup,hands,hair,banana',
  sam2_model_size: 'tiny',
  show_mask_offsets: false,
  mask_top: 0,
  mask_bottom: 0,
  mask_left: 0,
  mask_right: 0,
  face_mask_blend: 12,
  // Paste-matte edge ramp shape. 'gaussian' is the shipped behaviour that
  // face_mask_blend was calibrated against; 'distance' gives a ramp of
  // constant width regardless of the matte's local curvature.
  mask_edge_mode: 'gaussian',
  boundary_illumination_strength: 0,

  // Mouth & Angle math
  mouth_top_scale: 1,
  mouth_bottom_scale: 1,
  mouth_left_scale: 1,
  mouth_right_scale: 1,
  mouth_mask_blend: 10,
  // Both measured OFF, twice, and kept off deliberately — a reset must not
  // switch them back on. use_source_bank costs 0.05-0.11 of identity at EVERY
  // yaw (re-verified 2026-08-15 under the production swapper), and handing the
  // swap a correctly pose-matched source plate by hand is still worse than the
  // averaged faceset embedding, so the loss is not its pose estimate. Its
  // partner use_3d_recon is a structural no-op for this swapper family anyway
  // (ProcessMgr gates the 3D pose-warp to image-source swappers).
  use_3d_recon: false,
  use_source_bank: false,
  use_frontalization: false,
  frontalization_threshold: 15,
  jaw_reshape: false,
  jaw_reshape_strength: 0.5,
  detail_transfer_strength: 0.4,
  // DeepFaceLab merger post-ops. NO LONGER neutral: hist/sharpen/grain carry
  // real values, so a reset changes the render. Order in procmgr_merger is
  // hist -> degrade -> sharpen -> motion -> grain, which is why degrade stays
  // at 0 — a degrade/sharpen round trip blurs and then re-sharpens the blur,
  // and apply_sharpen(1.0) is 2*image - 1*blur, twice a standard unsharp.
  merger_hist_match: 0.4,
  merger_sharpen: 0.35,
  merger_motion_blur: 0,
  merger_grain_match: 0.45,
  merger_degrade: 0,
  // LAB micro-clarity on L + a soft-knee bound on chrominance. Was baked into
  // Enhance_UltraMax; 1.0 is exactly what it applied. Works with ANY enhancer.
  merger_clarity: 1,
  output_face_scale: 0,
  // Expression restore is an editable Face Swap control and a heavy GPU stage,
  // but was missing here — so "Reset defaults" left it at whatever it was.
  expression_restore_strength: 0,
  expression_restore_region: 'all',

  // Video parameters
  video_swapping_method: 'In-Memory processing',
  no_face_action: 'Use untouched original frame',
  temporal_detection: true,
  vr_mode: false,
  stabilize_enhancer: true,
  stabilize_enhancer_strength: 0.25,
  stabilize_mask: true,
  stabilize_mask_strength: 0.5,
  // Dense-landmark smoothing rides on stabilize_face; it is what keeps the
  // paste matte's OUTLINE steady, which the 5-point kps filter never touched.
  stabilize_landmarks: true,
  stabilize_hf_texture: false,
  stabilize_hf_texture_weight: 0.15,

  // System options
  autorotate_faces: true,
  skip_audio: false,
  keep_frames: false,
  wait_after_extraction: false,

  // Enhancements
  track_identities: true,
  stabilize_face: true,
  stabilize_method: 'one_euro',
  stabilize_min_cutoff: 0.1,
  stabilize_beta: 0.1,
  restore_original_mouth: false,
  restore_original_eyes: false,
  eyes_blend_amount: 1,
  eyes_feather_blend: 25,
  eyes_size_factor: 1,
  eyes_radius_x: 1,
  eyes_radius_y: 1,
  parser_regions: ['skin', 'brows', 'eyes', 'nose', 'mouth'],
  glasses_frame_protect: true,
  parser_region_grow: {},
  enhancer_align: false,
  color_match_after_enhance: true,
  lipsync_enabled: false,
  lipsync_audio_source: 'original',
  lipsync_audio_path: null,

  // Output
  output_method: 'File',
};
