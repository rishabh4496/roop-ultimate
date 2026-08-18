// Baked-in defaults for every setting editable inside the Face Swap tab.
// Snapshot of the user's preferred configuration (taken 2026-07-12 from
// app/config.yaml). The "Reset defaults" button in FaceSwap.jsx merges this
// over the live settings and persists it to the backend CFG.
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
  face_detector_threshold: 0.7,
  face_detector_nms: 0.4,
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
  selected_enhancer: 'Restoreformer++',
  codeformer_fidelity: 0.5,
  max_face_distance: 0.75,
  subsample_upscale: '128px',
  upscale_after_swap: true,
  upscale_model_after: 'esrganx2',
  interp_after_swap: 'off',
  color_transfer_mode: 'lct',
  blend_ratio: 1,

  // Masking parameters
  mask_engine: 'DFL XSeg',
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
  face_mask_blend: 20,

  // Mouth & Angle math
  mouth_top_scale: 1,
  mouth_bottom_scale: 1,
  mouth_left_scale: 1,
  mouth_right_scale: 1,
  mouth_mask_blend: 10,
  use_3d_recon: true,
  use_source_bank: true,
  use_frontalization: false,
  frontalization_threshold: 25,
  jaw_reshape: false,
  jaw_reshape_strength: 0.5,
  detail_transfer_strength: 0,
  // DeepFaceLab merger post-ops. All neutral by default — each is a
  // bit-identical no-op at 0, so the defaults change nothing about a render.
  merger_hist_match: 0,
  merger_sharpen: 0,
  merger_motion_blur: 0,
  merger_grain_match: 0,
  merger_degrade: 0,
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
  stabilize_enhancer_strength: 0.5,
  stabilize_mask: true,
  stabilize_mask_strength: 0.5,

  // System options
  autorotate_faces: true,
  skip_audio: false,
  keep_frames: false,
  wait_after_extraction: false,

  // Enhancements
  track_identities: true,
  stabilize_face: true,
  stabilize_method: 'one_euro',
  stabilize_min_cutoff: 0.05,
  stabilize_beta: 0.02,
  restore_original_mouth: false,
  restore_original_eyes: false,
  eyes_blend_amount: 1,
  eyes_feather_blend: 25,
  eyes_size_factor: 1,
  eyes_radius_x: 1,
  eyes_radius_y: 1,
  parser_regions: ['skin', 'brows', 'eyes', 'nose', 'mouth'],
  parser_region_grow: {},
  enhancer_align: false,
  color_match_after_enhance: false,
  lipsync_enabled: false,
  lipsync_audio_source: 'original',
  lipsync_audio_path: null,

  // Output
  output_method: 'File',
};
