// The four quick recipes on the inspector's preset bar, as DATA.
//
// A recipe is a block of `key: value` pairs written straight into the live
// settings, exactly as the old Fast / Balanced / Quality presets were. That is
// the part that can silently go wrong (an invented key is written, ignored by
// the backend and echoed back; a value outside an option list leaves a
// <select> blank), so every key here is a real setting in faceswap/defaults.js
// and every enum value is one api.py's /api/meta offers.
// app/tests/test_ui_preset_recipes.py::test_quick_recipes checks both.
//
// What a recipe does NOT touch, on purpose:
//   * global / Settings-tab keys (execution provider, threads, pools, codecs):
//     the provider is read when inference sessions are built, so a recipe could
//     only claim to have changed it. "Max Fidelity TRT" therefore sets the look
//     and says in its hint that TensorRT is chosen in Settings.
//   * the strength knobs a person tunes by eye (stabilize_enhancer_strength,
//     merger_sharpen on the 3060, ...). A recipe flips features and names a
//     model; it does not re-tune the secondary machine's hand-set look.
export const QUICK_RECIPES = [
  {
    id: 'fast',
    label: 'Fast Preview',
    hint: '320 px detection, 128 px swap, no restorer, no flicker passes. For scrubbing and checking framing.',
    patch: {
      default_det_size: false,
      face_detector_size: '320',
      face_detector_threshold: 0.5,
      subsample_upscale: '128px',
      selected_enhancer: 'None',
      num_swap_steps: 1,
      stabilize_enhancer: false,
      stabilize_mask: false,
      stabilize_hf_texture: false,
    },
  },
  {
    id: 'natural',
    label: 'Natural Studio',
    hint: 'The shipped look: UltraMax keeping the swap\'s tone, LCT colour, grain and light sharpening.',
    patch: {
      default_det_size: true,
      face_detector_size: '640',
      subsample_upscale: '256px',
      selected_enhancer: 'UltraMax',
      ultramax_swap_lowband: true,
      ultramax_detail_weight: 0.75,
      color_transfer_mode: 'lct',
      merger_grain_match: 0.45,
      merger_hist_match: 0.4,
      merger_sharpen: 0.35,
      detail_transfer_strength: 0.4,
      face_mask_blend: 12,
      mask_edge_mode: 'gaussian',
      stabilize_enhancer: true,
      stabilize_mask: true,
      stabilize_hf_texture: true,
    },
  },
  {
    id: 'fidelity',
    label: 'Max Fidelity TRT',
    hint: 'realswap, 512 px swap, Restore Ultra keeping identity. Slowest; run it on TensorRT (chosen in Settings).',
    patch: {
      swap_model: 'realswap',
      default_det_size: true,
      face_detector_size: '640',
      refine_landmarks: true,
      subsample_upscale: '512px',
      selected_enhancer: 'Restore Ultra',
      restore_ultra_frequency_blend: true,
      restore_ultra_detail_weight: 0.75,
      color_transfer_mode: 'lct',
      merger_hist_match: 0.4,
      mask_engine: 'RealityUX',
      stabilize_enhancer: true,
    },
  },
  {
    id: 'occlusion',
    label: 'High-Motion Occlusion',
    hint: 'Hands, hair and motion blur: RealityUX + Face Occluder, XSeg occlusion, flow-warped mask smoothing, permissive matching.',
    patch: {
      mask_engine: 'RealityUX',
      mask_engine_2: 'Face Occluder',
      enable_xseg_occlusion: true,
      occlusion_erosion: 3,
      face_mask_blend: 16,
      stabilize_mask: true,
      stabilize_mask_strength: 0.6,
      mask_flow_warp: true,
      temporal_detection: true,
      track_identities: true,
      rescue_small_faces: true,
      face_detector_threshold: 0.4,
      max_face_distance: 0.8,
      stabilize_face: true,
      stabilize_method: 'one_euro',
      stabilize_beta: 0.1,
    },
  },
];

const NEAR = 1e-6;

/** One key of a recipe against the live settings. `default_det_size` is true when absent, like everywhere else. */
function sameValue(key, want, have) {
  if (key === 'default_det_size') return (have !== false) === (want !== false);
  if (typeof want === 'boolean') return !!have === want;
  if (typeof want === 'number') return Math.abs(Number(have) - want) < NEAR;
  return have === want;
}

/** True when every key of `patch` already holds in `settings`. */
export const recipeMatches = (settings, patch) => {
  const s = settings || {};
  return Object.entries(patch).every(([k, v]) => sameValue(k, v, s[k]));
};

/** The id of the recipe the settings are currently exactly in, or null. */
export const activeRecipeId = (settings) => {
  const hit = QUICK_RECIPES.find((r) => recipeMatches(settings, r.patch));
  return hit ? hit.id : null;
};
