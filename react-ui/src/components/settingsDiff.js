// Shared vocabulary for describing and diffing a run's settings.
//
// Lived in RunHistory.jsx until the Outputs tab needed the same thing: when you
// put two rendered files side by side, the question that follows immediately is
// "what was different about them?", and answering it from a second copy of this
// table would let the two drift.

// Canonical settings keys → short human labels. Anything not listed still shows
// in a diff under its raw key; this map just gives the common ones nice names
// and drives the at-a-glance chip rows.
export const LABELS = {
  swap_model: 'Swapper',
  selected_enhancer: 'Enhancer',
  adaptive_enhancer_profile: 'Adaptive profile',
  face_detection_mode: 'Detection',
  mask_engine: 'Mask',
  mask_engine_2: 'Mask 2',
  max_threads: 'Threads',
  subsample_upscale: 'Pixel boost',
  video_swapping_method: 'Method',
  output_method: 'Output',
  blend_ratio: 'Blend',
  max_face_distance: 'Face distance',
  detector_engine: 'Detector',
  color_transfer_mode: 'Color transfer',
  target_conditioned_appearance: 'Target-conditioned lighting',
  codeformer_fidelity: 'CodeFormer fidelity',
  face_detector_size: 'Detector size',
  face_detector_threshold: 'Detector threshold',
  num_swap_steps: 'Swap steps',
  track_identities: 'Identity tracking',
  temporal_detection: 'Temporal detection',
  upscale_after_swap: 'AI upscale pass',
  upscale_model_after: 'Upscale model',
  interp_after_swap: 'Interpolation',
  refine_landmarks: 'Refine landmarks',
  swap_model_mask_strength: "Swap model's face mask",
  rescue_small_faces: 'Small-face rescue',
  jaw_reshape: 'Jaw reshape',
  autorotate_faces: 'Autorotate',
  vr_mode: 'VR mode',
  expression_restore_strength: 'Expression restore',
  detail_transfer_strength: 'Detail transfer',
  merger_hist_match: 'Histogram match',
  merger_sharpen: 'Sharpen / soften',
  merger_motion_blur: 'Motion blur',
  merger_grain_match: 'Grain match',
  merger_degrade: 'Degrade',
  output_face_scale: 'Face size',
  stabilize_face: 'Stabilize face',
};

// The handful of settings that most define what a run looked like, shown as
// chips. Missing/empty ones are skipped so the row stays tight.
export const CHIP_KEYS = ['swap_model', 'selected_enhancer', 'face_detection_mode',
                          'mask_engine', 'subsample_upscale', 'max_threads'];

// Keys that are noise in a side-by-side diff (paths, per-run junk, huge blobs).
export const DIFF_SKIP = new Set(['selected_theme', 'output_path', 'source_path',
                                  'target_path', 'clear_output', 'output_show_video']);

export const fmtVal = (v) => {
  if (v === true) return 'on';
  if (v === false) return 'off';
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2);
  return String(v);
};

export const fmtDur = (s) => {
  if (!s || s <= 0) return null;
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
};

// Primitive-only view of a run's settings, for chips + diffing. Objects/arrays
// (e.g. face_mapping) are not comparable at a glance, so they're dropped.
export const primitives = (settings) => {
  const out = {};
  Object.entries(settings || {}).forEach(([k, v]) => {
    if (DIFF_SKIP.has(k)) return;
    if (v === null || ['string', 'number', 'boolean'].includes(typeof v)) out[k] = v;
  });
  return out;
};

/** Union of both settings objects, split into what differs and what matches. */
export const diffSettings = (a, b) => {
  const pa = primitives(a), pb = primitives(b);
  const keys = Array.from(new Set([...Object.keys(pa), ...Object.keys(pb)])).sort();
  const changed = [], same = [];
  keys.forEach((k) => {
    const va = pa[k], vb = pb[k];
    (fmtVal(va) === fmtVal(vb) ? same : changed).push({ k, va, vb });
  });
  return { changed, same };
};
