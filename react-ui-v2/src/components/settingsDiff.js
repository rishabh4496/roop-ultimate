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
  output_method: 'Output',
  blend_ratio: 'Blend',
  max_face_distance: 'Face distance',
  detector_engine: 'Detector',
  color_transfer_mode: 'Color transfer',
  target_conditioned_appearance: 'Target-conditioned lighting',
  codeformer_fidelity: 'CodeFormer fidelity',
  ultramax_swap_lowband: 'UltraMax keeps swap tone/shape',
  ultramax_detail_weight: 'UltraMax detail',
  restoration_mode: 'Restoration mode',
  tile_diffusion_strength: 'Tile diffusion strength',
  tile_diffusion_steps: 'Tile diffusion steps',
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
  expression_gaze_follow: 'Eye-gaze follow',
  expression_blink_sync: 'Blink sync',
  pupil_gaze_retention: 'Pupil & eye-gaze retention',
  micro_expression_restore: 'Micro-expression & lip-sync tension',
  inner_mouth_preserve: 'Inner-mouth & teeth preservation',
  identity_lock: 'Identity crossover lock',
  identity_lock_margin: 'Identity lock margin',
  identity_lock_hold: 'Identity lock hold',
  detail_transfer_strength: 'Detail transfer',
  merger_hist_match: 'Histogram match',
  merger_sharpen: 'Sharpen / soften',
  merger_match_sharpness: 'Match sharpness to the footage',
  merger_sharpness_headroom: 'Sharpness headroom',
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

// ── checkpoint vs current ────────────────────────────────────────────────────
// History's "Compare Checkpoint vs. Current Settings": the saved run on the left, what is set NOW on
// the right, one row per parameter that differs, and a restore that writes the CHECKPOINT's value back.

// Pixel-valued settings read as "12px", not "12".
export const UNITS = {
  face_mask_blend: 'px',
  mouth_mask_blend: 'px',
  mask_erode_dilate_radius: 'px',
  eyes_feather_blend: 'px',
  face_detector_size: 'px',
};

// Labels the run chips never needed but a diff does.
Object.assign(LABELS, {
  face_mask_blend: 'Mask feather',
  mouth_mask_blend: 'Mouth mask feather',
  mask_erode_dilate_radius: 'Mask erode / dilate',
  restore_ultra_detail_weight: 'Restoration detail weight',
  stabilize_enhancer_strength: 'Flicker reduction strength',
  eyes_blend_amount: 'Eyes blend amount',
});

export const labelFor = (k) => LABELS[k] || k.replace(/_/g, ' ');
export const fmtWithUnit = (k, v) => {
  const base = fmtVal(v);
  return UNITS[k] && base !== '—' && typeof v === 'number' ? `${base}${UNITS[k]}` : base;
};

/**
 * Rows for every parameter whose checkpoint value differs from the current one.
 *   changed          set in both, different
 *   only-checkpoint  the checkpoint set it and the current settings have no value (restoring adds it)
 *   only-current     the current settings set it and the checkpoint never did (nothing to restore TO)
 * Sorted with the labelled, human-meaningful parameters first.
 */
export const checkpointDiff = (checkpoint, current) => {
  const a = primitives(checkpoint);
  const b = primitives(current);
  const rows = [];
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    const hasA = Object.prototype.hasOwnProperty.call(a, k) && a[k] !== undefined;
    const hasB = Object.prototype.hasOwnProperty.call(b, k) && b[k] !== undefined;
    if (hasA && hasB && fmtVal(a[k]) === fmtVal(b[k])) continue;
    rows.push({
      k,
      label: labelFor(k),
      known: k in LABELS,
      from: hasA ? a[k] : undefined,
      to: hasB ? b[k] : undefined,
      fromText: hasA ? fmtWithUnit(k, a[k]) : '—',
      toText: hasB ? fmtWithUnit(k, b[k]) : '—',
      kind: hasA && hasB ? 'changed' : hasA ? 'only-checkpoint' : 'only-current',
      restorable: hasA,
    });
  }
  rows.sort((x, y) => (Number(y.known) - Number(x.known)) || x.label.localeCompare(y.label));
  return rows;
};

/** {key: checkpointValue} for the selected rows that can be restored. Never writes a key the checkpoint lacks. */
export const restorePatch = (rows, selectedKeys) => {
  const want = new Set(selectedKeys);
  const patch = {};
  for (const r of rows) if (want.has(r.k) && r.restorable) patch[r.k] = r.from;
  return patch;
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
