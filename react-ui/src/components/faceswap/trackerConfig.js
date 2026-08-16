// Single source of truth for the Slider Tracker bar.
//
// These numbers used to live in four places — TRACKER_SLIDERS' defaultVal, the
// 'Default' built-in preset, resetTrackerSliders() and the bypass object inside
// buildPreviewPayload() — and had already drifted apart. Deriving the two value
// maps from the one slider list keeps them honest.
//
// Note that DEFAULT and BYPASS are deliberately NOT the same thing:
//   DEFAULT — the value the slider ships at ("↺ Reset", the default pip).
//   BYPASS  — the value that makes the stage a no-op ("Slider Effect: OFF").
// For jaw reshape and flicker reduction the default is 0.5 but the no-op is 0,
// so collapsing them would silently turn "bypass" into "half strength".
//
// `group` splits the bar into labelled, collapsible sections. Every slider must
// name one — an unlisted group would render a section the header row never
// accounts for, so GROUP_ORDER below is the whole set and TRACKER_GROUPS is
// derived from it rather than hand-listed.
//
// min/max/step here MUST match the same control in FaceSwap.jsx's settings
// panel: they are two views of one setting, and a range that disagreed would
// let one view produce a value the other cannot represent.
// tests/test_ui_slider_tracker.py checks that.

export const GROUP_CORE = 'Swap & blend';
export const GROUP_MERGER = 'Merger post-processing';
export const GROUP_ORDER = [GROUP_CORE, GROUP_MERGER];

export const TRACKER_SLIDERS = [
  {
    key: 'blend_ratio',
    group: GROUP_CORE,
    label: 'Original / Enhanced Blend',
    min: 0,
    max: 1,
    step: 0.01,
    defaultVal: 0.8,
    bypassVal: 0.8,
    format: (v) => Number(v ?? 0.8).toFixed(2),
    info: 'Blends between original target face and enhanced swapped result.',
  },
  {
    key: 'detail_transfer_strength',
    group: GROUP_CORE,
    label: 'Skin Detail Transfer',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0,
    bypassVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: "Transfers target's original skin texture (pores, grain, stubble) onto swapped face.",
  },
  {
    key: 'expression_restore_strength',
    group: GROUP_CORE,
    label: 'Expression Restore',
    min: 0,
    max: 2,
    step: 0.05,
    defaultVal: 0,
    bypassVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: "Re-applies target's original expression using LivePortrait model.",
  },
  {
    key: 'face_mask_blend',
    group: GROUP_CORE,
    label: 'Face Mask Edge Blend',
    min: 0,
    max: 200,
    step: 1,
    defaultVal: 20,
    bypassVal: 20,
    format: (v) => `${Math.round(v ?? 20)}px`,
    info: 'Feathering softness around the border of the face mask.',
  },
  {
    key: 'max_face_distance',
    group: GROUP_CORE,
    label: 'Max Face Distance',
    min: 0.01,
    max: 1,
    step: 0.01,
    // Mid-gap. Measured on a hard clip: the same person stayed under ~0.66,
    // different people sat at ~0.93-1.07. 0.75 clears the same-person ceiling
    // with headroom for a worse frame than that clip contained, while staying
    // well under stranger range. See procmgr_runtime._TRACK_VETO_DIST.
    defaultVal: 0.75,
    bypassVal: 0.75,
    format: (v) => Number(v ?? 0.75).toFixed(2),
    info: 'Cosine distance (0-2), NOT a similarity: higher is MORE permissive. '
        + 'Same person measured under ~0.66, different people ~0.93-1.07.',
  },
  {
    key: 'num_swap_steps',
    group: GROUP_CORE,
    label: 'Swapping Steps',
    min: 1,
    max: 5,
    step: 1,
    defaultVal: 1,
    bypassVal: 1,
    format: (v) => `${Math.round(v ?? 1)}×`,
    info: 'Swap iteration passes. Higher values increase source likeness.',
  },
  {
    key: 'jaw_reshape_strength',
    group: GROUP_CORE,
    label: 'Jaw Reshape Strength',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0.5,
    bypassVal: 0,
    format: (v) => Number(v ?? 0.5).toFixed(2),
    info: 'Blends source jaw & chin contours onto target face.',
  },
  {
    key: 'stabilize_enhancer_strength',
    group: GROUP_CORE,
    label: 'Flicker Reduction',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0.5,
    bypassVal: 0,
    format: (v) => Number(v ?? 0.5).toFixed(2),
    info: 'Temporal stabilization strength for video frames.',
  },
  {
    key: 'stabilize_mask_strength',
    group: GROUP_CORE,
    label: 'Mask Flicker Reduction',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0.5,
    bypassVal: 0,
    format: (v) => Number(v ?? 0.5).toFixed(2),
    info: 'Temporal smoothing strength for the mask edge (the swap/original blend boundary).',
  },

  // ── DeepFaceLab merger post-ops ────────────────────────────────────────
  // Every one is a no-op at 0, so default and bypass are the same value here
  // and the whole group vanishes from the render when untouched.
  {
    key: 'output_face_scale',
    group: GROUP_MERGER,
    label: 'Face Size',
    min: -0.2,
    max: 0.2,
    step: 0.01,
    defaultVal: 0,
    bypassVal: 0,
    // Signed, and shown as a percent: "+6%" reads as a size change where
    // "0.06" reads as a strength like every other slider on the bar.
    format: (v) => `${Number(v ?? 0) > 0 ? '+' : ''}${Math.round(Number(v ?? 0) * 100)}%`,
    info: 'Grows or shrinks the pasted face about its own centre. Swappers keep '
        + "the TARGET's head size, so this is the lever for a narrower or broader source.",
  },
  {
    key: 'merger_grain_match',
    group: GROUP_MERGER,
    label: 'Grain Match',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0,
    bypassVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: "Adds noise matched to the footage's own measured noise floor. Fixes the "
        + 'common "pasted" look, which is usually the face being too CLEAN for the shot.',
  },
  {
    key: 'merger_sharpen',
    group: GROUP_MERGER,
    label: 'Sharpen / Soften',
    min: -1,
    max: 1,
    step: 0.05,
    defaultVal: 0,
    bypassVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: 'Signed unsharp mask. Positive sharpens; negative softens, which is the '
        + 'denoise direction for crunchy over-enhanced skin.',
  },
  {
    key: 'merger_motion_blur',
    group: GROUP_MERGER,
    label: 'Motion Blur',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0,
    bypassVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: 'Blurs the face along the direction the camera smeared it. You set the '
        + "amount; the direction is measured off the original crop.",
  },
  {
    key: 'merger_hist_match',
    group: GROUP_MERGER,
    label: 'Histogram Match',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0,
    bypassVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: "Matches the face's per-channel histogram to the original crop — "
        + 'distribution shape, where RCT/LCT/MKL match statistical moments.',
  },
  {
    key: 'merger_degrade',
    group: GROUP_MERGER,
    label: 'Degrade',
    min: 0,
    max: 1,
    step: 0.05,
    defaultVal: 0,
    bypassVal: 0,
    format: (v) => Number(v ?? 0).toFixed(2),
    info: 'Bicubic down-and-up, to match a soft or heavily compressed plate that a '
        + 'crisp 512px swap does not belong in.',
  },
];

export const TRACKER_DEFAULT_VALUES = Object.fromEntries(
  TRACKER_SLIDERS.map((s) => [s.key, s.defaultVal])
);

export const TRACKER_BYPASS_VALUES = Object.fromEntries(
  TRACKER_SLIDERS.map((s) => [s.key, s.bypassVal])
);

// [{ name, sliders }] in GROUP_ORDER order — derived, so adding a slider above
// puts it in its section with nothing else to update. Empty groups are dropped
// rather than rendered as a header with nothing under it.
export const TRACKER_GROUPS = GROUP_ORDER
  .map((name) => ({ name, sliders: TRACKER_SLIDERS.filter((s) => s.group === name) }))
  .filter((g) => g.sliders.length > 0);
