// The 3-stage pipeline controls as DATA: which swappers are offered and how they read, the three restorer modes and
// the settings each one writes, and what the pre-pass stride means. Pure functions of the settings object, so the
// Face Swap tab, the Settings tab and the render-check read the same rules and cannot disagree about what a
// control means.
//
// (The file is named pipelineRules, not pipelineControls: a name that differs from PipelineControls.jsx only by case
// resolves to the component on a case-insensitive filesystem.)
//
// Nothing here invents a backend setting. Every key below already exists in app/settings.py and is read by the
// render: `swap_model`, `selected_enhancer`, `restore_ultra_frequency_blend`, `auto_bypass_restorer_native_swap`
// (roop/smart_restorer_bypass.py), `prepass_detect_interval` (roop/prepass_policy.py) and `temporal_smart_skip`.

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

// ── Swapper ────────────────────────────────────────────────────────────────────────────────────────────────────
// The five the brief names, by the key /api/meta lists them under. `native` is the swapper's own resolution: it
// decides whether the restorer is worth running (a 128px swap is upscaled to the subsample size; a 256/512px one
// is not).
export const SWAPPER_PRESETS = [
  { value: 'inswapper', label: 'inswapper_128', native: 128 },
  { value: 'reswapper_256', label: 'ReSwapper 256', native: 256 },
  { value: 'alpha_256', label: 'Alpha 256', native: 256 },
  { value: 'hyperswap_256', label: 'HyperSwap 256', native: 256 },
  { value: 'hyperswap_512', label: 'HyperSwap 512', native: 512 },
];

/** The restorer auto-bypass (roop/smart_restorer_bypass.py) only fires for these swappers. */
export const BYPASS_SWAPPERS = /^(alpha_256|alphaface_256|hyperswap)/;

const PRESET_BY_VALUE = Object.fromEntries(SWAPPER_PRESETS.map((s) => [s.value, s]));

/** A swapper key as a person reads it; unknown keys are shown as they are, never hidden. */
export const swapperLabel = (key) => (PRESET_BY_VALUE[key] ? PRESET_BY_VALUE[key].label : String(key ?? ''));

/**
 * The swapper `<select>` as groups, for the Select primitive's `{ group, options }` form:
 *   Default            realswap (the shipped pair)
 *   256 - 512 px       the curated five, only those this backend lists
 *   Other swappers     everything else /api/meta lists, plus the current value if the backend no longer does
 * so nothing the backend offers is lost behind the short list, and the select never renders blank.
 */
export function swapperOptions(available, current) {
  const have = Array.isArray(available) && available.length ? available : SWAPPER_PRESETS.map((s) => s.value).concat('realswap');
  const known = new Set(['realswap', ...SWAPPER_PRESETS.map((s) => s.value)]);
  const groups = [];
  if (have.includes('realswap')) groups.push({ group: 'Default', options: [{ value: 'realswap', label: 'realswap (hyperswap + hififace)' }] });
  const curated = SWAPPER_PRESETS.filter((s) => have.includes(s.value));
  if (curated.length) groups.push({ group: 'Native 128 - 512 px', options: curated.map(({ value, label }) => ({ value, label })) });
  const rest = have.filter((k) => !known.has(k));
  if (current && !known.has(current) && !rest.includes(current)) rest.unshift(current);
  if (rest.length) groups.push({ group: 'Other swappers', options: rest.map((k) => ({ value: k, label: k })) });
  return groups;
}

// ── Restorer mode ──────────────────────────────────────────────────────────────────────────────────────────────
// One three-way choice over three real settings. "RF++" is RestoreFormer++: `Restoreformer++` runs it plainly,
// `Restore Ultra` runs it and recombines by frequency (`restore_ultra_frequency_blend`: the swap's low band + w x the
// restorer's high band, roop/enhance_blend.py - measured to keep identity the linear blend gives away).
export const RESTORER_MODES = [
  { value: 'bypass', label: 'Off / Auto-Bypass ≥256px' },
  { value: 'laplacian', label: 'Laplacian Split RF++' },
  { value: 'full', label: 'Full RF++' },
];
const CUSTOM = 'custom';

/**
 * Which mode the current settings are in. `bypass` wins whenever auto-bypass is on or no restorer is selected, since
 * both skip the neural restorer for the faces that don't need it. Any other restorer (UltraMax, GPEN, CodeFormer...)
 * is `custom`: the control says so instead of pretending to be one of the three.
 */
export function restorerModeOf(p) {
  const s = p || {};
  const enhancer = String(s.selected_enhancer ?? '');
  if (s.auto_bypass_restorer_native_swap || enhancer === 'None') return 'bypass';
  if (enhancer === 'Restore Ultra') return s.restore_ultra_frequency_blend === false ? 'full' : 'laplacian';
  if (enhancer === 'Restoreformer++') return 'full';
  return CUSTOM;
}

/** The settings patch that puts the pipeline into `mode`. Never touches a setting the mode does not own. */
export function restorerPatch(mode) {
  switch (mode) {
    case 'bypass':
      // Keeps the selected restorer (it still runs for faces the bypass does not cover); only turns the bypass on.
      return { auto_bypass_restorer_native_swap: true };
    case 'laplacian':
      return { selected_enhancer: 'Restore Ultra', restore_ultra_frequency_blend: true, auto_bypass_restorer_native_swap: false };
    case 'full':
      return { selected_enhancer: 'Restoreformer++', auto_bypass_restorer_native_swap: false };
    default:
      return {};          // `custom` is a read-only description of the current restorer, never a target
  }
}

/** The select's options: the three modes, plus the current custom restorer when that is what is active. */
export function restorerOptions(p) {
  const out = RESTORER_MODES.map(({ value, label }) => ({ value, label }));
  if (restorerModeOf(p) === CUSTOM) out.push({ value: CUSTOM, label: `Custom · ${p.selected_enhancer}` });
  return out;
}

/** One line under the select: what actually runs, in the backend's own terms. */
export function restorerNote(p) {
  const s = p || {};
  const mode = restorerModeOf(s);
  const enhancer = String(s.selected_enhancer ?? '') || '—';
  if (mode === 'bypass') {
    const swapper = String(s.swap_model ?? '');
    const applies = BYPASS_SWAPPERS.test(swapper);
    if (enhancer === 'None') return 'No restorer runs.';
    return applies
      ? `${enhancer} is skipped (DoG sharpen instead) for faces ≤320 px tall; larger faces still get it.`
      : `Auto-bypass only fires with Alpha 256 or HyperSwap; with ${swapperLabel(swapper) || 'this swapper'} ${enhancer} runs on every face.`;
  }
  if (mode === 'laplacian') return 'RestoreFormer++, recombined by frequency: the swap keeps the low band.';
  if (mode === 'full') return 'RestoreFormer++ output replaces the swapped crop (linear blend).';
  return `${enhancer} is selected below; pick a mode to replace it.`;
}

// ── Pre-pass stride ────────────────────────────────────────────────────────────────────────────────────────────
export const STRIDE_MIN = 1;
export const STRIDE_MAX = 8;
export const clampStride = (v) => {
  const n = Math.round(Number(v));
  return isNum(n) ? Math.min(STRIDE_MAX, Math.max(STRIDE_MIN, n)) : STRIDE_MIN;
};

/** What a stride means, so the slider's value is never a bare number: K=1 reads every frame, K>1 carries by flow. */
export function strideText(k) {
  const n = clampStride(k);
  return n === 1 ? 'Detect every frame' : `Detect every ${n}${n === 2 ? 'nd' : n === 3 ? 'rd' : 'th'} frame, carry the rest by optical flow`;
}
