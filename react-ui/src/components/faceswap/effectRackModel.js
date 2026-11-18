// The effect rack as data: which modules there are, what switches each one, which real setting is its wet/dry mix, and what it
// costs. No React, no DOM: `.render-check/stage5-rack-check.mjs` runs it under node.
//
// NOTHING HERE INVENTS A SETTING. Every key below is one the render already reads (app/settings.py): `selected_enhancer`,
// `blend_ratio`, `merger_grain_match`, `color_transfer_mode`, `merger_hist_match`, `mask_engine_2`, `upscale_after_swap`,
// `upscale_model_after`.
//
// NOT HERE, ON PURPOSE: `enable_xseg_occlusion` / `occlusion_erosion`. They are read by roop/enhancers/anatomical_engine.py and
// roop/api/enhancer_routes.py, and nothing in ProcessMgr / procmgr_masking reads them, so a rack card bound to them would look
// wired and change no render. The occlusion module is the SECOND MASKING ENGINE (`mask_engine_2`), which ProcessMgr does read.
//
// WHAT A "MIX" IS. A module has a wet/dry mix only where the backend has a setting that blends its result over the input:
//
//   Face Restorer     blend_ratio          1 = the restorer's face, 0 = the swap as it came out        (paste_upscale)
//   Texture & Grain   merger_grain_match   the amount of matched noise added; 0 = the stage is off
//   Colour Match      merger_hist_match    how far the face's histogram is pulled toward the plate; 0 = off
//   Occlusion Mask    none                 a second masking engine (XSeg, occluder, ...): it decides WHERE the swap shows
//   Frame Upscaling   none                 it replaces the frame, there is no blend setting
//
// so the last two show a disabled 100% bar and say why, rather than a slider bound to nothing.
//
// REORDERING IS NOT OFFERED. The engine runs the chain in a fixed order (swap, restore, grain, colour, masking at the paste,
// upscale after the swap pass); a drag handle would rearrange the cards and not the render.
//
// LATENCY. `latencyFor` returns a level (Low / Med / High VRAM) with its PROVENANCE: `measured` where the project has a number
// (the restorer costs printed beside the restorer selector, measured on an RTX 4070), `estimate` where the level is read from
// the kind of work (a DSP pass vs a neural network), which the UI says. No number here comes from the 3060.

export const STAGE_IDS = Object.freeze(['swapper', 'restorer', 'grain', 'color', 'occlusion', 'upscale']);

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
export const num = (v, d) => { const x = Number(v); return Number.isFinite(x) ? x : d; };

// ── latency ─────────────────────────────────────────────────────────────────────────────────────────────────────────
export const LATENCY_LEVELS = Object.freeze({
  low: { level: 'low', label: 'Low' },
  med: { level: 'med', label: 'Med' },
  high: { level: 'high', label: 'High VRAM' },
});

/** ms per face (RTX 4070, measured; the same figures the restorer selector's tooltip states) -> a level. */
export const levelForMs = (ms) => (ms < 8 ? 'low' : ms < 40 ? 'med' : 'high');

const RESTORER_MS = [
  [/^gpen[-_ ]?256/i, 3, 'GPEN-256'],
  [/^gpen[-_ ]?512/i, 20, 'GPEN-512'],
  [/restoreformer/i, 23, 'RestoreFormer++'],
  [/codeformer/i, 24, 'CodeFormer'],
];

const mk = (level, basis, note) => ({ ...LATENCY_LEVELS[level], basis, note });

export function latencyFor(stage, p = {}) {
  switch (stage) {
    case 'swapper':
      return { level: 'core', label: 'Core', basis: 'estimate', note: 'The swap itself always runs.' };
    case 'restorer': {
      const name = String(p.selected_enhancer || '');
      if (!name || name === 'None') return mk('low', 'estimate', 'Off: nothing runs.');
      const hit = RESTORER_MS.find(([re]) => re.test(name));
      if (hit) return mk(levelForMs(hit[1]), 'measured', `${hit[2]} ~${hit[1]} ms per face (RTX 4070).`);
      if (/ultramax/i.test(name)) return mk('high', 'estimate', 'UltraMax runs a restorer plus a detail pass per face; not measured here.');
      return mk('med', 'estimate', `${name}: no measurement in the project; assumed a neural restorer.`);
    }
    case 'grain':
      return mk('low', 'estimate', 'A DSP noise pass on the face crop; no network.');
    case 'color': {
      const mode = String(p.color_transfer_mode || 'none').toLowerCase();
      if (mode === 'idt') return mk('high', 'estimate', 'IDT matches the full colour distribution: the quality ceiling and much the dearest.');
      if (mode === 'mkl') return mk('med', 'estimate', 'MKL is a full linear match.');
      return mk('low', 'estimate', `${mode.toUpperCase()} is a per-channel statistic match.`);
    }
    case 'occlusion': {
      const e = String(p.mask_engine_2 || 'None');
      if (e === 'None') return mk('low', 'estimate', 'Off: nothing runs.');
      return mk('med', 'estimate', `${e} runs a segmentation network per face, after the first masking engine. Not measured here.`);
    }
    case 'upscale': {
      const m = String(p.upscale_model_after || '');
      if (/(lanczos|fsr|spline|sinc)/i.test(m)) return mk('low', 'estimate', 'A resampling filter, no network.');
      if (/compact/i.test(m)) return mk('med', 'estimate', 'Compact is the fast AI upscaler.');
      if (/x2$/i.test(m)) return mk('med', 'estimate', 'A ×2 network over the whole frame.');
      return mk('high', 'estimate', 'A ×4 network over the whole frame: the heaviest stage in the chain.');
    }
    default:
      return mk('med', 'estimate', '');
  }
}

// ── per-module state ────────────────────────────────────────────────────────────────────────────────────────────────
/** Which setting is the module's wet/dry mix, or null when the backend has none. */
export const MIX_KEYS = Object.freeze({ restorer: 'blend_ratio', grain: 'merger_grain_match', color: 'merger_hist_match' });

export const MIX_UNAVAILABLE = Object.freeze({
  occlusion: 'An occlusion engine decides where the swap shows. It is a mask, so it has no amount to blend.',
  upscale: 'The upscaler replaces the frame. The render has no blend setting for it.',
  swapper: 'The swapper is the swap itself.',
});

/** { enabled, mixKey, mix } for one module from the settings object `p` and the defaults `d`. */
export function stageState(stage, p = {}, d = {}) {
  switch (stage) {
    case 'swapper':
      return { enabled: true, locked: true, mixKey: null, mix: 1 };
    case 'restorer': {
      const e = p.selected_enhancer ?? d.selected_enhancer;
      return { enabled: !!e && e !== 'None', mixKey: 'blend_ratio', mix: num(p.blend_ratio, d.blend_ratio ?? 1), mixDefault: d.blend_ratio ?? 1 };
    }
    case 'grain': {
      const g = num(p.merger_grain_match, d.merger_grain_match ?? 0);
      return { enabled: g > 0, mixKey: 'merger_grain_match', mix: g, mixDefault: d.merger_grain_match ?? 0 };
    }
    case 'color': {
      const m = p.color_transfer_mode || d.color_transfer_mode || 'none';
      return { enabled: m !== 'none', mixKey: 'merger_hist_match', mix: num(p.merger_hist_match, d.merger_hist_match ?? 0), mixDefault: d.merger_hist_match ?? 0 };
    }
    case 'occlusion': {
      const e = p.mask_engine_2 ?? d.mask_engine_2 ?? 'None';
      return { enabled: !!e && e !== 'None', mixKey: null, mix: 1 };
    }
    case 'upscale':
      return { enabled: !!p.upscale_after_swap, mixKey: null, mix: 1 };
    default:
      return { enabled: false, mixKey: null, mix: 1 };
  }
}

/**
 * The write a power switch makes, as [key, value]. A switch that is turned off REMEMBERS what it replaced (the previous
 * restorer, grain amount, colour mode) in `memo` and puts it back when turned on, instead of snapping to a default.
 * `memo` is mutated; keep one per page load so the memory survives the rack unmounting with its tab.
 */
export function togglePatch(stage, on, p = {}, d = {}, memo = {}) {
  switch (stage) {
    case 'restorer': {
      if (on) return ['selected_enhancer', memo.restorer || d.selected_enhancer || 'UltraMax'];
      memo.restorer = p.selected_enhancer ?? d.selected_enhancer;
      return ['selected_enhancer', 'None'];
    }
    case 'grain': {
      if (on) return ['merger_grain_match', memo.grain || d.merger_grain_match || 0.45];
      memo.grain = num(p.merger_grain_match, d.merger_grain_match ?? 0);
      return ['merger_grain_match', 0];
    }
    case 'color': {
      if (on) return ['color_transfer_mode', memo.color || d.color_transfer_mode || 'lct'];
      memo.color = p.color_transfer_mode || d.color_transfer_mode;
      return ['color_transfer_mode', 'none'];
    }
    case 'occlusion': {
      if (on) return ['mask_engine_2', memo.occlusion || 'DFL XSeg'];
      memo.occlusion = p.mask_engine_2 ?? d.mask_engine_2;
      return ['mask_engine_2', 'None'];
    }
    case 'upscale': return ['upscale_after_swap', !!on];
    default: return null;
  }
}

/** One line under the module title. */
export function stageSummary(stage, p = {}, d = {}, swapperLabel = (k) => String(k ?? '')) {
  const st = stageState(stage, p, d);
  switch (stage) {
    case 'swapper': return swapperLabel(p.swap_model || d.swap_model);
    case 'restorer': return st.enabled ? String(p.selected_enhancer ?? d.selected_enhancer) : 'Bypassed';
    case 'grain': return st.enabled ? `Grain match ${st.mix.toFixed(2)}` : 'Bypassed';
    case 'color': return st.enabled ? `${String(p.color_transfer_mode || d.color_transfer_mode).toUpperCase()} · histogram ${st.mix.toFixed(2)}` : 'Bypassed';
    case 'occlusion': return st.enabled ? String(p.mask_engine_2 ?? d.mask_engine_2) : 'Bypassed';
    case 'upscale': return st.enabled ? String(p.upscale_model_after || d.upscale_model_after || '') : 'Bypassed';
    default: return '';
  }
}

export const isPercentMix = (stage) => stage === 'restorer' || stage === 'grain' || stage === 'color';
export { isNum };
