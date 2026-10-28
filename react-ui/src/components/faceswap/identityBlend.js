import { getJSON } from '../../api';
import { reportFailure } from '../../failureLog';
import { normalizeKeyframes } from './identityKeyframes';

// Identity Blender recipe shape (app/roop/identity_algebra.py BlendRecipe).
//
// The dials the panel offers. identity_render_validation.json (65 paired faces through the swapper) marks
// age, gender and expression `writable: false`; jawline is the one that is writable. A recipe the server
// hands back (BlendRecipe.from_payload fills all four names with 0) or an older saved one can still hold a
// value for a removed dial, and with no slider to clear it, it would keep `has_dials` true for a no-op.
// normalizeBlend is the one place that drops them.
export const BLEND_DIALS = Object.freeze(['jawline']);
export const MIN_COSINE_DEFAULT = 0.8;

// Stage 20. Mirrors of identity_algebra.py: MAX_BLEND_SOURCES, BLEND_MODES / DEFAULT_BLEND_MODE and
// DEFAULT_PRIVACY_COSINE. `tests/test_identity_stage20_routes.py` reads them back from the server's `limits`.
export const MAX_BLEND_SOURCES = 5;
export const BLEND_MODES = Object.freeze(['slerp', 'linear']);
export const BLEND_MODE_LABELS = Object.freeze({ slerp: 'Slerp', linear: 'Linear' });
export const DEFAULT_BLEND_MODE = 'slerp';
export const PRIVACY_COSINE_DEFAULT = 0.3;

export const EMPTY_DEID = Object.freeze({
  enabled: false, references: Object.freeze([]), max_cosine: PRIVACY_COSINE_DEFAULT, min_primary_cosine: 0,
});

export const EMPTY_BLEND = Object.freeze({
  enabled: false,
  components: [],
  dials: Object.freeze({ jawline: 0 }),
  min_cosine: MIN_COSINE_DEFAULT,
  mode: DEFAULT_BLEND_MODE,
  keyframes: Object.freeze([]),
  deidentify: EMPTY_DEID,
});

const num = (v, d) => { const n = Number(v); return Number.isFinite(n) ? n : d; };

function normalizeDeid(d) {
  const r = d && typeof d === 'object' ? d : {};
  const refs = [];
  for (const id of Array.isArray(r.references) ? r.references : []) {
    const sid = String(id ?? '').trim();
    if (sid && !refs.includes(sid) && refs.length < MAX_BLEND_SOURCES) refs.push(sid);
  }
  return {
    enabled: !!r.enabled,
    references: refs,
    max_cosine: Math.max(0, Math.min(0.95, num(r.max_cosine, PRIVACY_COSINE_DEFAULT))),
    min_primary_cosine: Math.max(0, Math.min(0.99, num(r.min_primary_cosine, 0))),
  };
}

/** A copy of `recipe` carrying only the offered dials (each finite, clamped to ±1); safe for null / partial input. */
export function normalizeBlend(recipe) {
  const r = recipe && typeof recipe === 'object' ? recipe : {};
  const dials = {};
  for (const key of BLEND_DIALS) {
    const v = Number(r.dials?.[key]);
    dials[key] = Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0;
  }
  const mc = Number(r.min_cosine);
  return {
    enabled: !!r.enabled,
    components: Array.isArray(r.components) ? r.components : [],
    dials,
    min_cosine: Number.isFinite(mc) ? mc : MIN_COSINE_DEFAULT,
    mode: BLEND_MODES.includes(r.mode) ? r.mode : DEFAULT_BLEND_MODE,
    keyframes: normalizeKeyframes(r.keyframes),
    deidentify: normalizeDeid(r.deidentify),
  };
}

// Server-side recipe on mount: a Pinokio tab switch reloads the frontend, and
// the backend's live global is the state to restore (see ui-rehydrate memory).
export async function fetchIdentityBlend() {
  try {
    const res = await getJSON('/api/identity/blend');
    return res?.recipe || null;
  } catch (err) { reportFailure('Loading the identity blend', err, { toast: false }); return null; }
}
