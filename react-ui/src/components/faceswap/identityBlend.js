import { getJSON } from '../../api';
import { reportFailure } from '../../failureLog';

// Identity Blender recipe shape (app/roop/identity_algebra.py BlendRecipe).
//
// The dials the panel offers. identity_render_validation.json (65 paired faces through the swapper) marks
// age, gender and expression `writable: false`; jawline is the one that is writable. A recipe the server
// hands back (BlendRecipe.from_payload fills all four names with 0) or an older saved one can still hold a
// value for a removed dial, and with no slider to clear it, it would keep `has_dials` true for a no-op.
// normalizeBlend is the one place that drops them.
export const BLEND_DIALS = Object.freeze(['jawline']);
export const MIN_COSINE_DEFAULT = 0.8;

export const EMPTY_BLEND = Object.freeze({
  enabled: false,
  components: [],
  dials: Object.freeze({ jawline: 0 }),
  min_cosine: MIN_COSINE_DEFAULT,
});

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
