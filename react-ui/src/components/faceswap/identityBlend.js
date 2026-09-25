import { getJSON } from '../../api';

// Identity Blender recipe shape (app/roop/identity_algebra.py BlendRecipe).
export const EMPTY_BLEND = Object.freeze({
  enabled: false,
  components: [],
  dials: { age: 0, gender: 0, jawline: 0, expression: 0 },
  min_cosine: 0.8,
});

// Server-side recipe on mount: a Pinokio tab switch reloads the frontend, and
// the backend's live global is the state to restore (see ui-rehydrate memory).
export async function fetchIdentityBlend() {
  try {
    const res = await getJSON('/api/identity/blend');
    return res?.recipe || null;
  } catch { return null; }
}
