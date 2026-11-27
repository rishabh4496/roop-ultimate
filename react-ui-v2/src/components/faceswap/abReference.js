// The A/B reference: a previous render parked for comparison against the Face Swap preview.
//
// Outputs sends one here; Face Swap shows it beside the live preview. It travels in localStorage so it
// survives Pinokio reloading the webview on every tab switch, and in a same-document event so Face Swap
// updates the moment Outputs sets it. It is a convenience, never state the server owns: a quota error or
// private mode means "no reference", not a failure.

const KEY = 'roop_ab_reference';
const EVENT = 'roop:ab-reference';

/** A reference is {url, name, kind: 'image'|'video', fps?, offset?}. Anything else is rejected. */
export function normalizeReference(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const url = typeof raw.url === 'string' ? raw.url.trim() : '';
  if (!url) return null;
  const kind = raw.kind === 'video' ? 'video' : 'image';
  const fps = Number(raw.fps);
  const offset = Math.round(Number(raw.offset));
  return {
    url,
    name: typeof raw.name === 'string' && raw.name ? raw.name : url.split('/').pop() || 'reference',
    kind,
    fps: Number.isFinite(fps) && fps > 0 ? fps : null,
    // Frames between the clip's frame 1 and this render's first frame (a trimmed render starts later).
    offset: Number.isFinite(offset) && offset > 0 ? offset : 0,
    // The run that produced it, for "what changed" later; optional.
    runId: raw.runId ?? null,
  };
}

export function loadReference() {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? normalizeReference(JSON.parse(raw)) : null;
  } catch { return null; }
}

export function saveReference(ref) {
  const clean = normalizeReference(ref);
  try {
    if (clean) window.localStorage.setItem(KEY, JSON.stringify(clean));
    else window.localStorage.removeItem(KEY);
  } catch { /* storage unavailable: the event below still reaches a mounted Face Swap */ }
  try { window.dispatchEvent(new CustomEvent(EVENT, { detail: clean })); } catch { /* no window */ }
  return clean;
}

export const clearReference = () => saveReference(null);

/** Calls `fn(reference|null)` on every change, from this document or another tab. Returns the unsubscribe. */
export function subscribeReference(fn) {
  const onEvent = (e) => fn(e.detail ?? null);
  const onStorage = (e) => { if (e.key === KEY || e.key === null) fn(loadReference()); };
  window.addEventListener(EVENT, onEvent);
  window.addEventListener('storage', onStorage);
  return () => {
    window.removeEventListener(EVENT, onEvent);
    window.removeEventListener('storage', onStorage);
  };
}

/**
 * Where in the reference video to look for clip frame `frame` (1-based): the render starts `offset`
 * frames into the clip, and the video has `duration` seconds. Clamped inside the media so a frame the
 * reference does not cover shows its nearest end instead of a black seek error.
 */
export function referenceTime(ref, frame, fps, duration = Infinity) {
  const rate = (ref?.fps || fps || 25);
  const t = (Math.max(1, frame) - 1 - (ref?.offset || 0)) / rate;
  const end = Number.isFinite(duration) ? Math.max(0, duration - 1 / rate) : Infinity;
  return Math.min(Math.max(0, t), end);
}

/** Does the reference cover this frame at all? (false = the picture shown is the nearest end, not the frame). */
export function referenceCovers(ref, frame, fps, duration) {
  if (!ref || ref.kind !== 'video' || !Number.isFinite(duration)) return true;
  const rate = (ref.fps || fps || 25);
  const t = (Math.max(1, frame) - 1 - (ref.offset || 0)) / rate;
  return t >= 0 && t <= duration;
}
