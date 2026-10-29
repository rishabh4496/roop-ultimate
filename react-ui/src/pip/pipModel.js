// The floating mini-viewport's rules, as pure functions: where it may sit, how it resizes, when it shows.
// No React, no DOM - the component feeds in pixels and applies what comes back, and .render-check proves the rules.
//
// All geometry is in LAYOUT pixels (the unit CSS `left` / `width` use), not visual ones. The app applies its UI zoom as CSS
// `zoom` on <html>, so a pointer delta arrives in visual pixels and must be divided by the element's own scale before it is
// added to a position (faceswap/zoomPan.uiScale); the component does that and hands this module layout pixels.

export const PIP_DEFAULT = Object.freeze({ w: 280, h: 158 });
export const PIP_MIN_W = 200;
export const PIP_MAX_W = 720;
export const PIP_MARGIN = 16;
/** Releasing within this distance of a corner snaps (docks) to it. */
export const PIP_SNAP = 72;
/** The header strip's height; the picture area is `h` and the strip sits on top, so the whole window is h + PIP_BAR. */
export const PIP_BAR = 28;

/** The tabs the window follows you to: the ones a user goes to WHILE TUNING a project, away from the preview. */
export const PIP_TABS = Object.freeze(['facemgr', 'batch', 'settings']);

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/**
 * Should the window be on screen?
 *   enabled      the user turned it on
 *   hasFrame     there is a frame to show (the Face Swap tab has rendered one this session)
 *   tab          the current tab id
 *   stageVisible whether the Face Swap preview stage is in view (always true off the Face Swap tab)
 * It follows you to PIP_TABS, and on the Face Swap tab itself when the stage is scrolled or collapsed out of view.
 */
export function pipVisible({ enabled, hasFrame, tab, stageVisible }) {
  if (!enabled || !hasFrame) return false;
  if (tab === 'faceswap') return !stageVisible;
  return PIP_TABS.includes(tab);
}

/** Width/height of the picture area for a frame of aspect `ar` at width `w` (a missing aspect falls back to 16:9). */
export function pictureHeight(w, ar) {
  const a = Number.isFinite(ar) && ar > 0 ? ar : PIP_DEFAULT.w / PIP_DEFAULT.h;
  return Math.round(w / a);
}

/** Keep the whole window (picture + bar) on screen. */
export function clampPosition({ x, y }, { w, h }, { vw, vh }, margin = PIP_MARGIN) {
  const total = h + PIP_BAR;
  return {
    x: clamp(x, margin, Math.max(margin, vw - w - margin)),
    y: clamp(y, margin, Math.max(margin, vh - total - margin)),
  };
}

/** The four docked positions, in layout pixels. */
export function cornerPositions({ w, h }, { vw, vh }, margin = PIP_MARGIN) {
  const total = h + PIP_BAR;
  const right = Math.max(margin, vw - w - margin);
  const bottom = Math.max(margin, vh - total - margin);
  return {
    'top-left': { x: margin, y: margin },
    'top-right': { x: right, y: margin },
    'bottom-left': { x: margin, y: bottom },
    'bottom-right': { x: right, y: bottom },
  };
}

export const CORNER_ORDER = ['bottom-right', 'bottom-left', 'top-left', 'top-right'];

/** The corner to snap to if the window is within PIP_SNAP of one, else null. */
export function snapCorner(pos, size, viewport, snap = PIP_SNAP) {
  const corners = cornerPositions(size, viewport);
  let best = null;
  let bestD = snap;
  for (const [name, c] of Object.entries(corners)) {
    const d = Math.hypot(pos.x - c.x, pos.y - c.y);
    if (d <= bestD) { best = name; bestD = d; }
  }
  return best;
}

/** The next corner when the dock button is pressed. */
export function nextCorner(current) {
  const i = CORNER_ORDER.indexOf(current);
  return CORNER_ORDER[(i + 1) % CORNER_ORDER.length];
}

/** A resize to width `w`: clamped to the allowed range and to what fits on screen. The picture keeps its aspect. */
export function resizeTo(w, ar, pos, { vw, vh }, margin = PIP_MARGIN) {
  const aspect = Number.isFinite(ar) && ar > 0 ? ar : PIP_DEFAULT.w / PIP_DEFAULT.h;
  const roomW = Math.max(PIP_MIN_W, vw - pos.x - margin);
  let nw = clamp(Math.round(w), PIP_MIN_W, Math.min(PIP_MAX_W, roomW));
  // Too tall for the room under it: narrow it until it fits (never below the minimum width), keeping the aspect.
  const roomH = vh - pos.y - margin - PIP_BAR;
  if (pictureHeight(nw, aspect) > roomH) nw = Math.max(PIP_MIN_W, Math.floor(roomH * aspect));
  return { w: nw, h: pictureHeight(nw, aspect) };
}

/** Everything persisted: size, position and the corner it is docked to (null when free-floating). */
export function sanitizeSaved(raw) {
  const out = { enabled: false, w: PIP_DEFAULT.w, corner: 'bottom-right', x: null, y: null };
  if (!raw || typeof raw !== 'object') return out;
  if (typeof raw.enabled === 'boolean') out.enabled = raw.enabled;
  if (Number.isFinite(raw.w)) out.w = clamp(Math.round(raw.w), PIP_MIN_W, PIP_MAX_W);
  if (raw.corner === null || CORNER_ORDER.includes(raw.corner)) out.corner = raw.corner;
  if (Number.isFinite(raw.x) && Number.isFinite(raw.y)) { out.x = raw.x; out.y = raw.y; }
  return out;
}

// ── Refreshing the picture while Face Swap is not mounted ──────────────────
//
// FaceSwap owns every /api/preview request while it is on screen. When it is not (you are in Face Manager), the PiP asks
// for the same picture again - the same target frame, settings and people - but with the SOURCE the backend has selected
// NOW, because that is what changed: Morph Lab's Preview and a faceset load both re-select the live source on the server.
// The source is taken the way FaceSwap itself takes it when it mounts (hydrateWorkspaceFromState): `selected_source_id` from
// /api/state, the numeric index only as a fallback.

/** Which successful requests change the live source, and so make the PiP's picture stale. */
const SOURCE_CHANGING = [
  /^\/api\/facemgr\/morph\/preview$/,       // loads the blend as a source and selects it
  /^\/api\/faceset\/library\/load$/,        // loads a library faceset as the source
];
export const changesLiveSource = (method, path) =>
  String(method).toUpperCase() !== 'GET' && SOURCE_CHANGING.some((re) => re.test(String(path).split('?')[0]));

/** The id/name of the source the backend has selected, from an /api/state reply; null if it does not say. */
export function selectedSourceOf(state) {
  const info = Array.isArray(state?.source_faces_info) ? state.source_faces_info : [];
  const byId = state?.selected_source_id
    ? info.find((i) => String(i?.id) === String(state.selected_source_id)) : null;
  const idx = Number.isInteger(state?.selected_source_index) && state.selected_source_index >= 0
    && state.selected_source_index < info.length ? state.selected_source_index : (info.length ? 0 : -1);
  const chosen = byId || (idx >= 0 ? info[idx] : null);
  const id = state?.selected_source_id ?? chosen?.id ?? null;
  if (id === null || id === undefined || id === '') return null;
  return { id: String(id), name: chosen?.name ?? null };
}

/**
 * The body for a refresh: `last` is the body of the last /api/preview FaceSwap sent. Never mutates it.
 *   - a fresh request_id (the response is echoed back against it) and a selection_version that moved on
 *   - the live source, if /api/state named one
 */
export function replayBody(last, { state = null, requestId, version } = {}) {
  const body = { ...last };
  const src = selectedSourceOf(state);
  if (src) {
    body.selected_source_id = src.id;
    if (src.name !== null) body.selected_source_name = src.name;
  }
  if (last?.processing_selection && typeof last.processing_selection === 'object') {
    body.processing_selection = {
      ...last.processing_selection,
      request_id: requestId ?? last.processing_selection.request_id,
      selection_version: Number.isFinite(version) ? version : last.processing_selection.selection_version,
      ...(src ? { source_identity_id: src.id } : {}),
    };
  }
  if (Number.isFinite(version)) body.selection_version = version;
  return body;
}
