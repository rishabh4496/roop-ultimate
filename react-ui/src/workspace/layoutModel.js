// Saveable workspace layouts, as pure functions: monitor aspect classes, hotkey slots, renames, and how a saved split-pane
// proportion is carried to a different monitor. No React, no DOM, no store - store/workspaceStore.js calls these and
// .render-check/render-stage15-logic.mjs proves them.
//
// WHAT A SAVED LAYOUT IS. { id, name, label, mode, sizes, aspect, hotkey, viewport }
//   sizes     the split-pane widths/heights in px (leftWidth, rightWidth, bottomHeight, queueWidth)
//   aspect    '16:9' | '21:9' | '32:9' | 'any' - the monitor shape the proportions were made for
//   hotkey    1..5 or null - the Alt+<n> slot that switches to it (one layout per slot)
//   viewport  { w, h } of the window it was saved in, so the same layout on a bigger or smaller monitor of the SAME shape can
//             be scaled to the same proportions instead of the same pixels (see scaleSizes)

export const ASPECTS = Object.freeze(['16:9', '21:9', '32:9']);
export const ASPECT_ANY = 'any';
export const HOTKEY_SLOTS = Object.freeze([1, 2, 3, 4, 5]);

export const ASPECT_HINT = Object.freeze({
  '16:9': '16:9 - 1920x1080, 2560x1440, 3840x2160 (and 16:10 laptops)',
  '21:9': '21:9 - 2560x1080, 3440x1440, 3840x1600',
  '32:9': '32:9 - 3840x1080, 5120x1440',
  any: 'Any monitor',
});

// A window's aspect, bucketed to the nearest standard monitor. The cut points sit between the standard ratios: 16:9 = 1.78
// (16:10 = 1.6, 4:3 = 1.33 fall in the first bucket), 21:9 = 2.33-2.4, 32:9 = 3.56.
const CUT_21_9 = 2.05;
const CUT_32_9 = 3.0;

/** '16:9' | '21:9' | '32:9' for a width x height; '16:9' for anything unmeasurable. */
export function classifyAspect(width, height) {
  const w = Number(width);
  const h = Number(height);
  if (!(w > 0) || !(h > 0)) return '16:9';
  const r = w / h;
  if (r >= CUT_32_9) return '32:9';
  if (r >= CUT_21_9) return '21:9';
  return '16:9';
}

export const isHotkeySlot = (n) => HOTKEY_SLOTS.includes(Number(n));


/**
 * One saved layout from storage, made safe: a name, a known aspect, a slot in 1..5 or null, finite sizes. Returns null for
 * something that is not a layout at all (no id or name). `clampSize(panel, value)` is the store's own limit function.
 */
export function normalizeLayout(raw, clampSize = (_p, v) => v) {
  if (!raw || typeof raw !== 'object') return null;
  const id = typeof raw.id === 'string' && raw.id ? raw.id : null;
  const name = String(raw.name ?? raw.label ?? '').trim();
  if (!id || !name) return null;
  const sizes = {};
  for (const panel of ['leftWidth', 'rightWidth', 'bottomHeight', 'queueWidth']) {
    const v = Number(raw.sizes?.[panel]);
    if (Number.isFinite(v)) sizes[panel] = clampSize(panel, v);
  }
  const vp = raw.viewport && Number(raw.viewport.w) > 0 && Number(raw.viewport.h) > 0
    ? { w: Math.round(Number(raw.viewport.w)), h: Math.round(Number(raw.viewport.h)) } : null;
  return {
    id,
    name,
    label: name,
    mode: typeof raw.mode === 'string' ? raw.mode : 'default',
    sizes,
    aspect: ASPECTS.includes(raw.aspect) ? raw.aspect : ASPECT_ANY,
    hotkey: isHotkeySlot(raw.hotkey) ? Number(raw.hotkey) : null,
    viewport: vp,
  };
}

/** A list from storage: each entry normalised, junk dropped, and a hotkey slot kept by only the FIRST layout that claims it. */
export function normalizeLayouts(list, clampSize) {
  if (!Array.isArray(list)) return [];
  const seen = new Set();
  const used = new Set();
  const out = [];
  for (const raw of list) {
    const l = normalizeLayout(raw, clampSize);
    if (!l || seen.has(l.id)) continue;
    seen.add(l.id);
    if (l.hotkey !== null) {
      if (used.has(l.hotkey)) l.hotkey = null;
      else used.add(l.hotkey);
    }
    out.push(l);
  }
  return out;
}

/** Bind `slot` (1..5, or null to unbind) to layout `id`. The slot moves: whichever layout held it loses it. */
export function assignHotkey(layouts, id, slot) {
  const want = slot === null || slot === undefined ? null : (isHotkeySlot(slot) ? Number(slot) : undefined);
  if (want === undefined || !layouts.some((l) => l.id === id)) return layouts;
  return layouts.map((l) => {
    if (l.id === id) return { ...l, hotkey: want };
    if (want !== null && l.hotkey === want) return { ...l, hotkey: null };
    return l;
  });
}

/** The layout that Alt+<slot> switches to, or null. */
export const layoutForSlot = (layouts, slot) => layouts.find((l) => l.hotkey === Number(slot)) || null;

/**
 * Rename. The name must be non-empty and not another layout's (case-insensitive) - two layouts called the same would be
 * indistinguishable in the menu. Returns { ok, layouts } or { ok: false, error }.
 */
export function renameLayout(layouts, id, name) {
  const next = String(name ?? '').trim().slice(0, 40);
  if (!next) return { ok: false, error: 'A layout needs a name' };
  if (!layouts.some((l) => l.id === id)) return { ok: false, error: 'No such layout' };
  if (layouts.some((l) => l.id !== id && l.name.toLowerCase() === next.toLowerCase())) {
    return { ok: false, error: `Another layout is already called "${next}"` };
  }
  return { ok: true, layouts: layouts.map((l) => (l.id === id ? { ...l, name: next, label: next } : l)) };
}

/** Change the monitor shape a layout is for. */
export const setAspect = (layouts, id, aspect) => (
  ASPECTS.includes(aspect) || aspect === ASPECT_ANY
    ? layouts.map((l) => (l.id === id ? { ...l, aspect } : l))
    : layouts);

/**
 * The layouts to offer on a monitor of `aspect`, best first: made for this shape, then for any, then for another shape.
 * Within a group the saved order stands (stable). Nothing is hidden - a 21:9 layout is still reachable on a 16:9 screen.
 */
export function orderForAspect(layouts, aspect) {
  const rank = (l) => (l.aspect === aspect ? 0 : l.aspect === ASPECT_ANY ? 1 : 2);
  return layouts.map((l, i) => ({ l, i })).sort((a, b) => rank(a.l) - rank(b.l) || a.i - b.i).map((x) => x.l);
}

/**
 * Carry a layout's pixel sizes to a window of a different size: the horizontal panels scale with the width, the timeline
 * with the height, so a layout saved at 3440 wide keeps its proportions on a 2560-wide monitor of the same shape. Only
 * done when the layout knows the window it was made in; otherwise the sizes are used as saved. `limits` clamps the result.
 */
export function scaleSizes(sizes, from, to, clampSize = (_p, v) => v, tolerance = 0.08) {
  if (!from || !to || !(from.w > 0) || !(from.h > 0) || !(to.w > 0) || !(to.h > 0)) return sizes;
  const kw = to.w / from.w;
  const kh = to.h / from.h;
  // A window a few percent off (a scrollbar, an open devtools strip) is the same monitor: keep the saved pixels.
  if (Math.abs(kw - 1) < tolerance && Math.abs(kh - 1) < tolerance) return sizes;
  const out = { ...sizes };
  for (const panel of ['leftWidth', 'rightWidth', 'queueWidth']) {
    if (Number.isFinite(out[panel])) out[panel] = clampSize(panel, Math.round(out[panel] * kw));
  }
  if (Number.isFinite(out.bottomHeight)) out.bottomHeight = clampSize('bottomHeight', Math.round(out.bottomHeight * kh));
  return out;
}

/**
 * Can this window hold these panels at all? The centre stage needs `minCenter` px (WorkspaceSplitLayout's own floor); the
 * side panels shrink to fit when they do not (that is the shell's job), so this only answers whether the layout is
 * comfortable or will be squeezed - the menu shows it as a hint, it never refuses.
 */
export function fitsWindow(sizes, width, { minCenter = 360, queue = false } = {}) {
  const used = (sizes.leftWidth || 0) + (sizes.rightWidth || 0) + (queue ? sizes.queueWidth || 0 : 0) + 4;
  return { fits: width - used >= minCenter, centre: Math.max(0, Math.round(width - used)) };
}

/** A unique id for a new layout. */
export const newLayoutId = () => `custom_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

