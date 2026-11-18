// Stage 11: the 3x3 Parameter Sweep contact sheet's maths, with no React in it so .render-check/render-ui-overhaul-stage11.mjs can
// pin it down: the shared view (zoom + pan), the synchronized loupe, the face focus presets, the axis values and the warnings.
//
// COORDINATES. Every cell shows the same crop, so one view serves all nine. An image point is `i = (ix, iy)` in 0..1 of the CROP.
// The view maps it to the cell as   s = 0.5 + pan + zoom * (i - 0.5)   (cell fractions, 0..1; pan in cell fractions too), i.e. the
// image is scaled about the cell's centre and then shifted by `pan`. A pan of 0 and zoom 1 is "fit".

export const ZOOM_MIN = 1;
export const ZOOM_MAX = 8;
/** The loupe magnification the brief asks for, relative to what the cell shows. */
export const LOUPE_POWER = 3.5;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const FIT = Object.freeze({ zoom: 1, pan: Object.freeze({ x: 0, y: 0 }) });

/** The pan that keeps the image covering the cell: at zoom z the image overhangs by (z-1)/2 each side. */
export function clampPan(pan, zoom) {
  const lim = Math.max(0, (zoom - 1) / 2);
  return { x: clamp(pan.x, -lim, lim), y: clamp(pan.y, -lim, lim) };
}

/** A view with its zoom and pan forced into range. */
export function normalizeView(view) {
  const zoom = clamp(Number.isFinite(view?.zoom) ? view.zoom : 1, ZOOM_MIN, ZOOM_MAX);
  return { zoom, pan: clampPan(view?.pan || { x: 0, y: 0 }, zoom) };
}

/** Zoom by `factor` keeping the image point under `cursor` ((u,v) in 0..1 of the cell) where it is. */
export function zoomAt(view, factor, cursor = { u: 0.5, v: 0.5 }) {
  const zoom = clamp(view.zoom * factor, ZOOM_MIN, ZOOM_MAX);
  const k = zoom / view.zoom;
  const cx = cursor.u - 0.5;
  const cy = cursor.v - 0.5;
  // s = pan + zoom*p  =>  pan' = c - (zoom'/zoom) * (c - pan)
  const pan = { x: cx - k * (cx - view.pan.x), y: cy - k * (cy - view.pan.y) };
  return { zoom, pan: clampPan(pan, zoom) };
}

/** Drag by (dx, dy) cell fractions. */
export function panBy(view, dx, dy) {
  return { zoom: view.zoom, pan: clampPan({ x: view.pan.x + dx, y: view.pan.y + dy }, view.zoom) };
}

/** The image point (0..1 of the crop) under cell position (u, v), through the view. */
export function imagePointAt(view, u, v) {
  return { x: 0.5 + (u - 0.5 - view.pan.x) / view.zoom, y: 0.5 + (v - 0.5 - view.pan.y) / view.zoom };
}

/** The view that puts image point `i` at the middle of the cell at `zoom`. */
export function viewCentredOn(i, zoom) {
  const z = clamp(zoom, ZOOM_MIN, ZOOM_MAX);
  return { zoom: z, pan: clampPan({ x: -z * (i.x - 0.5), y: -z * (i.y - 0.5) }, z) };
}

/**
 * Where the loupe lens sits and what it shows, for a cell whose picture box is `frameW x frameH` px.
 *   lens      diameter in px
 *   u, v      the pointer in the cell (0..1)
 * The lens shows the image point under the pointer at `power` x the cell's own scale (the cell is already `view.zoom` x
 * fit), so 3.5x is 3.5x of what the cell shows, not of the unzoomed crop. Returns the CSS the lens needs:
 *   { left, top, bgW, bgH, bgX, bgY }  (px; background-size = bgW x bgH, background-position = bgX bgY)
 * The same u,v and view in all nine cells give the same image point in all nine - that IS the synchronization.
 */
export function loupeLayout({ frameW, frameH, view, u, v, lens, power = LOUPE_POWER }) {
  const pt = imagePointAt(view, u, v);
  const bgW = frameW * view.zoom * power;
  const bgH = frameH * view.zoom * power;
  return {
    left: u * frameW - lens / 2,
    top: v * frameH - lens / 2,
    bgW, bgH,
    bgX: lens / 2 - pt.x * bgW,
    bgY: lens / 2 - pt.y * bgH,
    point: pt,
  };
}

/** Where on the crop each focus preset looks, from the backend's face coordinates (crop px) or a sensible default. */
export function focusPoint(which, faceCoords) {
  const rect = faceCoords?.crop_rect;
  const w = Array.isArray(rect) && rect[2] > 0 ? rect[2] : null;
  const h = Array.isArray(rect) && rect[3] > 0 ? rect[3] : null;
  const norm = (pt, fallback) => (w && h && Array.isArray(pt) && pt.length >= 2
    ? { x: clamp(pt[0] / w, 0, 1), y: clamp(pt[1] / h, 0, 1) } : fallback);
  if (which === 'eyes') {
    const l = faceCoords?.eyes?.left;
    const r = faceCoords?.eyes?.right;
    if (w && h && Array.isArray(l) && Array.isArray(r)) return norm([(l[0] + r[0]) / 2, (l[1] + r[1]) / 2], { x: 0.5, y: 0.4 });
    return { x: 0.5, y: 0.4 };
  }
  if (which === 'mouth') return norm(faceCoords?.mouth?.center, { x: 0.5, y: 0.66 });
  if (which === 'jaw') return norm(faceCoords?.jawline?.center, { x: 0.5, y: 0.84 });
  return { x: 0.5, y: 0.5 };
}

export const FOCUS_ZOOM = Object.freeze({ eyes: 2.8, mouth: 3.0, jaw: 2.6 });

/** The shared view for a focus preset ('fit' | 'eyes' | 'mouth' | 'jaw'). */
export function focusView(which, faceCoords) {
  if (which === 'fit') return { zoom: 1, pan: { x: 0, y: 0 } };
  return viewCentredOn(focusPoint(which, faceCoords), FOCUS_ZOOM[which] || 2.5);
}

// ── axis values ────────────────────────────────────────────────────────────

/** The min / centre / max a numeric axis starts from, or its three categorical choices. */
export const defaultValues = (def) => [...def.defaultRange];

/**
 * A typed value forced into the parameter's range and step. Categorical values must be one of the options (else the first is kept);
 * a non-number falls back to `fallback`. Rounds to the step's precision so 0.1 + 0.2 never shows as 0.30000000000000004.
 */
export function normalizeValue(def, value, fallback) {
  if (def.type === 'categorical') return def.options.includes(value) ? value : (fallback ?? def.options[0]);
  const n = typeof value === 'number' ? value : parseFloat(value);
  if (!Number.isFinite(n)) return fallback ?? def.defaultRange[0];
  const lo = Number.isFinite(def.min) ? def.min : -Infinity;
  const hi = Number.isFinite(def.max) ? def.max : Infinity;
  const stepped = def.step > 0 ? Math.round(n / def.step) * def.step : n;
  const decimals = def.step > 0 ? Math.max(0, (String(def.step).split('.')[1] || '').length) : 6;
  return Number(clamp(stepped, lo, hi).toFixed(decimals));
}

/** True when the three values are not strictly the same (a sweep over one repeated value is nine copies of one picture). */
export const valuesVary = (values) => new Set(values.map(String)).size > 1;

// ── the result ─────────────────────────────────────────────────────────────

export const cellIndex = (col, row) => row * 3 + col;

/** The tile at grid position, or null while there is no result. */
export const tileAt = (result, col, row) => result?.tiles?.[cellIndex(col, row)] || null;

/** The picture's width / height ratio, from the first tile's pixel size when known, else the backend's crop rect, else 1. */
export function aspectOf(result, naturalSize) {
  if (naturalSize?.w > 0 && naturalSize?.h > 0) return naturalSize.w / naturalSize.h;
  const rect = result?.face_coordinates?.crop_rect;
  if (Array.isArray(rect) && rect[2] > 0 && rect[3] > 0) return rect[2] / rect[3];
  return 1;
}

/**
 * What the sheet must tell the user about a result, in order of importance. `[{ kind, text }]`.
 *   no_face    the backend found no face, so every tile is the same crop of the frame
 *   unbound    an axis drives nothing in a single-frame preview (its reason is the backend's)
 *   identical  the nine pictures are byte-identical whatever the axes say
 *   flat       the axis values were all equal, so there was nothing to compare
 */
export function resultWarnings(result, { xLabel = 'X', yLabel = 'Y', xValues = [], yValues = [] } = {}) {
  if (!result) return [];
  const out = [];
  if (result.no_face) out.push({ kind: 'no_face', text: 'No face was detected on this frame, so all nine tiles are the same crop of it.' });
  for (const [axis, label] of [['x', xLabel], ['y', yLabel]]) {
    const a = result.axes?.[axis];
    if (a && a.bound === false) out.push({ kind: 'unbound', text: `${label}: ${a.reason || 'nothing reads this parameter in a single-frame preview.'}` });
  }
  if (!result.no_face && result.identical === true && !out.some((w) => w.kind === 'unbound')) {
    out.push({ kind: 'identical', text: 'All nine tiles are pixel-identical: neither parameter changes this frame. Try a frame where the face is larger, or a wider range.' });
  }
  if (valuesVary(xValues) === false && valuesVary(yValues) === false) {
    out.push({ kind: 'flat', text: 'Both axes use one repeated value, so there is nothing to compare.' });
  }
  return out;
}

/** "Restore Ultra Weight 0.65 · Mask Erosion 0" for a tile. */
export function recipeText(xLabel, yLabel, tile) {
  return `${xLabel} ${tile.x_value} · ${yLabel} ${tile.y_value}`;
}

export const SLOT_ORDER = Object.freeze(['A', 'B', 'C', 'D']);
