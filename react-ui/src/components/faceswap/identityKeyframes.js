// Identity keyframes: the pure model behind the timeline's Identity Keyframe Track.
//
// A keyframe is `{ frame_index, curve, bezier, weights }` and nothing else, exactly what
// app/roop/identity_algebra.py BlendRecipe.from_payload keeps. `weights` is `{source_id: 0..100}` over the
// blender's own sources (absent = 0); `curve` shapes the segment that STARTS at this keyframe (hold / linear /
// cubic_bezier, the convention of every NLE), so the last keyframe's curve is unused.
//
// ONE frame convention: `frame_index` is 0-based and absolute in the clip, which is what the backend reads
// (ProcessMgr turns the render's clip-relative index, or the preview's frame, into it). The timeline speaks 1-based
// UI frames (chapter markers, the playhead, In/Out), so `frameIndexOf` / `uiFrameOf` below are the only two places
// that convert, and the rest of this file takes whichever its name says.
//
// The easing is a mirror of identity_algebra.bezier_ease (60-step bisection, x clamped to [0,1] so the inverse
// exists, y clamped to [0,1] so the identity never overshoots a keyframe). It is checked against values generated
// by the Python implementation in .render-check/render-identity-keyframes.mjs.

export const CURVES = Object.freeze(['hold', 'linear', 'cubic_bezier']);
export const CURVE_LABELS = Object.freeze({ hold: 'Hold', linear: 'Linear', cubic_bezier: 'Bezier' });
export const DEFAULT_BEZIER = Object.freeze([0.42, 0, 0.58, 1]);        // ease-in-out
export const MAX_KEYFRAMES = 64;                                         // identity_algebra.MAX_KEYFRAMES

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const finite = (v, d = 0) => { const n = Number(v); return Number.isFinite(n) ? n : d; };

/** The backend's 0-based absolute frame index for a 1-based UI frame. */
export const frameIndexOf = (uiFrame) => Math.round(finite(uiFrame, 1)) - 1;
/** The 1-based UI frame a keyframe sits on. */
export const uiFrameOf = (kf) => kf.frame_index + 1;

export function bezierEase(u, x1, y1, x2, y2) {
  const uu = clamp(finite(u), 0, 1);
  const cx1 = clamp(finite(x1), 0, 1);
  const cx2 = clamp(finite(x2), 0, 1);
  const cy1 = clamp(finite(y1), 0, 1);
  const cy2 = clamp(finite(y2), 0, 1);
  if (uu <= 0 || uu >= 1) return uu;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 60; i++) {
    const t = 0.5 * (lo + hi);
    const x = 3 * (1 - t) ** 2 * t * cx1 + 3 * (1 - t) * t ** 2 * cx2 + t ** 3;
    if (x < uu) lo = t; else hi = t;
  }
  const t = 0.5 * (lo + hi);
  return 3 * (1 - t) ** 2 * t * cy1 + 3 * (1 - t) * t ** 2 * cy2 + t ** 3;
}

/** Progress along a segment (0..1) for `u` = the elapsed fraction of it. */
export function ease(curve, u, bezier = DEFAULT_BEZIER) {
  if (curve === 'hold') return u < 1 ? 0 : 1;
  if (curve === 'cubic_bezier') return bezierEase(u, ...bezier);
  return clamp(finite(u), 0, 1);
}

const cleanBezier = (b) => {
  const raw = Array.isArray(b) ? b.slice(0, 4).map((x) => Number(x)) : [];
  if (raw.length !== 4 || !raw.every(Number.isFinite)) return [...DEFAULT_BEZIER];
  return raw.map((x) => clamp(x, 0, 1));
};

/**
 * Sanitised copy: integer frame >= 0, a known curve, a clamped bezier, weights finite in [0,100] (a keyframe with no
 * positive weight says nothing and is dropped), one keyframe per frame (the later in the list wins), sorted by
 * frame, at most MAX_KEYFRAMES. Weights for sources that are no longer in the blender are KEPT here (removing a
 * source for a moment must not destroy the track); the backend ignores them.
 */
export function normalizeKeyframes(list) {
  const byFrame = new Map();
  for (const k of Array.isArray(list) ? list.slice(0, MAX_KEYFRAMES * 4) : []) {
    if (!k || typeof k !== 'object') continue;
    const f = finite(k.frame_index, -1);
    if (f < 0) continue;
    const weights = {};
    for (const [sid, w] of Object.entries(k.weights && typeof k.weights === 'object' ? k.weights : {})) {
      weights[String(sid)] = clamp(finite(w), 0, 100);
    }
    if (!Object.values(weights).some((w) => w > 1e-12)) continue;
    const frame = Math.round(f);
    byFrame.set(frame, {
      frame_index: frame,
      curve: CURVES.includes(k.curve) ? k.curve : 'linear',
      bezier: cleanBezier(k.bezier),
      weights,
    });
  }
  return [...byFrame.values()].sort((a, b) => a.frame_index - b.frame_index).slice(0, MAX_KEYFRAMES);
}

const normalizedShares = (weights, ids) => {
  const raw = ids.map((id) => Math.max(0, finite(weights?.[id])));
  const total = raw.reduce((a, b) => a + b, 0);
  return total > 1e-12 ? raw.map((w) => w / total) : raw.map(() => 1 / Math.max(1, ids.length));
};

/** The keyframe a segment starts at, the next one, and the eased progress, for a 1-based UI frame. */
export function segmentAt(list, uiFrame) {
  if (!list.length) return null;
  const f = frameIndexOf(uiFrame);
  if (f <= list[0].frame_index) return { i: 0, next: list.length > 1 ? 1 : null, u: 0, s: 0 };
  const last = list.length - 1;
  if (f >= list[last].frame_index) return { i: last, next: null, u: 0, s: 0 };
  let i = 0;
  while (i + 1 < list.length && list[i + 1].frame_index <= f) i++;
  const a = list[i];
  const b = list[i + 1];
  const u = (f - a.frame_index) / (b.frame_index - a.frame_index);
  return { i, next: i + 1, u, s: ease(a.curve, u, a.bezier) };
}

/**
 * The shares (0..100, summing to 100) the track gives each source at a UI frame, from the eased mix of the two
 * bracketing keyframes' normalised weights. This SEEDS a new keyframe so adding one at the playhead does not change
 * what the preview shows; the backend does not use it (it blends each keyframe's vector and slerps between them).
 */
export function seedWeightsAt(list, components, uiFrame) {
  const ids = components.map((c) => c.source_id);
  const seg = segmentAt(list, uiFrame);
  const base = (w) => Object.fromEntries(ids.map((id, i) => [id, w[i] * 100]));
  if (!seg) return base(normalizedShares(Object.fromEntries(components.map((c) => [c.source_id, c.weight])), ids));
  const a = normalizedShares(list[seg.i].weights, ids);
  if (seg.next === null || seg.s <= 0) return base(a);
  const b = normalizedShares(list[seg.next].weights, ids);
  return base(a.map((x, i) => x + (b[i] - x) * seg.s));
}

/** Add a keyframe at a UI frame. A keyframe already there is not duplicated: `created` is false. */
export function addKeyframe(list, uiFrame, weights, maxFrames, curve = 'linear') {
  const frame = frameIndexOf(clamp(Math.round(finite(uiFrame, 1)), 1, Math.max(1, maxFrames)));
  if (list.some((k) => k.frame_index === frame)) return { list, created: false, frame_index: frame };
  if (list.length >= MAX_KEYFRAMES) return { list, created: false, frame_index: frame, full: true };
  const next = normalizeKeyframes([...list, { frame_index: frame, curve, bezier: DEFAULT_BEZIER, weights }]);
  return { list: next, created: next.some((k) => k.frame_index === frame), frame_index: frame };
}

/** Move a keyframe to a UI frame. Refused (same list) onto an occupied frame or outside the clip. */
export function moveKeyframe(list, frameIndex, toUiFrame, maxFrames) {
  const to = frameIndexOf(Math.round(finite(toUiFrame, 1)));
  if (to < 0 || to > Math.max(1, maxFrames) - 1 || to === frameIndex) return list;
  if (list.some((k) => k.frame_index === to)) return list;
  return normalizeKeyframes(list.map((k) => (k.frame_index === frameIndex ? { ...k, frame_index: to } : k)));
}

export const removeKeyframe = (list, frameIndex) => list.filter((k) => k.frame_index !== frameIndex);

/** Patch a keyframe's curve / bezier / weights (weights are merged, not replaced). */
export function updateKeyframe(list, frameIndex, patch) {
  return normalizeKeyframes(list.map((k) => (k.frame_index !== frameIndex ? k : {
    ...k,
    ...(patch.curve !== undefined ? { curve: patch.curve } : null),
    ...(patch.bezier !== undefined ? { bezier: patch.bezier } : null),
    ...(patch.weights !== undefined ? { weights: { ...k.weights, ...patch.weights } } : null),
  })));
}

/** Points [u, s] of a segment's easing for drawing: a step for hold, the diagonal for linear, the curve for bezier. */
export function curvePoints(curve, bezier = DEFAULT_BEZIER, steps = 24) {
  if (curve === 'hold') return [[0, 0], [1, 0], [1, 1]];
  if (curve !== 'cubic_bezier') return [[0, 0], [1, 1]];
  return Array.from({ length: steps + 1 }, (_, i) => [i / steps, bezierEase(i / steps, ...bezier)]);
}

/** SVG path for a segment's easing inside a `w` x `h` box (s=1 at the top). */
export function curvePath(curve, bezier, w, h, pad = 0) {
  const iw = Math.max(0, w - 2 * pad);
  const ih = Math.max(0, h - 2 * pad);
  return curvePoints(curve, bezier).map(([u, s], i) => `${i ? 'L' : 'M'}${(pad + u * iw).toFixed(2)},${(pad + (1 - s) * ih).toFixed(2)}`).join(' ');
}
