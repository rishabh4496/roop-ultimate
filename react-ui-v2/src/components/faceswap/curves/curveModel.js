// Parameter curves: the pure model behind the timeline's Curve Editor.
//
// A track is a sorted list of keyframes `{ frame, value, interp }` for ONE parameter. `frame` is the 1-based UI frame
// (the timeline's own convention: playhead, In/Out, markers) and `interp` shapes the segment that STARTS at the
// keyframe, as in every NLE (the last keyframe's interp is unused). Before the first keyframe and after the last the
// value is held.
//
// The interpolation is the same cubic-Bezier easing the Identity Keyframe Track already uses (identityKeyframes.js,
// mirrored from identity_algebra.bezier_ease), so one segment is always `value0 + (value1 - value0) * ease(u)`:
//
//   hold    a step: value0 until the next keyframe, then value1
//   linear  the diagonal
//   ease    ease-in / ease-out, the CSS `ease-in-out` (0.42, 0, 0.58, 1)
//   smooth  Auto-Smooth: Bezier handles worked out from the neighbouring keyframes (monotone cubic Hermite, the
//           Fritsch-Carlson limiter), so the curve passes through every keyframe, never overshoots between two of
//           them, and is flat at a local peak and at the two ends. A ramp that starts or stops at a scene override's
//           edge therefore arrives with zero slope instead of as a one-frame pop.
//
// A Hermite segment with tangents m0, m1 over `dx` frames and a rise of `d` is exactly the Bezier easing
// (1/3, m0*dx/(3d), 2/3, 1 - m1*dx/(3d)): x(t) = t for those control abscissae, so the easing is a function of the
// elapsed fraction alone. That is why a smooth segment can be evaluated, drawn and handed to the backend's identity
// keyframes through the one `bezierEase`.

import { DEFAULT_BEZIER, bezierEase } from '../identityKeyframes';

export const INTERPS = Object.freeze(['hold', 'linear', 'ease', 'smooth']);
export const INTERP_LABELS = Object.freeze({
  hold: 'Hold (step)', linear: 'Linear', ease: 'Ease-In / Ease-Out', smooth: 'Auto-Smooth Bezier',
});
export const DEFAULT_INTERP = 'smooth';
export const MAX_KEYS = 128;

/**
 * The curves on offer. `key` is the settings / swap-payload key a direct curve writes (the same name in both, which is
 * why one patch serves the preview and the render). `quantum` is the render's step: a queued render is cut into
 * pieces, so a ramp is rendered in steps of at most this size (curvePieces.js). `needs` names a setting that must be
 * on for the curve to do anything, so the editor can say so instead of quietly drawing a curve that changes nothing.
 */
export const CURVE_PARAMS = Object.freeze([
  { id: 'swap_opacity', key: 'blend_ratio', label: 'Swap Opacity', min: 0, max: 1, digits: 2, quantum: 0.05, base: 1, unit: '', color: '#34d399' },
  { id: 'restore_blend', key: 'restore_ultra_detail_weight', label: 'Restore Ultra Blend', min: 0, max: 1, digits: 2, quantum: 0.05, base: 0.75, unit: '', color: '#60a5fa' },
  { id: 'mask_feather', key: 'face_mask_blend', label: 'Mask Feather', min: 0, max: 200, digits: 0, quantum: 2, base: 20, unit: 'px', color: '#f59e0b' },
  { id: 'relight', key: 'light_harmonizer_key_intensity', label: 'Relight Strength', min: 0, max: 2, digits: 2, quantum: 0.1, base: 1, unit: '', color: '#f472b6',
    needs: { key: 'light_harmonizer', label: 'Light Harmonizer' } },
  // Not a settings key: the morph is compiled into the Identity Keyframe Track, which the backend already renders per frame.
  { id: 'identity_morph', key: null, label: 'Identity Morph Weight', min: 0, max: 1, digits: 2, quantum: 0.05, base: 0, unit: '', color: '#a78bfa', identity: true },
]);

export const paramById = (id) => CURVE_PARAMS.find((p) => p.id === id) || null;
/** The curves that write a settings key (everything but the identity morph). */
export const DIRECT_PARAMS = Object.freeze(CURVE_PARAMS.filter((p) => !p.identity));

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const finite = (v, d = 0) => { const n = Number(v); return Number.isFinite(n) ? n : d; };

/** A value clamped to the parameter's range and rounded to its precision. */
export function clampValue(param, value) {
  const f = 10 ** param.digits;
  return Math.round(clamp(finite(value, param.base), param.min, param.max) * f) / f;
}

/** Sanitised copy of one track: in-range frames and values, a known interp, one key per frame, sorted, capped. */
export function normalizeTrack(list, param, maxFrames) {
  const hi = Math.max(1, Math.round(finite(maxFrames, 1)));
  const byFrame = new Map();
  for (const k of Array.isArray(list) ? list.slice(0, MAX_KEYS * 4) : []) {
    if (!k || typeof k !== 'object' || !Number.isFinite(Number(k.frame)) || !Number.isFinite(Number(k.value))) continue;
    byFrame.set(clamp(Math.round(Number(k.frame)), 1, hi), {
      frame: clamp(Math.round(Number(k.frame)), 1, hi),
      value: clampValue(param, k.value),
      interp: INTERPS.includes(k.interp) ? k.interp : DEFAULT_INTERP,
    });
  }
  return [...byFrame.values()].sort((a, b) => a.frame - b.frame).slice(0, MAX_KEYS);
}

/** { paramId: keyframes } with unknown ids dropped and empty tracks omitted. */
export function normalizeTracks(raw, maxFrames) {
  const out = {};
  if (!raw || typeof raw !== 'object') return out;
  for (const p of CURVE_PARAMS) {
    const t = normalizeTrack(raw[p.id], p, maxFrames);
    if (t.length) out[p.id] = t;
  }
  return out;
}

// ── tangents and evaluation ──────────────────────────────────────────────────

/**
 * Per-keyframe tangents (value per frame) for the smooth interp: the Fritsch-Carlson monotone limiter over the
 * neighbouring secants. Zero at the ends, and wherever the two secants disagree in sign (a local peak or trough), so a
 * smooth segment never overshoots its own two keyframes.
 */
export function tangents(list) {
  const n = list.length;
  const m = new Array(n).fill(0);
  if (n < 3) return m;
  const secant = [];
  for (let i = 0; i < n - 1; i++) secant.push((list[i + 1].value - list[i].value) / (list[i + 1].frame - list[i].frame));
  for (let i = 1; i < n - 1; i++) {
    const a = secant[i - 1];
    const b = secant[i];
    if (a * b <= 0) { m[i] = 0; continue; }
    const wa = 2 * (list[i + 1].frame - list[i].frame) + (list[i].frame - list[i - 1].frame);
    const wb = (list[i + 1].frame - list[i].frame) + 2 * (list[i].frame - list[i - 1].frame);
    m[i] = (wa + wb) / (wa / a + wb / b);                     // weighted harmonic mean: never exceeds 3x either secant
  }
  return m;
}

/** The easing [x1, y1, x2, y2] of the segment starting at keyframe `i`, or null for hold / linear / the last key. */
export function segmentBezier(list, i, tans = tangents(list)) {
  const a = list[i];
  const b = list[i + 1];
  if (!a || !b) return null;
  if (a.interp === 'ease') return DEFAULT_BEZIER;
  if (a.interp !== 'smooth') return null;
  const dx = b.frame - a.frame;
  const d = b.value - a.value;
  if (Math.abs(d) < 1e-12) return null;                      // flat: every easing is the same constant
  const y1 = clamp((tans[i] * dx) / (3 * d), 0, 1);
  const y2 = clamp(1 - (tans[i + 1] * dx) / (3 * d), 0, 1);
  return [1 / 3, y1, 2 / 3, y2];
}

// Progress along a segment for the elapsed fraction `u`, given the segment's easing (null = linear). An Auto-Smooth
// segment's control abscissae are (1/3, 2/3), for which x(t) = t exactly, so its easing is the Bezier polynomial itself
// and needs no inverse; the 60-step bisection of bezierEase is only for the general case (Ease-In / Ease-Out). The
// distinction is worth having: a bake walks every frame of the clip, 100,000 of them on a long one.
const easeAlong = (bz, u) => {
  if (!bz) return u;
  if (bz[0] === 1 / 3 && bz[2] === 2 / 3) {
    const w = 1 - u;
    return 3 * w * w * u * bz[1] + 3 * w * u * u * bz[3] + u * u * u;
  }
  return bezierEase(u, ...bz);
};

/**
 * A function frame -> value for one track, with the tangents and every segment's easing worked out ONCE. This is what a
 * loop over many frames (the render bake, the graph's per-pixel line) should use; `evaluate` below is the one-off.
 */
export function makeEvaluator(list) {
  const n = list ? list.length : 0;
  if (!n) return () => null;
  const tans = tangents(list);
  const beziers = list.map((_, i) => segmentBezier(list, i, tans));
  return (frame) => {
    if (frame <= list[0].frame) return list[0].value;
    if (frame >= list[n - 1].frame) return list[n - 1].value;
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {                                     // last key at or before the frame
      const mid = (lo + hi) >> 1;
      if (list[mid].frame <= frame) lo = mid; else hi = mid;
    }
    const a = list[lo];
    if (a.interp === 'hold') return a.value;
    const b = list[lo + 1];
    return a.value + (b.value - a.value) * easeAlong(beziers[lo], (frame - a.frame) / (b.frame - a.frame));
  };
}

/** Value of a track at a frame (fractional frames allowed), or null for an empty track. Held outside the keys. */
export const evaluate = (list, frame) => makeEvaluator(list)(frame);

/** The value of every track that has keyframes, at a frame: { paramId: value } (clamped and rounded). */
export function valuesAt(tracks, frame) {
  const out = {};
  for (const p of CURVE_PARAMS) {
    const list = tracks?.[p.id];
    if (list && list.length) out[p.id] = clampValue(p, evaluate(list, frame));
  }
  return out;
}

export const hasCurves = (tracks) => !!tracks && Object.values(tracks).some((l) => Array.isArray(l) && l.length > 0);

// ── edits (each returns a new list; the same list when it refused) ───────────

/** Add a key at a frame. The value defaults to what the curve already says there, so adding one changes nothing. */
export function addKey(list, param, frame, value, maxFrames, interp = DEFAULT_INTERP) {
  const f = clamp(Math.round(finite(frame, 1)), 1, Math.max(1, maxFrames));
  if (list.some((k) => k.frame === f)) return { list, created: false, frame: f };
  if (list.length >= MAX_KEYS) return { list, created: false, frame: f, full: true };
  const v = value === undefined || value === null ? (evaluate(list, f) ?? param.base) : value;
  return { list: normalizeTrack([...list, { frame: f, value: v, interp }], param, maxFrames), created: true, frame: f };
}

/** The frames a key may be dragged between: strictly inside its neighbours, or null when it is boxed in. */
export function keyFrameBounds(list, frame, maxFrames) {
  const i = list.findIndex((k) => k.frame === frame);
  if (i < 0) return null;
  const lo = i > 0 ? list[i - 1].frame + 1 : 1;
  const hi = i < list.length - 1 ? list[i + 1].frame - 1 : Math.max(1, maxFrames);
  return hi < lo ? null : [lo, hi];
}

/**
 * Move a key to a new frame and value. A key cannot be dragged past its neighbours (that would silently reorder the
 * segments' interps), so the frame is held strictly between them.
 */
export function moveKey(list, param, frame, toFrame, toValue, maxFrames) {
  const i = list.findIndex((k) => k.frame === frame);
  const bounds = keyFrameBounds(list, frame, maxFrames);
  if (i < 0 || !bounds) return list;
  const f = clamp(Math.round(finite(toFrame, frame)), bounds[0], bounds[1]);
  const v = clampValue(param, toValue === undefined ? list[i].value : toValue);
  if (f === frame && v === list[i].value) return list;
  return list.map((k, j) => (j === i ? { ...k, frame: f, value: v } : k));
}

export const removeKey = (list, frame) => list.filter((k) => k.frame !== frame);

export function setInterp(list, frame, interp) {
  if (!INTERPS.includes(interp)) return list;
  return list.map((k) => (k.frame === frame && k.interp !== interp ? { ...k, interp } : k));
}

/** Set the value of an existing key. */
export function setValue(list, param, frame, value) {
  return list.map((k) => (k.frame === frame ? { ...k, value: clampValue(param, value) } : k));
}

// ── applying curves ──────────────────────────────────────────────────────────

/**
 * `settings` with the curves' values at `frame` written over the keys they own. Returns the SAME object when no curve
 * applies, so nothing downstream changes identity (the preview key, the memos) until the playhead is under a curve.
 */
export function patchSettingsWithCurves(settings, tracks, frame) {
  let out = null;
  for (const p of DIRECT_PARAMS) {
    const list = tracks?.[p.id];
    if (!list || !list.length) continue;
    const v = clampValue(p, evaluate(list, frame));
    if (settings && settings[p.key] === v) continue;
    out = out || { ...settings };
    out[p.key] = v;
  }
  return out || settings;
}

/** A swap payload with the values of one piece (valuesOf output) written over it. The same object when empty. */
export function applyCurveValues(payload, values) {
  if (!values) return payload;
  let out = null;
  for (const p of DIRECT_PARAMS) {
    if (values[p.id] === undefined) continue;
    out = out || { ...payload };
    out[p.key] = values[p.id];
  }
  return out || payload;
}

/** A curve that the current settings make a no-op, in words - or null. The editor shows it beside the curve. */
export function curveIneffective(param, settings) {
  if (param.needs && settings && !settings[param.needs.key]) return `${param.needs.label} is off, so this curve changes nothing until it is turned on.`;
  return null;
}

// ── identity morph ───────────────────────────────────────────────────────────

const IDENTITY_CURVE = { hold: 'hold', linear: 'linear', ease: 'cubic_bezier', smooth: 'cubic_bezier' };

/**
 * Compile the Identity Morph Weight curve into Identity Keyframe Track keyframes (identityKeyframes.js shape).
 * 0 = all of the blender's first source, 1 = all of the second. Needs exactly the first two sources of the recipe;
 * the others are weighted 0. The backend slerps between keyframe vectors with the same easing the curve uses.
 */
export function morphToIdentityKeyframes(list, components, maxFrames) {
  const ids = (components || []).map((c) => c.source_id);
  if (ids.length < 2 || !list || !list.length) return [];
  const tans = tangents(list);
  return list.map((k, i) => {
    const bz = segmentBezier(list, i, tans);
    const w = clamp(k.value, 0, 1);
    return {
      frame_index: clamp(k.frame, 1, Math.max(1, maxFrames)) - 1,
      curve: IDENTITY_CURVE[k.interp] || 'linear',
      bezier: bz ? [...bz] : [...DEFAULT_BEZIER],
      weights: { [ids[0]]: Math.round((1 - w) * 10000) / 100, [ids[1]]: Math.round(w * 10000) / 100 },
    };
  });
}
