// Per-segment recipe overrides: "for frames a-b only, use THIS mask / blend / swapper / no swap".
//
// How an override reaches a render. A run has exactly one trim range and one settings payload
// (the range lives on the target entry, not in the swap payload), so the way to vary settings
// inside one clip is the mechanism segments already use: one queue job per piece, each carrying
// its own frame_start/frame_end and its own payload, joined afterwards with a stream copy
// (routes_queue._apply_segment, /api/queue/join). `overridePieces` cuts the work range into those
// pieces; `patchPayload` is the only thing that touches a payload. Nothing here changes how a
// frame is processed, which is why nothing in the render loop had to move.

import { clamp } from './viewMath';

// The four things an override may change. `key` is the swap-payload / settings key it writes.
export const OVERRIDE_FIELDS = [
  { id: 'mask_erosion', key: 'mask_erode_dilate_radius', label: 'Mask erosion', kind: 'range', min: -32, max: 32, step: 1, base: 0,
    hint: 'Signed pixels: negative erodes the swap mask (keeps hair / hands out), positive dilates it.' },
  { id: 'mask_feather', key: 'face_mask_blend', label: 'Mask feather', kind: 'range', min: 0, max: 200, step: 1, base: 20,
    hint: 'Width of the soft edge between the swapped face and the plate.' },
  { id: 'restore_blend', key: 'restore_ultra_detail_weight', label: 'Restoration blend', kind: 'range', min: 0, max: 1, step: 0.05, base: 0.75,
    hint: 'How much of the restorer\'s fine detail goes on top of the swap (0 = the swap\'s own detail).' },
  { id: 'swapper', key: 'swap_model', label: 'Swapper model', kind: 'model',
    hint: 'Use a different swap model for this range.' },
];

// Face distance no face can satisfy: with it a targeted run matches nothing, so every frame in the
// piece passes through untouched. Only meaningful for the TARGETED detection modes (see canBypass).
export const BYPASS_FACE_DISTANCE = 0;

const TARGETED_MODES = new Set(['selected', 'selected_face', 'selected_multi', 'selected faces', 'selected face']);
/** Bypass is expressed through the face-match threshold, which "swap every face" modes never read. */
export const canBypass = (detectionMode) => TARGETED_MODES.has(String(detectionMode || '').trim().toLowerCase());

const newId = () => Math.random().toString(36).slice(2, 10);

export function normalizeOverrides(list, maxFrames) {
  const hi = Math.max(1, maxFrames);
  return (list || [])
    .filter((o) => o && Number.isFinite(o.start) && Number.isFinite(o.end))
    .map((o, i) => {
      const a = clamp(Math.round(Math.min(o.start, o.end)), 1, hi);
      const b = clamp(Math.round(Math.max(o.start, o.end)), 1, hi);
      return {
        id: o.id || newId(),
        start: a,
        end: b,
        label: typeof o.label === 'string' ? o.label : `Override ${i + 1}`,
        bypass: !!o.bypass,
        params: sanitizeParams(o.params),
        // Creation order is the layer order: a later override sits on top where two overlap.
        order: Number.isFinite(o.order) ? o.order : i,
      };
    })
    .sort((x, y) => x.start - y.start || x.end - y.end || x.order - y.order);
}

function sanitizeParams(params) {
  const out = {};
  for (const f of OVERRIDE_FIELDS) {
    const v = params?.[f.id];
    if (v === undefined || v === null || v === '') continue;
    if (f.kind === 'range') {
      const n = Number(v);
      if (Number.isFinite(n)) out[f.id] = clamp(n, f.min, f.max);
    } else if (typeof v === 'string') out[f.id] = v;
  }
  return out;
}

// `bypass` lets a caller create the override already bypassing: add() followed by update() in one handler would read the
// list from before the add (useSceneOverrides reads it from the last render) and write that stale list back.
export function addOverride(list, { start, end, label, bypass = false }, maxFrames) {
  const order = list.reduce((m, o) => Math.max(m, o.order ?? 0), -1) + 1;
  const created = {
    id: newId(), start, end, label: label || `Override ${list.length + 1}`, bypass: !!bypass, params: {}, order,
  };
  const next = normalizeOverrides([...list, created], maxFrames);
  return { list: next, created: next.find((o) => o.id === created.id) };
}

export function updateOverride(list, id, patch, maxFrames) {
  return normalizeOverrides(list.map((o) => (o.id === id ? {
    ...o, ...patch, params: patch.params ? { ...patch.params } : o.params,
  } : o)), maxFrames);
}

export const removeOverride = (list, id) => list.filter((o) => o.id !== id);

/** The override in force at a frame: the topmost (latest-created) one covering it. */
export function overrideAt(list, frame) {
  let hit = null;
  for (const o of list || []) {
    if (frame >= o.start && frame <= o.end && (hit === null || (o.order ?? 0) >= (hit.order ?? 0))) hit = o;
  }
  return hit;
}

/** An override that changes nothing is not worth a separate render. */
export const overrideIsEmpty = (o) => !o.bypass && Object.keys(o.params || {}).length === 0;

/**
 * Cut [lo, hi] into consecutive pieces, each tagged with the override that governs it (or null for
 * the base settings). Pieces tile the range exactly, never overlap, and adjacent pieces with the
 * same governor merge, so N overrides cost at most 2N+1 renders.
 */
export function overridePieces(list, lo, hi) {
  if (!(hi >= lo)) return [];
  const live = (list || []).filter((o) => !overrideIsEmpty(o) && o.end >= lo && o.start <= hi);
  const cuts = new Set([lo, hi + 1]);
  for (const o of live) { cuts.add(clamp(o.start, lo, hi + 1)); cuts.add(clamp(o.end + 1, lo, hi + 1)); }
  const points = [...cuts].sort((a, b) => a - b);
  const pieces = [];
  for (let i = 0; i < points.length - 1; i++) {
    const start = points[i];
    const end = points[i + 1] - 1;
    if (end < start) continue;
    const governor = overrideAt(live, start);
    const last = pieces[pieces.length - 1];
    if (last && (last.override?.id ?? null) === (governor?.id ?? null)) last.end = end;
    else pieces.push({ start, end, override: governor });
  }
  return pieces;
}

/**
 * The payload a piece renders with: the base payload with the override's fields written over it.
 * Pure: returns a new object. A bypass piece keeps every other setting (so it fingerprints the same
 * as its neighbours apart from the threshold) and swaps nothing.
 */
export function patchPayload(base, override) {
  if (!override) return base;
  const out = { ...base };
  for (const f of OVERRIDE_FIELDS) {
    if (override.params?.[f.id] !== undefined) out[f.key] = override.params[f.id];
  }
  if (override.bypass) out.face_distance = BYPASS_FACE_DISTANCE;
  return out;
}

/** The same patch applied to the settings object the preview reads, so scrubbing shows the override. */
export function patchSettings(settings, override) {
  if (!override) return settings;
  const out = { ...settings };
  for (const f of OVERRIDE_FIELDS) {
    if (override.params?.[f.id] !== undefined) out[f.key] = override.params[f.id];
  }
  if (override.bypass) out.max_face_distance = BYPASS_FACE_DISTANCE;
  return out;
}

/** One-line description for a pill / tooltip. */
export function describeOverride(o) {
  if (!o) return '';
  if (o.bypass) return 'Bypass swap';
  const bits = OVERRIDE_FIELDS.filter((f) => o.params?.[f.id] !== undefined)
    .map((f) => `${f.label} ${o.params[f.id]}`);
  return bits.length ? bits.join(' · ') : 'No changes yet';
}
