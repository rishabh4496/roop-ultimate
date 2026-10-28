// Pure data model behind the timeline lanes: interval algebra, the render/cache strip, scenes
// between cuts, and the per-person presence pills. No React, no DOM.
//
// Intervals are inclusive 1-based frame ranges [start, end], the numbering the whole UI uses.

// ── interval algebra ────────────────────────────────────────────────────────

/** Sorted, merged, clipped to [lo, hi]. Intervals that touch (end + 1 === start) merge too. */
export function mergeIntervals(list, lo = 1, hi = Infinity) {
  const rows = [];
  for (const it of list || []) {
    if (!it) continue;
    const s = Math.max(lo, Math.round(Math.min(it[0], it[1])));
    const e = Math.min(hi, Math.round(Math.max(it[0], it[1])));
    if (Number.isFinite(s) && Number.isFinite(e) && e >= s) rows.push([s, e]);
  }
  rows.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = [];
  for (const r of rows) {
    const last = out[out.length - 1];
    if (last && r[0] <= last[1] + 1) last[1] = Math.max(last[1], r[1]);
    else out.push([r[0], r[1]]);
  }
  return out;
}

/** The parts of [lo, hi] that `list` does not cover. */
export function complement(list, lo, hi) {
  const merged = mergeIntervals(list, lo, hi);
  const out = [];
  let cursor = lo;
  for (const [s, e] of merged) {
    if (s > cursor) out.push([cursor, s - 1]);
    cursor = Math.max(cursor, e + 1);
  }
  if (cursor <= hi) out.push([cursor, hi]);
  return out;
}

/** Intervals of `list` that intersect [lo, hi], clipped to it. Binary-searches a sorted list. */
export function visibleIntervals(list, lo, hi) {
  if (!list || !list.length) return [];
  let a = 0;
  let b = list.length;
  while (a < b) {                       // first interval whose end >= lo
    const mid = (a + b) >> 1;
    if (list[mid][1] < lo) a = mid + 1; else b = mid;
  }
  const out = [];
  for (let i = a; i < list.length && list[i][0] <= hi; i++) out.push([Math.max(lo, list[i][0]), Math.min(hi, list[i][1])]);
  return out;
}

/** Number of frames covered, counting overlaps once. */
export const coveredFrames = (list) => mergeIntervals(list).reduce((n, [s, e]) => n + e - s + 1, 0);

// ── audio track ──────────────────────────────────────────

/**
 * Where in the AUDIO FILE the sound heard at video time `t` lives, given the slip. A positive slip plays the audio later,
 * so the pixel at video time t shows what the file holds at t - slip (never before the file's start). This is the same
 * mapping the render applies to the mouth and to the remuxed track (app/roop/mouth_custody.py `audio_time_s`).
 */
export const audioSourceTime = (videoSeconds, slipSeconds = 0) => videoSeconds - slipSeconds;

/** True when video time `t` falls inside audio of length `audioSeconds` after the slip - outside it the track is silent. */
export const audioCovers = (videoSeconds, slipSeconds, audioSeconds) => {
  const src = audioSourceTime(videoSeconds, slipSeconds);
  return src >= 0 && src <= audioSeconds;
};

// ── render / smart-skip strip ───────────────────────────────────────────────

// Highest priority first. A frame is painted by the first layer that covers it.
export const RENDER_STATES = ['reuse', 'bypass', 'rendered', 'dirty'];

/**
 * Colour-code every frame of [lo, hi]:
 *   reuse     motion-compensated smart-skip ROI reuse (cyan)
 *   bypass    Stage 1/2 face-free, passed through untouched (purple)
 *   rendered  a render of these exact settings covers it (green)
 *   dirty     never rendered, or rendered with different settings (amber)
 *
 * Returns ordered, non-overlapping {start, end, state} spans that tile [lo, hi] exactly. Built by
 * a boundary sweep, so the cost follows the number of intervals, not the number of frames.
 */
export function buildRenderStrip({ lo = 1, hi, reuse = [], bypass = [], rendered = [] }) {
  if (!(hi >= lo)) return [];
  const layers = {
    reuse: mergeIntervals(reuse, lo, hi),
    bypass: mergeIntervals(bypass, lo, hi),
    rendered: mergeIntervals(rendered, lo, hi),
  };
  const cuts = new Set([lo, hi + 1]);
  for (const key of Object.keys(layers)) for (const [s, e] of layers[key]) { cuts.add(s); cuts.add(e + 1); }
  const points = [...cuts].filter((p) => p >= lo && p <= hi + 1).sort((a, b) => a - b);

  const cursors = { reuse: 0, bypass: 0, rendered: 0 };
  const stateAt = (f) => {
    for (const key of ['reuse', 'bypass', 'rendered']) {
      const list = layers[key];
      let i = cursors[key];
      while (i < list.length && list[i][1] < f) i++;
      cursors[key] = i;
      if (i < list.length && list[i][0] <= f) return key;
    }
    return 'dirty';
  };

  const spans = [];
  for (let i = 0; i < points.length - 1; i++) {
    const start = points[i];
    const end = points[i + 1] - 1;
    if (end < start) continue;
    const state = stateAt(start);
    const last = spans[spans.length - 1];
    if (last && last.state === state && last.end + 1 === start) last.end = end;
    else spans.push({ start, end, state });
  }
  return spans;
}

/** FNV-1a over a stable serialisation, as 8 hex chars. Collisions only ever read as "clean". */
export function fingerprint(value, ignore = IGNORED_FINGERPRINT_KEYS) {
  const text = stableStringify(value, ignore);
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

// Per-request noise that changes on every payload build without changing the render.
export const IGNORED_FINGERPRINT_KEYS = new Set([
  'request_id', 'selection_version', 'processing_selection', 'selection_state', 'imagemask',
  'frame_start', 'frame_end', 'segment', 'label',
]);

function stableStringify(v, ignore) {
  if (v === null || typeof v !== 'object') return JSON.stringify(v) ?? 'null';
  if (Array.isArray(v)) return `[${v.map((x) => stableStringify(x, ignore)).join(',')}]`;
  const keys = Object.keys(v).filter((k) => !ignore.has(k) && v[k] !== undefined).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(v[k], ignore)}`).join(',')}}`;
}

/**
 * Which frames hold a finished render of the settings in force NOW.
 *
 * `log` is [{start, end, fp}], one per finished render, fp = the fingerprint of the settings it used.
 * `pieces` is [{start, end, fp}] tiling the clip: the fingerprint the settings WOULD have at each
 * stretch (the clip's own, or the clip's with a scene override applied). A frame counts only when
 * a render of exactly that piece's fingerprint covers it - so changing a slider, or an override,
 * turns the affected frames dirty again instead of leaving them green.
 */
export function renderedForPieces(log, pieces) {
  const out = [];
  for (const piece of pieces || []) {
    for (const r of log || []) {
      if (!r || r.fp !== piece.fp) continue;
      const a = Math.max(piece.start, Math.min(r.start, r.end));
      const b = Math.min(piece.end, Math.max(r.start, r.end));
      if (b >= a) out.push([a, b]);
    }
  }
  return mergeIntervals(out);
}

/** Add a finished render to the log, dropping entries it fully supersedes. */
export function recordRender(log, entry, limit = 64) {
  const clean = { start: Math.min(entry.start, entry.end), end: Math.max(entry.start, entry.end), fp: entry.fp };
  const kept = (log || []).filter((r) => !(r.fp === clean.fp && r.start >= clean.start && r.end <= clean.end));
  return [...kept, clean].slice(-limit);
}

// ── scenes ──────────────────────────────────────────────────────────────────

/** Scene cuts as the FIRST frame of each new shot; returns the shots they split [1, maxFrames] into. */
export function scenesFromCuts(cuts, maxFrames) {
  const starts = [...new Set((cuts || []).map((c) => Math.round(c)).filter((c) => c > 1 && c <= maxFrames))]
    .sort((a, b) => a - b);
  const out = [];
  let from = 1;
  for (const c of starts) {
    out.push({ index: out.length, start: from, end: c - 1 });
    from = c;
  }
  out.push({ index: out.length, start: from, end: Math.max(from, maxFrames) });
  return out;
}

export function sceneAt(scenes, frame) {
  let lo = 0;
  let hi = scenes.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frame < scenes[mid].start) hi = mid - 1;
    else if (frame > scenes[mid].end) lo = mid + 1;
    else return scenes[mid];
  }
  return null;
}

/** The previous/next cut strictly before/after `frame`, or null. `cuts` need not be sorted. */
export function adjacentCut(cuts, frame, dir) {
  let best = null;
  for (const c of cuts || []) {
    if (dir > 0 ? c > frame && (best === null || c < best) : c < frame && (best === null || c > best)) best = c;
  }
  return best;
}

// ── presence ────────────────────────────────────────────────────────────────

// Categorical palette chosen to stay apart in hue AND lightness, so two neighbouring pills read as
// different people for colour-blind viewers too. Index by the person's rank, wrap past eight.
export const PERSON_COLORS = ['#38bdf8', '#f472b6', '#a3e635', '#fb923c', '#c084fc', '#2dd4bf', '#facc15', '#f87171'];
export const personColor = (rank) => PERSON_COLORS[((rank % PERSON_COLORS.length) + PERSON_COLORS.length) % PERSON_COLORS.length];

/**
 * Normalise the server's presence payload into rows the lane can draw and hit-test.
 * `people` is [{person_id, name, intervals: [[s,e],...]}]; rows keep the server's order, which is
 * "most frames first", so Person 1 is the main character.
 */
export function presenceRows(people, maxFrames) {
  return (people || []).map((p, i) => ({
    id: String(p.person_id ?? p.id ?? i),
    name: p.name || `Person ${i + 1}`,
    rank: i,
    color: personColor(i),
    intervals: mergeIntervals(p.intervals || [], 1, maxFrames),
  }));
}

/**
 * The identity-confidence rows the strip draws, one per person, keyed like the presence rows so the two
 * lanes agree on whose colour is whose. `points` is [[frame, confidence, amber], ...] in UI frames and
 * `amber` the frames worth a look; both are passed through as the backend decided them.
 */
export function identityRows(lanes) {
  return (lanes || []).map((l, i) => ({
    id: String(l.person_id ?? l.id ?? i),
    name: l.name || `Person ${i + 1}`,
    rank: i,
    color: personColor(i),
    points: (l.points || []).filter((p) => Array.isArray(p) && Number.isFinite(Number(p[0]))),
    amber: (l.amber || []).map(Number).filter(Number.isFinite),
    threshold: Number(l.threshold) || 0,
  }));
}

/** The row/interval under a frame in a given row, or null. */
export function presenceHit(row, frame) {
  if (!row) return null;
  for (const it of row.intervals) if (frame >= it[0] && frame <= it[1]) return it;
  return null;
}
