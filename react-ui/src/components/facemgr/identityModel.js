// Pure logic for the Identity Workstation (Face Manager, Stage 6). No React, no DOM, no network: everything
// here is a function of its arguments so .render-check/render-stage6-logic.mjs can pin it.

// ── Pose: yaw/pitch for a face ────────────────────────────────────────────────────────────────────────────
// A faceset reports numeric yaw/pitch (/api/facemgr/radar). A captured target person only reports the label
// estimate_face_pose_from_kps() gives ("Front", "Left Profile", "Up Tilt", "Right Profile + Down Tilt").
// poseFromLabel() turns that label into a representative angle and flags it `approx` so the UI can say so;
// it never invents precision the label does not carry.
const LABEL_YAW = { left: -60, right: 60 };
const LABEL_PITCH = { up: 22, down: -22 };

export function poseFromLabel(label) {
  const s = String(label || 'Front').toLowerCase();
  const yaw = s.includes('left') ? LABEL_YAW.left : s.includes('right') ? LABEL_YAW.right : 0;
  const pitch = s.includes('up') ? LABEL_PITCH.up : s.includes('down') ? LABEL_PITCH.down : 0;
  return { yaw, pitch, approx: true };
}

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** {yaw, pitch, approx} for one face from whatever its record carries. */
export function facePose(rec = {}) {
  const yaw = num(rec.yaw);
  const pitch = num(rec.pitch);
  if (yaw !== null && pitch !== null) return { yaw, pitch, approx: false };
  return poseFromLabel(rec.pose);
}

// ── Coverage zones ────────────────────────────────────────────────────────────────────────────────────────
// A 5 (yaw) x 3 (pitch) grid over the head-angle plane. Edges sit on the buckets identity_lab.py already uses
// (|yaw| >= 40 is a profile, |pitch| >= 20 is a high/low angle) so "covered" here agrees with the Face Manager.
export const YAW_BANDS = [
  { id: 'L90', label: 'Left 90° profile', lo: -90, hi: -60 },
  { id: 'L45', label: 'Left 45° profile', lo: -60, hi: -22 },
  { id: 'F', label: 'Front', lo: -22, hi: 22 },
  { id: 'R45', label: 'Right 45° profile', lo: 22, hi: 60 },
  { id: 'R90', label: 'Right 90° profile', lo: 60, hi: 90 },
];
export const PITCH_BANDS = [
  { id: 'U', label: 'Upward', lo: 14, hi: 45 },
  { id: 'M', label: 'Level', lo: -14, hi: 14 },
  { id: 'D', label: 'Downward', lo: -45, hi: -14 },
];

// The zones a swap actually needs. Pitch at the extreme profiles and the up/down corners is rarely
// photographed and rarely matters, so only these are reported as "missing"; the rest still plot.
const REQUIRED = ['F/M', 'L45/M', 'R45/M', 'L90/M', 'R90/M', 'F/U', 'F/D'];

const inBand = (b, v, last) => v >= b.lo && (v < b.hi || (last && v <= b.hi));
export const zoneOf = (yaw, pitch) => {
  const y = YAW_BANDS.find((b, i) => inBand(b, Math.max(-90, Math.min(90, yaw)), i === YAW_BANDS.length - 1));
  const p = PITCH_BANDS.find((b, i) => inBand(b, Math.max(-45, Math.min(45, pitch)), i === 0));
  return `${(y || YAW_BANDS[2]).id}/${(p || PITCH_BANDS[1]).id}`;
};

const zoneName = (id) => {
  const [y, p] = id.split('/');
  const yl = YAW_BANDS.find((b) => b.id === y)?.label || y;
  const pl = PITCH_BANDS.find((b) => b.id === p)?.label || p;
  if (p === 'M') return yl;
  return y === 'F' ? `${pl} pitch` : `${yl}, ${pl.toLowerCase()}`;
};

/**
 * @param {{yaw:number, pitch:number, sharpness?:number|null, key?:any}[]} points
 * @returns {{zones: Record<string,{count:number, keys:any[]}>, missing:{id:string,label:string}[],
 *            covered:number, required:number, approx:boolean}}
 */
export function computeCoverage(points = []) {
  const zones = {};
  for (const y of YAW_BANDS) for (const p of PITCH_BANDS) zones[`${y.id}/${p.id}`] = { count: 0, keys: [] };
  for (const pt of points) {
    const z = zones[zoneOf(pt.yaw, pt.pitch)];
    z.count += 1;
    z.keys.push(pt.key);
  }
  const missing = REQUIRED.filter((id) => zones[id].count === 0).map((id) => ({ id, label: `Missing ${zoneName(id)}` }));
  return {
    zones,
    missing,
    covered: REQUIRED.length - missing.length,
    required: REQUIRED.length,
    approx: points.some((p) => p.approx),
  };
}

// ── Library filtering ─────────────────────────────────────────────────────────────────────────────────────
export const fmtBytes = (n) => {
  const v = Number(n) || 0;
  if (v < 1024) return `${v} B`;
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(v < 10240 ? 1 : 0)} KB`;
  return `${(v / 1024 / 1024).toFixed(1)} MB`;
};

// Tags are derived, not stored: the library has no tag field, and inventing a stored one would be a second source of
// truth. A tag is the fact the user filters on - how big the set is, and how heavy the file is.
export function libraryTags(entry = {}) {
  const tags = [];
  const faces = Number(entry.faces) || 0;
  if (faces <= 1) tags.push('single');
  else if (faces >= 12) tags.push('rich');
  else tags.push('multi-angle');
  if ((Number(entry.size) || 0) >= 5 * 1024 * 1024) tags.push('large');
  return tags;
}

export function filterLibrary(entries = [], query = '', tag = null) {
  const q = String(query || '').trim().toLowerCase();
  return entries.filter((e) => (!q || String(e.name || '').toLowerCase().includes(q))
    && (!tag || libraryTags(e).includes(tag)));
}

// ── Selection ─────────────────────────────────────────────────────────────────────────────────────────────
// `sel` is a plain array of face indices in the order they were picked; `anchor` is the last plain/ctrl click,
// which is where a Shift+Click range starts. Returns the next {sel, anchor}.
export function applyClick(state, index, { shift = false, ctrl = false } = {}, order = []) {
  const { sel = [], anchor = null } = state || {};
  if (shift && anchor !== null) {
    const a = order.indexOf(anchor);
    const b = order.indexOf(index);
    if (a >= 0 && b >= 0) {
      const [lo, hi] = a < b ? [a, b] : [b, a];
      const range = order.slice(lo, hi + 1);
      // Ctrl+Shift extends the existing selection; Shift alone replaces it.
      const next = ctrl ? [...new Set([...sel, ...range])] : range;
      return { sel: next, anchor };
    }
  }
  if (ctrl) {
    const next = sel.includes(index) ? sel.filter((i) => i !== index) : [...sel, index];
    return { sel: next, anchor: index };
  }
  return { sel: [index], anchor: index };
}

/** Rects are {left, top, right, bottom} in the same coordinate space. */
export function rectFromPoints(a, b) {
  return { left: Math.min(a.x, b.x), top: Math.min(a.y, b.y), right: Math.max(a.x, b.x), bottom: Math.max(a.y, b.y) };
}
export const rectsIntersect = (r, s) => r.left < s.right && r.right > s.left && r.top < s.bottom && r.bottom > s.top;

/** Marquee: faces whose tile rect intersects the drag rect. `additive` (Shift/Ctrl held) keeps the prior selection. */
export function marqueeSelect(rect, tiles, prior = [], additive = false) {
  const hit = tiles.filter((t) => rectsIntersect(rect, t.rect)).map((t) => t.index);
  return additive ? [...new Set([...prior, ...hit])] : hit;
}

// ── Cluster edits ─────────────────────────────────────────────────────────────────────────────────────────
// `groups` is target_groups: one rank per captured face (parallel to target_faces). These return a NEW array
// for POST /api/target/group, which re-ranks, so a gap left by a vacated rank is harmless.

/** Move every face of `from` ranks (and the loose `indices`) into rank `into`. */
export function mergeGroups(groups, ranks, into) {
  const set = new Set(ranks);
  return groups.map((g) => (set.has(g) ? into : g));
}

/** Move just `indices` into a brand-new rank (a false positive split off into its own identity). */
export function splitGroup(groups, indices) {
  const next = (groups.length ? Math.max(...groups) : -1) + 1;
  const set = new Set(indices);
  return { groups: groups.map((g, i) => (set.has(i) ? next : g)), rank: next };
}

/** Ranks that the selected face indices touch, ascending. */
export const ranksOf = (groups, indices) => [...new Set(indices.map((i) => groups[i]).filter((g) => g !== undefined))].sort((a, b) => a - b);

// ── Quality ───────────────────────────────────────────────────────────────────────────────────────────────
// Outliers: faces under a quality threshold. `quality` is the best score available for the face (FIQA score for a
// faceset, face width for a target person - normalised by the caller to 0..1).
export const outliersBelow = (qualities, threshold) => qualities
  .map((q, i) => ({ q, i }))
  .filter(({ q }) => typeof q === 'number' && q < threshold)
  .map(({ i }) => i);

/** Quality 0..1 of a captured target face from its bbox width: 24px is unusable, 160px+ is plenty. */
export const qualityFromWidth = (w) => {
  const v = Number(w);
  if (!Number.isFinite(v) || v <= 0) return null;
  return Math.max(0, Math.min(1, (v - 24) / (160 - 24)));
};

export const clampThumb = (px) => Math.max(64, Math.min(240, Math.round(Number(px) || 112)));

// ── Hero thumbnail ────────────────────────────────────────────────────────────────────────────────────────
// The server has no "hero" field, so the hero is a per-person preference kept in the browser, keyed by the stable
// target_person_id. A hero whose face no longer exists (removed, or the person merged away) falls back to the
// person's first face rather than showing nothing.
export function resolveHero(heroes, personId, faceIndices) {
  const h = heroes?.[personId];
  if (typeof h === 'number' && faceIndices.includes(h)) return h;
  return faceIndices.length ? faceIndices[0] : null;
}
