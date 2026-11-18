// The Face Routing Dock as data: the per-person override pills, a person's frame presence, the route lines, and what to send.
// No React, no DOM: `.render-check/stage5-routing-check.mjs` runs it under node and compares OVERRIDE_PILLS with the backend's own
// table (app/roop/person_overrides.py `PARAMETERS`), so the two cannot drift apart.

// One pill per override the backend (roop/person_overrides.py) applies. `key` is the WIRE name; `global` is the ordinary
// setting the override replaces for ONE person, which is what the pill shows until it is overridden.
export const OVERRIDE_PILLS = Object.freeze([
  { key: 'mask_erode_dilate_radius', label: 'Erode', title: 'Mask erode / dilate (px). Negative keeps hair and objects out of this face; positive recovers clipped skin.',
    min: -32, max: 32, step: 1, global: 'mask_erode_dilate_radius', fallback: 0, unit: ' px', integer: true },
  { key: 'mask_feather', label: 'Feather', title: 'Edge feather of this face, in Face mask edge blend units.',
    min: 0, max: 200, step: 1, global: 'face_mask_blend', fallback: 12 },
  { key: 'restore_identity_weight', label: 'Likeness', title: 'Likeness ↔ fidelity of the restorer for this face (CodeFormer fidelity weight). Only CodeFormer and UltraMax read it.',
    min: 0, max: 1, step: 0.05, global: 'codeformer_fidelity', fallback: 0.55 },
  { key: 'color_match_strength', label: 'LAB', title: 'How strongly the skin colour is matched to the plate at the seam for this face (Lab transfer strength).',
    min: 0, max: 1, step: 0.05, global: 'skin_lab_match', fallback: 1, percent: true },
  { key: 'blend_ratio', label: 'Enhancer', title: 'How much of the restorer is mixed over the swap for this face. 100% = the restorer, 0% = the raw swap.',
    min: 0, max: 1, step: 0.05, global: 'blend_ratio', fallback: 1, percent: true },
]);

const PILL = Object.fromEntries(OVERRIDE_PILLS.map((p) => [p.key, p]));
const finite = (v) => typeof v === 'number' && Number.isFinite(v);

/** The ordinary (all faces) value a pill's override replaces. */
export function globalValue(pill, settings = {}) {
  const v = Number(settings?.[pill.global]);
  return Number.isFinite(v) ? v : pill.fallback;
}

/** What the render uses for this face: the person's override if it has one, else the ordinary setting. */
export function effectiveValue(pill, overrides, personId, settings) {
  const own = overrides?.[personId]?.[pill.key];
  return finite(own) ? own : globalValue(pill, settings);
}

export const hasOverride = (overrides, personId) => Object.keys(overrides?.[personId] || {}).length > 0;
export const overrideCount = (overrides, personId) => Object.keys(overrides?.[personId] || {}).length;

/** Display text for a pill value ("-4 px", "0.55", "80%"). */
export function formatPill(pill, value) {
  if (!finite(value)) return '—';
  if (pill.percent) return `${Math.round(value * 100)}%`;
  const text = pill.integer ? String(Math.round(value)) : String(Number(value.toFixed(2)));
  return `${text}${pill.unit || ''}`;
}

/** Set one override, returning a NEW map. A value outside the pill's range is clamped; an unknown key is ignored. */
export function withOverride(overrides, personId, key, value) {
  const pill = PILL[key];
  const n = Number(value);
  if (!pill || !personId || !Number.isFinite(n)) return overrides || {};
  const clamped = Math.min(pill.max, Math.max(pill.min, pill.integer ? Math.round(n) : n));
  return { ...(overrides || {}), [personId]: { ...(overrides?.[personId] || {}), [key]: clamped } };
}

/** Remove one override (or all of a person's when `key` is omitted), returning a NEW map with no empty people. */
export function withoutOverride(overrides, personId, key) {
  const next = { ...(overrides || {}) };
  if (key === undefined) { delete next[personId]; return next; }
  const own = { ...(next[personId] || {}) };
  delete own[key];
  if (Object.keys(own).length) next[personId] = own; else delete next[personId];
  return next;
}

/**
 * What goes on the wire as `person_overrides`: only people who still exist in the target bank (`personIds`), only known keys,
 * only finite numbers in range. A saved override for a person a clip change removed must not be sent, and must not land on
 * whoever now holds that rank (the backend resolves ids to ranks itself, and reports unknown ids).
 */
export function wireOverrides(overrides, personIds) {
  const ids = new Set((personIds || []).map(String));
  const out = {};
  for (const [person, values] of Object.entries(overrides || {})) {
    if (!ids.has(String(person))) continue;
    const clean = {};
    for (const [key, value] of Object.entries(values || {})) {
      const pill = PILL[key];
      if (pill && finite(value) && value >= pill.min && value <= pill.max) clean[key] = value;
    }
    if (Object.keys(clean).length) out[person] = clean;
  }
  return out;
}

// ── the swap modes that route by person (mirror of person_overrides.PERSON_ROUTED_MODES, by UI label) ─────────────────────
export const ROUTED_MODE_LABELS = Object.freeze(['Selected face', 'Selected people']);
export const overridesApply = (modeLabel) => ROUTED_MODE_LABELS.includes(modeLabel);

// ── presence ────────────────────────────────────────────────────────────────────────────────────────────────────────
/**
 * Percentage of the clip's frames a person is on screen, from their presence intervals ([start, end] inclusive, UI frames).
 * null when there is no scan to read (not 0: "not scanned" and "never appears" are different answers).
 */
export function presencePercent(row, maxFrames) {
  if (!row || !Array.isArray(row.intervals) || !(maxFrames > 0)) return null;
  const merged = [...row.intervals]
    .map(([s, e]) => [Math.max(1, Math.floor(s)), Math.min(maxFrames, Math.floor(e))])
    .filter(([s, e]) => e >= s)
    .sort((a, b) => a[0] - b[0]);
  let covered = 0; let end = 0;
  for (const [s, e] of merged) {
    const from = Math.max(s, end + 1);
    if (e >= from) covered += e - from + 1;
    end = Math.max(end, e);
  }
  return Math.min(100, (covered / maxFrames) * 100);
}

/** "Left Profile + Up Tilt" -> "Left · Up". Short enough for a badge; the full string stays in the title. */
export function poseBadge(pose) {
  const text = String(pose || 'Front');
  const short = text.replace(/\s*Profile/gi, '').replace(/\s*Tilt/gi, '').replace(/\s*\+\s*/g, ' · ').trim();
  return short || 'Front';
}

// ── route lines ─────────────────────────────────────────────────────────────────────────────────────────────────────
/** A cubic S-curve from (x1,y1) to (x2,y2): the line that joins a target face to the source it becomes. */
export function routePath(x1, y1, x2, y2) {
  const dx = Math.max(12, (x2 - x1) * 0.5);
  return `M${x1} ${y1} C${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

/**
 * The lines to draw. `rows` is [{ id, y, sourceIndex }] (y is the row's centre), `sources` is [{ y }] by source index.
 * A person mapped to a source that no longer exists, or to nobody (-1), has no line.
 */
export function routeLines(rows, sources, x1, x2) {
  const lines = [];
  for (const row of rows) {
    const s = sources[row.sourceIndex];
    if (row.sourceIndex < 0 || !s) continue;
    lines.push({ id: row.id, sourceIndex: row.sourceIndex, d: routePath(x1, row.y, x2, s.y) });
  }
  return lines;
}
