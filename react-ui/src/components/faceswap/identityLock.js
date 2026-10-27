// Identity pins and the confidence strip: the pure model.
//
//   pin      "this box, on this frame, is that person". The render binds the pre-pass track passing
//            through the box to the pinned source before any automatic gate runs (see
//            app/roop/identity_lock.py, resolve_pin_tracks, and procmgr_tracking._assign_track_sources).
//   strip    how far inside its match threshold each SAMPLED detection of a person sat, with the
//            samples worth a look marked amber. The backend (identity_lock.confidence_lane) decides what is
//            amber; this file only lays it out and answers "where is the next one".
//
// A pin's "Track #N" is the number the preview shows for the person (Person N): the pre-pass track ids
// do not exist until a render's pre-pass has run, and the renderer resolves the pin to its track from the
// box. Pure functions only (`.render-check/render-stage11-logic.mjs` runs them under node).

// Strict: Number(null) is 0 and Number('') is 0, so a missing frame or coordinate would pass a bare
// Number.isFinite(Number(v)) as a real zero (the trap faceRoles.js documents).
const finite = (v) => v !== null && v !== undefined && v !== '' && typeof v !== 'boolean' && Number.isFinite(Number(v));
const sourceOf = (v) => (Number.isInteger(v) && v >= 0 ? v : null);

/** The label the pin button and the pin list show for a person. */
export const trackLabel = (personId) => (finite(personId) ? `Track #${Number(personId) + 1}` : 'Track');

/**
 * One pin from a clicked face. `bbox` is the face box in frame pixels, `frame` the UI frame number,
 * `sourceIndex` what that person swaps from (the mapping the render uses). A pin that locks no source
 * is refused (null): the renderer would have nothing to bind.
 */
export function makePin({ frame, bbox, personId = null, sourceIndex = null, faceIndex = null, id = null }) {
  const source = sourceOf(sourceIndex);
  if (source === null) return null;
  if (!Array.isArray(bbox) || bbox.length < 4 || !bbox.slice(0, 4).every(finite)) return null;
  const [x0, y0, x1, y1] = bbox.slice(0, 4).map(Number);
  if (!(x1 > x0 && y1 > y0) || !finite(frame)) return null;
  return {
    id: id || `pin_${Math.round(Number(frame))}_${source}`,
    frame: Math.round(Number(frame)),
    bbox: [x0, y0, x1, y1],
    source_index: source,
    person_id: finite(personId) ? `tp_${Number(personId)}` : null,
    track_id: finite(personId) ? Number(personId) + 1 : null,
    face_index: Number.isInteger(faceIndex) ? faceIndex : null,
  };
}

/** Add (or replace the pin of the same id). Two pins for one person on one frame are one pin. */
export function addPin(pins, pin) {
  if (!pin) return pins || [];
  return [...(pins || []).filter((p) => p.id !== pin.id), pin].sort((a, b) => a.frame - b.frame);
}

export const removePin = (pins, id) => (pins || []).filter((p) => p.id !== id);

/** The pin, if any, that covers face `bbox` on `frame` (same source, within `tol` frames). */
export function pinFor(pins, frame, source, tol = 0) {
  return (pins || []).find((p) => p.source_index === source && Math.abs(p.frame - frame) <= tol) || null;
}

/** What the request carries (UI-only fields stripped). */
export function serializePins(pins) {
  return (pins || []).map((p) => ({
    id: p.id, frame: p.frame, bbox: p.bbox, source_index: p.source_index,
    person_id: p.person_id, track_id: p.track_id,
  }));
}

// ── confidence strip ────────────────────────────────────────────────────────

/** Tone for a sample: 'amber' when flagged, else 'ok' (strong) or 'fair'. */
export function toneOf(confidence, amber) {
  if (amber) return 'amber';
  return confidence >= 0.6 ? 'ok' : 'fair';
}

/**
 * Lay a lane out for drawing. `points` is [[frame, confidence, amber], ...] in UI frames; each sample
 * covers half the gap to its neighbours (we looked at one frame in `stride`), clamped to [1, total].
 * Returns [{ start, end, confidence, amber, tone }].
 */
export function laneCells(points, total) {
  const pts = (points || []).filter((p) => p && finite(p[0]) && finite(p[1])).sort((a, b) => a[0] - b[0]);
  const lastSample = pts.length ? pts[pts.length - 1][0] : 1;
  const last = Math.max(1, Math.round(Number(total)) || lastSample);
  return pts.map((p, i) => {
    const prev = i > 0 ? pts[i - 1][0] : null;
    const next = i + 1 < pts.length ? pts[i + 1][0] : null;
    const lo = prev === null ? p[0] : Math.floor((prev + p[0]) / 2) + 1;
    const hi = next === null ? p[0] : Math.floor((p[0] + next) / 2);
    return {
      start: Math.max(1, lo),
      end: Math.min(last, Math.max(lo, hi)),
      frame: p[0],
      confidence: Number(p[1]),
      amber: !!p[2],
      tone: toneOf(Number(p[1]), !!p[2]),
    };
  });
}

/** The nearest amber frame after (dir 1) or before (dir -1) `frame`, wrapping at the ends; null if none. */
export function nextAmber(amber, frame, dir = 1) {
  const list = [...new Set((amber || []).map(Number).filter(finite))].sort((a, b) => a - b);
  if (!list.length) return null;
  if (dir >= 0) return list.find((f) => f > frame) ?? list[0];
  for (let i = list.length - 1; i >= 0; i -= 1) if (list[i] < frame) return list[i];
  return list[list.length - 1];
}

/** All amber frames across lanes, merged and sorted - the strip's "jump to next" list. */
export function allAmber(lanes) {
  return [...new Set((lanes || []).flatMap((l) => l.amber || []))].sort((a, b) => a - b);
}
