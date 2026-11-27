// How a parameter curve reaches a render.
//
// The preview evaluates a curve exactly, per frame. A render cannot: a run has one settings payload, and the only way
// to vary settings inside one clip is the one scene overrides already use - one queue job per piece, each with its own
// frame range and payload, joined afterwards (timeline/overrides.js). So a curve is rendered as a STAIRCASE: the range
// is cut wherever the curve moves by one `quantum` (the parameter's render step), and each piece renders at the value
// the curve has in the middle of it.
//
// Every piece is a job, and a job has a fixed start-up cost, so the number of pieces is capped. When a curve would need
// more, the step is widened (x1.5 at a time) until it fits, and the widened step is reported so the UI can say "stepped
// in 0.15s" rather than imply the render follows the curve exactly. Pieces from scene overrides are never merged away:
// an override boundary is always a piece boundary.

import { DIRECT_PARAMS, clampValue, makeEvaluator } from './curveModel';

export const MAX_RENDER_PIECES = 24;
const MAX_WIDEN = 4096;

const sameValues = (a, b) => {
  const ka = Object.keys(a);
  return ka.length === Object.keys(b).length && ka.every((k) => a[k] === b[k]);
};

function splitPiece(pc, active, factor) {
  const cuts = new Set([pc.start]);
  for (const { p, ev } of active) {
    const step = p.quantum * factor;
    let prev = Math.round(ev(pc.start) / step);
    for (let f = pc.start + 1; f <= pc.end; f++) {
      const lvl = Math.round(ev(f) / step);
      if (lvl !== prev) cuts.add(f);
      prev = lvl;
    }
  }
  const points = [...cuts].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < points.length; i++) {
    const start = points[i];
    const end = i + 1 < points.length ? points[i + 1] - 1 : pc.end;
    const mid = (start + end) / 2;
    const curves = {};
    for (const { p, ev } of active) curves[p.id] = clampValue(p, ev(mid));
    const last = out[out.length - 1];
    if (last && sameValues(last.curves, curves)) last.end = end;
    else out.push({ start, end, override: pc.override, curves });
  }
  return out;
}

/**
 * Cut override pieces (overridePieces output) further, wherever a curve moves by a render step.
 *
 * @returns {{pieces: Array<{start, end, override, curves: object|null}>, widen: number, steps: object}}
 *   `widen` is 1 when every curve renders at its natural step, more when the cap forced coarser steps;
 *   `steps` is { paramId: the step actually used }.
 */
export function bakeCurvePieces(pieces, tracks, { maxPieces = MAX_RENDER_PIECES } = {}) {
  const active = DIRECT_PARAMS
    .filter((p) => tracks?.[p.id]?.length)
    .map((p) => ({ p, ev: makeEvaluator(tracks[p.id]) }));
  if (!active.length) return { pieces: pieces.map((pc) => ({ ...pc, curves: null })), widen: 1, steps: {} };

  const limit = Math.max(maxPieces, pieces.length);
  let widen = 1;
  let out = pieces.flatMap((pc) => splitPiece(pc, active, widen));
  while (out.length > limit && widen < MAX_WIDEN) {
    widen *= 1.5;
    out = pieces.flatMap((pc) => splitPiece(pc, active, widen));
  }
  const steps = Object.fromEntries(active.map(({ p }) => [p.id, Math.round(p.quantum * widen * 1000) / 1000]));
  return { pieces: out, widen, steps };
}

/**
 * True when a piece carries a curve value that differs from the payload's own - i.e. rendering this range as pieces
 * would change a frame. With no `base`, any curve value counts. A curve sitting exactly on its slider changes nothing,
 * so it must not turn a plain Start into a queue of identical jobs.
 */
export const piecesHaveCurves = (pieces, base = null) => pieces.some((pc) => pc.curves && DIRECT_PARAMS.some(
  (p) => pc.curves[p.id] !== undefined && (!base || Number(base[p.key]) !== pc.curves[p.id])));

/** One line for the editor: what a render of this range will do with the curves. */
export function describeBake({ pieces, widen, steps }) {
  const withCurves = pieces.filter((pc) => pc.curves);
  if (!withCurves.length) return '';
  const n = pieces.length;
  const stepText = Object.entries(steps)
    .map(([id, s]) => `${DIRECT_PARAMS.find((p) => p.id === id)?.label ?? id} ${s}`).join(' · ');
  return `A render cuts this range into ${n} piece${n === 1 ? '' : 's'} (${stepText})${widen > 1 ? ' - steps widened to stay under the piece cap' : ''}.`;
}
