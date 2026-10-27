// Pointer hit-testing for the lanes, as pure functions of a frame number and a pixel row. The
// canvases draw; these decide what a click or right-click landed on.

import { SCENE_CUT_H, SCENE_ROW_H, MAX_PRESENCE_ROWS } from './lanes';
import { presenceHit, sceneAt } from './laneModel';
import { overrideAt } from './overrides';

/** The cut within `tolFrames` of `frame` (nearest wins), or null. */
export function cutNear(cuts, frame, tolFrames) {
  let best = null;
  let bestD = tolFrames + 1e-9;
  for (const c of cuts || []) {
    const d = Math.abs(c - frame);
    if (d <= bestD) { best = c; bestD = d; }
  }
  return best;
}

/** Which presence row a y offset inside the scene lane falls on (0-based), or -1. */
export function presenceRowAtY(y, rowCount) {
  if (y < SCENE_CUT_H) return -1;
  const r = Math.floor((y - SCENE_CUT_H) / SCENE_ROW_H);
  return r >= 0 && r < Math.min(rowCount, MAX_PRESENCE_ROWS) ? r : -1;
}

/**
 * What is under (frame, y) in the scene lane:
 *   {kind:'cut', frame}                      a scene-cut flag
 *   {kind:'presence', row, interval}         a person's pill
 *   {kind:'scene', scene}                    a shot with nothing more specific on it
 */
export function sceneLaneHit({ frame, y, cuts, rows, scenes, tolFrames }) {
  const cut = y < SCENE_CUT_H + 2 ? cutNear(cuts, frame, tolFrames) : null;
  if (cut !== null) return { kind: 'cut', frame: cut };
  const r = presenceRowAtY(y, rows.length);
  if (r >= 0) {
    const interval = presenceHit(rows[r], frame);
    if (interval) return { kind: 'presence', row: rows[r], interval };
  }
  const scene = sceneAt(scenes, frame);
  return scene ? { kind: 'scene', scene } : null;
}

/**
 * An override under a frame, and which part of it: an edge within `edgeFrames` is a resize grip,
 * anywhere else is the body (move / select).
 */
export function overrideLaneHit(overrides, frame, edgeFrames) {
  const o = overrideAt(overrides, frame);
  if (!o) return null;
  if (frame - o.start <= edgeFrames) return { override: o, part: 'left' };
  if (o.end - frame <= edgeFrames) return { override: o, part: 'right' };
  return { override: o, part: 'body' };
}

/**
 * The ranges a right-click can turn into an override: the shot under the cursor, and the In/Out
 * work area when the cursor is inside it (and it is not simply the whole clip).
 */
export function overrideCandidates({ frame, scenes, startFrame, endFrame, maxFrames }) {
  const out = [];
  const scene = sceneAt(scenes, frame);
  if (scene) out.push({ kind: 'scene', start: scene.start, end: scene.end, label: `Scene ${scene.index + 1}` });
  const wholeClip = startFrame <= 1 && endFrame >= maxFrames;
  if (!wholeClip && frame >= startFrame && frame <= endFrame) {
    const dup = out.some((c) => c.start === startFrame && c.end === endFrame);
    if (!dup) out.push({ kind: 'inout', start: startFrame, end: endFrame, label: 'In/Out range' });
  }
  return out;
}
