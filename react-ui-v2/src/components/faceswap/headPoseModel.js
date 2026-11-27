// The head-pose HUD's model: what `GET /api/trajectory/pose` returned -> the rows the HUD shows.
//
// The numbers come from the Stage 2 pre-pass's SMOOTHED keypoints (roop/trajectory_cache.py), i.e. the pose of
// the alignment the swap will actually use, not a second estimate made for display. Angles are degrees in
// head_pose.py's convention: +yaw = nose toward the viewer's right, +pitch = tilted DOWN, +roll = clockwise.
//
// Pure: no React, no request, no timer, so it is checked on its own (.render-check/render-head-pose-hud.mjs).

/** Half-range of each dial's bar, in degrees: a head at this angle pins the marker to the end of its bar. */
export const BAR_RANGE = { yaw: 90, pitch: 60, roll: 45 };

export const CONVENTION = '+yaw: nose toward your right · +pitch: looking down · +roll: clockwise';

const FLAG_NOTES = [
  [4, 'predicted'],      // Kalman-coasted past the gap limit
  [2, 'flow-tracked'],   // placed by optical flow between two detections
  [1, 'gap-filled'],     // interpolated between detections
];

export function fmtAngle(value) {
  const v = Number(value);
  if (value == null || !Number.isFinite(v)) return '—';
  const r = Math.round(v * 10) / 10;
  return `${r > 0 ? '+' : ''}${r.toFixed(1)}°`;
}

/** Marker position along a bar, as a percentage of its width (50 = centred); a missing angle has none. */
export function markerPct(value, range) {
  const v = Number(value);
  if (value == null || !Number.isFinite(v) || !(range > 0)) return null;
  return 50 + Math.max(-1, Math.min(1, v / range)) * 50;
}

/** The word for a face that was not detected on this frame ('' for a real detection). */
export function noteOf(flags) {
  const f = Number(flags) || 0;
  for (const [bit, word] of FLAG_NOTES) if (f & bit) return word;
  return '';
}

/**
 * @param payload  the JSON from /api/trajectory/pose, or undefined before the first answer
 * @param opts.active  whether a job has frames at all (an idle app has no trajectory to ask for)
 * @returns {{status: 'idle'|'waiting'|'none'|'empty'|'live', caption: string, frame: number|null, faces: object[]}}
 */
export function hudModel(payload, { active = true } = {}) {
  if (!active) {
    return { status: 'idle', caption: 'Appears while a job runs, once the pre-pass has finished.', frame: null, faces: [] };
  }
  if (payload === undefined) {
    return { status: 'waiting', caption: 'Reading the trajectory…', frame: null, faces: [] };
  }
  if (!payload || payload.available !== true) {
    return {
      status: 'none',
      caption: (payload && payload.reason) || 'No trajectory yet.',
      frame: null,
      faces: [],
    };
  }
  const faces = (Array.isArray(payload.faces) ? payload.faces : []).map((f) => ({
    track: Number(f.track),
    label: `Face ${Number(f.track)}`,
    note: noteOf(f.flags),
    dials: ['yaw', 'pitch', 'roll'].map((name) => ({
      name,
      label: name[0].toUpperCase() + name.slice(1),
      text: fmtAngle(f[name]),
      pct: markerPct(f[name], BAR_RANGE[name]),
    })),
  }));
  const used = Number(payload.frame);
  const asked = Number(payload.requested);
  if (!faces.length) {
    return { status: 'empty', caption: `No face near frame ${Number.isFinite(asked) ? asked : '?'}.`, frame: null, faces: [] };
  }
  // A frame that is not the one asked for says so (the HUD is polled at an approximate frame number).
  const offset = Number.isFinite(used) && Number.isFinite(asked) ? used - (asked + Number(payload.frame_start || 0)) : 0;
  return {
    status: 'live',
    caption: `Frame ${Number.isFinite(used) ? used : '?'}${offset ? ` (${offset > 0 ? '+' : ''}${offset} from now)` : ''}`,
    frame: Number.isFinite(used) ? used : null,
    faces,
  };
}
