// Vector roto zones: the pure model behind the Pen tool (P).
//
// A zone is a closed Bezier / polygon outline in FRAME PIXELS plus how it is applied:
//
//   mode 'exclude'  the original pixels are restored inside it (a microphone, a hand, a watermark)
//   mode 'include'  the swap is forced inside it (it cancels the occluder mask there, and any
//                   exclusion zone it overlaps)
//   feather         edge softness in frame pixels; the outline is the 50% line, so a feather never
//                   grows or shrinks the zone that was drawn
//   frames          [first, last] UI frame numbers (1-based, inclusive) or null for the whole clip
//   pin             null, or { ref_nose, ref_chin, source_index }: the zone follows a face. Two
//                   landmarks fix a similarity transform exactly (translate + rotate + uniform
//                   scale), so it needs no tracker state and renders identically on any worker.
//
// `flatten`, `pinTransform` and the anchor rule are the SAME algorithms as app/roop/roto_polygons.py,
// pinned to its output by .render-check/render-stage11-logic.mjs: the outline the user draws is the
// outline that renders. Everything here is a pure function returning new objects; the stage holds the
// state. Pure so the maths can be checked under node, without a browser.

export const BEZIER_STEPS = 16;
export const MAX_ZONES = 24;
export const MAX_POINTS = 512;
export const MAX_FEATHER = 300;
export const MODES = ['exclude', 'include'];

// Strict: Number(null) is 0 and Number('') is 0, so a missing frame or coordinate would pass a bare
// Number.isFinite(Number(v)) as a real zero (the trap faceRoles.js documents).
const finite = (v) => v !== null && v !== undefined && v !== '' && typeof v !== 'boolean' && Number.isFinite(Number(v));
const num = (v, d = 0) => (finite(v) ? Number(v) : d);

export const point = (x, y) => ({ x, y, hin: null, hout: null });

export function newZone({ id, mode = 'exclude', feather = 8, frames = null } = {}) {
  return {
    id: id || `roto_${Math.random().toString(36).slice(2, 8)}`,
    mode: MODES.includes(mode) ? mode : 'exclude',
    enabled: true,
    closed: false,
    feather: Math.min(MAX_FEATHER, Math.max(0, num(feather, 8))),
    frames,
    pin: null,
    points: [],
  };
}

// ── editing ─────────────────────────────────────────────────────────────────

export function addPoint(zone, x, y) {
  if (zone.points.length >= MAX_POINTS) return zone;
  return { ...zone, points: [...zone.points, point(x, y)] };
}

export function movePoint(zone, i, x, y) {
  if (i < 0 || i >= zone.points.length) return zone;
  const points = zone.points.map((p, k) => (k === i ? { ...p, x, y } : p));
  return { ...zone, points };
}

/**
 * Set a Bezier handle (an OFFSET from its anchor). With `mirror` the opposite handle takes the
 * negated offset, which is a smooth point; a click-drag with the pen creates exactly that.
 */
export function setHandle(zone, i, which, dx, dy, mirror = true) {
  if (i < 0 || i >= zone.points.length || (which !== 'hin' && which !== 'hout')) return zone;
  const other = which === 'hout' ? 'hin' : 'hout';
  const points = zone.points.map((p, k) => {
    if (k !== i) return p;
    const next = { ...p, [which]: [dx, dy] };
    if (mirror) next[other] = [-dx, -dy];
    return next;
  });
  return { ...zone, points };
}

export function clearHandles(zone, i) {
  const points = zone.points.map((p, k) => (k === i ? { ...p, hin: null, hout: null } : p));
  return { ...zone, points };
}

export function removePoint(zone, i) {
  if (i < 0 || i >= zone.points.length) return zone;
  const points = zone.points.filter((_, k) => k !== i);
  // a closed zone that falls below a triangle is no longer a zone
  return { ...zone, points, closed: zone.closed && points.length >= 3 };
}

export const closeZone = (zone) => (zone.points.length >= 3 ? { ...zone, closed: true } : zone);
export const setMode = (zone, mode) => (MODES.includes(mode) ? { ...zone, mode } : zone);
export const setFeather = (zone, px) => ({ ...zone, feather: Math.min(MAX_FEATHER, Math.max(0, num(px, zone.feather))) });

/** `range` is [first, last] (any order) or null for the whole clip. */
export function setRange(zone, range) {
  if (!range) return { ...zone, frames: null };
  const a = Math.round(num(range[0], 1));
  const b = Math.round(num(range[1], a));
  return { ...zone, frames: [Math.min(a, b), Math.max(a, b)] };
}

// ── geometry (identical to roto_polygons.flatten) ───────────────────────────

/** The outline as [[x, y], ...]: straight segments stay straight, a segment with a handle on either
 *  end becomes a cubic Bezier sampled at `steps`. `closed` (the default, and what the backend always
 *  gets) includes the segment back to the first point; an open zone being drawn does not. */
export function flatten(points, steps = BEZIER_STEPS, closed = true) {
  const n = points.length;
  const out = [];
  const segs = closed ? n : n - 1;
  for (let i = 0; i < segs; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % n];
    out.push([a.x, a.y]);
    if (!a.hout && !b.hin) continue;
    const ho = a.hout || [0, 0];
    const hi = b.hin || [0, 0];
    const c1 = [a.x + ho[0], a.y + ho[1]];
    const c2 = [b.x + hi[0], b.y + hi[1]];
    for (let s = 1; s < steps; s += 1) {
      const t = s / steps;
      const u = 1 - t;
      const w0 = u * u * u;
      const w1 = 3 * u * u * t;
      const w2 = 3 * u * t * t;
      const w3 = t * t * t;
      out.push([
        w0 * a.x + w1 * c1[0] + w2 * c2[0] + w3 * b.x,
        w0 * a.y + w1 * c1[1] + w2 * c2[1] + w3 * b.y,
      ]);
    }
  }
  if (!closed && n > 0) out.push([points[n - 1].x, points[n - 1].y]);
  return out;
}

export function pointInPolygon(poly, x, y) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function polygonArea(poly) {
  let a = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    a += (poly[j][0] + poly[i][0]) * (poly[j][1] - poly[i][1]);
  }
  return Math.abs(a / 2);
}

const dist = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);

/** Index of the control point within `tol` of (x, y), nearest first, or -1. */
export function hitPoint(zone, x, y, tol) {
  let best = -1;
  let bd = tol;
  zone.points.forEach((p, i) => {
    const d = dist(p.x, p.y, x, y);
    if (d <= bd) { bd = d; best = i; }
  });
  return best;
}

/** {i, which} of the Bezier handle knob within `tol`, or null. Handles are drawn at anchor + offset. */
export function hitHandle(zone, x, y, tol) {
  let best = null;
  let bd = tol;
  zone.points.forEach((p, i) => {
    ['hin', 'hout'].forEach((which) => {
      const h = p[which];
      if (!h) return;
      const d = dist(p.x + h[0], p.y + h[1], x, y);
      if (d <= bd) { bd = d; best = { i, which }; }
    });
  });
  return best;
}

/** The segment (and parameter t along it) of the outline nearest (x, y) within `tol`, or null. */
export function hitSegment(zone, x, y, tol, steps = BEZIER_STEPS) {
  const n = zone.points.length;
  if (n < 2) return null;
  let best = null;
  let bd = tol;
  const segs = zone.closed ? n : n - 1;
  for (let i = 0; i < segs; i += 1) {
    const a = zone.points[i];
    const b = zone.points[(i + 1) % n];
    const curved = a.hout || b.hin;
    const sub = curved ? steps : 1;
    // flatten([a, b]) also wraps b -> a; its first `steps` samples are a -> b with b left off
    const line = curved ? [...flatten([a, b], steps).slice(0, steps), [b.x, b.y]] : [[a.x, a.y], [b.x, b.y]];
    for (let k = 0; k < line.length - 1; k += 1) {
      const [px, py] = line[k];
      const [qx, qy] = line[k + 1];
      const vx = qx - px;
      const vy = qy - py;
      const len2 = vx * vx + vy * vy || 1;
      const u = Math.max(0, Math.min(1, ((x - px) * vx + (y - py) * vy) / len2));
      const d = dist(px + u * vx, py + u * vy, x, y);
      if (d <= bd) { bd = d; best = { seg: i, t: (k + u) / sub }; }
    }
  }
  return best;
}

/** Insert a point on segment `seg` at parameter `t`, splitting a curved segment so its shape is kept
 *  (de Casteljau) and a straight one in place. */
export function insertPoint(zone, seg, t) {
  const n = zone.points.length;
  if (seg < 0 || seg >= n || zone.points.length >= MAX_POINTS) return zone;
  const a = zone.points[seg];
  const b = zone.points[(seg + 1) % n];
  let np;
  let na = a;
  let nb = b;
  if (!a.hout && !b.hin) {
    np = point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
  } else {
    const ho = a.hout || [0, 0];
    const hi = b.hin || [0, 0];
    const p0 = [a.x, a.y];
    const p1 = [a.x + ho[0], a.y + ho[1]];
    const p2 = [b.x + hi[0], b.y + hi[1]];
    const p3 = [b.x, b.y];
    const lerp = (u, v) => [u[0] + (v[0] - u[0]) * t, u[1] + (v[1] - u[1]) * t];
    const q0 = lerp(p0, p1);
    const q1 = lerp(p1, p2);
    const q2 = lerp(p2, p3);
    const r0 = lerp(q0, q1);
    const r1 = lerp(q1, q2);
    const s = lerp(r0, r1);
    np = { x: s[0], y: s[1], hin: [r0[0] - s[0], r0[1] - s[1]], hout: [r1[0] - s[0], r1[1] - s[1]] };
    na = { ...a, hout: [q0[0] - a.x, q0[1] - a.y] };
    nb = { ...b, hin: [q2[0] - b.x, q2[1] - b.y] };
  }
  const points = [...zone.points];
  points[seg] = na;
  const bi = (seg + 1) % n;
  if (bi === 0) {
    points[0] = nb;
    points.push(np);
  } else {
    points[bi] = nb;
    points.splice(seg + 1, 0, np);
  }
  return { ...zone, points };
}

// ── the 2-point landmark pin (identical to roto_polygons.pin_transform / face_anchors) ──────────

/** Preview face i's [nose, chin] as reported by /api/preview (`anchors[i]`), or null. */
export function anchorsOf(anchors, i) {
  const a = anchors && anchors[i];
  if (!a || !a[0] || !a[1]) return null;
  if (![a[0][0], a[0][1], a[1][0], a[1][1]].every(finite)) return null;
  if (dist(a[0][0], a[0][1], a[1][0], a[1][1]) < 1e-3) return null;
  return { nose: [Number(a[0][0]), Number(a[0][1])], chin: [Number(a[1][0]), Number(a[1][1])] };
}

/** The similarity taking the reference anchor pair onto the current one: x' = a x + b y + tx,
 *  y' = c x + d y + ty. null when either pair is degenerate. */
export function pinTransform(refNose, refChin, curNose, curChin) {
  const rvx = refChin[0] - refNose[0];
  const rvy = refChin[1] - refNose[1];
  const cvx = curChin[0] - curNose[0];
  const cvy = curChin[1] - curNose[1];
  const rl = Math.hypot(rvx, rvy);
  const cl = Math.hypot(cvx, cvy);
  if (rl < 1e-6 || cl < 1e-6) return null;
  const s = cl / rl;
  const th = Math.atan2(cvy, cvx) - Math.atan2(rvy, rvx);
  const c = s * Math.cos(th);
  const si = s * Math.sin(th);
  return {
    a: c, b: -si, tx: curNose[0] - (c * refNose[0] - si * refNose[1]),
    c: si, d: c, ty: curNose[1] - (si * refNose[0] + c * refNose[1]),
  };
}

export const applyTransform = (m, [x, y]) => [m.a * x + m.b * y + m.tx, m.c * x + m.d * y + m.ty];

/** Pin a zone to the face whose anchors are given. The reference is THIS frame's pose. */
export function pinZone(zone, anchors, sourceIndex = null) {
  if (!anchors) return zone;
  return {
    ...zone,
    pin: {
      ref_nose: anchors.nose,
      ref_chin: anchors.chin,
      source_index: Number.isInteger(sourceIndex) && sourceIndex >= 0 ? sourceIndex : null,
    },
  };
}

export const unpinZone = (zone) => ({ ...zone, pin: null });

/**
 * Where the zone sits on a frame. `faces` is [{ source_index, nose, chin }] for the faces on THAT
 * frame. Picks the face the way the renderer does: the pin's source index if it is present, else the
 * face whose nose is nearest the reference nose. Returns { outline, tracked }; an unpinned zone, or a
 * pinned one with no face to follow, stays where it was drawn (tracked: false).
 */
export function placedOutline(zone, faces = []) {
  const outline = flatten(zone.points);
  if (!zone.pin || !faces.length) return { outline, tracked: false };
  let face = null;
  if (zone.pin.source_index != null) face = faces.find((f) => f.source_index === zone.pin.source_index) || null;
  if (!face) {
    face = faces.reduce((best, f) => {
      const d = dist(f.nose[0], f.nose[1], zone.pin.ref_nose[0], zone.pin.ref_nose[1]);
      return !best || d < best.d ? { f, d } : best;
    }, null)?.f || null;
  }
  const m = face && pinTransform(zone.pin.ref_nose, zone.pin.ref_chin, face.nose, face.chin);
  if (!m) return { outline, tracked: false };
  return { outline: outline.map((p) => applyTransform(m, p)), tracked: true };
}

// ── per frame / request ─────────────────────────────────────────────────────

export function activeAt(zone, frame) {
  if (!zone.enabled) return false;
  if (!zone.frames || frame == null) return true;
  return frame >= zone.frames[0] && frame <= zone.frames[1];
}

const r2 = (v) => Math.round(v * 100) / 100;

/** What the request carries: closed zones with a triangle or more, rounded to 0.01 px, no UI state. */
export function serialize(zones) {
  return (zones || [])
    .filter((z) => z && z.enabled !== false && z.closed && z.points.length >= 3)
    .slice(0, MAX_ZONES)
    .map((z) => ({
      id: z.id,
      mode: z.mode,
      feather: r2(z.feather),
      frames: z.frames ? [z.frames[0], z.frames[1]] : null,
      points: z.points.slice(0, MAX_POINTS).map((p) => ({
        x: r2(p.x),
        y: r2(p.y),
        hin: p.hin ? [r2(p.hin[0]), r2(p.hin[1])] : null,
        hout: p.hout ? [r2(p.hout[0]), r2(p.hout[1])] : null,
      })),
      pin: z.pin ? {
        enabled: true,
        ref_nose: z.pin.ref_nose.map(r2),
        ref_chin: z.pin.ref_chin.map(r2),
        source_index: z.pin.source_index,
      } : null,
    }));
}

export const MODE_COLOR = { exclude: '#fb7185', include: '#34d399' };

/** Which preset the zone's frame range is: 'clip' | 'frame' | 'from' | 'until' (the toolbar's select). */
export function rangeKind(zone) {
  if (!zone || !zone.frames) return 'clip';
  const [a, b] = zone.frames;
  if (a === b) return 'frame';
  if (a === 1) return 'until';
  return 'from';
}

/** The [first, last] range a preset means at `frame` of a clip of `maxFrames`; null = whole clip. */
export function rangeFor(kind, frame, maxFrames) {
  if (kind === 'frame') return [frame, frame];
  if (kind === 'from') return [frame, Math.max(frame, maxFrames)];
  if (kind === 'until') return [1, frame];
  return null;
}

/** A short label for the zone list: "Exclude 1 · 6 pts · pinned · f120-180". */
export function describe(zone, index = 0) {
  const parts = [`${zone.mode === 'include' ? 'Include' : 'Exclude'} ${index + 1}`, `${zone.points.length} pts`];
  if (!zone.closed) parts.push('open');
  if (zone.pin) parts.push('pinned');
  if (zone.frames) parts.push(`f${zone.frames[0]}-${zone.frames[1]}`);
  return parts.join(' · ');
}
