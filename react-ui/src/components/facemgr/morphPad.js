// Geometry and weight rules for the Morph Lab's identity pad. Pure: no React, no DOM.
//
// The pad is a unit square. 2 identities sit on a horizontal line, 3 on a triangle (true barycentric
// weights), 4 on the corners of a square (bilinear weights), 5 on a regular pentagon (mean-value
// coordinates: non-negative, a partition of unity, exact linear precision on a convex polygon). In every
// case the weights are a partition of unity that is LINEAR in the cursor position for 2, 3 and 5
// identities and bilinear for 4, and `pointAtWeights(weightsAtPoint(p)) === p` for a point inside the pad -
// so dragging the cursor and dragging the sliders edit the same thing and never disagree.

export const MIN_IDENTITIES = 2;
export const MAX_IDENTITIES = 5;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Vertex positions (0..1) for n identities, in the order the identities were picked. */
export function padVertices(n) {
  if (n === 2) return [[0.1, 0.5], [0.9, 0.5]];
  if (n === 3) return [[0.5, 0.1], [0.1, 0.86], [0.9, 0.86]];
  if (n === 4) return [[0.12, 0.12], [0.88, 0.12], [0.88, 0.88], [0.12, 0.88]];
  if (n === 5) {
    // Regular pentagon, point up, inscribed in a circle of radius 0.4 about the pad's centre.
    return Array.from({ length: 5 }, (_, i) => {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
      return [0.5 + 0.4 * Math.cos(a), 0.5 + 0.4 * Math.sin(a)];
    });
  }
  throw new RangeError(`the pad holds ${MIN_IDENTITIES} to ${MAX_IDENTITIES} identities, not ${n}`);
}

function triangleWeights([px, py], [a, b, c]) {
  const d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  const wa = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d;
  const wb = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d;
  return [wa, wb, 1 - wa - wb];
}

/**
 * Mean-value coordinates (Floater 2003) of `p` in the convex polygon `verts`: w_i is proportional to
 * (tan(a_{i-1}/2) + tan(a_i/2)) / |p - v_i| where a_i is the angle at p between v_i and v_{i+1}. Non-negative
 * inside a convex polygon, sums to 1 after normalising, and reproduces p exactly (sum w_i v_i = p). On the
 * boundary the formula is singular, so a vertex gets weight 1 and a point on an edge is shared linearly by
 * that edge's two vertices.
 */
function meanValueWeights([px, py], verts) {
  const m = verts.length;
  const EPS = 1e-12;
  const d = verts.map((v) => Math.hypot(v[0] - px, v[1] - py));
  for (let i = 0; i < m; i++) {
    if (d[i] < EPS) return verts.map((_, j) => (i === j ? 1 : 0));
  }
  for (let i = 0; i < m; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % m];
    const ex = b[0] - a[0];
    const ey = b[1] - a[1];
    const cross = (px - a[0]) * ey - (py - a[1]) * ex;
    const dot = (px - a[0]) * ex + (py - a[1]) * ey;
    const len2 = ex * ex + ey * ey;
    if (Math.abs(cross) < 1e-9 * Math.sqrt(len2) && dot >= 0 && dot <= len2) {
      const t = dot / len2;
      return verts.map((_, j) => (j === i ? 1 - t : j === (i + 1) % m ? t : 0));
    }
  }
  const tanHalf = verts.map((v, i) => {
    const nx = verts[(i + 1) % m];
    const ax = v[0] - px;
    const ay = v[1] - py;
    const bx = nx[0] - px;
    const by = nx[1] - py;
    const dotp = ax * bx + ay * by;
    const crs = ax * by - ay * bx;
    return Math.abs(crs) / (d[i] * d[(i + 1) % m] + dotp);      // tan(angle/2), angle in [0, pi)
  });
  const raw = verts.map((_, i) => (tanHalf[(i + m - 1) % m] + tanHalf[i]) / d[i]);
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map((w) => w / total);
}

/** The point of the polygon's boundary closest to `(x, y)`. */
function closestOnPolygon(verts, x, y) {
  let best = null;
  for (let i = 0; i < verts.length; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % verts.length];
    const abx = b[0] - a[0];
    const aby = b[1] - a[1];
    const t = clamp(((x - a[0]) * abx + (y - a[1]) * aby) / (abx * abx + aby * aby), 0, 1);
    const q = [a[0] + t * abx, a[1] + t * aby];
    const dist = (q[0] - x) ** 2 + (q[1] - y) ** 2;
    if (!best || dist < best.d) best = { q, d: dist };
  }
  return best.q;
}

/** True when `(x, y)` is inside (or on) the convex polygon `verts`, whichever way it winds. */
function insideConvex(verts, x, y) {
  let sign = 0;
  for (let i = 0; i < verts.length; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % verts.length];
    const c = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    if (Math.abs(c) < 1e-12) continue;
    if (sign === 0) sign = Math.sign(c);
    else if (Math.sign(c) !== sign) return false;
  }
  return true;
}

/** Weights (sum 1, each >= 0) for a cursor position; positions outside the pad are pulled onto it. */
export function weightsAtPoint(point, n) {
  const verts = padVertices(n);
  const p = clampToPad(point, n);
  if (n === 2) {
    const t = clamp((p[0] - verts[0][0]) / (verts[1][0] - verts[0][0]), 0, 1);
    return [1 - t, t];
  }
  if (n === 3) return triangleWeights(p, verts).map((w) => Math.max(0, w));
  if (n === 5) return meanValueWeights(p, verts);
  const u = clamp((p[0] - verts[0][0]) / (verts[1][0] - verts[0][0]), 0, 1);
  const v = clamp((p[1] - verts[0][1]) / (verts[3][1] - verts[0][1]), 0, 1);
  return [(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v];
}

/** Where the cursor sits for a set of weights: their weighted centroid of the pad's vertices. */
export function pointAtWeights(weights, n) {
  const verts = padVertices(n);
  const total = weights.reduce((a, b) => a + Math.max(0, b), 0) || 1;
  let x = 0;
  let y = 0;
  weights.forEach((w, i) => { x += (Math.max(0, w) / total) * verts[i][0]; y += (Math.max(0, w) / total) * verts[i][1]; });
  return [x, y];
}

/** The nearest point of the pad to `point` (the segment, the triangle, the square, or the pentagon). */
export function clampToPad(point, n) {
  const verts = padVertices(n);
  const [x, y] = point;
  if (n === 4) return [clamp(x, verts[0][0], verts[1][0]), clamp(y, verts[0][1], verts[3][1])];
  if (n === 2) return [clamp(x, verts[0][0], verts[1][0]), verts[0][1]];
  if (n === 5) return insideConvex(verts, x, y) ? [x, y] : closestOnPolygon(verts, x, y);
  const w = triangleWeights([x, y], verts);
  if (w.every((v) => v >= 0)) return [x, y];
  // Outside: the closest point on the triangle's boundary.
  return closestOnPolygon(verts, x, y);
}

/** Percentages that add to exactly 100 (largest-remainder), for display. */
export function toPercents(weights) {
  const total = weights.reduce((a, b) => a + Math.max(0, b), 0);
  if (total <= 0) return weights.map(() => 0);
  const raw = weights.map((w) => (Math.max(0, w) / total) * 100);
  const floors = raw.map(Math.floor);
  let left = 100 - floors.reduce((a, b) => a + b, 0);
  raw.map((r, i) => [r - floors[i], i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => {
    if (left > 0) { floors[i] += 1; left -= 1; }
  });
  return floors;
}

/**
 * Move one weight (0..100) and return the new weight set.
 *   auto : the others keep their RATIOS and share what is left, so the set always totals 100
 *   free : only this weight changes; the shares are whatever the ratios come to
 */
export function setWeight(weights, index, value, mode = 'auto') {
  const v = clamp(Number.isFinite(value) ? value : 0, 0, 100);
  const next = weights.map((w) => Math.max(0, w));
  if (mode === 'free') { next[index] = v; return next; }
  const others = next.reduce((a, w, i) => (i === index ? a : a + w), 0);
  const room = 100 - v;
  return next.map((w, i) => {
    if (i === index) return v;
    if (others <= 1e-9) return room / (next.length - 1);      // everyone else was 0: spread evenly
    return (w / others) * room;
  });
}

/** Equal weights summing to 100. */
export const evenWeights = (n) => Array.from({ length: n }, () => 100 / n);

/** Add or remove an identity id from the pick list (max MAX_IDENTITIES, in pick order); weights reset to even. */
export function togglePick(picked, id) {
  if (picked.includes(id)) return picked.filter((p) => p !== id);
  if (picked.length >= MAX_IDENTITIES) return picked;
  return [...picked, id];
}
