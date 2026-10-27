// Geometry and weight rules for the Morph Lab's identity pad. Pure: no React, no DOM.
//
// The pad is a unit square. 2 identities sit on a horizontal line, 3 on a triangle (true barycentric
// weights), 4 on the corners of a square (bilinear weights). In every case the weights are a partition
// of unity that is LINEAR in the cursor position for 2 and 3 identities and bilinear for 4, and
// `pointAtWeights(weightsAtPoint(p)) === p` for a point inside the pad - so dragging the cursor and
// dragging the sliders edit the same thing and never disagree.

export const MIN_IDENTITIES = 2;
export const MAX_IDENTITIES = 4;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Vertex positions (0..1) for n identities, in the order the identities were picked. */
export function padVertices(n) {
  if (n === 2) return [[0.1, 0.5], [0.9, 0.5]];
  if (n === 3) return [[0.5, 0.1], [0.1, 0.86], [0.9, 0.86]];
  if (n === 4) return [[0.12, 0.12], [0.88, 0.12], [0.88, 0.88], [0.12, 0.88]];
  throw new RangeError(`the pad holds ${MIN_IDENTITIES} to ${MAX_IDENTITIES} identities, not ${n}`);
}

function triangleWeights([px, py], [a, b, c]) {
  const d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  const wa = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d;
  const wb = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d;
  return [wa, wb, 1 - wa - wb];
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

/** The nearest point of the pad to `point` (the segment, the triangle, or the square). */
export function clampToPad(point, n) {
  const verts = padVertices(n);
  const [x, y] = point;
  if (n === 4) return [clamp(x, verts[0][0], verts[1][0]), clamp(y, verts[0][1], verts[3][1])];
  if (n === 2) return [clamp(x, verts[0][0], verts[1][0]), verts[0][1]];
  const w = triangleWeights([x, y], verts);
  if (w.every((v) => v >= 0)) return [x, y];
  // Outside: the closest point on the triangle's boundary.
  let best = null;
  for (let i = 0; i < 3; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % 3];
    const abx = b[0] - a[0];
    const aby = b[1] - a[1];
    const t = clamp(((x - a[0]) * abx + (y - a[1]) * aby) / (abx * abx + aby * aby), 0, 1);
    const q = [a[0] + t * abx, a[1] + t * aby];
    const d = (q[0] - x) ** 2 + (q[1] - y) ** 2;
    if (!best || d < best.d) best = { q, d };
  }
  return best.q;
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

/** Add or remove an identity id from the pick list (max 4, in pick order); weights reset to even. */
export function togglePick(picked, id) {
  if (picked.includes(id)) return picked.filter((p) => p !== id);
  if (picked.length >= MAX_IDENTITIES) return picked;
  return [...picked, id];
}
