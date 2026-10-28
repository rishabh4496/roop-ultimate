// Mock of the Stage 20 identity algebra (app/roop/identity_algebra.py, app/routes_identity.py,
// app/routes_identity_lab.py): recipe sanitising, the blender's diagnostics, the keyframe readout and the Morph Lab
// summary - with the SHAPES of the real responses, so the dev server and the e2e suite drive the same UI contract.
//
// The NUMBERS are a stand-in, not a measurement. The mock has no embeddings, so it models every pair of different
// sources at one fixed cosine (MOCK_PAIR_COSINE, 0 = orthogonal) and derives the rest from that Gram matrix: exact Slerp for two
// sources, the normalised weighted sum for three or more and for `linear`, keyframes as blends of those. Nothing
// here proves the backend's maths; app/tests/test_identity_slerp_keyframes.py and tests/identity_algebra_bench.py do.

export const MOCK_PAIR_COSINE = 0;     // the mock treats different sources as orthogonal, as the Morph Lab mock always has
export const MAX_SOURCES = 5;
export const MAX_KEYFRAMES = 64;
export const BLEND_MODES = ['slerp', 'linear'] as const;
export const CURVES = ['hold', 'linear', 'cubic_bezier'] as const;
export const DEFAULT_BEZIER = [0.42, 0, 0.58, 1];
export const DEFAULT_PRIVACY_COSINE = 0.3;
const DIAL_NAMES = ['age', 'gender', 'jawline', 'expression'];

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const finite = (v: any, d = 0) => { const n = Number(v); return Number.isFinite(n) ? n : d; };

export interface MockRecipe {
  enabled: boolean;
  components: { source_id: string; weight: number }[];
  dials: Record<string, number>;
  min_cosine: number;
  mode: string;
  keyframes: { frame_index: number; curve: string; bezier: number[]; weights: Record<string, number> }[];
  deidentify: { enabled: boolean; references: string[]; max_cosine: number; min_primary_cosine: number };
}

export function emptyRecipe(): MockRecipe {
  return sanitizeRecipe({});
}

/** BlendRecipe.from_payload. */
export function sanitizeRecipe(raw: any): MockRecipe {
  const p = raw && typeof raw === 'object' ? raw : {};
  const components: MockRecipe['components'] = [];
  const seen = new Set<string>();
  for (const item of Array.isArray(p.components) ? p.components : []) {
    if (components.length >= MAX_SOURCES) break;
    const sid = String(item?.source_id ?? '').trim();
    if (!item || typeof item !== 'object' || !sid || seen.has(sid)) continue;
    seen.add(sid);
    components.push({ source_id: sid, weight: clamp(finite(item.weight), 0, 100) });
  }
  const dials: Record<string, number> = {};
  for (const name of DIAL_NAMES) dials[name] = clamp(finite(p.dials?.[name]), name === 'age' ? -30 : -1, name === 'age' ? 30 : 1);
  const mode = String(p.mode ?? 'slerp').trim().toLowerCase();
  const refsIn = Array.isArray(p.deidentify?.references) ? p.deidentify.references : [];
  const references: string[] = [];
  for (const r of refsIn) {
    const sid = String(r ?? '').trim();
    if (sid && !references.includes(sid) && references.length < MAX_SOURCES) references.push(sid);
  }
  const byFrame = new Map<number, MockRecipe['keyframes'][number]>();
  for (const k of (Array.isArray(p.keyframes) ? p.keyframes : []).slice(0, MAX_KEYFRAMES)) {
    const f = finite(k?.frame_index, -1);
    if (!k || typeof k !== 'object' || f < 0) continue;
    const weights: Record<string, number> = {};
    for (const [sid, w] of Object.entries(k.weights && typeof k.weights === 'object' ? k.weights : {})) {
      if (seen.has(sid)) weights[sid] = clamp(finite(w), 0, 100);
    }
    if (Object.values(weights).reduce((a, b) => a + b, 0) <= 1e-12) continue;
    const curve = String(k.curve ?? 'linear').trim().toLowerCase();
    const b = Array.isArray(k.bezier) ? k.bezier.slice(0, 4).map((x: any) => clamp(finite(x, NaN), 0, 1)) : [];
    byFrame.set(Math.round(f), {
      frame_index: Math.round(f),
      curve: (CURVES as readonly string[]).includes(curve) ? curve : 'linear',
      bezier: b.length === 4 && b.every(Number.isFinite) ? b : [...DEFAULT_BEZIER],
      weights,
    });
  }
  return {
    enabled: !!p.enabled,
    components,
    dials,
    min_cosine: clamp(finite(p.min_cosine, 0.8), 0.5, 0.99),
    mode: (BLEND_MODES as readonly string[]).includes(mode) ? mode : 'slerp',
    keyframes: [...byFrame.keys()].sort((a, b) => a - b).map((f) => byFrame.get(f)!),
    deidentify: {
      enabled: !!p.deidentify?.enabled,
      references,
      max_cosine: clamp(finite(p.deidentify?.max_cosine, DEFAULT_PRIVACY_COSINE), 0, 0.95),
      min_primary_cosine: clamp(finite(p.deidentify?.min_primary_cosine, 0), 0, 0.99),
    },
  };
}

export const recipeActive = (r: MockRecipe) =>
  r.enabled && (r.components.length >= 2 || Object.values(r.dials).some((v) => Math.abs(v) > 1e-9)
    || (r.deidentify.enabled && r.deidentify.references.length > 0));

/** identity_algebra.bezier_ease: bisection on x(t) = u. */
export function bezierEase(u: number, x1: number, y1: number, x2: number, y2: number): number {
  const uu = clamp(u, 0, 1);
  const [a, b, c, d] = [clamp(x1, 0, 1), clamp(y1, 0, 1), clamp(x2, 0, 1), clamp(y2, 0, 1)];
  if (uu <= 0 || uu >= 1) return uu;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 60; i++) {
    const t = 0.5 * (lo + hi);
    const x = 3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t ** 2 * c + t ** 3;
    if (x < uu) lo = t; else hi = t;
  }
  const t = 0.5 * (lo + hi);
  return 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * d + t ** 3;
}

export function ease(curve: string, u: number, bezier: number[]): number {
  if (curve === 'hold') return u < 1 ? 0 : 1;
  if (curve === 'cubic_bezier') return bezierEase(u, bezier[0], bezier[1], bezier[2], bezier[3]);
  return clamp(u, 0, 1);
}

// ── geometry: weights over sources, Gram matrix G = (1 - c) I + c 1 1^T ─────
const gram = (a: number[], b: number[]) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) s += a[i] * b[j] * (i === j ? 1 : MOCK_PAIR_COSINE);
  return s;
};
const normed = (w: number[]) => { const t = w.reduce((a, b) => a + Math.max(0, b), 0); return t > 1e-12 ? w.map((x) => Math.max(0, x) / t) : w.map(() => 1 / Math.max(1, w.length)); };
const cosOf = (a: number[], b: number[]) => gram(a, b) / Math.sqrt(gram(a, a) * gram(b, b));

/** cos(blend, source k) for normalised weights `w`: exact Slerp for two sources in slerp mode, else the weighted sum. */
function cosToEach(w: number[], mode: string): number[] {
  if (w.length === 2 && mode === 'slerp') {
    const theta = Math.acos(MOCK_PAIR_COSINE);
    // the blend sits at angle w[1]*theta from source 0 and w[0]*theta from source 1
    return [Math.cos(w[1] * theta), Math.cos(w[0] * theta)];
  }
  return w.map((_, k) => cosOf(w, w.map((__, i) => (i === k ? 1 : 0))));
}

/** The identity guard's report for an explicit-cosine de-identification of a vector at `cosBefore` to each reference. */
function deidReport(cosBefore: number[], maxCosine: number, floor: number) {
  const after = cosBefore.map((c) => Math.min(c, maxCosine));
  const moved = after.some((c, i) => c < cosBefore[i] - 1e-12);
  const cmax = Math.max(...cosBefore);
  // |r| for one reference is sqrt(1 - c^2); the closed form of the real algebra needs the actual vectors, so the mock
  // reports the primary cosine for the single-reference case and a plausible value otherwise.
  const phi = Math.acos(Math.sqrt(Math.max(0, 1 - cmax * cmax)));
  const s = moved && phi > 1e-9 ? 1 - Math.asin(Math.min(1, (maxCosine / cmax) * Math.sin(phi))) / phi : 0;
  const toPrimary = moved ? Math.cos(s * phi) : 1;
  const floored = floor > 0 && toPrimary < floor;
  return {
    applied: moved, satisfied: !floored, reason: !moved ? 'already_below' : floored ? 'floor' : '',
    cosine_before: cosBefore.map((c) => Number(c.toFixed(4))), cosine_after: after.map((c) => Number(c.toFixed(4))),
    cosine_to_primary: Number(toPrimary.toFixed(4)), progress: Number(s.toFixed(4)),
  };
}

/** describe_keyframes(): each keyframe's angle to the next, sampled eased progress and cosines, the smoothness. */
export function evaluateKeyframes(recipe: MockRecipe, frames?: number[], samples = 96) {
  const ids = recipe.components.map((c) => c.source_id);
  const kfs = recipe.keyframes;
  const vecs = kfs.map((k) => normed(ids.map((id) => k.weights[id] ?? 0)));
  const angle = (i: number, j: number) => Math.acos(clamp(cosOf(vecs[i], vecs[j]), -1, 1));
  const out: any = { mode: recipe.mode, keyframes: [], samples: [], smoothness: null, skipped: 0 };
  if (!kfs.length) return out;
  out.keyframes = kfs.map((k, i) => ({
    frame_index: k.frame_index, curve: k.curve, bezier: k.bezier,
    ...(i + 1 < kfs.length ? { angle_to_next_deg: (angle(i, i + 1) * 180) / Math.PI } : {}),
  }));
  const first = kfs[0].frame_index;
  const last = kfs[kfs.length - 1].frame_index;
  const at = (f: number) => {
    if (f <= first) return { i: 0, s: 0 };
    if (f >= last) return { i: kfs.length - 1, s: 0 };
    let i = 0;
    while (i + 1 < kfs.length && kfs[i + 1].frame_index <= f) i++;
    const a = kfs[i];
    return { i, s: ease(a.curve, (f - a.frame_index) / (kfs[i + 1].frame_index - a.frame_index), a.bezier) };
  };
  const n = Math.max(2, Math.min(samples, last - first + 1));
  const picked = Array.isArray(frames) ? frames.slice(0, 2000)
    : last > first ? [...new Set(Array.from({ length: n }, (_, k) => Math.round(first + ((last - first) * k) / (n - 1))))] : [first];
  const cosines = (f: number) => {
    const { i, s } = at(f);
    if (i + 1 >= kfs.length || s <= 0) return vecs.map((_, m) => (m === i ? 1 : cosOf(vecs[i], vecs[m])));
    const th = angle(i, i + 1);
    const sinTh = Math.sin(th) || 1;
    return vecs.map((_, m) => {
      const ci = cosOf(vecs[i], vecs[m]);
      const cn = cosOf(vecs[i + 1], vecs[m]);
      return (Math.sin((1 - s) * th) / sinTh) * ci + (Math.sin(s * th) / sinTh) * cn;
    });
  };
  out.samples = picked.map((f) => ({ frame_index: f, segment: at(f).i, progress: at(f).s,
    cosine_to_keyframes: cosines(f).map((c) => Number(c.toFixed(4))) }));
  // per-frame steps over the whole range (a hold contributes one step, the cut, at its end)
  let max = 0; let sum = 0; let count = 0; let cuts = 0; let maxCut = 0;
  for (let f = first; f < last; f++) {
    const a = at(f);
    const b = at(f + 1);
    const th = angle(a.i, Math.min(a.i + 1, kfs.length - 1));
    const step = a.i === b.i ? Math.abs(b.s - a.s) * th : (1 - a.s) * th;
    if (kfs[a.i].curve === 'hold' && a.i !== b.i && step > 1e-9) { cuts++; maxCut = Math.max(maxCut, step); continue; }
    max = Math.max(max, step); sum += step; count++;
  }
  out.smoothness = { frames: last - first + 1, max_step: max, mean_step: count ? sum / count : 0, cuts, max_cut: maxCut, min_norm: 1, max_norm: 1 };
  return out;
}

/** describe_blend(): the readout the Identity Blender panel and the keyframe inspector read. */
export function describeBlend(recipe: MockRecipe) {
  const ids = recipe.components.map((c) => c.source_id);
  const out: any = { directions_available: ['jawline'], heldout: {}, mode: recipe.mode, components: [], missing: [] };
  if (ids.length) {
    const w = normed(recipe.components.map((c) => c.weight));
    const cos = ids.length >= 2 ? cosToEach(w, recipe.mode) : [1];
    out.components = ids.map((id, i) => ({ source_id: id, weight: w[i], cosine_to_blend: cos[i] }));
    if (ids.length >= 2) {
      out.pairwise_cosine = ids.map((_, i) => ids.map((__, j) => (i === j ? 1 : MOCK_PAIR_COSINE)));
      out.pairwise_ids = ids;
    }
    const d = recipe.deidentify;
    if (d.enabled && d.references.length) {
      // A reference that is one of the blended sources sits at the blend's own cosine to it; an outside one is at the
      // fixed pair cosine from every input, so cos(blend, outside) = sum(w) * c / |blend| with sum(w) = 1.
      const before = d.references.map((ref) => (ids.includes(ref) ? (ids.length >= 2 ? cos[ids.indexOf(ref)] : 1)
        : MOCK_PAIR_COSINE / Math.sqrt(gram(w, w))));
      out.deidentify = { references: d.references, max_cosine: d.max_cosine, ...deidReport(before, d.max_cosine, d.min_primary_cosine) };
    }
  }
  if (recipe.keyframes.length && ids.length >= 2) out.keyframes = evaluateKeyframes(recipe);
  return out;
}

/** morph_summary(): the Face Manager Morph Lab's blend readout over candidate ids. */
export function morphSummary(weightsRaw: number[], mode: string, deid: { references: number[]; max_cosine: number; min_primary_cosine: number } | null) {
  const raw = weightsRaw.map((x) => Math.max(0, x));
  const w = normed(raw);
  const cos = cosToEach(w, mode);
  const dominant = w.indexOf(Math.max(...w));
  let deidInfo: any = null;
  let finalCos = cos;
  if (deid && deid.references.length) {
    // the references are other candidates: the mock puts them at the same fixed cosine from everything, except a
    // reference that IS one of the blended inputs (index into `weightsRaw`), whose cosine is the blend's own.
    const before = deid.references.map((r) => (r >= 0 ? cos[r] : MOCK_PAIR_COSINE));
    // identity_lab.morph_summary's shape: `cosine_to_blend` (the blender's describe_blend calls the same number
    // `cosine_to_primary`, and also reports `progress`), so this one is built key by key, not spread.
    const r = deidReport(before, clamp(deid.max_cosine, 0, 0.99), deid.min_primary_cosine);
    deidInfo = { applied: r.applied, satisfied: r.satisfied, reason: r.reason, max_cosine: clamp(deid.max_cosine, 0, 0.99),
      cosine_before: r.cosine_before, cosine_after: r.cosine_after, cosine_to_blend: r.cosine_to_primary };
    finalCos = cos.map((c, i) => Math.min(c, deid.references.includes(i) ? deidInfo.cosine_after[deid.references.indexOf(i)] : c));
  }
  return {
    mode,
    weights: w.map((x) => Number(x.toFixed(4))), weights_as_given: raw, cosine_to_each: finalCos.map((c) => Number(c.toFixed(4))), dominant,
    pairwise_cosine: w.map((_, i) => w.map((__, j) => (i === j ? 1 : MOCK_PAIR_COSINE))),
    deidentify: deidInfo,
    warning: !deidInfo && finalCos[dominant] < 0.5
      ? `the blend sits far from every source (cosine to the dominant one is ${finalCos[dominant].toFixed(2)}); the swap may not resemble any of them` : null,
  };
}
