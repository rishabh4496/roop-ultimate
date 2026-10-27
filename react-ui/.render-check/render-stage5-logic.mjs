/**
 * Stage 5's pure logic (plain node, no browser): the Morph Lab's pad geometry and weight rules, the Segment
 * Chunk Matrix's pill/summary rules, the A/B reference store, and the checkpoint-vs-current settings diff.
 *
 *   node .render-check/run-module.mjs render-stage5-logic.mjs
 *
 * The pad and the weights are property checks (a few hundred random cases each) because their failures are
 * the quiet kind: a cursor that drifts a few percent off the sliders, weights that total 99.
 */
import {
  MAX_IDENTITIES, clampToPad, evenWeights, padVertices, pointAtWeights, setWeight, toPercents, togglePick, weightsAtPoint,
} from '../src/components/facemgr/morphPad.js';
import {
  PILLS, canRerender, fileMissing, groupSummary, isActive, joinDisabledReason, pillCounts, pillOf, verifyLabel,
  visibleGroups, wornPill,
} from '../src/components/segments/segmentModel.js';
import {
  clearReference, loadReference, normalizeReference, referenceCovers, referenceTime, saveReference, subscribeReference,
} from '../src/components/faceswap/abReference.js';
import { LABELS, checkpointDiff, fmtWithUnit, labelFor, restorePatch } from '../src/components/settingsDiff.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const near = (a, b, e = 1e-9) => Math.abs(a - b) <= e;
const sum = (a) => a.reduce((x, y) => x + y, 0);

let seed = 0x1badf00d;
const rnd = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return ((seed >>> 0) % 1_000_000) / 1_000_000; };

// ── morph pad ───────────────────────────────────────────────────────────────
console.log('morph pad');
{
  ok('2, 3 and 4 identities have a pad; 1 and 5 do not', [2, 3, 4].every((n) => padVertices(n).length === n)
    && [1, 5, 0].every((n) => { try { padVertices(n); return false; } catch (e) { return e instanceof RangeError; } }));
  for (const n of [2, 3, 4]) {
    const verts = padVertices(n);
    ok(`${n}: each vertex is 100% of its own identity`, verts.every((v, i) => {
      const w = weightsAtPoint(v, n);
      return w.every((x, j) => near(x, i === j ? 1 : 0, 1e-9));
    }));
    ok(`${n}: the centroid is an even blend`, weightsAtPoint(pointAtWeights(evenWeights(n), n), n).every((w) => near(w, 1 / n, 1e-9)));
  }
  ok('4: the centre of the square is an even blend', weightsAtPoint([0.5, 0.5], 4).every((w) => near(w, 0.25)));

  let bad = 0;
  let roundTrip = 0;
  for (let t = 0; t < 600; t++) {
    const n = 2 + (t % 3);
    const p = [rnd() * 1.4 - 0.2, rnd() * 1.4 - 0.2];                // inside AND outside the pad
    const w = weightsAtPoint(p, n);
    if (w.length !== n || w.some((x) => x < -1e-12 || !Number.isFinite(x)) || !near(sum(w), 1, 1e-9)) bad++;
    const q = clampToPad(p, n);
    const q2 = clampToPad(q, n);
    if (!near(q[0], q2[0], 1e-9) || !near(q[1], q2[1], 1e-9)) bad++;           // clamping is idempotent
    const back = pointAtWeights(weightsAtPoint(q, n), n);
    if (!near(back[0], q[0], 1e-9) || (n !== 2 && !near(back[1], q[1], 1e-9))) roundTrip++;
  }
  ok('weights are a partition of unity at any point, inside or outside the pad (600 random)', bad === 0, `${bad} violations`);
  ok('cursor -> weights -> cursor lands where it started (so pad and sliders cannot disagree)', roundTrip === 0, `${roundTrip} drifted`);
  ok('a point outside the triangle is pulled onto its boundary, not clamped to a corner',
    (() => { const q = clampToPad([0.5, 1.2], 3); return near(q[1], 0.86, 1e-9) && q[0] > 0.2 && q[0] < 0.8; })());
  ok('3: dragging toward a vertex raises exactly that identity', (() => {
    const [a, b] = [weightsAtPoint([0.5, 0.3], 3), weightsAtPoint([0.5, 0.2], 3)];
    return b[0] > a[0] && b[1] < a[1] && b[2] < a[2];
  })());

  let pctBad = 0;
  for (let t = 0; t < 300; t++) {
    const n = 2 + (t % 3);
    const w = Array.from({ length: n }, () => rnd() * 100);
    const p = toPercents(w);
    if (sum(p) !== 100 || p.some((x) => !Number.isInteger(x) || x < 0)) pctBad++;
  }
  ok('percentages are whole numbers that total exactly 100 (300 random)', pctBad === 0, `${pctBad} bad`);
  ok('all-zero weights show 0%, not NaN', eq(toPercents([0, 0, 0]), [0, 0, 0]));

  let wBad = 0;
  for (let t = 0; t < 400; t++) {
    const n = 2 + (t % 3);
    const w = Array.from({ length: n }, () => 1 + rnd() * 99);
    const total = sum(w);
    const scaled = w.map((x) => (x / total) * 100);
    const i = t % n;
    const v = Math.round(rnd() * 100);
    const next = setWeight(scaled, i, v, 'auto');
    if (!near(sum(next), 100, 1e-9) || !near(next[i], v, 1e-9)) wBad++;
    // the other weights keep their RATIOS
    const j = (i + 1) % n;
    const k = (i + 2) % n;
    if (n > 2 && scaled[k] > 1e-6 && next[k] > 1e-6 && !near(next[j] / next[k], scaled[j] / scaled[k], 1e-6)) wBad++;
  }
  ok('auto mode: always totals 100, the dragged weight is exact, the rest keep their ratios (400 random)', wBad === 0, `${wBad} bad`);
  ok('auto mode: 100% leaves the others at 0', eq(setWeight([50, 30, 20], 0, 100).map((x) => Math.round(x)), [100, 0, 0]));
  ok('auto mode: others that were all 0 share the remainder evenly', eq(setWeight([100, 0, 0], 0, 40), [40, 30, 30]));
  ok('free mode changes only the dragged weight', eq(setWeight([50, 30, 20], 1, 80, 'free'), [50, 80, 20]));
  ok('weights are clamped to 0..100 and NaN is 0', eq(setWeight([50, 50], 0, 140, 'free'), [100, 50]) && eq(setWeight([50, 50], 0, NaN, 'free'), [0, 50]));
  ok('even weights total 100', near(sum(evenWeights(3)), 100));
  ok('picks are capped at four, kept in order, and toggle off', (() => {
    let p = [];
    for (const id of ['a', 'b', 'c', 'd', 'e']) p = togglePick(p, id);
    return eq(p, ['a', 'b', 'c', 'd']) && MAX_IDENTITIES === 4 && eq(togglePick(p, 'b'), ['a', 'c', 'd']);
  })());
}

// ── segment matrix ──────────────────────────────────────────────────────────
console.log('segment chunk matrix');
{
  const states = { QUEUED: 'queued', RECOVERABLE: 'queued', PREPARING: 'rendering', PROCESSING: 'rendering',
    PAUSE_REQUESTED: 'rendering', PAUSED: 'rendering', COMPLETED: 'completed', FAILED: 'failed',
    CANCELLED: 'failed', INTERRUPTED: 'failed' };
  ok('all ten queue states map to one of the four pills', Object.entries(states).every(([s, p]) => pillOf(s) === p));
  ok('an unknown or missing state reads as queued, never as done', pillOf('WAT') === 'queued' && pillOf(undefined) === 'queued');
  ok('the pills are Queued / Rendering / Completed / Failed', eq(Object.values(PILLS).map((p) => p.label), ['Queued', 'Rendering', 'Completed', 'Failed']));
  ok('only finished or failed slices can be re-rendered',
    canRerender({ state: 'COMPLETED' }) && canRerender({ state: 'FAILED' }) && canRerender({ state: 'INTERRUPTED' })
    && canRerender({ state: 'RECOVERABLE' }) && !canRerender({ state: 'PROCESSING' }) && !canRerender({ state: 'QUEUED' }));
  const seg = (i, state, extra = {}) => ({ index: i, state, exists: state === 'COMPLETED', name: `seg_000${i}.mp4`, error: '', ...extra });
  ok('a slice that finished but whose file is gone is not shown green', (() => {
    const s = seg(2, 'COMPLETED', { exists: false });
    return fileMissing(s) && wornPill(s).key === 'failed' && wornPill(s).label === 'Missing' && wornPill(s).note.includes('seg_0002');
  })());
  ok('a failed slice carries its error', wornPill(seg(1, 'FAILED', { error: 'boom' })).note === 'boom');
  const group = (segs, extra = {}) => ({ group: 'G', count: segs.length, segments: segs, can_join: false, joined: null, ...extra });
  const g1 = group([seg(1, 'COMPLETED'), seg(2, 'PROCESSING'), seg(3, 'QUEUED'), seg(4, 'FAILED')]);
  ok('counts per pill', eq(pillCounts(g1), { queued: 1, rendering: 1, completed: 1, failed: 1 }));
  ok('the summary line', groupSummary(g1) === '1 of 4 rendered · 1 rendering · 1 queued · 1 failed', groupSummary(g1));
  const joined = group([seg(1, 'COMPLETED'), seg(2, 'COMPLETED')], { can_join: true, joined: { exists: true, result: { verify: { ok: true, joined_s: 3, expected_s: 3 } } } });
  ok('a verified join says joined', groupSummary(joined) === '2 of 2 rendered · joined');
  const bad = group([seg(1, 'COMPLETED'), seg(2, 'COMPLETED')], { joined: { exists: true, result: { verify: { ok: false, reason: 'x' } } } });
  ok('a join that failed its duration check says so in the summary', groupSummary(bad).includes('DURATION MISMATCH'));
  ok('verify labels: pass / fail / unmeasured / none',
    verifyLabel({ ok: true, joined_s: 3, expected_s: 3.04 }).text === 'duration 3.00 s matches 3.04 s'
    && verifyLabel({ ok: false, reason: 'joined video is 2.00s' }).text === 'joined video is 2.00s'
    && verifyLabel({ ok: null }).ok === null && verifyLabel(null) === null);
  ok('the join button explains itself', joinDisabledReason(joined) === '' && joinDisabledReason(g1).length > 0
    && joinDisabledReason(joined, true) === 'Joining…' && joinDisabledReason({ ...g1, join_blocked: 'slice(s) 2 not finished' }) === 'slice(s) 2 not finished');
  ok('only real splits are listed', visibleGroups([{ count: 1, segments: [seg(1, 'COMPLETED')] }, g1]).length === 1);
  ok('polling continues while anything is queued or rendering, stops when settled',
    isActive([g1]) && !isActive([joined]) && !isActive([]) && !isActive(undefined));
}

// ── A/B reference ───────────────────────────────────────────────────────────
console.log('A/B reference');
{
  const store = new Map();
  const listeners = {};
  globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init?.detail; } };
  globalThis.window = {
    localStorage: { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) },
    addEventListener: (t, f) => { (listeners[t] ||= new Set()).add(f); },
    removeEventListener: (t, f) => { listeners[t]?.delete(f); },
    dispatchEvent: (e) => { for (const f of listeners[e.type] || []) f(e); return true; },
  };
  ok('rejects junk', normalizeReference(null) === null && normalizeReference({}) === null && normalizeReference({ url: '  ' }) === null && normalizeReference('x') === null);
  const ref = normalizeReference({ url: '/outputs/a.mp4', kind: 'video', fps: '29.97', offset: '9.6' });
  ok('normalises kind, fps and offset', ref.kind === 'video' && near(ref.fps, 29.97) && ref.offset === 10 && ref.name === 'a.mp4');
  ok('an unknown kind is an image, a bad fps / offset is dropped', (() => { const r = normalizeReference({ url: 'x.png', kind: 'gif', fps: -3, offset: -4 }); return r.kind === 'image' && r.fps === null && r.offset === 0; })());
  const seen = [];
  const off = subscribeReference((r) => seen.push(r?.name ?? null));
  saveReference({ url: '/outputs/b.png', name: 'B' });
  ok('save is stored, announced to a mounted Face Swap, and read back', loadReference().name === 'B' && eq(seen, ['B']));
  clearReference();
  ok('clear removes it and announces null', loadReference() === null && eq(seen, ['B', null]));
  off();
  saveReference({ url: '/outputs/c.png' });
  ok('an unsubscribed listener hears nothing more', eq(seen, ['B', null]));
  window.localStorage.setItem('roop_ab_reference', '{not json');
  ok('corrupt storage is "no reference", not a crash', loadReference() === null);
  window.localStorage.setItem = () => { throw new Error('quota'); };
  ok('a storage error does not throw out of save', (() => { try { saveReference({ url: '/x.png' }); return true; } catch { return false; } })());

  const v = { kind: 'video', fps: 25, offset: 100 };
  ok('time is the clip frame minus the render\'s own start', near(referenceTime(v, 101, 25, 10), 0) && near(referenceTime(v, 126, 25, 10), 1));
  ok('frames before the render start show its first frame', referenceTime(v, 5, 25, 10) === 0);
  ok('frames past its end show its last frame, not a seek error', near(referenceTime(v, 9999, 25, 10), 10 - 1 / 25));
  ok('the reference\'s own fps beats the clip\'s', near(referenceTime({ ...v, fps: 50, offset: 0 }, 51, 25, 10), 1));
  ok('coverage says whether the picture is the real frame or the nearest end',
    referenceCovers(v, 150, 25, 10) && !referenceCovers(v, 50, 25, 10) && !referenceCovers(v, 9999, 25, 10) && referenceCovers({ kind: 'image' }, 1, 25, 0));
}

// ── checkpoint vs current ───────────────────────────────────────────────────
console.log('checkpoint diff');
{
  const checkpoint = { codeformer_fidelity: 0.5, face_mask_blend: 12, swap_model: 'realswap', same: 1, only_then: 7, selected_theme: 'a', face_mapping: [1] };
  const current = { codeformer_fidelity: 0.8, face_mask_blend: 24, swap_model: 'realswap', same: 1, only_now: 9, selected_theme: 'b', face_mapping: [2] };
  const rows = checkpointDiff(checkpoint, current);
  const by = Object.fromEntries(rows.map((r) => [r.k, r]));
  ok('the brief\'s examples read as the brief writes them',
    by.codeformer_fidelity.fromText === '0.50' && by.codeformer_fidelity.toText === '0.80' && by.face_mask_blend.fromText === '12px' && by.face_mask_blend.toText === '24px');
  ok('parameters have human labels', by.codeformer_fidelity.label === 'CodeFormer fidelity' && by.face_mask_blend.label === 'Mask feather');
  ok('unchanged, noise (theme) and non-primitive (mapping) settings are not rows', !('swap_model' in by) && !('same' in by) && !('selected_theme' in by) && !('face_mapping' in by));
  ok('rows are classified', by.codeformer_fidelity.kind === 'changed' && by.only_then.kind === 'only-checkpoint' && by.only_now.kind === 'only-current');
  ok('labelled parameters come first', rows[0].known && rows[1].known && !rows.at(-1).known);
  ok('a number that differs only past two decimals is the same', checkpointDiff({ a: 0.5 }, { a: 0.500001 }).length === 0);
  ok('booleans read on/off', checkpointDiff({ a: true }, { a: false })[0].fromText === 'on' && checkpointDiff({ a: true }, { a: false })[0].toText === 'off');
  ok('units only attach to numbers', fmtWithUnit('face_mask_blend', 12) === '12px' && fmtWithUnit('face_mask_blend', undefined) === '—' && fmtWithUnit('swap_model', 'x') === 'x');
  ok('an unlabelled key still gets a readable name', labelFor('some_new_setting') === 'some new setting' && labelFor('swap_model') === LABELS.swap_model);

  ok('restoring writes the CHECKPOINT value for the selected rows only',
    eq(restorePatch(rows, ['codeformer_fidelity']), { codeformer_fidelity: 0.5 }));
  ok('restoring every row restores every restorable one', eq(Object.keys(restorePatch(rows, rows.map((r) => r.k))).sort(), ['codeformer_fidelity', 'face_mask_blend', 'only_then']));
  ok('a key the checkpoint never set cannot be "restored"', !('only_now' in restorePatch(rows, ['only_now'])));
  ok('applying the patch makes those parameters match', (() => {
    const next = { ...current, ...restorePatch(rows, ['face_mask_blend']) };
    const after = Object.fromEntries(checkpointDiff(checkpoint, next).map((r) => [r.k, r]));
    return !('face_mask_blend' in after) && 'codeformer_fidelity' in after;
  })());
  ok('identical settings have no rows', checkpointDiff(checkpoint, checkpoint).length === 0 && checkpointDiff(null, null).length === 0);
}

console.log(failures === 0 ? '\nall stage 5 logic checks passed' : `\n${failures} stage 5 logic check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
