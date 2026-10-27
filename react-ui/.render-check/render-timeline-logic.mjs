/**
 * The multi-track timeline's pure logic (plain node, no browser): the view window, the adaptive
 * ruler, thumbnail slots, the render/cache strip, scene + presence geometry, per-segment overrides,
 * the J/K/L shuttle and the lane hit-testing.
 *
 *   node .render-check/run-module.mjs render-timeline-logic.mjs
 *
 * Most groups are PROPERTY checks against a brute-force oracle (a per-frame array), because the
 * failure these functions can have is an off-by-one at a boundary that a hand-picked example would
 * never hit: a zoom that drifts off its anchor by a frame, a strip that leaves one frame untiled, a
 * piece boundary that renders a frame twice.
 */
import {
  FRAME_CELL_MIN_PX, FRAME_VIEW_PX, fitRange, fmtFrameLabel, fmtSeconds, fmtTimecode, followFrame, frameAtX, minSpan, normalizeView,
  panView, placeView, rulerTicks, spanAtFraction, thumbSlots, viewSpan, xAtFrame, zoomFraction, zoomView,
} from '../src/components/faceswap/timeline/viewMath.js';
import {
  adjacentCut, buildRenderStrip, complement, coveredFrames, fingerprint, mergeIntervals, personColor, presenceHit,
  presenceRows, recordRender, renderedForPieces, sceneAt, scenesFromCuts, visibleIntervals, PERSON_COLORS,
} from '../src/components/faceswap/timeline/laneModel.js';
import {
  BYPASS_FACE_DISTANCE, OVERRIDE_FIELDS, addOverride, canBypass, normalizeOverrides, overrideAt, overrideIsEmpty,
  overridePieces, patchPayload, patchSettings, removeOverride, updateOverride,
} from '../src/components/faceswap/timeline/overrides.js';
import {
  IDLE, SHUTTLE_SPEEDS, reverseTick, shuttleKey, shuttleLabel, shuttleRate, stepSize,
} from '../src/components/faceswap/timeline/shuttle.js';
import { cutNear, overrideCandidates, overrideLaneHit, presenceRowAtY, sceneLaneHit } from '../src/components/faceswap/timeline/hit.js';
import { MAX_PRESENCE_ROWS, SCENE_CUT_H, SCENE_ROW_H, sceneLaneHeight } from '../src/components/faceswap/timeline/lanes.js';
import { FACESWAP_DEFAULTS } from '../src/components/faceswap/defaults.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// Small deterministic RNG so a failure reproduces.
let seed = 0x2545f491;
const rnd = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return ((seed >>> 0) % 1_000_000) / 1_000_000; };
const rint = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));

// ── view window ────────────────────────────────────────────────────────────
console.log('view window');
ok('a degenerate view is the whole clip', eq(normalizeView({ start: 5, end: 5 }, 100), { start: 1, end: 100 }) && eq(normalizeView(null, 100), { start: 1, end: 100 }));
ok('a view past the end is clamped', eq(normalizeView({ start: 90, end: 500 }, 100), { start: 90, end: 100 }));
ok('the zoom floor is the frame-by-frame view: a frame is as wide as one still',
  FRAME_VIEW_PX * minSpan(1100, 100000) <= 1100 && minSpan(1100, 100000) === Math.floor(1100 / FRAME_VIEW_PX));
ok('and a still is always wider than the narrowest label cell', FRAME_VIEW_PX > FRAME_CELL_MIN_PX);
ok('the floor never exceeds the clip', minSpan(1100, 10) <= 9 && minSpan(1100, 3) === 2 && minSpan(0, 1000) >= 4);

{
  let worstDrift = 0;
  let outOfBounds = 0;
  let notShrinking = 0;
  for (let n = 0; n < 2000; n++) {
    const max = rint(30, 120000);
    const w = rint(300, 2400);
    const span = rint(minSpan(w, max), max - 1);
    const start = rint(1, max - span);
    const view = { start, end: start + span };
    const anchor = rint(view.start, view.end);
    const ratio = (anchor - view.start) / span;
    const factor = rnd() < 0.5 ? 0.5 + rnd() * 0.45 : 1.05 + rnd() * 1.5;
    const next = zoomView(view, max, w, factor, anchor);
    const nspan = viewSpan(next);
    if (next.start < 1 || next.end > max || nspan < minSpan(w, max) || nspan > max - 1) outOfBounds++;
    // The anchored frame must stay under the same pixel (to the frame): that is the whole point of
    // anchoring a zoom at the cursor.
    const after = (anchor - next.start) / nspan;
    const atEdge = next.start === 1 || next.end === max;                     // clamped by the clip's edge
    if (!atEdge) worstDrift = Math.max(worstDrift, Math.abs(after - ratio) * nspan);
    if (factor < 1 && span > minSpan(w, max) && nspan >= span) notShrinking++;
  }
  ok('zoom keeps the anchor frame under the cursor (2000 random zooms)', worstDrift <= 1.01, `worst drift ${worstDrift.toFixed(2)} frames`);
  ok('zoom never leaves the clip or the floor/ceiling', outOfBounds === 0, `${outOfBounds} violations`);
  ok('zooming in always moves while above the floor', notShrinking === 0, `${notShrinking} stuck`);
}
{
  const v = zoomView({ start: 1, end: 1000 }, 1000, 1100, 0.0001, 500);
  ok('zooming in hard stops exactly at the floor', viewSpan(v) === minSpan(1100, 1000), `span ${viewSpan(v)}`);
  const out = zoomView(v, 1000, 1100, 1000, 500);
  ok('zooming out hard returns the whole clip', eq(out, { start: 1, end: 1000 }), JSON.stringify(out));
}
ok('pan is clamped to the clip', eq(panView({ start: 10, end: 60 }, 100, -999), { start: 1, end: 51 }) && eq(panView({ start: 10, end: 60 }, 100, 999), { start: 50, end: 100 }));
ok('placeView keeps the width', viewSpan(placeView(95, 20, 100)) === 20);
ok('fitRange never goes below the floor', viewSpan(fitRange(500, 502, 100000, 1100)) === minSpan(1100, 100000));
ok('fitRange shows the whole range', (() => { const v = fitRange(1000, 5000, 100000, 1100); return v.start <= 1000 && v.end >= 5000; })());
{
  const view = { start: 100, end: 200 };
  ok('follow leaves an on-screen playhead alone (same object)', followFrame(view, 1000, 150) === normalizeView(view, 1000) || eq(followFrame(view, 1000, 150), view));
  const f = followFrame(view, 1000, 300);
  ok('follow scrolls to a playhead that ran off, keeping the width', f.start <= 300 && f.end >= 300 && viewSpan(f) === 100, JSON.stringify(f));
}
{
  let bad = 0;
  for (let n = 0; n < 500; n++) {
    const max = rint(50, 100000);
    const w = rint(300, 2000);
    const t = rnd();
    const span = spanAtFraction(t, max, w);
    const back = zoomFraction({ start: 1, end: 1 + span }, max, w);
    if (Math.abs(back - t) > 0.02 && span > minSpan(w, max) + 2 && span < max - 3) bad++;
  }
  ok('the log zoom slider round-trips', bad === 0, `${bad} off`);
}
ok('x <-> frame mapping inverts', (() => {
  const v = { start: 1000, end: 2000 };
  for (const f of [1000, 1234, 2000]) if (frameAtX(xAtFrame(f, 800, v), 800, v, 5000) !== f) return false;
  return true;
})());

// ── time formatting ────────────────────────────────────────────────────────
console.log('time formatting');
ok('frame 1 is timecode zero', fmtTimecode(1, 25) === '00:00:00:00');
ok('frame 26 at 25 fps is one second', fmtTimecode(26, 25) === '00:00:01:00');
ok('the frame field counts to fps-1', fmtTimecode(25, 25) === '00:00:00:24' && fmtTimecode(26, 25) === '00:00:01:00');
ok('29.97 fps counts at the nominal 30', fmtTimecode(31, 29.97) === '00:00:01:00');
ok('an hour rolls over', fmtTimecode(3600 * 25 + 1, 25) === '01:00:00:00');
ok('frame labels pad to five digits', fmtFrameLabel(412, 50000) === '#00412');
ok('frame labels grow with the clip', fmtFrameLabel(412, 1_234_567) === '#0000412');
ok('seconds read m:ss then h:mm:ss', fmtSeconds(1, 25) === '0:00' && fmtSeconds(25 * 75 + 1, 25) === '1:15' && fmtSeconds(25 * 3725 + 1, 25) === '1:02:05');

// ── adaptive ruler ─────────────────────────────────────────────────────────
console.log('adaptive ruler');
{
  const full = rulerTicks({ start: 1, end: 100000 }, 100000, 25, 1100);
  ok('a 100k-frame clip fully out labels in seconds', full.mode === 'seconds', full.mode);
  ok('and walks a bounded number of ticks', full.major.length > 0 && full.major.length <= 40 && full.minor.length <= 160, `${full.major.length}/${full.minor.length}`);
  const mid = rulerTicks({ start: 5000, end: 5100 }, 100000, 25, 1100);
  ok('a hundred frames across (labels under a second apart) reads HH:MM:SS:FF', mid.mode === 'timecode', mid.mode);
  const secs = rulerTicks({ start: 5000, end: 5400 }, 100000, 25, 1100);
  ok('four hundred frames across (labels over a second apart) reads in seconds', secs.mode === 'seconds', secs.mode);
  const tight = rulerTicks({ start: 5000, end: 5030 }, 100000, 25, 1100);
  ok('~36 px per frame reads in exact frame numbers', tight.mode === 'frames', tight.mode);
  ok('frame-mode labels are spaced for their text', (() => {
    const ppf = 1100 / 30;
    return tight.step * ppf >= FRAME_CELL_MIN_PX;
  })(), `step ${tight.step}`);
  for (const [name, view] of [['full', { start: 1, end: 100000 }], ['mid', { start: 5000, end: 5400 }], ['tight', { start: 5000, end: 5030 }]]) {
    const t = rulerTicks(view, 100000, 25, 1100);
    ok(`${name}: every major tick is inside the view and on the step grid`,
      t.major.every((f) => f >= view.start && f <= view.end && (f - 1) % t.step === 0));
  }
  const a = rulerTicks({ start: 1000, end: 2000 }, 100000, 25, 1100);
  const b = rulerTicks({ start: 1040, end: 2040 }, 100000, 25, 1100);
  ok('panning does not move a tick off its frame (labels do not crawl)',
    a.major.filter((f) => f >= 1040 && f <= 2000).every((f) => b.major.includes(f)));
  const small = rulerTicks({ start: 1, end: 100 }, 100, 25, 1100);
  ok('a 100-frame clip fully out (11 px a frame) reads in timecode', small.mode === 'timecode', small.mode);
  const floor = zoomView({ start: 1, end: 100 }, 100, 1100, 0.0001, 50);
  ok('zoomed to the floor it reads in exact frame numbers', rulerTicks(floor, 100, 25, 1100).mode === 'frames');
  ok('no ticks without a width', rulerTicks({ start: 1, end: 100 }, 100, 25, 0).major.length === 0);
}

// ── thumbnail slots ────────────────────────────────────────────────────────
console.log('thumbnail slots');
{
  const out = thumbSlots({ start: 1, end: 1_000_000 }, 1_000_000, 1100, 139);
  ok('a million-frame clip asks for ~8 stills, not a million', out.length <= Math.ceil(1100 / 139) + 1, `${out.length}`);
  ok('slots tile the lane left to right', out.every((s, i) => i === 0 || s.x >= out[i - 1].x) && Math.abs(out.at(-1).x + out.at(-1).w - 1100) < 1);
  const floorView = zoomView({ start: 1, end: 5000 }, 5000, 1100, 0.00001, 100);
  const fine = thumbSlots(floorView, 5000, 1100, 139);
  ok('at the zoom floor the strip is frame-by-frame: one still per frame, in order',
    fine.length === viewSpan(floorView) + 1 && fine.every((s, i) => s.frame === floorView.start + i), `${fine.length} slots for ${JSON.stringify(floorView)}`);
  ok('and each still sits exactly on its own frame', fine.every((s) => Math.abs((s.x + s.w / 2) - xAtFrame(s.frame, 1100, floorView)) < 1e-6));
  ok('and is as wide as a still', fine.every((s) => s.w >= 139));
  ok('a 100-frame clip stays bounded too', thumbSlots({ start: 1, end: 100 }, 100, 1100, 139).length <= 9);
  ok('no width, no slots', thumbSlots({ start: 1, end: 100 }, 100, 0, 139).length === 0);
}

// ── intervals ──────────────────────────────────────────────────────────────
console.log('intervals');
{
  const bitmap = (list, n) => { const b = new Uint8Array(n + 2); for (const [s, e] of list) for (let f = s; f <= e; f++) if (f >= 1 && f <= n) b[f] = 1; return b; };
  let bad = 0;
  for (let t = 0; t < 400; t++) {
    const n = rint(5, 120);
    const list = Array.from({ length: rint(0, 8) }, () => { const a = rint(1, n); return [a, Math.min(n, a + rint(0, 15))]; });
    const merged = mergeIntervals(list, 1, n);
    const comp = complement(list, 1, n);
    const want = bitmap(list, n);
    const gm = bitmap(merged, n);
    const gc = bitmap(comp, n);
    for (let f = 1; f <= n; f++) if (gm[f] !== want[f] || gc[f] === want[f]) bad++;
    for (let i = 1; i < merged.length; i++) if (merged[i][0] <= merged[i - 1][1] + 1) bad++;
    if (coveredFrames(list) !== want.reduce((a, b) => a + b, 0)) bad++;
  }
  ok('merge and complement agree with a per-frame bitmap (400 random sets)', bad === 0, `${bad} mismatches`);
  ok('touching intervals merge', eq(mergeIntervals([[1, 5], [6, 9]]), [[1, 9]]));
  ok('reversed ends are repaired', eq(mergeIntervals([[9, 5]]), [[5, 9]]));
  const sorted = [[1, 10], [20, 30], [40, 50], [60, 70]];
  ok('visibleIntervals clips and bisects', eq(visibleIntervals(sorted, 25, 45), [[25, 30], [40, 45]]) && eq(visibleIntervals(sorted, 11, 19), []));
}

// ── render / cache strip ───────────────────────────────────────────────────
console.log('render / cache strip');
{
  const rank = { reuse: 0, bypass: 1, rendered: 2, dirty: 3 };
  let bad = 0;
  for (let t = 0; t < 300; t++) {
    const hi = rint(1, 200);
    const mk = () => Array.from({ length: rint(0, 6) }, () => { const a = rint(1, hi); return [a, Math.min(hi, a + rint(0, 40))]; });
    const layers = { reuse: mk(), bypass: mk(), rendered: mk() };
    const strip = buildRenderStrip({ lo: 1, hi, ...layers });
    const covers = (list, f) => list.some(([s, e]) => f >= s && f <= e);
    // tiles [1, hi] exactly
    let cursor = 1;
    for (const s of strip) { if (s.start !== cursor || s.end < s.start) bad++; cursor = s.end + 1; }
    if (cursor !== hi + 1) bad++;
    for (let i = 1; i < strip.length; i++) if (strip[i].state === strip[i - 1].state) bad++;   // adjacent runs merged
    for (let f = 1; f <= hi; f++) {
      const want = covers(layers.reuse, f) ? 'reuse' : covers(layers.bypass, f) ? 'bypass' : covers(layers.rendered, f) ? 'rendered' : 'dirty';
      const got = strip.find((s) => f >= s.start && f <= s.end)?.state;
      if (got !== want) bad++;
    }
  }
  ok('the strip tiles the clip and paints the top layer per frame (300 random)', bad === 0, `${bad} mismatches`);
  ok('priority is reuse > bypass > rendered > dirty', rank.reuse < rank.bypass && rank.bypass < rank.rendered && rank.rendered < rank.dirty);
  ok('an empty strip is one dirty span', eq(buildRenderStrip({ lo: 1, hi: 50 }), [{ start: 1, end: 50, state: 'dirty' }]));
  ok('a zero-length clip draws nothing', buildRenderStrip({ lo: 1, hi: 0 }).length === 0);
  const t0 = performance.now();
  const big = buildRenderStrip({
    lo: 1, hi: 100000,
    rendered: Array.from({ length: 4000 }, (_, i) => [i * 25 + 1, i * 25 + 12]),
    bypass: Array.from({ length: 2000 }, (_, i) => [i * 50 + 5, i * 50 + 30]),
    reuse: Array.from({ length: 1000 }, (_, i) => [i * 100 + 40, i * 100 + 60]),
  });
  const dt = performance.now() - t0;
  ok('7,000 intervals over 100,000 frames build in well under a frame budget', dt < 120 && big.length > 1000, `${dt.toFixed(1)} ms, ${big.length} spans`);
}

// ── fingerprint / rendered-for-pieces ──────────────────────────────────────
console.log('fingerprints');
{
  const a = { x: 1, y: { b: 2, a: 1 }, request_id: 'r1', processing_selection: { request_id: 'q1' } };
  const b = { processing_selection: { request_id: 'q2' }, y: { a: 1, b: 2 }, x: 1, request_id: 'r2' };
  ok('key order and per-request ids do not change the fingerprint', fingerprint(a) === fingerprint(b));
  ok('a changed setting does', fingerprint({ ...a, x: 2 }) !== fingerprint(a));
  ok('a nested change does', fingerprint({ ...a, y: { b: 3, a: 1 } }) !== fingerprint(a));
  ok('frame_start / frame_end (the range) are not part of the settings', fingerprint({ ...a, frame_start: 1, frame_end: 50 }) === fingerprint(a));
  const pieces = [{ start: 1, end: 100, fp: 'base' }, { start: 101, end: 150, fp: 'ovr' }, { start: 151, end: 300, fp: 'base' }];
  const log = [{ start: 1, end: 300, fp: 'base' }, { start: 120, end: 140, fp: 'ovr' }];
  ok('a base render does NOT count under an override, and an override render counts only inside it',
    eq(renderedForPieces(log, pieces), [[1, 100], [120, 140], [151, 300]]), JSON.stringify(renderedForPieces(log, pieces)));
  ok('a render of old settings counts for nothing', renderedForPieces([{ start: 1, end: 300, fp: 'old' }], pieces).length === 0);
  ok('recording drops entries a new render fully supersedes',
    recordRender([{ start: 10, end: 20, fp: 'a' }, { start: 5, end: 8, fp: 'b' }], { start: 1, end: 30, fp: 'a' }).length === 2);
}

// ── scenes ─────────────────────────────────────────────────────────────────
console.log('scenes');
{
  const scenes = scenesFromCuts([30, 70, 70, 5000, 1, 0], 100);
  ok('cuts are the first frame of each new shot; duplicates and out-of-range cuts are dropped',
    eq(scenes, [{ index: 0, start: 1, end: 29 }, { index: 1, start: 30, end: 69 }, { index: 2, start: 70, end: 100 }]), JSON.stringify(scenes));
  ok('no cuts = one scene', eq(scenesFromCuts([], 50), [{ index: 0, start: 1, end: 50 }]));
  ok('sceneAt finds the shot and both of its edges', sceneAt(scenes, 29)?.index === 0 && sceneAt(scenes, 30)?.index === 1 && sceneAt(scenes, 100)?.index === 2 && sceneAt(scenes, 101) === null);
  ok('prev / next cut are strict and unsorted-safe', adjacentCut([70, 30], 30, 1) === 70 && adjacentCut([70, 30], 30, -1) === null && adjacentCut([70, 30], 71, -1) === 70 && adjacentCut([], 5, 1) === null);
}

// ── presence ───────────────────────────────────────────────────────────────
console.log('presence');
{
  const rows = presenceRows([{ person_id: 'tp_0', name: 'Actor 1', intervals: [[10, 20], [21, 30], [50, 60]] }, { person_id: 'tp_1', intervals: [] }], 100);
  ok('rows keep the server order and merge touching stretches', rows[0].intervals.length === 2 && rows[0].id === 'tp_0' && rows[1].name === 'Person 2');
  ok('each person gets their own colour, wrapping past the palette', rows[0].color !== rows[1].color && personColor(PERSON_COLORS.length) === personColor(0));
  ok('hit-testing a pill', eq(presenceHit(rows[0], 55), [50, 60]) && presenceHit(rows[0], 40) === null && presenceHit(null, 1) === null);
}

// ── overrides ──────────────────────────────────────────────────────────────
console.log('scene overrides');
{
  ok('every field an override writes exists in the factory settings (so "inherit" has a value to show)',
    OVERRIDE_FIELDS.every((f) => f.key in FACESWAP_DEFAULTS), OVERRIDE_FIELDS.filter((f) => !(f.key in FACESWAP_DEFAULTS)).map((f) => f.key).join(','));
  ok('bypass is only offered where a match threshold exists', canBypass('Selected face') && canBypass('selected') && !canBypass('All faces') && !canBypass(''));

  let bad = 0;
  for (let t = 0; t < 400; t++) {
    const max = rint(20, 400);
    const lo = rint(1, max);
    const hi = rint(lo, max);
    let list = [];
    for (let i = 0; i < rint(0, 6); i++) {
      const a = rint(1, max);
      list = addOverride(list, { start: a, end: Math.min(max, a + rint(0, 60)) }, max).list;
      list = updateOverride(list, list.at(-1).id, { bypass: rnd() < 0.3, params: rnd() < 0.7 ? { mask_erosion: rint(-10, 10) } : {} }, max);
    }
    const pieces = overridePieces(list, lo, hi);
    let cursor = lo;
    for (const p of pieces) { if (p.start !== cursor || p.end < p.start) bad++; cursor = p.end + 1; }
    if (cursor !== hi + 1) bad++;
    for (let i = 1; i < pieces.length; i++) if ((pieces[i].override?.id ?? null) === (pieces[i - 1].override?.id ?? null)) bad++;
    for (const p of pieces) {
      for (const f of [p.start, p.end, (p.start + p.end) >> 1]) {
        const live = list.filter((o) => !overrideIsEmpty(o));
        const want = overrideAt(live, f)?.id ?? null;
        if ((p.override?.id ?? null) !== want) bad++;
      }
    }
  }
  ok('pieces tile the work range exactly, never overlap, and each is governed by the topmost override (400 random)', bad === 0, `${bad} violations`);

  const one = addOverride([], { start: 40, end: 60 }, 100).list;
  ok('an override that changes nothing is not a piece', eq(overridePieces(one, 1, 100).map((p) => [p.start, p.end, p.override]), [[1, 100, null]]));
  const set = updateOverride(one, one[0].id, { params: { mask_feather: 33 } }, 100);
  ok('a real override cuts the range into before / during / after',
    eq(overridePieces(set, 1, 100).map((p) => [p.start, p.end, !!p.override]), [[1, 39, false], [40, 60, true], [61, 100, false]]));
  ok('an override outside the work range is ignored', overridePieces(set, 70, 100).length === 1 && overridePieces(set, 70, 100)[0].override === null);
  ok('an override overlapping an edge is clipped to the range', eq(overridePieces(set, 50, 100).map((p) => [p.start, p.end, !!p.override]), [[50, 60, true], [61, 100, false]]));
  const top = addOverride(set, { start: 50, end: 70 }, 100);
  const layered = updateOverride(top.list, top.created.id, { params: { mask_feather: 5 } }, 100);
  ok('where two overlap the later one wins', overrideAt(layered, 55).params.mask_feather === 5 && overrideAt(layered, 45).params.mask_feather === 33);

  const base = { face_distance: 0.75, mask_erode_dilate_radius: 0, face_mask_blend: 20, restore_ultra_detail_weight: 0.75, swap_model: 'realswap' };
  const patched = patchPayload(base, { params: { mask_erosion: -4, mask_feather: 50, restore_blend: 0.3, swapper: 'hyperswap' }, bypass: false });
  ok('each field lands on the key the swap route reads',
    patched.mask_erode_dilate_radius === -4 && patched.face_mask_blend === 50 && patched.restore_ultra_detail_weight === 0.3 && patched.swap_model === 'hyperswap' && patched.face_distance === 0.75);
  ok('patching never mutates the base', base.face_mask_blend === 20 && base.swap_model === 'realswap');
  ok('an unset field inherits', patchPayload(base, { params: { mask_feather: 9 }, bypass: false }).swap_model === 'realswap');
  ok('bypass closes the match threshold', patchPayload(base, { params: {}, bypass: true }).face_distance === BYPASS_FACE_DISTANCE && BYPASS_FACE_DISTANCE === 0);
  ok('no override is the same object', patchPayload(base, null) === base && patchSettings(base, null) === base);
  ok('the preview reads the settings key for the threshold', patchSettings({ max_face_distance: 0.75 }, { params: {}, bypass: true }).max_face_distance === 0);
  const norm = normalizeOverrides([{ start: 90, end: 5, params: { mask_erosion: 999, bogus: 1, swapper: '' } }], 50);
  ok('normalising clamps, orders the ends and drops unknown / empty params',
    norm[0].start === 5 && norm[0].end === 50 && norm[0].params.mask_erosion === 32 && !('bogus' in norm[0].params) && !('swapper' in norm[0].params));
  ok('removal is by id', removeOverride(one, one[0].id).length === 0);
}

// ── shuttle ────────────────────────────────────────────────────────────────
console.log('J / K / L shuttle');
{
  let s = IDLE;
  s = shuttleKey(s, 'l'); ok('L plays forward at 1x', s.dir === 1 && shuttleRate(s) === 1);
  s = shuttleKey(s, 'L'); ok('L again doubles', shuttleRate(s) === 2);
  s = shuttleKey(s, 'l'); ok('and again quadruples', shuttleRate(s) === 4);
  s = shuttleKey(s, 'l'); ok('and stops at 4x', shuttleRate(s) === 4 && SHUTTLE_SPEEDS.at(-1) === 4);
  s = shuttleKey(s, 'j'); ok('J while going forward reverses at 1x (it does not slow down)', s.dir === -1 && shuttleRate(s) === 1);
  s = shuttleKey(s, 'j'); ok('J again doubles the reverse', s.dir === -1 && shuttleRate(s) === 2);
  ok('the read-out names direction and speed', shuttleLabel(s) === '◀ 2×' && shuttleLabel(IDLE) === '');
  s = shuttleKey(s, 'k'); ok('K pauses and resets the ladder', s.dir === 0 && shuttleRate(s) === 0);
  ok('an unrelated key changes nothing', shuttleKey(shuttleKey(IDLE, 'l'), 'x').dir === 1);
  s = shuttleKey(IDLE, 'l'); s = shuttleKey(s, 'k'); s = shuttleKey(s, 'l');
  ok('after K the ladder starts again at 1x', shuttleRate(s) === 1);

  // One second of 60 Hz ticks at 25 fps should move 25 frames at 1x, 100 at 4x - fractions carried.
  const run = (rate) => {
    let st = { frame: 5000, carry: 0 };
    for (let i = 0; i < 60; i++) st = reverseTick(st, 1000 / 60, rate, 25);
    return 5000 - st.frame;
  };
  ok('reverse at 1x moves 25 frames in a second (no rounding loss)', Math.abs(run(1) - 25) <= 1, `${run(1)}`);
  ok('reverse at 4x moves 100', Math.abs(run(4) - 100) <= 1, `${run(4)}`);
  const end = reverseTick({ frame: 3, carry: 0 }, 1000, 1, 25);
  ok('reverse stops at frame 1 and says so', end.frame === 1 && end.atStart);
  ok('with a loop floor it stops at the In point', reverseTick({ frame: 120, carry: 0 }, 1000, 1, 25, 100).frame === 100);
  ok('a stalled clock (dt 0 / negative) never moves backwards in time', reverseTick({ frame: 50, carry: 0.2 }, -5, 1, 25).frame === 50);
  ok('step keys move 1, Shift moves 10', stepSize(false) === 1 && stepSize(true) === 10);
}

// ── hit testing ────────────────────────────────────────────────────────────
console.log('lane hit-testing');
{
  ok('the nearest cut within tolerance wins', cutNear([100, 105], 103, 3) === 105 && cutNear([100, 105], 120, 3) === null);
  const rows = presenceRows([{ person_id: 'a', name: 'A', intervals: [[10, 40]] }, { person_id: 'b', name: 'B', intervals: [[30, 80]] }], 100);
  const scenes = scenesFromCuts([50], 100);
  ok('the cut strip hits a cut, a pill row hits its pill, anything else is the scene',
    sceneLaneHit({ frame: 50, y: 4, cuts: [50], rows, scenes, tolFrames: 1 })?.kind === 'cut'
    && sceneLaneHit({ frame: 35, y: SCENE_CUT_H + SCENE_ROW_H * 1.5, cuts: [50], rows, scenes, tolFrames: 1 })?.row.id === 'b'
    && sceneLaneHit({ frame: 90, y: SCENE_CUT_H + 2, cuts: [50], rows, scenes, tolFrames: 1 })?.kind === 'scene');
  ok('row index from y', presenceRowAtY(SCENE_CUT_H + 1, 2) === 0 && presenceRowAtY(SCENE_CUT_H + SCENE_ROW_H + 1, 2) === 1 && presenceRowAtY(2, 2) === -1 && presenceRowAtY(SCENE_CUT_H + SCENE_ROW_H * 5, 2) === -1);
  ok('the lane grows with the people up to a cap', sceneLaneHeight(1) < sceneLaneHeight(3) && sceneLaneHeight(50) === sceneLaneHeight(MAX_PRESENCE_ROWS));
  const ovs = normalizeOverrides([{ id: 'o', start: 20, end: 60, params: { mask_feather: 1 } }], 100);
  ok('override edges resize, the body moves', overrideLaneHit(ovs, 21, 2).part === 'left' && overrideLaneHit(ovs, 59, 2).part === 'right' && overrideLaneHit(ovs, 40, 2).part === 'body' && overrideLaneHit(ovs, 90, 2) === null);
  const cands = overrideCandidates({ frame: 40, scenes: scenesFromCuts([30, 70], 100), startFrame: 35, endFrame: 80, maxFrames: 100 });
  ok('right-click offers the scene and the In/Out range', eq(cands.map((c) => [c.kind, c.start, c.end]), [['scene', 30, 69], ['inout', 35, 80]]), JSON.stringify(cands));
  ok('a whole-clip In/Out is not offered', overrideCandidates({ frame: 40, scenes: scenesFromCuts([], 100), startFrame: 1, endFrame: 100, maxFrames: 100 }).every((c) => c.kind === 'scene'));
  ok('outside the In/Out only the scene is offered', overrideCandidates({ frame: 10, scenes: scenesFromCuts([30], 100), startFrame: 35, endFrame: 80, maxFrames: 100 }).length === 1);
}

// ── scale ──────────────────────────────────────────────────────────────────
console.log('scale: 100 frames and 1,000,000 frames cost the same');
{
  const time = (max) => {
    const t0 = performance.now();
    for (let i = 0; i < 200; i++) {
      const v = zoomView({ start: 1, end: max }, max, 1100, 0.9, rint(1, max));
      rulerTicks(v, max, 25, 1100);
      thumbSlots(v, max, 1100, 139);
    }
    return (performance.now() - t0) / 200;
  };
  const small = time(100);
  const huge = time(1_000_000);
  ok('a zoom + ruler + thumbnail pass is sub-millisecond at both sizes', small < 1 && huge < 1, `100 f: ${small.toFixed(3)} ms, 1M f: ${huge.toFixed(3)} ms`);
}

console.log(failures === 0 ? '\nall timeline logic checks passed' : `\n${failures} timeline logic check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
