/**
 * Stage 25's pure logic (plain node, no browser): the parameter-curve model, how a curve reaches a render as pieces,
 * and the face context menu's geometry and skin-tone maths.
 *
 *   node .render-check/run-module.mjs render-stage25-logic.mjs
 *
 * Most groups are PROPERTY checks against a brute-force oracle (a per-frame array), because the failures these
 * functions can have are a curve that overshoots between two keys, a piece boundary that renders a frame twice, and a
 * menu that hangs off the screen - none of which a hand-picked example finds.
 */
import {
  CURVE_PARAMS, DIRECT_PARAMS, INTERPS, MAX_KEYS, addKey, applyCurveValues, clampValue, curveIneffective, evaluate,
  keyFrameBounds, moveKey, morphToIdentityKeyframes, normalizeTrack, normalizeTracks, paramById, patchSettingsWithCurves,
  removeKey, segmentBezier, setInterp, setValue, tangents, valuesAt,
} from '../src/components/faceswap/curves/curveModel.js';
import { MAX_RENDER_PIECES, bakeCurvePieces, describeBake, piecesHaveCurves } from '../src/components/faceswap/curves/curvePieces.js';
import { describeCurvesChange } from '../src/components/faceswap/curves/curveHistory.js';
import {
  MENU_MARGIN, MIN_SKIN_SAMPLES, clampMenuPosition, cropRectFor, faceBadgeText, faceRois, isSkinPixel, nextWarmth, rgbToLab8,
  skinStats, warmthDeltaFor,
} from '../src/components/faceswap/faceContextModel.js';
import { focusPanFor } from '../src/components/faceswap/zoomPan.js';
import { ease, normalizeKeyframes } from '../src/components/faceswap/identityKeyframes.js';
import { overridePieces } from '../src/components/faceswap/timeline/overrides.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

// Small deterministic RNG so a failure reproduces.
let seed = 25;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const int = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));

const OPACITY = paramById('swap_opacity');
const FEATHER = paramById('mask_feather');
const key = (frame, value, interp = 'linear') => ({ frame, value, interp });

function randomTrack(param, maxFrames, n, interp) {
  const frames = new Set();
  while (frames.size < n) frames.add(int(1, maxFrames));
  return normalizeTrack([...frames].map((f) => ({
    frame: f, value: param.min + rnd() * (param.max - param.min), interp: interp || INTERPS[int(0, INTERPS.length - 1)],
  })), param, maxFrames);
}

console.log('curve model');
{
  ok('the parameter list is the five the brief names', CURVE_PARAMS.map((p) => p.id).join() === 'swap_opacity,restore_blend,mask_feather,relight,identity_morph');
  ok('every direct curve names a settings key, the morph does not',
    DIRECT_PARAMS.every((p) => typeof p.key === 'string') && paramById('identity_morph').key === null);

  const messy = normalizeTrack([
    key(50, 0.5), key(10, 2, 'bogus'), key(10, 0.3), key(-4, 0.1), key(999, 0.9), { frame: 'x', value: 1 }, null, key(20, NaN),
  ], OPACITY, 100);
  ok('normalizeTrack clamps frames and values, drops junk, keeps one key per frame, sorts',
    messy.map((k) => k.frame).join() === '1,10,50,100' && messy.every((k) => k.value >= 0 && k.value <= 1), JSON.stringify(messy));
  ok('later duplicates win and an unknown interp becomes the default', messy.find((k) => k.frame === 10).value === 0.3
    && messy.every((k) => INTERPS.includes(k.interp)));
  ok('a track is capped', normalizeTrack(Array.from({ length: 500 }, (_, i) => key(i + 1, 0.5)), OPACITY, 1000).length === MAX_KEYS);
  ok('normalizeTracks drops unknown parameters and empty tracks',
    JSON.stringify(Object.keys(normalizeTracks({ swap_opacity: [key(3, 0.5)], nope: [key(1, 1)], mask_feather: [] }, 10))) === '["swap_opacity"]');
  ok('clampValue rounds to the parameter precision', clampValue(FEATHER, 20.6) === 21 && clampValue(OPACITY, 0.12345) === 0.12 && clampValue(OPACITY, 7) === 1);

  // evaluation
  ok('an empty track has no value', evaluate([], 5) === null && evaluate(undefined, 5) === null);
  const lin = [key(10, 0), key(20, 1, 'linear')];
  ok('the value is held before the first key and after the last', evaluate(lin, 1) === 0 && evaluate(lin, 99) === 1);
  ok('linear is the diagonal', near(evaluate(lin, 15), 0.5) && near(evaluate(lin, 12), 0.2));
  const hold = [key(10, 0.2, 'hold'), key(20, 0.9)];
  ok('hold is a step: the first value until the next key, then the second',
    evaluate(hold, 19.999) === 0.2 && evaluate(hold, 20) === 0.9 && evaluate(hold, 15) === 0.2);
  const easeT = [key(0 + 1, 0, 'ease'), key(101, 1)];
  ok('ease-in / ease-out is symmetric and slower than linear at the ends',
    near(evaluate(easeT, 51), 0.5, 1e-6) && evaluate(easeT, 26) < 0.25 && evaluate(easeT, 76) > 0.75);
  ok('ease-in / ease-out is monotone', (() => { let prev = -1; for (let f = 1; f <= 101; f++) { const v = evaluate(easeT, f); if (v < prev - 1e-12) return false; prev = v; } return true; })());

  // smooth: the property the brief asks for - no overshoot, no pop
  let overshoot = 0;
  let passesThrough = 0;
  let trials = 0;
  for (let t = 0; t < 200; t++) {
    const track = randomTrack(OPACITY, 400, int(2, 9), 'smooth');
    for (const k of track) { trials += 1; if (near(evaluate(track, k.frame), k.value, 1e-9)) passesThrough += 1; }
    for (let i = 0; i < track.length - 1; i++) {
      const lo = Math.min(track[i].value, track[i + 1].value) - 1e-9;
      const hi = Math.max(track[i].value, track[i + 1].value) + 1e-9;
      for (let f = track[i].frame; f <= track[i + 1].frame; f++) {
        const val = evaluate(track, f);
        if (val < lo || val > hi) overshoot += 1;
      }
    }
  }
  ok('smooth never overshoots the two keys a segment sits between (200 random tracks, every frame)', overshoot === 0, `${overshoot} frames outside`);
  ok('every curve passes exactly through its keys', passesThrough === trials, `${passesThrough}/${trials}`);

  const peak = [key(1, 0, 'smooth'), key(50, 1, 'smooth'), key(100, 0, 'smooth')];
  ok('a local peak is flat: the tangent there is zero and the curve stays at or under it', tangents(peak)[1] === 0
    && Array.from({ length: 100 }, (_, i) => evaluate(peak, i + 1)).every((x) => x <= 1 + 1e-12));
  const ramp = [key(100, 0.2, 'smooth'), key(130, 0.9, 'smooth'), key(200, 0.9, 'smooth')];
  ok('Auto-Smooth leaves a ramp with zero slope at both ends: no one-frame pop',
    Math.abs(evaluate(ramp, 100.2) - 0.2) < 0.01 && Math.abs(evaluate(ramp, 129.8) - 0.9) < 0.01
    && Math.abs(evaluate(ramp, 101) - evaluate(ramp, 100)) < Math.abs(evaluate(ramp, 116) - evaluate(ramp, 115)));
  const smoothPop = [key(10, 0.2, 'smooth'), key(11, 0.9, 'smooth'), key(50, 0.9)];
  ok('the same ramp done as a hard step is a pop (the thing smoothing exists to prevent)',
    Math.abs(evaluate([key(10, 0.2, 'hold'), key(11, 0.9)], 11) - evaluate([key(10, 0.2, 'hold'), key(11, 0.9)], 10.99)) > 0.6
    && evaluate(smoothPop, 10.5) > 0.2 && evaluate(smoothPop, 10.5) < 0.9);
  ok('a flat segment has no easing to speak of', segmentBezier([key(1, 0.5, 'smooth'), key(9, 0.5)], 0) === null);

  // The compile claim: a smooth / ease segment's easing, handed to the backend's identity easing, gives the SAME curve.
  let compileGap = 0;
  for (let t = 0; t < 100; t++) {
    const track = randomTrack(OPACITY, 300, int(2, 7), t % 2 ? 'smooth' : 'ease');
    const idk = morphToIdentityKeyframes(track, [{ source_id: 'a' }, { source_id: 'b' }], 300);
    for (let i = 0; i < track.length - 1; i++) {
      const a = track[i];
      const b = track[i + 1];
      for (let f = a.frame; f <= b.frame; f += 3) {
        const u = (f - a.frame) / (b.frame - a.frame);
        const viaBackend = a.value + (b.value - a.value) * ease(idk[i].curve, u, idk[i].bezier);
        compileGap = Math.max(compileGap, Math.abs(viaBackend - evaluate(track, f)));
      }
    }
  }
  ok('compiled identity keyframes evaluate to the curve (backend easing == editor easing)', compileGap < 1e-6, `max gap ${compileGap}`);

  // edits
  const base = [key(10, 0.2), key(40, 0.8)];
  const added = addKey(base, OPACITY, 25, undefined, 100);
  ok('a key added with no value keeps what the curve already says there (adding changes nothing)',
    added.created && near(added.list.find((k) => k.frame === 25).value, 0.5) && near(evaluate(added.list, 17), evaluate(base, 17)));
  ok('a key is not added twice at one frame', addKey(base, OPACITY, 10, 0.5, 100).created === false);
  ok('a key on an empty track takes the parameter base', addKey([], FEATHER, 5, undefined, 100).list[0].value === FEATHER.base);
  ok('adding past the cap is refused', addKey(Array.from({ length: MAX_KEYS }, (_, i) => key(i + 1, 0.5)), OPACITY, 500, 0.1, 1000).full === true);
  ok('keyFrameBounds is strictly inside the neighbours', JSON.stringify(keyFrameBounds([key(10, 0), key(20, 0), key(30, 0)], 20, 100)) === '[11,29]'
    && JSON.stringify(keyFrameBounds([key(10, 0), key(20, 0)], 10, 100)) === '[1,19]'
    && JSON.stringify(keyFrameBounds([key(10, 0), key(11, 0), key(12, 0)], 11, 100)) === '[11,11]'     // wedged: it can only stay
    && keyFrameBounds([key(10, 0)], 99, 100) === null);
  const three = [key(10, 0.1), key(20, 0.5), key(30, 0.9)];
  ok('a key cannot be dragged past a neighbour', moveKey(three, OPACITY, 20, 99, 0.5, 100).find((k) => k.value === 0.5).frame === 29
    && moveKey(three, OPACITY, 20, -5, 0.5, 100).find((k) => k.value === 0.5).frame === 11);
  ok('a move that changes nothing returns the same list', moveKey(three, OPACITY, 20, 20, 0.5, 100) === three);
  ok('a move clamps the value into range', moveKey(three, OPACITY, 20, 21, 7, 100).find((k) => k.frame === 21).value === 1);
  ok('remove / interp / value edits', removeKey(three, 20).length === 2 && setInterp(three, 20, 'hold')[1].interp === 'hold'
    && setInterp(three, 20, 'nope') === three && setValue(three, OPACITY, 20, 0.77)[1].value === 0.77);

  // applying
  const tracks = { swap_opacity: [key(1, 1), key(11, 0.5)], mask_feather: [key(1, 30)], identity_morph: [key(1, 1)] };
  const settings = { blend_ratio: 1, face_mask_blend: 20, other: 7 };
  const at = patchSettingsWithCurves(settings, tracks, 6);
  ok('patchSettingsWithCurves writes the curve values at the frame and leaves the rest alone',
    near(at.blend_ratio, 0.75) && at.face_mask_blend === 30 && at.other === 7 && settings.blend_ratio === 1);
  ok('the identity morph writes no settings key', !('identity_morph' in at) && Object.keys(at).length === 3);
  ok('no tracks gives back the SAME object (nothing downstream changes identity)', patchSettingsWithCurves(settings, {}, 5) === settings
    && patchSettingsWithCurves(settings, undefined, 5) === settings);
  ok('a curve already equal to the setting gives back the same object', patchSettingsWithCurves(settings, { swap_opacity: [key(1, 1)] }, 5) === settings);
  ok('valuesAt lists every track that has keys', JSON.stringify(Object.keys(valuesAt(tracks, 5)).sort()) === '["identity_morph","mask_feather","swap_opacity"]');
  ok('applyCurveValues writes the piece values over a payload and keeps the original intact',
    (() => { const p = { blend_ratio: 1, x: 1 }; const o = applyCurveValues(p, { swap_opacity: 0.4 }); return o.blend_ratio === 0.4 && o.x === 1 && p.blend_ratio === 1 && applyCurveValues(p, null) === p && applyCurveValues(p, {}) === p; })());
  ok('a relight curve says so when Light Harmonizer is off', /Light Harmonizer is off/.test(curveIneffective(paramById('relight'), { light_harmonizer: false }) || '')
    && curveIneffective(paramById('relight'), { light_harmonizer: true }) === null && curveIneffective(OPACITY, {}) === null);

  // identity morph
  const comps = [{ source_id: 'a' }, { source_id: 'b' }, { source_id: 'c' }];
  const morph = morphToIdentityKeyframes([key(1, 0, 'hold'), key(61, 1, 'smooth'), key(101, 0.25)], comps, 100);
  ok('the morph compiles to keyframes the identity track accepts, unchanged by its own normaliser',
    JSON.stringify(normalizeKeyframes(morph)) === JSON.stringify(morph), JSON.stringify(morph));
  ok('frames become 0-based, the first two sources carry the weight and it sums to 100',
    morph[0].frame_index === 0 && morph[1].frame_index === 60 && morph[2].frame_index === 99 && !('c' in morph[1].weights)
    && morph.every((k) => near(k.weights.a + k.weights.b, 100, 1e-9)) && morph[0].weights.a === 100 && near(morph[2].weights.b, 25));
  ok('interps map onto the backend curves', morph[0].curve === 'hold' && morph[1].curve === 'cubic_bezier' && morph[2].curve === 'linear');
  ok('fewer than two sources compiles to nothing', morphToIdentityKeyframes([key(1, 1)], [comps[0]], 100).length === 0);
}

console.log('how a curve reaches a render');
{
  const MAX = 600;
  const tiles = (pieces, lo, hi) => pieces[0].start === lo && pieces[pieces.length - 1].end === hi
    && pieces.every((pc, i) => pc.end >= pc.start && (i === 0 || pc.start === pieces[i - 1].end + 1));

  const none = bakeCurvePieces([{ start: 1, end: 100, override: null }], {});
  ok('no curves: the pieces come back untouched, with no curve values', none.pieces.length === 1 && none.pieces[0].curves === null && none.widen === 1);

  const flat = bakeCurvePieces([{ start: 1, end: 100, override: null }], { swap_opacity: [key(1, 0.5)] });
  ok('a flat curve is one piece carrying its value', flat.pieces.length === 1 && flat.pieces[0].curves.swap_opacity === 0.5);
  ok('a curve sitting on its slider changes nothing, one that does not changes the render',
    !piecesHaveCurves(flat.pieces, { blend_ratio: 0.5 }) && piecesHaveCurves(flat.pieces, { blend_ratio: 1 }) && piecesHaveCurves(flat.pieces));

  let tilingBad = 0;
  let errBad = 0;
  let capBad = 0;
  let boundaryBad = 0;
  for (let t = 0; t < 120; t++) {
    const tracks = {};
    for (const p of DIRECT_PARAMS.slice(0, 3)) if (rnd() < 0.7) tracks[p.id] = randomTrack(p, MAX, int(1, 8));
    const ovr = rnd() < 0.5 ? [{ id: 'o', start: int(50, 200), end: int(250, 400), params: { mask_erosion: 3 }, bypass: false, order: 0 }] : [];
    const lo = int(1, 80);
    const hi = int(450, MAX);
    const base = overridePieces(ovr, lo, hi);
    const baked = bakeCurvePieces(base, tracks);
    if (!tiles(baked.pieces, lo, hi)) tilingBad += 1;
    if (baked.pieces.length > Math.max(MAX_RENDER_PIECES, base.length)) capBad += 1;
    // every override boundary is still a piece boundary, and the governor is unchanged inside it
    for (const b of base) {
      const inside = baked.pieces.filter((pc) => pc.start >= b.start && pc.end <= b.end);
      if (!inside.length || inside[0].start !== b.start || inside[inside.length - 1].end !== b.end
        || inside.some((pc) => (pc.override?.id ?? null) !== (b.override?.id ?? null))) boundaryBad += 1;
    }
    // every frame renders within one (possibly widened) step of the exact curve
    for (const pc of baked.pieces) {
      for (const p of DIRECT_PARAMS) {
        const list = tracks[p.id];
        if (!list) continue;
        const tol = p.quantum * baked.widen + 10 ** -p.digits;
        for (let f = pc.start; f <= pc.end; f++) if (Math.abs(pc.curves[p.id] - evaluate(list, f)) > tol) errBad += 1;
      }
    }
  }
  ok('pieces tile the range exactly: no gap, no overlap, no frame rendered twice (120 random plans)', tilingBad === 0, `${tilingBad} plans`);
  ok('the piece count stays under the cap', capBad === 0, `${capBad} plans over`);
  ok('scene-override boundaries are kept, and so is the override each piece belongs to', boundaryBad === 0, `${boundaryBad}`);
  ok('every frame renders within one step of the exact curve', errBad === 0, `${errBad} frames outside`);

  const one = bakeCurvePieces([{ start: 1, end: 600, override: null }], { swap_opacity: [key(1, 0, 'linear'), key(600, 1)] });
  ok('one full 0-1 ramp fits under the cap at its natural step (21 levels)', one.widen === 1 && one.pieces.length <= MAX_RENDER_PIECES && one.pieces.length >= 20);
  // Three ramps with different render steps cut at different frames, so together they need ~50 pieces: over the cap.
  const steep = bakeCurvePieces([{ start: 1, end: 600, override: null }], {
    swap_opacity: [key(1, 0, 'linear'), key(600, 1)], mask_feather: [key(1, 0, 'linear'), key(600, 200)],
    relight: [key(1, 0, 'linear'), key(600, 2)],
  });
  ok('curves that need more pieces than the cap are capped, and the widened step is reported rather than hidden',
    steep.pieces.length <= MAX_RENDER_PIECES && steep.widen > 1 && steep.steps.swap_opacity > OPACITY.quantum);
  ok('the report says so', /widened/.test(describeBake(steep)) && /pieces/.test(describeBake(steep)));
  const gentle = bakeCurvePieces([{ start: 1, end: 100, override: null }], { swap_opacity: [key(40, 1, 'linear'), key(60, 0.5)] });
  ok('a short ramp renders at the natural step', gentle.widen === 1 && gentle.pieces.length > 2 && gentle.pieces.length <= MAX_RENDER_PIECES);
  const held = bakeCurvePieces([{ start: 1, end: 100, override: null }], { swap_opacity: [key(30, 1, 'hold'), key(70, 0.3)] }).pieces;
  ok('a hold is two values, so two pieces, cut exactly where the step lands', held.length === 2 && held[0].end === 69 && held[1].start === 70
    && held[0].curves.swap_opacity === 1 && held[1].curves.swap_opacity === 0.3);
  ok('nothing to report without curves', describeBake(none) === '');
}

console.log('undo labels');
{
  const a = { c1: { swap_opacity: [key(1, 0.5)] } };
  ok('added / removed / moved / interp / cleared read like the rest of the history',
    describeCurvesChange({}, a) === 'Added Swap Opacity Keyframe'
    && describeCurvesChange(a, { c1: { swap_opacity: [] } }) === 'Cleared Swap Opacity Curve'
    && describeCurvesChange({ c1: { swap_opacity: [key(1, 0.5), key(9, 1)] } }, a) === 'Removed Swap Opacity Keyframe'
    && describeCurvesChange(a, { c1: { swap_opacity: [key(3, 0.5)] } }) === 'Moved Swap Opacity Keyframe'
    && describeCurvesChange(a, { c1: { swap_opacity: [key(1, 0.5, 'hold')] } }) === 'Changed Swap Opacity Interpolation');
}

console.log('face context menu: placement');
{
  const VW = 1280;
  const VH = 720;
  const inside = (c, w, h) => c.left >= MENU_MARGIN && c.top >= MENU_MARGIN && c.left + w <= VW - MENU_MARGIN && c.top + h <= VH - MENU_MARGIN;
  ok('a menu in open space opens at the pointer', (() => { const c = clampMenuPosition({ x: 300, y: 200, w: 260, h: 240, vw: VW, vh: VH }); return c.left === 300 && c.top === 200 && !c.flipX && !c.flipY; })());
  ok('near the right edge it opens to the left of the pointer', (() => { const c = clampMenuPosition({ x: 1200, y: 100, w: 260, h: 240, vw: VW, vh: VH }); return c.flipX && c.left === 940 && c.left + 260 <= 1200; })());
  ok('near the bottom edge it opens above the pointer', (() => { const c = clampMenuPosition({ x: 100, y: 650, w: 260, h: 240, vw: VW, vh: VH }); return c.flipY && c.top === 410; })());
  ok('in the bottom-right corner it flips both ways', (() => { const c = clampMenuPosition({ x: 1270, y: 715, w: 260, h: 240, vw: VW, vh: VH }); return c.flipX && c.flipY && inside(c, 260, 240); })());

  let escaped = 0;
  for (let t = 0; t < 4000; t++) {
    const w = int(120, 400);
    const h = int(80, 560);
    const c = clampMenuPosition({ x: int(-50, VW + 50), y: int(-50, VH + 50), w, h, vw: VW, vh: VH });
    if (!inside(c, Math.min(w, VW - 2 * MENU_MARGIN), Math.min(h, c.maxHeight ?? h))) escaped += 1;
  }
  ok('wherever the click lands - even off the window - the menu stays inside the margins (4000 random clicks)', escaped === 0, `${escaped} escaped`);
  ok('a menu taller than the window is pinned to the top and told how tall it may be', (() => {
    const c = clampMenuPosition({ x: 50, y: 400, w: 200, h: 900, vw: VW, vh: VH });
    return c.top === MENU_MARGIN && c.maxHeight === VH - 2 * MENU_MARGIN;
  })());
  ok('a menu that fits needs no max-height', clampMenuPosition({ x: 5, y: 5, w: 200, h: 200, vw: VW, vh: VH }).maxHeight === null);
  ok('a window smaller than the menu does not produce a negative position', (() => { const c = clampMenuPosition({ x: 10, y: 10, w: 400, h: 400, vw: 200, vh: 200 }); return c.left >= 0 && c.top >= 0; })());
}

console.log('face context menu: badge and crop');
{
  ok('a swapped face names its track and the identity it wears',
    faceBadgeText({ faceIndex: 0, personId: 0, sourceName: 'rishabh_v3.fsz', role: 'swapped' }) === 'Track #1 → rishabh_v3.fsz');
  ok('a bystander says it is not swapped, an excluded one says never', /not swapped/.test(faceBadgeText({ faceIndex: 1, personId: 1, role: 'bystander' }))
    && /excluded/.test(faceBadgeText({ faceIndex: 1, personId: 1, role: 'bystander', excluded: true })));
  ok('an uncaptured face is named by its box number', faceBadgeText({ faceIndex: 2, personId: null, role: 'bystander' }).startsWith('Face 3'));
  ok('a swapped face with a source that left the gallery says so rather than showing nothing',
    /not in gallery/.test(faceBadgeText({ faceIndex: 0, personId: 0, sourceName: null, role: 'target' })));
  ok('the badge never invents a match score', !/\[\d/.test(faceBadgeText({ faceIndex: 0, personId: 0, sourceName: 'a', role: 'swapped' })));

  const r = cropRectFor([100, 100, 200, 220], { w: 1000, h: 500 }, { w: 2000, h: 1000 }, 0.5);
  // box 200..400 x 200..440 in the 2x frame (200 x 240), +50% each side: x 100..500, y 80..560
  ok('the crop is the box grown by the margin, in the frame’s own pixels', r.x === 100 && r.y === 80 && r.w === 400 && r.h === 480, JSON.stringify(r));
  ok('a face at the frame edge is cropped at the edge, never outside', (() => {
    const e = cropRectFor([-30, -30, 60, 60], { w: 640, h: 360 }, { w: 640, h: 360 }, 0.6);
    return e.x === 0 && e.y === 0 && e.w > 0 && e.h > 0 && e.x + e.w <= 640 && e.y + e.h <= 360;
  })());
  ok('a speck, or no frame, has no crop', cropRectFor([5, 5, 7, 7], { w: 100, h: 100 }, { w: 100, h: 100 }) === null && cropRectFor([0, 0, 50, 50], null, { w: 1, h: 1 }) === null);
}

console.log('face context menu: skin tone');
{
  const px = (w, h, rgb) => ({ width: w, height: h, data: Uint8ClampedArray.from({ length: w * h * 4 }, (_, i) => (i % 4 === 3 ? 255 : rgb[i % 4])) });
  const SKIN = [224, 172, 140];

  ok('OpenCV 8-bit LAB: white is L 255, neutral grey sits on a=b=128', (() => {
    const w = rgbToLab8(255, 255, 255);
    const g = rgbToLab8(128, 128, 128);
    return Math.abs(w[0] - 255) < 1 && Math.abs(w[1] - 128) < 1 && Math.abs(g[1] - 128) < 1 && Math.abs(g[2] - 128) < 1;
  })());
  ok('warm skin has a* and b* above neutral', (() => { const l = rgbToLab8(...SKIN); return l[1] > 134 && l[2] > 134; })());
  // It is the backend's own gate (procmgr_color._skin_region_mask): chroma ranges and not near-black. Like that gate it
  // cannot tell a neutral grey from skin by colour alone - the regions sampled (forehead, cheeks) are what do that.
  ok('skin evidence accepts a skin tone and rejects blue sky, green foliage and black hair',
    isSkinPixel(...SKIN) && !isSkinPixel(80, 140, 230) && !isSkinPixel(40, 160, 60) && !isSkinPixel(10, 8, 6));

  const s = skinStats([px(20, 20, SKIN)]);
  ok('skinStats of a flat patch is that colour', s.n === 400 && Math.abs(s.a - Math.round(rgbToLab8(...SKIN)[1])) <= 1);
  ok('hair, eyes and background in the patch do not move the median', (() => {
    const mixed = { width: 10, height: 10, data: new Uint8ClampedArray(400) };
    for (let i = 0; i < 100; i++) {
      const c = i < 70 ? SKIN : [20, 15, 10];
      mixed.data.set([...c, 255], i * 4);
    }
    const m = skinStats([mixed]);
    return m.n === 70 && Math.abs(m.a - s.a) <= 1 && Math.abs(m.b - s.b) <= 1;
  })());
  ok('no skin at all counts zero samples', skinStats([px(8, 8, [10, 10, 10])]).n === 0 && skinStats([]).n === 0 && skinStats(null).n === 0);

  // The headline property: warmth is solved in the renderer's own units. Shift a face by exactly the amount warmth w
  // does on skin (1.5 w on a*, 4.0 w on b*) and the solver must hand w back.
  const plate = { a: 150, b: 150, L: 140, n: 500 };
  let worst = 0;
  for (const w of [-80, -35, -25, 25, 40, 75]) {        // smaller shifts are below NEGLIGIBLE_SHIFT: "matched", tested below
    const swapped = { a: plate.a - 1.5 * (w / 100), b: plate.b - 4.0 * (w / 100), L: 140, n: 500 };
    worst = Math.max(worst, Math.abs(warmthDeltaFor(plate, swapped).warmthDelta - w));
  }
  ok('the solver returns the warmth that undoes a shift made with the renderer’s gain (to the slider’s precision)', worst < 0.2, `worst error ${worst}`);
  ok('a pure green / magenta error is declined, not "fixed" with a big warmth move that barely reduces it', (() => {
    const m = warmthDeltaFor({ a: 140, b: 150, n: 500 }, { a: 150, b: 150, n: 500 });
    return m.verdict === 'tint' && m.warmthDelta === 0 && m.residual > 8 && m.explained < 0.3;
  })());
  ok('a shift along the warm axis is applied and explained almost entirely', (() => {
    const m = warmthDeltaFor({ a: 150, b: 152, n: 500 }, { a: 148.5, b: 148, n: 500 });
    return m.verdict === 'apply' && m.explained > 0.99 && Math.abs(m.warmthDelta - 100) < 1;
  })());
  ok('a difference nobody can see is "already matched"', warmthDeltaFor({ a: 150, b: 150.4, n: 500 }, { a: 150.3, b: 150, n: 500 }).verdict === 'matched');
  ok('every verdict that is not "apply" writes no change', ['matched', 'tint'].every((v) => {
    const m = v === 'tint' ? warmthDeltaFor({ a: 140, b: 150, n: 500 }, { a: 150, b: 150, n: 500 }) : warmthDeltaFor(plate, plate);
    return m.verdict === v && m.warmthDelta === 0;
  }));
  ok('too few skin pixels on either side measures nothing', warmthDeltaFor({ a: 1, b: 1, n: MIN_SKIN_SAMPLES - 1 }, plate) === null
    && warmthDeltaFor(plate, { a: 1, b: 1, n: 0 }) === null && warmthDeltaFor(null, plate) === null);
  ok('the new warmth is the current one plus the change, kept on the slider', nextWarmth(10, 15) === 25 && nextWarmth(90, 40) === 100
    && nextWarmth(-90, -40) === -100 && nextWarmth(undefined, 5) === 5);
  ok('applying the answer closes the loop: measuring again after the correction says ~0', (() => {
    const swapped = { a: 146, b: 141, L: 130, n: 400 };
    const first = warmthDeltaFor(plate, swapped).warmthDelta / 100;
    const corrected = { ...swapped, a: swapped.a + 1.5 * first, b: swapped.b + 4.0 * first };
    return Math.abs(warmthDeltaFor(plate, corrected).warmthDelta) < Math.abs(first * 100) * 0.4;
  })());

  const upright = faceRois([100, 100, 200, 220], [[130, 140], [170, 140], [150, 165], [133, 190], [167, 190]]);
  ok('the keypoints give a forehead above the eyes and a cheek on each side', upright.length === 3 && upright[0].y + upright[0].h / 2 < 140
    && upright[1].x + upright[1].w / 2 < 150 && upright[2].x + upright[2].w / 2 > 150, JSON.stringify(upright));
  ok('every region lies inside the face box', upright.every((q) => q.x >= 100 && q.y >= 90 && q.x + q.w <= 200 && q.y + q.h <= 220), JSON.stringify(upright));
  // Roll the same five points by 30 degrees about the nose: the forehead must follow the head.
  const rot = (pt, deg, c) => { const t = (deg * Math.PI) / 180; const dx = pt[0] - c[0]; const dy = pt[1] - c[1]; return [c[0] + dx * Math.cos(t) - dy * Math.sin(t), c[1] + dx * Math.sin(t) + dy * Math.cos(t)]; };
  const k0 = [[130, 140], [170, 140], [150, 165], [133, 190], [167, 190]];
  const rolled = faceRois([80, 80, 220, 240], k0.map((q) => rot(q, 30, [150, 165])));
  const fore = rolled[0];
  const eyeMid = rot([150, 140], 30, [150, 165]);
  const up = rot([150, 100], 30, [150, 165]);
  ok('on a rolled head the forehead follows the face’s own up, not the screen’s',
    Math.hypot(fore.x + fore.w / 2 - up[0], fore.y + fore.h / 2 - up[1]) < Math.hypot(fore.x + fore.w / 2 - eyeMid[0], fore.y + fore.h / 2 - (eyeMid[1] - 25)) + 12
    && fore.x + fore.w / 2 > eyeMid[0] + 5, JSON.stringify({ fore, eyeMid, up }));
  const noKps = faceRois([100, 100, 200, 220], null);
  ok('without keypoints the regions are fractions of the box, still inside it', noKps.length === 3
    && noKps.every((q) => q.x >= 100 && q.y >= 100 && q.x + q.w <= 200 && q.y + q.h <= 220));
  ok('a degenerate box has no regions', faceRois([5, 5, 6, 6], null).length === 0 && faceRois(null, null).length === 0);
}

console.log('focus and zoom 200%');
{
  const stage = { offsetWidth: 800, offsetHeight: 450, getBoundingClientRect: () => ({ width: 800, height: 450, left: 0, top: 0 }) };
  const dim = { w: 1600, h: 900 };
  const centre = focusPanFor([700, 400, 900, 500], dim, stage, 2);
  ok('a face dead centre needs no pan', Math.abs(centre.x) < 1e-6 && Math.abs(centre.y) < 1e-6);
  const left = focusPanFor([100, 400, 300, 500], dim, stage, 2);
  ok('a face on the left pans the picture right so it lands in the middle', left.x > 0 && Math.abs(left.y) < 1e-6);
  ok('and the pan never exposes the picture’s edge', Math.abs(left.x) <= (800 * 2 - 800) / 2 + 1e-6);
  ok('no stage or no box, no pan', focusPanFor(null, dim, stage, 2) === null && focusPanFor([1, 1, 2, 2], dim, null, 2) === null && focusPanFor([1, 1, 2, 2], null, stage, 2) === null);
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nstage 25 logic: all checks passed' : `\nstage 25 logic: ${failures} FAILED`);
