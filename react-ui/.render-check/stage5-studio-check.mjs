/**
 * Stage 5 studio controls: scrubber geometry, the effect rack's rules, the face routing dock's rules.
 *
 *   node .render-check/run-module.mjs stage5-studio-check.mjs
 *
 * Pure rules only. What they cannot show - that a drag lands on the bar, that the power switch glows, that a route line meets its
 * source - is checked in a real browser by e2e/stage5-studio.spec.js, and that the BACKEND applies an override per face is
 * app/tests/test_person_overrides.py (the production paste, bit for bit).
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const sm = await import('../src/components/scrubberModel.js');
const rack = await import('../src/components/faceswap/effectRackModel.js');
const rt = await import('../src/components/faceswap/routingModel.js');
const scrub = await import('../src/numeric/scrubMath.js');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const section = (t) => console.log(`\n── ${t} ${'─'.repeat(Math.max(2, 64 - t.length))}`);
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Scrubber: where the fill runs');
{
  const a = sm.fillExtent(0.25, 0, 1, 0.5);
  ok('a unipolar range fills from the left edge to the value', a.from === 0 && near(a.to, 0.25) && near(a.pct, 0.25));
  ok('the default tick sits at its own fraction', near(a.pctDefault, 0.5));
  const neg = sm.fillExtent(-8, -32, 32, 0);
  ok('a range through zero fills OUTWARD FROM ZERO: -8 of -32..32 fills left of the centre line',
    near(neg.to, 0.5) && near(neg.from, 0.375) && neg.bipolar === true, JSON.stringify(neg));
  const pos = sm.fillExtent(8, -32, 32, 0);
  ok('and +8 fills right of it', near(pos.from, 0.5) && near(pos.to, 0.625));
  ok('a value outside the range is clamped, never a fill past the bar', sm.fillExtent(999, 0, 1).to === 1 && sm.fillExtent(-999, 0, 1).to === 0);
  ok('a degenerate range draws nothing instead of NaN', Object.values(sm.fillExtent(1, 5, 5)).every((v) => v === 0 || v === null || v === false));
  ok('no default means no tick', sm.fillExtent(0.5, 0, 1).pctDefault === null);
  ok('decimals follow the step: 0.05 -> 2, 1 -> 0, 0.001 -> 3, 0.5 -> 2', sm.scrubberDecimals(0.05) === 2 && sm.scrubberDecimals(1) === 0
    && sm.scrubberDecimals(0.001) === 3 && sm.scrubberDecimals(0.5) === 2);
}

section('Scrubber: Shift x10 / Alt x0.1 (the pace the bar and the label share)');
{
  const cfg = { min: 0, max: 1, step: 0.01 };
  const run = (mods, dx) => scrub.advanceScrub(scrub.beginScrub(0.5, cfg), dx, mods).value;
  const plain = run({}, 40) - 0.5;
  ok('a plain 40 px drag moves the value', plain > 0.05, String(plain));
  // Compared on the accumulator, inside the range, so neither the clamp nor the snap grid blurs the ratio.
  const wide = { min: 0, max: 1000, step: 1 };
  const acc = (mods) => scrub.advanceScrub(scrub.beginScrub(100, wide), 10, mods).acc - 100;
  ok('Shift is ten times as fast', near(acc({ shiftKey: true }) / acc({}), 10, 1e-9), String(acc({ shiftKey: true }) / acc({})));
  ok('Alt is a tenth as fast', near(acc({ altKey: true }) / acc({}), 0.1, 1e-9));
  ok('Shift and Alt together cancel to x1 (releasing one key mid-drag never jumps the value)', near(acc({ shiftKey: true, altKey: true }), acc({}), 1e-9));
  ok('Alt snaps a fractional control to 0.001', Math.abs(Math.round(run({ altKey: true }, 7) * 1000) - run({ altKey: true }, 7) * 1000) < 1e-6);
  ok('an integer control keeps its step under Alt (a fraction of a thread is not a value)',
    Number.isInteger(scrub.advanceScrub(scrub.beginScrub(8, { min: 1, max: 32, step: 1 }), 40, { altKey: true }).value));
  ok('the value never leaves the range', run({ shiftKey: true }, 100000) === 1 && run({ shiftKey: true }, -100000) === 0);
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Effect rack: which modules have a real wet/dry mix');
{
  const d = { selected_enhancer: 'UltraMax', blend_ratio: 1, merger_grain_match: 0.45, color_transfer_mode: 'lct', merger_hist_match: 0.4, mask_engine_2: 'None', upscale_after_swap: false };
  ok('six modules in engine order', rack.STAGE_IDS.join() === 'swapper,restorer,grain,color,occlusion,upscale');
  ok('restorer / grain / colour mix are settings the render reads', rack.MIX_KEYS.restorer === 'blend_ratio' && rack.MIX_KEYS.grain === 'merger_grain_match' && rack.MIX_KEYS.color === 'merger_hist_match');
  ok('the occlusion mask and the upscaler have NO mix key, and say why',
    rack.stageState('occlusion', d, d).mixKey === null && rack.stageState('upscale', d, d).mixKey === null
    && !!rack.MIX_UNAVAILABLE.occlusion && !!rack.MIX_UNAVAILABLE.upscale);
  ok('the swapper is locked on', rack.stageState('swapper', d, d).locked === true && rack.stageState('swapper', d, d).enabled === true);
  ok('enabled flags follow the settings', rack.stageState('restorer', d, d).enabled && rack.stageState('grain', d, d).enabled
    && rack.stageState('color', d, d).enabled && !rack.stageState('occlusion', d, d).enabled && !rack.stageState('upscale', d, d).enabled);
  ok('Restorer "None" is bypassed; grain 0 is bypassed; colour "none" is bypassed',
    !rack.stageState('restorer', { selected_enhancer: 'None' }, d).enabled && !rack.stageState('grain', { merger_grain_match: 0 }, d).enabled
    && !rack.stageState('color', { color_transfer_mode: 'none' }, d).enabled);
  ok('the dead XSeg-governance keys are NOT bound by any module (nothing in ProcessMgr reads them)',
    !JSON.stringify(rack).includes('enable_xseg_occlusion') || /NOT HERE/.test(readFileSync(join(here, '../src/components/faceswap/effectRackModel.js'), 'utf8')));
}

section('Effect rack: bypass remembers and restores');
{
  const d = { selected_enhancer: 'UltraMax', merger_grain_match: 0.45, color_transfer_mode: 'lct', mask_engine_2: 'None' };
  const memo = {};
  let p = { selected_enhancer: 'GPEN', merger_grain_match: 0.8, color_transfer_mode: 'mkl', mask_engine_2: 'DFL XSeg' };
  const apply = (patch) => { p = { ...p, [patch[0]]: patch[1] }; return patch; };
  ok('restorer off -> None', apply(rack.togglePatch('restorer', false, p, d, memo))[1] === 'None');
  ok('restorer on again -> the one it had (GPEN), not the default', apply(rack.togglePatch('restorer', true, p, d, memo))[1] === 'GPEN');
  ok('grain off -> 0, back on -> 0.8', apply(rack.togglePatch('grain', false, p, d, memo))[1] === 0 && apply(rack.togglePatch('grain', true, p, d, memo))[1] === 0.8);
  ok('colour off -> none, back on -> mkl', apply(rack.togglePatch('color', false, p, d, memo))[1] === 'none' && apply(rack.togglePatch('color', true, p, d, memo))[1] === 'mkl');
  ok('occlusion off -> None, back on -> DFL XSeg', apply(rack.togglePatch('occlusion', false, p, d, memo))[1] === 'None' && apply(rack.togglePatch('occlusion', true, p, d, memo))[1] === 'DFL XSeg');
  ok('turning on with nothing remembered falls back to a sensible default',
    rack.togglePatch('restorer', true, { selected_enhancer: 'None' }, d, {})[1] === 'UltraMax' && rack.togglePatch('occlusion', true, { mask_engine_2: 'None' }, d, {})[1] === 'DFL XSeg');
  ok('the swapper has no toggle', rack.togglePatch('swapper', false, p, d, memo) === null);
  const w = rack.togglePatch('upscale', true, p, d, memo);
  ok('upscale writes upscale_after_swap', w[0] === 'upscale_after_swap' && w[1] === true);
}

section('Effect rack: latency badge and its provenance');
{
  const L = (stage, p) => rack.latencyFor(stage, p);
  ok('GPEN-256 ~3 ms is Low, MEASURED', L('restorer', { selected_enhancer: 'GPEN-256' }).level === 'low' && L('restorer', { selected_enhancer: 'GPEN-256' }).basis === 'measured');
  ok('GPEN-512 / RestoreFormer++ / CodeFormer are Med, measured',
    ['GPEN-512', 'Restoreformer++', 'Codeformer'].every((m) => L('restorer', { selected_enhancer: m }).level === 'med' && L('restorer', { selected_enhancer: m }).basis === 'measured'));
  ok('a restorer with no measurement is Med and says ESTIMATE', L('restorer', { selected_enhancer: 'DMDNet' }).basis === 'estimate');
  ok('UltraMax is High and an estimate (not measured here)', L('restorer', { selected_enhancer: 'UltraMax' }).level === 'high' && L('restorer', { selected_enhancer: 'UltraMax' }).basis === 'estimate');
  ok('a bypassed restorer costs nothing (Low)', L('restorer', { selected_enhancer: 'None' }).level === 'low');
  ok('the Low / Med / High thresholds are 8 ms and 40 ms', rack.levelForMs(7.9) === 'low' && rack.levelForMs(8) === 'med' && rack.levelForMs(39.9) === 'med' && rack.levelForMs(40) === 'high');
  ok('colour: RCT/LCT Low, MKL Med, IDT High', L('color', { color_transfer_mode: 'rct' }).level === 'low' && L('color', { color_transfer_mode: 'lct' }).level === 'low'
    && L('color', { color_transfer_mode: 'mkl' }).level === 'med' && L('color', { color_transfer_mode: 'idt' }).level === 'high');
  ok('upscale: resampling Low, Compact / x2 Med, x4 networks High',
    L('upscale', { upscale_model_after: 'fsr_x2' }).level === 'low' && L('upscale', { upscale_model_after: 'compact_x4' }).level === 'med'
    && L('upscale', { upscale_model_after: 'esrganx2' }).level === 'med' && L('upscale', { upscale_model_after: 'esrganx4' }).level === 'high');
  ok('occlusion off is Low, an engine is Med', L('occlusion', { mask_engine_2: 'None' }).level === 'low' && L('occlusion', { mask_engine_2: 'DFL XSeg' }).level === 'med');
  ok('"High VRAM" is the label for the top level', rack.LATENCY_LEVELS.high.label === 'High VRAM');
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Routing: the pills are the backend\'s table, name for name and range for range');
{
  const py = readFileSync(join(here, '../../app/roop/person_overrides.py'), 'utf8');
  const table = {};
  for (const m of py.matchAll(/^\s*"(\w+)":\s*\(([-\d.]+),\s*([-\d.]+),\s*(True|False)\),/gm)) table[m[1]] = { lo: Number(m[2]), hi: Number(m[3]), integer: m[4] === 'True' };
  ok('the backend table was found (5 names)', Object.keys(table).length === 5, Object.keys(table).join());
  ok('every pill is a backend name and every backend name has a pill', rt.OVERRIDE_PILLS.map((p) => p.key).sort().join() === Object.keys(table).sort().join());
  for (const pill of rt.OVERRIDE_PILLS) {
    const t = table[pill.key];
    ok(`${pill.key}: the pill range sits inside the backend's [${t?.lo}, ${t?.hi}]`, !!t && pill.min >= t.lo && pill.max <= t.hi);
    ok(`${pill.key}: whole-number control iff the backend wants an integer`, !!t && !!pill.integer === t.integer);
  }
  const modes = py.match(/PERSON_ROUTED_MODES\s*=\s*frozenset\(\{([^}]*)\}\)/)?.[1] ?? '';
  ok('the dock applies overrides in exactly the modes the backend routes by person',
    rt.overridesApply('Selected face') && rt.overridesApply('Selected people') && !rt.overridesApply('All faces')
    && !rt.overridesApply('First found') && /"selected"/.test(modes) && /"selected_multi"/.test(modes));
}

section('Routing: override state');
{
  const pill = rt.OVERRIDE_PILLS.find((p) => p.key === 'mask_feather');
  let o = {};
  o = rt.withOverride(o, 'p1', 'mask_feather', 40);
  ok('set one override', o.p1.mask_feather === 40);
  const before = o;
  o = rt.withOverride(o, 'p1', 'blend_ratio', 0.5);
  ok('a second key keeps the first and the first map is untouched (new object)', o.p1.mask_feather === 40 && o.p1.blend_ratio === 0.5 && before.p1.blend_ratio === undefined);
  ok('a value over the range is clamped to the pill', rt.withOverride({}, 'p1', 'mask_feather', 999).p1.mask_feather === 200);
  ok('the radius is rounded to a whole number', rt.withOverride({}, 'p1', 'mask_erode_dilate_radius', 3.6).p1.mask_erode_dilate_radius === 4);
  ok('an unknown key / NaN / no person is ignored', Object.keys(rt.withOverride({}, 'p1', 'face_mask_blend', 3)).length === 0
    && Object.keys(rt.withOverride({}, 'p1', 'mask_feather', NaN)).length === 0 && Object.keys(rt.withOverride({}, '', 'mask_feather', 3)).length === 0);
  ok('clearing the last key removes the person', Object.keys(rt.withoutOverride(rt.withOverride({}, 'p1', 'mask_feather', 3), 'p1', 'mask_feather')).length === 0);
  ok('clearing a person removes all their keys', !('p1' in rt.withoutOverride(o, 'p1')));
  ok('effective value is the override, else the ordinary setting, else the pill fallback',
    rt.effectiveValue(pill, o, 'p1', { face_mask_blend: 20 }) === 40 && rt.effectiveValue(pill, o, 'p2', { face_mask_blend: 20 }) === 20
    && rt.effectiveValue(pill, {}, 'p2', {}) === 12);
  ok('pill text: px, plain, percent', rt.formatPill(rt.OVERRIDE_PILLS[0], -4) === '-4 px' && rt.formatPill(rt.OVERRIDE_PILLS[2], 0.55) === '0.55'
    && rt.formatPill(rt.OVERRIDE_PILLS[4], 0.8) === '80%' && rt.formatPill(rt.OVERRIDE_PILLS[1], NaN) === '—');
}

section('Routing: what goes on the wire');
{
  const saved = { p1: { mask_feather: 30, blend_ratio: 0.5, bogus: 1 }, gone: { mask_feather: 5 }, p2: { mask_feather: 9999 }, p3: {} };
  const wire = rt.wireOverrides(saved, ['p1', 'p2', 'p3']);
  ok('a person the clip no longer has is not sent', !('gone' in wire));
  ok('an unknown key and an out-of-range value are dropped, in-range ones kept', wire.p1.mask_feather === 30 && wire.p1.blend_ratio === 0.5 && !('bogus' in wire.p1) && !('p2' in wire));
  ok('an empty person is not sent', !('p3' in wire));
  ok('nothing saved is an empty object, not undefined', JSON.stringify(rt.wireOverrides(undefined, ['p1'])) === '{}');
}

section('Routing: presence, pose, lines');
{
  const row = { id: 'p1', intervals: [[1, 50], [40, 60], [200, 250]] };
  ok('presence % merges overlapping intervals (1-60 and 200-250 of 500 = 111 frames = 22.2%)', near(rt.presencePercent(row, 500), (60 + 51) / 500 * 100, 1e-9), String(rt.presencePercent(row, 500)));
  ok('presence is clamped to the clip', rt.presencePercent({ intervals: [[1, 9999]] }, 100) === 100);
  ok('no scan / no clip length is null (not 0%)', rt.presencePercent(undefined, 100) === null && rt.presencePercent(row, 0) === null);
  ok('a scanned person who never appears is 0, not null', rt.presencePercent({ intervals: [] }, 100) === 0);
  ok('pose badges are short', rt.poseBadge('Left Profile + Up Tilt') === 'Left · Up' && rt.poseBadge('Front') === 'Front' && rt.poseBadge('') === 'Front');
  const lines = rt.routeLines(
    [{ id: 'a', y: 24, sourceIndex: 1 }, { id: 'b', y: 78, sourceIndex: -1 }, { id: 'c', y: 132, sourceIndex: 9 }],
    [{ y: 24 }, { y: 78 }], 0, 56);
  ok('a route per mapped person; skip and a vanished source draw nothing', lines.length === 1 && lines[0].id === 'a' && lines[0].sourceIndex === 1);
  ok('the line runs from the target socket to the source socket', /^M0 24 C.* 56 78$/.test(lines[0].d), lines[0].d);
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nstage 5 studio: all checks passed' : `\nstage 5 studio: ${failures} FAILED`);
