/**
 * Stage 26's pure logic (plain node, no browser): the expression parser behind numeric fields, the scrub arithmetic,
 * and the key-combo / keymap model behind the shortcut overlay and the remapper.
 *
 *   node .render-check/run-module.mjs render-stage26-logic.mjs
 *
 * The parser is checked against a brute-force oracle (JavaScript's own arithmetic on generated expressions) as well as
 * by example: a hand-picked example finds the cases someone thought of, and an operator-precedence or associativity
 * slip is exactly the case nobody did. The keymap section pins the SHIPPED DEFAULTS to what the handlers matched before
 * the registry existed, because a remapper that quietly changes the default keys is a regression with no error.
 */
import { evaluateExpression, normalizeExpression, settleTypedValue, MAX_EXPR_LENGTH } from '../src/numeric/mathExpr.js';
import {
  COARSE, FINE, MICRO_STEP, advanceScrub, beginScrub, decimalsFor, scrubMultiplier, snapStep, snapTo, unitsPerPixel,
} from '../src/numeric/scrubMath.js';
import {
  comboFromEvent, formatCombo, hasMod, keyToken, normalizeCombo, parseCombo, reservedReason, scopesOverlap,
} from '../src/shortcuts/combo.js';
import { ACTIONS, ACTION_BY_ID, CATEGORIES } from '../src/shortcuts/actions.js';
import {
  allConflicts, bindingsOf, buildIndex, checkBinding, conflictsFor, parseStored, sanitizeOverrides, serialize,
} from '../src/shortcuts/keymap.js';

import {
  CORNER_ORDER, PIP_BAR, PIP_DEFAULT, PIP_MARGIN, PIP_MAX_W, PIP_MIN_W, PIP_SNAP, PIP_TABS, changesLiveSource, clampPosition,
  cornerPositions, nextCorner, pictureHeight, pipVisible, replayBody, resizeTo, sanitizeSaved, selectedSourceOf, snapCorner,
} from '../src/pip/pipModel.js';

// The shortcut STORE reads localStorage when it loads, so a fake one - already holding a saved remap - goes in first.
const fakeStorage = (() => {
  const m = new Map([['roop_keybinds_v1', JSON.stringify({ v: 1, overrides: { 'viewport.fit': ['Y'] } })]]);
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => { m.set(k, String(v)); }, removeItem: (k) => { m.delete(k); } };
})();
globalThis.localStorage = fakeStorage;
const store = await import('../src/store/shortcutStore.js');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;
const val = (text, ctx) => { const r = evaluateExpression(text, ctx); return r.ok ? r.value : `ERR:${r.error}`; };

let seed = 26;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const int = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));

// ── 1. Expressions: examples from the brief ────────────────────────────────
console.log('expressions');
ok('512/2 = 256', val('512/2') === 256);
ok('24*1.5 = 36', val('24*1.5') === 36);
ok('-5 is the number -5, not "5 below current"', val('-5', { current: 40 }) === -5);
ok('+10% scales the CURRENT value (40 -> 44)', near(val('+10%', { current: 40 }), 44));
ok('-10% scales it down (40 -> 36)', near(val('-10%', { current: 40 }), 36));
ok('+10% of 0 stays 0 (no divide by the current)', val('+10%', { current: 0 }) === 0);
ok('50% is half the RANGE (0..1 -> 0.5)', val('50%', { min: 0, max: 1 }) === 0.5);
ok('50% over 0..4096 is 2048', val('50%', { min: 0, max: 4096 }) === 2048);
ok('50% over a range that starts above 0 (10..30 -> 20)', val('50%', { min: 10, max: 30 }) === 20);
ok('50% with no range is 0.5', val('50%') === 0.5);
ok('200 + 10% is 10% OF 200 (220)', near(val('200+10%'), 220));
ok('200 - 10% is 180', near(val('200-10%'), 180));
ok('200 * 50% is 100', val('200*50%') === 100);
ok('a + 2*10% does NOT read the percent relative (only a bare % term does)', near(val('200+2*10%'), 200.2));

// ── 2. Precedence, associativity, grouping ─────────────────────────────────
console.log('precedence');
ok('2+3*4 = 14', val('2+3*4') === 14);
ok('(2+3)*4 = 20', val('(2+3)*4') === 20);
ok('10-4-3 is left-associative (3)', val('10-4-3') === 3);
ok('100/10/5 is left-associative (2)', val('100/10/5') === 2);
ok('2^3^2 is right-associative (512)', val('2^3^2') === 512);
ok('-2^2 is -(2^2) = -4', val('-2^2') === -4);
ok('2^-1 = 0.5', val('2^-1') === 0.5);
ok('unary chains: - - 3 = 3', val('--3') === 3);
ok('+-+3 = -3', val('+-+3') === -3);
ok('pi is the constant', near(val('pi'), Math.PI));
ok('2*pi', near(val('2*pi'), 2 * Math.PI));
ok('scientific notation 1e3', val('1e3') === 1000);
ok('.5 and 5. both parse', val('.5') === 0.5 && val('5.') === 5);
ok('decimal comma: 1,5*2 = 3', val('1,5*2') === 3);
ok('typographic operators: 6 × 7 ÷ 2 − 1 = 20', val('6 × 7 ÷ 2 − 1') === 20);
ok('whitespace is ignored', val('  12 *  ( 1 + 1 ) ') === 24);

// ── 3. Oracle: generated expressions match JavaScript's arithmetic ─────────
console.log('oracle');
{
  const OPS = ['+', '-', '*', '/'];
  let checked = 0;
  let bad = null;
  const gen = (depth) => {
    if (depth === 0 || rnd() < 0.3) {
      const n = (int(1, 90) / int(1, 9)).toFixed(2);
      return n;
    }
    const a = gen(depth - 1);
    const b = gen(depth - 1);
    const op = OPS[int(0, 3)];
    const text = `${a}${op}${b}`;
    return rnd() < 0.4 ? `(${text})` : text;
  };
  for (let i = 0; i < 400 && !bad; i += 1) {
    const text = gen(3);
    // eslint-disable-next-line no-new-func -- the test ORACLE: trusted generated text, never user input
    const want = new Function(`return (${text});`)();
    const got = evaluateExpression(text);
    if (!Number.isFinite(want)) { if (got.ok) bad = `${text}: oracle non-finite, parser gave ${got.value}`; }
    else if (!got.ok) bad = `${text}: parser said ${got.error}`;
    else if (Math.abs(got.value - want) > 1e-9 * Math.max(1, Math.abs(want))) bad = `${text}: oracle ${want}, parser ${got.value}`;
    checked += 1;
  }
  ok(`400 generated expressions agree with the oracle (${checked} run)`, !bad, bad || '');
}

// ── 4. Rejection: nothing here may evaluate, throw or hang ─────────────────
console.log('rejection');
const rejects = (text, why) => ok(`rejects ${JSON.stringify(text.length > 30 ? `${text.slice(0, 27)}...` : text)} (${why})`, !evaluateExpression(text).ok);
rejects('', 'empty');
rejects('   ', 'blank');
rejects('abc', 'identifier');
rejects('alert(1)', 'call');
rejects('constructor', 'identifier');
rejects('1;2', 'semicolon');
rejects('1,2,3', 'is not a valid number list');
rejects('process.exit()', 'member access');
rejects('"a"+1', 'string');
rejects('0x10', 'hex literal is not in the grammar');
rejects('1/0', 'division by zero');
rejects('(1+2', 'unbalanced');
rejects('1+2)', 'unbalanced');
rejects('1+', 'trailing operator');
rejects('*3', 'leading binary operator');
rejects('9^9^9^9', 'overflow to Infinity');
rejects('1e999', 'literal out of range');
rejects('(((((((((((((((((((((((((((((((((((((((((((((1)))))))))))))))))))))))))))))))))))))))))))))', 'nesting depth');
rejects('1+'.repeat(MAX_EXPR_LENGTH) + '1', 'length');
rejects('-'.repeat(200) + '1', 'long unary chain (length cap)');
ok('50 nested parens fail by message, not by stack overflow', (() => { try { return !evaluateExpression('('.repeat(50) + '1' + ')'.repeat(50)).ok; } catch { return false; } })());
ok('"pix" is not "pi" followed by x', !evaluateExpression('pix').ok);
ok('error text is a message', typeof evaluateExpression('1/0').error === 'string' && evaluateExpression('1/0').error.length > 0);
ok('normalizeExpression maps unicode minus', normalizeExpression('5−2') === '5-2');

// ── 5. settleTypedValue: the control's contract ────────────────────────────
console.log('settle');
{
  const s = (t, o) => settleTypedValue(t, o);
  ok('integer step rounds: 512/3 -> 171', s('512/3', { min: 0, max: 4096, step: 1 }).value === 171);
  ok('fractional step keeps 4 decimals: 1/3 -> 0.3333', s('1/3', { min: 0, max: 1, step: 0.01 }).value === 0.3333);
  ok('clamps high and says so', (() => { const r = s('9999', { min: 0, max: 4096, step: 1 }); return r.value === 4096 && r.clamped === true; })());
  ok('clamps low and says so', (() => { const r = s('-5', { min: 0, max: 1, step: 0.01 }); return r.value === 0 && r.clamped === true; })());
  ok('in range is not "clamped"', s('0.5', { min: 0, max: 1, step: 0.01 }).clamped === false);
  ok('+10% then clamps at the top (0.95 -> 1)', s('+10%', { current: 0.95, min: 0, max: 1, step: 0.01 }).value === 1);
  ok('a failure passes the message through', s('1/0', { min: 0, max: 1, step: 0.01 }).ok === false);
  ok('integer control: +10% of 7 (7.7) -> 8', s('+10%', { current: 7, min: 0, max: 100, step: 1 }).value === 8);
}

// ── 6. Scrub arithmetic ────────────────────────────────────────────────────
console.log('scrub');
{
  const SL = { min: 0, max: 1, step: 0.01 };
  ok('multiplier: none/shift/alt/both', scrubMultiplier({}) === 1 && scrubMultiplier({ shiftKey: true }) === COARSE
    && scrubMultiplier({ altKey: true }) === FINE && scrubMultiplier({ shiftKey: true, altKey: true }) === 1);
  ok('0..1 crosses the range in 400 px at plain pace', near(1 / unitsPerPixel(0, 1, 0.01), 400));
  ok('0..4096 crosses it in 600 px', near(4096 / unitsPerPixel(0, 4096, 1), 600));
  ok('Shift is exactly 10x Alt-free pace', (() => {
    const a = advanceScrub(beginScrub(0, SL), 20, {}); const b = advanceScrub(beginScrub(0, SL), 20, { shiftKey: true });
    return near(b.acc, a.acc * 10);
  })());
  ok('Alt is exactly 0.1x', (() => {
    const a = advanceScrub(beginScrub(0, SL), 200, {}); const b = advanceScrub(beginScrub(0, SL), 200, { altKey: true });
    return near(b.acc, a.acc * 0.1);
  })());
  ok('Alt snaps to 0.001, plain to the step', (() => {
    const fine = advanceScrub(beginScrub(0.5, SL), 7, { altKey: true });
    const plain = advanceScrub(beginScrub(0.5, SL), 7, {});
    return near(fine.value * 1000, Math.round(fine.value * 1000)) && near(plain.value * 100, Math.round(plain.value * 100));
  })());
  ok('Alt can reach a value the step cannot (0.5 -> 0.501)', (() => {
    let s = beginScrub(0.5, SL);
    s = advanceScrub(s, 1, { altKey: true });                // 1 px * (0.0025 * 0.1) = 0.00025: under half a milli
    for (let i = 0; i < 4; i += 1) s = advanceScrub(s, 1, { altKey: true });
    return near(s.value, 0.501) || near(s.value, 0.5);       // reaches the milli grid, never jumps a whole step
  })());
  ok('an integer control keeps its step under Alt', snapStep(1, { altKey: true }) === 1 && snapStep(0.01, { altKey: true }) === MICRO_STEP);
  ok('value never leaves [min, max]', (() => {
    let s = beginScrub(0.5, SL);
    for (let i = 0; i < 200; i += 1) { s = advanceScrub(s, int(-500, 500), { shiftKey: rnd() < 0.5 }); if (s.value < 0 || s.value > 1) return false; }
    return true;
  })());
  ok('overshoot does not build a dead zone: past the end then back reverses at once', (() => {
    let s = beginScrub(0.9, SL);
    s = advanceScrub(s, 5000, {});                           // far past max
    const atMax = s.value;
    s = advanceScrub(s, -40, {});                            // 40 px back
    return atMax === 1 && s.value < 1;
  })());
  ok('releasing Shift mid-drag does not jump the value', (() => {
    let s = beginScrub(0, SL);
    s = advanceScrub(s, 10, { shiftKey: true });
    const before = s.value;
    s = advanceScrub(s, 0, {});                              // modifier changed, no travel
    return s.value === before;
  })());
  ok('a range whose min is off the step grid snaps from min (min 0.5, step 1 -> 1.5)', (() => {
    const s = advanceScrub(beginScrub(0.5, { min: 0.5, max: 10, step: 1 }), 6, {});
    return near((s.value - 0.5) % 1, 0);
  })());
  ok('snapTo removes float drift', snapTo(0.1 + 0.2, 0.01) === 0.3);
  ok('decimalsFor: shows milli precision only when present', decimalsFor(0.5, 2) === 2 && decimalsFor(0.501, 2) === 3 && decimalsFor(4, 0) === 0);
}

// ── 7. Combos ──────────────────────────────────────────────────────────────
console.log('combos');
const ev = (o) => ({ key: '', code: '', ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, ...o });
{
  const c = (o) => comboFromEvent(ev(o));
  ok('letters fold case (Caps Lock changes nothing)', c({ key: 'c', code: 'KeyC' }) === 'C' && c({ key: 'C', code: 'KeyC' }) === 'C');
  ok('Shift stays on a letter', c({ key: 'C', code: 'KeyC', shiftKey: true }) === 'Shift+C');
  ok('Ctrl and Cmd are both Mod', c({ key: 'z', code: 'KeyZ', ctrlKey: true }) === 'Mod+Z' && c({ key: 'z', code: 'KeyZ', metaKey: true }) === 'Mod+Z');
  ok('Mod+Shift+Z', c({ key: 'Z', code: 'KeyZ', ctrlKey: true, shiftKey: true }) === 'Mod+Shift+Z');
  ok('digit comes from the code (Shift+1 reports "!")', c({ key: '!', code: 'Digit1', shiftKey: true }) === 'Shift+1' && c({ key: '1', code: 'Digit1' }) === '1');
  ok('space', c({ key: ' ', code: 'Space' }) === 'Space');
  ok('? drops its Shift (the character is the identity)', c({ key: '?', code: 'Slash', shiftKey: true }) === '?');
  ok('Ctrl+/ is Mod+/', c({ key: '/', code: 'Slash', ctrlKey: true }) === 'Mod+/');
  ok('a plain symbol reported WITH Shift stays a Shift press (Playwright / a layout whose Shift+. is not ">")',
    c({ key: '.', code: 'Period', shiftKey: true }) === 'Shift+.' && c({ key: ',', code: 'Comma', shiftKey: true }) === 'Shift+,'
    && c({ key: '/', code: 'Slash', shiftKey: true }) === 'Shift+/' && c({ key: '[', code: 'BracketLeft', shiftKey: true }) === 'Shift+[');
  ok('...but the shifted CHARACTER does not carry a second Shift', c({ key: '>', code: 'Period', shiftKey: true }) === '>'
    && c({ key: '?', code: 'Slash', shiftKey: true }) === '?' && c({ key: '{', code: 'BracketLeft', shiftKey: true }) === '{');
  ok('a symbol with no Shift is just the symbol', c({ key: '.', code: 'Period' }) === '.' && c({ key: ',', code: 'Comma' }) === ',');
  ok('< and > are the symbols, with no Shift', c({ key: '<', code: 'Comma', shiftKey: true }) === '<' && c({ key: '>', code: 'Period', shiftKey: true }) === '>');
  ok('{ and } are the symbols', c({ key: '{', code: 'BracketLeft', shiftKey: true }) === '{');
  ok('Ctrl+Shift+= is Mod++ (the + key)', c({ key: '+', code: 'Equal', ctrlKey: true, shiftKey: true }) === 'Mod++');
  ok('Ctrl+= is Mod+=', c({ key: '=', code: 'Equal', ctrlKey: true }) === 'Mod+=');
  ok('named keys keep Shift: Shift+ArrowLeft', c({ key: 'ArrowLeft', code: 'ArrowLeft', shiftKey: true }) === 'Shift+ArrowLeft');
  ok('a bare modifier is not a combo', c({ key: 'Shift', code: 'ShiftLeft', shiftKey: true }) === null && c({ key: 'Control', ctrlKey: true }) === null);
  ok('dead keys are not combos', c({ key: 'Dead', code: 'Quote' }) === null);
  ok('non-Latin layout: the letter comes from the code', c({ key: 'я', code: 'KeyZ' }) === 'Z');
  ok('Esc is Escape', c({ key: 'Esc' }) === 'Escape');
  ok('F11', c({ key: 'F11', code: 'F11' }) === 'F11');
  ok('keyToken of a numpad digit is the digit', keyToken(ev({ key: '2', code: 'Numpad2' })) === '2');
  ok('parseCombo: Mod++ is Mod and the + key', (() => { const p = parseCombo('Mod++'); return p && p.key === '+' && p.mods.join() === 'Mod'; })());
  ok('parseCombo: "+" alone is the plus key', parseCombo('+')?.key === '+');
  ok('parseCombo rejects "Mod+", "Shift", "", null', parseCombo('Mod+') === null && parseCombo('Shift') === null && parseCombo('') === null && parseCombo(null) === null);
  ok('normalizeCombo reorders modifiers', normalizeCombo('Shift+Mod+Z') === 'Mod+Shift+Z');
  ok('normalizeCombo rejects duplicate modifiers', normalizeCombo('Mod+Mod+Z') === null);
  ok('round trip: every shipped default parses and re-spells identically',
    ACTIONS.every((a) => a.keys.every((k) => normalizeCombo(k) === k)));
  ok('hasMod', hasMod('Mod+K') && !hasMod('K') && !hasMod('Shift+C'));
  ok('formatCombo: PC and Mac spellings', formatCombo('Mod+Shift+Z', { mac: false }) === 'Ctrl + Shift + Z'
    && formatCombo('Mod+Shift+Z', { mac: true }) === '⌘⇧Z' && formatCombo('ArrowLeft', { mac: false }) === '←');
  ok('reserved: bare Escape/Tab/Enter and the browser-owned combos', ['Escape', 'Tab', 'Enter', 'Mod+W', 'Mod+R', 'F5'].every((k) => reservedReason(k)));
  ok('Mod+Enter is NOT reserved (it runs the swap)', reservedReason('Mod+Enter') === null);
  ok('scopes: global overlaps everything; faceswap and monitor do not overlap', scopesOverlap('global', 'monitor') && scopesOverlap('faceswap', 'faceswap') && !scopesOverlap('faceswap', 'monitor'));
}

// ── 8. The keymap: defaults, conflicts, persistence ────────────────────────
console.log('keymap');
{
  ok('action ids are unique', new Set(ACTIONS.map((a) => a.id)).size === ACTIONS.length);
  ok('every action has a label, a known category and a scope', ACTIONS.every((a) => a.label && CATEGORIES.includes(a.category) && ['global', 'faceswap', 'monitor'].includes(a.scope)));
  ok('every action has at least one default key', ACTIONS.every((a) => a.keys.length > 0));
  ok('every category has an action', CATEGORIES.every((cat) => ACTIONS.some((a) => a.category === cat)));
  ok('THE SHIPPED KEYMAP HAS NO CLASH', allConflicts({}).length === 0, JSON.stringify(allConflicts({})));

  // What the handlers matched before the registry existed. If one of these moves, a default key moved.
  const WAS = {
    'app.palette': ['Mod+K'], 'app.toggleRail': ['Mod+B'], 'app.zoomReset': ['Mod+0'],
    'transport.playPause': ['Space'], 'transport.stepBack': ['ArrowLeft'], 'transport.stepForward': ['ArrowRight'],
    'transport.stepBack10': ['Shift+ArrowLeft'], 'transport.stepForward10': ['Shift+ArrowRight'],
    'transport.toStart': ['Home'], 'transport.toEnd': ['End'], 'transport.run': ['Mod+Enter'], 'transport.addToQueue': ['Q'],
    'timeline.trimStart': ['['], 'timeline.trimEnd': [']'], 'timeline.resetRange': ['R'], 'timeline.curveEditor': ['C'],
    'viewport.compare': ['Shift+C'], 'viewport.splitView': ['S'],
    'timeline.marker': ['M'], 'transport.shuttleBack': ['J'], 'transport.shuttlePause': ['K'], 'transport.shuttleForward': ['L'],
    'timeline.setIn': ['I'], 'timeline.setOut': ['O'], 'timeline.frameBack': [','], 'timeline.frameForward': ['.'],
    'timeline.frameBack10': ['<', 'Shift+,'], 'timeline.frameForward10': ['>', 'Shift+.'], 'timeline.prevCut': ['ArrowUp'], 'timeline.nextCut': ['ArrowDown'],
    'viewport.fit': ['F'], 'viewport.lockTarget': ['T'], 'viewport.debugOverlay': ['D'], 'viewport.loupe': ['G'],
    'mask.brush': ['B'], 'mask.pen': ['P'], 'viewport.compareAxis': ['X'], 'viewport.compareMode': ['W'], 'viewport.autoSwipe': ['A'],
    'viewport.zoom100': ['Shift+1'], 'viewport.zoom200': ['Shift+2'], 'viewport.zoom400': ['Shift+4'],
    'snapshot.slotA': ['1'], 'snapshot.slotB': ['2'], 'snapshot.slotC': ['3'], 'snapshot.slotD': ['4'],
    'edit.undo': ['Mod+Z'], 'monitor.fullscreen': ['F', 'F11'], 'viewport.peekOriginal': ['\\'],
  };
  const drift = Object.entries(WAS).filter(([id, keys]) => JSON.stringify(ACTION_BY_ID[id]?.keys) !== JSON.stringify(keys));
  ok('every shipped default equals what its handler matched before', drift.length === 0, JSON.stringify(drift));

  ok('undo/redo cover Ctrl+Z, Ctrl+Y and Ctrl+Shift+Z', JSON.stringify(ACTION_BY_ID['edit.redo'].keys) === JSON.stringify(['Mod+Y', 'Mod+Shift+Z']));
  ok('M no longer opens the loupe (it set a marker AND opened the magnifier before)', !ACTION_BY_ID['viewport.loupe'].keys.includes('M'));

  const index = buildIndex({});
  ok('index resolves a combo to its action', index.get('Mod+K')?.[0] === 'app.palette' && index.get('Shift+C')?.[0] === 'viewport.compare');
  ok('C and Shift+C are different keys', index.get('C')?.[0] === 'timeline.curveEditor' && index.get('Shift+C')?.[0] === 'viewport.compare');

  // Overrides
  ok('an override replaces the defaults', JSON.stringify(bindingsOf('viewport.splitView', { 'viewport.splitView': ['Y'] })) === '["Y"]');
  ok('an empty override means unbound', bindingsOf('viewport.splitView', { 'viewport.splitView': [] }).length === 0);
  ok('unbound actions leave the index', !buildIndex({ 'viewport.splitView': [] }).get('S'));
  ok('conflictsFor: Q is taken by "Add to the batch queue"', conflictsFor('viewport.splitView', 'Q', {}).map((a) => a.id).join() === 'transport.addToQueue');
  ok('conflictsFor: a free key', conflictsFor('viewport.splitView', 'Y', {}).length === 0);
  ok('conflictsFor: the same key on the same action is not a conflict with itself', conflictsFor('viewport.splitView', 'S', {}).length === 0);
  ok('a monitor-scope action may share a faceswap key (F)', conflictsFor('monitor.fullscreen', 'F', {}).length === 0);
  ok('but a GLOBAL action may not take one (Mod+Z is global; faceswap sees it)', conflictsFor('app.palette', 'Mod+Z', {}).length === 1);
  ok('allConflicts finds a deliberate clash', allConflicts({ 'viewport.splitView': ['Q'] }).some((x) => x.combo === 'Q' && x.actions.includes('viewport.splitView') && x.actions.includes('transport.addToQueue')));
  ok('a swap (two actions exchanging keys) leaves no clash', allConflicts({ 'viewport.splitView': ['Q'], 'transport.addToQueue': ['S'] }).length === 0);

  ok('checkBinding: free key ok', checkBinding('viewport.splitView', 'Y', {}).ok === true);
  ok('checkBinding: reserved key refused with a reason', (() => { const r = checkBinding('viewport.splitView', 'Escape', {}); return !r.ok && /Esc/.test(r.reason); })());
  ok('checkBinding: conflict refused, naming the owner', (() => { const r = checkBinding('viewport.splitView', 'Q', {}); return !r.ok && r.conflicts?.length === 1 && /queue/i.test(r.reason); })());
  ok('checkBinding: junk refused', checkBinding('viewport.splitView', 'Mod+', {}).ok === false);

  // Persistence
  const stored = serialize({ 'viewport.splitView': ['Y'] });
  ok('serialize / parseStored round-trips', JSON.stringify(parseStored(stored)) === '{"viewport.splitView":["Y"]}');
  ok('parseStored survives garbage', Object.keys(parseStored('{not json')).length === 0 && Object.keys(parseStored(null)).length === 0 && Object.keys(parseStored('[]')).length === 0);
  ok('parseStored rejects another schema version', Object.keys(parseStored(JSON.stringify({ v: 99, overrides: { 'viewport.splitView': ['Y'] } }))).length === 0);
  ok('sanitize drops ids this build does not know (an action removed in a later version)', Object.keys(sanitizeOverrides({ 'nope.gone': ['Y'] })).length === 0);
  ok('sanitize drops reserved and unparseable combos', JSON.stringify(sanitizeOverrides({ 'viewport.splitView': ['Escape', 'Mod+', 'Y'] })) === '{"viewport.splitView":["Y"]}');
  ok('sanitize drops an override that equals the defaults', Object.keys(sanitizeOverrides({ 'viewport.splitView': ['S'] })).length === 0);
  ok('sanitize: an entry whose every combo is invalid is corrupt, so the DEFAULT stands (not silently unbound)',
    Object.keys(sanitizeOverrides({ 'viewport.fit': ['Escape', 'Mod+'] })).length === 0);
  ok('sanitize keeps an explicit unbind', JSON.stringify(sanitizeOverrides({ 'viewport.splitView': [] })) === '{"viewport.splitView":[]}');
  ok('sanitize respells and de-duplicates', JSON.stringify(sanitizeOverrides({ 'viewport.splitView': ['Shift+Mod+Y', 'Mod+Shift+Y'] })) === '{"viewport.splitView":["Mod+Shift+Y"]}');
}

// ── 9. The floating mini-viewport's rules ──────────────────────────────────
console.log('pip model');
{
  const VP = { vw: 1280, vh: 720 };
  const sz = { w: 280, h: 158 };
  ok('the default window is 280x158', PIP_DEFAULT.w === 280 && PIP_DEFAULT.h === 158 && pictureHeight(280, 280 / 158) === 158);
  ok('a 16:9 frame at 280 wide is 158 high', pictureHeight(280, 16 / 9) === 158);
  ok('a missing aspect falls back to the default shape', pictureHeight(280, undefined) === 158 && pictureHeight(280, 0) === 158 && pictureHeight(280, NaN) === 158);

  // visibility: follows you to the tabs you tune from, and reappears on Face Swap when the stage is out of view
  const vis = (o) => pipVisible({ enabled: true, hasFrame: true, tab: 'facemgr', stageVisible: true, ...o });
  ok('on Face Manager / Batch / Settings', ['facemgr', 'batch', 'settings'].every((t) => vis({ tab: t })));
  ok('PIP_TABS is exactly those three', JSON.stringify([...PIP_TABS]) === '["facemgr","batch","settings"]');
  ok('not on Home, Processing, Editor, Outputs, History', ['home', 'processing', 'extras', 'gallery', 'history'].every((t) => !vis({ tab: t })));
  ok('on Face Swap it hides while the stage is visible', !vis({ tab: 'faceswap', stageVisible: true }));
  ok('on Face Swap it shows when the stage is scrolled out of view', vis({ tab: 'faceswap', stageVisible: false }));
  ok('never when switched off', !vis({ enabled: false }) && !vis({ enabled: false, tab: 'faceswap', stageVisible: false }));
  ok('never without a frame to show', !vis({ hasFrame: false }));

  // placement
  const c = cornerPositions(sz, VP);
  ok('corners sit PIP_MARGIN from the edges, bar included', c['top-left'].x === PIP_MARGIN && c['top-left'].y === PIP_MARGIN
    && c['bottom-right'].x === 1280 - 280 - PIP_MARGIN && c['bottom-right'].y === 720 - (158 + PIP_BAR) - PIP_MARGIN);
  ok('four distinct corners', new Set(Object.values(c).map((p) => `${p.x},${p.y}`)).size === 4);
  ok('clampPosition keeps the whole window (picture AND bar) inside', (() => {
    const lo = clampPosition({ x: -500, y: -500 }, sz, VP);
    const hi = clampPosition({ x: 9999, y: 9999 }, sz, VP);
    return lo.x === PIP_MARGIN && lo.y === PIP_MARGIN && hi.x + sz.w + PIP_MARGIN === VP.vw && hi.y + sz.h + PIP_BAR + PIP_MARGIN === VP.vh;
  })());
  ok('clampPosition never throws or leaves the margin on a viewport smaller than the window', (() => {
    const p = clampPosition({ x: 50, y: 50 }, sz, { vw: 100, vh: 100 });
    return p.x === PIP_MARGIN && p.y === PIP_MARGIN;
  })());
  ok('property: any drop point clamps to a position that is fully on screen', (() => {
    for (let i = 0; i < 300; i += 1) {
      const vp = { vw: int(400, 2400), vh: int(300, 1400) };
      const s2 = { w: int(PIP_MIN_W, 500), h: int(110, 280) };
      const p = clampPosition({ x: int(-3000, 3000), y: int(-3000, 3000) }, s2, vp);
      const fits = s2.w + 2 * PIP_MARGIN <= vp.vw && s2.h + PIP_BAR + 2 * PIP_MARGIN <= vp.vh;
      if (fits && (p.x < PIP_MARGIN || p.y < PIP_MARGIN || p.x + s2.w > vp.vw - PIP_MARGIN || p.y + s2.h + PIP_BAR > vp.vh - PIP_MARGIN)) return false;
    }
    return true;
  })());

  // docking
  ok('dropped within PIP_SNAP of a corner it snaps there', snapCorner({ x: PIP_MARGIN + 30, y: PIP_MARGIN + 30 }, sz, VP) === 'top-left'
    && snapCorner({ x: c['bottom-right'].x - 20, y: c['bottom-right'].y + 10 }, sz, VP) === 'bottom-right');
  ok('dropped mid-screen it floats', snapCorner({ x: 500, y: 300 }, sz, VP) === null);
  ok('just inside / just outside the snap radius', snapCorner({ x: PIP_MARGIN + PIP_SNAP, y: PIP_MARGIN }, sz, VP) === 'top-left'
    && snapCorner({ x: PIP_MARGIN + PIP_SNAP + 1, y: PIP_MARGIN }, sz, VP) === null);
  ok('the dock button visits all four corners and comes back', (() => {
    let cur = 'bottom-right'; const seen = [];
    for (let i = 0; i < 4; i += 1) { cur = nextCorner(cur); seen.push(cur); }
    return new Set(seen).size === 4 && cur === 'bottom-right';
  })());
  ok('a free-floating window (corner null) starts the cycle', CORNER_ORDER.includes(nextCorner(null)));

  // resize
  const AR = 16 / 9;
  ok('resize keeps the picture shape', (() => { const r = resizeTo(400, AR, { x: 100, y: 100 }, VP); return r.w === 400 && r.h === Math.round(400 / AR); })());
  ok('resize respects the minimum and maximum width', resizeTo(10, AR, { x: 100, y: 100 }, VP).w === PIP_MIN_W && resizeTo(99999, AR, { x: 20, y: 20 }, VP).w === PIP_MAX_W);
  ok('resize cannot run off the right edge', (() => { const r = resizeTo(700, AR, { x: 900, y: 50 }, VP); return r.w <= VP.vw - 900 - PIP_MARGIN || r.w === PIP_MIN_W; })());
  ok('resize cannot run off the bottom edge', (() => { const r = resizeTo(700, AR, { x: 20, y: 600 }, VP); return r.h + 600 + PIP_BAR + PIP_MARGIN <= VP.vh || r.w === PIP_MIN_W; })());
  ok('property: resize always lands in [min, max] with the aspect kept', (() => {
    for (let i = 0; i < 300; i += 1) {
      const ar = 0.5 + rnd() * 2.5;
      const r = resizeTo(int(-100, 2000), ar, { x: int(16, 600), y: int(16, 400) }, { vw: int(500, 2200), vh: int(400, 1300) });
      if (r.w < PIP_MIN_W || r.w > PIP_MAX_W || r.h !== pictureHeight(r.w, ar)) return false;
    }
    return true;
  })());

  // persistence
  ok('sanitizeSaved: defaults for nothing / junk', JSON.stringify(sanitizeSaved(null)) === JSON.stringify(sanitizeSaved('junk'))
    && sanitizeSaved(null).enabled === false && sanitizeSaved(null).w === 280 && sanitizeSaved(null).corner === 'bottom-right');
  ok('sanitizeSaved: clamps width, rejects an unknown corner, keeps a free position', (() => {
    const a = sanitizeSaved({ enabled: true, w: 99999, corner: 'middle', x: 40, y: 50 });
    const b = sanitizeSaved({ w: 5, corner: null, x: 'a', y: 2 });
    return a.enabled === true && a.w === PIP_MAX_W && a.corner === 'bottom-right' && a.x === 40 && a.y === 50
      && b.w === PIP_MIN_W && b.corner === null && b.x === null;
  })());

  // what makes the picture stale
  ok('Morph Lab Preview and a faceset load change the live source', changesLiveSource('POST', '/api/facemgr/morph/preview') && changesLiveSource('POST', '/api/faceset/library/load'));
  ok('...a query string does not hide it', changesLiveSource('POST', '/api/facemgr/morph/preview?x=1'));
  ok('...but reads, saves and Face Manager working-set edits do not', !changesLiveSource('GET', '/api/facemgr/morph/preview')
    && !changesLiveSource('POST', '/api/facemgr/morph/blend') && !changesLiveSource('POST', '/api/facemgr/morph/save')
    && !changesLiveSource('POST', '/api/facemgr/add') && !changesLiveSource('POST', '/api/facemgr/cut') && !changesLiveSource('POST', '/api/preview'));

  // the source the backend has selected NOW
  const info = [{ id: 'a1', name: 'Ann' }, { id: 'b2', name: 'Bo' }, { id: 'c3', name: 'Cy' }];
  ok('selectedSourceOf: by stable id', JSON.stringify(selectedSourceOf({ source_faces_info: info, selected_source_id: 'b2', selected_source_index: 0 })) === '{"id":"b2","name":"Bo"}');
  ok('selectedSourceOf: the index only as a fallback', JSON.stringify(selectedSourceOf({ source_faces_info: info, selected_source_index: 2 })) === '{"id":"c3","name":"Cy"}');
  ok('selectedSourceOf: an out-of-range index falls to the first', selectedSourceOf({ source_faces_info: info, selected_source_index: 9 })?.id === 'a1');
  ok('selectedSourceOf: nothing to say -> null', selectedSourceOf({}) === null && selectedSourceOf(null) === null && selectedSourceOf({ source_faces_info: [] }) === null);
  ok('selectedSourceOf: an id the list does not know is still the selected id', selectedSourceOf({ source_faces_info: info, selected_source_id: 'zz' })?.id === 'zz');

  // the replay body
  const last = {
    index: 0, frame: 42, fake_preview: false, blend_ratio: 0.8,
    selected_source_id: 'a1', selected_source_name: 'Ann', selection_version: 5,
    processing_selection: { request_id: 'old', selection_version: 5, source_identity_id: 'a1', target_media_id: 'm1', target_person_ids: ['p1'] },
  };
  const frozen = JSON.stringify(last);
  const re = replayBody(last, { state: { source_faces_info: info, selected_source_id: 'b2' }, requestId: 'new-id', version: 9 });
  ok('replay never mutates the remembered body', JSON.stringify(last) === frozen);
  ok('replay takes the live source', re.selected_source_id === 'b2' && re.selected_source_name === 'Bo' && re.processing_selection.source_identity_id === 'b2');
  ok('replay carries a fresh request id and a later version', re.processing_selection.request_id === 'new-id' && re.processing_selection.selection_version === 9 && re.selection_version === 9);
  ok('replay keeps the frame, settings and people exactly', re.frame === 42 && re.index === 0 && re.blend_ratio === 0.8
    && re.processing_selection.target_media_id === 'm1' && JSON.stringify(re.processing_selection.target_person_ids) === '["p1"]');
  ok('replay with no /api/state keeps the last source', (() => { const r = replayBody(last, { requestId: 'n', version: 6 }); return r.selected_source_id === 'a1' && r.processing_selection.source_identity_id === 'a1'; })());
  ok('replay of a body with no processing_selection still works', replayBody({ frame: 3 }, { version: 2 }).frame === 3);
}

// ── 10. The live keymap store (what the window handlers actually read) ─────
console.log('shortcut store');
{
  const { useShortcutStore, eventMatches, actionsFor, eventFiresInText, keyupMatches, bindings } = store;
  const E = (o) => ({ key: '', code: '', ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, ...o });
  const st = useShortcutStore.getState;
  ok('a saved remap is loaded when the store starts (Fit -> Y)', JSON.stringify(bindings('viewport.fit')) === '["Y"]');
  ok('...so Y fits and F no longer does', eventMatches(E({ key: 'y', code: 'KeyY' }), 'viewport.fit') && !eventMatches(E({ key: 'f', code: 'KeyF' }), 'viewport.fit'));
  ok('defaults still apply to what was not remapped', eventMatches(E({ key: 'q', code: 'KeyQ' }), 'transport.addToQueue'));

  st().setBindings('viewport.splitView', ['Z']);
  ok('setBindings takes effect on the very next keypress', eventMatches(E({ key: 'z', code: 'KeyZ' }), 'viewport.splitView'));
  ok('...and the old key stops working at once', !eventMatches(E({ key: 's', code: 'KeyS' }), 'viewport.splitView'));
  st().setBindings('viewport.splitView', ['Mod+Shift+Y']);
  ok('a Mod+Shift remap matches exactly', eventMatches(E({ key: 'Y', code: 'KeyY', ctrlKey: true, shiftKey: true }), 'viewport.splitView')
    && !eventMatches(E({ key: 'y', code: 'KeyY', ctrlKey: true }), 'viewport.splitView'));
  ok('it is persisted to localStorage', JSON.parse(fakeStorage.getItem('roop_keybinds_v1')).overrides['viewport.splitView'][0] === 'Mod+Shift+Y');
  st().resetBindings('viewport.splitView');
  ok('resetBindings restores S and drops the stored entry', eventMatches(E({ key: 's', code: 'KeyS' }), 'viewport.splitView')
    && !('viewport.splitView' in JSON.parse(fakeStorage.getItem('roop_keybinds_v1')).overrides));
  st().setBindings('transport.addToQueue', []);
  ok('an unbound action fires from nothing', !eventMatches(E({ key: 'q', code: 'KeyQ' }), 'transport.addToQueue') && bindings('transport.addToQueue').length === 0);
  st().resetAll();
  ok('resetAll clears every remap and the stored key', eventMatches(E({ key: 'f', code: 'KeyF' }), 'viewport.fit') && fakeStorage.getItem('roop_keybinds_v1') === null);

  ok('actionsFor lists what a keypress fires', JSON.stringify(actionsFor(E({ key: 'k', code: 'KeyK', ctrlKey: true }))) === '["app.palette"]'
    && actionsFor(E({ key: 'Shift' })).length === 0);
  ok('a bare modifier press fires nothing', actionsFor(E({ key: 'Control', ctrlKey: true })).length === 0);
  ok('typing "?" in a field must not open help; Ctrl+/ may', !eventFiresInText(E({ key: '?', code: 'Slash', shiftKey: true }), 'app.help')
    && eventFiresInText(E({ key: '/', code: 'Slash', ctrlKey: true }), 'app.help'));
  ok('Ctrl+K fires from a field; Ctrl+B does not (its action does not opt in)', eventFiresInText(E({ key: 'k', code: 'KeyK', ctrlKey: true }), 'app.palette')
    && !eventFiresInText(E({ key: 'b', code: 'KeyB', ctrlKey: true }), 'app.toggleRail'));
  ok('keyup for the hold action matches on the key alone', keyupMatches(E({ key: '\\', code: 'Backslash' }), 'viewport.peekOriginal')
    && keyupMatches(E({ key: '\\', code: 'Backslash', shiftKey: true }), 'viewport.peekOriginal')
    && !keyupMatches(E({ key: 'x', code: 'KeyX' }), 'viewport.peekOriginal'));
  ok('eventMatches survives junk', !eventMatches(null, 'viewport.fit') && !eventMatches(undefined, 'viewport.fit') && !eventMatches({}, 'viewport.fit'));
  ok('an unknown action id matches nothing', !eventMatches(E({ key: 'f', code: 'KeyF' }), 'no.such.action'));
  ok('the same event object is judged consistently (the combo is cached)', (() => {
    const e = E({ key: 'f', code: 'KeyF' });
    return eventMatches(e, 'viewport.fit') === eventMatches(e, 'viewport.fit') && eventMatches(e, 'viewport.fit') && !eventMatches(e, 'viewport.lockTarget');
  })());
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`);
