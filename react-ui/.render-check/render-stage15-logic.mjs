/**
 * Stage 15's pure logic (plain node, no browser): expression maths (relative forms, min/max, the live preview text), the
 * saved-workspace-layout model and its store, the ANSI key remapper's model and hotkey profiles, and the floating preview's
 * picture / transport rules.
 *
 *   node .render-check/run-module.mjs render-stage15-logic.mjs
 *
 * Same discipline as render-stage26-logic: the parser is checked against JavaScript's own arithmetic on generated
 * expressions (so a precedence slip in the new functions cannot hide behind hand-picked examples), and the keyboard is checked
 * against the keymap it draws (every default key must land on a drawn key, or a bound key would be invisible).
 */
import {
  FUNCTIONS, evaluateExpression, normalizeExpression, previewExpression, settleTypedValue,
} from '../src/numeric/mathExpr.js';
import {
  ASPECTS, assignHotkey, classifyAspect, fitsWindow, layoutForSlot, normalizeLayout, normalizeLayouts, orderForAspect,
  renameLayout, scaleSizes, setAspect,
} from '../src/workspace/layoutModel.js';
import {
  ANSI_MAIN, ANSI_NAV, BINDABLE_KEYS, GROUPS, LAYER_BY_ID, actionsOnKey, analyzeKeymap, describeConflicts, groupOfAction,
  keyView, modifierUse, physicalPlacement,
} from '../src/shortcuts/ansiLayout.js';
import { ACTIONS, ACTION_BY_ID, LAYOUT_SLOT_ACTION_IDS } from '../src/shortcuts/actions.js';
import { allConflicts, bindingsOf, exportProfile, parseProfile, profileFileName } from '../src/shortcuts/keymap.js';
import { parseCombo } from '../src/shortcuts/combo.js';
import {
  AB_MODES, PIP_TABS, clampWipe, nextAbMode, nextFrame, pickPicture, pipVisible, stepBody, transportKind,
} from '../src/pip/pipModel.js';

// The workspace store reads storage and the window when it loads, so both exist first.
const memory = new Map();
globalThis.window = {
  innerWidth: 3440,
  innerHeight: 1440,
  localStorage: { getItem: (k) => (memory.has(k) ? memory.get(k) : null), setItem: (k, v) => { memory.set(k, String(v)); }, removeItem: (k) => { memory.delete(k); } },
};
const { useWorkspaceStore, clampPanelSize, DEFAULT_PANEL_SIZES } = await import('../src/store/workspaceStore.js');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;
const val = (text, ctx) => { const r = evaluateExpression(text, ctx); return r.ok ? r.value : `ERR:${r.error}`; };
const ws = () => useWorkspaceStore.getState();

let seed = 15;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const int = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));

// ── 1. Relative forms: the brief's *2, /1.5, +16, -10% ────────────────────
console.log('relative expressions');
ok('*2 doubles the current value (40 -> 80)', val('*2', { current: 40 }) === 80);
ok('/1.5 divides it (45 -> 30)', near(val('/1.5', { current: 45 }), 30));
ok('+16 adds to it (40 -> 56)', val('+16', { current: 40 }) === 56);
ok('-10% scales it down (40 -> 36)', near(val('-10%', { current: 40 }), 36));
ok('+10% still scales it up (40 -> 44)', near(val('+10%', { current: 40 }), 44));
ok('a bare -5 is still the number -5, not "5 below"', val('-5', { current: 40 }) === -5);
ok('the right side of a relative form is any expression: *2+1 -> 40*3', val('*2+1', { current: 40 }) === 120);
ok('+2*8 -> 40 + 16', val('+2*8', { current: 40 }) === 56);
ok('*(1+1) -> 80', val('*(1+1)', { current: 40 }) === 80);
ok('whitespace after the operator is fine: * 2', val('* 2', { current: 40 }) === 80);
ok('*2 on a field holding 0 stays 0 (no NaN)', val('*2', { current: 0 }) === 0);
ok('with no current value, *2 is 0 rather than an error', val('*2') === 0);
ok('relative forms respect the ctx step: integer control rounds', settleTypedValue('/3', { current: 100, min: 0, max: 4096, step: 1 }).value === 33);
ok('relative result clamps to the range and says so', (() => { const r = settleTypedValue('*100', { current: 1000, min: 0, max: 4096, step: 1 }); return r.value === 4096 && r.clamped; })());
ok('v is the current value: v-16 -> 24', val('v-16', { current: 40 }) === 24);
ok('value and current are the same name', val('value*2', { current: 5 }) === 10 && val('current+1', { current: 5 }) === 6);
ok('v with no current value is an error, not 0', typeof val('v+1') === 'string' && String(val('v+1')).startsWith('ERR:'));
ok('percent inside the right side: +10% of v is unchanged (still the scale form)', near(val('+10%', { current: 200 }), 220));
ok('a relative form that divides by zero says so', String(val('/0', { current: 40 })).includes('Division by zero'));
ok('/ then an expression that is zero: /(1-1)', String(val('/(1-1)', { current: 40 })).includes('Division by zero'));

// ── 2. min() / max() and friends ──────────────────────────────────────────
console.log('functions');
ok('min(3,5) = 3, max(3,5) = 5', val('min(3,5)') === 3 && val('max(3,5)') === 5);
ok('min with one value is that value', val('min(7)') === 7);
ok('up to eight values', val('max(1,2,3,4,5,6,7,8)') === 8);
ok('nine values are refused', String(val('max(1,2,3,4,5,6,7,8,9)')).startsWith('ERR:'));
ok('arguments are expressions: min(512/2, 100*3)', val('min(512/2, 100*3)') === 256);
ok('functions nest: max(min(10,4), 2)', val('max(min(10,4), 2)') === 4);
ok('with the current value: min(v*2, 512) on 300 -> 512', val('min(v*2, 512)', { current: 300 }) === 512);
ok('with the current value: max(v/2, 16) on 20 -> 16', val('max(v/2, 16)', { current: 20 }) === 16);
ok('a function is a primary: 2*min(3,4)+1 = 7', val('2*min(3,4)+1') === 7);
ok('function names are case-insensitive', val('MIN(3,5)') === 3 && val('Max(3,5)') === 5);
ok('min(1,5) is a call with two arguments, NOT the decimal comma 1.5', val('min(1,5)') === 1);
ok('a decimal comma still works with no function: 1,5*2 = 3', val('1,5*2') === 3);
ok('decimal comma ISN\'T applied inside a call: max(1,5) = 5', val('max(1,5)') === 5);
ok('abs / round / floor / ceil', val('abs(-4)') === 4 && val('round(2.6)') === 3 && val('floor(2.9)') === 2 && val('ceil(2.1)') === 3);
ok('clamp(x, lo, hi)', val('clamp(9, 0, 5)') === 5 && val('clamp(-3, 0, 5)') === 0 && val('clamp(3, 0, 5)') === 3);
ok('a relative form can start a function: *min(2,3)', val('*min(2,3)', { current: 10 }) === 20);
ok('FUNCTIONS is the closed list', Object.keys(FUNCTIONS).sort().join() === 'abs,ceil,clamp,floor,max,min,round');
const refuses = (text, why) => ok(`refuses ${JSON.stringify(text)} (${why})`, !evaluateExpression(text, { current: 1 }).ok);
refuses('alert(1)', 'unknown function');
refuses('constructor(1)', 'inherited name is not a function');
refuses('toString(1)', 'inherited name is not a function');
refuses('__proto__(1)', 'inherited name is not a function');
refuses('min()', 'no arguments');
refuses('min(1,)', 'dangling comma');
refuses('min(,1)', 'leading comma');
refuses('min(1', 'unclosed');
refuses('min 1', 'no parentheses');
refuses('clamp(1,2)', 'wrong arity');
refuses('abs(1,2)', 'wrong arity');
refuses('1, 2', 'a spaced comma outside a call (1,2 is the decimal comma 1.2)');
refuses('(1, 2)', 'a comma in plain parentheses');
refuses('*', 'a lone operator');
refuses('**2', 'two leading operators');
refuses('+*2', 'operator pair');
refuses('Math.max(1,2)', 'member access');
ok('50 nested calls fail by message, not by stack overflow', (() => {
  const text = `${'min('.repeat(50)}1${')'.repeat(50)}`;
  try { return !evaluateExpression(text).ok; } catch { return false; }
})());
ok('normalizeExpression keeps the comma in a call', normalizeExpression('min(1,5)') === 'min(1,5)' && normalizeExpression('1,5') === '1.5');

// ── 3. Oracle: min/max/relative agree with JavaScript ────────────────────
console.log('oracle');
{
  let bad = null;
  let checked = 0;
  const num = () => (int(1, 90) / int(1, 9)).toFixed(2);
  const gen = (depth) => {
    if (depth === 0 || rnd() < 0.3) return num();
    const roll = rnd();
    if (roll < 0.25) return `min(${gen(depth - 1)},${gen(depth - 1)})`;
    if (roll < 0.5) return `max(${gen(depth - 1)},${gen(depth - 1)},${gen(depth - 1)})`;
    const op = ['+', '-', '*'][int(0, 2)];
    const text = `${gen(depth - 1)}${op}${gen(depth - 1)}`;
    return rnd() < 0.4 ? `(${text})` : text;
  };
  for (let i = 0; i < 1500 && !bad; i += 1) {
    const text = gen(2);
    if (text.length > 100) continue;                  // the parser caps expressions at 120 characters; a longer one is a different test
    // eslint-disable-next-line no-new-func -- the test ORACLE: trusted generated text, never user input
    const want = new Function('min', 'max', `return (${text});`)(Math.min, Math.max);
    const got = evaluateExpression(text);
    if (!got.ok) bad = `${text}: parser said ${got.error}`;
    else if (Math.abs(got.value - want) > 1e-9 * Math.max(1, Math.abs(want))) bad = `${text}: oracle ${want}, parser ${got.value}`;
    checked += 1;
  }
  ok(`generated min/max expressions agree with JavaScript (${checked} run)`, !bad && checked > 150, bad || `only ${checked} fit under the length cap`);
  let rbad = null;
  for (let i = 0; i < 300 && !rbad; i += 1) {
    const cur = int(1, 900) / 4;
    const k = (int(1, 90) / int(1, 9)).toFixed(2);
    const op = ['*', '/', '+'][int(0, 2)];
    const got = evaluateExpression(`${op}${k}`, { current: cur });
    const want = op === '*' ? cur * Number(k) : op === '/' ? cur / Number(k) : cur + Number(k);
    if (!got.ok || Math.abs(got.value - want) > 1e-9 * Math.max(1, Math.abs(want))) rbad = `${op}${k} on ${cur}: want ${want}, got ${JSON.stringify(got)}`;
  }
  ok('300 relative forms agree with current <op> k', !rbad, rbad || '');
}

// ── 4. The live preview pill's text ────────────────────────────────────────
console.log('preview');
{
  const opts = { current: 40, min: 0, max: 4096, step: 1 };
  ok('a plain number gets no pill', previewExpression('256', opts) === null && previewExpression('-5', opts) === null && previewExpression('0.5', opts) === null);
  ok('empty gets no pill', previewExpression('', opts) === null && previewExpression('   ', opts) === null);
  const half = previewExpression('512/2', opts);
  ok('512/2 -> "256"', half?.ok && half.formula === '512/2' && half.text === '256', JSON.stringify(half));
  const dbl = previewExpression('*2', opts);
  ok('*2 on 40 -> "80"', dbl?.ok && dbl.text === '80' && dbl.formula === '*2', JSON.stringify(dbl));
  ok('/1.5 on an integer control rounds like the field will (40 -> 27)', previewExpression('/1.5', opts)?.text === '27');
  ok('-10% on 40 -> "36"', previewExpression('-10%', opts)?.text === '36');
  const hi = previewExpression('*1000', opts);
  ok('an out-of-range result says (max) and is clamped', hi?.ok && hi.clamped && hi.text === '4096 (max)', JSON.stringify(hi));
  const lo = previewExpression('min(-5, -6)', opts);
  ok('and (min) at the bottom', lo?.ok && lo.clamped && lo.text === '0 (min)', JSON.stringify(lo));
  const bad = previewExpression('min(1,', opts);
  ok('an unfinished expression is an error pill, not nothing', bad && bad.ok === false && typeof bad.error === 'string' && bad.error.length > 0, JSON.stringify(bad));
  ok('the pill and the commit agree for every example (same settleTypedValue)', ['512/2', '*2', '/1.5', '+16', '-10%', 'min(v*2, 64)', 'max(v/2, 8)', '50%'].every((t) => {
    const p = previewExpression(t, opts);
    const s = settleTypedValue(t, opts);
    return p?.ok === s.ok && (!s.ok || p.result === s.value);
  }));
  ok('fractional controls keep 4 decimals in the pill', previewExpression('1/3', { current: 0, min: 0, max: 1, step: 0.01 })?.text === '0.3333');
  ok('typographic operators preview too', previewExpression('6 × 7', opts)?.text === '42');
}

// ── 5. Monitor aspect classes ──────────────────────────────────────────────
console.log('aspect');
{
  const cases = [
    [1920, 1080, '16:9'], [2560, 1440, '16:9'], [3840, 2160, '16:9'], [1366, 768, '16:9'], [1920, 1200, '16:9'], [1280, 1024, '16:9'],
    [2560, 1080, '21:9'], [3440, 1440, '21:9'], [3840, 1600, '21:9'], [2560, 1080 - 40, '21:9'],
    [3840, 1080, '32:9'], [5120, 1440, '32:9'], [5120, 1440 - 80, '32:9'],
  ];
  for (const [w, h, want] of cases) ok(`${w}x${h} is ${want}`, classifyAspect(w, h) === want, `got ${classifyAspect(w, h)}`);
  ok('an unmeasured window is 16:9, not an error', classifyAspect(0, 0) === '16:9' && classifyAspect(NaN, 10) === '16:9' && classifyAspect(undefined, undefined) === '16:9');
  ok('the three classes are the three monitors', ASPECTS.join() === '16:9,21:9,32:9');
}

// ── 6. Layout model: normalise, hotkeys, rename, order, scale ──────────────
console.log('layout model');
{
  const L = (id, name, extra = {}) => ({ id, name, label: name, mode: 'default', sizes: { leftWidth: 300, rightWidth: 340, bottomHeight: 260 }, aspect: 'any', hotkey: null, viewport: null, ...extra });
  const clampTo = (panel, v) => Math.min(600, Math.max(100, v));
  ok('a pre-Stage-15 layout (no aspect / hotkey / viewport) normalises to any / null / null', (() => {
    const n = normalizeLayout({ id: 'custom_1', name: 'Old', label: 'Old', mode: 'default', sizes: { leftWidth: 300, rightWidth: 340, bottomHeight: 260 } }, clampTo);
    return n && n.aspect === 'any' && n.hotkey === null && n.viewport === null && n.sizes.leftWidth === 300;
  })());
  ok('junk is dropped: no id, no name, not an object', normalizeLayout({ name: 'x' }) === null && normalizeLayout({ id: 'a' }) === null && normalizeLayout(null) === null && normalizeLayout('x') === null);
  ok('an unknown aspect becomes any; a bad hotkey becomes null', (() => { const n = normalizeLayout({ id: 'a', name: 'n', aspect: '4:3', hotkey: 9 }); return n.aspect === 'any' && n.hotkey === null; })());
  ok('sizes pass through the store\'s clamp', normalizeLayout({ id: 'a', name: 'n', sizes: { leftWidth: 9999 } }, clampTo).sizes.leftWidth === 600);
  ok('a non-finite size is dropped, not stored as NaN', !('rightWidth' in normalizeLayout({ id: 'a', name: 'n', sizes: { rightWidth: 'wide' } }).sizes));
  ok('a duplicated id and a slot claimed twice keep the first', (() => {
    const out = normalizeLayouts([L('a', 'A', { hotkey: 2 }), L('b', 'B', { hotkey: 2 }), L('a', 'A again')]);
    return out.length === 2 && out[0].hotkey === 2 && out[1].hotkey === null;
  })());
  ok('not an array -> []', normalizeLayouts(undefined).length === 0 && normalizeLayouts({}).length === 0);

  const three = [L('a', 'A'), L('b', 'B'), L('c', 'C')];
  const bound = assignHotkey(three, 'b', 3);
  ok('binding a slot sets it', bound.find((l) => l.id === 'b').hotkey === 3);
  ok('a slot moves: binding 3 to c takes it off b', assignHotkey(bound, 'c', 3).filter((l) => l.hotkey === 3).map((l) => l.id).join() === 'c');
  ok('binding null unbinds', assignHotkey(bound, 'b', null).every((l) => l.hotkey === null));
  ok('slots are 1..5 only', assignHotkey(three, 'a', 6) === three && assignHotkey(three, 'a', 0) === three && assignHotkey(three, 'a', 1.5) === three);
  ok('an unknown layout id changes nothing', assignHotkey(three, 'zzz', 2) === three);
  ok('layoutForSlot finds it, and null for an empty slot', layoutForSlot(bound, 3)?.id === 'b' && layoutForSlot(bound, 4) === null);

  ok('rename works', renameLayout(three, 'a', '  Roto  ').layouts.find((l) => l.id === 'a').name === 'Roto');
  ok('rename keeps label in step with name', renameLayout(three, 'a', 'Roto').layouts.find((l) => l.id === 'a').label === 'Roto');
  ok('rename to the same name (other case) as ANOTHER layout is refused', renameLayout(three, 'a', 'b').ok === false);
  ok('rename to its own name in another case is allowed', renameLayout(three, 'a', 'a').ok === true);
  ok('an empty name is refused', renameLayout(three, 'a', '   ').ok === false && renameLayout(three, 'a', '').error.length > 0);
  ok('renaming a layout that is not there is refused', renameLayout(three, 'nope', 'x').ok === false);
  ok('a name is capped at 40 characters', renameLayout(three, 'a', 'x'.repeat(100)).layouts[0].name.length === 40);
  ok('setAspect only takes the known shapes', setAspect(three, 'a', '21:9')[0].aspect === '21:9' && setAspect(three, 'a', 'any')[0].aspect === 'any' && setAspect(three, 'a', '4:3') === three);

  const mixed = [L('p', 'plain', { aspect: '16:9' }), L('u', 'ultra', { aspect: '21:9' }), L('x', 'anyone'), L('s', 'super', { aspect: '32:9' }), L('u2', 'ultra2', { aspect: '21:9' })];
  ok('on 21:9: this shape first (saved order), then any, then the rest', orderForAspect(mixed, '21:9').map((l) => l.id).join() === 'u,u2,x,p,s');
  ok('on 32:9 the 32:9 layout leads', orderForAspect(mixed, '32:9')[0].id === 's');
  ok('nothing is hidden by the ordering', orderForAspect(mixed, '16:9').length === mixed.length);

  const sizes = { leftWidth: 460, rightWidth: 400, bottomHeight: 300, queueWidth: 300 };
  const big = { w: 3440, h: 1440 };
  const small = { w: 2560, h: 1080 };
  const scaled = scaleSizes(sizes, big, small, clampTo);
  ok('3440 -> 2560 keeps the proportions: left 460 -> 342', scaled.leftWidth === Math.round(460 * 2560 / 3440), `got ${scaled.leftWidth}`);
  ok('the timeline scales with the HEIGHT: 300 -> 225', scaled.bottomHeight === Math.round(300 * 1080 / 1440));
  ok('a window a few % off is the same monitor: sizes untouched', scaleSizes(sizes, big, { w: 3440 - 50, h: 1440 - 30 }, clampTo) === sizes);
  ok('no viewport saved -> sizes as saved', scaleSizes(sizes, null, small, clampTo) === sizes && scaleSizes(sizes, big, null, clampTo) === sizes);
  ok('a nonsense viewport changes nothing', scaleSizes(sizes, { w: 0, h: 0 }, small, clampTo) === sizes);
  ok('the scaled result is clamped', scaleSizes({ leftWidth: 500 }, { w: 1000, h: 1000 }, { w: 5000, h: 1000 }, clampTo).leftWidth === 600);
  ok('fitsWindow: 1920 wide holds 460+400 with 1060 left', (() => { const f = fitsWindow(sizes, 1920); return f.fits && f.centre === 1920 - 864; })());
  ok('fitsWindow: 1280 wide with 620+580 does not', fitsWindow({ leftWidth: 620, rightWidth: 580 }, 1280).fits === false);
  ok('fitsWindow counts the queue pane only when asked', fitsWindow(sizes, 2000, { queue: true }).centre === fitsWindow(sizes, 2000).centre - 300);
}

// ── 7. The workspace store: save, rename, bind, switch, scale ───────────────
console.log('workspace store');
{
  const s0 = ws();
  ok('the store starts with no saved layouts', s0.customLayouts.length === 0);
  ws().setPanelSize('leftWidth', 440);
  ws().setPanelSize('rightWidth', 380);
  ws().setPanelSize('bottomHeight', 300);
  const roto = ws().saveCustomLayout('Roto 21:9');
  ok('save returns the layout, tagged with this window\'s shape (3440x1440 = 21:9)', roto && roto.aspect === '21:9' && roto.viewport.w === 3440 && roto.viewport.h === 1440);
  ok('the saved sizes are the panels at save time', roto.sizes.leftWidth === 440 && roto.sizes.rightWidth === 380 && roto.sizes.bottomHeight === 300);
  ok('it becomes the active layout', ws().activeLayoutId === roto.id);
  ok('it was persisted', JSON.parse(memory.get('roop_workspace')).customLayouts.some((l) => l.id === roto.id && l.aspect === '21:9'));

  ws().setPanelSize('leftWidth', 300);
  const wide = ws().saveCustomLayout('Super 32:9', { aspect: '32:9', hotkey: 2 });
  ok('an explicit aspect and hotkey are kept', wide.aspect === '32:9' && wide.hotkey === 2);

  ok('binding slot 2 to Roto moves it off Super', (() => {
    ws().bindLayoutHotkey(roto.id, 2);
    const ls = ws().customLayouts;
    return ls.find((l) => l.id === roto.id).hotkey === 2 && ls.find((l) => l.id === wide.id).hotkey === null;
  })());
  ok('saving under an existing name replaces it and KEEPS its slot', (() => {
    const again = ws().saveCustomLayout('roto 21:9');
    return again.id === roto.id && again.hotkey === 2 && ws().customLayouts.filter((l) => l.name.toLowerCase() === 'roto 21:9').length === 1;
  })());
  // That save captured the panels as they were (left 300); put Roto's proportions back for the checks below.
  ws().setPanelSize('leftWidth', 440);
  ws().setPanelSize('rightWidth', 380);
  ws().overwriteCustomLayout(roto.id);
  ok('rename through the store: refused names return the error', ws().renameCustomLayout(wide.id, 'Roto 21:9').ok === false && ws().renameCustomLayout(wide.id, 'Super').ok === true);
  ok('the rename shows in the list', ws().customLayouts.find((l) => l.id === wide.id).name === 'Super');

  ws().setPanelSize('leftWidth', 240);
  ok('switching to a slot applies that layout\'s sizes (Roto: left 440)', (() => {
    const applied = ws().applyLayoutSlot(2);
    return applied?.id === roto.id && ws().panelSizes.leftWidth === 440 && ws().activeLayoutId === roto.id;
  })());
  ok('an empty slot applies nothing and says null', ws().applyLayoutSlot(5) === null);

  // The same layout on a smaller monitor of the same shape: proportions, not pixels.
  globalThis.window.innerWidth = 2560;
  globalThis.window.innerHeight = 1080;
  ws().applyLayout(roto.id);
  ok('on a 2560x1080 window the 3440-wide layout keeps its proportions (440 -> 328)', ws().panelSizes.leftWidth === clampPanelSize('leftWidth', Math.round(440 * 2560 / 3440)), `got ${ws().panelSizes.leftWidth}`);
  globalThis.window.innerWidth = 3440 - 40;
  globalThis.window.innerHeight = 1440 - 20;
  ws().applyLayout(roto.id);
  ok('a window a few % smaller keeps the saved pixels', ws().panelSizes.leftWidth === 440, `got ${ws().panelSizes.leftWidth}`);
  globalThis.window.innerWidth = 3440;
  globalThis.window.innerHeight = 1440;

  ws().setPanelSize('rightWidth', 500);
  ws().overwriteCustomLayout(roto.id);
  ok('"update from current panels" replaces the sizes and keeps name, slot and shape', (() => {
    const l = ws().customLayouts.find((x) => x.id === roto.id);
    return l.sizes.rightWidth === 500 && l.name.toLowerCase() === 'roto 21:9' && l.hotkey === 2 && l.aspect === '21:9';
  })());
  ok('setLayoutAspect changes the monitor shape', (() => { ws().setLayoutAspect(roto.id, '16:9'); return ws().customLayouts.find((l) => l.id === roto.id).aspect === '16:9'; })());
  ok('deleting the active layout falls back to standard', (() => { ws().applyLayout(roto.id); ws().deleteCustomLayout(roto.id); return ws().activeLayoutId === 'standard'; })());
  ok('presets still apply as written (no scaling)', (() => { ws().applyLayout('roto'); return ws().panelSizes.leftWidth === 440; })());
  ok('DEFAULT_PANEL_SIZES is untouched by any of this', DEFAULT_PANEL_SIZES.leftWidth === 260 && DEFAULT_PANEL_SIZES.rightWidth === 340);
}

// ── 8. Workspace-layout hotkey actions in the keymap ───────────────────────
console.log('hotkey actions');
{
  ok('five slot actions exist and are in slot order', LAYOUT_SLOT_ACTION_IDS.length === 5 && LAYOUT_SLOT_ACTION_IDS.every((id, i) => ACTION_BY_ID[id] && bindingsOf(id).join() === `Alt+${i + 1}`));
  ok('they are global-scope workspace actions', LAYOUT_SLOT_ACTION_IDS.every((id) => ACTION_BY_ID[id].scope === 'global' && ACTION_BY_ID[id].category === 'Workspace'));
  ok('the default keymap has no conflicts (Alt+1..5 clash with nothing)', allConflicts({}).length === 0, JSON.stringify(allConflicts({})));
  ok('every default combo parses', ACTIONS.every((a) => a.keys.every((c) => parseCombo(c))));
}

// ── 9. The ANSI keyboard ───────────────────────────────────────────────────
console.log('ansi keyboard');
{
  const tokens = [...ANSI_MAIN, ...ANSI_NAV].flat().filter((k) => k.t);
  ok('the layout draws a full ANSI main block: 6 rows, 15 units wide in the five typing rows', ANSI_MAIN.length === 6
    && ANSI_MAIN.slice(1).every((row) => Math.abs(row.reduce((n, k) => n + (k.gap ?? k.w), 0) - 15) < 1e-9));
  ok('bindable tokens are unique', new Set(BINDABLE_KEYS).size === BINDABLE_KEYS.length);
  ok('modifier keys are drawn but not bindable', tokens.filter((k) => k.mod).length === 6 && BINDABLE_KEYS.every((t) => !String(t).startsWith('mod:')));
  ok('every letter, digit and arrow is a bindable key', [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'].every((c) => BINDABLE_KEYS.includes(c)) && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Space', 'Escape'].every((t) => BINDABLE_KEYS.includes(t)));

  // The invariant that matters: a default binding the keyboard cannot show is a binding the user cannot see.
  const analysis = analyzeKeymap({});
  const missing = [];
  for (const a of ACTIONS) {
    for (const c of a.keys) {
      const p = parseCombo(c);
      const at = physicalPlacement(p);
      if (!BINDABLE_KEYS.includes(at.key)) missing.push(`${a.id}:${c}`);
    }
  }
  ok('every default combo lands on a drawn key', missing.length === 0, missing.join(', '));
  ok('shifted symbols are placed on their physical key with Shift (? -> /, + -> =)', (() => {
    const q = physicalPlacement(parseCombo('?'));
    const p = physicalPlacement(parseCombo('Mod++'));
    return q.key === '/' && q.mods.join() === 'Shift' && p.key === '=' && p.mods.join() === 'Mod,Shift';
  })());
  ok('a plain letter is not moved', (() => { const p = physicalPlacement(parseCombo('Alt+C')); return p.key === 'C' && p.mods.join() === 'Alt'; })());

  const alt = LAYER_BY_ID.alt;
  ok('the Alt layer lights 1-5 as Workspace', ['1', '2', '3', '4', '5'].every((n) => { const v = keyView(n, analysis, alt); return v.bound && v.primary === 'workspace' && v.actions.some((a) => a.id === `workspace.layout${n}`); }));
  ok('the plain layer\'s 1-4 are Snapshots, not the Alt layouts', ['1', '2', '3', '4'].every((n) => keyView(n, analysis).groups.includes('snapshots') && !keyView(n, analysis).actions.some((a) => a.id.startsWith('workspace.layout'))));
  ok('the plain layer\'s Space is Transport', keyView('Space', analysis).primary === 'transport');
  ok('B is Masking in the plain layer; the flow bar\'s bypass B also lives there as Pipeline Soloing', (() => {
    const v = keyView('B', analysis);
    return v.primary === 'masking' && v.groups.includes('soloing');
  })());
  ok('plain C is Viewport; Alt+C (curve editor) is Workspace & Timeline in the Alt layer', keyView('C', analysis).primary === 'viewport' && keyView('C', analysis, alt).primary === 'workspace');
  ok('Shift+C (compare) is in the Shift layer, not the plain one', keyView('C', analysis, LAYER_BY_ID.shift).bound && !keyView('C', analysis).actions.some((a) => a.id === 'viewport.compare'));
  ok('? is the Shift layer\'s /', keyView('/', analysis, LAYER_BY_ID.shift).actions.some((a) => a.id === 'app.help'));
  ok('Ctrl+K is the palette in the Ctrl layer', keyView('K', analysis, LAYER_BY_ID.ctrl).actions.some((a) => a.id === 'app.palette'));
  ok('an unbound key reports not bound and points at other layers', (() => { const v = keyView('Q', analysis, alt); return !v.bound && v.elsewhere >= 1; })());
  ok('the default keymap shows no conflict on any key', BINDABLE_KEYS.every((t) => !keyView(t, analysis).conflict));
  ok('the layer tabs count lit keys per layer', analysis.layerCount.plain > 20 && analysis.layerCount.alt >= 7 && analysis.layerCount.ctrl >= 8);
  ok('the modifier keys light when bindings use them', (() => { const u = modifierUse(analysis); return u.Mod > 8 && u.Alt >= 7 && u.Shift > 8; })());

  const groupSeen = new Set(ACTIONS.map((a) => groupOfAction(a)));
  ok('every family has at least one default action', GROUPS.every((g) => groupSeen.has(g.id)), [...groupSeen].join());
  ok('flow and sweep scopes are Pipeline Soloing', groupOfAction(ACTION_BY_ID['flow.solo1']) === 'soloing' && groupOfAction(ACTION_BY_ID['sweep.cell']) === 'soloing');
  ok('Transport keeps its verbs; transport.run is Workspace (it is a Run button, not a transport)', groupOfAction(ACTION_BY_ID['transport.playPause']) === 'transport' && groupOfAction(ACTION_BY_ID['transport.run']) === 'workspace');
  ok('Mask & Roto is Masking, Snapshots is Snapshots', groupOfAction(ACTION_BY_ID['mask.brush']) === 'masking' && groupOfAction(ACTION_BY_ID['snapshot.slotA']) === 'snapshots');
  ok('an unknown action is Workspace, not a throw', groupOfAction(undefined) === 'workspace');

  // Conflicts, live: rebind two same-scope actions to one key.
  const clash = analyzeKeymap({ 'viewport.fit': ['S'] });          // S is viewport.splitView on the same tab
  ok('a remap that doubles a key lights it as a conflict', keyView('S', clash).conflict === true);
  ok('and lists both actions', describeConflicts(clash).some((c) => c.combo === 'S' && c.labels.length >= 2), JSON.stringify(describeConflicts(clash)));
  ok('actionsOnKey lists everything on S, any modifier', actionsOnKey('S', clash).map((a) => a.id).includes('viewport.fit') && actionsOnKey('S', clash).map((a) => a.id).includes('viewport.splitView'));
  ok('scopes that cannot be live together are NOT a conflict (monitor F vs faceswap F)', allConflicts({}).every((c) => c.combo !== 'F'));
  ok('fixing the clash clears the red', keyView('S', analyzeKeymap({ 'viewport.fit': ['Y'] })).conflict === false);
}

// ── 10. Hotkey profiles: export / import ───────────────────────────────────
console.log('profiles');
{
  const mine = { 'viewport.fit': ['Y'], 'viewport.splitView': [] };
  const file = exportProfile(mine, 'Roto desk', new Date('2026-11-19T00:00:00Z'));
  ok('an export names its format, version, name and time', file.format === 'roop-ultimate-keymap' && file.v === 1 && file.name === 'Roto desk' && file.exportedAt === '2026-11-19T00:00:00.000Z');
  ok('an export carries the overrides (an emptied action stays emptied)', file.overrides['viewport.fit'].join() === 'Y' && Array.isArray(file.overrides['viewport.splitView']) && file.overrides['viewport.splitView'].length === 0);
  const back = parseProfile(JSON.stringify(file));
  ok('export -> import round-trips', back.ok && back.name === 'Roto desk' && JSON.stringify(back.overrides) === JSON.stringify(file.overrides) && back.ignored === 0);
  ok('an empty keymap exports and imports as empty', (() => { const r = parseProfile(JSON.stringify(exportProfile({}, 'Defaults'))); return r.ok && Object.keys(r.overrides).length === 0; })());
  ok('a blank name falls back', exportProfile({}, '   ').name === 'My keymap');
  const garbage = (t) => parseProfile(t);
  ok('not JSON is refused with a reason', garbage('{nope').ok === false && garbage('{nope').error.includes('JSON'));
  ok('empty / non-string is refused', garbage('').ok === false && garbage(undefined).ok === false && garbage('   ').ok === false);
  ok('JSON that is not a keymap is refused', garbage('{"a":1}').ok === false && garbage('[]').ok === false && garbage('null').ok === false && garbage('"x"').ok === false);
  ok('a future version is refused by number', (() => { const r = parseProfile(JSON.stringify({ ...file, v: 2 })); return !r.ok && r.error.includes('2'); })());
  ok('a profile with no overrides object is refused', parseProfile(JSON.stringify({ format: 'roop-ultimate-keymap', v: 1 })).ok === false && parseProfile(JSON.stringify({ format: 'roop-ultimate-keymap', v: 1, overrides: [] })).ok === false);
  ok('an oversized file is refused', parseProfile(' '.repeat(300 * 1024)).ok === false && parseProfile(`{"format":"roop-ultimate-keymap","v":1,"overrides":{},"pad":"${'x'.repeat(300 * 1024)}"}`).ok === false);
  const dirty = parseProfile(JSON.stringify({
    format: 'roop-ultimate-keymap', v: 1, name: 'x',
    overrides: { 'viewport.fit': ['Y'], 'no.such.action': ['Q'], 'viewport.zoomIn': ['Mod+W'], 'viewport.loupe': 'Z', 'viewport.compare': ['Shift+C'] },
  }));
  ok('unknown actions and reserved/invalid entries are ignored and COUNTED', dirty.ok && dirty.ignored === 3, `ignored ${dirty.ignored}`);
  ok('...and the good entry survives; the default-equal one is not an override', dirty.ok && dirty.overrides['viewport.fit']?.join() === 'Y' && !('viewport.compare' in dirty.overrides) && !('no.such.action' in dirty.overrides));
  ok('a profile with a prototype-pollution key does not pollute', (() => {
    const r = parseProfile('{"format":"roop-ultimate-keymap","v":1,"overrides":{"__proto__":["Q"],"constructor":["Q"]}}');
    return r.ok && r.ignored === 2 && Object.keys(r.overrides).length === 0 && ({}).Q === undefined;
  })());
  ok('importing a keymap that clashes is allowed, and the conflict is visible', (() => {
    const r = parseProfile(JSON.stringify(exportProfile({ 'viewport.fit': ['S'] }, 'Clashy')));
    return r.ok && allConflicts(r.overrides).length > 0;
  })());
  ok('file names are slugs', profileFileName('Roto desk #2!') === 'roop-keymap-roto-desk-2.json' && profileFileName('') === 'roop-keymap-keymap.json' && profileFileName('../../x') === 'roop-keymap-x.json');
}

// ── 11. The floating preview: picture, transport, tabs ─────────────────────
console.log('floating preview');
{
  ok('it follows you to Processing as well as Batch / Face Manager / Settings', ['batch', 'processing', 'facemgr', 'settings'].every((t) => PIP_TABS.includes(t)));
  ok('it does not follow you to Home / Outputs / History', ['home', 'gallery', 'history'].every((t) => !PIP_TABS.includes(t)));
  ok('visible on Processing with a frame, hidden without one', pipVisible({ enabled: true, hasFrame: true, tab: 'processing', stageVisible: true }) && !pipVisible({ enabled: true, hasFrame: false, tab: 'processing', stageVisible: true }));
  ok('on Face Swap it shows only once the stage has scrolled away', pipVisible({ enabled: true, hasFrame: true, tab: 'faceswap', stageVisible: false }) && !pipVisible({ enabled: true, hasFrame: true, tab: 'faceswap', stageVisible: true }));
  ok('off means off', !pipVisible({ enabled: false, hasFrame: true, tab: 'processing', stageVisible: false }));

  ok('A/B cycles After -> A | B -> Before -> After', nextAbMode('after') === 'wipe' && nextAbMode('wipe') === 'before' && nextAbMode('before') === 'after' && AB_MODES.length === 3);
  ok('an unknown mode cycles to the first real one, not undefined', AB_MODES.includes(nextAbMode('???')));
  ok('the wipe clamps to 0..100 and survives NaN', clampWipe(-5) === 0 && clampWipe(130) === 100 && clampWipe(NaN) === 50 && clampWipe(37) === 37);

  ok('a running render shows its live frames, and A/B is off', (() => { const p = pickPicture({ running: true, hasLive: true, hasAfter: true, hasBase: true, view: 'wipe' }); return p.kind === 'live' && !p.ab; })());
  ok('running with no live frame yet falls back to the still', pickPicture({ running: true, hasLive: false, hasAfter: true, hasBase: false, view: 'after' }).kind === 'after');
  ok('with a base, the chosen view is honoured and A/B is on', ['after', 'wipe', 'before'].every((v) => { const p = pickPicture({ running: false, hasLive: false, hasAfter: true, hasBase: true, view: v }); return p.kind === v && p.ab; }));
  ok('without a base, wipe and before degrade to after instead of painting nothing', ['wipe', 'before'].every((v) => { const p = pickPicture({ running: false, hasLive: false, hasAfter: true, hasBase: false, view: v }); return p.kind === 'after' && !p.ab; }));
  ok('nothing at all -> none', pickPicture({ running: false, hasLive: false, hasAfter: false, hasBase: false, view: 'after' }).kind === 'none');
  ok('a live frame with no still still shows (the window can open on the first live frame)', pickPicture({ running: true, hasLive: true, hasAfter: false, hasBase: false, view: 'after' }).kind === 'live');

  ok('transport: a render freezes the live view', transportKind({ running: true, faceswapMounted: true, canMounted: true, canStep: true }) === 'live');
  ok('transport: with Face Swap mounted it is Face Swap\'s own', transportKind({ running: false, faceswapMounted: true, canMounted: true, canStep: true }) === 'mounted');
  ok('transport: mounted but with nothing to play is none (not the stepper: Face Swap owns previews)', transportKind({ running: false, faceswapMounted: true, canMounted: false, canStep: true }) === 'none');
  ok('transport: unmounted steps itself when there is a request and frames left', transportKind({ running: false, faceswapMounted: false, canMounted: false, canStep: true }) === 'stepper');
  ok('transport: a still is none', transportKind({ running: false, faceswapMounted: false, canMounted: false, canStep: false }) === 'none');

  ok('nextFrame steps and stops at the end', nextFrame(1, 10) === 2 && nextFrame(9, 10) === 10 && nextFrame(10, 10) === null);
  ok('a single frame / unknown length has no next', nextFrame(1, 1) === null && nextFrame(1, null) === null && nextFrame(NaN, 10) === null);
  ok('nextFrame accepts a step', nextFrame(5, 100, 10) === 15 && nextFrame(95, 100, 10) === null);
  ok('stepBody moves the frame and never mutates the remembered request', (() => {
    const last = { frame: 3, index: 0, processing_selection: { request_id: 'a' } };
    const b = stepBody(last, 4);
    return b.frame === 4 && last.frame === 3 && b.processing_selection === last.processing_selection;
  })());
}

console.log(failures === 0 ? '\nall checks passed' : `\n${failures} FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
