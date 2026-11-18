/**
 * UI overhaul Stage 9 (hotkeys, palette, cheat-sheet): the comparison cycle, the palette ranking, and the virtual
 * keyboard's mapping from the live keymap. The shuttle ladder is pinned by workstation-shortcuts-check.mjs.
 *
 *   node .render-check/run-module.mjs render-ui-overhaul-stage9.mjs
 */
import { COMPARE_STEPS, compareLabel, nextCompareIndex } from '../src/components/faceswap/compareCycle.js';
import { groupBySection, rankCommands, scoreCommand, scoreText } from '../src/components/commandPaletteModel.js';
import {
  DRAWN_KEYS, KEY_ROWS, bindingsByKey, boundCount, keyState, tokenOfEvent,
} from '../src/shortcuts/keyboardModel.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ── C cycles the four comparison modes ───────────────────────────────────────
console.log('compare cycle');
{
  ok('four steps, in the order the requirement names', eq(COMPARE_STEPS.map((s) => s.id), ['wipe', 'crossfade', 'diff', 'side']));
  ok('wipe is the split-wipe slider, crossfade the blend, diff the diff', COMPARE_STEPS[0].sub === 'slider' && COMPARE_STEPS[1].sub === 'blend' && COMPARE_STEPS[2].sub === 'diff');
  ok('only side-by-side turns the split view on', COMPARE_STEPS.filter((s) => s.split).map((s) => s.id).join() === 'side');
  ok('compare off: the first press shows the first mode', nextCompareIndex(-1, false) === 0 && nextCompareIndex(2, false) === 0);
  ok('compare on, no known step: the first press is the first mode', nextCompareIndex(-1, true) === 0);
  let i = 0;
  const seen = [];
  for (let n = 0; n < 5; n += 1) { i = nextCompareIndex(i, true); seen.push(i); }
  ok('C steps through all four and wraps back to wipe', eq(seen, [1, 2, 3, 0, 1]));
  ok('labels read for the transient read-out', compareLabel(0) === 'Split wipe' && compareLabel(3) === 'Side-by-side' && compareLabel(9) === '');
}

// ── the palette: title matches first, stable, sectioned ───────────────────────
console.log('palette search');
{
  ok('a contiguous match scores above a scattered one', scoreText('comp', 'Compare') > scoreText('cmp', 'Compare') && scoreText('cmp', 'Compare') > 0);
  ok('no match scores zero', scoreText('zzz', 'Compare') === 0);
  ok('an empty query matches everything', scoreText('', 'anything') === 1);
  ok('a title match outranks a match only in the subtitle', scoreCommand('loupe', { title: 'Magnifier loupe' }) > scoreCommand('loupe', { title: 'Brush', subtitle: 'loupe' }));

  const cmds = [
    { id: 'a', title: 'Brush radius: larger', section: 'Viewport' },
    { id: 'b', title: 'Compare: Split wipe', subtitle: 'C cycles these four', section: 'Viewport' },
    { id: 'c', title: 'Go to Settings', section: 'Navigate' },
    { id: 'd', title: 'Thread benchmark', subtitle: 'Measures pools and compare-free', section: 'Actions' },
    { id: 'e', title: 'Load checkpoint: Good render', section: 'Checkpoints' },
  ];
  ok('an empty query keeps the palette order', eq(rankCommands(cmds, '').map((c) => c.id), ['a', 'b', 'c', 'd', 'e']));
  ok('a query ranks its title match first', rankCommands(cmds, 'compare')[0].id === 'b');
  ok('a checkpoint is found by its name', rankCommands(cmds, 'good')[0].id === 'e');
  ok('a query with no match returns nothing', rankCommands(cmds, 'xylophone').length === 0);
  ok('the query is trimmed', rankCommands(cmds, '  settings ')[0]?.id === 'c');
  ok('equal scores keep the original order (stable)', eq(rankCommands([{ id: 'p', title: 'x a' }, { id: 'q', title: 'y a' }], 'a').map((c) => c.id), ['p', 'q']));

  const groups = groupBySection(rankCommands(cmds, ''));
  ok('sections appear in the order their first command does', eq(groups.map((g) => g.section), ['Viewport', 'Navigate', 'Actions', 'Checkpoints']));
  ok('each grouped item keeps its index in the ranked list', groups[0].items[1].i === 1 && groups[0].items[1].c.id === 'b');
}

// ── the virtual keyboard: the live keymap, lit ───────────────────────────────
console.log('virtual keyboard');
{
  const tokens = DRAWN_KEYS;
  ok('every drawn key is listed once', new Set(tokens).size === tokens.length);
  ok('the rows are non-empty and cover the number row, the letters and the arrows',
    KEY_ROWS.length === 5 && tokens.includes('1') && tokens.includes('Q') && tokens.includes('ArrowLeft'));

  const byKey = bindingsByKey({});
  const c = keyState('C', byKey);
  ok('C is lit by the comparison cycle', c.bound && c.entries.some((e) => e.combo === 'C' && e.actions.some((a) => a.id === 'viewport.compareCycle')));
  ok('C also lists Alt+C (the curve editor) under the same key', c.entries.some((e) => e.combo === 'Alt+C' && e.actions.some((a) => a.id === 'timeline.curveEditor')));
  ok('Z and G both light the loupe', keyState('Z', byKey).bound && keyState('G', byKey).bound
    && keyState('Z', byKey).entries[0].actions[0].id === 'viewport.loupe');
  ok('Tab lights cinema mode', keyState('Tab', byKey).entries.some((e) => e.actions.some((a) => a.id === 'workspace.cinema')));
  ok('Space lights play / pause', keyState('Space', byKey).entries.some((e) => e.actions.some((a) => a.id === 'transport.playPause')));
  ok('the brush keys: B paint, E erase, [ and ] radius',
    keyState('E', byKey).entries.some((e) => e.actions.some((a) => a.id === 'mask.erase'))
    && keyState('[', byKey).entries.some((e) => e.actions.some((a) => a.id === 'mask.brushSmaller'))
    && keyState(']', byKey).entries.some((e) => e.actions.some((a) => a.id === 'mask.brushLarger')));
  ok('an unused key reads unbound', !keyState('`', byKey).bound && keyState('`', byKey).title === 'Not bound');
  ok('the bound count is a real number of drawn keys', boundCount(byKey) > 20 && boundCount(byKey) <= DRAWN_KEYS.length);

  const remapped = bindingsByKey({ 'viewport.compareCycle': ['Y'] });
  ok('a remap moves the highlight: Y lit, C no longer the cycle', keyState('Y', remapped).bound && !keyState('C', remapped).entries.some((e) => e.actions.some((a) => a.id === 'viewport.compareCycle')));

  ok('tokenOfEvent: space, Escape, and letters fold to the keymap spelling',
    tokenOfEvent({ key: ' ' }) === 'Space' && tokenOfEvent({ key: 'Esc' }) === 'Escape' && tokenOfEvent({ key: 'c' }) === 'C');
  ok('tokenOfEvent: digits and arrows pass through', tokenOfEvent({ key: '1' }) === '1' && tokenOfEvent({ key: 'ArrowLeft' }) === 'ArrowLeft');
}

console.log(failures === 0 ? '\nall stage 9 checks passed' : `\n${failures} stage 9 check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
if (failures) process.exitCode = 1;
