/**
 * React UI 2.0's visual layer, the parts that are logic rather than pixels (plain node, no browser):
 *
 *   node .render-check/run-module.mjs render-v2fx.mjs
 *
 *   * the terminal dock's engine feed: a polled 250-line tail arrives many times and must print once,
 *     oldest lines evict first, severities map to levels, the filter chips partition correctly;
 *   * the UI feed (termFeed): bounded, ordered, clearable, listeners told;
 *   * every tab transition ends at the IDENTITY - a resting transform / clip-path on the tab panel
 *     would make it the containing block of every `fixed` dialog inside it;
 *   * the FX preferences: defaults, persistence, and a corrupt stored value falling back to defaults.
 *
 * The mock backend sends no `log`, so the browser run cannot prove the engine half of the terminal;
 * this does.
 */
let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

// localStorage + matchMedia for the store module (node has neither).
const mem = new Map();
globalThis.window = globalThis.window || globalThis;
globalThis.localStorage = {
  getItem: (k) => (mem.has(k) ? mem.get(k) : null),
  setItem: (k, v) => mem.set(k, String(v)),
  removeItem: (k) => mem.delete(k),
};
window.localStorage = globalThis.localStorage;
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });

const { engineLine, foldEngineLog, matchesFilter, FILTERS } = await import('../src/v2fx/termModel.js');
const feed = await import('../src/v2fx/termFeed.js');
const { TRANSITION_VARIANTS, variantsFor } = await import('../src/v2fx/transitions.js');

console.log('── engine feed ─────────────────────────────────────────────');
{
  const tail = (from, to) => Array.from({ length: to - from }, (_, i) => ({ seq: from + i, t: '01:00:00', msg: `line ${from + i}` }));
  const map = new Map();
  ok('a first poll adds every line', foldEngineLog(map, tail(0, 250)) === 250 && map.size === 250);
  ok('the same tail again adds nothing', foldEngineLog(map, tail(0, 250)) === 0 && map.size === 250);
  ok('an overlapping tail adds only the new lines', foldEngineLog(map, tail(100, 350)) === 100 && map.size === 350);
  const capped = new Map();
  foldEngineLog(capped, tail(0, 30), 20);
  const keys = [...capped.keys()];
  ok('past the cap the OLDEST lines go', capped.size === 20 && keys[0] === 'en-10' && keys[19] === 'en-29', `${keys[0]}..${keys[19]}`);
  ok('an empty / missing tail is a no-op', foldEngineLog(new Map(), null) === 0 && foldEngineLog(new Map(), []) === 0);
  ok('a null entry is skipped, not thrown on', foldEngineLog(new Map(), [null, { seq: 1, msg: 'x' }]) === 1);
  const noSeq = new Map();
  foldEngineLog(noSeq, [{ t: '01', msg: 'a' }, { t: '01', msg: 'a' }, { t: '02', msg: 'a' }]);
  ok('lines without seq dedupe by time + text', noSeq.size === 2);

  ok('ERROR level -> error', engineLine({ seq: 1, msg: 'x', level: 'ERROR' }).level === 'error');
  ok('a failure message -> error', engineLine({ seq: 2, msg: 'swap failed on frame 12' }).level === 'error');
  ok('WARNING -> warn', engineLine({ seq: 3, msg: 'slow', level: 'WARNING' }).level === 'warn');
  ok('a throughput line -> perf', engineLine({ seq: 4, msg: '31.4 fps, 18 faces/s' }).level === 'perf');
  ok('a plain line -> info', engineLine({ seq: 5, msg: 'loaded model' }).level === 'info');
  ok('the category becomes the source column', engineLine({ seq: 6, msg: 'x', category: 'PERFORMANCE' }).src === 'performance');
  ok('engine lines are marked engine', engineLine({ seq: 7, msg: 'x' }).origin === 'engine');

  const lines = [
    { origin: 'engine', level: 'info' }, { origin: 'engine', level: 'warn' }, { origin: 'ui', level: 'error' },
    { origin: 'ui', level: 'ok' }, { origin: 'engine', level: 'perf' },
  ];
  const n = (f) => lines.filter((l) => matchesFilter(l, f)).length;
  ok('ALL shows everything', n('ALL') === 5);
  ok('ENGINE / UI partition by origin', n('ENGINE') === 3 && n('UI') === 2);
  ok('WARN / ERROR / PERF pick their level', n('WARN') === 1 && n('ERROR') === 1 && n('PERF') === 1);
  ok('the chips are the six filters', FILTERS.join() === 'ALL,ENGINE,UI,WARN,ERROR,PERF');
}

console.log('── ui feed ─────────────────────────────────────────────────');
{
  feed.clearTermLines();
  let told = 0;
  const off = feed.onTermLines(() => { told += 1; });
  feed.termLog('a', 'info', 'ui');
  feed.termLog('b', 'warn', 'nav');
  const ls = feed.termLines();
  ok('lines are kept in order', ls.length === 2 && ls[0].msg === 'a' && ls[1].msg === 'b');
  ok('listeners are told each line', told === 2);
  ok('ui lines carry time, level, source', /^\d\d:\d\d:\d\d$/.test(ls[0].t) && ls[1].level === 'warn' && ls[1].src === 'nav' && ls[0].origin === 'ui');
  for (let i = 0; i < 1000; i++) feed.termLog(`x${i}`);
  ok('the buffer is bounded (400)', feed.termLines().length === 400 && feed.termLines()[399].msg === 'x999');
  off();
  const before = told;
  feed.termLog('after unsubscribe');
  ok('an unsubscribed listener is not called', told === before);
  feed.clearTermLines();
  ok('clear empties it', feed.termLines().length === 0);
  ok('toast types map to levels', feed.levelOfToast('error') === 'error' && feed.levelOfToast('success') === 'ok' && feed.levelOfToast('weird') === 'info');
}

console.log('── transitions end at the identity ─────────────────────────');
{
  for (const [id, v] of Object.entries(TRANSITION_VARIANTS)) {
    if (id === 'none') continue;
    const end = v.animate.transitionEnd || {};
    const clipped = 'clipPath' in v.initial || 'clipPath' in v.animate;
    const transforms = ['x', 'y', 'scale', 'scaleX', 'scaleY', 'rotateX', 'rotateY', 'skewX', 'transformPerspective']
      .some((k) => k in v.initial);
    ok(`${id}: a moving style resets its transform when it lands`, !transforms || end.transform === 'none', JSON.stringify(end));
    ok(`${id}: a clipping style removes its clip-path when it lands`, !clipped || end.clipPath === 'none', JSON.stringify(end));
    const finalOpacity = Array.isArray(v.animate.opacity) ? v.animate.opacity.at(-1) : v.animate.opacity;
    ok(`${id}: lands fully opaque`, finalOpacity === 1);
  }
  ok('an unknown style falls back to depth', variantsFor('nope') === TRANSITION_VARIANTS.depth);
}

console.log('── FX preferences ──────────────────────────────────────────');
{
  mem.set('roop_v2_fx', '{not json');
  const { useFxStore, DEFAULT_FX } = await import('../src/v2fx/fxStore.js');
  const s = useFxStore.getState();
  ok('a corrupt stored value falls back to the defaults', s.transition === DEFAULT_FX.transition && s.ambient === true);
  s.setFx({ transition: 'portal', crt: true });
  const saved = JSON.parse(mem.get('roop_v2_fx'));
  ok('a change is persisted', saved.transition === 'portal' && saved.crt === true);
  ok('only known keys are persisted (no functions)', Object.keys(saved).every((k) => k in DEFAULT_FX));
  useFxStore.getState().toggleFx('crt');
  ok('toggle flips and persists', JSON.parse(mem.get('roop_v2_fx')).crt === false);
  useFxStore.getState().resetFx();
  ok('reset restores the defaults', JSON.parse(mem.get('roop_v2_fx')).transition === 'depth');
  ok('the ambient render-hold is ON by default (the engine is GPU-bound)', DEFAULT_FX.ambientDuringRender === false);
}

console.log(failures ? `v2fx: ${failures} check(s) FAILED` : 'v2fx: all checks passed');
globalThis.__RENDER_CHECK_FAILURES__ = failures;
