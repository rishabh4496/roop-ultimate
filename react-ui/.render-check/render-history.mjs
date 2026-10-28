/**
 * Stage 23: undo/redo history engine, transactions, and the A/B/C/D snapshot slots (plain node).
 *
 *   node .render-check/run-module.mjs render-history.mjs
 *
 * The properties that matter and that a screenshot cannot show:
 *   - a drag is ONE entry, named from where it started to where it ended, however many onChange fired
 *   - undo / redo / time-travel restore the owner's state and do NOT record themselves as new edits
 *   - the ring holds at most 50 states, the redo branch dies on a new edit
 *   - a slot recall is one undoable action and leaves machine settings (provider, theme) alone
 *   - every ImageBitmap a slot ever held is closed exactly once when overwritten or cleared
 */
globalThis.window = globalThis;
globalThis.location = { protocol: 'http:', host: '127.0.0.1:42003', origin: 'http://127.0.0.1:42003', hash: '' };
globalThis.window.location = globalThis.location;

const E = await import('../src/history/historyEngine.js');
const { useHistoryStore, resetHistoryForTests, onHistoryApplied } = await import('../src/store/historyStore.js');
const C = await import('../src/history/historyChannels.js');
const D = await import('../src/history/historyDom.js');
const S = await import('../src/store/slotStore.js');

let failures = 0;
const ok = (name, cond, extra = '') => {
  if (!cond) { failures += 1; console.error(`  FAIL  ${name}${extra ? ` — ${extra}` : ''}`); } else console.log(`  PASS  ${name}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const tick = () => new Promise((r) => queueMicrotask(r));

console.log('── engine ──');
{
  let h = E.createHistory({ a: 1 }, { now: 0 });
  ok('no-op commit returns the same history', E.commit(h, { a: 1 }, { now: 1 }) === h);

  h = E.commit(h, { a: 2 }, { now: 10, label: 'two' });
  h = E.commit(h, { a: 3 }, { now: 20, label: 'three' });
  ok('two edits, three entries', h.entries.length === 3 && h.index === 2);
  const u = E.undo(h);
  ok('undo returns the previous state', u && u.state.a === 2 && u.h.index === 1);
  ok('undo at the start is null', E.undo(E.createHistory({ a: 1 })) === null);
  ok('redo at the tip is null', E.redo(h) === null);
  const h2 = E.commit(u.h, { a: 9 }, { now: 30, label: 'nine' });
  ok('a new edit ends the redo branch', h2.entries.length === 3 && E.redo(h2) === null && E.head(h2).state.a === 9);
  ok('jump to an entry', E.jump(h, 0).state.a === 1);

  let big = E.createHistory({ n: 0 });
  for (let i = 1; i <= 80; i += 1) big = E.commit(big, { n: i }, { now: i * 1000, label: `n${i}` });
  ok('ring holds 50 states', big.entries.length === 50 && big.index === 49);
  ok('oldest fell off, newest kept', big.entries[0].state.n === 31 && big.entries[49].state.n === 80);
  ok('undo all the way back stops at the oldest kept', (() => {
    let x = big; let steps = 0;
    for (let r = E.undo(x); r; r = E.undo(x)) { x = r.h; steps += 1; }
    return steps === 49 && E.head(x).state.n === 31;
  })());

  // coalescing: arrow-keying a value
  let c = E.createHistory({ v: 0.5 }, { now: 0 });
  const desc = (b, a) => E.describeSettingsChange(b, a, () => 'Restore Ultra Blend');
  c = E.commit(c, { v: 0.6 }, { now: 100, coalesceKey: 'v', describe: desc });
  c = E.commit(c, { v: 0.7 }, { now: 300, coalesceKey: 'v', describe: desc });
  c = E.commit(c, { v: 0.78 }, { now: 500, coalesceKey: 'v', describe: desc });
  ok('three nearby changes of one thing are one entry', c.entries.length === 2);
  ok('its label reads from where it started', E.head(c).label === 'Changed Restore Ultra Blend (0.50 -> 0.78)', E.head(c).label);
  c = E.commit(c, { v: 0.9 }, { now: 5000, coalesceKey: 'v', describe: desc });
  ok('a change after the window is a new entry', c.entries.length === 3);
  let back = E.commit(E.createHistory({ v: 1 }, { now: 0 }), { v: 2 }, { now: 10, coalesceKey: 'v', describe: desc });
  back = E.commit(back, { v: 1 }, { now: 20, coalesceKey: 'v', describe: desc });
  ok('dragging back to the start leaves no entry', back.entries.length === 1 && back.index === 0);

  ok('describe: numbers', E.describeSettingsChange({ x: 0.5 }, { x: 0.78 }, () => 'X') === 'Changed X (0.50 -> 0.78)');
  ok('describe: integers stay integers', E.describeSettingsChange({ x: 3 }, { x: 4 }, () => 'X') === 'Changed X (3 -> 4)');
  ok('describe: booleans', E.describeSettingsChange({ x: false }, { x: true }, () => 'X') === 'Turned on X');
  ok('describe: several keys', E.describeSettingsChange({ a: 1, b: 1, c: 1, d: 1 }, { a: 2, b: 2, c: 2, d: 2 }) === 'Changed 4 settings');
  ok('describe: no change is null', E.describeSettingsChange({ a: { z: 1 } }, { a: { z: 1 } }) === null);
  ok('settingsEqual sees through new object identity', E.settingsEqual({ a: { z: 1 } }, { a: { z: 1 } }));
  ok('roto labels', E.describeRotoChange({}, { m: [1] }) === 'Added Roto Polygon' && E.describeRotoChange({ m: [1] }, {}) === 'Deleted Roto Polygon' && E.describeRotoChange({ m: [{ a: 1 }] }, { m: [{ a: 2 }] }) === 'Edited Roto Polygon');
  ok('stroke labels', E.describeStrokesChange([], [{}]) === 'Painted Occlusion Mask Stroke' && E.describeStrokesChange([{}], [{}, { erase: true }]) === 'Erased Occlusion Mask Stroke' && E.describeStrokesChange([{}], []) === 'Cleared Occlusion Mask');
  ok('ago()', E.ago(0, 2000) === 'just now' && E.ago(0, 30000) === '30s ago' && E.ago(0, 200000) === '3m ago');
}

console.log('── keys ──');
{
  const k = (key, o = {}) => D.historyKeyAction({ key, ctrlKey: true, ...o });
  ok('Ctrl+Z undo', k('z') === 'undo' && k('Z') === 'undo');
  ok('Ctrl+Y redo', k('y') === 'redo');
  ok('Ctrl+Shift+Z redo', k('z', { shiftKey: true }) === 'redo');
  ok('Cmd+Z undo', D.historyKeyAction({ key: 'z', metaKey: true }) === 'undo');
  ok('bare Z and Alt+Ctrl+Z are not history', D.historyKeyAction({ key: 'z' }) === null && k('z', { altKey: true }) === null);
  ok('Ctrl+Shift+Y is not redo', k('y', { shiftKey: true }) === null);
  ok('text inputs keep their own undo', D.isTextEntry({ tagName: 'INPUT', type: 'text' }) && D.isTextEntry({ tagName: 'INPUT' }) && D.isTextEntry({ tagName: 'TEXTAREA' }) && D.isTextEntry({ tagName: 'DIV', isContentEditable: true }) && D.isTextEntry({ tagName: 'INPUT', type: 'number' }));
  ok('a focused slider or checkbox does not', !D.isTextEntry({ tagName: 'INPUT', type: 'range' }) && !D.isTextEntry({ tagName: 'INPUT', type: 'checkbox' }) && !D.isTextEntry({ tagName: 'BUTTON' }) && !D.isTextEntry(null));
}

// A fake owner of a channel: React state reduced to a variable plus the re-render that reports it.
function owner(id, initial, def = {}) {
  const o = { id, value: initial, applied: [], alive: true };
  o.unregister = useHistoryStore.getState().register(id, {
    ...def,
    apply: (v) => {
      o.applied.push(v);
      o.value = v;
      // the owner re-renders with exactly what it was given and reports it
      queueMicrotask(() => { if (o.alive) useHistoryStore.getState().observe(id, o.value); });
    },
  });
  const unreg = o.unregister;
  o.unregister = () => { o.alive = false; unreg(); };
  o.set = (v) => { o.value = v; useHistoryStore.getState().observe(id, v); };
  o.report = () => useHistoryStore.getState().observe(id, o.value);
  return o;
}
const H = () => useHistoryStore.getState();

console.log('── store: transactions ──');
{
  resetHistoryForTests();
  const s = owner('settings', { blend: 0.5, name: 'x' }, C.SETTINGS_CHANNEL);
  s.report();
  ok('baseline is one entry', H().entries.length === 1);

  H().beginTransaction();
  for (let i = 1; i <= 60; i += 1) s.set({ blend: 0.5 + i * 0.005, name: 'x' });   // 60 onChange events
  await sleep(20);
  ok('nothing is recorded mid-drag', H().entries.length === 1 && H().inTransaction());
  H().endTransaction();
  await sleep(120);
  ok('the drag is ONE entry', H().entries.length === 2, `entries=${H().entries.length}`);
  ok('named with the start and end value', H().entries[1].label === 'Changed Blend (0.50 -> 0.80)', H().entries[1].label);
  ok('transaction closed', !H().inTransaction());

  // undo / redo restore the owner and do not record themselves
  const before = H().entries.length;
  ok('undo succeeds', H().undo() === true);
  await sleep(20);
  ok('owner got the old value back', s.value.blend === 0.5 && s.applied.length === 1);
  ok('undo recorded nothing new', H().entries.length === before && H().index === 0, `entries=${H().entries.length} index=${H().index}`);
  ok('canRedo', H().canRedo && !H().canUndo);
  ok('redo succeeds', H().redo() === true);
  await sleep(20);
  ok('owner got the new value back', Math.abs(s.value.blend - 0.8) < 1e-9);
  ok('redo recorded nothing new', H().entries.length === before && H().index === 1);

  // one tick, several channels => one entry
  const roto = owner('roto', {}, C.ROTO_CHANNEL);
  roto.report();
  await tick();
  s.set({ blend: 0.1, name: 'x' });
  roto.set({ clip: [{ id: 1 }] });
  await sleep(20);
  ok('two channels changed in one tick make one entry', H().entries.length === before + 1, `entries=${H().entries.length}`);
  ok('and are named for both', /\(\+1 more\)/.test(H().entries.at(-1).label), H().entries.at(-1).label);

  // roto polygon is its own entry with its own name
  roto.set({ clip: [{ id: 1 }, { id: 2 }] });
  await sleep(20);
  ok('roto add is named', H().entries.at(-1).label === 'Added Roto Polygon', H().entries.at(-1).label);
  roto.unregister(); s.unregister();
}

console.log('── store: time travel ──');
{
  resetHistoryForTests();
  const s = owner('settings', { v: 0 }, C.SETTINGS_CHANNEL);
  s.report();
  for (let i = 1; i <= 5; i += 1) { s.set({ v: i * 10 }); await sleep(750); }      // outside the coalesce window
  ok('five edits, six entries', H().entries.length === 6, `entries=${H().entries.length}`);
  H().jumpTo(2);
  await sleep(20);
  ok('jumping to a past row restores that state', s.value.v === 20 && H().index === 2);
  ok('and records nothing', H().entries.length === 6);
  s.set({ v: 99 });
  await sleep(20);
  ok('editing after time-travel ends the redo branch', H().entries.length === 4 && !H().canRedo && H().entries.at(-1).state.settings.v === 99);
  let seen = null;
  const off = onHistoryApplied((e) => { seen = e; });
  H().undo();
  off();
  ok('listeners hear what was undone', seen && seen.kind === 'undo' && /Changed/.test(seen.label), JSON.stringify(seen));
  s.unregister();
}

console.log('── store: channels ──');
{
  resetHistoryForTests();
  const s = owner('settings', { v: 1 }, C.SETTINGS_CHANNEL);
  s.report();
  // a bundle the owner rebuilds with a NEW identity on every render
  const mask = owner('mask', { dataUrl: null, kps: null, version: 0 }, C.MASK_CHANNEL);
  mask.report();
  await tick();
  mask.set({ dataUrl: 'data:a', kps: [[1, 2]], version: 1 });
  await sleep(20);
  ok('applying a mask is one named entry', H().entries.length === 2 && H().entries[1].label === 'Applied Occlusion Mask', H().entries[1].label);
  H().undo();
  await sleep(20);
  mask.value = { ...mask.value };                 // re-render builds an equal object with a new identity
  mask.report();
  await sleep(20);
  ok('an equal-but-new bundle is not an edit', H().entries.length === 2 && H().index === 0);
  ok('and the undo emptied the mask', mask.applied.at(-1).dataUrl === null);

  // a channel that mounts later is part of the baseline, not an edit
  const n = H().entries.length;
  const late = owner('strokes', [], C.STROKES_CHANNEL);
  late.report();
  await sleep(20);
  ok('a late channel does not make an entry', H().entries.length === n);
  late.set([{ erase: false }]);
  await sleep(20);
  ok('its first stroke does (and, being a new edit, ends the redo branch)', H().entries.length === 2 && H().index === 1 && H().entries.at(-1).label === 'Painted Occlusion Mask Stroke', H().entries.at(-1).label);

  // an unmounted channel keeps its place and undo skips it
  late.unregister();
  const apps = late.applied.length;
  H().undo();
  await sleep(20);
  ok('undo skips a channel nobody owns', late.applied.length === apps);
  s.unregister(); mask.unregister();
}

console.log('── slots ──');
{
  resetHistoryForTests();
  S.resetSlotsForTests();
  const settings = owner('settings', { blend: 0.4, provider: 'cuda', selected_theme: 'Dark', merger_sharpen: 0.2 }, C.SETTINGS_CHANNEL);
  settings.report();
  const roto = owner('roto', {}, C.ROTO_CHANNEL);
  roto.report();
  const extras = owner('extrasOpts', { fps: 30 }, C.EXTRAS_OPTS_CHANNEL);
  extras.report();
  await tick();

  const closed = [];
  const bmp = (name) => ({ name, width: 640, height: 360, close() { closed.push(name); } });

  S.captureSlot('faceswap', 'A', { bitmap: bmp('A1'), frameKey: 'm_1' });
  ok('a slot captures its route and the global recipe, not the other route', eq(Object.keys(S.useSlotStore.getState().banks.faceswap.A.state).sort(), ['roto', 'settings']));
  ok('the extras route has its own bank', Object.keys(S.useSlotStore.getState().banks.extras.A?.state || {}).length === 0);

  settings.set({ blend: 0.9, provider: 'cpu', selected_theme: 'Light', merger_sharpen: 0.8 });
  roto.set({ clip: [{ id: 1 }] });
  await sleep(20);
  const n = H().entries.length;
  S.captureSlot('faceswap', 'B', { bitmap: bmp('B1'), frameKey: 'm_1' });

  const slot = S.recallSlot('faceswap', 'A');
  await sleep(20);
  ok('recall returns the slot', slot && slot.id === 'A');
  ok('recall applies the recipe', settings.value.blend === 0.4 && settings.value.merger_sharpen === 0.2 && Object.keys(roto.value).length === 0);
  ok('recall leaves machine settings alone', settings.value.provider === 'cpu' && settings.value.selected_theme === 'Light', JSON.stringify(settings.value));
  ok('recall is ONE history entry, named', H().entries.length === n + 1 && H().entries.at(-1).label === 'Recalled Slot A', `${H().entries.length - n} ${H().entries.at(-1).label}`);
  H().undo();
  await sleep(20);
  ok('Ctrl+Z flips back to the look you left', settings.value.blend === 0.9 && settings.value.provider === 'cpu');
  ok('recalling an empty slot does nothing', S.recallSlot('faceswap', 'D') === null);
  ok('active slot tracks the recall', S.useSlotStore.getState().active.faceswap === 'A');

  // bitmaps: closed once, when overwritten or cleared, never while the stage may still hold them
  ok('nothing closed yet', closed.length === 0);
  S.captureSlot('faceswap', 'A', { bitmap: bmp('A2'), frameKey: 'm_2' });
  ok('overwriting retires the old bitmap but does not close it in the same tick', closed.length === 0);
  S.flushRetired();
  ok('overwrite closes the old bitmap', eq(closed, ['A1']), JSON.stringify(closed));
  S.useSlotStore.getState().setReference('faceswap', 'B');
  S.useSlotStore.getState().clear('faceswap', 'B');
  ok('clearing the reference slot drops the reference', S.useSlotStore.getState().reference.faceswap === null);
  S.flushRetired();
  ok('clear closes its bitmap', eq(closed, ['A1', 'B1']), JSON.stringify(closed));
  await sleep(S.RETIRE_MS + 60);
  ok('the retire timer does it unaided', (() => { S.useSlotStore.getState().clear('faceswap', 'A'); return true; })());
  await sleep(S.RETIRE_MS + 60);
  ok('timer closed the last one, nothing twice', eq(closed.slice().sort(), ['A1', 'A2', 'B1']), JSON.stringify(closed));
  S.captureSlot('extras', 'C', { bitmap: null });
  S.useSlotStore.getState().clear('extras', 'C');
  ok('a slot with no frame clears cleanly', S.useSlotStore.getState().banks.extras.C === null);

  settings.unregister(); roto.unregister(); extras.unregister();
}

console.log('── labels ──');
ok('catalog / diff labels are used', C.settingLabel('blend_ratio') === 'Blend' && C.settingLabel('server_port') === 'Server port');
ok('unknown keys are humanised', C.settingLabel('some_new_key') === 'Some new key');
ok('machine keys are not recipe keys', !C.isRecipeKey('provider') && !C.isRecipeKey('theme_dark') && !C.isRecipeKey('server_share') && C.isRecipeKey('blend_ratio'));

console.log(failures === 0 ? '\nall history checks passed' : `\n${failures} history check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
