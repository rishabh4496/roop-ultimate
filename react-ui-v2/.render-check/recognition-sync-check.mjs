// Behavioural check for the recognition-selection <-> settings-state sync.
// Runs the real module the UI ships. Run with: node .render-check/recognition-sync-check.mjs
import assert from 'node:assert/strict';
import process from 'node:process';
import { mergeSelection } from '../src/components/recognitionSync.js';

let pass = 0;
const fails = [];
const check = (name, fn) => {
  try { fn(); pass++; console.log(`  PASS  ${name}`); }
  catch (e) { fails.push(name); console.log(`  FAIL  ${name}\n        ${e.message}`); }
};

// What App.jsx does: settingsDirtyRef.current = settings; postJSON('/api/settings', settings).
const autosaveBody = (settings) => settings;

const loaded = { recognition_model: 'default', recognition_provider: 'app', blend_ratio: 0.8 };
const applied = { model: 'adaface', provider: 'tensorrt' };

console.log('\n── recognition selection stays in the autosaved settings ──');
check('WITHOUT the merge, an unrelated edit posts the stale selection back (the bug)', () => {
  const afterSlider = { ...loaded, blend_ratio: 0.9 };
  assert.equal(autosaveBody(afterSlider).recognition_model, 'default');
});
check('WITH the merge, an unrelated edit after an apply keeps the applied selection', () => {
  const afterApply = mergeSelection(loaded, applied);
  const afterSlider = { ...afterApply, blend_ratio: 0.9 };
  assert.equal(autosaveBody(afterSlider).recognition_model, 'adaface');
  assert.equal(autosaveBody(afterSlider).recognition_provider, 'tensorrt');
  assert.equal(afterSlider.blend_ratio, 0.9);
});
check('every other setting is carried over untouched', () => {
  const out = mergeSelection({ a: 1, b: { c: 2 } }, applied);
  assert.equal(out.a, 1);
  assert.deepEqual(out.b, { c: 2 });
});
check('the previous state object is not mutated', () => {
  const before = JSON.stringify(loaded);
  mergeSelection(loaded, applied);
  assert.equal(JSON.stringify(loaded), before);
});
check('null / undefined settings (not loaded yet) still produce a usable object', () => {
  for (const s of [null, undefined]) {
    assert.deepEqual(mergeSelection(s, applied), { recognition_model: 'adaface', recognition_provider: 'tensorrt' });
  }
});

console.log(`\n${fails.length ? 'FAILURES' : 'ALL GREEN'}: ${pass}/${pass + fails.length} checks passed`);
process.exit(fails.length ? 1 : 0);
