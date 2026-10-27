/**
 * The workstation UI's pure logic (plain node, no browser): the quick recipes, the
 * temporal stability modes, the telemetry pill's reading, the runtime section rows,
 * the Settings tab map and the workspace store.
 *
 *   node .render-check/run-module.mjs render-workstation-logic.mjs
 *
 * What each group pins, because each is a place a control can look right and be wrong:
 *   * recipes      applying one makes it the active one, and the factory settings are
 *                  "Natural Studio" (so the bar does not claim "Custom" on a fresh install);
 *   * temporal     a mode's patch reads back as THAT mode, the shipped defaults are Cinema,
 *                  and changing one parameter reads back as Custom, never as a stale mode;
 *   * telemetry    an absent gpu_util is null (not 0): "unmeasured" is not "idle";
 *   * runtime      nested and single-value-wrapper sections flatten the way the backend sends them;
 *   * tabs         every setting key resolves to one of the five tabs;
 *   * store        an unknown layout or tab is refused instead of being stored.
 */
import process from 'node:process';
import { FACESWAP_DEFAULTS } from '../src/components/faceswap/defaults.js';
import { QUICK_RECIPES, activeRecipeId, recipeMatches } from '../src/components/faceswap/recipes.js';
import { TEMPORAL_MODES, temporalModeOf, temporalPatch, temporalNote } from '../src/components/faceswap/temporalModes.js';
import { readingOf, pct } from '../src/components/shell/telemetryReading.js';
import { RUNTIME_SECTIONS, rowsOf } from '../src/components/runtimeRows.js';
import { TAB_LABELS, tabForSetting } from '../src/components/settingsTabs.js';
import { useWorkspaceStore, WORKSPACE_MODES, LAYOUT_PRESETS, SETTINGS_TABS } from '../src/store/workspaceStore.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ── recipes ────────────────────────────────────────────────────────────────
console.log('recipes');
ok('four recipes', QUICK_RECIPES.length === 4 && eq(QUICK_RECIPES.map((r) => r.id), ['fast', 'natural', 'fidelity', 'occlusion']));
for (const r of QUICK_RECIPES) {
  const applied = { ...FACESWAP_DEFAULTS, ...r.patch };
  ok(`applying "${r.label}" makes it the active recipe`, activeRecipeId(applied) === r.id, `got ${activeRecipeId(applied)}`);
}
ok('factory settings are Natural Studio', activeRecipeId({ ...FACESWAP_DEFAULTS }) === 'natural');
ok('a changed value is no longer a match', !recipeMatches({ ...FACESWAP_DEFAULTS, merger_grain_match: 0.9 }, QUICK_RECIPES[1].patch));
ok('numbers compare by value, not by float identity', recipeMatches({ merger_sharpen: 0.35000000001 }, { merger_sharpen: 0.35 }));
ok('default_det_size absent counts as true', recipeMatches({}, { default_det_size: true }) && !recipeMatches({}, { default_det_size: false }));
ok('no recipe touches a global (provider / threads) setting',
  QUICK_RECIPES.every((r) => !Object.keys(r.patch).some((k) => /^(provider|max_threads|perf_|cpu_|trt_)/.test(k))));

// ── temporal modes ─────────────────────────────────────────────────────────
console.log('temporal modes');
ok('four modes in order', eq(TEMPORAL_MODES.map((m) => m.id), ['off', 'smart_skip', 'balanced', 'cinema']));
for (const m of TEMPORAL_MODES) {
  ok(`"${m.label}" reads back as ${m.id}`, temporalModeOf({ ...FACESWAP_DEFAULTS, ...temporalPatch(m.id) }) === m.id,
    `got ${temporalModeOf({ ...FACESWAP_DEFAULTS, ...temporalPatch(m.id) })}`);
}
ok('the shipped defaults are Cinema Cut-Aware', temporalModeOf({ ...FACESWAP_DEFAULTS }) === 'cinema');
ok('an empty settings object is the same as the defaults', temporalModeOf({}) === 'cinema');
ok('smart-skip is part of "off" and "smart_skip" only',
  temporalPatch('off').temporal_smart_skip === false && temporalPatch('smart_skip').temporal_smart_skip === true
  && !('temporal_smart_skip' in temporalPatch('balanced')) && !('temporal_smart_skip' in temporalPatch('cinema')));
ok('no mode re-tunes a strength slider',
  TEMPORAL_MODES.every((m) => !Object.keys(m.patch).some((k) => /strength|weight|cutoff|beta/.test(k))));
for (const m of TEMPORAL_MODES) {
  const edited = { ...FACESWAP_DEFAULTS, ...m.patch, stabilize_mask: !m.patch.stabilize_mask };
  ok(`editing one parameter of ${m.id} reads back as custom`, temporalModeOf(edited) === 'custom', `got ${temporalModeOf(edited)}`);
}
ok('custom has no patch (it describes, never targets)', eq(temporalPatch('custom'), {}) && eq(temporalPatch('nope'), {}));
ok('the note names what a custom state is made of', /^Custom: .*tracked detection/.test(temporalNote({ ...FACESWAP_DEFAULTS, stabilize_mask: false })));
ok('balanced uses EMA, cinema uses One Euro', temporalPatch('balanced').stabilize_method === 'ema' && temporalPatch('cinema').stabilize_method === 'one_euro');

// ── telemetry pill ─────────────────────────────────────────────────────────
console.log('telemetry reading');
ok('no sample, no reading', readingOf(null) === null && readingOf(undefined) === null && readingOf('x') === null);
const withSmi = readingOf({ gpu_util: 37, vram_used: 6, vram_total: 12, cpu_percent: 14 });
ok('with nvidia-smi: util, vram %, cpu', withSmi.gpu === 37 && withSmi.vram === 50 && withSmi.cpu === 14 && withSmi.primary === 37);
const noSmi = readingOf({ vram_used: 3, vram_total: 12, cpu_percent: 9 });
ok('without nvidia-smi: gpu is null, not 0, and the trace is VRAM', noSmi.gpu === null && noSmi.primary === 25);
ok('gpu_util 0 is a real zero', readingOf({ gpu_util: 0, vram_used: 1, vram_total: 2 }).gpu === 0);
ok('no GPU at all: vram is null', readingOf({ gpu: 'CPU Only', vram_total: 0, vram_used: 0, cpu_percent: 5 }).vram === null);
ok('percentages clamp', pct(20, 10) === 100 && pct(-5, 10) === 0 && pct(1, 0) === null);

// ── runtime sections ───────────────────────────────────────────────────────
console.log('runtime sections');
ok('fourteen sections, backend order', RUNTIME_SECTIONS.length === 14 && RUNTIME_SECTIONS[0] === 'SYSTEM' && RUNTIME_SECTIONS[13] === 'CHECKPOINT');
ok('a { value } wrapper shows under its parent', eq(rowsOf({ gpu: { value: 'RTX 4070' } }), [['gpu', 'RTX 4070']]));
ok('nested objects flatten to parent.child', eq(rowsOf({ vram: { total_gb: 12, free_gb: 9 } }), [['vram.total_gb', '12'], ['vram.free_gb', '9']]));
ok('empty and null read as a dash, arrays are joined', eq(rowsOf({ a: null, b: '', c: ['x', 'y'], d: [] }), [['a', '—'], ['b', '—'], ['c', 'x, y'], ['d', 'none']]));
ok('a non-object has no rows', eq(rowsOf(undefined), []) && eq(rowsOf('s'), []));

// ── settings tabs ──────────────────────────────────────────────────────────
console.log('settings tabs');
ok('five tabs, each with a label', SETTINGS_TABS.length === 5 && SETTINGS_TABS.every((t) => TAB_LABELS[t]));
ok('provider and TensorRT go to providers', tabForSetting('provider', 'Performance') === 'providers' && tabForSetting('trt_cuda_graph', 'Advanced performance') === 'providers');
ok('thread controls go to cpu', ['max_threads', 'auto_thread_selection', 'memory_limit'].every((k) => tabForSetting(k, 'Performance') === 'cpu') && tabForSetting('perf_trt_pool', 'Advanced performance') === 'cpu');
ok('server and output go to paths', tabForSetting('server_port', 'Server') === 'paths' && tabForSetting('output_video_codec', 'Output') === 'paths');
ok('detector defaults leave "Performance"', tabForSetting('face_detector_nms', 'Performance') === 'general');
ok('anything else is general', tabForSetting('selected_theme', 'Appearance & Theme') === 'general' && tabForSetting('x', undefined) === 'general');

// ── workspace store ────────────────────────────────────────────────────────
console.log('workspace store');
const st = () => useWorkspaceStore.getState();
ok('defaults', st().workspaceMode === 'default' && st().railCollapsed === false && st().settingsTab === 'providers');
st().setWorkspaceMode('cinema');
ok('a known layout is stored', st().workspaceMode === 'cinema');
st().setWorkspaceMode('nonsense');
ok('an unknown layout is refused', st().workspaceMode === 'cinema');
st().toggleRail();
ok('the rail toggles', st().railCollapsed === true);
st().setSettingsTab('diagnostics');
st().setSettingsTab('bogus');
ok('an unknown settings tab is refused', st().settingsTab === 'diagnostics');
ok('every layout preset names a real mode', LAYOUT_PRESETS.every((l) => WORKSPACE_MODES.includes(l.mode)));
ok('Max Canvas hides every panel, Inspection Suite keeps the timeline',
  LAYOUT_PRESETS.find((l) => l.id === 'canvas').mode === 'cinema' && LAYOUT_PRESETS.find((l) => l.id === 'inspect').mode === 'timeline');

console.log(failures === 0 ? '\nall workstation logic checks passed' : `\n${failures} check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
process.exitCode = failures === 0 ? 0 : 1;
