/**
 * Stage 27 Logic Render Check (plain node, no browser):
 * 1. Resizable Multi-Panel Workspace Store:
 *    - Panel size defaults (leftWidth: 350, rightWidth: 330, bottomHeight: 280)
 *    - Strict dimension clamping (min/max boundaries)
 *    - Custom layout creation, application, and deletion
 *    - Built-in layout presets (Standard, Max Canvas, Inspection Suite, My Roto Setup, Dual-Monitor Color, Minimal Laptop)
 * 2. Per-Frame Bottleneck Flamegraph Model:
 *    - 7 distinct pipeline stages in sequence
 *    - Stage color schemes (Cyan, Purple, Emerald, Blue, Amber, Pink, Slate)
 *    - Frame-time percentage allocations
 *    - Min / Avg / P95 / Max percentile statistics
 *    - Hardware residency tags (GPU vs CPU)
 *    - Full telemetry report packaging
 *
 *   node .render-check/run-module.mjs render-stage27-logic.mjs
 */
import process from 'node:process';
import { useWorkspaceStore, LAYOUT_PRESETS, DEFAULT_PANEL_SIZES, PANEL_LIMITS } from '../src/store/workspaceStore.js';
import {
  FLAMEGRAPH_STAGES,
  STAGE_COLORS,
  STAGE_RESIDENCY,
  computeStageMetrics,
  computeFlamePercentages,
  exportTelemetryReport,
} from '../src/components/faceswap/flamegraphModel.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ── 1. Resizable Multi-Panel Workspace Layouts ──────────────────────────────
console.log('workspace panel sizes & custom layouts');
const st = () => useWorkspaceStore.getState();

// Check defaults
ok('default leftWidth is 350px', st().panelSizes.leftWidth === DEFAULT_PANEL_SIZES.leftWidth);
ok('default rightWidth is 330px', st().panelSizes.rightWidth === DEFAULT_PANEL_SIZES.rightWidth);
ok('default bottomHeight is 280px', st().panelSizes.bottomHeight === DEFAULT_PANEL_SIZES.bottomHeight);

// Check clamping
st().setPanelSize('leftWidth', 100);
ok('leftWidth clamps to minimum 240px', st().panelSizes.leftWidth === PANEL_LIMITS.leftWidth.min);
st().setPanelSize('leftWidth', 900);
ok('leftWidth clamps to maximum 620px', st().panelSizes.leftWidth === PANEL_LIMITS.leftWidth.max);

st().setPanelSize('rightWidth', 150);
ok('rightWidth clamps to minimum 240px', st().panelSizes.rightWidth === PANEL_LIMITS.rightWidth.min);
st().setPanelSize('rightWidth', 1200);
ok('rightWidth clamps to maximum 580px', st().panelSizes.rightWidth === PANEL_LIMITS.rightWidth.max);

st().setPanelSize('bottomHeight', 50);
ok('bottomHeight clamps to minimum 140px', st().panelSizes.bottomHeight === PANEL_LIMITS.bottomHeight.min);
st().setPanelSize('bottomHeight', 800);
ok('bottomHeight clamps to maximum 600px', st().panelSizes.bottomHeight === PANEL_LIMITS.bottomHeight.max);

// Reset to standard defaults
st().resetPanelSizes();
ok('resetPanelSizes restores standard proportions',
  st().panelSizes.leftWidth === DEFAULT_PANEL_SIZES.leftWidth &&
  st().panelSizes.rightWidth === DEFAULT_PANEL_SIZES.rightWidth &&
  st().panelSizes.bottomHeight === DEFAULT_PANEL_SIZES.bottomHeight
);

// Preset coverage
const presetIds = LAYOUT_PRESETS.map((p) => p.id);
ok('includes built-in layout presets',
  presetIds.includes('standard') &&
  presetIds.includes('canvas') &&
  presetIds.includes('inspect') &&
  presetIds.includes('roto') &&
  presetIds.includes('dual_color') &&
  presetIds.includes('laptop')
);

// Custom Layout Creation & Persistence
st().setPanelSize('leftWidth', 400);
st().setPanelSize('rightWidth', 450);
st().setPanelSize('bottomHeight', 350);
const savedLayout = st().saveCustomLayout('My Studio Roto');
const savedId = savedLayout ? savedLayout.id : null;
ok('saveCustomLayout returns created layout with id', Boolean(savedId) && savedId.startsWith('custom_'));
ok('activeLayoutId is set to created layout', st().activeLayoutId === savedId);
ok('custom layout appears in customLayouts list', st().customLayouts.some((l) => l.id === savedId && l.name === 'My Studio Roto'));

// Switch away to standard preset
st().applyLayout('standard');
ok('applying standard preset restores activeLayoutId standard', st().activeLayoutId === 'standard');
ok('standard preset restores standard dimensions', st().panelSizes.leftWidth === DEFAULT_PANEL_SIZES.leftWidth);

// Re-apply custom layout
st().applyLayout(savedId);
ok('re-applying custom layout restores activeLayoutId', st().activeLayoutId === savedId);
ok('re-applying custom layout restores custom dimensions',
  st().panelSizes.leftWidth === 400 &&
  st().panelSizes.rightWidth === 450 &&
  st().panelSizes.bottomHeight === 350
);

// Delete custom layout
st().deleteCustomLayout(savedId);
ok('deleteCustomLayout removes layout from list', !st().customLayouts.some((l) => l.id === savedId));
ok('deleteCustomLayout falls back to standard when active deleted', st().activeLayoutId === 'standard');

// ── 2. Live Per-Frame Bottleneck Flamegraph Model ────────────────────────────
console.log('flamegraph model & stage breakdown');

// 7 Stages in required sequence
const expectedStages = ['decode', 'detect', 'swap', 'mask', 'restore', 'blend', 'encode'];
ok('flamegraph defines exactly 7 stages', FLAMEGRAPH_STAGES.length === 7);
ok('flamegraph stages match sequence', eq(FLAMEGRAPH_STAGES.map((s) => s.key), expectedStages));

// Visual Palette Verification
ok('decode stage is Cyan', STAGE_COLORS.decode.bar.includes('cyan'));
ok('detect stage is Purple', STAGE_COLORS.detect.bar.includes('purple'));
ok('swap stage is Emerald', STAGE_COLORS.swap.bar.includes('emerald'));
ok('mask stage is Blue', STAGE_COLORS.mask.bar.includes('blue'));
ok('restore stage is Amber', STAGE_COLORS.restore.bar.includes('amber'));
ok('blend (relight) stage is Pink', STAGE_COLORS.blend.bar.includes('pink'));
ok('encode stage is Slate', STAGE_COLORS.encode.bar.includes('slate'));

// Hardware Residency
ok('decode and encode are CPU/NVDEC residency', STAGE_RESIDENCY.decode.residency.includes('CPU') && STAGE_RESIDENCY.encode.residency.includes('GPU'));
ok('detect, swap, mask, restore, blend are GPU resident',
  STAGE_RESIDENCY.detect.residency === 'GPU' &&
  STAGE_RESIDENCY.swap.residency === 'GPU' &&
  STAGE_RESIDENCY.mask.residency === 'GPU' &&
  STAGE_RESIDENCY.restore.residency === 'GPU' &&
  STAGE_RESIDENCY.blend.residency === 'GPU'
);

// Percentage calculations
const sampleFrameTimes = {
  decode: 5.0,
  detect: 8.0,
  swap: 15.0,
  mask: 4.0,
  restore: 12.0,
  blend: 2.0,
  encode: 4.0,
};
const percentages = computeFlamePercentages(sampleFrameTimes);
ok('flame percentage for swap is 30%', percentages.swap === 30);
ok('flame percentage for restore is 24%', percentages.restore === 24);
ok('flame percentage for decode is 10%', percentages.decode === 10);
const sumPct = Object.values(percentages).reduce((a, b) => a + b, 0);
ok('flame percentages sum to 100%', Math.round(sumPct) === 100);

// Percentile & summary metrics
const history = [
  { decode: 4.0, detect: 7.0, swap: 14.0, mask: 3.5, restore: 10.0, blend: 2.0, encode: 3.5 },
  { decode: 5.0, detect: 8.0, swap: 15.0, mask: 4.0, restore: 12.0, blend: 2.0, encode: 4.0 },
  { decode: 6.0, detect: 9.0, swap: 18.0, mask: 4.5, restore: 14.0, blend: 2.5, encode: 4.5 },
];
const metrics = computeStageMetrics('swap', history);
ok('computes min correctly', metrics.min === 14.0);
ok('computes max correctly', metrics.max === 18.0);
ok('computes avg correctly', Math.abs(metrics.avg - 15.67) < 0.05);
ok('computes p95 correctly', metrics.p95 >= 15.0 && metrics.p95 <= 18.0);

// Telemetry Export
const report = exportTelemetryReport({
  jobId: 'job_stage27_demo',
  recipe: { swap_model: 'realswap', face_enhancer: 'gpen_256_pro' },
  history,
  summary: { fps: 28.5, totalFrames: 120 },
});
ok('telemetry report contains metadata', report.job_id === 'job_stage27_demo' && report.app === 'roop-ultimate');
ok('telemetry report includes recipe', report.recipe.swap_model === 'realswap');
ok('telemetry report includes stage_metrics for all 7 stages',
  expectedStages.every((s) => report.stage_metrics && report.stage_metrics[s] !== undefined)
);

console.log(failures === 0 ? '\nall Stage 27 logic checks passed' : `\n${failures} check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
process.exitCode = failures === 0 ? 0 : 1;
