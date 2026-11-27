import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab, startOnSettingsTab } from './helpers.js';

// The 3-stage pipeline controls (#/faceswap, #/settings) and their live telemetry (#/processing, #/settings) in a
// real browser, against the mock server.
//
//   * the four controls are on Face Swap AND Settings, edit the one settings object, and reach the backend as the
//     setting names the render reads (the autosave POST is intercepted and read);
//   * the restorer mode is one choice over three real settings;
//   * the stride and the toggle work from the keyboard alone;
//   * Face Swap still has exactly one Start control;
//   * the live HUD names Stage 1 in keyframes/s, Stage 2's flow engine only when it was reported, labels Stage 3's
//     columns from the backend's own backend names, and shows the canary precision status (PASSED / FAILED / none);
//   * axe finds nothing in any of those states.
//
// Telemetry and runtime-report payloads are scripted in the page, so what is under test is what the SHELL does with a
// backend's real shapes, not the mock's invented numbers.

const AXE_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];
const S1 = 'STAGE_1_SPARSE_PRESCAN';
const S2 = 'STAGE_2_ACTIVE_PREPASS';
const S3 = 'STAGE_3_MAIN_SWAP';

/** Resolves with the next autosave body (the whole settings object) that satisfies `want`. */
const nextSave = (page, want) => page.waitForRequest((r) => {
  if (!r.url().endsWith('/api/settings') || r.method() !== 'POST') return false;
  try { return want(r.postDataJSON()); } catch { return false; }
}, { timeout: 10_000 }).then((r) => r.postDataJSON());

// The mock server is ONE process for the whole run and keeps what the app POSTs, so a test that moves a setting would
// hand the next test (and the next spec file) its value. These are the defaults of app/settings.py.
const DEFAULTS = {
  swap_model: 'realswap', selected_enhancer: 'UltraMax', prepass_detect_interval: 1, temporal_smart_skip: false,
  auto_bypass_restorer_native_swap: false, restore_ultra_frequency_blend: true,
};
const resetSettings = (page) => page.request.post('/api/settings', { data: DEFAULTS });
test.beforeEach(async ({ page }) => { await resetSettings(page); });
test.afterEach(async ({ page }) => { await resetSettings(page); });

const controls = (page) => page.getByTestId('pipeline-controls');
const swapper = (page) => controls(page).locator('select').nth(0);
const restorer = (page) => controls(page).locator('select').nth(1);
const stride = (page) => controls(page).locator('input[type="range"]');
const smartSkip = (page) => controls(page).locator('input[type="checkbox"]');

test('faceswap: the four controls are one labelled group and write the settings the render reads', async ({ page }) => {
  await openTab(page, 'faceswap');
  const group = page.getByRole('group', { name: '3-stage pipeline' });
  await expect(group).toBeVisible();
  await expect(controls(page)).toHaveCount(1);

  // swapper: the five named models are there under their own names, and nothing the backend lists is lost
  const labels = await swapper(page).locator('option').allTextContents();
  for (const name of ['inswapper_128', 'ReSwapper 256', 'Alpha 256', 'HyperSwap 256', 'HyperSwap 512']) expect(labels).toContain(name);
  expect(labels, 'the long tail is still selectable').toContain('ghost_1');
  await expect(swapper(page).locator('optgroup')).toHaveCount(3);

  const swapSave = nextSave(page, (s) => s.swap_model === 'alpha_256');
  await swapper(page).selectOption('alpha_256');
  expect((await swapSave).swap_model).toBe('alpha_256');

  // restorer mode: Laplacian Split RF++ = Restore Ultra + the frequency blend, with the bypass off
  await expect(restorer(page).locator('option')).toHaveText(['Off / Auto-Bypass ≥256px', 'Laplacian Split RF++', 'Full RF++', 'Custom · UltraMax']);
  const lapSave = nextSave(page, (s) => s.selected_enhancer === 'Restore Ultra');
  await restorer(page).selectOption('laplacian');
  const lap = await lapSave;
  expect(lap).toMatchObject({ selected_enhancer: 'Restore Ultra', restore_ultra_frequency_blend: true, auto_bypass_restorer_native_swap: false });
  await expect(page.getByTestId('restorer-note')).toContainText('recombined by frequency');
  await expect(restorer(page).locator('option')).toHaveCount(3);          // the "Custom" entry went away with UltraMax

  const bypassSave = nextSave(page, (s) => s.auto_bypass_restorer_native_swap === true);
  await restorer(page).selectOption('bypass');
  const bypass = await bypassSave;
  expect(bypass.selected_enhancer, 'bypass keeps the restorer, it only turns the bypass on').toBe('Restore Ultra');
  await expect(page.getByTestId('restorer-note')).toContainText('≤320 px');   // alpha_256 is a swapper the bypass applies to
});

test('faceswap: the stride and the smart-skip toggle work from the keyboard alone', async ({ page }) => {
  await openTab(page, 'faceswap');
  const range = stride(page);
  await expect(range).toHaveAttribute('min', '1');
  await expect(range).toHaveAttribute('max', '8');
  await expect(page.getByTestId('stride-note')).toContainText('K = 1: Detect every frame');

  await range.focus();
  const save = nextSave(page, (s) => s.prepass_detect_interval === 4);
  for (let i = 0; i < 3; i += 1) await page.keyboard.press('ArrowRight');
  await expect(page.getByTestId('stride-note')).toContainText('K = 4: Detect every 4th frame, carry the rest by optical flow');
  expect((await save).prepass_detect_interval).toBe(4);
  await page.keyboard.press('End');
  await expect(page.getByTestId('stride-note')).toContainText('K = 8');

  const toggle = smartSkip(page);
  await toggle.focus();
  const skipSave = nextSave(page, (s) => s.temporal_smart_skip === true);
  await page.keyboard.press('Space');
  await expect(toggle).toBeChecked();
  expect((await skipSave).temporal_smart_skip).toBe(true);
});

test('faceswap: the controls added no second Start (the run bar stays the one run control)', async ({ page }) => {
  await openTab(page, 'faceswap');
  await expect(page.getByRole('button', { name: /^(▶\s*)?Start( Swap(ping)?)?$/ })).toHaveCount(1);
  const own = controls(page).getByRole('button');
  for (const b of await own.all()) expect(await b.textContent()).toMatch(/^(\?|MOD)$/);   // info badges only
});

test('settings: the same four controls, bound to the same settings (an edit on one tab is the other\'s value)', async ({ page }) => {
  await startOnSettingsTab(page, 'general');
  await openTab(page, 'settings');
  await expect(page.getByRole('heading', { name: '3-stage pipeline' })).toBeVisible();
  await expect(controls(page)).toHaveCount(1);
  for (const key of ['swap_model', 'selected_enhancer', 'auto_bypass_restorer_native_swap', 'prepass_detect_interval', 'temporal_smart_skip']) {
    await expect(page.locator(`[data-setting="${key}"]`), `${key} is addressable from the command palette`).toHaveCount(1);
  }

  await swapper(page).selectOption('hyperswap_512');
  await stride(page).focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await expect(page.getByTestId('stride-note')).toContainText('K = 3');

  await page.goto('/#/faceswap');
  await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
  await expect(swapper(page)).toHaveValue('hyperswap_512');
  await expect(stride(page)).toHaveValue('3');
});

test('settings: searching for a pipeline control keeps its section; an unrelated search hides it', async ({ page }) => {
  await openTab(page, 'settings');
  const search = page.getByPlaceholder(/search/i).first();
  await search.fill('smart-skip');
  await expect(page.getByTestId('pipeline-controls')).toHaveCount(1);
  await search.fill('zzzz-no-such-setting');
  await expect(page.getByTestId('pipeline-controls')).toHaveCount(0);
});

// ── telemetry ──────────────────────────────────────────────────────────────────────────────────────────────────────

const frame = (extra) => ({
  event: 'progress', processing: true, paused: false, progress: 0.4, desc: 'Processing frame 144 / 360',
  error: '', current_frame: 144, total_frames: 360, fps: 30, fps_now: 30, frame_ms: 33, eta_s: 7,
  started_at: Date.now() / 1000 - 20, live_seq: 0, engine_state: 'IDLE', engine_label: '', engine_model: '',
  engine_precision: '', engine_since: null, ...extra,
});
const common = { active_ranges_count: 2, verified_target_frames: 180, skipped_frames_pct: 46.7, eta_seconds: 0 };
const stages = (flowBackend, ops) => ({
  [S1]: { ...common, state: 'DONE', stage_progress_pct: 100, wall_s: 6.2, fps: 10, scene_cuts: 3 },
  [S2]: { ...common, state: 'DONE', stage_progress_pct: 100, wall_s: 4.5, fps: 20, neural_detected_frames: 25, flow_tracked_frames: 75, flow_ratio_pct: 75, flow_backend: flowBackend },
  [S3]: { ...common, state: 'RUNNING', stage_progress_pct: 40, wall_s: 5.5, fps: 30, eta_seconds: 7, ops },
});
const OPS = {
  decode: { ms: 2.1, calls: 144, backend: 'NVDEC' },
  warp: { ms: 3.4, calls: 144, backend: 'cv2 (CPU)' },
  model: { ms: 11, calls: 144, backend: 'TensorRT (CUDA graph)' },
  restore: null,
  blend: { ms: 0.4, calls: 144, backend: 'Triton (fused)' },
  encode: { ms: 1.2, calls: 144, backend: 'hevc_nvenc' },
};

/** Scripted telemetry socket + a /api/progress that agrees with it and carries a given runtime report. */
async function scriptProcessing(page, frames, runtime) {
  await page.route('**/api/progress', async (route) => {
    const res = await route.fetch();
    let body = {};
    try { body = await res.json(); } catch { body = {}; }
    await route.fulfill({ response: res, json: { ...body, processing: true, ...(runtime ? { runtime } : {}) } });
  });
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => {
    let i = 0;
    const send = () => { ws.send(JSON.stringify(frames[Math.min(i, frames.length - 1)])); i += 1; };
    send();
    const timer = setInterval(send, 400);
    ws.onClose(() => clearInterval(timer));
  });
}

const precision = (verdicts) => ({
  sections: { PRECISION: { status: 'AVAILABLE', source: 'runtime configuration/profile', values: {
    configured: 'mixed', effective: 'mixed', surgeon_version: 'v2', canary_verdicts: verdicts } } },
});

test('processing: Stage 1 keyframes/s, Stage 2 "NVOFA tracked %", Stage 3 TRT Graph | Triton Blend | NVENC ms', async ({ page }) => {
  await scriptProcessing(page, [frame({ event: 'hello', stage: S3, stages: stages('nvof', OPS), summary: {} })],
    precision({ xseg: { passed: true, reason: '' }, '1k3d68': { passed: true, reason: '' } }));
  await page.goto('/#/processing');

  const live = page.getByTestId('pipeline-live');
  await expect(live).toBeVisible();
  await expect(live.locator(`[data-stage="${S1}"]`)).toContainText('10.0 keyframes/s');
  await expect(live.locator(`[data-stage="${S1}"]`)).toContainText('3 scene cuts');
  await expect(page.getByTestId(`${S2}-detail`)).toContainText('75.0% NVOFA tracked');
  const ops = page.getByTestId('stage3-ops');
  await expect(ops).toContainText('Decode 2.10 ms');
  await expect(ops).toContainText('TRT Graph 11.0 ms');
  await expect(ops).toContainText('Triton Blend 0.40 ms');
  await expect(ops).toContainText('NVENC 1.20 ms');
  await expect(ops, 'the warp stays a CPU op, labelled as such').toContainText('Warp 3.40 ms');

  const canary = page.getByTestId('canary-status');
  await expect(canary).toHaveAttribute('data-state', 'PASSED');
  await expect(canary).toContainText('mixed → mixed');
  await expect(canary.locator('[data-canary="xseg"]')).toContainText('✓ pass');
});

test('processing: an engine the backend did not name is plain "flow", and a failed canary says FAILED with its reason', async ({ page }) => {
  await scriptProcessing(page, [frame({ event: 'hello', stage: S3, stages: stages(null, { ...OPS, model: { ms: 11, calls: 144, backend: 'TensorrtExecutionProvider' }, blend: { ms: 4, calls: 144, backend: 'cv2 (CPU)' } }), summary: {} })],
    precision({ xseg: { passed: true, reason: '' }, '1k3d68': { passed: false, reason: 'ssim 0.81 below the 0.95 floor' } }));
  await page.goto('/#/processing');

  await expect(page.getByTestId(`${S2}-detail`)).toContainText('75.0% flow');
  await expect(page.getByTestId(`${S2}-detail`)).not.toContainText('NVOFA');
  const ops = page.getByTestId('stage3-ops');
  await expect(ops).toContainText('TRT 11.0 ms');
  await expect(ops).not.toContainText('Graph');
  await expect(ops, 'a cv2 blend is not called Triton').not.toContainText('Triton');
  await expect(ops).toContainText('Blend 4.00 ms');

  const canary = page.getByTestId('canary-status');
  await expect(canary).toHaveAttribute('data-state', 'FAILED');
  await expect(canary.locator('[data-canary="1k3d68"]')).toContainText('✗ fail · ssim 0.81 below the 0.95 floor');
});

test('processing: a canary running now is shown', async ({ page }) => {
  await scriptProcessing(page, [
    frame({ event: 'hello', stage: S3, stages: stages('cv', OPS), summary: {}, engine_state: 'RUNNING_CANARY', engine_label: 'Verifying TensorRT Engine...', engine_model: 'gpen_256', engine_precision: 'FP16' }),
  ], precision({}));
  await page.goto('/#/processing');
  const canary = page.getByTestId('canary-status');
  await expect(canary).toHaveAttribute('data-state', 'RUNNING');
  await expect(page.getByTestId('canary-live')).toContainText('gpen_256 · FP16');
});

test('processing: during a run with no verdicts recorded the status says NONE - never "passed"', async ({ page }) => {
  await scriptProcessing(page, [frame({ event: 'hello', stage: S3, stages: stages('nvof', OPS), summary: {} })], precision({}));
  await page.goto('/#/processing');
  const canary = page.getByTestId('canary-status');
  await expect(canary).toHaveAttribute('data-state', 'NONE');
  await expect(canary).toContainText('No canary verdict recorded yet');
  await expect(canary).not.toContainText('PASSED');
});

test('processing: the structured runtime report leads its PRECISION card with the canary status', async ({ page }) => {
  const verdicts = Object.fromEntries(['a', 'b', 'c', 'd', 'e', 'f'].map((k) => [`model_${k}`, { passed: true, reason: '' }]));
  verdicts.model_z = { passed: false, reason: 'cosine 0.91' };
  await scriptProcessing(page, [frame({ event: 'hello', stage: S3, stages: stages('nvof', OPS), summary: {} })], precision(verdicts));
  await page.goto('/#/processing');
  await page.getByTitle('Toggle structured runtime report').click();
  const card = page.locator('div', { has: page.getByText('PRECISION', { exact: true }) }).filter({ hasText: 'canary precision' }).last();
  await expect(card).toContainText('canary precision');
  await expect(card).toContainText('FAILED (6 pass, 1 fail)');
  await expect(card).toContainText('canary.model_z');
});

test('settings: the canary precision status sits under the last-run report', async ({ page }) => {
  await page.route('**/api/runtime/state', (route) => route.fulfill({
    json: { state: 'ready', initialized: true, ...precision({ xseg: { passed: true, reason: '' } }),
      sections: { ...precision({ xseg: { passed: true, reason: '' } }).sections, PERFORMANCE: { status: 'AVAILABLE', values: { pipeline_report: 'UNKNOWN' } } } },
  }));
  await startOnSettingsTab(page, 'diagnostics');
  await openTab(page, 'settings');
  await expect(page.getByTestId('canary-status')).toHaveAttribute('data-state', 'PASSED');
});

test('axe: the pipeline controls and every telemetry state above have no violations', async ({ page }) => {
  await openTab(page, 'faceswap');
  const onFaceSwap = await new AxeBuilder({ page }).include('[data-testid="pipeline-controls"]').withTags(AXE_TAGS).analyze();
  expect(onFaceSwap.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`)).toEqual([]);

  // Same document as Face Swap (a hash change), so seeding the stored tab would not be re-read: click it.
  await openTab(page, 'settings');
  await page.getByRole('tab', { name: /^Appearance/ }).click();
  const onSettings = await new AxeBuilder({ page }).include('[data-testid="pipeline-controls"]').withTags(AXE_TAGS).analyze();
  expect(onSettings.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`)).toEqual([]);
});

test('axe: the processing HUD with a failed canary and live engine labels', async ({ page }) => {
  await scriptProcessing(page, [frame({ event: 'hello', stage: S3, stages: stages('nvof', OPS), summary: {} })],
    precision({ xseg: { passed: true, reason: '' }, '1k3d68': { passed: false, reason: 'ssim 0.81' } }));
  await page.goto('/#/processing');
  await expect(page.getByTestId('canary-status')).toBeVisible();
  const { violations } = await new AxeBuilder({ page }).include('[data-testid="pipeline-hud"]').withTags(AXE_TAGS).analyze();
  expect(violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`)).toEqual([]);
});
