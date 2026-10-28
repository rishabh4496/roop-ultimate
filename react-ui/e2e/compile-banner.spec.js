import { test, expect } from '@playwright/test';
import { MOCK_PORT } from './playwright.config.js';
import { openTab } from './helpers.js';

// The backend announces STAGE_COMPILING_TRT / STAGE_RUNNING_CANARY on /ws/v1/pipeline/status
// (app/roop/pipeline_status.py). The mock replays that channel and is driven through
// POST /__mock/pipeline. Two things must hold in a real browser:
//   * a non-blocking banner names the model and the precision (FP32 vs FP16), survives
//     hash-route navigation, and goes away when the stage ends;
//   * a request that outlives the 15 s deadline while a stage is active is NOT timed out;
//     it gets a fresh deadline when the stage ends.

const mock = (request, active) => request.post(`http://127.0.0.1:${MOCK_PORT}/__mock/pipeline`, { data: { active } });
const stage = (kind, model, precision, elapsed_s = 3) => ({ kind, model, precision, elapsed_s });

test.afterEach(async ({ request }) => { await mock(request, []); });

test('banner: names model + precision, persists across routes, clears when the stage ends', async ({ page, request }) => {
  await openTab(page, 'faceswap');
  const banner = page.getByTestId('compile-banner');
  await expect(banner).toHaveCount(0);

  await mock(request, [stage('STAGE_COMPILING_TRT', 'gpen_2048', 'FP16')]);
  await expect(banner).toBeVisible();
  await expect(banner).toContainText('Compiling TensorRT engine');
  await expect(banner.locator('[data-field="model"]')).toHaveText('gpen_2048');
  await expect(banner.locator('[data-field="precision"]')).toHaveText('FP16');
  await expect(banner, 'a status, never a modal').toHaveAttribute('role', 'status');

  // Not tied to the tab that was open when it started.
  for (const id of ['batch', 'processing', 'settings']) {
    await page.evaluate((h) => { window.location.hash = `#/${h}`; }, id);
    await expect(page).toHaveURL(new RegExp(`#/${id}$`));
    await expect(banner, `still shown on #/${id}`).toBeVisible();
  }

  // The canary follows the build; an FP32 model says FP32.
  await mock(request, [stage('STAGE_RUNNING_CANARY', 'xseg', 'FP32')]);
  await expect(banner).toContainText('Verifying engine fidelity');
  await expect(banner.locator('[data-field="precision"]')).toHaveText('FP32');

  await mock(request, []);
  await expect(banner).toHaveCount(0);

  // Blocking nothing: the page under it stayed interactive the whole time.
  await mock(request, [stage('STAGE_COMPILING_TRT', 'codeformer', 'FP16')]);
  await expect(banner).toBeVisible();
  await page.getByRole('button', { name: /^Settings$/ }).first().click().catch(() => {});
  await expect(page.locator('nav[aria-label="Main"] button[aria-current="page"]')).toBeVisible();
});

const compileDetail = { stage: 'TRT_COMPILE', workspace_mb: 2048, cache_path: 'C:/roop/trt_cache/fp16_mixed' };
const lastCompile = (cache_hit, elapsed_s) => ({
  kind: 'STAGE_COMPILING_TRT', state: 'finished', model: 'gpen_256', precision: 'FP16', cache_hit, ok: true,
  elapsed_s, time: Date.now() / 1000, ...compileDetail,
});

test('banner: shows workspace + cache directory while building, then a hit/miss receipt that expires', async ({ page, request }) => {
  test.setTimeout(60_000);
  await openTab(page, 'faceswap');
  const banner = page.getByTestId('compile-banner');

  await mock(request, [{ ...stage('STAGE_COMPILING_TRT', 'gpen_256', 'FP16'), ...compileDetail }]);
  await expect(banner.locator('[data-field="workspace"]')).toHaveText('workspace 2048 MB');
  await expect(banner.locator('[data-field="cache"]')).toHaveText('cache C:/roop/trt_cache/fp16_mixed');
  await expect(banner.locator('[data-field="precision"]')).toHaveText('FP16');

  // The build ends having written a plan: a MISS, with how long it took. Not a modal, and it goes away by itself.
  await request.post(`http://127.0.0.1:${MOCK_PORT}/__mock/pipeline`, { data: { active: [], last_compile: lastCompile(false, 143) } });
  const receipt = page.getByTestId('compile-receipt');
  await expect(receipt).toBeVisible();
  await expect(receipt).toHaveAttribute('data-cache-hit', 'false');
  await expect(receipt).toContainText('TRT cache miss · built in 2m 23s');
  await expect(banner).toHaveAttribute('role', 'status');
  await expect(banner, 'the receipt expires on its own').toHaveCount(0, { timeout: 15_000 });

  // The next build deserialises from the cache: a HIT, with its sub-second time.
  await mock(request, [{ ...stage('STAGE_COMPILING_TRT', 'gpen_256', 'FP16'), ...compileDetail }]);
  await expect(banner).toBeVisible();
  await request.post(`http://127.0.0.1:${MOCK_PORT}/__mock/pipeline`, { data: { active: [], last_compile: lastCompile(true, 0.18) } });
  await expect(receipt).toHaveAttribute('data-cache-hit', 'true');
  await expect(receipt).toContainText('TRT cache hit · 0.18s');
});

test('banner: a build that finished before the page connected leaves no receipt', async ({ page, request }) => {
  await request.post(`http://127.0.0.1:${MOCK_PORT}/__mock/pipeline`, { data: { active: [], last_compile: lastCompile(false, 90) } });
  await openTab(page, 'faceswap');
  await page.waitForTimeout(800);
  await expect(page.getByTestId('compile-banner')).toHaveCount(0);
  await request.post(`http://127.0.0.1:${MOCK_PORT}/__mock/pipeline`, { data: { active: [], last_compile: null } });
});

test('deadline: a 15 s request is not timed out while compiling, and gets a fresh deadline after', async ({ page, request }) => {
  test.setTimeout(90_000);
  const logs = [];
  page.on('console', (m) => { if (/\[roop-ui\]/.test(m.text())) logs.push(m.text()); });
  await mock(request, [stage('STAGE_COMPILING_TRT', 'gpen_2048', 'FP16')]);
  await page.clock.install();
  await page.route('**/api/settings/defaults', () => new Promise(() => {}));     // never answers

  const asked = page.waitForRequest(/\/api\/settings\/defaults/);
  await page.goto('/#/settings');
  await page.locator('nav[aria-label="Main"] button[aria-current="page"]').waitFor();
  await asked;

  await page.clock.fastForward(20_000);                                          // well past 15 s
  await page.waitForTimeout(300);
  expect(logs.filter((l) => /defaults/.test(l)), 'compiling: no timeout at 20 s').toHaveLength(0);
  await expect(page.getByTestId('compile-banner')).toBeVisible();

  await mock(request, []);                                                        // the build ends
  // The confirm poll (2 s) notices; the request then gets a fresh 15 s.
  for (let i = 0; i < 6; i++) { await page.clock.fastForward(2_000); await page.waitForTimeout(150); }
  expect(logs.filter((l) => /defaults/.test(l)), 'a fresh deadline: not yet').toHaveLength(0);
  await page.clock.fastForward(16_000);
  await expect.poll(() => logs.find((l) => /Loading the setting defaults failed/.test(l)), { timeout: 10_000 })
    .toMatch(/Request timed out after \d+ s \(GET \/api\/settings\/defaults\)/);
  // (The banner going away is asserted in the first test; under this test's fake clock framer's
  // exit animation cannot finish, so the DOM node would outlive the state.)
});
