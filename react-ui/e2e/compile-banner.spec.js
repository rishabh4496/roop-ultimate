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
  await expect(page.locator('header nav button[aria-current="page"]')).toBeVisible();
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
  await page.locator('header nav button[aria-current="page"]').waitFor();
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
