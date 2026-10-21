import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// The 3-Step Pipeline Pill (1 Pre-Scan -> 2 Range Lock -> 3 Render & Mux) in a real browser, driven by a scripted
// /ws/telemetry the way the backend's frames arrive (stage_telemetry.snapshot()): on #/processing and on
// #/faceswap, with axe run WHILE the pill is on screen (the app's bar is zero violations, a11y.spec.js).
//
// The pill reads the socket only. This spec also counts the requests the page makes while the stages advance:
// moving from stage to stage must not add a single REST call (nothing in the pill can then run into the 15 s API
// deadline, however long a stage takes).

const here = dirname(fileURLToPath(import.meta.url));
const report = JSON.parse(readFileSync(join(here, '..', '.render-check', 'pipeline-report.fixture.json'), 'utf8'));
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];

const S1 = 'STAGE_1_SPARSE_PRESCAN';
const S2 = 'STAGE_2_ACTIVE_PREPASS';
const S3 = 'STAGE_3_MAIN_SWAP';

const frame = (extra) => ({
  event: 'progress', processing: true, paused: false, progress: 0.4, desc: 'Processing frame 144 / 360',
  error: '', current_frame: 144, total_frames: 360, fps: 11.8, fps_now: 14.4, frame_ms: 69, eta_s: 7,
  started_at: Date.now() / 1000 - 20, live_seq: 0, engine_state: 'IDLE', engine_label: '', engine_model: '',
  engine_precision: '', engine_since: null, ...extra,
});
const st = (state, pct, extra = {}) => ({ state, stage_progress_pct: pct, active_ranges_count: 2, verified_target_frames: 180, skipped_frames_pct: 46.7, eta_seconds: 5, wall_s: 4.2, ...extra });

const STAGE2 = { [S1]: st('DONE', 100), [S2]: st('RUNNING', 35), [S3]: st('PENDING', 0) };
const STAGE3 = { [S1]: st('DONE', 100), [S2]: st('DONE', 100), [S3]: st('RUNNING', 40) };
const MUXING = { [S1]: st('DONE', 100), [S2]: st('DONE', 100), [S3]: st('DONE', 100) };

/**
 * Replace /ws/telemetry with a scripted socket: `frames` in order, `gap` ms apart, the last one repeated.
 * The app also polls GET /api/progress (the socket's fallback) and the mock answers "idle", which would flip
 * `processing` off between two scripted frames; the poll is answered with the script's own `processing` so the two
 * channels agree, as they do against the real backend.
 */
async function scriptTelemetry(page, frames, gap = 700) {
  const live = { processing: !!frames[0].processing };
  await page.route('**/api/progress', async (route) => {
    const res = await route.fetch();
    let body = {};
    try { body = await res.json(); } catch { body = {}; }
    await route.fulfill({ response: res, json: { ...body, processing: live.processing } });
  });
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => {
    let i = 0;
    const send = () => {
      const f = frames[Math.min(i, frames.length - 1)];
      live.processing = !!f.processing;
      ws.send(JSON.stringify(f));
      i += 1;
    };
    send();
    const timer = setInterval(send, gap);
    ws.onClose(() => clearInterval(timer));
  });
}

const script = [
  frame({ event: 'hello', stage: S2, stages: STAGE2, summary: {} }),
  frame({ stage: S3, stages: STAGE3, summary: {} }),
  frame({ progress: 0.99, stage: null, stages: MUXING, summary: {} }),
  frame({ processing: false, progress: 1, desc: 'Done', stage: null, stages: MUXING, summary: report }),
];

test('processing: the pill walks Range Lock -> Render -> Muxing -> done with the backend\'s numbers', async ({ page }) => {
  await scriptTelemetry(page, script, 1600);
  await page.goto('/#/processing');

  const pill = page.getByTestId('pipeline-pill');
  await expect(pill).toBeVisible();
  // step order and labels
  const order = pill.locator('li[data-step]');
  await expect(order).toHaveCount(3);
  await expect(order.nth(0)).toContainText('Pre-Scan');
  await expect(order.nth(1)).toContainText('Range Lock');
  await expect(order.nth(2)).toContainText('Render & Mux');

  // frame 0: Stage 2 running at 35%
  await expect(order.nth(0)).toHaveAttribute('data-state', 'done');
  await expect(order.nth(1)).toHaveAttribute('data-state', 'active');
  await expect(order.nth(1)).toHaveAttribute('aria-current', 'step');
  await expect(order.nth(1)).toContainText('35.0%');
  await expect(order.nth(2)).toHaveAttribute('data-state', 'pending');
  await expect(page.getByTestId('pill-culled')).toHaveText('46.7% culled');
  await expect(page.getByTestId('pill-fps'), 'fps only beside step 3').toHaveText('');

  // axe with the pill on screen, mid-run
  const { violations } = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  expect(violations.map((v) => ({ id: v.id, targets: v.nodes.slice(0, 3).map((n) => n.target.join(' ')) })), 'axe on #/processing while the pill shows').toEqual([]);

  // Stage 3: step 3 active, fps shown, exactly one aria-current
  await expect(order.nth(2)).toHaveAttribute('data-state', 'active', { timeout: 10_000 });
  await expect(order.nth(2)).toContainText('40.0%');
  await expect(page.getByTestId('pill-fps')).toHaveText('14.4 fps');
  await expect(pill.locator('[aria-current="step"]')).toHaveCount(1);
  await expect(page.getByTestId('pill-announce')).toContainText('Step 3 of 3, Render & Mux');

  // Stage 3 over, Stage 4 (mux) not reported yet: step 3 stays active as Muxing
  await expect(order.nth(2)).toContainText('Muxing', { timeout: 10_000 });
  await expect(page.getByTestId('pill-announce')).toContainText('muxing audio');

  // the report arrives: every step done
  await expect(order.nth(2)).toHaveAttribute('data-state', 'done', { timeout: 10_000 });
  await expect(order.nth(0)).toHaveAttribute('data-state', 'done');
  await expect(page.getByTestId('pill-announce')).toHaveText('Render finished');
});

test('processing: stage changes add no REST request (the pill is socket-only)', async ({ browser }) => {
  // Two arms run the SAME scripted socket: A carries the stage tables (the pill shows), B the same frames with the
  // tables emptied (no pill). Whatever the shell requests on its own - polls, the end-of-run refreshes - it requests
  // in both; a request that only arm A makes would be the pill's. Distinct `METHOD path` sets, the whole script.
  const strip = (f) => ({ ...f, stage: null, stages: {}, summary: {} });
  const arm = async (frames, pillExpected) => {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    const seen = new Set();
    page.on('request', (r) => { if (/\/api\//.test(r.url())) seen.add(`${r.method()} ${new URL(r.url()).pathname}`); });
    await scriptTelemetry(page, frames, 700);
    await page.goto('/#/processing');
    await page.locator('header nav button[aria-current="page"]').waitFor();
    if (pillExpected) await expect(page.locator('li[data-step="render"]')).toHaveAttribute('data-state', 'done', { timeout: 15_000 });
    else await page.waitForTimeout(4500);
    await page.waitForTimeout(1500);
    await expect(page.getByTestId('pipeline-pill')).toHaveCount(pillExpected ? 1 : 0);
    await ctx.close();
    return seen;
  };
  const withPill = await arm(script, true);
  const without = await arm(script.map(strip), false);
  const onlyPill = [...withPill].filter((r) => !without.has(r));
  expect(onlyPill, 'requests made only when the pill is on screen').toEqual([]);
});

test('face swap: the run bar carries the pill while a job runs, and axe is clean with it', async ({ page }) => {
  await scriptTelemetry(page, [frame({ event: 'hello', stage: S3, stages: STAGE3, summary: {} })], 700);
  await page.goto('/#/faceswap');
  await page.locator('header nav button[aria-current="page"]').waitFor();
  const pill = page.getByTestId('pipeline-pill');
  await expect(pill, 'visible in the Face Swap run bar while processing').toBeVisible({ timeout: 15_000 });
  await expect(pill.locator('li[data-step="render"]')).toHaveAttribute('aria-current', 'step');
  await expect(pill.getByTestId('pill-culled')).toHaveText('46.7% culled');
  const { violations } = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  expect(violations.map((v) => ({ id: v.id, targets: v.nodes.slice(0, 3).map((n) => n.target.join(' ')) })), 'axe on #/faceswap while the pill shows').toEqual([]);
});

test('face swap: no pill while idle', async ({ page }) => {
  await scriptTelemetry(page, [frame({ event: 'hello', processing: false, progress: 0, desc: '', stage: null, stages: {}, summary: {} })], 700);
  await page.goto('/#/faceswap');
  await page.locator('header nav button[aria-current="page"]').waitFor();
  await page.waitForTimeout(1500);
  await expect(page.getByTestId('pipeline-pill')).toHaveCount(0);
});

// The run bar's own controls are part of the bar the pill sits in. "Pause Render" (white on amber-600, 3.19:1) and
// "Resume Render" (white on emerald-600, 3.7:1) failed color-contrast, but only while a run was in flight - the
// idle-state axe suite (a11y.spec.js) never sees them - so the running and paused states are audited here.
for (const [name, extra] of [['running', {}], ['paused', { paused: true }]]) {
  test(`processing: axe is clean in the ${name} state (run bar controls + pill)`, async ({ page }) => {
    await scriptTelemetry(page, [frame({ event: 'hello', stage: S3, stages: STAGE3, summary: {}, ...extra })], 700);
    await page.route('**/api/progress', async (route) => {
      const res = await route.fetch();
      let body = {};
      try { body = await res.json(); } catch { body = {}; }
      await route.fulfill({ response: res, json: { ...body, processing: true, ...extra } });
    });
    await page.goto('/#/processing');
    await expect(page.getByTestId('pipeline-pill')).toBeVisible();
    await expect(page.getByRole('button', { name: name === 'paused' ? 'Resume Render' : 'Pause Render' })).toBeVisible();
    const { violations } = await new AxeBuilder({ page }).withTags(TAGS).analyze();
    expect(violations.map((v) => ({ id: v.id, targets: v.nodes.slice(0, 3).map((n) => n.target.join(' ')) }))).toEqual([]);
  });
}
