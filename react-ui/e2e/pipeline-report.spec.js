import { test, expect } from '@playwright/test';
import { readFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// The per-job pipeline report in a real browser, fed a REAL report (the `cfr_3stage_trim` run of
// app/tests/verify_stage4_report.py: a trimmed 3-stage render of the Phase 5 clip through the production path).
//
//   * #/processing  the Diagnostics HUD shows the three stages live from /ws/telemetry, then, when the backend's
//                   frame carries `summary`, the finished report in their place;
//   * #/settings    "Last run report" reads PERFORMANCE.pipeline_report of /api/runtime/state;
//   * #/history     a run's `report` is a collapsed card that opens to the stage blocks; a run recorded before the
//                   report existed shows no card at all (not a table of zeros).
//
// The app's own request/WebSocket traffic is intercepted in the page, so nothing here depends on the mock
// server's invented numbers; what is under test is what the SHELL does with the backend's real payloads.

const here = dirname(fileURLToPath(import.meta.url));
const report = JSON.parse(readFileSync(join(here, '..', '.render-check', 'pipeline-report.fixture.json'), 'utf8'));
const SHOTS = process.env.PIPELINE_REPORT_SHOTS || '';
if (SHOTS) mkdirSync(SHOTS, { recursive: true });
const shot = (page, name) => (SHOTS ? page.screenshot({ path: join(SHOTS, name), fullPage: true }) : Promise.resolve());

const S1 = 'STAGE_1_SPARSE_PRESCAN';
const S2 = 'STAGE_2_ACTIVE_PREPASS';
const S3 = 'STAGE_3_MAIN_SWAP';

const baseFrame = (extra) => ({
  event: 'progress', processing: true, paused: false, progress: 0.4, desc: 'Processing frame 144 / 360',
  error: '', current_frame: 144, total_frames: 360, fps: 30, fps_now: 30, frame_ms: 33, eta_s: 7,
  started_at: Date.now() / 1000 - 20, live_seq: 0, engine_state: 'IDLE', engine_label: '', engine_model: '',
  engine_precision: '', engine_since: null, ...extra,
});

const liveStages = {
  [S1]: { state: 'DONE', stage_progress_pct: 100, active_ranges_count: 2, verified_target_frames: null, skipped_frames_pct: 46.7, eta_seconds: 0, wall_s: 6.2 },
  [S2]: { state: 'DONE', stage_progress_pct: 100, active_ranges_count: 2, verified_target_frames: 180, skipped_frames_pct: 46.7, eta_seconds: 0, wall_s: 4.49 },
  [S3]: { state: 'RUNNING', stage_progress_pct: 40, active_ranges_count: 2, verified_target_frames: 180, skipped_frames_pct: 46.7, eta_seconds: 7, wall_s: 5.5 },
};

/** Replace /ws/telemetry with a scripted socket: `frames` are sent in order, `gap` ms apart, the last repeated. */
async function scriptTelemetry(page, frames, gap = 250) {
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => {
    let i = 0;
    const send = () => { ws.send(JSON.stringify(frames[Math.min(i, frames.length - 1)])); i += 1; };
    send();
    const timer = setInterval(send, gap);
    ws.onClose(() => clearInterval(timer));
  });
}

test('processing: live stages, then the finished report replaces them', async ({ page }) => {
  // frame 0: hello with the live stages; later frames carry the report (the run ended)
  const frames = [
    baseFrame({ event: 'hello', stage: S3, stages: liveStages, summary: {} }),
    baseFrame({ stage: S3, stages: liveStages, summary: {} }),
    baseFrame({ processing: false, progress: 1, desc: 'Done', stage: null, stages: liveStages, summary: report }),
  ];
  await scriptTelemetry(page, frames, 1500);
  await page.goto('/#/processing');

  const live = page.getByTestId('pipeline-live');
  await expect(live).toBeVisible();
  await expect(live).toContainText('Stage 1 · Sparse pre-scan');
  await expect(live.locator(`[data-stage="${S3}"]`)).toContainText('RUNNING');
  await expect(live.locator(`[data-stage="${S3}"]`)).toContainText('40.0%');
  await expect(live.locator(`[data-stage="${S1}"]`)).toContainText('6.20s');
  await shot(page, 'ui_processing_live.png');

  const card = page.getByTestId('pipeline-report');
  await expect(card, 'the finished report arrives on the same socket').toBeVisible({ timeout: 15_000 });
  await expect(live, 'and replaces the live rows').toHaveCount(0);
  await expect(card).toContainText('verified');
  await expect(card).toContainText('speedup ≈ 1.35× (estimate)');
  for (const label of ['Stage 1 · Sparse pre-scan', 'Stage 2 · Active-range pre-pass', 'Stage 3 · Main swap', 'Stage 4 · Finalize']) {
    await expect(card).toContainText(label);
  }
  const s3 = card.locator(`[data-stage="${S3}"]`);
  await expect(s3).toContainText('Zero-GPU passthrough frames');
  await expect(s3).toContainText('168');
  await expect(s3).toContainText('192');
  await expect(card.locator('[data-stage="STAGE_4_FINALIZE"]')).toContainText('passed');
  await shot(page, 'ui_processing_report.png');
});

test('settings: "Last run report" is the runtime state\'s pipeline_report', async ({ page }) => {
  await page.route('**/api/runtime/state', (route) => route.fulfill({
    json: { state: 'ready', initialized: true, sections: { PERFORMANCE: { status: 'AVAILABLE', source: 'runtime monitor/profile', values: { pipeline_report: report } } } },
  }));
  await page.goto('/#/settings');
  const card = page.getByTestId('pipeline-report');
  await expect(card).toBeVisible();
  await expect(page.getByText('The render that finished most recently in this session.')).toBeVisible();
  await expect(card).toContainText('360 frames');
  await expect(card.locator(`[data-stage="${S2}"]`)).toContainText('cap.set() seeks');
  await expect(card.locator(`[data-stage="${S3}"]`)).toContainText('Peak VRAM');
  await shot(page, 'ui_settings_report.png');
});

test('settings: with no live report the newest history entry that has one is used, and says so', async ({ page }) => {
  await page.route('**/api/runtime/state', (route) => route.fulfill({
    json: { state: 'ready', initialized: true, sections: { PERFORMANCE: { status: 'AVAILABLE', values: { pipeline_report: 'UNKNOWN' } } } },
  }));
  await page.route('**/api/history', (route) => route.fulfill({
    json: { entries: [
      { id: 2, time: 1790000000, outputs: ['older.mp4'], settings: {}, duration_s: 10, frames: 100, fps: 10 },
      { id: 1, time: 1789990000, outputs: ['with_report.mp4'], settings: {}, duration_s: 50, frames: 360, fps: 7, report },
    ] },
  }));
  await page.goto('/#/settings');
  await expect(page.getByText(/From History: with_report\.mp4/)).toBeVisible();
  await expect(page.getByTestId('pipeline-report')).toBeVisible();
});

test('history: a run with a report has a collapsed card that opens; an older run has none', async ({ page }) => {
  await page.route('**/api/history', (route) => route.fulfill({
    json: { entries: [
      { id: 2, time: 1790000000, outputs: ['older.mp4'], settings: { enhancer: 'GPEN' }, duration_s: 10, frames: 100, fps: 10 },
      { id: 1, time: 1789990000, outputs: ['with_report.mp4'], settings: { enhancer: 'GPEN' }, duration_s: 50, frames: 360, fps: 7, report },
    ] },
  }));
  await page.goto('/#/history');
  await expect(page.getByTitle('with_report.mp4')).toBeVisible();
  await expect(page.getByTitle('older.mp4')).toBeVisible();
  const cards = page.getByTestId('pipeline-report');
  await expect(cards, 'only the run that has a report gets a card').toHaveCount(1);
  const card = cards.first();
  expect(await card.evaluate((el) => el.open), 'collapsed by default in a list').toBe(false);
  await card.locator('summary').click();
  expect(await card.evaluate((el) => el.open)).toBe(true);
  await expect(card.locator(`[data-stage="${S3}"]`)).toContainText('GPU frames (swap / enhance)');
  await shot(page, 'ui_history_report.png');
});

test('a failed Stage 4 is shown as FAILED with its reason', async ({ page }) => {
  const failed = JSON.parse(JSON.stringify(report));
  failed.ok = false;
  failed.stages.STAGE_4_FINALIZE = { ...failed.stages.STAGE_4_FINALIZE, ok: false, state: 'FAILED', problems: ['the audio mux failed'] };
  await page.route('**/api/runtime/state', (route) => route.fulfill({
    json: { state: 'ready', initialized: true, sections: { PERFORMANCE: { status: 'AVAILABLE', values: { pipeline_report: failed } } } },
  }));
  await page.goto('/#/settings');
  const card = page.getByTestId('pipeline-report');
  await expect(card).toContainText('FAILED');
  await expect(card).toContainText('the audio mux failed');
});
