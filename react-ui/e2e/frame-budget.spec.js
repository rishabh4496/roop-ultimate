import { test, expect } from '@playwright/test';
import { TABS, openTab, makePng } from './helpers.js';

// Frame-time, long-task and React-commit measurements for the workstation, in a real (headless) Chromium
// against the production build, with the mock server standing in for app/api.py.
//
// What it measures, per scenario window:
//   * requestAnimationFrame deltas  -> p50 / p95 / max frame time and the share of frames over 20 ms
//   * `longtask` entries            -> main-thread blocks of >= 50 ms (count, worst)
//   * React commits per second      -> through a stub __REACT_DEVTOOLS_GLOBAL_HOOK__ (React calls
//                                      onCommitFiberRoot in production builds when a hook is present)
//
// Scenarios: every tab idle; a running job (the mock pushes /ws/telemetry and progress) on the three tabs that
// show it; a 600-event pointer flood with the Mask Brush and with the 3.5x Lens; a timeline drag.
//
// HONESTY ABOUT WHAT THIS IS. Headless Chromium rasterises in software, on whatever machine runs it, so the
// absolute frame times say nothing about a real GPU-backed browser. They are a main-thread jank budget: a
// regression that blocks the thread (a 30 Hz setState on a 4,000-line panel, a synchronous canvas redraw per
// pointer event) shows up here whatever the display does. The budgets below are therefore generous ceilings
// taken from measured numbers with headroom, not a claim of 60 FPS.

const WINDOW_MS = 4000;

// Ceilings, from the numbers measured on 2026-10-19 (headless Chromium, mock server) with headroom. Each is a statement
// about work, not about a GPU:
//   idle   measured 59.6-59.9 fps, 0% of frames over 20 ms, 0 long tasks, 1.7-2.5 React commits/s (background polls)
//   job    measured 59.4-59.8 fps, <= 0.4% over 20 ms, 4.2-4.7 commits/s: a 4 Hz telemetry feed costs ~2 extra commits/s,
//          because the store sits outside React state and the readouts are throttled leaf components (store/telemetryStore.js)
//   lens   600 pointermoves: 2.1 commits/s since the glass is moved by an rAF writer (InteractivePreview.jsx); it was 58/s
//          when the position was React state. This is the guard for that fix: the old code fails it by 5x.
// NOT asserted: brush painting (11% of frames over 20 ms, but only ~45 ms of the main thread's time is script: software
// canvas raster in headless, not something a main-thread change moves) and the timeline drag (about 94 commits/s of the
// panel that owns the playhead; frames stay at 59 fps here). Both are reported so a change shows up, and left open.
const BUDGET = {
  idle: { over20: 5, longTasks: 0, commitsPerSec: 6 },
  job: { over20: 5, longTasks: 0, commitsPerSec: 10 },
  lens: { commitsPerSec: 10 },
};

function expectWithin(r, budget) {
  for (const [k, ceiling] of Object.entries(budget)) {
    expect(r[k], `${r.label}: ${k} = ${r[k]} exceeds the budget of ${ceiling}`).toBeLessThanOrEqual(ceiling);
  }
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    const perf = { ts: [], long: [], commits: 0, on: false };
    window.__perf = perf;
    const tick = (t) => { if (perf.on) perf.ts.push(t); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    try {
      new PerformanceObserver((list) => {
        if (perf.on) for (const e of list.getEntries()) perf.long.push(e.duration);
      }).observe({ entryTypes: ['longtask'] });
    } catch { /* longtask unsupported: the count stays 0 and the report says so */ }
    window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
      supportsFiber: true, isDisabled: false, renderers: new Map(),
      inject(r) { this.renderers.set(1, r); return 1; },
      onCommitFiberRoot() { if (perf.on) perf.commits += 1; },
      onCommitFiberUnmount() {}, onPostCommitFiberRoot() {}, checkDCE() {},
      on() {}, off() {}, emit() {}, sub() { return () => {}; },
    };
  });
});

const pct = (a, q) => (a.length ? a[Math.min(a.length - 1, Math.floor(a.length * q))] : 0);

/** Run `during` (or just wait) inside a measurement window and summarise it. */
async function measure(page, label, during, ms = WINDOW_MS) {
  // Chromium's own accounting of main-thread work, independent of whether a frame missed vsync: how long the
  // renderer spent running script, laying out and recalculating style inside the window (seconds, cumulative).
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Performance.enable');
  const metrics = async () => Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map((m) => [m.name, m.value]));
  const m0 = await metrics();
  await page.evaluate(() => { const p = window.__perf; p.ts = []; p.long = []; p.commits = 0; p.on = true; });
  const t0 = Date.now();
  if (during) await during(); else await page.waitForTimeout(ms);
  const raw = await page.evaluate(() => { const p = window.__perf; p.on = false; return { ts: p.ts, long: p.long, commits: p.commits }; });
  const m1 = await metrics();
  const busy = (k) => Number((1000 * ((m1[k] || 0) - (m0[k] || 0))).toFixed(0));
  const secs = (Date.now() - t0) / 1000;
  const d = raw.ts.slice(1).map((t, i) => t - raw.ts[i]).sort((a, b) => a - b);
  const r = {
    label,
    secs: Number(secs.toFixed(1)),
    frames: d.length,
    fps: Number((d.length / secs).toFixed(1)),
    p50: Number(pct(d, 0.5).toFixed(1)),
    p95: Number(pct(d, 0.95).toFixed(1)),
    max: Number((d[d.length - 1] || 0).toFixed(1)),
    over20: Number((100 * d.filter((x) => x > 20).length / Math.max(1, d.length)).toFixed(1)),
    longTasks: raw.long.length,
    longMax: Number(Math.max(0, ...raw.long).toFixed(0)),
    commitsPerSec: Number((raw.commits / secs).toFixed(1)),
    // main-thread ms spent in the window (cumulative counters from Performance.getMetrics)
    taskMs: busy('TaskDuration'),
    scriptMs: busy('ScriptDuration'),
    layoutMs: busy('LayoutDuration'),
    styleMs: busy('RecalcStyleDuration'),
  };
  // eslint-disable-next-line no-console
  console.log(`[perf] ${label.padEnd(34)} ${r.fps} fps | frame p50 ${r.p50} p95 ${r.p95} max ${r.max} ms | >20ms ${r.over20}% | `
    + `long tasks ${r.longTasks} (worst ${r.longMax} ms) | React commits ${r.commitsPerSec}/s | `
    + `main thread: task ${r.taskMs} ms, script ${r.scriptMs}, layout ${r.layoutMs}, style ${r.styleMs}`);
  test.info().annotations.push({ type: 'perf', description: JSON.stringify(r) });
  return r;
}

for (const tab of TABS) {
  test(`idle frame budget: #/${tab.id}`, async ({ page }) => {
    await openTab(page, tab.id);
    expectWithin(await measure(page, `idle #/${tab.id}`), BUDGET.idle);
  });
}

test.describe('with a running job (mock telemetry 4 Hz)', () => {
  test.beforeEach(async ({ page, request }) => {
    await request.post('/api/stop').catch(() => {});
    const res = await request.post('/api/swap', { data: {} });
    expect([200, 409]).toContain(res.status());
    await page.waitForTimeout(500);
  });
  test.afterEach(async ({ request }) => { await request.post('/api/stop').catch(() => {}); });

  for (const id of ['faceswap', 'processing', 'batch']) {
    test(`job running: #/${id}`, async ({ page }) => {
      await openTab(page, id);
      expectWithin(await measure(page, `job running #/${id}`), BUDGET.job);
    });
  }
});

const FRAME = { w: 640, h: 360 };

async function openStage(page) {
  const png = makePng(FRAME.w, FRAME.h);
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: png }));
  await openTab(page, 'faceswap');
  const stage = page.locator('.preview-stage').first();
  await expect(stage).toBeVisible({ timeout: 15_000 });
  await stage.hover();
  return stage;
}

async function flood(page, stage, events = 600) {
  const b = await stage.boundingBox();
  const cx = b.x + b.width * 0.5; const cy = b.y + b.height * 0.5;
  const rx = b.width * 0.3; const ry = b.height * 0.25;
  await page.mouse.move(cx + rx, cy);
  for (let i = 0; i < events; i++) {
    const a = (i / 40) * Math.PI;
    await page.mouse.move(cx + rx * Math.cos(a), cy + ry * Math.sin(a * 1.3));
  }
}

test('pointer flood: Mask Brush painting', async ({ page }) => {
  const stage = await openStage(page);
  await stage.getByRole('button', { name: 'Mask Brush' }).click();
  const b = await stage.boundingBox();
  await page.mouse.move(b.x + b.width * 0.5, b.y + b.height * 0.5);
  await page.mouse.down();
  await measure(page, 'brush painting, 600 pointermoves', () => flood(page, stage));
  await page.mouse.up();
});

test('pointer flood: control (no tool active)', async ({ page }) => {
  const stage = await openStage(page);
  await measure(page, 'no tool hover, 600 pointermoves', () => flood(page, stage));
});

test('pointer flood: 3.5x Lens hover', async ({ page }) => {
  const stage = await openStage(page);
  await stage.getByRole('button', { name: /Magnifier|Lens/i }).first().click();
  expectWithin(await measure(page, 'lens hover, 600 pointermoves', () => flood(page, stage)), BUDGET.lens);
});

test('timeline drag', async ({ page }) => {
  await openTab(page, 'faceswap');
  const track = page.locator('div.cursor-ew-resize').first();
  if (!(await track.isVisible().catch(() => false))) {
    test.info().annotations.push({ type: 'perf', description: 'timeline track not mounted in the mock state: skipped' });
    test.skip(true, 'no timeline track in the mock state');
    return;
  }
  // The track is below the fold at the default viewport: mouse events at its page coordinates land nowhere.
  await track.scrollIntoViewIfNeeded();
  const b = await track.boundingBox();
  const frameInput = page.getByLabel('Frame number');
  const before = await frameInput.inputValue();
  await measure(page, 'timeline drag, 240 pointermoves', async () => {
    await page.mouse.move(b.x + b.width * 0.1, b.y + b.height / 2);
    await page.mouse.down();
    for (let i = 0; i <= 240; i++) await page.mouse.move(b.x + b.width * (0.1 + 0.8 * (i / 240)), b.y + b.height / 2);
    await page.mouse.up();
  });
  // A drag that moved nothing would still "measure" fine: prove the playhead went somewhere.
  const after = await frameInput.inputValue();
  test.info().annotations.push({ type: 'perf', description: `playhead frame ${before} -> ${after}` });
  expect(after, 'the drag must have moved the playhead, or the measurement above is of nothing').not.toBe(before);
});

// The same budget with the 3-step Pipeline Pill on screen, measured AGAINST A CONTROL.
//
// The mock's own telemetry is the old shape (`status`, no `processing`, no `started_at`), so the running-job tests above
// never mount the pill and never start the run clocks. A real backend's frame (api._telemetry_snapshot) carries both, and
// with `started_at` set the elapsed/ETA clocks of the run bars tick at 14-27 Hz: ~55-70 React commits/s on #/faceswap
// and #/processing whether or not the pill exists (measured on HEAD without any pill code: 70.7/s). So the absolute
// `BUDGET.job.commitsPerSec` (10) does not describe a real run and is not applied here.
//
// What this asserts is the pill's OWN cost: the same scripted socket twice, once with the stage tables emptied (no pill)
// and once with a Stage 3 percentage that moves on every 4 Hz frame (the pill re-renders on each). The difference must stay
// small, and the frame-time and long-task budgets must hold with it on screen.
const PILL_EXTRA_COMMITS_PER_SEC = 6;

test.describe('with a running job and the pipeline pill (stage tables moving at 4 Hz)', () => {
  test.afterEach(async ({ request }) => { await request.post('/api/stop').catch(() => {}); });

  for (const id of ['faceswap', 'processing']) {
    test(`job running, pill vs control: #/${id}`, async ({ page, request }) => {
      await request.post('/api/stop').catch(() => {});
      const res = await request.post('/api/swap', { data: {} });
      expect([200, 409]).toContain(res.status());
      const st = (state, pct) => ({ state, stage_progress_pct: pct, active_ranges_count: 2, verified_target_frames: 180, skipped_frames_pct: 46.7, eta_seconds: 5, wall_s: 4.2 });
      const startedAt = Date.now() / 1000 - 20;      // fixed for the run, as the backend's is
      const arm = { withStages: false };
      // The mock's own job ends after a few seconds and its /api/progress poll would then say "idle" over the socket's
      // "processing" (the two channels agree on a real backend); keep the poll consistent for the whole test.
      await page.route('**/api/progress', async (route) => {
        const r = await route.fetch();
        let b = {};
        try { b = await r.json(); } catch { b = {}; }
        await route.fulfill({ response: r, json: { ...b, processing: true, paused: false, error: '' } });
      });
      await page.routeWebSocket(/\/ws\/telemetry/, (ws) => {
        let n = 0;
        const send = () => {
          n += 1;
          const pct = (n * 1.7) % 100;
          ws.send(JSON.stringify({
            event: 'progress', processing: true, paused: false, progress: pct / 100, desc: `Processing frame ${Math.round(pct * 3.6)} / 360`,
            error: '', current_frame: Math.round(pct * 3.6), total_frames: 360, fps: 11.8, fps_now: 14 + (n % 5) / 10, frame_ms: 69, eta_s: 7,
            started_at: startedAt, live_seq: 0, engine_state: 'IDLE', engine_label: '', engine_model: '', engine_precision: '', engine_since: null,
            stage: 'STAGE_3_MAIN_SWAP',
            stages: arm.withStages
              ? { STAGE_1_SPARSE_PRESCAN: st('DONE', 100), STAGE_2_ACTIVE_PREPASS: st('DONE', 100), STAGE_3_MAIN_SWAP: st('RUNNING', pct) }
              : {},
            summary: {},
          }));
        };
        send();
        const timer = setInterval(send, 250);
        ws.onClose(() => clearInterval(timer));
      });

      await openTab(page, id);
      await expect(page.getByTestId('pipeline-pill')).toHaveCount(0);
      const control = await measure(page, `job running, no pill   #/${id}`);

      arm.withStages = true;
      // A real reload, not openTab alone: the hash is unchanged, so a bare goto keeps the first arm's app state and the
      // two arms would not be comparable. The new socket carries the stage tables.
      await page.reload();
      await openTab(page, id);
      await expect(page.getByTestId('pipeline-pill'), 'the pill must be on screen for this measurement to mean anything').toBeVisible();
      const withPill = await measure(page, `job running, with pill #/${id}`);

      expectWithin(withPill, { over20: BUDGET.job.over20, longTasks: BUDGET.job.longTasks });
      expect(withPill.commitsPerSec - control.commitsPerSec,
        `the pill costs ${(withPill.commitsPerSec - control.commitsPerSec).toFixed(1)} extra React commits/s (control ${control.commitsPerSec}, pill ${withPill.commitsPerSec})`)
        .toBeLessThanOrEqual(PILL_EXTRA_COMMITS_PER_SEC);
    });
  }
});
