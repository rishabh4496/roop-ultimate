import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// The terminal dashboard on #/processing in a real browser, fed a REAL snapshot: `console-mirror.fixture.json` is
// produced by roop/terminal_dashboard.py (see .render-check/render-console-mirror.mjs), and the backend paints the
// console from the very object it puts on /ws/telemetry as `dashboard`. So what is under test is what the shell does
// with that payload: the four stages, the latency split with the bottleneck flagged, the hit rates, nothing at idle,
// accessibility, and no horizontal overflow at desktop and phone width.

const here = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(join(here, '..', '.render-check', 'console-mirror.fixture.json'), 'utf8'));
const AXE = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];

const frame = (extra) => ({
  event: 'progress', processing: true, paused: false, progress: 0.17, desc: 'Processing frame 300 / 1800',
  error: '', current_frame: 300, total_frames: 1800, fps: 50, fps_now: 50, frame_ms: 20, eta_s: 30,
  started_at: Date.now() / 1000 - 20, live_seq: 0, engine_state: 'IDLE', engine_label: '', engine_model: '',
  engine_precision: '', engine_since: null, ...extra,
});

async function scriptTelemetry(page, frames, gap = 250) {
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => {
    let i = 0;
    const send = () => { ws.send(JSON.stringify(frames[Math.min(i, frames.length - 1)])); i += 1; };
    send();
    const timer = setInterval(send, gap);
    ws.onClose(() => clearInterval(timer));
  });
}

test('processing: the console mirror shows the terminal\'s stages, split and hit rates', async ({ page }) => {
  await scriptTelemetry(page, [frame({ event: 'hello', dashboard: fx.running }), frame({ dashboard: fx.running })]);
  await page.goto('/#/processing');

  const mirror = page.getByTestId('console-mirror');
  await expect(mirror).toBeVisible();
  await expect(mirror).toHaveAttribute('data-active', 'true');
  await expect(mirror.locator('[data-stage]')).toHaveCount(4);
  const s3 = mirror.locator('[data-stage="S3"]');
  await expect(s3).toHaveAttribute('data-state', 'RUNNING');
  await expect(s3).toContainText('16.7%');
  await expect(s3).toContainText('300/1,800 frames');
  await expect(s3).toContainText('ETA: 00m 30s');
  await expect(mirror.locator('[data-stage="S4"]')).toContainText('pending');

  // the seven latency cells, the bottleneck column flagged (and only it)
  await expect(mirror.locator('[data-col]')).toHaveCount(7);
  await expect(mirror.locator('[data-bottleneck="true"]')).toHaveCount(1);
  await expect(mirror.locator('[data-col="restore"]')).toHaveAttribute('data-bottleneck', 'true');
  await expect(mirror.locator('[data-col="restore"]')).toContainText('Restore 32.0');
  await expect(mirror).toContainText('[BOTTLENECK: Restore 54%]');
  await expect(mirror.locator('[data-hit="roi"]')).toContainText('10.0%');
  await expect(mirror.locator('[data-hit="vram"]')).toContainText('7.25 / 8.50 GB');

  // every percentage the bars carry is exposed to assistive tech
  const bars = mirror.getByRole('progressbar');
  await expect(bars).toHaveCount(4);
  await expect(bars.nth(2)).toHaveAttribute('aria-valuenow', '17');
});

test('processing: nothing is shown before a job, and the last job stays readable after it', async ({ page }) => {
  await scriptTelemetry(page, [frame({ event: 'hello', processing: false, progress: 0, desc: '', dashboard: fx.idle })]);
  await page.goto('/#/processing');
  await page.waitForTimeout(600);
  await expect(page.getByTestId('console-mirror')).toHaveCount(0);

  await page.unroute(/\/ws\/telemetry/).catch(() => {});
  const p2 = await page.context().newPage();
  await scriptTelemetry(p2, [frame({ event: 'hello', processing: false, progress: 1, desc: 'Done', dashboard: fx.final })]);
  await p2.goto('/#/processing');
  const mirror = p2.getByTestId('console-mirror');
  await expect(mirror).toBeVisible();
  await expect(mirror).toHaveAttribute('data-active', 'false');
  await expect(mirror).toContainText('last run');
  await expect(mirror.locator('[data-stage="S4"]')).toContainText('output verification');
});

test('axe and layout: no violations, no horizontal scroll at 1280 and at phone width', async ({ page }) => {
  await scriptTelemetry(page, [frame({ event: 'hello', dashboard: fx.running }), frame({ dashboard: fx.running })]);
  for (const [w, h] of [[1280, 800], [390, 844]]) {
    await page.setViewportSize({ width: w, height: h });
    await page.goto('/#/processing');
    const mirror = page.getByTestId('console-mirror');
    await expect(mirror).toBeVisible();
    const { violations } = await new AxeBuilder({ page }).include('[data-testid="console-mirror"]').withTags(AXE).analyze();
    expect(violations.map((v) => ({ rule: v.id, targets: v.nodes.slice(0, 3).map((n) => n.target.join(' ')) })), `axe at ${w}`).toEqual([]);
    const overflow = await page.evaluate(() => {
      const el = document.querySelector('[data-testid="console-mirror"]');
      const r = el.getBoundingClientRect();
      return { doc: document.documentElement.scrollWidth - document.documentElement.clientWidth, mirror: el.scrollWidth - el.clientWidth, right: r.right - window.innerWidth };
    });
    expect(overflow.doc, `page overflows at ${w}`).toBeLessThanOrEqual(0);
    expect(overflow.mirror, `mirror overflows itself at ${w}`).toBeLessThanOrEqual(0);
    expect(overflow.right, `mirror leaves the viewport at ${w}`).toBeLessThanOrEqual(1);
  }
});
