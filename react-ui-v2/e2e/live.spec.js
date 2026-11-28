// React UI 2.0's actions against the REAL engine. Run with playwright.live.config.js only; read
// that file for what this needs and what it leaves behind.
//
// The mock cannot show these. Each one found a defect the mock suite passed:
// - a real swap preview (a "N swapped" frame, not just any status text);
// - In/Out: the frame on screen is the frame rendered. In 300 / Out 329 must render 30 frames that
//   start on source frame 300 (2617e3c; it used to render 29 from 301);
// - a finished render's /api/progress.output is an object, which blanked the app (d888e2d);
// - Pause, Resume and Stop as the engine reports them, and a stopped run kept out of History.
import { test as base, expect } from '@playwright/test';

const test = base.extend({
  api: async ({ page }, use) => {
    await use(async (path, body) => page.evaluate(async ([p, b]) => {
      const r = await fetch(p, b === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) });
      return r.json();
    }, [path, body]));
  },
});

test('Studio → preview, In/Out, Start/Pause/Resume/Stop, a full render, and the batch queue on the real engine', async ({ page, api }) => {
  const posts = [];
  const errors = [];
  page.on('request', (r) => { if (r.method() === 'POST') posts.push(new URL(r.url()).pathname); });
  page.on('pageerror', (e) => errors.push(e.message));
  await page.addInitScript(() => {
    try {
      sessionStorage.setItem('roop_v2_booted', '1');
      localStorage.setItem('roop_v2_fx', JSON.stringify({ ambient: false }));
    } catch { /* */ }
  });
  await page.goto('#/studio', { waitUntil: 'domcontentloaded' });
  await page.getByTestId('studio').waitFor({ timeout: 30_000 });
  await page.waitForTimeout(3000);

  const pre = await api('/api/progress');
  test.skip(!!pre.processing, 'the engine is busy: nothing was changed');
  const st = await api('/api/state');
  const t = st.targets?.[st.selected_target_index];
  test.skip(!t, 'no target is loaded in the engine');
  const readiness = page.getByTestId('readiness');
  const why = (await readiness.count()) ? await readiness.innerText() : '';
  test.skip(!!why, `the studio is not ready to render: ${why}`);

  const mediaId = t.media_id || t.id;
  const orig = { start: t.start_frame, end: t.end_frame };
  const histBefore = (await api('/api/history')).entries.length;

  try {
    let faceAt = 0;
    await test.step('the preview swaps a real face', async () => {
      let status = '';
      for (const f of [1, 300, 1200, 2500, 5000, 9000, 15000, 24000, 36000, 50000, 65000]) {
        if (f > t.frames) break;
        await page.locator('#tl-frame').fill(String(f));
        await page.locator('#tl-frame').press('Enter');
        for (let i = 0; i < 40; i++) {
          await page.waitForTimeout(500);
          status = await page.getByTestId('viewport-status').innerText().catch(() => '');
          if (/swapped|no face|refused|error/i.test(status)) break;
        }
        if (/\d+ swapped/.test(status)) { faceAt = f; break; }
      }
      expect(faceAt, `no frame swapped a face; last status: ${status}`).toBeGreaterThan(0);
    });

    const inAt = Math.min(Math.max(1, faceAt), Math.max(1, t.frames - 40));
    await test.step('In/Out through the timeline buttons: the frames on screen, 30 inclusive', async () => {
      await page.locator('#tl-frame').fill(String(inAt));
      await page.waitForTimeout(300);
      await page.getByRole('button', { name: /^In$/ }).click();
      await page.locator('#tl-frame').fill(String(inAt + 29));
      await page.waitForTimeout(300);
      await page.getByRole('button', { name: /^Out$/ }).click();
      await page.waitForTimeout(1500);
      const t2 = (await api('/api/state')).targets[st.selected_target_index];
      expect([t2.start_frame, t2.end_frame], 'the target list reports In/Out as set').toEqual([inAt, inAt + 29]);
      expect([t2.startframe, t2.endframe], 'the engine seeks to 0-based In-1 and ends after Out').toEqual([inAt - 1, inAt + 29]);
      await expect(page.locator('text=/in \\d+ · out \\d+/').first()).toContainText('30 f');
    });

    await test.step('Start → Pause → Resume → Stop, as the engine reports them', async () => {
      await page.getByTestId('run-start').click();
      await page.getByTestId('run-pause').first().waitFor({ timeout: 120_000 });
      expect(posts).toContain('/api/swap');
      await page.waitForTimeout(6000);
      await page.getByTestId('run-pause').first().click();
      await page.getByTestId('run-resume').first().waitFor({ timeout: 60_000 });
      const paused = await api('/api/progress');
      expect(paused.pause_requested || paused.paused, 'the engine reports the pause').toBeTruthy();
      await page.getByTestId('run-resume').first().click();
      await page.getByTestId('run-pause').first().waitFor({ timeout: 60_000 });
      expect((await api('/api/progress')).paused).toBeFalsy();
      await page.waitForTimeout(3000);
      await page.getByTestId('run-stop').first().click();
      await expect.poll(async () => (await api('/api/progress')).processing, { timeout: 120_000, intervals: [1000] }).toBeFalsy();
      expect(['/api/pause', '/api/resume', '/api/stop'].every((p) => posts.includes(p))).toBe(true);
      await page.waitForTimeout(3000);
      expect((await api('/api/history')).entries.length, 'a stopped run is not recorded').toBe(histBefore);
    });

    await test.step('a full In..Out render completes, writes 30 frames and lands in History', async () => {
      const topBefore = (await api('/api/history')).entries[0]?.id;
      await page.getByTestId('run-start').click();
      await page.getByTestId('run-pause').first().waitFor({ timeout: 120_000 });
      await expect.poll(async () => (await api('/api/progress')).processing, { timeout: 15 * 60_000, intervals: [2000] }).toBeFalsy();
      const done = await api('/api/progress');
      expect(done.error || '', `render ended: ${done.desc}`).toBe('');
      expect(done.desc).toBe('Done');
      await page.waitForTimeout(3000);
      const newest = (await api('/api/history')).entries[0];
      expect(newest?.id).not.toBe(topBefore);
      expect((newest.outputs || []).some((o) => /\.mp4$/i.test(o)), (newest.outputs || []).join(', ')).toBe(true);
      expect(newest.frames, 'In..Out inclusive: 30 frames, not 29').toBe(30);
      // progress.output is now an object: the studio must still be there, naming it.
      await expect(page.getByTestId('run-deck')).toContainText('last output ·');
      await expect(page.getByTestId('screen-error')).toHaveCount(0);
    });

    await test.step('Batch → "Add current setup" queues a real job, and Remove takes it out', async () => {
      await page.evaluate(() => { window.location.hash = '#/batch'; });
      await page.getByTestId('batch').waitFor();
      await page.waitForTimeout(1500);
      const before = (await api('/api/queue')).jobs.length;
      await page.getByTestId('batch-add-current').click();
      await expect.poll(async () => (await api('/api/queue')).jobs.length).toBe(before + 1);
      const q = await api('/api/queue');
      expect(q.jobs.at(-1)?.payload?.processing_selection?.target_media_id).toBe(mediaId);
      await page.getByTestId('queue-job').last().getByRole('button', { name: 'Remove' }).click();
      await expect.poll(async () => (await api('/api/queue')).jobs.length).toBe(before);
    });
  } finally {
    // The target's In/Out exactly as found (both 1-based).
    await api('/api/target/set_frame', { which: 'start', frame: orig.start, target_media_id: mediaId });
    await api('/api/target/set_frame', { which: 'end', frame: orig.end, target_media_id: mediaId });
  }
  const t3 = (await api('/api/state')).targets[st.selected_target_index];
  expect([t3.start_frame, t3.end_frame], 'In/Out restored').toEqual([orig.start, orig.end]);
  expect(errors).toEqual([]);
});
