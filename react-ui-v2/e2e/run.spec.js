// Start, Pause, Resume and Stop, and the path that queues a render in pieces.
import { test, expect, readyStudio, stopEverything } from './fixtures.js';

test.afterEach(async ({ page }) => { await stopEverything(page); });

test('Start renders; Pause, Resume and Stop reach the engine from the deck and the top bar', async ({ page, v2 }) => {
  await readyStudio(page, v2);
  const start = page.getByTestId('run-start');
  const readiness = page.getByTestId('readiness');
  const why = (await readiness.count()) ? await readiness.innerText() : '';
  expect(why, 'nothing is missing for a render').toBe('');
  await expect(start).toBeEnabled();
  await start.click();
  await expect.poll(() => v2.posted('/api/swap'), 'a plain render is one /api/swap').toBe(true);

  await expect(page.getByTestId('run-controls')).toHaveCount(1);
  await expect(page.getByTestId('run-controls-compact'), 'pause/stop on every screen').toHaveCount(1);

  await page.getByTestId('run-pause').first().click();
  await expect(page.getByTestId('run-resume').first()).toBeVisible();
  await page.getByTestId('run-resume').first().click();
  await expect(page.getByTestId('run-pause').first()).toBeVisible();
  await page.getByTestId('run-stop').first().click();
  await expect.poll(() => ['/api/pause', '/api/resume', '/api/stop'].every((p) => v2.posted(p))).toBe(true);
  await expect(page.getByTestId('run-start')).toBeVisible({ timeout: 15_000 });
  expect(v2.errors).toEqual([]);
});

test('a range cut by a scene override is queued as pieces and the queue is started', async ({ page, v2 }) => {
  await readyStudio(page, v2);
  const total = await page.evaluate(async () => {
    const st = await (await fetch('/api/state')).json();
    return st.targets[st.selected_target_index]?.frames || 0;
  });
  test.skip(total < 120, `the mock target has ${total} frames`);
  const mark = async (frame, which) => {
    await page.locator('#tl-frame').fill(String(frame));
    await page.locator('#tl-frame').press('Enter');
    await page.getByRole('button', { name: new RegExp(`^${which}$`) }).click();
    await expect.poll(() => v2.bodiesOf('/api/target/set_frame').some((b) => b.which === (which === 'In' ? 'start' : 'end') && b.frame === frame)).toBe(true);
  };

  // An override covers the In–Out range it is added on: 50..80. Then the range goes back to the
  // whole clip, so the render is [1, 49] plain + [50, 80] overridden + [81, end] plain.
  await mark(50, 'In');
  await mark(80, 'Out');
  await page.getByTestId('tool-tab-scenes').click();
  await page.getByTestId('tool-scenes').getByRole('button', { name: /Override In/ }).click();
  await page.getByTestId('tool-scenes').getByRole('button', { name: /Mask feather/ }).first().click();
  await mark(1, 'In');
  await mark(total, 'Out');
  await expect(page.getByTestId('tools')).toContainText('renders in 3 pieces');

  await page.getByTestId('run-start').click();
  await expect.poll(() => v2.posted('/api/queue/add_batch')).toBe(true);
  await expect.poll(() => v2.posted('/api/queue/start')).toBe(true);
  expect(v2.posted('/api/swap'), 'pieces go through the queue, not one swap').toBe(false);
  const jobs = v2.bodiesOf('/api/queue/add_batch')[0]?.jobs || [];
  expect(jobs.map((j) => [j.frame_start, j.frame_end]), 'three pieces that tile the clip, 1-based inclusive').toEqual([[1, 49], [50, 80], [81, total]]);
  expect(new Set(jobs.map((j) => j.segment?.group)).size, 'one segment group').toBe(1);
  expect(v2.errors).toEqual([]);
});
