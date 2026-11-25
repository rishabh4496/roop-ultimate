import { expect, test } from '@playwright/test';
import { openState } from './helpers.js';

test('a refused batch keeps its rows so all targets can be queued on retry', async ({ page, request }) => {
  await request.post('/api/queue/clear', { data: {} });
  await page.route('**/api/state', async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    const targets = state.targets.map((target, index) => ({ ...target, has_target_faces: index === 0 }));
    await route.fulfill({ response, json: { ...state, targets } });
  });
  let enqueues = 0;
  await page.route('**/api/queue/add_batch', async (route) => {
    enqueues += 1;
    if (enqueues === 1) {
      await route.fulfill({ status: 400, json: { message: 'Batch rejected for this test' } });
    } else {
      await route.continue();
    }
  });

  try {
    await openState(page, { tab: 'batch', strategy: 2 });
    await page.getByRole('button', { name: 'Generate Per-File Matrix Batch Jobs' }).click();
    const launch = page.getByRole('button', { name: 'Enqueue & Start Batch' });
    await launch.click();
    await expect(page.getByRole('button', { name: /Dismiss notification: Batch rejected for this test/ })).toBeVisible();
    await expect(launch).toBeVisible();
    expect((await (await request.get('/api/queue')).json()).jobs).toHaveLength(0);

    const start = page.waitForResponse((response) => response.url().endsWith('/api/queue/start'));
    await launch.click();
    const started = await start;
    expect(started.ok()).toBe(true);
    const { jobs } = await started.json();
    expect(jobs.map((job) => job.payload.detection)).toEqual(['Selected face', 'First found']);
    await expect(launch).toBeHidden();
  } finally {
    await request.post('/api/queue/stop', { data: {} });
    await request.post('/api/queue/clear', { data: {} });
  }
});
