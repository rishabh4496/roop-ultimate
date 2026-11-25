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

test('removing the first target keeps the settings of the next Matrix row', async ({ page }) => {
  let removed = false;
  await page.route('**/api/state', async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    await route.fulfill({ response, json: { ...state, targets: removed ? state.targets.slice(1) : state.targets } });
  });
  await page.route('**/api/target/remove', async (route) => {
    removed = true;
    await route.fulfill({ json: { success: true } });
  });
  await openState(page, { tab: 'batch', strategy: 2 });
  const modes = page.getByRole('combobox', { name: /: swap mode$/ });
  await expect(modes).toHaveCount(2);
  await modes.nth(0).selectOption('All faces');
  await modes.nth(1).selectOption('First found');
  await page.getByRole('button', { name: 'Remove interview_scene_1080p.mp4' }).click();
  await expect(modes).toHaveCount(1);
  await expect(modes.first()).toHaveValue('First found');
});

test('removing a faceset keeps surviving assignments and exposes missing ones', async ({ page }) => {
  let removed = false;
  await page.route('**/api/state', async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    if (removed) {
      state.source_faces_info = state.source_faces_info.slice(1);
      state.source_faces = state.source_faces.slice(1);
    }
    await route.fulfill({ response, json: state });
  });
  await page.route('**/api/source/remove', async (route) => {
    removed = true;
    await route.fulfill({ json: { success: true } });
  });
  await openState(page, { tab: 'batch', strategy: 2 });
  const assignments = page.getByRole('combobox', { name: /assigned faceset/ });
  await expect(assignments).toHaveCount(2);
  await assignments.nth(1).selectOption({ label: 'source_beta.png' });
  await page.getByRole('button', { name: 'Remove Faceset #1' }).click();
  await expect(assignments.nth(0)).toHaveValue('-1');
  await expect(assignments.nth(1).locator('option:checked')).toHaveText('source_beta.png');
  await page.getByRole('button', { name: 'Generate Per-File Matrix Batch Jobs' }).click();
  await expect(page.getByRole('button', { name: /Dismiss notification: Assign a loaded faceset/ })).toBeVisible();
});
