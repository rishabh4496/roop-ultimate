import { expect, test } from '@playwright/test';
import { openState, openTab, PNG_1X1 } from './helpers.js';

test('Face Swap starts queued jobs from its main run button', async ({ page, request }) => {
  await request.post('/api/queue/clear', { data: {} });
  const state = await (await request.get('/api/state')).json();
  const source = state.source_faces_info[0];
  const jobs = state.targets.slice(0, 2).map((target) => ({
    target_name: target.name, target_media_id: target.media_id || target.id,
    source_index: 0, source_id: source.id, payload: { detection: 'First found' },
  }));
  expect((await request.post('/api/queue/add_batch', { data: { jobs } })).ok()).toBe(true);
  const calls = [];
  for (const path of ['/api/swap', '/api/queue/start']) {
    await page.route(`**${path}`, async (route) => {
      calls.push(path);
      if (path === '/api/swap') await route.fulfill({ json: { success: true } });
      else await route.continue();
    });
  }
  try {
    await openTab(page, 'faceswap');
    await page.getByRole('button', { name: /Start (Swapping|Batch)/ }).click();
    await expect.poll(() => calls.length).toBe(1);
    expect(calls).toEqual(['/api/queue/start']);
  } finally {
    await request.post('/api/queue/stop', { data: {} });
    await request.post('/api/queue/clear', { data: {} });
  }
});

test('Face Swap queues fresh videos with the shared faceset and no previous clip selection', async ({ page }) => {
  let original;
  await page.route('**/api/state', async (route) => {
    const response = await route.fetch();
    const state = await response.json();
    original = {
      ...state, targets: state.targets.slice(0, 1), selected_target_index: 0,
      source_faces: state.source_faces.slice(0, 1), source_faces_info: state.source_faces_info.slice(0, 1),
      selected_source_index: 0,
      target_person_ids: ['previous-person'], target_reference_face_ids: ['previous-reference'],
      selected_target_person_id: 'previous-person', selected_reference_face_id: 'previous-reference',
    };
    await route.fulfill({ response, json: original });
  });
  await page.route('**/api/settings', async (route) => {
    const response = await route.fetch();
    await route.fulfill({ response, json: { ...await response.json(), face_detection_mode: 'Selected face' } });
  });
  const fresh = ['one', 'two'].map((id) => ({ id, media_id: id, name: `${id}.mp4`, frames: 30, has_target_faces: false }));
  await page.route('**/api/target/add', (route) => route.fulfill({ json: {
    targets: [...original.targets, ...fresh], selected_target_index: 1, target_media_id: fresh[0].id,
    target_faces: [], target_groups: [],
  } }));
  await page.route('**/api/queue/add_batch', (route) => route.fulfill({ json: { jobs: [] } }));

  await openTab(page, 'faceswap');
  const enqueue = page.waitForRequest('**/api/queue/add_batch');
  await page.locator('input[type=file][accept*="video/"]').first().setInputFiles(fresh.map((target) => ({
    name: target.name, mimeType: 'video/mp4', buffer: Buffer.from('mock video'),
  })));
  const { jobs } = (await enqueue).postDataJSON();
  expect(jobs.map((job) => job.target_media_id)).toEqual(['one', 'two']);
  for (const job of jobs) {
    expect(job.source_id).toBe(original.source_faces_info[0].id);
    expect(job.payload).toMatchObject({ detection: 'First found', face_mapping: [0] });
    expect(job.payload.processing_selection).toMatchObject({
      target_media_id: job.target_media_id, target_person_id: null, target_person_ids: [],
      target_reference_face_id: null, target_person_source_mapping: {},
    });
  }
});

test('Face Swap can queue its loaded videos after a faceset is added', async ({ page, request }) => {
  await request.post('/api/queue/clear', { data: {} });
  const state = await (await request.get('/api/state')).json();
  const targets = state.targets.map((target) => ({ ...target, name: target.name.replace(/\.png$/, '.mp4'), frames: 30, fps: 30 }));
  await page.route('**/api/state', (route) => route.fulfill({ json: {
    ...state, targets, source_faces: [], source_faces_info: [], faceset_count: 0,
  } }));
  await page.route('**/api/source/add', (route) => route.fulfill({ json: {
    source_faces: state.source_faces.slice(0, 1), source_faces_info: state.source_faces_info.slice(0, 1), faceset_count: 1,
  } }));
  try {
    await openTab(page, 'faceswap');
    const addVideos = page.getByRole('button', { name: 'Queue 2 videos' });
    await expect(addVideos).toHaveCount(0);
    await page.locator('input[type=file][accept*=".fsz"]').first().setInputFiles({
      name: 'source.png', mimeType: 'image/png', buffer: PNG_1X1,
    });
    const [response] = await Promise.all([page.waitForResponse('**/api/queue/add_batch', { timeout: 10000 }), addVideos.click()]);
    expect(response.ok()).toBe(true);
    const { jobs } = await response.json();
    expect(jobs.map((job) => job.target_name)).toEqual(targets.map((target) => target.name));
    expect(jobs.map((job) => job.source_id)).toEqual([state.source_faces_info[0].id, state.source_faces_info[0].id]);
    await expect(addVideos).toHaveCount(0);
    await expect(page.getByRole('button', { name: /Start Batch \(2\)/ })).toBeVisible();
  } finally {
    await request.post('/api/queue/clear', { data: {} });
  }
});

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
