import { expect, test } from '@playwright/test';

const CLIP = 'interview_scene_1080p.mp4';
const ALPHA = 'source_alpha.png';
const BETA = 'source_beta.png';
const matrix = (page) => page.getByRole('region', { name: 'Batch matrix', exact: true });
const cell = (page, name) => matrix(page).getByRole('button', { name });

test('source cells, row and column toggles control the staged split payload', async ({ page }) => {
  let submitted;
  let queueJobs = [];
  await page.route('**/api/queue', (route) => route.fulfill({ json: { jobs: queueJobs, running: false } }));
  await page.route('**/api/queue/add_batch', async (route) => {
    submitted = route.request().postDataJSON().jobs;
    queueJobs = submitted.map((job, i) => ({ ...job, id: String(i), state: i ? 'QUEUED' : 'COMPLETED' }));
    await route.fulfill({ json: { jobs: queueJobs, running: false } });
  });
  await page.goto('/#/batch');
  await expect(cell(page, new RegExp(ALPHA + ' × ' + CLIP + ': Planned'))).toBeVisible();
  await cell(page, 'Toggle row ' + ALPHA).click();
  await expect(cell(page, new RegExp(ALPHA + ' × ' + CLIP + ': Skipped'))).toBeVisible();
  await cell(page, 'Toggle column ' + CLIP).click();
  await expect(cell(page, new RegExp(BETA + ' × ' + CLIP + ': Planned'))).toBeVisible();
  await cell(page, 'Toggle column ' + CLIP).click();
  await expect(cell(page, new RegExp(BETA + ' × ' + CLIP + ': Skipped'))).toBeVisible();
  await cell(page, new RegExp(ALPHA + ' × ' + CLIP + ': Skipped')).click();

  await page.getByRole('checkbox', { name: 'Automatically split video' }).check();
  await page.getByRole('spinbutton', { name: 'Target duration' }).fill('3');
  const timeline = page.getByTestId('batch-split-timeline');
  await expect(timeline).toContainText('1–90 · 91–180');
  await page.getByRole('button', { name: 'Align to GOP boundaries' }).click();
  await expect(timeline).toContainText('GOP aligned');
  await expect(timeline).toContainText('1–120 · 121–180');
  await cell(page, new RegExp(BETA + ' × ' + CLIP + ' chunk 2: Skipped')).click();
  await page.getByRole('button', { name: 'Stage selected jobs' }).click();
  await expect(cell(page, new RegExp(BETA + ' × ' + CLIP + ' chunk 2: Staged'))).toBeVisible();
  await page.getByRole('button', { name: 'Enqueue to Queue', exact: true }).click();
  await expect(cell(page, new RegExp(BETA + ' × ' + CLIP + ' chunk 2: Queued'))).toBeVisible();
  const jobs = submitted.filter((job) => job.target_name === CLIP);
  expect(jobs.map(({ source_name, frame_start, frame_end }) => [source_name, frame_start, frame_end])).toEqual([
    [ALPHA, 1, 120], [ALPHA, 121, 180], [BETA, 121, 180],
  ]);
  expect(jobs[0].target_media_id).toBe('mock-media-0001');
  expect(jobs[0].segment.group).toBe(jobs[1].segment.group);
  expect(jobs[2].segment).toMatchObject({ index: 2, count: 2 });
  expect(jobs[2].segment.group).not.toBe(jobs[0].segment.group);
  await expect(cell(page, new RegExp(ALPHA + ' × ' + CLIP + ' chunk 1: Rendered'))).toBeVisible();
  await page.setViewportSize({ width: 420, height: 900 });
  await expect.poll(() => matrix(page).evaluate((grid) => ({
    scrolls: grid.scrollWidth > grid.clientWidth,
    fits: grid.getBoundingClientRect().right <= window.innerWidth,
  }))).toEqual({ scrolls: true, fits: true });
});

test('grouped and per-file previews reflect assignments and recipe chips show only changes', async ({ page }) => {
  await page.goto('/#/batch');
  await page.getByRole('button', { name: /Grouped Multi-Batch/ }).click();
  await page.getByRole('button', { name: '+ ' + CLIP, exact: true }).click();
  await expect(cell(page, 'Toggle column Batch Group A · ' + CLIP)).toBeVisible();
  await page.getByRole('button', { name: /Per-File Matrix Full matrix/ }).click();
  await expect(cell(page, new RegExp(ALPHA + ' × ' + CLIP + ': Planned'))).toBeVisible();

  await page.getByRole('button', { name: /Recipe Matrix Compare/ }).click();
  await page.getByRole('button', { name: 'Add recipe' }).click();
  const differences = page.locator('[aria-label="Recipe 1 differences"]');
  await expect(differences).toHaveText('Matches baseline');
  await page.getByRole('combobox', { name: 'Recipe 1 Mask' }).selectOption('Clip2Seg');
  await expect(differences).toContainText('→ Clip2Seg');
  await expect(differences).not.toContainText(/Enhancer|Swap model/);
  await expect(cell(page, ALPHA + ' × Recipe 1: Planned')).toBeVisible();

  const request = page.waitForRequest((req) => req.url().endsWith('/api/queue/add_batch'));
  await page.route('**/api/queue/add_batch', (route) => route.fulfill({
    status: 400, json: { message: 'Fixture rejection' },
  }));
  await page.getByRole('button', { name: 'Stage selected jobs' }).click();
  await page.getByRole('button', { name: 'Enqueue to Queue', exact: true }).click();
  const jobs = (await request).postDataJSON().jobs;
  expect(jobs).toHaveLength(6);
  expect(jobs.every((job) => job.target_name === CLIP)).toBe(true);
  expect(jobs.filter((job) => job.label.startsWith('Recipe 1')).map((job) => job.payload.mask_engine))
    .toEqual(['Clip2Seg', 'Clip2Seg', 'Clip2Seg']);
  await expect(page.getByRole('status').filter({ hasText: 'Fixture rejection' })).toBeVisible();
  await expect(cell(page, ALPHA + ' × Recipe 1: Staged')).toBeVisible();
});
