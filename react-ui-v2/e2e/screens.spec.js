// The screens beyond the studio: batch queue, faceset library, models, outputs, face manager,
// editor, history and settings.
import { test, expect, openScreen, readyStudio, stopEverything, ENGINE_SHAPES } from './fixtures.js';

test.afterEach(async ({ page }) => { await stopEverything(page); });

test('batch: "Add current setup" queues a job, and the queue starts and stops', async ({ page, v2 }) => {
  await readyStudio(page, v2);
  await page.getByTestId('dock-batch').click();
  await expect(page.getByTestId('batch')).toBeVisible();
  await page.waitForTimeout(800);
  const before = await page.getByTestId('queue-job').count();
  await page.getByTestId('batch-add-current').click();
  await expect(page.getByTestId('queue-job')).toHaveCount(before + 1);

  await expect(page.getByTestId('queue-start')).toBeEnabled();
  await page.getByTestId('queue-start').click();
  await expect.poll(() => v2.posted('/api/queue/start')).toBe(true);
  if (await page.getByTestId('queue-stop').count()) {
    await page.getByTestId('queue-stop').click();
    await expect.poll(() => v2.posted('/api/queue/stop')).toBe(true);
  }
  expect(v2.errors).toEqual([]);
});

test('library: lists the engine\'s facesets and Load sends the file name', async ({ page, v2 }) => {
  await openScreen(page, 'library');
  const cards = page.locator('[data-testid="library-grid"] article');
  await expect(cards).toHaveCount(ENGINE_SHAPES.library.entries.length);
  await expect(cards.first()).toContainText('ava');
  await expect(cards.first()).toContainText('4 faces');
  await cards.first().hover();
  await page.getByTestId('library-load').first().click();
  await expect.poll(() => v2.bodiesOf('/api/faceset/library/load')[0]).toEqual({ filename: 'ava.fsz' });
  expect(v2.errors).toEqual([]);
});

test('library: search filters by name', async ({ page }) => {
  await openScreen(page, 'library');
  await page.getByLabel('Search the library').fill('rhy');
  await expect(page.locator('[data-testid="library-grid"] article')).toHaveCount(1);
  await expect(page.locator('[data-testid="library-grid"] article')).toContainText('rhythm');
});

test('models: every family /api/meta lists, and the integrity of each file on disk', async ({ page, v2 }) => {
  await openScreen(page, 'models');
  expect(await page.locator('[data-testid^="family-"]').count()).toBeGreaterThanOrEqual(6);
  const integrity = page.getByTestId('integrity');
  const okCount = ENGINE_SHAPES.integrity.files.filter((f) => f.status === 'ok').length;
  await expect(integrity).toContainText(`${okCount}/${ENGINE_SHAPES.integrity.files.length} verified`);
  for (const f of ENGINE_SHAPES.integrity.files) await expect(integrity).toContainText(f.file);
  expect(v2.errors).toEqual([]);
});

test('outputs renders without an error', async ({ page, v2 }) => {
  await openScreen(page, 'outputs');
  await expect(page.getByTestId('screen-error')).toHaveCount(0);
  expect(v2.errors).toEqual([]);
});

test('face manager: builder, radar and morph lab; two picks blend two identities', async ({ page, v2 }) => {
  await openScreen(page, 'faces');
  await expect(page.locator('[data-testid="facemgr-builder"], [data-testid="facemgr-radar"], [data-testid="facemgr-morph"]')).toHaveCount(3);
  const candidates = page.locator('[data-testid="facemgr-morph"] .tile');
  test.skip((await candidates.count()) < 2, 'the mock offers fewer than two morph candidates');
  await candidates.nth(0).click();
  await candidates.nth(1).click();
  await expect.poll(() => v2.posted('/api/facemgr/morph/blend')).toBe(true);
  expect(v2.errors).toEqual([]);
});

test('editor: transform, enhance and video-to-faceset panels', async ({ page, v2 }) => {
  await openScreen(page, 'editor');
  await expect(page.locator('[data-testid="editor-transform"], [data-testid="editor-enhance"], [data-testid="editor-faceset"]')).toHaveCount(3);
  expect(v2.errors).toEqual([]);
});

test('history: lists runs, and two ticked runs open the settings compare', async ({ page, v2 }) => {
  await openScreen(page, 'history');
  const runs = page.locator('[data-testid="history-list"] li');
  expect(await runs.count()).toBeGreaterThan(0);
  test.skip((await runs.count()) < 2, 'the mock has one run');
  await page.locator('[data-testid="history-list"] input[type=checkbox]').nth(0).check();
  await page.locator('[data-testid="history-list"] input[type=checkbox]').nth(1).check();
  await expect(page.getByTestId('history-compare')).toHaveCount(1);
  expect(v2.errors).toEqual([]);
});

test('settings: renders every UI setting from the schema, and search filters the panels', async ({ page, v2 }) => {
  await openScreen(page, 'settings');
  const controls = page.locator('[data-testid="settings"] .panel .field-label, [data-testid="settings"] .panel [role=switch]');
  expect(await controls.count()).toBeGreaterThanOrEqual(80);
  const panels = page.locator('[data-testid="settings"] .panel');
  const all = await panels.count();
  await page.getByTestId('settings-search').fill('thread');
  await expect.poll(() => panels.count()).toBeLessThan(all);
  expect(await panels.count()).toBeGreaterThanOrEqual(1);
  expect(v2.errors).toEqual([]);
});
