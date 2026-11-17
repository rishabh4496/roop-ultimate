import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

test('Studio keys edit, undo, redo and recover after reload without remote requests', async ({ page }) => {
  const errors = [], remote = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => {
    const url = new URL(request.url());
    if (['http:', 'https:', 'ws:', 'wss:'].includes(url.protocol) && !['localhost', '127.0.0.1', '::1'].includes(url.hostname)) remote.push(request.url());
  });
  await openTab(page, 'faceswap');
  await page.getByTitle('Toggle Pro Studio NLE View & Viewport').click();
  const parameter = page.getByTitle('Active Keyframe Parameter');
  await expect(parameter).toBeVisible();
  await parameter.selectOption('grain_match_strength');
  await page.getByTitle(/Add grain_match_strength keyframe/).click();
  const value = page.getByRole('spinbutton', { name: 'Keyframe value' });
  await expect(value).toBeVisible();
  await page.keyboard.press('Control+z');
  await expect(value).toBeHidden();
  await page.keyboard.press('Control+y');
  await expect(value).toBeVisible();
  await value.fill('0.6');
  await value.evaluate((element) => element.blur());
  await expect(value).toHaveValue('0.6');
  await page.waitForTimeout(400);
  await page.keyboard.press('Control+z');
  await expect(value).not.toHaveValue('0.6');
  await page.keyboard.press('Control+y');
  await expect(value).toHaveValue('0.6');
  await page.waitForTimeout(400);
  await page.reload();
  await expect(page.getByTitle('Active Keyframe Parameter')).toBeVisible();
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('roop_studio_projects_v1')));
  expect(Object.values(saved.projects).some((project) => project.keyframes?.some((key) => key.param === 'grain_match_strength' && key.value === .6))).toBeTruthy();
  expect(errors).toEqual([]);
  expect(remote).toEqual([]);
});

test('Pre-Scan requests the backend and its faceset bindings persist in the project', async ({ page }) => {
  const scans = [];
  page.on('request', (request) => {
    if (request.url().endsWith('/api/target/prescan')) scans.push(request.postDataJSON());
  });
  await openTab(page, 'faceswap');
  await page.getByTitle('Toggle Pro Studio NLE View & Viewport').click();
  await page.getByRole('button', { name: /Pre-Scan/ }).last().click();
  const binding = page.getByRole('combobox', { name: 'Faceset for Track #1' });
  await expect(binding).toBeVisible();
  const previousBinding = await binding.inputValue();
  await binding.selectOption('/mock/facesets/source_beta.png');
  await expect(binding).toHaveValue('/mock/facesets/source_beta.png');
  await page.keyboard.press('Control+z');
  await expect(binding).toHaveValue(previousBinding);
  await page.keyboard.press('Control+y');
  await expect(binding).toHaveValue('/mock/facesets/source_beta.png');
  await page.waitForTimeout(400);
  expect(scans).toHaveLength(1);
  expect(scans[0]).toHaveProperty('detector_model');
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'Faceset for Track #1' })).toHaveValue('/mock/facesets/source_beta.png');
});
