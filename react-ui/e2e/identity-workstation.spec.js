import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

// Stage 6, in a real Chromium against the production build with the mock backend:
//   #/facemgr   the Identity Workstation - .fsz library pane, face grid (zoom, marquee, Shift/Ctrl, merge / split /
//               rename / hero / exclude, quality filter) and the head-angle coverage radar in the inspector.
//
// WHAT THIS DOES NOT PROVE. The mock stands in for app/api.py: it applies /api/target/group and /name the way the
// real route does, but its pose labels and library entries are fixtures. The backend rules behind a merge or a
// rename are pinned in app/tests; the pure rules here (zones, ranges, marquee maths) in
// .render-check/render-identity-workstation.mjs.

async function seedTargets(request, groups) {
  await request.post('/api/target/clear_faces', { data: {} });
  for (let i = 0; i < groups.length; i++) await request.post('/api/target/use_face', { data: {} });
  await request.post('/api/target/group', { data: { groups } });
}
async function restoreTargets(request) {
  await seedTargets(request, [0]);
  await request.post('/api/target/name', { data: { person: 0, name: 'Actor Lead (Person 1)' } });
}

const tile = (page, i) => page.locator(`[data-testid="ws-grid"] [data-face-index="${i}"]`);
const selected = (page) => page.locator('[data-testid="ws-grid"] [data-face-index][data-selected="true"]');

test.describe('Identity Workstation', () => {
  test.beforeEach(async ({ request }) => { await request.post('/api/facemgr/clear', { data: {} }); });
  test.afterEach(async ({ request }) => {
    await request.post('/api/facemgr/clear', { data: {} });
    await restoreTargets(request);
  });

  test('is the default Face Manager view and fills the viewport in three panes', async ({ page }) => {
    await openTab(page, 'facemgr');
    const ws = page.getByTestId('identity-workstation');
    await expect(ws).toBeVisible();
    await expect(page.getByTestId('ws-library')).toBeVisible();
    await expect(page.getByTestId('ws-stage')).toBeVisible();
    await expect(page.getByTestId('ws-inspector')).toBeVisible();
    const [root, box] = await Promise.all([page.locator('main.studio-main').boundingBox(), ws.boundingBox()]);
    expect(box.height, 'the workstation fills the route').toBeGreaterThan(root.height * 0.8);
    expect(box.width).toBeGreaterThan(root.width * 0.95);
  });

  test('library: search, tag filter, size badges, open loads the archive into the grid', async ({ page }) => {
    await openTab(page, 'facemgr');
    await expect(page.getByTestId('ws-entry')).toHaveCount(3);
    await expect(page.getByTestId('ws-library-total')).toContainText('3 ·');
    await expect(page.getByTestId('ws-size').first()).toHaveText(/\d+(\.\d)? (KB|MB)/);

    await page.getByTestId('ws-search').fill('actors');
    await expect(page.getByTestId('ws-entry')).toHaveCount(1);
    await page.getByTestId('ws-search').fill('nobody-by-this-name');
    await expect(page.getByTestId('ws-library-empty')).toBeVisible();
    await page.getByTestId('ws-search').fill('');

    await page.getByTestId('ws-tag-single').click();
    await expect(page.getByTestId('ws-entry')).toHaveCount(1);
    await expect(page.getByTestId('ws-entry')).toContainText('Lead Portrait');
    await page.getByTestId('ws-tag-single').click();

    await page.getByRole('button', { name: 'Select Studio Cast 2026' }).click();
    await expect(page.getByTestId('ws-export')).toHaveAttribute('download', 'Studio Cast 2026.fsz');
    await page.getByTestId('ws-open').click();
    await expect(page.locator('[data-testid="ws-faces"] [data-face-index]')).toHaveCount(5);
    await expect(page.getByTestId('ws-inspector-title')).toHaveText('Studio Cast 2026');
  });

  test('thumbnail zoom slider resizes the tiles between 64 and 240 px', async ({ page, request }) => {
    for (let i = 0; i < 3; i++) await request.post('/api/facemgr/cut', { data: {} });
    await openTab(page, 'facemgr');
    const slider = page.getByTestId('ws-thumb-slider');
    await expect(slider).toHaveAttribute('min', '64');
    await expect(slider).toHaveAttribute('max', '240');
    await slider.fill('64');
    await expect(page.getByTestId('ws-thumb-value')).toHaveText('64px');
    expect((await tile(page, 0).boundingBox()).width).toBeCloseTo(64, 0);
    await slider.fill('240');
    expect((await tile(page, 0).boundingBox()).width).toBeCloseTo(240, 0);
  });

  test('coverage radar plots the faceset and names the angles it is missing', async ({ page, request }) => {
    for (let i = 0; i < 2; i++) await request.post('/api/facemgr/cut', { data: {} });   // yaw 0 and yaw -58
    await openTab(page, 'facemgr');
    const radar = page.getByTestId('coverage-radar');
    await expect(radar.getByTestId('coverage-dot')).toHaveCount(2);
    const missing = page.getByTestId('coverage-missing');
    await expect(missing).toContainText('Missing Right 90° profile');
    await expect(missing).toContainText('Missing Upward pitch');
    await expect(missing).toContainText('Missing Downward pitch');
    await expect(missing).not.toContainText('Missing Front');
    await expect(missing).not.toContainText('Missing Left 45° profile');
    await expect(radar.locator('[data-zone="F/U"]')).toHaveAttribute('data-missing', 'true');
    await expect(radar.locator('[data-zone="F/M"]')).toHaveAttribute('data-covered', 'true');

    // add the next three seeded angles (right profile 62, high 27, low -26), then reload: the set is re-read from
    // the server, and exactly the angles still not photographed remain
    for (let i = 0; i < 3; i++) await request.post("/api/facemgr/cut", { data: {} });
    await page.reload();
    await expect(radar.getByTestId("coverage-dot")).toHaveCount(5);
    await expect(missing).not.toContainText("Missing Right 90° profile");
    await expect(missing).not.toContainText("Missing Upward pitch");
    await expect(missing).not.toContainText("Missing Downward pitch");
    await expect(missing).toContainText("Missing Left 90° profile");
    await expect(missing).toContainText("Missing Right 45° profile");
  });

  test('Shift selects a range, Ctrl toggles, the radar highlights the selection, Delete excludes it', async ({ page, request }) => {
    for (let i = 0; i < 6; i++) await request.post('/api/facemgr/cut', { data: {} });
    await openTab(page, 'facemgr');
    await tile(page, 1).click();
    await tile(page, 4).click({ modifiers: ['Shift'] });
    await expect(selected(page)).toHaveCount(4);
    await expect(page.getByTestId('ws-selected-count')).toHaveText('4 selected');
    await tile(page, 2).click({ modifiers: ['Control'] });
    await expect(selected(page)).toHaveCount(3);

    await page.getByTestId('ws-grid').click({ position: { x: 4, y: 4 } });
    await expect(selected(page), 'a click on empty grid space deselects').toHaveCount(0);

    await tile(page, 0).click();
    await tile(page, 5).click({ modifiers: ['Control'] });
    await page.keyboard.press('Delete');
    await expect(page.locator('[data-testid="ws-faces"] [data-face-index]')).toHaveCount(4);
  });

  test('marquee drag selects the tiles it crosses', async ({ page, request }) => {
    for (let i = 0; i < 5; i++) await request.post('/api/facemgr/cut', { data: {} });
    await openTab(page, 'facemgr');
    await page.getByTestId('ws-thumb-slider').fill('96');
    const a = await tile(page, 1).boundingBox();
    const b = await tile(page, 3).boundingBox();
    const grid = await page.getByTestId('ws-grid').boundingBox();
    // start in the padding above-left of tile 1 (empty grid background), drag to the far corner of tile 3
    await page.mouse.move(Math.max(grid.x + 2, a.x - 4), Math.max(grid.y + 2, a.y - 4));
    await page.mouse.down();
    await page.mouse.move(b.x + b.width - 6, b.y + b.height - 6, { steps: 8 });
    await expect(page.getByTestId('ws-marquee')).toBeVisible();
    await page.mouse.up();
    const ids = await selected(page).evaluateAll((els) => els.map((e) => Number(e.getAttribute('data-face-index'))));
    expect(ids).toEqual(expect.arrayContaining([1, 2, 3]));
    expect(ids).not.toContain(0);
    expect(ids).not.toContain(4);
    await expect(page.getByTestId('ws-marquee')).toHaveCount(0);
  });

  test('quality filter: threshold dims outliers, selects them, and excludes them', async ({ page, request }) => {
    for (let i = 0; i < 7; i++) await request.post('/api/facemgr/cut', { data: {} });   // scores .82 .7 .3 .88 .66 .2 .8
    await openTab(page, 'facemgr');
    await page.getByTestId('ws-threshold').fill('0.5');
    await expect(page.getByTestId('ws-outlier-count')).toHaveText('2 below');
    await expect(page.locator('[data-face-index][data-outlier="true"]')).toHaveCount(2);
    await page.getByTestId('ws-select-outliers').click();
    await expect(selected(page)).toHaveCount(2);
    await page.getByTestId('ws-exclude-outliers').click();
    await expect(page.locator('[data-testid="ws-faces"] [data-face-index]')).toHaveCount(5);
    await expect(page.getByTestId('ws-outlier-count')).toHaveText('0 below');
  });

  test('target people: rename inline, split a false positive, merge two identities', async ({ page, request }) => {
    await seedTargets(request, [0, 0, 1, 1]);
    await openTab(page, 'facemgr');
    await page.getByTestId('ws-mode-people').click();
    await expect(page.getByTestId('ws-person')).toHaveCount(2);

    // rename: the server's reply is what the field shows afterwards
    const name = page.getByTestId('ws-person-name').first();
    await name.fill('Marta');
    await name.press('Enter');
    await expect(page.getByTestId('ws-person-name').first()).toHaveValue('Marta');
    await page.reload();
    await page.getByTestId('ws-mode-people').click();
    await expect(page.getByTestId('ws-person-name').first()).toHaveValue('Marta');

    // split face 1 (person 1's second face) into its own identity
    await tile(page, 1).click();
    await expect(page.getByTestId('ws-merge'), 'one identity touched: nothing to merge').toBeDisabled();
    await page.getByTestId('ws-split').click();
    await expect(page.getByTestId('ws-person')).toHaveCount(3);

    // merge a face from two different identities: they collapse into one
    await tile(page, 0).click();
    await tile(page, 2).click({ modifiers: ['Control'] });
    await expect(page.getByTestId('ws-merge')).toBeEnabled();
    await page.getByTestId('ws-merge').click();
    await expect(page.getByTestId('ws-person')).toHaveCount(2);
  });

  test('hero thumbnail is per person, marked, and survives a reload', async ({ page, request }) => {
    await seedTargets(request, [0, 0, 0]);
    await openTab(page, 'facemgr');
    await page.getByTestId('ws-mode-people').click();
    await expect(page.getByTitle('Hero thumbnail')).toHaveCount(1);
    await expect(tile(page, 0).getByTitle('Hero thumbnail')).toHaveCount(1);
    await tile(page, 2).click();
    await page.getByTestId('ws-hero').click();
    await expect(tile(page, 2).getByTitle('Hero thumbnail')).toHaveCount(1);
    await expect(tile(page, 0).getByTitle('Hero thumbnail')).toHaveCount(0);
    await page.reload();
    await page.getByTestId('ws-mode-people').click();
    await expect(tile(page, 2).getByTitle('Hero thumbnail')).toHaveCount(1);
  });

  test('people mode with nothing captured says so', async ({ page, request }) => {
    await request.post('/api/target/clear_faces', { data: {} });
    await openTab(page, 'facemgr');
    await page.getByTestId('ws-mode-people').click();
    await expect(page.getByTestId('ws-empty-people')).toBeVisible();
    await expect(page.getByTestId('coverage-empty')).toBeVisible();
  });
});
