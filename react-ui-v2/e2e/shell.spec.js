// The shell: boot, every screen, the terminal, widths, and 2.0's independence from 1.0.
import { test, expect, openScreen, SCREENS } from './fixtures.js';

test('boots straight into the studio with one React and no 1.0 markup', async ({ page, v2 }) => {
  await openScreen(page, 'studio');
  await expect(page.getByTestId('studio')).toHaveCount(1);
  // 1.0's rail nav and its glass panels: none of 1.0's visuals ship in 2.0.
  await expect(page.locator('nav[aria-label="Main"] ul, .glass-panel')).toHaveCount(0);
  expect(v2.errors.filter((e) => /Invalid hook call|more than one copy of React/i.test(e)), 'one React (vite dedupe)').toEqual([]);
  expect(v2.errors).toEqual([]);
});

test('every screen opens from the dock, settles with no transform, and reports no error', async ({ page, v2 }) => {
  await openScreen(page, 'studio');
  for (const { route: screen, root } of SCREENS) {
    await page.getByTestId(`dock-${screen}`).click();
    await expect(page.getByTestId(root), `${screen} renders`).toBeVisible({ timeout: 15_000 });
    await expect(page).toHaveURL(new RegExp(`#/${screen}$`));
    await expect(page.getByTestId('screen-error'), `${screen} did not hit its error boundary`).toHaveCount(0);
    // The screen transition must END: a transform left on the screen blurs text and breaks
    // position:fixed children.
    await expect.poll(() => page.evaluate(() => {
      const el = document.querySelector('[data-transition]');
      return el ? getComputedStyle(el).transform : 'missing';
    }), `${screen} ends its transition`).toBe('none');
  }
  expect(v2.errors).toEqual([]);
});

test('every screen also opens by its hash route, and none links back into 1.0', async ({ page, v2 }) => {
  for (const { route: screen } of SCREENS) {
    await openScreen(page, screen);
    await expect(page.getByTestId('screen-error')).toHaveCount(0);
    await expect(page.locator('a[href="/"], a[href^="/#/"]'), `${screen} has no link into 1.0`).toHaveCount(0);
  }
  expect(v2.errors).toEqual([]);
});

test('the terminal opens and closes with Ctrl+`', async ({ page }) => {
  await openScreen(page, 'studio');
  await page.keyboard.press('Control+Backquote');
  await expect(page.locator('.v2-term.is-open')).toHaveCount(1);
  await page.keyboard.press('Control+Backquote');
  await expect(page.locator('.v2-term.is-open')).toHaveCount(0);
});

for (const width of [1280, 420]) {
  test(`no horizontal page scroll at ${width} px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 860 });
    for (const { route: screen } of SCREENS) {
      await openScreen(page, screen);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `${screen} at ${width} px`).toBeLessThanOrEqual(1);
    }
  });
}

// Regression (d888e2d): once a render finishes the engine's /api/progress.output is an OBJECT
// ({path, kind, version, source, url, name, absolute_path}). Rendered raw it threw React error #31
// and blanked the whole app.
test('a finished render whose progress.output is an object is shown by name, not a blank app', async ({ page, v2 }) => {
  await page.route((u) => u.pathname === '/api/progress', async (route) => {
    const res = await route.fetch();
    const body = await res.json();
    await route.fulfill({
      response: res,
      json: {
        ...body,
        processing: false,
        output: { path: 'C:/out/clip_swapped.mp4', kind: 'video', version: 1, source: 'render', url: '/outputs/clip_swapped.mp4', name: 'clip_swapped.mp4', absolute_path: 'C:/out/clip_swapped.mp4' },
      },
    });
  });
  await openScreen(page, 'studio');
  await expect(page.getByTestId('run-deck')).toContainText('last output · clip_swapped.mp4');
  await expect(page.getByTestId('screen-error')).toHaveCount(0);
  expect(v2.errors).toEqual([]);
});
