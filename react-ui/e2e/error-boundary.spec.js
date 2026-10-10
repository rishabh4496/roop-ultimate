import { test, expect } from '@playwright/test';

// The live path of the one ErrorBoundary (the markup and state machine are covered
// in .render-check/render-error-boundary.mjs): a tab's lazy chunk fails to
// download -- the server restarted, or the connection dropped mid-navigation.
//
// Without a boundary React 19 unmounts the whole tree on the rejected import() and
// the app goes blank white. With it the shell survives, says what happened in
// words, and Retry re-runs the import once the server is back.
test('a lazy tab chunk that fails to load is contained to its panel, and Retry recovers', async ({ page }) => {
  let blocked = true;
  // Gallery is the Outputs tab; its chunk is `assets/Gallery-<hash>.js`.
  await page.route(/\/assets\/Gallery-[^/]+\.js$/, (route) => (blocked ? route.abort('failed') : route.continue()));

  await page.goto('/#/gallery');

  const alert = page.getByRole('alert').filter({ hasText: 'This panel could not be loaded' });
  await expect(alert, 'the panel says the chunk failed').toBeVisible();
  await expect(alert).toContainText('usually the server restarted');

  // The shell is still there: the header, and the nav still works.
  await expect(page.locator('header nav')).toBeVisible();
  await expect(page.locator('header nav button[aria-current="page"]')).toHaveAccessibleName(/Outputs/);

  // Navigating to a healthy tab frees the shell without a reload (resetKey).
  await page.locator('header nav > div:first-child button').nth(1).click();
  await expect(page).toHaveURL(/#\/faceswap$/);
  await expect(alert).toHaveCount(0);

  // Back to the broken one, bring the "server" back, and Retry.
  await page.locator('header nav > div:first-child button').nth(3).click();
  await expect(alert).toBeVisible();
  blocked = false;
  await alert.getByRole('button', { name: 'Retry' }).click();
  await expect(alert, 'Retry re-ran the import and the panel loaded').toHaveCount(0);
  await expect(page.locator('.deferred-fallback')).toHaveCount(0);
  await expect(page.getByRole('heading', { name: /Outputs|Output/i }).first()).toBeVisible();
});

// React.lazy memoizes its first rejection, so before lazyPanel.js a healed server
// did not help: the broken tab re-threw its cached error until a full reload.
test('a failed chunk is retried on its own when the user comes back to the tab', async ({ page }) => {
  let blocked = true;
  const requests = [];
  page.on('request', (r) => { if (/\/assets\/Gallery-[^/]+\.js$/.test(r.url())) requests.push(blocked); });
  await page.route(/\/assets\/Gallery-[^/]+\.js$/, (route) => (blocked ? route.abort('failed') : route.continue()));

  await page.goto('/#/gallery');
  const alert = page.getByRole('alert').filter({ hasText: 'This panel could not be loaded' });
  await expect(alert).toBeVisible();

  blocked = false;                                   // the "server" is back
  await page.locator('header nav > div:first-child button').nth(1).click();
  await expect(page).toHaveURL(/#\/faceswap$/);
  // Wait for the other tab to actually render: the failed panel's boundary is
  // unmounted when its exit animation ends, and that unmount is what releases it.
  await expect(page.getByRole('button', { name: /Start Swapping/ })).toBeVisible();
  await expect(alert).toHaveCount(0);
  await page.locator('header nav > div:first-child button').nth(3).click();
  await expect(page).toHaveURL(/#\/gallery$/);
  await expect(alert, 'no Retry click needed').toHaveCount(0);
  await expect(page.locator('.deferred-fallback')).toHaveCount(0);
  expect(requests.at(-1), 'the last request for the chunk went through the healed route').toBe(false);
});
