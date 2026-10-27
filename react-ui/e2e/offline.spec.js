import { test, expect } from '@playwright/test';
import { TABS, openTab, startOnSettingsTab } from './helpers.js';

// A local app must not need the internet to look right or to say so honestly.
//
//   * the fonts are bundled (src/assets/fonts, @font-face in index.css): no tab makes a request
//     to any origin but the app's own, and the bundled face is the one in use;
//   * Settings -> "Check compatibility" reaches the network on the backend (`git ls-remote`);
//     offline it must say UNVERIFIED rather than imply the install is current or broken.

test('self-hosted fonts: no tab makes a request to a foreign origin', async ({ page, baseURL }) => {
  test.setTimeout(120_000);
  const own = new URL(baseURL).origin;
  const foreign = [];
  page.on('request', (r) => {
    const u = new URL(r.url());
    if (/^https?:$/.test(u.protocol) && u.origin !== own) foreign.push(`${r.method()} ${r.url()}`);
  });
  // Belt and braces: if something DOES try to leave the machine, it fails visibly here.
  await page.route((url) => /^https?:$/.test(url.protocol) && url.origin !== own, (route) => route.abort('internetdisconnected'));

  for (const tab of TABS) await openTab(page, tab.id);

  expect(foreign, 'requests to foreign origins').toEqual([]);

  const fonts = await page.evaluate(async () => {
    await document.fonts.ready;
    return [...document.fonts].map((f) => ({ family: f.family.replace(/"/g, ''), status: f.status }));
  });
  expect(fonts.length, '@font-face rules registered').toBeGreaterThan(0);
  expect(fonts.some((f) => /Plus Jakarta Sans/i.test(f.family) && f.status === 'loaded'),
    `the bundled font is loaded: ${JSON.stringify(fonts)}`).toBe(true);
});

test('Settings: "Check compatibility" reports UNVERIFIED when the machine is offline', async ({ page }) => {
  // The request itself fails (no route to anything): the UI must still answer.
  await page.route('**/api/update/check*', (route) => route.abort('internetdisconnected'));
  await startOnSettingsTab(page, 'diagnostics');
  await openTab(page, 'settings');
  const button = page.getByRole('button', { name: 'Check compatibility' });
  await button.scrollIntoViewIfNeeded();
  await button.click();
  await expect(page.getByText('UNVERIFIED', { exact: true })).toBeVisible();
  // Nothing else changed: the button is usable again and nothing claims SAFE.
  await expect(button).toBeEnabled();
  await expect(page.getByText('SAFE', { exact: true })).toHaveCount(0);
});

test('Settings: a backend that answers UNVERIFIED (git unreachable) is shown as UNVERIFIED', async ({ page }) => {
  await page.route('**/api/update/check*', (route) => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      classification: 'UNVERIFIED', available: false,
      reasons: ['the remote could not be reached'],
      current: { sha: 'a'.repeat(40), date: '2026-09-22T00:00:00+00:00', branch: 'main' },
    }),
  }));
  await startOnSettingsTab(page, 'diagnostics');
  await openTab(page, 'settings');
  const button = page.getByRole('button', { name: 'Check compatibility' });
  await button.scrollIntoViewIfNeeded();
  await button.click();
  await expect(page.getByText('UNVERIFIED', { exact: true })).toBeVisible();
  await expect(page.getByText('the remote could not be reached')).toBeVisible();
});
