import { test, expect } from '@playwright/test';
import { TABS, openTab } from './helpers.js';

// Every hash route mounts cleanly: no uncaught exception, no console.error, no
// error-boundary fallback, and no request that leaves the machine. Each route
// is its own test so a failure names the route that broke.
//
// The list is the nine routes the React shell exposes (App.jsx ALL_TABS). The
// `#/monitor` pop-out is a tenth route, reached only from inside Face Swap and
// Editor, and is covered by the monitor-specific specs rather than here.

const ROUTES = ['home', 'faceswap', 'batch', 'processing', 'facemgr', 'extras', 'gallery', 'history', 'settings'];

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '::1', '[::1]']);
const CRASH = /The app crashed|The app could not be loaded/;

test('the spec covers exactly the nine shell routes', () => {
  expect(TABS.map((t) => t.id)).toEqual(ROUTES);
});

for (const id of ROUTES) {
  test(`#/${id} mounts with no runtime, console or foreign-network errors`, async ({ page }) => {
    test.setTimeout(90_000);
    const problems = [];

    page.on('pageerror', (error) => problems.push(`uncaught exception: ${error.message}`));
    // Browsers print a failed fetch as a bare "404 (Not Found)" with no URL, so the
    // URL is taken from the response event that carries it and listed beside it.
    const failedResponses = [];
    page.on('response', (response) => {
      if (response.status() >= 400) failedResponses.push(`${response.status()} ${response.request().method()} ${response.url()}`);
    });
    page.on('console', (message) => {
      if (message.type() === 'error') problems.push(`console.error: ${message.text()}`);
    });
    page.on('request', (request) => {
      const url = new URL(request.url());
      if (['data:', 'blob:'].includes(url.protocol)) return;
      if (!LOCAL_HOSTS.has(url.hostname)) problems.push(`request left the machine: ${request.url()}`);
    });

    await openTab(page, id);

    // A route that throws is replaced by the error boundary; the page then has
    // no nav-selected content, so the crash heading is the signal to look for.
    await expect(page.getByText(CRASH)).toHaveCount(0);
    await expect(page.locator(`nav[aria-label="Main"] button[aria-current="page"]`)).toHaveCount(1);
    expect(page.url()).toContain(`#/${id}`);

    for (const line of failedResponses) problems.push(`HTTP error response: ${line}`);
    expect(problems, problems.join('\n')).toEqual([]);
  });
}
