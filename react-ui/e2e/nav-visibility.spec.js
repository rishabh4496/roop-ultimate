import { test, expect } from '@playwright/test';
import { openTab, loadAllowlist, saveAllowlist, UPDATE_BASELINE } from './helpers.js';

// Every tab in the header nav must be reachable at common desktop widths
// WITHOUT scrolling the strip: fully inside the viewport and inside the nav's
// own box (the nav is `overflow-x-auto`, so a clipped tab is silently scrolled
// away rather than wrapped), and the page itself must not scroll sideways.
// The allowlist records, per width, the tab labels already known to be clipped.
const WIDTHS = [1024, 1280, 1440, 1920];

for (const width of WIDTHS) {
  test(`nav visibility @ ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    await openTab(page, 'faceswap');

    const report = await page.evaluate(() => {
      const nav = document.querySelector('header nav');
      const navBox = nav.getBoundingClientRect();
      const vw = document.documentElement.clientWidth;
      const clipped = [...nav.querySelectorAll('button')]
        .filter((b) => {
          const r = b.getBoundingClientRect();
          return r.left < Math.max(0, navBox.left) - 1 || r.right > Math.min(vw, navBox.right) + 1;
        })
        .map((b) => b.textContent.trim());
      return {
        clipped,
        navScrolls: nav.scrollWidth > nav.clientWidth + 1,
        pageScrollsSideways: document.documentElement.scrollWidth > vw + 1,
      };
    });

    // Sideways page scroll is reported as a pseudo-entry so it rides the same
    // allowlist as a clipped tab.
    const problems = [...report.clipped];
    if (report.pageScrollsSideways) problems.push('(page scrolls horizontally)');
    if (report.navScrolls && !problems.length) problems.push('(nav strip scrolls)');

    if (UPDATE_BASELINE) {
      saveAllowlist((a) => {
        if (problems.length) a.navVisibility[width] = problems;
        else delete a.navVisibility[width];
      });
      return;
    }

    const known = loadAllowlist().navVisibility[width] ?? [];
    const fresh = problems.filter((p) => !known.includes(p));
    if (known.some((k) => !problems.includes(k))) {
      testInfo.annotations.push({
        type: 'tighten allowlist',
        description: `${width}px: some known problems are gone -- run npm run test:e2e:baseline`,
      });
    }
    expect(fresh, `newly clipped or overflowing at ${width}px`).toEqual([]);
  });
}
