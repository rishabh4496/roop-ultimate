import { test, expect } from '@playwright/test';
import { VIEWS } from './helpers.js';
import { measureContrast } from './contrast.js';

// WCAG AA (4.5:1, or 3:1 for large text) for every visible text node on every
// view, measured against the COMPUTED backdrop -- see contrast.js for how, and
// for why axe's own contrast rule is not enough here (it returns ~1,000
// "incomplete" on this gradient-and-glass UI, which is neither pass nor fail).
//
// There is deliberately no allowlist: the muted-text token (`text-muted`,
// `--muted-pct`) and the accent-as-ink token (`text-accent`, `--accent-ink`) were
// introduced to get this to zero, and a new low-contrast text colour should fail
// here and be fixed with one of them, not recorded.
//
// Default (dark) theme only. Every theme defines its own surfaces, so a theme
// sweep is a different test; this one guards the theme users get.
for (const view of VIEWS) {
  test(`contrast: ${view.id}`, async ({ page }, testInfo) => {
    await view.open(page);
    const r = await page.evaluate(measureContrast, {});

    testInfo.annotations.push({
      type: 'measured',
      description: `${r.checked} text nodes, ${r.fails.length} below AA, ${r.unmeasurable.length} over pictures`,
    });
    expect(r.checked, 'measured no text at all -- is the view empty?').toBeGreaterThan(0);

    const lines = r.fails.map((f) => `${f.ratio}:1 (need ${f.need}) ${f.fg} on ${f.bg}  "${f.text}"  ${f.target}`);
    expect(lines, `text below WCAG AA on ${view.id}`).toEqual([]);
  });
}
