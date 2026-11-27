import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { openTab, loadAllowlist, saveAllowlist, UPDATE_BASELINE } from './helpers.js';
import { measureContrast } from './contrast.js';

// Computed-background contrast under EVERY preset theme. Opt-in (about 5 minutes):
//
//   npm run test:e2e:themes            fail if a theme got WORSE than its baseline
//   npm run test:e2e:themes:baseline   re-record e2e/allowlist.json -> themeContrast
//
// contrast.spec.js guards the theme users get by default and allows nothing. This
// is the other 36: each theme's count of text nodes below AA (home + face swap +
// settings) is a ceiling, so the debt can only shrink. It exists because the
// default-theme suite cannot see a regression that only some themes have -- the
// first version of the `--bg-base` change painted the five AMOLED themes WHITE
// (a plain-colour `--bg-gradient` is only legal as the last background layer) and
// every default-theme check stayed green.
//
// Themes are applied the way applyThemeToDom does it (class on <html> and <body>,
// data-theme-mode), and the spec CHECKS it took: an earlier sweep that relied on
// the settings API reported every theme at the Default's numbers.
test.skip(process.env.E2E_THEMES !== '1', 'opt-in: npm run test:e2e:themes');

const src = readFileSync(new URL('../src/themes.js', import.meta.url), 'utf8');
const THEMES = [...src.matchAll(/\{ name: '([^']+)',\s+className: '([^']*)'[^}]*mode: '(dark|light)'/g)]
  .map((m) => ({ name: m[1], cls: m[2], mode: m[3] }));
const TABS = ['home', 'faceswap', 'settings'];

test('the theme list was parsed', () => {
  expect(THEMES.length, 'themes parsed out of src/themes.js').toBeGreaterThanOrEqual(30);
});

for (const theme of THEMES) {
  test(`theme contrast: ${theme.name}`, async ({ page }, testInfo) => {
    test.setTimeout(60_000);
    let failing = 0;
    const samples = [];
    for (const tab of TABS) {
      await openTab(page, tab);
      await page.evaluate(({ cls, mode }) => {
        const root = document.documentElement;
        for (const el of [root, document.body]) {
          [...el.classList].filter((c) => c.startsWith('theme-')).forEach((c) => el.classList.remove(c));
          if (cls) el.classList.add(cls);
        }
        root.setAttribute('data-theme-mode', mode);
      }, theme);
      await page.waitForTimeout(900);   // the body background transitions over 0.6 s

      // It took, or the numbers below are the Default's.
      const applied = await page.evaluate(() => ({
        mode: document.documentElement.getAttribute('data-theme-mode'),
        cls: [...document.documentElement.classList].find((c) => c.startsWith('theme-')) || '',
      }));
      expect(applied, `${theme.name} was not applied`).toEqual({ mode: theme.mode, cls: theme.cls });
      // A painted canvas: a page that lost its background shows the browser's white.
      const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      expect(bg, `${theme.name}: body has no background colour`).not.toBe('rgba(0, 0, 0, 0)');

      const r = await page.evaluate(measureContrast, {});
      failing += r.fails.length;
      samples.push(...r.fails.slice(0, 2).map((f) => `${tab}: ${f.ratio}:1 ${f.fg} on ${f.bg} "${f.text}"`));
    }
    testInfo.annotations.push({ type: 'below AA', description: `${failing}` });

    if (UPDATE_BASELINE) {
      saveAllowlist((a) => { a.themeContrast = { ...(a.themeContrast || {}), [theme.name]: failing }; });
      return;
    }
    const ceiling = (loadAllowlist().themeContrast || {})[theme.name];
    expect(ceiling, `no themeContrast baseline for ${theme.name} -- run npm run test:e2e:themes:baseline`).toBeDefined();
    if (failing < ceiling) {
      testInfo.annotations.push({ type: 'tighten allowlist', description: `${theme.name}: ${failing} < ${ceiling}` });
    }
    expect(failing, `${theme.name}: text below AA (baseline ${ceiling}). Examples:\n${samples.join('\n')}`).toBeLessThanOrEqual(ceiling);
  });
}
