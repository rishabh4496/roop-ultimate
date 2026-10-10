import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab, TABS, loadAllowlist, saveAllowlist, UPDATE_BASELINE } from './helpers.js';

// The header nav's contract at the widths the app is actually used at:
//
//   * every tab is either in the strip or one click away under "More" -- none is
//     clipped, scrolled out of the nav's box or off the screen;
//   * the page never scrolls sideways and the header stays on one row;
//   * below 1280px the tabs are icon-only (named by aria-label, with a tooltip),
//     from 1280px up they carry their text;
//   * hash routing is the same: every tab, wherever it lives, is #/<id>.
//
// The first test keeps the allowlist mechanism the old version had (a recorded
// list of tab names known to be clipped per width). It is EMPTY now; a clipped
// tab is a regression.
const WIDTHS = [1024, 1280, 1440, 1920];

// The tab strip's own contract, from App.jsx ALL_TABS. `processing` is transient
// (only while a run exists) and is covered by its own test.
const PRIMARY = ['Home', 'Face Swap', 'Batch Matrix', 'Outputs', 'Settings'];
const SECONDARY = [
  { id: 'facemgr', label: 'Face Manager' },
  { id: 'extras', label: 'Editor' },
  { id: 'history', label: 'History' },
];

const NAV = 'header nav';
const MORE_LIST = '#header-more-tabs';

/** Every nav button's name (the strip, then the More trigger): aria-label when icon-only, else the visible text. */
const stripNames = () => [...document.querySelectorAll('header nav button')]
  .map((b) => b.getAttribute('aria-label') || b.textContent.trim());

async function openMore(page) {
  await page.locator(`${NAV} button[aria-controls="header-more-tabs"]`).click();
  await expect(page.locator(MORE_LIST)).toBeVisible();
}

for (const width of WIDTHS) {
  test(`nav visibility @ ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    await openTab(page, 'faceswap');

    const report = await page.evaluate(() => {
      const nav = document.querySelector('header nav');
      const navBox = nav.getBoundingClientRect();
      const vw = document.documentElement.clientWidth;
      const name = (b) => b.getAttribute('aria-label') || b.textContent.trim();
      const clipped = [...nav.querySelectorAll('button')]
        .filter((b) => {
          const r = b.getBoundingClientRect();
          return r.left < Math.max(0, navBox.left) - 1 || r.right > Math.min(vw, navBox.right) + 1;
        })
        .map(name);
      const header = document.querySelector('header');
      const brand = header.firstElementChild.getBoundingClientRect();
      const controls = nav.parentElement.getBoundingClientRect();
      // Everything in the header must sit inside the header's own box.
      const hb = header.getBoundingClientRect();
      const spill = [...header.querySelectorAll('button')].filter((b) => {
        const r = b.getBoundingClientRect();
        return r.width && (r.left < hb.left - 1 || r.right > hb.right + 1);
      }).map(name);
      return {
        clipped,
        spill,
        navScrolls: nav.firstElementChild.scrollWidth > nav.firstElementChild.clientWidth + 1,
        pageScrollsSideways: document.documentElement.scrollWidth > vw + 1,
        // Wrapped = the controls dropped below the brand instead of beside it.
        wrapped: controls.top >= brand.bottom - 1,
      };
    });

    // Sideways page scroll and a wrapped header are reported as pseudo-entries so
    // they ride the same allowlist as a clipped tab.
    const problems = [...report.clipped, ...report.spill.map((n) => `${n} (outside header)`)];
    if (report.pageScrollsSideways) problems.push('(page scrolls horizontally)');
    if (report.navScrolls) problems.push('(nav strip scrolls)');
    if (report.wrapped) problems.push('(header wraps to two rows)');

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

  test(`every tab is visible or one click away @ ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openTab(page, 'faceswap');

    // The strip: the five primaries, in order, each named (aria-label when
    // icon-only, so a screen reader still hears "Home" and not nothing).
    const names = await page.evaluate(stripNames);
    expect(names.slice(0, PRIMARY.length)).toEqual(PRIMARY);
    expect(names.at(-1), 'the last strip button is the More trigger').toMatch(/more tabs/i);

    // Icon-only below 1280, labelled from 1280 up.
    const visibleText = await page.evaluate(
      () => [...document.querySelectorAll('header nav > div:first-child button')].map((b) => b.textContent.trim()));
    if (width < 1280) expect(visibleText.slice(0, 5)).toEqual(['', '', '', '', '']);
    else expect(visibleText.slice(0, 5)).toEqual(PRIMARY);

    // One click reveals every other tab, fully on screen and not covered by
    // anything (elementFromPoint at its centre is the item itself).
    await openMore(page);
    const listed = await page.evaluate(() => [...document.querySelectorAll('#header-more-tabs button')].map((b) => {
      const r = b.getBoundingClientRect();
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return {
        text: b.textContent.trim(),
        inside: r.left >= 0 && r.right <= document.documentElement.clientWidth && r.top >= 0 && r.bottom <= window.innerHeight,
        reachable: b === hit || b.contains(hit),
      };
    }));
    expect(listed.map((i) => i.text)).toEqual(SECONDARY.map((s) => s.label));
    for (const item of listed) {
      expect(item.inside, `${item.text} is on screen`).toBe(true);
      expect(item.reachable, `${item.text} is not covered by anything`).toBe(true);
    }

    // Each one routes by hash exactly as it did when it was in the strip, and the
    // trigger then names it as the current tab.
    for (const s of SECONDARY) {
      if (!(await page.locator(MORE_LIST).isVisible())) await openMore(page);
      await page.locator(`${MORE_LIST} button`, { hasText: s.label }).click();
      await expect(page).toHaveURL(new RegExp(`#/${s.id}$`));
      await expect(page.locator(`${NAV} button[aria-current="page"]`)).toHaveAccessibleName(new RegExp(`^${s.label}, more tabs`));
      await expect(page.locator(MORE_LIST)).toHaveCount(0);
    }

    // And a primary still routes the same way.
    await page.locator(`${NAV} > div:first-child button`).nth(0).click();
    await expect(page).toHaveURL(/#\/home$/);
  });
}

// The widest the header gets is a run in flight: the chip beside the brand is
// ~186px of extra width. A run is started BEFORE the page loads because the app
// only learns of a run it did not start itself on load (job recovery). The mock
// run lasts ~4s, so measure straight away and fail loudly, not silently, if the
// chip was already gone.
//
// 1280-1366 are deliberately absent: there the labelled tabs plus the chip are
// wider than the row and the header wraps to a second row (nothing is clipped).
// Keeping that on one row would mean removing text from the chip or the brand.
for (const width of [1024, 1440, 1920]) {
  test(`header stays on one row with a run in flight @ ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.request.post('/api/stop', { data: {} });
    await page.request.post('/api/swap', { data: {} });
    try {
      await page.goto('/#/facemgr');
      await page.locator('header nav button[aria-current="page"]').waitFor();
      const m = await page.evaluate(() => {
        const header = document.querySelector('header');
        const brand = header.firstElementChild.getBoundingClientRect();
        const controls = header.querySelector('nav').parentElement.getBoundingClientRect();
        return {
          chip: /Processing/.test(header.firstElementChild.innerText),
          wrapped: controls.top >= brand.bottom - 1,
          pageScrollsSideways: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
        };
      });
      expect(m.chip, 'the run chip is showing (else this measured nothing)').toBe(true);
      expect(m.wrapped, 'controls dropped to a second row').toBe(false);
      expect(m.pageScrollsSideways).toBe(false);
    } finally {
      await page.request.post('/api/stop', { data: {} });
    }
  });
}

test('tooltips: icon-only tabs name themselves on hover and keyboard focus, labelled tabs do not', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 900 });
  await openTab(page, 'faceswap');
  const tip = page.locator('header span.pointer-events-none[aria-hidden="true"]');

  const settings = page.locator(`${NAV} > div:first-child button`).nth(4);
  await settings.hover();
  await expect(tip).toHaveText('Settings');
  const [t, b] = await Promise.all([tip.boundingBox(), settings.boundingBox()]);
  expect(t.y, 'tooltip sits below its button').toBeGreaterThanOrEqual(b.y + b.height - 1);
  expect(t.x, 'tooltip is not cut off at the left').toBeGreaterThanOrEqual(0);
  expect(t.x + t.width, 'tooltip is not cut off at the right').toBeLessThanOrEqual(1024);
  await page.mouse.move(5, 400);
  await expect(tip, 'moving the pointer away dismisses a hover tooltip').toHaveCount(0);

  // Keyboard focus (a Tab press, so :focus-visible matches) shows it too, and
  // Escape on the focused control dismisses it.
  await page.locator(`${NAV} > div:first-child button`).nth(0).focus();
  await page.keyboard.press('Tab');
  await expect(tip).toHaveText('Face Swap');
  await page.keyboard.press('Escape');
  await expect(tip).toHaveCount(0);
  await page.keyboard.press('Tab');
  await expect(tip).toHaveText('Batch Matrix');

  // The More trigger sits against the right edge; its tooltip must stay on screen.
  await page.locator(`${NAV} button[aria-controls="header-more-tabs"]`).hover();
  await expect(tip).toHaveText('More tabs');
  const m = await tip.boundingBox();
  expect(m.x + m.width).toBeLessThanOrEqual(1024);

  // At 1280 the text is on the tab itself, so there is nothing to add.
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.locator(`${NAV} > div:first-child button`).nth(4).hover();
  await expect(tip).toHaveCount(0);
});

test('More is a keyboard disclosure: Enter opens onto the list, arrows move, Escape returns focus', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'faceswap');
  const trigger = page.locator(`${NAV} button[aria-controls="header-more-tabs"]`);
  await expect(trigger).toHaveAttribute('aria-expanded', 'false');

  await trigger.focus();
  await page.keyboard.press('Enter');
  await expect(trigger).toHaveAttribute('aria-expanded', 'true');
  const items = page.locator(`${MORE_LIST} button`);
  await expect(items.nth(0)).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(items.nth(1)).toBeFocused();
  await page.keyboard.press('End');
  await expect(items.nth(2)).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(items.nth(0)).toBeFocused();

  await page.keyboard.press('Escape');
  await expect(page.locator(MORE_LIST)).toHaveCount(0);
  await expect(trigger).toBeFocused();

  // Enter on an item navigates, closes, and leaves focus on the trigger.
  await page.keyboard.press('Enter');
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#\/extras$/);
  await expect(trigger).toBeFocused();

  // Clicking away closes it too.
  await trigger.click();
  await expect(page.locator(MORE_LIST)).toBeVisible();
  await page.mouse.click(10, 500);
  await expect(page.locator(MORE_LIST)).toHaveCount(0);
});

test('a tab under More is current from the URL alone, and Back returns', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await openTab(page, 'facemgr');
  const current = page.locator(`${NAV} button[aria-current="page"]`);
  await expect(current).toHaveText('Face Manager');
  await expect(current).toHaveAccessibleName('Face Manager, more tabs');

  await page.locator(`${NAV} > div:first-child button`).nth(1).click();
  await expect(page).toHaveURL(/#\/faceswap$/);
  await page.goBack();
  await expect(page).toHaveURL(/#\/facemgr$/);
  await expect(current).toHaveText('Face Manager');
});

test('Processing (transient) lives under More, and follows a run there', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'processing');
  const current = page.locator(`${NAV} button[aria-current="page"]`);
  await expect(current).toHaveText('Processing');
  await openMore(page);
  await expect(page.locator(`${MORE_LIST} button`)).toHaveText(['Processing', 'Face Manager', 'Editor', 'History']);
  await expect(page.locator(`${MORE_LIST} button[aria-current="page"]`)).toHaveText('Processing');
});

test('the active tab is scrolled into view when the strip is too narrow for all of them', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'home');
  // Force the overflow the strip is only a safety net for (a phone, huge fonts).
  await page.evaluate(() => { document.querySelector('header nav > div:first-child').style.maxWidth = '120px'; });
  const strip = page.locator(`${NAV} > div:first-child`);
  expect(await strip.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);

  for (const [id, index] of [['settings', 4], ['gallery', 3], ['home', 0]]) {
    await page.evaluate((h) => { window.location.hash = h; }, `#/${id}`);
    const button = page.locator(`${NAV} > div:first-child button`).nth(index);
    await expect(button).toHaveAttribute('aria-current', 'page');
    const inView = await page.evaluate((i) => {
      const sc = document.querySelector('header nav > div:first-child');
      const el = sc.querySelectorAll('button')[i];
      const s = sc.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      return r.left >= s.left - 1 && r.right <= s.right + 1;
    }, index);
    expect(inView, `${id} is inside the strip's visible area`).toBe(true);
  }
});

test('the UI zoom is accounted for: 125% on a 1440px window lays out like 1152px', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('roop_zoom', '1.25'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'faceswap');
  const visibleText = await page.evaluate(
    () => [...document.querySelectorAll('header nav > div:first-child button')].map((b) => b.textContent.trim()));
  expect(visibleText.slice(0, 5), '1440 / 1.25 < 1280, so icon-only').toEqual(['', '', '', '', '']);

  // The tooltip is positioned in layout pixels under a CSS zoom; it must still
  // land under its own button rather than drifting by the zoom factor.
  const home = page.locator(`${NAV} > div:first-child button`).nth(0);
  await home.hover();
  const tip = page.locator('header span.pointer-events-none[aria-hidden="true"]');
  await expect(tip).toHaveText('Home');
  const [t, b] = await Promise.all([tip.boundingBox(), home.boundingBox()]);
  expect(Math.abs((t.x + t.width / 2) - (b.x + b.width / 2)), 'tooltip centred on its button').toBeLessThan(6);
  expect(Math.abs(t.y - (b.y + b.height)), 'tooltip directly under its button').toBeLessThan(16);
});

test('axe: the header with More open', async ({ page }) => {
  for (const width of [1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await openTab(page, 'faceswap');
    await openMore(page);
    const { violations } = await new AxeBuilder({ page })
      .include('header')
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'])
      .analyze();
    expect(
      violations.map((v) => ({ rule: v.id, targets: v.nodes.slice(0, 3).map((n) => n.target.join(' ')) })),
      `axe violations in the header @ ${width}px`,
    ).toEqual([]);
    // Leave it closed: the next iteration's click would otherwise close it.
    await page.keyboard.press('Escape');
    await expect(page.locator(MORE_LIST)).toHaveCount(0);
  }
});

test('every tab id in the app still has a hash route (strip + More = ALL_TABS)', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  for (const t of TABS) {
    await openTab(page, t.id);
    await expect(page.locator(`${NAV} button[aria-current="page"]`)).toBeVisible();
    await expect(page).toHaveURL(new RegExp(`#/${t.id}$`));
  }
});
