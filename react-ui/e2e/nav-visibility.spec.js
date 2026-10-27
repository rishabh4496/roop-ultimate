import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab, TABS } from './helpers.js';

// The workstation shell's contract at the widths the app is actually used at.
//
//   * every route is a button in the rail, on screen and not covered by anything;
//   * the page never scrolls sideways and nothing in the header spills out of it;
//   * the rail collapses to icons (still named by aria-label), remembers that
//     across a reload, and Ctrl+B does the same;
//   * hash routing is unchanged: every route is #/<id>, and Back returns;
//   * Processing is transient: it joins the rail with a live count while a run
//     (or queued work) exists;
//   * the header's layout switcher, telemetry pill and the Ctrl+K palette work
//     from wherever you are, a text field included.
//
// (This file replaced the old tab-strip contract: allowlisted clipping, the More
// menu and its tooltips. A rail has the window's whole height, so none of those
// can happen; the first test below is the one that would catch a regression.)
const WIDTHS = [1024, 1280, 1440, 1920];

const NAV = 'nav[aria-label="Main"]';
const RAIL_IDLE = ['Home', 'Face Swap', 'Batch Matrix', 'Face Manager', 'Editor', 'Outputs', 'History', 'Settings'];

const railNames = (page) => page.locator(`${NAV} ul button`).evaluateAll(
  (els) => els.map((b) => b.getAttribute('aria-label')));

for (const width of WIDTHS) {
  test(`shell fits @ ${width}px: rail routes reachable, header inside its box, no sideways scroll`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openTab(page, 'faceswap');

    expect(await railNames(page)).toEqual(RAIL_IDLE);

    const report = await page.evaluate(() => {
      const vw = document.documentElement.clientWidth;
      const vh = window.innerHeight;
      const rail = document.querySelector('nav[aria-label="Main"]');
      const items = [...rail.querySelectorAll('ul button')].map((b) => {
        const r = b.getBoundingClientRect();
        const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        return {
          name: b.getAttribute('aria-label'),
          inside: r.left >= 0 && r.right <= vw && r.top >= 0 && r.bottom <= vh,
          reachable: b === hit || b.contains(hit),
        };
      });
      const header = document.querySelector('header');
      const hb = header.getBoundingClientRect();
      const spill = [...header.querySelectorAll('button')].filter((b) => {
        const r = b.getBoundingClientRect();
        return r.width && (r.left < hb.left - 1 || r.right > hb.right + 1);
      }).map((b) => b.getAttribute('aria-label') || b.textContent.trim());
      return { items, spill, sideways: document.documentElement.scrollWidth > vw + 1 };
    });

    for (const it of report.items) {
      expect(it.inside, `${it.name} is on screen`).toBe(true);
      expect(it.reachable, `${it.name} is not covered by anything`).toBe(true);
    }
    expect(report.spill, 'header controls spill out of the header').toEqual([]);
    expect(report.sideways, 'the page scrolls horizontally').toBe(false);
  });
}

test('every route is #/<id>: the rail routes by hash, and Back returns', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  for (const t of TABS) {
    await openTab(page, t.id);
    await expect(page.locator(`${NAV} button[aria-current="page"]`)).toBeVisible();
    await expect(page).toHaveURL(new RegExp(`#/${t.id}$`));
  }

  await openTab(page, 'facemgr');
  await expect(page.locator(`${NAV} button[aria-current="page"]`)).toHaveAccessibleName('Face Manager');
  await page.locator(`${NAV} button[aria-label="Face Swap"]`).click();
  await expect(page).toHaveURL(/#\/faceswap$/);
  await page.goBack();
  await expect(page).toHaveURL(/#\/facemgr$/);
  await expect(page.locator(`${NAV} button[aria-current="page"]`)).toHaveAccessibleName('Face Manager');
});

test('the rail collapses to icons, keeps its names, remembers it across a reload; Ctrl+B toggles', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'faceswap');
  const toggle = page.locator(`${NAV} button[aria-expanded]`);
  const faceSwap = page.locator(`${NAV} ul button[aria-label="Face Swap"]`);

  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await expect(faceSwap).toHaveText('Face Swap');
  const wide = (await page.locator(NAV).boundingBox()).width;

  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(faceSwap, 'icon-only: no visible text, still named').toHaveText('', { useInnerText: true });
  await expect(faceSwap).toHaveAccessibleName('Face Swap');
  expect((await page.locator(NAV).boundingBox()).width).toBeLessThan(wide - 100);

  await openTab(page, 'faceswap');                                  // a fresh load
  await expect(toggle, 'the choice survives a reload').toHaveAttribute('aria-expanded', 'false');

  await page.evaluate(() => document.activeElement?.blur());
  await page.keyboard.press('Control+b');
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await page.keyboard.press('Control+b');
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
});

// A run outlives its test in the shared mock, and the header is widest while one
// is in flight (the chip). It is started BEFORE the page loads because the app
// only learns of a run it did not start itself on load (job recovery); the mock
// run lasts ~4 s, so measure straight away.
test('Processing joins the rail with a count while a run exists, and nothing spills', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.request.post('/api/stop', { data: {} });
  await page.request.post('/api/swap', { data: {} });
  try {
    await page.goto('/#/facemgr');
    await page.locator(`${NAV} button[aria-current="page"]`).waitFor();
    const proc = page.locator(`${NAV} ul button[aria-label^="Processing"]`);
    await expect(proc, 'Processing is in the rail for the run').toBeVisible();
    await expect(proc).toHaveAccessibleName(/1 job/);
    const m = await page.evaluate(() => {
      const header = document.querySelector('header');
      const hb = header.getBoundingClientRect();
      return {
        chip: /Processing/.test(header.innerText),
        spill: [...header.querySelectorAll('button')].some((b) => {
          const r = b.getBoundingClientRect();
          return r.width && (r.left < hb.left - 1 || r.right > hb.right + 1);
        }),
        sideways: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
      };
    });
    expect(m.chip, 'the run chip is showing (else this measured nothing)').toBe(true);
    expect(m.spill).toBe(false);
    expect(m.sideways).toBe(false);
  } finally {
    await page.request.post('/api/stop', { data: {} });
  }
});

test('the layout switcher drives the Face Swap workspace and is the same state as the dock', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'faceswap');
  const group = page.getByRole('group', { name: 'Workspace layout' });
  const standard = group.getByRole('button', { name: 'Standard' });
  const canvas = group.getByRole('button', { name: 'Max Canvas' });
  const inspect = group.getByRole('button', { name: 'Inspection Suite' });
  const inspector = page.getByTestId('inspector');

  await expect(standard).toHaveAttribute('aria-pressed', 'true');
  await expect(inspector).toBeVisible();

  await canvas.click();
  await expect(canvas).toHaveAttribute('aria-pressed', 'true');
  await expect(standard).toHaveAttribute('aria-pressed', 'false');
  await expect(inspector, 'Max Canvas hides the settings column').toBeHidden();

  // It is stored, so it survives leaving the tab and coming back.
  await page.locator(`${NAV} button[aria-label="Home"]`).click();
  await expect(page).toHaveURL(/#\/home$/);
  await page.locator(`${NAV} button[aria-label="Face Swap"]`).click();
  await expect(inspector).toBeHidden();
  await expect(canvas).toHaveAttribute('aria-pressed', 'true');

  // From another tab the switcher takes you to Face Swap, where it applies.
  await page.locator(`${NAV} button[aria-label="Settings"]`).click();
  await expect(page).toHaveURL(/#\/settings$/);
  await standard.click();
  await expect(page).toHaveURL(/#\/faceswap$/);
  await expect(inspector).toBeVisible();

  await inspect.click();
  await expect(inspect).toHaveAttribute('aria-pressed', 'true');
  await expect(inspector, 'Inspection Suite is preview + timeline, no side panels').toBeHidden();
});

test('Ctrl+K opens the palette from inside a text field and runs a layout command', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'settings');
  const field = page.getByRole('searchbox').or(page.locator('input[type="search"], input[placeholder*="earch"]')).first();
  await field.focus();
  await page.keyboard.press('Control+k');
  const palette = page.getByPlaceholder('Search actions, themes, settings…');
  await expect(palette, 'Ctrl+K works while a field has focus').toBeVisible();

  await palette.fill('max canvas');
  await expect(page.getByRole('button', { name: /Layout: Max Canvas/ })).toBeVisible();
  await page.keyboard.press('Enter');
  await expect(palette).toHaveCount(0);
  await expect(page).toHaveURL(/#\/faceswap$/);
  await expect(page.getByTestId('inspector')).toBeHidden();

  await page.keyboard.press('Control+k');
  await palette.fill('provider');
  await expect(page.getByRole('button', { name: /Execution provider: tensorrt/ })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(palette).toHaveCount(0);
});

test('the telemetry pill reads the hardware poll, opens a popover, Escape closes it and returns focus', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'home');
  const pill = page.getByRole('button', { name: /^Hardware telemetry:/ });
  await expect(pill).toContainText('VRAM');
  await expect(pill).toContainText('CPU');
  await expect(pill.locator('canvas')).toHaveCount(1);

  await pill.click();
  const dialog = page.getByRole('dialog', { name: 'Hardware telemetry' });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('System RAM');
  await expect(dialog).toContainText('Execution provider');

  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(pill).toBeFocused();

  // The duplicates it replaced are gone: no HUD toggle in the header, no GPU tile on Home.
  await expect(page.getByRole('button', { name: 'Hardware telemetry HUD' })).toHaveCount(0);
});

test('the UI zoom is accounted for: 125% on a 1440px window still fits', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('roop_zoom', '1.25'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await openTab(page, 'faceswap');
  const sideways = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
  expect(sideways).toBe(false);
  await expect(page.locator(`${NAV} ul button[aria-label="Home"]`)).toBeVisible();
});

test('axe: the rail and the header, expanded and collapsed', async ({ page }) => {
  for (const width of [1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await openTab(page, 'faceswap');
    for (const collapsed of [false, true]) {
      const toggle = page.locator(`${NAV} button[aria-expanded]`);
      const isCollapsed = (await toggle.getAttribute('aria-expanded')) === 'false';
      if (isCollapsed !== collapsed && width >= 768) await toggle.click();
      const { violations } = await new AxeBuilder({ page })
        .include(NAV)
        .include('header')
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'])
        .analyze();
      expect(
        violations.map((v) => ({ rule: v.id, targets: v.nodes.slice(0, 3).map((n) => n.target.join(' ')) })),
        `axe violations in the shell @ ${width}px, collapsed=${collapsed}`,
      ).toEqual([]);
    }
  }
});
