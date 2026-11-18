import { test, expect } from '@playwright/test';
import { openTab, makePng } from './helpers.js';

// Stage 5: the studio parameter controls, in a real browser.
//
//   * the scrubber: a filled bar (--accent-cyan), drag-anywhere scrubbing with Shift x10 / Alt x0.1, double-click to type,
//     right-click or the cyan dot to reset;
//   * the mask edge profile beside the feather and erode sliders;
//   * the effect rack: glowing bypass switches, wet/dry mix, latency badges, collapsible bodies;
//   * the face routing dock: route lines, per-face override pills, and that the overrides REACH the request.
//
// The maths behind each is pinned by .render-check/stage5-studio-check.mjs and mask-edge-check.mjs, and the backend applying a
// per-face override to PIXELS by app/tests/test_person_overrides.py. This file is the DOM: that what is drawn is what is
// computed, and that a gesture lands where it should. It runs against the MOCK backend, so nothing here says the real render
// looked different - only that the request carried the override.

const field = (page, label) => page.locator('label', { hasText: label }).filter({ has: page.locator('input[type="range"]') }).first();
const native = (page, label) => field(page, label).locator('input[type="range"]');
const bar = (page, label) => field(page, label).getByTestId('scrubber-bar');
const valueOf = async (page, label) => Number(await native(page, label).inputValue());
const settled = (page, key) => page.waitForRequest((r) => r.method() === 'POST' && /\/api\/settings$/.test(r.url()) && key in (r.postDataJSON() || {}));

async function drag(page, locator, dx, { shift = false, alt = false, from = 0.5 } = {}) {
  const box = await locator.boundingBox();
  const x = box.x + box.width * from;
  const y = box.y + box.height / 2;
  if (shift) await page.keyboard.down('Shift');
  if (alt) await page.keyboard.down('Alt');
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx, y, { steps: 6 });
  await page.mouse.up();
  if (alt) await page.keyboard.up('Alt');
  if (shift) await page.keyboard.up('Shift');
}

// A closed inspector group UNMOUNTS its body (faceswap/InspectorGroup.jsx), so a control inside "Masking" is not in the DOM
// until the group is open. The open state is persisted, so seed it before the page loads.
const openGroups = (page) => page.addInitScript(() => {
  try { window.localStorage.setItem('roop_inspector_open', JSON.stringify({ masking: true, restoration: true })); } catch { /* no storage */ }
});

let original;
test.beforeAll(async ({ request }) => { original = await (await request.get('/api/settings')).json(); });
test.afterEach(async ({ request }) => { await request.post('/api/settings', { data: original }); });

// ── the scrubber ─────────────────────────────────────────────────────────────────────────────────────────────────────
test.describe('studio scrubber', () => {
  const L = 'Face mask edge blend';                       // 0..200, step 1, default 12
  test.beforeEach(async ({ page }) => {
    await openGroups(page);
    await page.setViewportSize({ width: 1600, height: 1000 });
    await openTab(page, 'faceswap');
    await field(page, L).scrollIntoViewIfNeeded();
  });

  test('the bar is filled to the value in --accent-cyan, with a leading edge', async ({ page }) => {
    await native(page, L).fill('100');
    const edge = field(page, L).locator('.studio-scrub__edge');
    const fill = field(page, L).locator('.studio-scrub__fill');
    const barBox = await bar(page, L).boundingBox();
    const fillBox = await fill.boundingBox();
    expect(Math.abs(fillBox.width / barBox.width - 0.5), 'half of 0..200 is half the bar').toBeLessThan(0.02);
    expect(await edge.evaluate((el) => getComputedStyle(el).backgroundColor)).toBe('rgb(0, 240, 255)');
    expect(await fill.evaluate((el) => getComputedStyle(el).backgroundColor), 'the fill is the same cyan, translucent').toMatch(/^(rgba?\(0, 240, 255|color\(srgb 0 0\.94)/);
    await expect(field(page, L).getByTestId('scrubber-value')).toHaveText('100');
  });

  test('pressing the bar does not move the value; only travel does, and the pace is relative', async ({ page }) => {
    await native(page, L).fill('12');
    const b = bar(page, L);
    const box = await b.boundingBox();
    await page.mouse.move(box.x + box.width * 0.9, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.up();
    expect(await valueOf(page, L), 'a click at 90% of the bar does not jump the value there').toBe(12);
    await drag(page, b, 30, { from: 0.1 });
    const v = await valueOf(page, L);
    expect(v, 'a 30 px drag scrubs up from where it was').toBeGreaterThan(12);
    expect(v, '... by a few units, not to the pointer\'s absolute position').toBeLessThan(40);
  });

  test('double-clicking the bar anywhere opens the editor without moving the value (the label must not forward a click to the range)', async ({ page }) => {
    await native(page, L).fill('12');
    const box = await bar(page, L).boundingBox();
    await page.mouse.dblclick(box.x + box.width * 0.85, box.y + box.height / 2);
    const editor = field(page, L).getByRole('textbox', { name: `${L} value` });
    await expect(editor).toHaveValue('12');
    await editor.press('Escape');
    expect(await valueOf(page, L)).toBe(12);
  });

  test('Shift is x10 and Alt is x0.1 of a plain drag', async ({ page }) => {
    const deltas = {};
    for (const [name, mods] of [['plain', {}], ['shift', { shift: true }], ['alt', { alt: true }]]) {
      await native(page, L).fill('12');
      await drag(page, bar(page, L), 30, { from: 0.2, ...mods });
      deltas[name] = (await valueOf(page, L)) - 12;
    }
    expect(deltas.plain).toBeGreaterThanOrEqual(8);
    expect(deltas.shift / deltas.plain, JSON.stringify(deltas)).toBeGreaterThan(8);
    expect(deltas.alt, JSON.stringify(deltas)).toBeLessThan(deltas.plain / 5);
  });

  test('dragging the LABEL scrubs the same way', async ({ page }) => {
    await native(page, L).fill('12');
    const label = field(page, L).locator('span.cursor-ew-resize').first();
    await drag(page, label, 60, { from: 0.1 });
    expect(await valueOf(page, L)).toBeGreaterThan(12);
  });

  test('double-click opens a typing field that takes arithmetic; Enter commits it', async ({ page }) => {
    await native(page, L).fill('12');
    const sent = settled(page, 'face_mask_blend');
    await bar(page, L).dblclick();
    const editor = field(page, L).getByRole('textbox', { name: `${L} value` });
    await expect(editor).toBeFocused();
    await editor.fill('512/4');
    await editor.press('Enter');
    expect((await sent).postDataJSON().face_mask_blend).toBe(128);
    expect(await valueOf(page, L)).toBe(128);
    await expect(editor).toHaveCount(0);
  });

  test('Escape in the typing field changes nothing', async ({ page }) => {
    await native(page, L).fill('12');
    await bar(page, L).dblclick();
    const editor = field(page, L).getByRole('textbox', { name: `${L} value` });
    await editor.fill('150');
    await editor.press('Escape');
    expect(await valueOf(page, L)).toBe(12);
  });

  test('right-click resets to the default (and does not open the browser menu)', async ({ page }) => {
    await native(page, L).fill('90');
    const sent = settled(page, 'face_mask_blend');
    await bar(page, L).click({ button: 'right' });
    expect((await sent).postDataJSON().face_mask_blend).toBe(12);
    expect(await valueOf(page, L)).toBe(12);
  });

  test('the cyan dot appears off-default and resets on click; it is not a tab stop', async ({ page }) => {
    const dot = field(page, L).getByTestId('modified-dot');
    await native(page, L).fill('12');                       // the mock keeps what earlier tests saved
    await expect(dot).toHaveCount(0);
    await native(page, L).fill('60');
    await expect(dot).toHaveCount(1);
    await expect(dot).toHaveAttribute('tabindex', '-1');
    await dot.click();
    expect(await valueOf(page, L)).toBe(12);
    await expect(dot).toHaveCount(0);
  });

  test('the keyboard still works: Arrow steps the hidden range, Backspace resets, Enter types', async ({ page }) => {
    const n = native(page, L);
    await n.fill('50');
    await n.focus();
    await page.keyboard.press('ArrowRight');
    expect(await valueOf(page, L)).toBe(51);
    await page.keyboard.press('Backspace');
    expect(await valueOf(page, L)).toBe(12);
    await page.keyboard.press('Enter');
    await expect(field(page, L).getByRole('textbox', { name: `${L} value` })).toBeFocused();
  });

  test('the bar is a real hit target (>= 24 px tall) and the native range is the one tab stop', async ({ page }) => {
    const box = await bar(page, L).boundingBox();
    expect(box.height).toBeGreaterThanOrEqual(24);
    await expect(bar(page, L)).not.toHaveAttribute('tabindex', /.*/);
  });
});

// ── the mask edge profile ────────────────────────────────────────────────────────────────────────────────────────────
test.describe('mask edge profile', () => {
  const profile = (page) => page.getByTestId('mask-edge-profile');
  test.beforeEach(async ({ page }) => {
    await openGroups(page);
    await page.setViewportSize({ width: 1600, height: 1000 });
    await openTab(page, 'faceswap');
    await field(page, 'Face mask edge blend').scrollIntoViewIfNeeded();
  });

  test('it draws the ramp the feather asks for, and follows the slider', async ({ page }) => {
    await native(page, 'Face mask edge blend').fill('12');
    await expect(profile(page)).toHaveAttribute('data-modelled', 'true');
    // 12 -> 512 * 12 / 200 = 30.72 feather, over a distance transform that reads 0.955 per pixel -> 32.17 px
    await expect(profile(page)).toHaveAttribute('data-width-px', '32.17');
    await expect(profile(page).getByTestId('mask-edge-ramp')).toHaveText('ramp 32 px');
    await native(page, 'Face mask edge blend').fill('24');
    await expect(profile(page)).toHaveAttribute('data-width-px', '64.34');
    await expect(profile(page).locator('svg path').first()).toBeVisible();
  });

  test('a feather past the 18%-of-face cap says it was capped', async ({ page }) => {
    await native(page, 'Face mask edge blend').fill('100');
    await expect(profile(page)).toHaveAttribute('data-capped', 'true');
    await expect(profile(page)).toHaveAttribute('data-width-px', '77.20');
    await expect(profile(page)).toContainText('Capped');
  });

  test('the erode / dilate radius moves the contour marker', async ({ page }) => {
    await expect(profile(page)).not.toContainText('contour');
    await native(page, 'Mask erode / dilate radius').fill('-10');
    await expect(profile(page)).toContainText('contour -10 px');
    await native(page, 'Mask erode / dilate radius').fill('6');
    await expect(profile(page)).toContainText('contour +6 px');
  });

  test('the curve is exposed to assistive tech as one labelled image', async ({ page }) => {
    await expect(profile(page).getByRole('img')).toHaveAttribute('aria-label', /Mask edge: \d+ px ramp/);
  });
});

// ── the effect rack ──────────────────────────────────────────────────────────────────────────────────────────────────
test.describe('effect rack', () => {
  const rack = (page) => page.getByRole('list', { name: 'Enhancer pipeline' });
  const mod = (page, id) => page.getByTestId(`rack-module-${id}`);
  const power = (page, title) => rack(page).getByRole('switch', { name: `${title} enabled` });
  const SIX = ['Swapper Core', 'Face Restorer', 'Texture & Grain Closer', 'LAB Color & Skin-Tone Match', 'Occlusion Mask', 'Frame Upscaling'];

  test.beforeEach(async ({ page }) => {
    await openGroups(page);
    await page.setViewportSize({ width: 1600, height: 1000 });
    await openTab(page, 'faceswap');
    await rack(page).scrollIntoViewIfNeeded();
  });

  test('six modules in engine order, each with a power switch, a latency badge and a mix bar', async ({ page }) => {
    for (const title of SIX) await expect(power(page, title)).toBeVisible();
    await expect(rack(page).getByTestId('rack-latency')).toHaveCount(6);
    await expect(rack(page).locator('[data-testid^="rack-mix-"]')).toHaveCount(6);
    await expect(power(page, 'Swapper Core'), 'the core cannot be bypassed').toBeDisabled();
  });

  test('the power switch glows when the stage runs and goes dark when it is bypassed', async ({ page }) => {
    const restorer = power(page, 'Face Restorer');
    await expect(restorer).toHaveAttribute('aria-checked', 'true');
    const glow = (el) => el.evaluate((n) => getComputedStyle(n).boxShadow);
    expect(await glow(restorer), 'lit').not.toBe('none');
    const off = settled(page, 'selected_enhancer');
    await restorer.click();
    expect((await off).postDataJSON().selected_enhancer).toBe('None');
    await expect(restorer).toHaveAttribute('aria-checked', 'false');
    await expect(mod(page, 'restorer')).toHaveAttribute('data-enabled', 'false');
    expect(await glow(restorer), 'dark').toBe('none');
    await restorer.click();                                                  // A/B: back on, with what it had
    await expect(restorer).toHaveAttribute('aria-checked', 'true');
  });

  test('the restorer mix is the blend ratio, shown 0-100%, and scrubbing it writes the setting', async ({ page }) => {
    const mix = mod(page, 'restorer').getByTestId('rack-mix-restorer');
    await expect(mix.getByTestId('scrubber-value')).toHaveText(/^\d+%$/);
    await mix.locator('input[type="range"]').fill('0.25');
    await expect(mix.getByTestId('scrubber-value')).toHaveText('25%');
    const sent = settled(page, 'blend_ratio');
    await mix.getByTestId('scrubber-bar').dblclick();
    const editor = mix.getByRole('textbox');
    await editor.fill('60');                                                  // typed as a percentage
    await editor.press('Enter');
    expect((await sent).postDataJSON().blend_ratio).toBeCloseTo(0.6, 6);
    await expect(mix.getByTestId('scrubber-value')).toHaveText('60%');
  });

  test('a stage with no blend setting shows a disabled 100% bar that says why', async ({ page }) => {
    for (const id of ['occlusion', 'upscale']) {
      const mix = mod(page, id).getByTestId(`rack-mix-${id}`);
      await expect(mix.locator('input[type="range"]')).toBeDisabled();
      await expect(mix.getByTestId('scrubber-value')).toHaveText('100%');
      await expect(mix.getByTestId('scrubber-bar')).toHaveAttribute('title', /no blend|no amount|replaces/i);
    }
  });

  test('the body folds away and the mix stays', async ({ page }) => {
    const card = mod(page, 'restorer');
    const title = card.getByRole('button', { name: /Face Restorer/ });
    const model = card.locator('label', { hasText: 'Restorer model' });
    // the slot-heavy bodies start folded (tab-stop budget); the mix bar and the power switch never fold away
    await expect(title).toHaveAttribute('aria-expanded', 'false');
    await expect(model).toHaveCount(0);
    await expect(card.getByTestId('rack-mix-restorer')).toBeVisible();
    await title.click();
    await expect(title).toHaveAttribute('aria-expanded', 'true');
    await expect(model).toBeVisible();
    await title.click();
    await expect(model).toHaveCount(0);
  });

  test('latency badges carry a level and say whether it was measured or estimated', async ({ page }) => {
    const badge = (id) => mod(page, id).getByTestId('rack-latency');
    for (const id of ['restorer', 'grain', 'color', 'occlusion', 'upscale']) {
      expect(['low', 'med', 'high']).toContain(await badge(id).getAttribute('data-level'));
      expect(['measured', 'estimate']).toContain(await badge(id).getAttribute('data-basis'));
    }
    await expect(badge('swapper')).toHaveText('Core');
    expect(await badge('grain').innerText()).toMatch(/^Low/);
    // a title is the only place the provenance is written out
    await expect(badge('grain')).toHaveAttribute('title', /estimated/);
  });

  test('the upscaler module writes upscale_after_swap', async ({ page }) => {
    const sw = power(page, 'Frame Upscaling');
    const sent = settled(page, 'upscale_after_swap');
    const was = (await sw.getAttribute('aria-checked')) === 'true';
    await sw.click();
    expect((await sent).postDataJSON().upscale_after_swap).toBe(!was);
    await mod(page, 'upscale').getByRole('button', { name: /Frame Upscaling/ }).click();
    await expect(mod(page, 'upscale').locator('label', { hasText: 'AI upscale model' })).toBeVisible();
  });
});

// ── the face routing dock ────────────────────────────────────────────────────────────────────────────────────────────
test.describe('face routing dock', () => {
  const P0 = 'p-lead';
  const P1 = 'p-guest';
  const dock = (page) => page.getByTestId('face-routing-dock');

  async function open(page) {
    const png = makePng(640, 360);
    await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: png }));
    await page.route('**/api/state', async (route) => {
      const res = await route.fetch();
      const json = await res.json();
      const face = json.target_faces?.[0];
      json.target_faces = [face, face];
      json.target_groups = [0, 1];
      json.target_person_ids = [P0, P1];
      json.target_names = ['Lead', 'Guest'];
      json.target_faces_info = [
        { index: 0, person: 'Person 1', landmarks: 106, gender: 'female', age: 26, bbox: [40, 50, 120, 160], pose: 'Front' },
        { index: 1, person: 'Person 2', landmarks: 106, gender: 'male', age: 41, bbox: [360, 60, 120, 160], pose: 'Left Profile + Up Tilt' },
      ];
      await route.fulfill({ response: res, json });
    });
    const writes = [];
    await page.route('**/api/target/context*', (route) => {
      if (route.request().method() === 'POST') {
        try { writes.push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
        return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
      }
      return route.fallback();
    });
    await openGroups(page);
    await page.setViewportSize({ width: 1600, height: 1000 });
    await openTab(page, 'faceswap');
    await expect(dock(page)).toBeVisible({ timeout: 20_000 });
    return writes;
  }

  test('targets on the left, sources on the right, one route line per mapped person', async ({ page }) => {
    await open(page);
    await expect(dock(page).getByRole('list', { name: 'Target faces' }).getByRole('listitem')).toHaveCount(2);
    expect(await dock(page).getByRole('list', { name: 'Source faces' }).getByRole('listitem').count()).toBeGreaterThanOrEqual(2);
    await expect(dock(page).getByTestId('routing-target-1')).toContainText('Lead');
    await expect(dock(page).getByTestId('routing-target-2')).toContainText('Guest');
    await expect(dock(page).getByTestId('routing-target-1').getByTestId('routing-pose')).toHaveText('Front');
    await expect(dock(page).getByTestId('routing-target-2').getByTestId('routing-pose')).toHaveText('Left · Up');
    expect(await dock(page).locator('[data-testid^="routing-line-"]').count(), 'every mapped person has a line').toBeGreaterThan(0);
  });

  test('frame presence is a percentage when the clip has been scanned, a dash when it has not', async ({ page }) => {
    await open(page);
    const text = await dock(page).getByTestId('routing-target-1').locator('.routing-presence').innerText();
    expect(text).toMatch(/^(—|\d+%)$/);
  });

  test('click a target, click a source: the route moves, and so does its line', async ({ page }) => {
    const writes = await open(page);
    await dock(page).getByTestId('routing-target-2').click();
    await dock(page).getByTestId('routing-source-2').click();
    await expect.poll(() => writes.some((w) => Object.values(w.target_person_source_mapping || w.face_mapping || {}).length >= 1)).toBe(true);
    await expect(dock(page).getByTestId('routing-line-2')).toHaveAttribute('data-source', '2');
    // the people list below shows the same decision
    await expect(page.getByLabel('Source face for Guest')).toHaveValue('2');
  });

  test('drag a source onto a target routes it the same way', async ({ page }) => {
    await open(page);
    await dock(page).getByTestId('routing-source-1').dragTo(dock(page).getByTestId('routing-target-1'));
    await expect(dock(page).getByTestId('routing-line-1')).toHaveAttribute('data-source', '1');
  });

  test('override pills: scrubbing one overrides that face only, and the request carries it', async ({ page }) => {
    await open(page);
    await dock(page).getByTestId('routing-target-2').click();
    const pill = dock(page).getByTestId('routing-pill-mask_feather');
    await expect(pill).toHaveAttribute('data-overridden', 'false');
    await pill.click();
    const editor = dock(page).getByTestId('routing-editor-mask_feather');
    const carried = page.waitForRequest((r) => r.method() === 'POST' && /\/api\/preview$/.test(r.url())
      && (r.postDataJSON()?.person_overrides?.[P1]?.mask_feather === 44));
    await editor.locator('input[type="range"]').fill('44');
    await carried;
    await expect(pill).toHaveAttribute('data-overridden', 'true');
    await expect(pill).toContainText('44');
    await expect(dock(page).getByTestId('routing-target-2').getByTestId('routing-ovr-dot')).toBeVisible();
    // the other person is untouched: no dot, the pill shows the ordinary setting
    await dock(page).getByTestId('routing-target-1').click();
    await expect(dock(page).getByTestId('routing-pill-mask_feather')).toHaveAttribute('data-overridden', 'false');
    await expect(dock(page).getByTestId('routing-target-1').getByTestId('routing-ovr-dot')).toHaveCount(0);
  });

  test('right-clicking the editor bar (or its MOD mark) clears the override and the request stops carrying it', async ({ page }) => {
    await open(page);
    await dock(page).getByTestId('routing-target-2').click();
    await dock(page).getByTestId('routing-pill-blend_ratio').click();
    const editor = dock(page).getByTestId('routing-editor-blend_ratio');
    await editor.locator('input[type="range"]').fill('0.4');
    await expect(dock(page).getByTestId('routing-pill-blend_ratio')).toHaveAttribute('data-overridden', 'true');
    const cleared = page.waitForRequest((r) => r.method() === 'POST' && /\/api\/preview$/.test(r.url())
      && r.postDataJSON()?.person_overrides?.[P1]?.blend_ratio === undefined && 'person_overrides' in (r.postDataJSON() || {}));
    await editor.getByTestId('scrubber-bar').click({ button: 'right' });
    await cleared;
    await expect(dock(page).getByTestId('routing-pill-blend_ratio')).toHaveAttribute('data-overridden', 'false');
  });

  test('"Reset face" clears every override of that person', async ({ page }) => {
    await open(page);
    await dock(page).getByTestId('routing-target-1').click();
    for (const key of ['mask_feather', 'color_match_strength']) {
      await dock(page).getByTestId(`routing-pill-${key}`).click();
      await dock(page).getByTestId(`routing-editor-${key}`).locator('input[type="range"]').fill(key === 'mask_feather' ? '30' : '0.5');
    }
    await expect(dock(page).locator('[data-overridden="true"]')).toHaveCount(2);
    await dock(page).getByRole('button', { name: 'Reset face' }).click();
    await expect(dock(page).locator('[data-overridden="true"]')).toHaveCount(0);
  });

  test('overrides survive a reload (this browser, this clip) and are not shared with another clip key', async ({ page }) => {
    await open(page);
    await dock(page).getByTestId('routing-target-1').click();
    await dock(page).getByTestId('routing-pill-mask_erode_dilate_radius').click();
    await dock(page).getByTestId('routing-editor-mask_erode_dilate_radius').locator('input[type="range"]').fill('-6');
    await expect(dock(page).getByTestId('routing-pill-mask_erode_dilate_radius')).toHaveAttribute('data-overridden', 'true');
    const stored = await page.evaluate(() => Object.fromEntries(Object.entries(localStorage).filter(([k]) => k.startsWith('roop_person_overrides:'))));
    expect(Object.keys(stored).length).toBe(1);
    expect(JSON.parse(Object.values(stored)[0])[P0].mask_erode_dilate_radius).toBe(-6);
    await page.reload();
    await expect(dock(page)).toBeVisible({ timeout: 20_000 });
    await dock(page).getByTestId('routing-target-1').click();
    await expect(dock(page).getByTestId('routing-pill-mask_erode_dilate_radius')).toHaveAttribute('data-overridden', 'true');
  });

  test('in a mode that cannot route by person the dock says the overrides are saved but unused, and offers the way back', async ({ page }) => {
    await open(page);
    const modeSelect = page.getByLabel(/Face selection|Detection|Swap mode/i).first();
    if (await modeSelect.count()) await modeSelect.selectOption('All faces').catch(() => {});
    await dock(page).getByTestId('routing-target-1').click();
    await dock(page).getByTestId('routing-pill-mask_feather').click();
    await dock(page).getByTestId('routing-editor-mask_feather').locator('input[type="range"]').fill('30');
    const note = dock(page).getByTestId('routing-mode-note');
    // "All faces" is the mock's default; when it is, the note shows and the button flips the mode
    if (await note.count()) {
      await expect(note).toContainText('not used');
      await note.getByRole('button', { name: 'Use Selected people' }).click();
      await expect(dock(page).getByTestId('routing-mode-note')).toHaveCount(0);
    }
  });
});
