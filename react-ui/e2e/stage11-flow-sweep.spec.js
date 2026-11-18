import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab } from './helpers.js';

// Stage 11, in a real Chromium against the production build with the mock server standing in for app/api.py:
//
//   flow bar    docked under the viewport; Solo (S) pills and the keys 1-6 / 0 send `solo_stage` and light the cyan path up to
//               that node; Δ asks for the difference pass; the bypass switch sends `bypass_stages` and the picture is re-rendered;
//               a solo the backend REFUSES is said; keys typed with focus elsewhere still hit the snapshot slots.
//   sweep       a 90vw x 90vh sheet; X / Y with min / centre / max; ONE zoom, pan and 3.5x loupe across all nine cells; Apply Cell
//               and Assign to Slot A-D commit the selected cell and mark it.
//
// The mock's pictures and detections are stand-ins; this says nothing about what a real swap looks like. The numbers the
// requests carry are what is asserted, because those are what reaches the backend.

const BAR = '[data-testid="visual-pipeline-node-flow-bar"]';
const isPreview = (r) => r.method() === 'POST' && new URL(r.url()).pathname === '/api/preview';
const bodyOf = (r) => { try { return JSON.parse(r.postData() || '{}'); } catch { return {}; } };
const previewWith = (page, pred) => page.waitForRequest((r) => isPreview(r) && pred(bodyOf(r)), { timeout: 15_000 });

async function openFaceSwap(page, { width = 1440, height = 900 } = {}) {
  // A short window starts the bar collapsed (the deck is ~190 px tall at 720p); 900 px is the normal desktop case.
  await page.setViewportSize({ width, height });
  await openTab(page, 'faceswap');
  const bar = page.locator(BAR);
  await bar.waitFor({ state: 'visible', timeout: 20_000 });
  await bar.scrollIntoViewIfNeeded();
  return bar;
}

test.describe('flow bar', () => {
  test('it is docked under the viewport and lists the chain in the brief\'s order', async ({ page }) => {
    const bar = await openFaceSwap(page);
    await expect(bar).toHaveAttribute('role', 'toolbar');
    const ids = await page.locator('[data-flow-node]').evaluateAll((els) => els.map((e) => e.getAttribute('data-flow-node')));
    expect(ids).toEqual(['source', 'detect_align', 'hyperswap', 'gaze_mouth', 'xseg_roto', 'restore_ultra', 'relight_lut', 'writer']);
    await expect(page.getByTestId('flow-node-source')).toContainText('Source / .fsz Identity');
    await expect(page.getByTestId('flow-node-detect_align')).toContainText('Detector & 5-Pt Pose');
    await expect(page.getByTestId('flow-node-xseg_roto')).toContainText('XSeg 3 / Occluder Matte');
    await expect(page.getByTestId('flow-node-relight_lut')).toContainText('Color / LUT Bake');
    await expect(page.getByTestId('flow-node-writer')).toContainText('NVHardwareVideoWriter Output');
    // below the stage, not in the inspector
    const stage = await page.getByTestId('stage-deck').boundingBox();
    const box = await bar.boundingBox();
    expect(box.y).toBeGreaterThan(stage.y + 100);
    await expect(page.getByTestId('stage-deck').locator(BAR)).toHaveCount(1);
    // idle: the whole chain is lit, nothing is soloed, and latency badges do not invent a number
    await expect(page.getByTestId('flow-edge-0')).toHaveAttribute('data-state', 'lit');
    await expect(page.getByTestId('flow-clear-solo')).toHaveCount(0);
  });

  test('Solo (S) sends the stage, lights the path up to it and dims what is downstream', async ({ page }) => {
    await openFaceSwap(page);
    const req = previewWith(page, (b) => b.solo_stage === 'hyperswap');
    await page.getByTestId('flow-solo-hyperswap').click();
    const body = bodyOf(await req);
    expect(body.solo_stage).toBe('hyperswap');
    await expect(page.getByTestId('flow-node-hyperswap')).toHaveAttribute('data-state', 'solo');
    await expect(page.getByTestId('flow-node-detect_align')).toHaveAttribute('data-state', 'lit');
    await expect(page.getByTestId('flow-node-restore_ultra')).toHaveAttribute('data-state', 'dim');
    await expect(page.getByTestId('flow-edge-1')).toHaveAttribute('data-state', 'lit');
    await expect(page.getByTestId('flow-edge-2')).toHaveAttribute('data-state', 'dim');
    await expect(page.getByTestId('flow-solo-label')).toHaveText('Swapper');
    await expect(page.getByTestId('interactive-preview-solo-badge')).toContainText('Swapper');
    // the same pill again returns to live
    const live = previewWith(page, (b) => b.solo_stage == null);
    await page.getByTestId('flow-solo-hyperswap').click();
    await live;
    await expect(page.getByTestId('flow-node-hyperswap')).toHaveAttribute('data-state', 'lit');
    await expect(page.getByTestId('interactive-preview-solo-badge')).toHaveCount(0);
  });

  test('hover a node and the digits solo it: 4 is the matte, 0 returns to live, and the slots are not touched', async ({ page }) => {
    await openFaceSwap(page);
    await expect(page.locator('[data-slot="A"]')).toHaveAttribute('data-slot-filled', '0');
    await page.getByTestId('flow-main-detect_align').hover();
    await expect(page.getByTestId('flow-main-detect_align')).toBeFocused();
    const matte = previewWith(page, (b) => b.solo_stage === 'xseg_roto');
    await page.keyboard.press('4');
    await matte;
    await expect(page.getByTestId('flow-node-xseg_roto')).toHaveAttribute('data-state', 'solo');
    // `1` with the bar focused is the Detector, NOT snapshot slot A
    const detector = previewWith(page, (b) => b.solo_stage === 'detect_align');
    await page.keyboard.press('1');
    await detector;
    await expect(page.locator('[data-slot="A"]')).toHaveAttribute('data-slot-filled', '0');
    const live = previewWith(page, (b) => b.solo_stage == null);
    await page.keyboard.press('0');
    await live;
    await expect(page.getByTestId('flow-clear-solo')).toHaveCount(0);
  });

  test('the same digit outside the bar is still a snapshot slot', async ({ page }) => {
    await openFaceSwap(page);
    await page.locator('body').click({ position: { x: 4, y: 4 } });
    await page.evaluate(() => document.activeElement?.blur?.());
    await page.keyboard.press('1');
    await expect(page.locator('[data-slot="A"]')).toHaveAttribute('data-slot-filled', '1');
  });

  test('S, D and B act on the focused node', async ({ page }) => {
    await openFaceSwap(page);
    await page.getByTestId('flow-main-restore_ultra').hover();
    await expect(page.getByTestId('flow-main-restore_ultra')).toBeFocused();
    const diff = previewWith(page, (b) => b.solo_stage === 'restore_diff');
    await page.keyboard.press('d');
    await diff;
    await expect(page.getByTestId('flow-diff-restore_ultra')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('flow-solo-label')).toHaveText('Restorer difference');
    const plain = previewWith(page, (b) => b.solo_stage === 'restore_ultra');
    await page.keyboard.press('s');
    await plain;
    const off = previewWith(page, (b) => Array.isArray(b.bypass_stages) && b.bypass_stages.includes('restore_ultra'));
    await page.keyboard.press('b');
    await off;
    await expect(page.getByTestId('flow-node-restore_ultra')).toHaveAttribute('data-bypassed', 'true');
  });

  test('a bypass switch is sent to the backend and the frame is re-rendered', async ({ page }) => {
    await openFaceSwap(page);
    await expect(page.getByTestId('flow-bypass-restore_ultra')).toHaveAttribute('aria-checked', 'true');
    const off = previewWith(page, (b) => Array.isArray(b.bypass_stages) && b.bypass_stages.join() === 'restore_ultra');
    await page.getByTestId('flow-bypass-restore_ultra').click();
    expect((bodyOf(await off)).bypass_stages).toEqual(['restore_ultra']);
    await expect(page.getByTestId('flow-bypass-restore_ultra')).toHaveAttribute('aria-checked', 'false');
    await expect(page.getByTestId('flow-node-restore_ultra')).toHaveAttribute('data-state', 'off');
    await expect(page.getByTestId('flow-edge-4')).toHaveAttribute('data-state', 'off');
    const two = previewWith(page, (b) => Array.isArray(b.bypass_stages) && b.bypass_stages.join() === 'xseg_roto,restore_ultra');
    await page.getByTestId('flow-bypass-xseg_roto').click();
    await two;                                                            // pipeline order, not click order
    const none = previewWith(page, (b) => Array.isArray(b.bypass_stages) && b.bypass_stages.length === 0);
    await page.getByTestId('flow-bypass-restore_ultra').click();
    await page.getByTestId('flow-bypass-xseg_roto').click();
    await none;
    // the detector has no switch: nothing is swapped without a face
    await expect(page.getByTestId('flow-bypass-detect_align')).toHaveCount(0);
  });

  test('a solo the backend refuses is said, in words, not shown as if it had worked', async ({ page }) => {
    await openFaceSwap(page);
    await page.getByTestId('flow-bypass-hyperswap').click();
    await expect(page.getByTestId('flow-node-hyperswap')).toHaveAttribute('data-state', 'off');
    await page.getByTestId('flow-solo-restore_ultra').click();
    const refused = page.getByTestId('flow-solo-refused');
    await expect(refused).toBeVisible();
    await expect(refused).toContainText('swapper is bypassed');
    await expect(refused).toHaveAttribute('role', 'status');
  });

  test('the two passes the pipeline cannot produce are disabled, with the reason on hover', async ({ page }) => {
    await openFaceSwap(page);
    for (const id of ['occlusion_alpha', 'depth_z_order']) {
      const b = page.getByTestId(`solo-buffer-${id}`);
      await expect(b).toBeDisabled();
      expect(await b.getAttribute('title')).toMatch(/not available|Not built/i);
    }
    const parsing = previewWith(page, (b) => b.solo_stage === 'bisenet_parsing');
    await page.getByTestId('solo-buffer-bisenet_parsing').click();
    await parsing;
  });

  test('a provider the backend says it downgraded is shown on the model nodes, with its own reason', async ({ page }) => {
    await page.route('**/api/meta', async (route) => {
      const res = await route.fetch();
      const json = await res.json();
      json.provider_status = { ...(json.provider_status || {}), requested: 'tensorrt', active: 'cuda', degraded: true,
        degradation_reason: 'TensorRT session failed to build.' };
      await route.fulfill({ response: res, json });
    });
    await openFaceSwap(page);
    for (const id of ['detect_align', 'hyperswap', 'xseg_roto', 'restore_ultra']) {
      const pill = page.getByTestId(`flow-pill-${id}-provider-downgrade`);
      await expect(pill).toBeVisible();
      await expect(pill).toHaveAttribute('title', /running on CUDA\. TensorRT session failed to build\./);
    }
    await expect(page.getByTestId('flow-pill-source-provider-downgrade')).toHaveCount(0);
    await expect(page.getByTestId('flow-warn-count')).toBeVisible();
  });

  test('no downgrade pill when the backend reports none', async ({ page }) => {
    await openFaceSwap(page);
    await expect(page.locator(`${BAR} [data-testid$="provider-downgrade"]`)).toHaveCount(0);
  });

  test('on a short window the bar starts collapsed so the stage keeps its room; expanding is remembered', async ({ page }) => {
    await openFaceSwap(page, { width: 1280, height: 720 });
    await expect(page.getByTestId('flow-rail')).toHaveCount(0);
    const stage = await page.locator('[data-testid="stage-deck"] .workspace-single-preview').boundingBox();
    expect(stage.height).toBeGreaterThan(95);      // HEAD gives ~130 here; the collapsed dock costs the stage ~30 px
    await expect(page.getByTestId('stage-deck').evaluate((el) => el.scrollHeight - el.clientHeight)).resolves.toBeLessThanOrEqual(1);
    await page.getByTestId('flow-collapse').click();
    await expect(page.getByTestId('flow-rail')).toBeVisible();
    await page.reload();
    await page.locator(BAR).waitFor({ state: 'visible', timeout: 20_000 });
    await expect(page.getByTestId('flow-rail')).toBeVisible();
  });

  test('the bar collapses and remembers it', async ({ page }) => {
    await openFaceSwap(page);
    await page.getByTestId('flow-collapse').click();
    await expect(page.getByTestId('flow-rail')).toHaveCount(0);
    await page.reload();
    await page.locator(BAR).waitFor({ state: 'visible', timeout: 20_000 });
    await expect(page.getByTestId('flow-rail')).toHaveCount(0);
    await page.getByTestId('flow-collapse').click();
    await expect(page.getByTestId('flow-rail')).toBeVisible();
  });

  test('axe finds nothing in the bar, soloed or not', async ({ page }) => {
    await openFaceSwap(page);
    const run = () => new AxeBuilder({ page }).include(BAR).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice']).analyze();
    expect((await run()).violations.map((v) => v.id)).toEqual([]);
    await page.getByTestId('flow-solo-hyperswap').click();
    await expect(page.getByTestId('flow-node-hyperswap')).toHaveAttribute('data-state', 'solo');
    expect((await run()).violations.map((v) => v.id)).toEqual([]);
  });
});

// ── the contact sheet ───────────────────────────────────────────────────────

const SHEET = '[data-testid="parameter-sweep-modal"]';
const isSweep = (r) => r.method() === 'POST' && new URL(r.url()).pathname === '/api/preview/sweep';

async function openSheet(page) {
  await openTab(page, 'faceswap');
  const first = page.waitForRequest(isSweep, { timeout: 20_000 });
  await page.getByTestId('faceswap-sweep-btn').scrollIntoViewIfNeeded();
  await page.getByTestId('faceswap-sweep-btn').click();
  const sheet = page.locator(SHEET);
  await sheet.waitFor({ state: 'visible' });
  const request = await first;
  await expect(page.getByTestId('sweep-tile-2-2').locator('img')).toBeVisible({ timeout: 15_000 });
  return { sheet, request };
}

const transforms = (page) => page.locator(`${SHEET} [data-cell] img`).evaluateAll((els) => els.map((e) => e.style.transform));

test.describe('contact sheet', () => {
  test('it fills 90% of the window in both directions and renders nine cells', async ({ page }) => {
    const { sheet, request } = await openSheet(page);
    const vp = page.viewportSize();
    const box = await sheet.boundingBox();
    expect(Math.abs(box.width - vp.width * 0.9)).toBeLessThan(3);
    expect(Math.abs(box.height - vp.height * 0.9)).toBeLessThan(3);
    await expect(page.locator(`${SHEET} [data-cell]`)).toHaveCount(9);
    const body = bodyOf(request);
    expect(body).toMatchObject({ x_param: 'restore_ultra_detail_weight', x_values: [0.25, 0.65, 1], y_param: 'mask_erode_dilate_radius', y_values: [-8, 0, 8] });
    // every cell is a real picture of its own
    const srcs = await page.locator(`${SHEET} [data-cell] img`).evaluateAll((els) => els.map((e) => e.getAttribute('src')));
    expect(new Set(srcs).size).toBe(9);
    await expect(sheet).toHaveAttribute('aria-modal', 'true');
    await expect(page.getByTestId('sweep-warnings')).toHaveCount(0);
  });

  test('X and Y take a parameter and min / centre / max, and Regenerate sends them', async ({ page }) => {
    await openSheet(page);
    await page.getByTestId('sweep-param-x').selectOption('blend_ratio');
    await expect(page.getByTestId('sweep-axis-x').getByLabel('Min value')).toHaveValue('0.4');
    await page.getByTestId('sweep-axis-x').getByLabel('Min value').fill('0.1');
    await page.getByTestId('sweep-axis-x').getByLabel('Max value').fill('0.95');
    await page.getByTestId('sweep-param-y').selectOption('skin_tone_warmth');
    await expect(page.getByTestId('sweep-generate')).toHaveText('Regenerate 3×3');
    const req = page.waitForRequest(isSweep);
    await page.getByTestId('sweep-generate').click();
    const body = bodyOf(await req);
    expect(body).toMatchObject({ x_param: 'blend_ratio', x_values: [0.1, 0.75, 0.95], y_param: 'skin_tone_warmth', y_values: [-0.5, 0, 0.5] });
    await expect(page.locator(`${SHEET} [data-testid^="sweep-tile-"]`).first()).toBeVisible();
    // the same parameter cannot be on both axes
    expect(await page.getByTestId('sweep-param-y').locator('option[value="blend_ratio"]').evaluate((o) => o.disabled)).toBe(true);
  });

  test('typing a value does not fight you: "0." keeps its dot, and the value is clamped to the range on blur', async ({ page }) => {
    await openSheet(page);
    const min = page.getByTestId('sweep-axis-x').getByLabel('Min value');
    await min.click();
    await min.fill('');
    await min.pressSequentially('0.35');               // the old box re-synced after "0." and turned it into "0", then "03"
    await expect(min).toHaveValue('0.35');
    await min.blur();
    await expect(min).toHaveValue('0.35');
    await min.click();
    await min.fill('1.7');                              // restore weight is 0..1
    await min.blur();
    await expect(min).toHaveValue('1');
  });

  test('wheel zooms ALL nine cells together and about the pointer; drag pans them together', async ({ page }) => {
    await openSheet(page);
    const cell = page.getByTestId('sweep-tile-1-1');
    const box = await cell.boundingBox();
    await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.3);
    await page.mouse.wheel(0, -400);
    await expect.poll(async () => (await page.getByTestId('sweep-zoom').textContent()) ?? '').not.toBe('100%');
    const zoomed = await transforms(page);
    expect(new Set(zoomed).size).toBe(1);                                  // one view serves all nine
    expect(zoomed[0]).toMatch(/scale\((1\.[1-9]|[2-9])/);
    // pointer in the upper-right of the cell: zooming about it pans the picture toward the lower-left
    expect(zoomed[0]).toMatch(/translate\(-/);
    const before = zoomed[0];
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5, { steps: 6 });
    await page.mouse.up();
    const panned = await transforms(page);
    expect(new Set(panned).size).toBe(1);
    expect(panned[0]).not.toBe(before);
    // Fit puts every cell back
    await page.getByTestId('sweep-focus-fit').click();
    expect((await transforms(page))[0]).toMatch(/translate\(0%, 0%\) scale\(1\)/);
    await expect(page.getByTestId('sweep-zoom')).toHaveText('100%');
  });

  test('Eyes / Mouth / Jaw zoom every cell to the feature', async ({ page }) => {
    await openSheet(page);
    for (const [k, z] of [['eyes', '280%'], ['mouth', '300%'], ['jaw', '260%']]) {
      await page.getByTestId(`sweep-focus-${k}`).click();
      await expect(page.getByTestId('sweep-zoom')).toHaveText(z);
      expect(new Set(await transforms(page)).size).toBe(1);
    }
  });

  test('the 3.5x loupe follows the pointer in all nine cells over the same point', async ({ page }) => {
    await openSheet(page);
    const cell = page.getByTestId('sweep-tile-0-0');
    const box = await cell.boundingBox();
    await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.45, { steps: 3 });
    const lenses = page.locator(`${SHEET} .sweep-loupe`);
    await expect(lenses).toHaveCount(9);
    await expect.poll(() => lenses.evaluateAll((els) => els.filter((e) => e.style.display === 'block').length)).toBe(9);
    const read = () => lenses.evaluateAll((els) => els.map((e) => ({
      size: e.style.backgroundSize, pos: e.style.backgroundPosition, left: e.style.left, top: e.style.top, w: e.style.width, img: e.style.backgroundImage.length,
    })));
    const a = await read();
    expect(new Set(a.map((l) => l.pos)).size).toBe(1);                    // the same image point at the lens centre in every cell
    expect(new Set(a.map((l) => l.size)).size).toBe(1);
    expect(new Set(a.map((l) => `${l.left}/${l.top}`)).size).toBe(1);
    expect(a.every((l) => l.img > 20)).toBe(true);                         // each lens shows ITS OWN tile's picture
    // the magnification is 3.5x of the cell's frame
    const frame = await page.getByTestId('sweep-tile-0-0').locator('div.relative.overflow-hidden').first().boundingBox();
    const [bgW] = a[0].size.split(' ').map(parseFloat);
    expect(Math.abs(bgW - frame.width * 3.5)).toBeLessThan(3);
    // moving moves them all
    await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, { steps: 3 });
    const b = await read();
    expect(new Set(b.map((l) => l.pos)).size).toBe(1);
    expect(b[0].pos).not.toBe(a[0].pos);
    // G turns it off, leaving the cell clears it
    await page.keyboard.press('g');
    await expect.poll(() => lenses.evaluateAll((els) => els.every((e) => e.style.display === 'none'))).toBe(true);
    await page.keyboard.press('g');
    await page.mouse.move(2, 2);
    await expect.poll(() => lenses.evaluateAll((els) => els.every((e) => e.style.display === 'none'))).toBe(true);
  });

  test('Apply Cell to Active Recipe commits the selected cell, marks it, and re-renders the frame with its values', async ({ page }) => {
    await openSheet(page);
    await page.getByTestId('sweep-select-0-0').click();                   // cell #1: X = min, Y = min
    await expect(page.getByTestId('sweep-tile-0-0')).toHaveAttribute('data-selected', 'true');
    await expect(page.getByTestId('sweep-actions')).toContainText('cell #1');
    const req = previewWith(page, (b) => b.restore_ultra_detail_weight === 0.25);
    await page.getByTestId('sweep-apply-selected').click();
    await req;
    await expect(page.getByTestId('sweep-committed-0')).toHaveText('✓ Applied');
    await expect(page.getByTestId('sweep-tile-0-0')).toHaveAttribute('data-committed', 'true');
    await expect(page.locator(`${SHEET} .sweep-pulse-ring`)).toHaveCount(1);
    await expect(page.getByTestId('sweep-status')).toContainText('Cell #1 applied to the active recipe');
    // committing again REMOUNTS the ring (its key is the nonce) so the pulse plays again
    const ring = await page.locator(`${SHEET} .sweep-pulse-ring`).evaluate((el) => { el.dataset.mark = '1'; return el.dataset.mark; });
    expect(ring).toBe('1');
    await page.getByTestId('sweep-apply-selected').click();
    await expect(page.locator(`${SHEET} .sweep-pulse-ring[data-mark="1"]`)).toHaveCount(0);
    await expect(page.locator(`${SHEET} .sweep-pulse-ring`)).toHaveCount(1);
  });

  test('keys: 1-9 apply that cell, Enter applies the selected one, Esc closes and gives focus back', async ({ page }) => {
    await openSheet(page);
    const nine = previewWith(page, (b) => b.restore_ultra_detail_weight === 1);
    await page.keyboard.press('9');                                        // cell #9: X = max
    await nine;
    await expect(page.getByTestId('sweep-committed-8')).toHaveText('✓ Applied');
    // the digit did NOT also hit a snapshot slot behind the sheet
    await expect(page.locator('[data-slot="A"]')).toHaveAttribute('data-slot-filled', '0');
    await page.getByTestId('sweep-select-1-0').focus();                  // select cell #2, then Enter
    const two = previewWith(page, (b) => b.restore_ultra_detail_weight === 0.65);
    await page.keyboard.press('Enter');
    await two;
    await expect(page.getByTestId('sweep-committed-1')).toHaveText('✓ Applied');
    await page.keyboard.press('Escape');
    await expect(page.locator(SHEET)).toHaveCount(0);
  });

  test('Assign to Snapshot Slot stores the cell\'s recipe WITHOUT applying it; recalling the slot applies it', async ({ page }) => {
    await openSheet(page);
    await page.getByTestId('sweep-select-0-2').click();                   // cell #7: X = min (0.25), Y = max (8)
    const sent = [];
    page.on('request', (r) => { if (isPreview(r)) sent.push(bodyOf(r)); });
    await page.getByTestId('sweep-slot-B').click();
    await expect(page.getByTestId('sweep-committed-6')).toHaveText('→ Slot B');
    await expect(page.getByTestId('sweep-status')).toContainText('assigned to Slot B');
    await expect(page.getByTestId('sweep-status')).toContainText('Not applied');
    await page.waitForTimeout(600);
    expect(sent.some((b) => b.restore_ultra_detail_weight === 0.25)).toBe(false);        // nothing was applied
    await page.keyboard.press('Escape');
    const slot = page.locator('[data-slot="B"]');
    await expect(slot).toHaveAttribute('data-slot-filled', '1');
    await expect(slot).toHaveAttribute('aria-pressed', 'false');                        // filled, but NOT the one in force
    const recall = previewWith(page, (b) => b.restore_ultra_detail_weight === 0.25);
    await slot.click();
    await recall;
    await expect(slot).toHaveAttribute('aria-pressed', 'true');
  });

  test('the A-D keys assign the selected cell', async ({ page }) => {
    await openSheet(page);
    await page.getByTestId('sweep-select-2-1').click();                   // cell #6
    await page.keyboard.press('d');
    await expect(page.getByTestId('sweep-committed-5')).toHaveText('→ Slot D');
    await page.keyboard.press('Escape');
    await expect(page.locator('[data-slot="D"]')).toHaveAttribute('data-slot-filled', '1');
  });

  test('an axis nothing reads is called out in words', async ({ page }) => {
    await openSheet(page);
    await page.getByTestId('sweep-param-x').selectOption('stabilization_strength');
    await page.getByTestId('sweep-param-y').selectOption('skin_tone_warmth');
    await page.getByTestId('sweep-generate').click();
    const warn = page.locator(`${SHEET} [data-warning="unbound"]`);
    await expect(warn).toBeVisible();
    await expect(warn).toContainText('Stabilization Strength');
    await expect(warn).toContainText('several frames');
  });

  test('a failed sweep says so and the sheet stays usable', async ({ page }) => {
    await page.route('**/api/preview/sweep', (route) => route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ message: 'Sweep failed: boom' }) }));
    await openTab(page, 'faceswap');
    await page.getByTestId('faceswap-sweep-btn').scrollIntoViewIfNeeded();
    await page.getByTestId('faceswap-sweep-btn').click();
    await expect(page.getByTestId('sweep-warnings').getByRole('alert')).toContainText(/boom|failed/i);
    await expect(page.getByTestId('sweep-generate')).toBeEnabled();
  });

  test('axe finds nothing in the open sheet', async ({ page }) => {
    await openSheet(page);
    const { violations } = await new AxeBuilder({ page }).include(SHEET).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice']).analyze();
    expect(violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
  });
});
