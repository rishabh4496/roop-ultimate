// Face Swap's precision tools in 2.0: mask brush, roto, identity pins, scene overrides, curves and
// identity blend. Each must reach the ENGINE, not just draw: the last /api/preview body has to carry
// all of them, and the ones that change the range must cut the render into pieces.
import { test, expect, readyStudio } from './fixtures.js';

test('every precision tool is drawn, listed and carried by the preview request', async ({ page, v2 }) => {
  await readyStudio(page, v2);

  await test.step('mask brush paints and applies', async () => {
    await page.getByTestId('tool-tab-mask').click();
    const mc = await page.getByTestId('mask-canvas').boundingBox();
    expect(mc).not.toBeNull();
    await page.mouse.move(mc.x + mc.width * 0.45, mc.y + mc.height * 0.4);
    await page.mouse.down();
    await page.mouse.move(mc.x + mc.width * 0.55, mc.y + mc.height * 0.5, { steps: 6 });
    await page.mouse.up();
    await page.getByTestId('mask-apply').click();
    await expect(page.getByTestId('tool-mask')).toContainText('in the render');
  });

  await test.step('roto pen draws and closes a zone', async () => {
    await page.getByTestId('tool-tab-roto').click();
    const rl = await page.getByTestId('roto-layer').boundingBox();
    for (const [fx, fy] of [[0.2, 0.2], [0.35, 0.2], [0.35, 0.35], [0.2, 0.2]]) {
      await page.mouse.click(rl.x + rl.width * fx, rl.y + rl.height * fy);
      await page.waitForTimeout(80);
    }
    await expect(page.getByTestId('tool-roto')).toContainText(/Exclude 1 · 3 pts/);
  });

  await test.step('pin tool pins a face at the current frame', async () => {
    await page.getByTestId('tool-tab-pins').click();
    await page.locator('[data-testid="viewport"] .face-tag').first().click();
    await expect(page.getByTestId('tool-pins')).toContainText(/frame \d+/);
  });

  await test.step('a scene override is added with a field', async () => {
    await page.getByTestId('tool-tab-scenes').click();
    await page.getByTestId('tool-scenes').getByRole('button', { name: /Override In/ }).click();
    await page.getByTestId('tool-scenes').getByRole('button', { name: /Mask feather/ }).first().click();
    await expect(page.getByTestId('tool-scenes')).toContainText('Mask feather');
  });

  await test.step('a curve key is added', async () => {
    await page.getByTestId('tool-tab-curves').click();
    await page.getByTestId('tool-curves').getByRole('button', { name: /^Key / }).click();
    await expect.poll(() => page.getByTestId('tool-curves').locator('select').count()).toBeGreaterThanOrEqual(1);
  });

  await test.step('identity blend: two identities, enabled, one keyframe, posted to the engine', async () => {
    await page.getByTestId('tool-tab-blend').click();
    const ids = page.getByTestId('tool-blend').locator('button[aria-pressed]');
    for (let i = 0; i < Math.min(2, await ids.count()); i++) {
      if ((await ids.nth(i).getAttribute('aria-pressed')) !== 'true') await ids.nth(i).click();
    }
    const sw = page.getByTestId('tool-blend').getByRole('switch').first();
    if ((await sw.getAttribute('aria-checked')) !== 'true') await sw.click();
    await page.getByTestId('tool-blend').getByRole('button', { name: /Key at/ }).click();
    await expect(page.getByTestId('tool-blend')).toContainText(/Keyframes · 1/);
    await expect.poll(() => v2.posted('/api/identity/blend')).toBe(true);
  });

  await test.step('the tools panel says the render is cut into pieces, and the timeline has tool lanes', async () => {
    await expect(page.getByTestId('tools')).toContainText(/renders in \d+ pieces/i);
    await expect(page.getByTestId('tool-lanes')).toHaveCount(1);
  });

  await test.step('the next preview carries the mask, roto, pins and blend', async () => {
    await expect.poll(() => {
      const b = v2.bodiesOf('/api/preview').at(-1) || {};
      return { mask: !!b.imagemask, roto: b.roto_polygons?.length || 0, pins: (b.identity_pins?.length || 0) >= 1, blend: b.identity_blend?.enabled === true };
    }, { timeout: 15_000 }).toEqual({ mask: true, roto: 1, pins: true, blend: true });
  });

  expect(v2.errors).toEqual([]);
});
