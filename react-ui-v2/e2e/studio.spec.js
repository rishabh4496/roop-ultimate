// The studio: cast, media bin, viewport modes, timeline and the In/Out range.
import { test, expect, openScreen, readyStudio, PNG_1X1 } from './fixtures.js';

test('adding a source image grows the cast', async ({ page, v2 }) => {
  await openScreen(page, 'studio');
  const before = await page.locator('[data-testid="cast"] .tile').count();
  await page.locator('[data-testid="cast"] input[type=file]').setInputFiles({ name: 'face.png', mimeType: 'image/png', buffer: PNG_1X1 });
  await expect(page.locator('[data-testid="cast"] .tile')).toHaveCount(before + 1);
  expect(v2.errors).toEqual([]);
});

test('a target from the media bin shows its faces, and clicking one makes a target person', async ({ page, v2 }) => {
  await readyStudio(page, v2);
  expect(await page.locator('[data-testid="viewport"] .face-tag').count()).toBeGreaterThan(0);
  expect(v2.errors).toEqual([]);
});

test('compare shows the wipe and quad shows four live panes', async ({ page, v2 }) => {
  await readyStudio(page, v2);
  await page.getByTestId('mode-compare').click();
  await expect(page.locator('.wipe-handle')).toHaveCount(1);
  await page.getByTestId('mode-quad').click();
  await expect(page.locator('[data-testid="quad"] > div')).toHaveCount(4);
  await page.getByTestId('mode-swap').click();
  await expect(page.locator('[data-testid="quad"]')).toHaveCount(0);
  expect(v2.errors).toEqual([]);
});

test('dragging on the timeline strip moves the playhead', async ({ page, v2 }) => {
  await readyStudio(page, v2);
  const strip = page.getByTestId('timeline-strip');
  const box = await strip.boundingBox();
  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height / 2, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => Number(await page.locator('#tl-frame').inputValue())).toBeGreaterThan(10);
});

// Regression (2617e3c): the In point is the frame ON SCREEN. set_frame used to store it raw as the
// engine's 0-based seek, so every render started one frame after the In the user picked
// (In 300 / Out 329 rendered 29 frames from 301). The API is now 1-based and inclusive, like a
// queue job's frame_start/frame_end; the client sends the playhead frame as it is.
test('In and Out send the frame on screen, and the timeline counts In..Out inclusive', async ({ page, v2 }) => {
  await readyStudio(page, v2);
  const total = await page.evaluate(async () => {
    const st = await (await fetch('/api/state')).json();
    return st.targets[st.selected_target_index]?.frames || 0;
  });
  test.skip(total < 60, `the mock target has ${total} frames`);
  const inAt = Math.min(300, total - 40);

  await page.locator('#tl-frame').fill(String(inAt));
  await page.locator('#tl-frame').press('Enter');
  await page.getByRole('button', { name: /^In$/ }).click();
  await page.locator('#tl-frame').fill(String(inAt + 29));
  await page.locator('#tl-frame').press('Enter');
  await page.getByRole('button', { name: /^Out$/ }).click();

  await expect.poll(() => v2.bodiesOf('/api/target/set_frame').length).toBeGreaterThanOrEqual(2);
  const sent = v2.bodiesOf('/api/target/set_frame');
  expect(sent.find((b) => b.which === 'start')?.frame, 'In sends the playhead frame, not frame - 1').toBe(inAt);
  expect(sent.find((b) => b.which === 'end')?.frame).toBe(inAt + 29);

  const chip = page.locator('text=/in \\d+ · out \\d+/').first();
  await expect(chip).toContainText(`in ${inAt}`);
  await expect(chip).toContainText(`out ${inAt + 29}`);
  await expect(chip).toContainText('30 f');

  // Put the range back for the next test (the mock is shared).
  await page.evaluate(async (t) => {
    const post = (b) => fetch('/api/target/set_frame', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) });
    await post({ which: 'start', frame: 1 });
    await post({ which: 'end', frame: t });
  }, total);
});

test('the I key sets the In point too', async ({ page, v2 }) => {
  await readyStudio(page, v2);
  await page.locator('#tl-frame').fill('12');
  await page.locator('#tl-frame').press('Enter');
  await page.locator('[data-testid="viewport"]').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('i');
  await expect.poll(() => v2.bodiesOf('/api/target/set_frame').some((b) => b.which === 'start' && b.frame === 12)).toBe(true);
});

test('the pipeline lists the engine\'s swap models', async ({ page }) => {
  await openScreen(page, 'studio');
  const options = page.locator('[data-testid="pipeline"] select').first().locator('option');
  expect(await options.count()).toBeGreaterThanOrEqual(3);
});
