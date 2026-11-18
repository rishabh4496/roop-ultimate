import { expect, test } from '@playwright/test';

// UI overhaul Stage 9: the global palette, the `?` cheat-sheet's keyboard view, and cinema mode on Tab.
// The mapping and ranking are pinned by .render-check/render-ui-overhaul-stage9.mjs; these check the wiring in a
// real page.

test('Ctrl+K opens the command palette from anywhere, filters it, and Escape closes it', async ({ page }) => {
  await page.goto('/#/home');
  await page.keyboard.press('Control+k');
  const search = page.getByPlaceholder('Search actions, themes, settings…');
  await expect(search).toBeFocused();
  await search.fill('thread benchmark');
  await expect(page.getByRole('button', { name: /Run the thread benchmark/ })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(search).toHaveCount(0);
});

test('the palette reaches the viewport modes and the Diff heatmap', async ({ page }) => {
  await page.goto('/#/faceswap');
  await page.keyboard.press('Control+k');
  await page.getByPlaceholder('Search actions, themes, settings…').fill('heatmap');
  await expect(page.getByRole('button', { name: /Compare: Diff heatmap/ })).toBeVisible();
  await page.keyboard.press('Escape');
});

test('? opens the cheat-sheet, and its keyboard view lights the keys the workstation uses', async ({ page }) => {
  await page.goto('/#/faceswap');
  await page.keyboard.press('?');
  const dialog = page.getByRole('dialog', { name: 'Keyboard shortcuts' });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: 'Keyboard', exact: true }).click();
  const keyboard = dialog.getByTestId('virtual-keyboard');
  await expect(keyboard).toBeVisible();
  await expect(keyboard.locator('[data-key="C"]')).toHaveAttribute('data-lit', 'yes');
  await expect(keyboard.locator('[data-key="Tab"]')).toHaveAttribute('data-lit', 'yes');
  await expect(keyboard.locator('[data-key="Space"]')).toHaveAttribute('data-lit', 'yes');
  await expect(keyboard.locator('[data-key="E"]')).toHaveAttribute('data-lit', 'yes');
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
});

test('Tab toggles cinema mode from the focused preview, and returns to the previous layout', async ({ page }) => {
  await page.goto('/#/faceswap');
  const standard = page.getByRole('button', { name: 'Standard NLE', exact: true });
  const cinema = page.getByRole('button', { name: 'Cinema Preview', exact: true });
  await expect(standard).toHaveAttribute('aria-pressed', 'true');
  await page.locator('[data-cinema-zone]').first().focus();
  await page.keyboard.press('Tab');
  await expect(cinema).toHaveAttribute('aria-pressed', 'true');
  await page.locator('[data-cinema-zone]').first().focus();
  await page.keyboard.press('Tab');
  await expect(standard).toHaveAttribute('aria-pressed', 'true');
});

test('Tab from the page still moves focus and does not toggle cinema', async ({ page }) => {
  await page.goto('/#/faceswap');
  const standard = page.getByRole('button', { name: 'Standard NLE', exact: true });
  await expect(standard).toHaveAttribute('aria-pressed', 'true');
  await page.evaluate(() => document.activeElement?.blur());
  await page.keyboard.press('Tab');
  await expect(standard).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: 'Cinema Preview', exact: true })).toHaveAttribute('aria-pressed', 'false');
});

test('Tab still moves focus when a control has it', async ({ page }) => {
  await page.goto('/#/faceswap');
  const before = page.getByRole('button', { name: 'Cinema Preview', exact: true });
  await before.focus();
  await page.keyboard.press('Tab');
  await expect(before).not.toBeFocused();
  await expect(before).toHaveAttribute('aria-pressed', 'false');
});
