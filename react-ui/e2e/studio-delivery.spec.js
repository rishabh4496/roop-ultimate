import { test, expect } from '@playwright/test';
import { openTab, PNG_1X1 } from './helpers.js';

test.use({ viewport: { width: 1600, height: 1000 } });

test('Editor loads a shelf LUT and previews the current frame in both output modes', async ({ page }) => {
  await openTab(page, 'extras');
  await page.locator('input[type="file"]').first().setInputFiles({
    name: 'studio-frame.png', mimeType: 'image/png', buffer: PNG_1X1,
  });
  const grading = page.getByTestId('color-grading-panel');
  await grading.locator('input[type="file"]').setInputFiles({
    name: 'studio-grade.cube', mimeType: 'text/plain',
    buffer: Buffer.from('LUT_3D_SIZE 2\n' + '0 0 0\n'.repeat(8)),
  });
  await expect(grading.getByLabel('Saved LUTs')).toHaveValue('/mock/luts/studio-grade.cube');
  await expect(grading.getByTestId('lut-live-thumbnails').locator('canvas')).toHaveCount(2);
  const bake = grading.getByRole('button', { name: 'Bake LUT into Hardware Encode' });
  await bake.click();
  await expect(bake).toHaveAttribute('aria-pressed', 'true');
  const display = grading.getByRole('button', { name: 'Display-Only Preview LUT' });
  await display.click();
  await expect(display).toHaveAttribute('aria-pressed', 'true');
});

test('Face Swap displays the verified output hash and exposes standalone matte controls', async ({ page }) => {
  await openTab(page, 'faceswap');
  await page.getByRole('button', { name: 'Studio delivery & interchange' }).click();
  const mattes = page.getByTestId('vfx-matte-export');
  await mattes.getByText('Enable matte export', { exact: true }).click();
  await expect(mattes.getByRole('checkbox', { name: 'Export XSeg 3 occlusion matte' })).toBeChecked();
  await expect(mattes.getByRole('checkbox', { name: 'Export Face parsing matte' })).toBeChecked();
  await mattes.getByRole('checkbox', { name: 'Export Mask difference matte' }).check();

  const provenance = page.getByTestId('provenance-inspector');
  await provenance.getByLabel('Rendered output path').fill('/outputs/beauty.mp4');
  await provenance.getByRole('button', { name: 'Verify output' }).click();
  await expect(provenance.getByRole('status')).toContainText('Output hash and signature verified');
  await expect(provenance.getByRole('status')).toContainText('mock_payload_hash_1234');

  const interchange = page.getByTestId('nle-interchange');
  for (const format of ['edl', 'otio']) {
    const [download, request] = await Promise.all([
      page.waitForEvent('download'),
      page.waitForRequest((entry) => entry.url().endsWith(`/api/timeline/export_${format}`)),
      interchange.getByRole('button', { name: `Export ${format.toUpperCase()}` }).click(),
    ]);
    expect(download.suggestedFilename()).toMatch(new RegExp(`\\.${format}$`));
    const { document, target_media_id } = request.postDataJSON();
    expect(target_media_id).toBeTruthy();
    expect(document.timeline.frame_end).toBeGreaterThan(document.timeline.frame_start);
  }
});
