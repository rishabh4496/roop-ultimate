import { expect, test } from '@playwright/test';

// UI overhaul Stage 8: the mission-control screens, against stubbed backend answers.
//
// The mock server invents telemetry and has no projects, so each test stubs the one endpoint it is about and
// asserts on what the screen does with it. The pure logic (badges, speed, rack lanes, pills) is pinned by
// .render-check/render-ui-overhaul-stage8.mjs; these check that it reaches the DOM and the actions are wired.

const JOBS = [
  { id: 'j1', label: 'Clip A', target_name: 'clipA.mp4', source_name: 'Ann', state: 'PROCESSING',
    payload: { swap_model: 'hyperswap', enhancer: 'gpen' },
    progress: { fraction: 0.4, frames_done: 40, frames_total: 100, fps: 12.5, eta_s: 30, phase: 'Swapping' } },
  { id: 'j2', label: 'Clip B', target_name: 'clipB.mp4', source_name: 'Ben', state: 'QUEUED', payload: {}, progress: {} },
  { id: 'j3', label: 'Clip C', target_name: 'clipC.mp4', source_name: 'Cam', state: 'FAILED', payload: {}, progress: {}, error: 'no source faces' },
  { id: 'j4', label: 'Clip D', target_name: 'clipD.mp4', source_name: 'Dee', state: 'RECOVERABLE', payload: {}, progress: {}, project_id: 'p-d' },
  { id: 'j5', label: 'Clip E', target_name: 'clipE.mp4', source_name: 'Eve', state: 'COMPLETED', payload: {}, progress: { fraction: 1 } },
];

test('the queue mission control shows every state, filters, and switches to a table', async ({ page }) => {
  await page.route('**/api/queue', (route) => route.fulfill({ json: { jobs: JOBS, running: true, paused: false, current: 'j1' } }));
  await page.route('**/api/queue/segments', (route) => route.fulfill({ json: { groups: [] } }));
  await page.goto('/#/processing');

  const mission = page.getByTestId('queue-mission-control');
  await expect(mission).toBeVisible();
  await expect(mission.locator('[data-job-id]')).toHaveCount(5);
  await expect(mission.locator('span[data-state="PROCESSING"]')).toContainText('Running');
  await expect(mission.locator('span[data-state="RECOVERABLE"]')).toContainText('Recoverable');
  await expect(mission.getByTestId('speedometer').first()).toContainText('13');

  // A filter narrows the cards to one state.
  await mission.locator('[data-filter="FAILED"]').click();
  await expect(mission.locator('[data-job-id]')).toHaveCount(1);
  await expect(mission.locator('[data-job-id]')).toContainText('no source faces');
  await mission.locator('[data-filter="ALL"]').click();

  // The table view carries the same jobs, one row each, with the speed column.
  await mission.getByRole('button', { name: 'table', exact: true }).click();
  const table = mission.locator('[data-view="table"]');
  await expect(table).toBeVisible();
  await expect(table.locator('tbody tr')).toHaveCount(5);
  await expect(table.locator('tbody tr').first()).toContainText('13');
});

test('a checkpoint shows its badge; only a resumable one can be resumed in the workstation', async ({ page }) => {
  let loaded = null;
  await page.route('**/api/projects', (route) => route.fulfill({ json: { projects: [
    { id: 'p-green', name: 'Good render', state: 'PAUSED', recoverable: true, validation_errors: [],
      safe_frame: 120, next_frame: 121, segments: 1, updated_at: 1_790_000_000 },
    { id: 'p-amber', name: 'Moved clip', state: 'RECOVERABLE', recoverable: false,
      validation_errors: ['source 1 is missing: C:\\clips\\ann.mp4'], safe_frame: 40, next_frame: 41, segments: 0,
      updated_at: 1_790_000_000 },
  ] } }));
  await page.route('**/api/projects/p-green/load', (route) => {
    loaded = 'p-green';
    return route.fulfill({ json: { project: { id: 'p-green' }, state: {} } });
  });
  await page.goto('/#/history');

  const green = page.locator('[data-project-id="p-green"]');
  const amber = page.locator('[data-project-id="p-amber"]');
  await expect(green).toHaveAttribute('data-badge', 'green');
  await expect(amber).toHaveAttribute('data-badge', 'amber');
  await expect(amber.getByRole('button', { name: 'Relink source 1' })).toBeVisible();
  await expect(amber.getByRole('button', { name: 'Resume Exact State in Workstation' })).toBeDisabled();

  await green.getByRole('button', { name: 'Resume Exact State in Workstation' }).click();
  await expect.poll(() => loaded).toBe('p-green');
  await expect(page).toHaveURL(/#\/faceswap$/);
});

test('two to four outputs open the quad view with one tile per render and a shared playhead', async ({ page }) => {
  const files = ['alpha.mp4', 'beta.mp4', 'gamma.mp4'].map((name, i) => ({
    name, kind: 'video', url: `/outputs/${name}`, path: `/outputs/${name}`, absolute_path: `/outputs/${name}`,
    mtime: 1_790_000_000 - i, size: 1000 + i,
  }));
  await page.route('**/api/output', (route) => route.fulfill({ json: { output_path: '/outputs', files } }));
  await page.route('**/api/history', (route) => route.fulfill({ json: { entries: [
    { id: 1, time: 1_790_000_000, outputs: ['alpha.mp4'], fps: 9.8, duration_s: 125, frames: 1200, settings: { swap_model: 'hyperswap' } },
  ] } }));
  await page.goto('/#/gallery');

  // The quad button only exists once two or more renders are selected.
  await expect(page.getByTestId('quad-compare-button')).toHaveCount(0);
  await page.getByRole('button', { name: 'Select All', exact: true }).click();
  await expect(page.getByTestId('quad-compare-button')).toBeEnabled();
  await page.getByTestId('quad-compare-button').click();

  const dialog = page.getByTestId('quad-lightbox');
  await expect(dialog).toBeVisible();
  await expect(dialog.locator('[data-tile]')).toHaveCount(3);
  await expect(dialog.locator('[data-tile="0"] [data-pill="model"]')).toContainText('hyperswap');
  await expect(dialog.locator('[data-tile="0"] [data-pill="dur"]')).toContainText('2m 05s');
  await expect(dialog.locator('input[aria-label="Shared playhead"]')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
});

test('the launchpad shows model readiness and deep-links into a module', async ({ page }) => {
  await page.route('**/api/models/offline', (route) => route.fulfill({ json: { models: {
    'inswapper_128.onnx': { status: 'ok' },
    'gpen_bfr_512.onnx': { status: 'missing' },
  } } }));
  await page.goto('/#/home');

  const band = page.getByTestId('launchpad-band');
  await expect(band).toBeVisible();
  await expect(page.getByTestId('model-readiness')).toContainText('1 model needs attention');
  await expect(page.getByTestId('model-readiness')).toContainText('gpen_bfr_512.onnx');
  await band.locator('[data-deeplink="history"]').click();
  await expect(page).toHaveURL(/#\/history$/);
});
