// Shared setup for React UI 2.0's browser tests.
//
// `test` here is Playwright's, extended with one automatic fixture per page:
// - the boot sequence is marked as seen and the WebGL horizon is off (`roop_v2_fx`), so a screen
//   is usable on load and no test waits on, or samples, an animation;
// - the three answers where 1.0's mock server does not match the engine are replaced by the
//   ENGINE's shape (see ENGINE_SHAPES below), so a test checks what 2.0 does with what the engine
//   really sends rather than with the mock's guess;
// - every console error, page error and HTTP >= 400, and every POST with its JSON body, is recorded
//   for the test to assert on (`v2.errors`, `v2.posts`).
import { test as base, expect } from '@playwright/test';

export { expect };

// 1x1 PNG: a valid image, so the happy path of an <img> or an upload runs.
export const PNG_1X1 = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==',
  'base64',
);
const PNG_DATA_URL = `data:image/png;base64,${PNG_1X1.toString('base64')}`;

// Where the mock and the engine disagree, the engine wins:
// - /api/faceset/library: the engine returns {entries: [{filename, name, path, size, mtime, faces,
//   thumb}], dir} (app/routes_faceset.py _faceset_library_payload); the mock returns {libraries}.
// - /api/models/integrity: the engine lists every manifest file as {file, family, required, status,
//   bytes_done, bytes_total} (app/routes_models.py -> model_integrity); the mock lists none.
// - /outputs/*: the engine serves the files; the mock serves nothing, so every output <img> 404s.
export const ENGINE_SHAPES = {
  library: {
    dir: 'C:/mock/facesets',
    entries: [
      { filename: 'ava.fsz', name: 'ava', path: 'C:/mock/facesets/ava.fsz', size: 48_213, mtime: 1_764_300_000, faces: 4, thumb: PNG_DATA_URL },
      { filename: 'rhythm.fsz', name: 'rhythm', path: 'C:/mock/facesets/rhythm.fsz', size: 91_550, mtime: 1_764_310_000, faces: 1, thumb: PNG_DATA_URL },
    ],
  },
  integrity: {
    phase: 'idle',
    message: 'All model files verified.',
    busy: false,
    files: [
      { file: 'inswapper_128.onnx', family: 'inswapper', required: true, status: 'ok', bytes_done: 554_253_681, bytes_total: 554_253_681 },
      { file: 'hyperswap_1a_256.onnx', family: 'hyperswap', required: false, status: 'ok', bytes_done: 403_000_000, bytes_total: 403_000_000 },
      { file: 'GFPGANv1.4.onnx', family: 'gfpgan', required: false, status: 'missing', bytes_done: 0, bytes_total: 340_000_000 },
    ],
    download_bytes_done: 0,
    download_bytes_total: 0,
    download_percent: null,
    started_at: null,
    finished_at: null,
  },
};

// Noise that says nothing about 2.0: the favicon, and the telemetry sockets reconnecting when a
// test closes its page.
const IGNORED_ERRORS = /favicon|ERR_CONNECTION_REFUSED|WebSocket|ws:\/\//i;

export const test = base.extend({
  v2: [async ({ page }, use) => {
    const errors = [];
    const posts = [];
    page.on('console', (m) => { if (m.type() === 'error' && !IGNORED_ERRORS.test(m.text())) errors.push(m.text()); });
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
    page.on('response', (r) => {
      const u = new URL(r.url());
      if (r.status() >= 400 && !IGNORED_ERRORS.test(u.pathname)) errors.push(`HTTP ${r.status()} ${r.request().method()} ${u.pathname}`);
    });
    page.on('request', (r) => {
      const u = new URL(r.url());
      if (r.method() !== 'POST' || !u.pathname.startsWith('/api/')) return;
      let body = null;
      try { body = JSON.parse(r.postData() || 'null'); } catch { /* multipart upload */ }
      posts.push({ path: u.pathname, body });
    });

    await page.addInitScript(() => {
      try {
        sessionStorage.setItem('roop_v2_booted', '1');
        localStorage.setItem('roop_v2_fx', JSON.stringify({ ambient: false }));
      } catch { /* storage blocked: the app still renders */ }
    });
    await page.route((u) => u.pathname === '/api/faceset/library', (route) => route.fulfill({ json: ENGINE_SHAPES.library }));
    await page.route((u) => u.pathname === '/api/models/integrity', (route) => route.fulfill({ json: ENGINE_SHAPES.integrity }));
    await page.route((u) => u.pathname.startsWith('/outputs/'), (route) => route.fulfill({ contentType: 'image/png', body: PNG_1X1 }));

    const v2 = {
      errors,
      posts,
      posted: (path) => posts.some((p) => p.path === path),
      bodiesOf: (path) => posts.filter((p) => p.path === path).map((p) => p.body),
    };
    await use(v2);
  }, { auto: true }],
});

// Every screen 2.0 has (App.jsx SCREENS): its hash route, which is also its dock button's
// `dock-<route>`, and the test id on its root. Only Face Manager's two differ.
export const SCREENS = [
  { route: 'studio', root: 'studio' },
  { route: 'batch', root: 'batch' },
  { route: 'library', root: 'library' },
  { route: 'faces', root: 'facemgr' },
  { route: 'models', root: 'models' },
  { route: 'editor', root: 'editor' },
  { route: 'outputs', root: 'outputs' },
  { route: 'history', root: 'history' },
  { route: 'settings', root: 'settings' },
];
const rootOf = (route) => SCREENS.find((s) => s.route === route)?.root || route;

export async function openScreen(page, route = 'studio') {
  await page.goto(`#/${route}`, { waitUntil: 'domcontentloaded' });
  await page.getByTestId(rootOf(route)).waitFor({ timeout: 20_000 });
  // Let the screen's first fetches land so the first read is the loaded view, not its skeleton.
  // Not `networkidle`: the telemetry socket and the progress poll never go quiet.
  await page.waitForTimeout(1200);
}

// The studio with everything Start needs: a source in the cast, a target selected, and a person
// picked from the viewport. The mock is shared by every test, so each step only acts when the
// state is not already there.
export async function readyStudio(page, v2) {
  await openScreen(page, 'studio');
  if ((await page.locator('[data-testid="cast"] .tile').count()) === 0) {
    await page.locator('[data-testid="cast"] input[type=file]').setInputFiles({ name: 'face.png', mimeType: 'image/png', buffer: PNG_1X1 });
    await expect(page.locator('[data-testid="cast"] .tile').first()).toBeVisible();
  }
  await expect(page.locator('[data-testid="targets"] .tile').first()).toBeVisible();
  await page.locator('[data-testid="targets"] .tile button').first().click();
  const tag = page.locator('[data-testid="viewport"] .face-tag').first();
  await expect(tag, 'the swap preview shows a face').toBeVisible({ timeout: 15_000 });
  if ((await tag.textContent()).includes('+ pick')) {
    await tag.click();
    await expect.poll(() => v2.posted('/api/target/use_face')).toBe(true);
  }
  await expect(page.locator('[data-testid="people"] select').first(), 'a target person exists').toBeVisible();
}

// Leave the shared mock idle for the next test: no render, no queue run.
export async function stopEverything(page) {
  await page.evaluate(async () => {
    const post = (p) => fetch(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});
    await post('/api/queue/stop');
    await post('/api/stop');
  }).catch(() => {});
}
