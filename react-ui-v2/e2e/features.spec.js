// Features 2.0 gained on 11-28 that 1.0 had and 2.0 did not: the intended-use terms gate, renaming
// and removing a target person, Settings' Storage and System pages, reset-all, the output-codec and
// tristate dropdowns, and the render pre-flight check.
import { test, expect, openScreen, readyStudio } from './fixtures.js';

const TERMS = { text: 'Use it only on material you have the right to use.', version: 'v-test-1', acknowledged: false };

test('a fresh install shows the terms gate; nothing else is reachable until they are accepted', async ({ page, v2 }) => {
  // The engine refuses /api/swap (403 terms_required) until the terms are accepted, so without this
  // screen a fresh install could never render from 2.0.
  let accepted = false;
  await page.route((u) => u.pathname === '/api/terms', (r) => r.fulfill({ json: { ...TERMS, acknowledged: accepted } }));
  await page.route((u) => u.pathname === '/api/terms/acknowledge', (r) => { accepted = true; r.fulfill({ json: { acknowledged: true, version: TERMS.version } }); });
  await page.goto('#/studio', { waitUntil: 'domcontentloaded' });
  const gate = page.getByTestId('terms-gate');
  await expect(gate).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId('terms-text')).toContainText('right to use');
  await expect(page.getByTestId('terms-accept')).toBeDisabled();
  await expect(page.locator('[data-testid^="dock-"]')).toHaveCount(0);
  await page.getByTestId('terms-tick').check();
  await page.getByTestId('terms-accept').click();
  await expect(gate).toHaveCount(0);
  expect(v2.bodiesOf('/api/terms/acknowledge')).toEqual([{ version: TERMS.version, accept: true }]);
  await expect(page.getByTestId('dock-studio')).toBeVisible();
  // Accepted terms stay accepted on reload: no gate.
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.getByTestId('studio').waitFor({ timeout: 20_000 });
  await page.waitForTimeout(800);
  await expect(gate).toHaveCount(0);
  expect(v2.errors).toEqual([]);
});

test('a target person can be renamed and removed', async ({ page, v2 }) => {
  await readyStudio(page, v2);
  const people = page.getByTestId('person');
  const before = await people.count();
  expect(before).toBeGreaterThan(0);
  await people.first().getByTestId('person-rename').click();
  await page.getByTestId('person-name-input').fill('Bride');
  await page.getByTestId('person-name-input').press('Enter');
  await expect(people.first().getByTestId('person-name')).toHaveText('Bride');
  const named = v2.bodiesOf('/api/target/name').at(-1);
  expect(named.name).toBe('Bride');
  // Escape abandons an edit without a request.
  const sent = v2.bodiesOf('/api/target/name').length;
  await people.first().getByTestId('person-rename').click();
  await page.getByTestId('person-name-input').fill('Groom');
  await page.getByTestId('person-name-input').press('Escape');
  await expect(people.first().getByTestId('person-name')).toHaveText('Bride');
  expect(v2.bodiesOf('/api/target/name').length).toBe(sent);

  await people.first().getByTestId('person-remove').click();
  await expect(people).toHaveCount(before - 1);
  expect(v2.posted('/api/target/remove_face')).toBe(true);
  expect(v2.errors).toEqual([]);
});

test('person names follow the engine: target_names is by person rank, not by captured face', async ({ page, v2 }) => {
  // Engine shape (api._target_context_payload): three captured faces, two of them angles of p-ann.
  // target_names is _target_names_ranked(): one entry per PERSON. Read by face index, Bob's name
  // landed on nobody and the second angle of Ann read "Bob".
  await page.route((u) => u.pathname === '/api/state', async (route) => {
    const res = await route.fetch();
    const st = await res.json();
    const face = (st.target_faces && st.target_faces[0]) || null;
    await route.fulfill({ json: {
      ...st,
      target_faces: [face, face, face],
      target_groups: [0, 0, 1],
      target_person_ids: ['p-ann', 'p-ann', 'p-bob'],
      target_reference_face_ids: ['r1', 'r2', 'r3'],
      target_names: ['Ann', 'Bob'],
      target_faces_info: [{}, {}, {}],
      selected_target_person_id: 'p-ann',
    } });
  });
  await openScreen(page, 'studio');
  const names = page.getByTestId('person-name');
  await expect(names).toHaveText(['Ann', 'Bob']);
  // Ann has two captured angles; removing her removes both, highest face index first.
  await page.unroute((u) => u.pathname === '/api/state');
  await page.route((u) => u.pathname === '/api/target/remove_face', (r) => r.fulfill({ json: {} }));
  await page.getByTestId('person').first().getByTestId('person-remove').click();
  await expect.poll(() => v2.bodiesOf('/api/target/remove_face').map((b) => b.face_index)).toEqual([1, 0]);
  expect(v2.errors).toEqual([]);
});

test('settings: the output codec and the tristate flags are dropdowns of the values the engine takes', async ({ page, v2 }) => {
  await openScreen(page, 'settings');
  // `output_video_codec` was mapped as `video_codec` (no such setting), so it was a free-text box.
  const codec = page.locator('#set-output_video_codec');
  await expect(codec).toHaveJSProperty('tagName', 'SELECT');
  expect(await codec.locator('option').allTextContents()).toEqual(expect.arrayContaining(['libx264', 'h264_nvenc']));
  await expect(page.locator('#set-output_video_format')).toHaveJSProperty('tagName', 'SELECT');
  const nvdec = page.locator('#set-perf_nvdec');
  await expect(nvdec).toHaveJSProperty('tagName', 'SELECT');
  expect(await nvdec.locator('option').allTextContents()).toEqual(expect.arrayContaining(['auto', 'on', 'off']));
  expect(v2.errors).toEqual([]);
});

test('settings: reset all puts every changed setting back to the default, after a confirmation', async ({ page, v2 }) => {
  await openScreen(page, 'settings');
  const defaults = await page.evaluate(() => fetch('/api/settings/defaults').then((r) => r.json()));
  const changed = Math.max(1, Number(defaults.max_threads) || 10) + 3;
  await page.evaluate((n) => fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ max_threads: n }) }), changed);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.getByTestId('settings').waitFor();
  await page.waitForTimeout(800);
  let prompt = '';
  // v2.posts also records this test's own setup POST: count only what the click sends.
  const sent = v2.bodiesOf('/api/settings').length;
  page.once('dialog', (d) => { prompt = d.message(); d.accept(); });
  await page.getByTestId('settings-reset-all').click();
  await expect.poll(() => v2.bodiesOf('/api/settings').length, { timeout: 5000 }).toBeGreaterThan(sent);
  expect(prompt).toMatch(/Reset \d+ setting/);
  const patch = Object.assign({}, ...v2.bodiesOf('/api/settings').slice(sent));
  expect(patch.max_threads).toBe(defaults.max_threads);
  const now = await page.evaluate(() => fetch('/api/settings').then((r) => r.json()));
  expect(now.max_threads).toBe(defaults.max_threads);
  expect(v2.errors).toEqual([]);
});

test('settings → storage lists the engine review and deletes one safe item after a confirmation', async ({ page, v2 }) => {
  await openScreen(page, 'settings');
  await page.getByTestId('settings-tab-storage').click();
  const items = page.getByTestId('storage-item');
  await expect(items.first()).toBeVisible();
  const before = await items.count();
  const deletable = page.getByTestId('storage-delete');
  // Only SAFE_TO_DELETE items carry a delete button.
  expect(await deletable.count()).toBeLessThan(before);
  // Dismissing the confirmation deletes nothing.
  page.once('dialog', (d) => d.dismiss());
  await deletable.first().click();
  await page.waitForTimeout(300);
  expect(v2.posted('/api/storage/delete')).toBe(false);
  page.once('dialog', (d) => d.accept());
  await deletable.first().click();
  await expect(items).toHaveCount(before - 1);
  expect(v2.bodiesOf('/api/storage/delete')[0]).toMatchObject({ confirm: true });
  expect(v2.errors).toEqual([]);
});

test('settings → system shows the machine and the update status', async ({ page, v2 }) => {
  await openScreen(page, 'settings');
  await page.getByTestId('settings-tab-system').click();
  await expect(page.getByTestId('hardware')).toContainText(/RTX|GPU/);
  await expect(page.getByTestId('hardware')).toContainText('VRAM');
  await expect(page.getByTestId('update-status')).toHaveText(/Up to date|update is available/);
  await page.getByTestId('update-check').click();
  await expect(page.getByTestId('update-status')).toBeVisible();
  expect(v2.errors).toEqual([]);
});

test('the render deck runs the engine pre-flight, and a one-click fix sets the setting it names', async ({ page, v2 }) => {
  await page.route((u) => u.pathname === '/api/render/preflight', (r) => r.fulfill({ json: {
    status: 'warning',
    checks: {},
    warnings: [{ check: 'encoder', message: 'NVIDIA GPU detected but the CPU encoder libx264 is selected.' }],
    auto_fixes: [{ action: 'switch_nvenc', label: 'Switch to NVENC', codec: 'h264_nvenc' }],
  } }));
  await readyStudio(page, v2);
  await expect(page.getByTestId('preflight-encoder')).toBeVisible();
  const before = v2.bodiesOf('/api/render/preflight').at(-1);
  expect(before).toMatchObject({ batch_count: 1 });
  expect(before.max_frames).toBeGreaterThan(0);
  await page.getByTestId('preflight-fix-switch_nvenc').click();
  await expect.poll(() => Object.assign({}, ...v2.bodiesOf('/api/settings')).output_video_codec).toBe('h264_nvenc');
  // The pipeline diagram's output node reads the real setting now (it read `video_codec`, always "file").
  await expect(page.getByTestId('pipeline')).toContainText('h264_nvenc');
  // Start is advisory-only: never blocked by a warning.
  await expect(page.getByTestId('run-start')).toBeEnabled();
  await page.evaluate(() => fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ output_video_codec: 'libx264' }) }));
  expect(v2.errors).toEqual([]);
});
