import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab, PNG_1X1 } from './helpers.js';

// Stage 2 of the workstation overhaul, in a real browser against the mock backend:
//   * the studio slider gestures (double-click reset, Shift+drag, typed values);
//   * the Face Swap inspector (recipes, padding link, region pills, enhancer stack, temporal modes);
//   * Home (quick launch, resumable checkpoints), the Editor's .fsz extractor, the Settings tabs.
//
// The mock is ONE process shared by every spec, and these tests move settings, so
// each one puts the settings it found back afterwards.
const AXE = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];

let original;
test.beforeAll(async ({ request }) => {
  original = await (await request.get('/api/settings')).json();
});
test.afterEach(async ({ request }) => {
  await request.post('/api/settings', { data: original });
});

const range = (page, label) => page.locator('label', { hasText: label }).locator('input[type="range"]').first();
const posted = (page, key) => page.waitForRequest((r) => r.method() === 'POST' && /\/api\/settings$/.test(r.url())
  && key in (r.postDataJSON() || {}));

// ── sliders ────────────────────────────────────────────────────────────────
test.describe('slider gestures', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 1000 });
    await openTab(page, 'faceswap');
  });

  test('double-click resets to the factory default, Backspace does the same from the keyboard', async ({ page }) => {
    const s = range(page, 'Face detection threshold');           // default 0.5
    await s.fill('0.7');
    await expect(s).toHaveValue('0.7');
    await s.dblclick();
    await expect(s).toHaveValue('0.5');

    await s.fill('0.3');
    await s.focus();
    await page.keyboard.press('Backspace');
    await expect(s).toHaveValue('0.5');
  });

  test('Shift+Arrow is a fine 0.01 step; a plain Arrow is the slider\'s own step', async ({ page }) => {
    const s = range(page, 'Face detection threshold');
    await s.fill('0.5');
    await s.focus();
    await page.keyboard.press('ArrowRight');
    await expect(s).toHaveValue('0.55');
    await page.keyboard.press('Shift+ArrowLeft');
    // The DOM range snaps to its own 0.05 grid for display; the committed value is exact.
    const req = await posted(page, 'face_detector_threshold');
    expect(Math.abs(req.postDataJSON().face_detector_threshold - 0.54)).toBeLessThan(1e-6);
  });

  test('Shift+drag moves the value a tenth as fast as a plain drag', async ({ page }) => {
    const s = range(page, 'Face detection threshold');
    await s.fill('0.5');
    const box = await s.boundingBox();
    const y = box.y + box.height / 2;
    const x0 = box.x + box.width * 0.5;
    const read = async () => Number(await s.inputValue());

    await page.mouse.move(x0, y);
    await page.keyboard.down('Shift');
    await page.mouse.down();
    await page.mouse.move(x0 + 100, y, { steps: 8 });
    await page.mouse.up();
    await page.keyboard.up('Shift');
    const fine = Math.abs((await read()) - 0.5);

    await s.fill('0.5');
    await page.mouse.move(x0, y);
    await page.mouse.down();
    await page.mouse.move(x0 + 100, y, { steps: 8 });
    await page.mouse.up();
    const coarse = Math.abs((await read()) - 0.5);

    expect(coarse, 'a plain drag of 100px moves the value a lot').toBeGreaterThan(0.1);
    expect(fine, 'the same Shift+drag moves it far less').toBeLessThan(coarse / 3);
  });

  test('click the number to type an exact value; Enter commits, Escape cancels; Enter on the slider opens it', async ({ page }) => {
    const s = range(page, 'Face detection threshold');
    const label = page.locator('label', { hasText: 'Face detection threshold' });
    await s.fill('0.5');

    await label.locator('button[title="Click to type a value"]').click();
    const box = page.getByRole('textbox', { name: 'Face detection threshold value' });
    await expect(box).toBeFocused();
    await box.fill('0.62');
    const sent = posted(page, 'face_detector_threshold');
    await page.keyboard.press('Enter');
    expect((await sent).postDataJSON().face_detector_threshold).toBe(0.62);
    await expect(label.locator('button[title="Click to type a value"]')).toHaveText('0.62');

    await s.focus();
    await page.keyboard.press('Enter');                          // the keyboard way in
    await expect(box).toBeFocused();
    await box.fill('0.9');
    await page.keyboard.press('Escape');
    await expect(box).toHaveCount(0);
    await expect(label.locator('button[title="Click to type a value"]')).toHaveText('0.62');

    await label.locator('button[title="Click to type a value"]').click();
    await page.getByRole('textbox', { name: 'Face detection threshold value' }).fill('7');   // beyond the max
    await page.keyboard.press('Enter');
    await expect(label.locator('button[title="Click to type a value"]')).toHaveText('0.90');
  });
});

// ── the Face Swap inspector ────────────────────────────────────────────────
test.describe('inspector', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1600, height: 1000 });
    await openTab(page, 'faceswap');
  });

  test('quick recipes apply real settings and light up when the settings match', async ({ page }) => {
    const bar = page.getByRole('region', { name: 'Quick recipes' });
    const fast = bar.getByRole('button', { name: 'Fast Preview' });
    const sent = posted(page, 'selected_enhancer');
    await fast.click();
    await expect(fast).toHaveAttribute('aria-pressed', 'true');
    const body = (await sent).postDataJSON();
    expect(body.selected_enhancer).toBe('None');
    expect(body.face_detector_size).toBe('320');
    expect(body.subsample_upscale).toBe('128px');

    await bar.getByRole('button', { name: 'Natural Studio' }).click();
    await expect(bar.getByRole('button', { name: 'Natural Studio' })).toHaveAttribute('aria-pressed', 'true');
    await expect(fast).toHaveAttribute('aria-pressed', 'false');

    // Changing one value of the applied recipe leaves no recipe lit.
    await range(page, 'Original/Enhanced blend').fill('0.3');         // not in the recipe: still lit
    await expect(bar.getByRole('button', { name: 'Natural Studio' })).toHaveAttribute('aria-pressed', 'true');
  });

  test('custom presets: save the current settings under a name, apply, delete', async ({ page }) => {
    const bar = page.getByRole('region', { name: 'Quick recipes' });
    await bar.getByRole('button', { name: 'Save custom preset' }).click();
    await page.getByRole('textbox', { name: 'Custom preset name' }).fill('Stage2 probe');
    await page.keyboard.press('Enter');
    const mine = bar.getByRole('group', { name: 'Your presets' });
    await expect(mine.getByRole('button', { name: 'Stage2 probe', exact: true })).toBeVisible();
    await mine.getByRole('button', { name: 'Delete preset Stage2 probe' }).click();
    await expect(mine).toHaveCount(0);
  });

  test('Detection keeps Face selection, Detector engine and Confidence up front; the rest is behind Advanced Detector Tuning', async ({ page }) => {
    const inspector = page.getByTestId('inspector');
    await expect(inspector.getByText('Face selection', { exact: true })).toBeVisible();
    await expect(inspector.getByText('Detector engine', { exact: true })).toBeVisible();
    await expect(inspector.getByText('Face detection threshold', { exact: true })).toBeVisible();
    await expect(inspector.getByText('Overlap NMS threshold')).toHaveCount(0);

    await inspector.getByRole('button', { name: 'Advanced Detector Tuning' }).click();
    await expect(inspector.getByText('Overlap NMS threshold')).toBeVisible();
    await expect(inspector.getByText('Face detection resolution')).toBeVisible();
  });

  test('mask padding is one slider with a link button that splits it into four', async ({ page }) => {
    const inspector = page.getByTestId('inspector');
    await inspector.getByRole('button', { name: /^Masking/ }).click();
    const link = inspector.getByRole('button', { name: /^(Unlink|Link) top, bottom, left and right padding$/ });
    await expect(link).toHaveAttribute('aria-pressed', 'true');                  // 0/0/0/0 starts linked
    await expect(range(page, 'Padding (all sides)')).toBeVisible();
    await expect(inspector.getByText('Padding top', { exact: true })).toHaveCount(0);

    const sent = posted(page, 'mask_top');
    await range(page, 'Padding (all sides)').fill('0.4');
    const body = (await sent).postDataJSON();
    expect([body.mask_top, body.mask_bottom, body.mask_left, body.mask_right]).toEqual([0.4, 0.4, 0.4, 0.4]);

    await link.click();
    await expect(link).toHaveAttribute('aria-pressed', 'false');
    for (const side of ['top', 'bottom', 'left', 'right']) {
      await expect(inspector.getByText(`Padding ${side}`, { exact: true })).toBeVisible();
    }
    const one = posted(page, 'mask_left');
    await range(page, 'Padding left').fill('0.9');
    const b2 = (await one).postDataJSON();
    expect([b2.mask_left, b2.mask_top]).toEqual([0.9, 0.4]);                      // only the one side moved
  });

  test('Face Parser region pills: toggle, Select all, Occlusion only; the last region cannot be removed', async ({ page }) => {
    const inspector = page.getByTestId('inspector');
    await inspector.getByRole('button', { name: /^Masking/ }).click();
    await expect(inspector.getByRole('group', { name: 'Regions to swap' })).toHaveCount(0);   // RealityUX engine: no pills
    await inspector.locator('label', { hasText: 'Masking engine' }).first().locator('select').selectOption('Face Parser (BiSeNet)');

    const pills = inspector.getByRole('group', { name: 'Regions to swap' });
    await expect(pills.getByRole('button')).toHaveCount(11);
    await expect(pills.getByRole('button', { name: 'Skin' })).toHaveAttribute('aria-pressed', 'true');
    await expect(pills.getByRole('button', { name: 'Hair' })).toHaveAttribute('aria-pressed', 'false');

    await pills.getByRole('button', { name: 'Hair' }).click();
    await expect(pills.getByRole('button', { name: 'Hair' })).toHaveAttribute('aria-pressed', 'true');

    await inspector.getByRole('button', { name: 'Select all' }).click();
    for (const b of await pills.getByRole('button').all()) await expect(b).toHaveAttribute('aria-pressed', 'true');

    const sent = posted(page, 'parser_regions');
    await inspector.getByRole('button', { name: 'Occlusion only' }).click();
    expect((await sent).postDataJSON().parser_regions).toEqual(['skin', 'brows', 'eyes', 'nose', 'mouth']);
    await expect(pills.getByRole('button', { name: 'Glasses' })).toHaveAttribute('aria-pressed', 'false');
  });

  test('the enhancer stack: four stages, switches that remember, a locked core, double-click reset on every slider', async ({ page }) => {
    const inspector = page.getByTestId('inspector');
    const stack = inspector.getByRole('list', { name: 'Enhancer pipeline' });
    await expect(stack.getByRole('listitem').filter({ has: page.getByRole('switch') })).toHaveCount(4);
    for (const t of ['Swapper Core', 'Face Restorer', 'Texture & Grain Closer', 'LAB Color & Skin-Tone Match']) {
      await expect(stack.getByRole('switch', { name: `${t} enabled` })).toBeVisible();
    }
    await expect(stack.getByRole('switch', { name: 'Swapper Core enabled' }), 'the core cannot be turned off').toBeDisabled();

    // Restorer: off -> None, on -> the one it replaced (not a default).
    const restorer = stack.getByRole('switch', { name: 'Face Restorer enabled' });
    await inspector.locator('label', { hasText: 'Restorer model' }).locator('select').selectOption('GPEN');
    await expect(restorer).toHaveAttribute('aria-checked', 'true');
    const off = posted(page, 'selected_enhancer');
    await restorer.click();
    expect((await off).postDataJSON().selected_enhancer).toBe('None');
    await expect(restorer).toHaveAttribute('aria-checked', 'false');
    const on = posted(page, 'selected_enhancer');
    await restorer.click();
    expect((await on).postDataJSON().selected_enhancer).toBe('GPEN');

    // Grain: off -> 0, on -> what it was.
    const grain = stack.getByRole('switch', { name: 'Texture & Grain Closer enabled' });
    await range(page, 'Grain match').fill('0.8');
    const z = posted(page, 'merger_grain_match');
    await grain.click();
    expect((await z).postDataJSON().merger_grain_match).toBe(0);
    const back = posted(page, 'merger_grain_match');
    await grain.click();
    expect((await back).postDataJSON().merger_grain_match).toBe(0.8);

    // Every slider in the stack resets on double-click.
    await range(page, 'Original/Enhanced blend').fill('0.2');
    await range(page, 'Original/Enhanced blend').dblclick();
    await expect(range(page, 'Original/Enhanced blend')).toHaveValue('1');
    await range(page, 'Histogram match').fill('0.9');
    await range(page, 'Histogram match').dblclick();
    await expect(range(page, 'Histogram match')).toHaveValue('0.4');
    await range(page, 'Swapping steps').fill('4');
    await range(page, 'Swapping steps').dblclick();
    await expect(range(page, 'Swapping steps')).toHaveValue('1');
  });

  test('temporal stability: four modes, read back from the settings, Custom when a parameter is edited', async ({ page }) => {
    const inspector = page.getByTestId('inspector');
    await inspector.getByRole('button', { name: /^Temporal/ }).click();
    const modes = inspector.getByRole('group', { name: 'Stability mode' });
    const m = (n) => modes.getByRole('button', { name: n });

    // (That the factory settings ARE Cinema is checked in render-workstation-logic.mjs; the mock's own settings are not the factory ones.)
    await m('Cinema Cut-Aware').click();
    await expect(m('Cinema Cut-Aware'), 'a mode reads back as itself').toHaveAttribute('aria-pressed', 'true');

    const sent = posted(page, 'stabilize_method');
    await m('Balanced EMA').click();
    const body = (await sent).postDataJSON();
    expect(body.stabilize_method).toBe('ema');
    expect(body.stabilize_hf_texture).toBe(false);
    await expect(m('Balanced EMA')).toHaveAttribute('aria-pressed', 'true');
    await expect(inspector.getByRole('button', { name: /^Temporal/ })).toContainText('Balanced');

    await m('Smart-Skip ROI').click();
    await expect(m('Smart-Skip ROI')).toHaveAttribute('aria-pressed', 'true');
    await m('Off').click();
    await expect(m('Off')).toHaveAttribute('aria-pressed', 'true');

    // The drawer holds the individual switches; flipping one moves the control to Custom.
    await m('Cinema Cut-Aware').click();
    await inspector.getByRole('button', { name: 'Custom parameters' }).click();
    await inspector.getByText('Reduce mask flicker', { exact: true }).click();
    await expect(inspector.getByText('Custom', { exact: true }).first()).toBeVisible();
    for (const b of await modes.getByRole('button').all()) await expect(b).toHaveAttribute('aria-pressed', 'false');
  });

  test('nothing the old column offered was lost: the controls moved into groups', async ({ page }) => {
    const inspector = page.getByTestId('inspector');
    for (const g of [/^Masking/, /^Temporal/, /^Eyes, mouth/, /^Pose, angle/]) {
      await inspector.getByRole('button', { name: g }).click();
    }
    await inspector.getByRole('button', { name: 'Custom parameters' }).click();         // the stabiliser switches live here
    for (const must of [
      'Masking engine', 'Second masking engine (occlusion)', 'Face mask edge blend', 'Mask edge falloff',
      'Boundary illumination match', 'Mask erode / dilate radius', 'Protect eyeglass frames',
      'Show mask overlay in preview', 'Temporal detection (anti-flicker)', 'Stabilize face (video)',
      'Reduce enhancer flicker', 'Flow-warped texture carry (pore boiling)', 'Restore original eyes',
      'Expression transfer strength', '3D source pose matching', 'Reshape jaw/chin to source', 'VR mode',
    ]) {
      await expect(inspector.getByText(must, { exact: false }).first(), must).toBeAttached();
    }
    // The empty select was a real bug: array options were not read by Select.
    await expect(inspector.locator('label', { hasText: 'Mask edge falloff' }).locator('select')).toHaveValue('gaussian');
  });

  test('axe: the inspector with every group open', async ({ page }) => {
    const inspector = page.getByTestId('inspector');
    for (const g of [/^Masking/, /^Temporal/, /^Eyes, mouth/, /^Pose, angle/]) {
      await inspector.getByRole('button', { name: g }).click();
    }
    await inspector.getByRole('button', { name: 'Advanced Detector Tuning' }).click();
    await inspector.getByRole('button', { name: 'Custom parameters' }).click();
    const { violations } = await new AxeBuilder({ page }).include('[data-testid="inspector"]').withTags(AXE).analyze();
    expect(violations.map((v) => ({ rule: v.id, targets: v.nodes.slice(0, 3).map((n) => n.target.join(' ')) }))).toEqual([]);
  });
});

// ── Home ───────────────────────────────────────────────────────────────────
test.describe('home', () => {
  test('quick launch: a source and a target load into the workspace and open Face Swap', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await openTab(page, 'home');
    const adds = [];
    page.on('request', (r) => { if (r.method() === 'POST' && /\/api\/(source|target)\/add$/.test(r.url())) adds.push(new URL(r.url()).pathname); });

    const inputs = page.locator('input[type="file"]');
    await expect(inputs).toHaveCount(2);
    await inputs.nth(0).setInputFiles({ name: 'face.png', mimeType: 'image/png', buffer: PNG_1X1 });
    await expect(page.getByText('face.png')).toBeVisible();
    await expect(page).toHaveURL(/#\/home$/);                          // one slot filled: stay
    await inputs.nth(1).setInputFiles({ name: 'clip.png', mimeType: 'image/png', buffer: PNG_1X1 });
    await expect(page).toHaveURL(/#\/faceswap$/, { timeout: 10_000 });
    expect(adds).toEqual(['/api/source/add', '/api/target/add']);
  });

  test('resumable checkpoints: only valid, resumable projects are offered, and Resume asks first', async ({ page }) => {
    await page.route('**/api/projects', (route) => route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        projects: [
          { id: 'p-ok', name: 'interview.mp4', state: 'INTERRUPTED', recoverable: true, safe_frame: 840, segments: 3, updated_at: Date.now() / 1000 - 600 },
          { id: 'p-blocked', name: 'blocked.mp4', state: 'INTERRUPTED', recoverable: false, safe_frame: 10, segments: 0, updated_at: Date.now() / 1000 - 60 },
          { id: 'p-done', name: 'done.mp4', state: 'COMPLETED', recoverable: true, safe_frame: 900, segments: 9, updated_at: Date.now() / 1000 },
        ],
      }),
    }));
    let resumed = null;
    await page.route('**/api/projects/*/resume', (route) => { resumed = route.request().url(); return route.fulfill({ contentType: 'application/json', body: '{"ok":true}' }); });

    await page.setViewportSize({ width: 1440, height: 900 });
    await openTab(page, 'home');
    const strip = page.getByTestId('resume-strip');
    await expect(strip.getByText('interview.mp4')).toBeVisible();
    await expect(strip.getByText('blocked.mp4'), 'a project the backend refused is never offered').toHaveCount(0);
    await expect(strip.getByText('done.mp4'), 'a completed project is not resumable').toHaveCount(0);
    await expect(strip.getByText(/frame 840 · 3 kept segments/)).toBeVisible();

    await strip.getByRole('button', { name: 'Resume' }).click();
    await page.getByRole('button', { name: 'Resume project' }).click();
    await expect.poll(() => resumed).toMatch(/\/api\/projects\/p-ok\/resume$/);
  });

  test('with no history the page is the launch pad only: no filler cards, no GPU tile', async ({ page }) => {
    await page.route('**/api/history*', (r) => r.fulfill({ contentType: 'application/json', body: '{"entries":[]}' }));
    await page.route('**/api/output*', (r) => r.fulfill({ contentType: 'application/json', body: '{"files":[],"output_path":""}' }));
    await page.route('**/api/queue', (r) => r.fulfill({ contentType: 'application/json', body: '{"jobs":[]}' }));
    await page.route('**/api/projects', (r) => r.fulfill({ contentType: 'application/json', body: '{"projects":[]}' }));
    await page.setViewportSize({ width: 1440, height: 900 });
    await openTab(page, 'home');
    await expect(page.getByRole('heading', { name: 'Quick launch' })).toBeVisible();
    for (const gone of ['Recent runs', 'Latest outputs', 'Resume where you stopped']) {
      await expect(page.getByRole('heading', { name: gone })).toHaveCount(0);
    }
    await expect(page.getByText('Finished runs are recorded here')).toHaveCount(0);
    await expect(page.getByText('GPU', { exact: true })).toHaveCount(0);
  });
});

// ── Editor ─────────────────────────────────────────────────────────────────
test.describe('editor', () => {
  test('video to faceset: load a clip, scrub, cut faces, build the .fsz', async ({ page, request }) => {
    await request.post('/api/facemgr/clear', { data: {} });
    await page.setViewportSize({ width: 1440, height: 900 });
    await openTab(page, 'extras');
    const card = page.getByTestId('video-faceset');
    await card.locator('input[type="file"]').setInputFiles({ name: 'clip.mp4', mimeType: 'video/mp4', buffer: Buffer.from('not really a video') });
    await expect(card.getByText(/clip\.mp4 · 240 frames/)).toBeVisible();
    await expect(card.getByRole('img', { name: /Frame 1 of clip\.mp4/ })).toBeVisible();

    await card.locator('label', { hasText: 'Frame' }).locator('input[type="range"]').fill('120');
    await card.getByRole('button', { name: 'Cut faces from this frame' }).click();
    await expect(card.getByText('1 face so far, 1 cut')).toBeVisible();
    await card.getByRole('button', { name: 'Build .fsz' }).click();
    await expect(card.getByRole('link', { name: /Download faceset\.fsz/ })).toBeVisible();
    await request.post('/api/facemgr/clear', { data: {} });
  });

  test('crop sliders reset on double-click, and the swap enhancement stack is mounted (collapsed)', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await openTab(page, 'extras');
    const left = page.locator('label').filter({ has: page.getByText('Left', { exact: true }) }).locator('input[type="range"]');
    await left.fill('20');
    await left.dblclick();
    await expect(left).toHaveValue('0');
    await expect(page.getByRole('list', { name: 'Enhancer pipeline' })).toHaveCount(0);
    await page.getByRole('button', { name: 'Swap enhancement pipeline' }).click();
    await expect(page.getByRole('list', { name: 'Enhancer pipeline' })).toBeVisible();
  });
});

// ── Settings tabs ──────────────────────────────────────────────────────────
test.describe('settings', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await openTab(page, 'settings');
  });

  test('five tabs; each shows its own sections and not the others\'', async ({ page }) => {
    const tabs = page.getByRole('tablist', { name: 'Settings sections' });
    await expect(tabs.getByRole('tab')).toHaveText([
      'Execution Providers & TensorRT Engines', 'CPU P/E-Core & Thread Pools', 'Paths & Storage',
      'Runtime Diagnostics & Benchmark Runner', 'Appearance & Pipeline Defaults',
    ]);
    await expect(tabs.getByRole('tab', { selected: true })).toHaveText('Execution Providers & TensorRT Engines');
    await expect(page.getByText('Provider', { exact: true }).first()).toBeVisible();
    await expect(page.getByText('Auto thread selection')).toHaveCount(0);

    await tabs.getByRole('tab', { name: /^CPU/ }).click();
    await expect(page.getByText('Auto thread selection')).toBeVisible();
    await expect(page.getByText('P/E-core placement')).toBeVisible();
    await expect(page.getByText('ROOP_CPU_DISTRIBUTION')).toBeVisible();

    await tabs.getByRole('tab', { name: /^Paths/ }).click();
    await expect(page.getByText('Faceset library folder')).toBeVisible();
    await expect(page.getByText('Video codec', { exact: true })).toBeVisible();

    await tabs.getByRole('tab', { name: /^Appearance/ }).click();
    await expect(page.getByText('Interface Theme')).toBeVisible();
    await expect(page.getByText('Detection defaults')).toBeVisible();
  });

  test('the diagnostics tab renders all fourteen runtime sections, naming the ones the engine did not send', async ({ page }) => {
    await page.getByRole('tab', { name: /^Runtime Diagnostics/ }).click();
    const sections = page.locator('[data-runtime-section]');
    await expect(sections).toHaveCount(14);
    await expect(sections.first()).toHaveAttribute('data-runtime-section', 'SYSTEM');
    await expect(sections.last()).toHaveAttribute('data-runtime-section', 'CHECKPOINT');
    await expect(page.getByText(/\d+ of 14 sections reported by the engine/)).toBeVisible();
    await expect(page.locator('[data-runtime-section="HARDWARE"]')).toContainText('NVIDIA GeForce');
    // The mock reports some sections and not others; an unreported one says so.
    await expect(page.getByText('Not reported').first()).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Hardware Benchmark & Optimization Suite' }).or(page.getByText(/Benchmark/).first())).toBeVisible();
  });

  test('searching looks across every tab; the palette opens a setting on the tab it lives on', async ({ page }) => {
    const search = page.getByRole('searchbox', { name: 'Search settings' });
    await search.fill('thread');
    await expect(page.getByText('Showing matches from every section')).toBeVisible();
    await expect(page.getByText('Auto thread selection')).toBeVisible();       // a CPU-tab control, found from the providers tab
    await search.fill('');

    await page.keyboard.press('Control+k');
    await page.getByPlaceholder('Search actions, themes, settings…').fill('Auto thread selection');
    await page.keyboard.press('Enter');
    await expect(page.getByRole('tab', { name: /^CPU/ })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByText('Auto thread selection')).toBeVisible();
  });

  test('"Open runtime diagnostics" from the telemetry popover lands on the diagnostics tab and the choice persists', async ({ page }) => {
    await openTab(page, 'home');
    await page.getByRole('button', { name: /^Hardware telemetry:/ }).click();
    await page.getByRole('button', { name: 'Open runtime diagnostics' }).click();
    await expect(page).toHaveURL(/#\/settings$/);
    await expect(page.getByRole('tab', { name: /^Runtime Diagnostics/ })).toHaveAttribute('aria-selected', 'true');
    await page.reload();
    await expect(page.getByRole('tab', { name: /^Runtime Diagnostics/ })).toHaveAttribute('aria-selected', 'true');
  });

  test('tab keys: arrows move between tabs', async ({ page }) => {
    const first = page.getByRole('tab', { name: /^Execution Providers/ });
    await first.focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.getByRole('tab', { name: /^CPU/ })).toBeFocused();
    await expect(page.getByRole('tab', { name: /^CPU/ })).toHaveAttribute('aria-selected', 'true');
    await page.keyboard.press('End');
    await expect(page.getByRole('tab', { name: /^Appearance/ })).toBeFocused();
  });
});
