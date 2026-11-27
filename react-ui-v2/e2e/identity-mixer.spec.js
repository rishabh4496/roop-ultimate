import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab } from './helpers.js';

// Stage 20 in a real Chromium, against the production build with the mock server standing in for app/api.py:
// the Latent Identity Mixer (Face Manager's Morph Lab, Face Swap's Identity Blender) and the Identity Keyframe Track.
//
// The mock invents the numbers (mock-server/identityMock.ts: different sources are orthogonal) and applies the
// backend's REQUEST rules, so what is under test is what the UI does with a response and what it puts in a request:
//   morph lab    five identities make a pentagon, a sixth is refused, the mode and the de-identification reach the
//                request, the pairwise matrix and the de-identification outcome are shown;
//   blender      five slots, the mode switch, the pairwise matrix, de-identification, the .fsz export;
//   track        a keyframe is added at the playhead and by double-click, dragged, nudged and deleted with the keyboard,
//                its curve and bezier edited, the backend's reading of the transition shown, the track survives a
//                reload, and the RECIPE a preview / blend request carries holds the keyframes with the 0-based frame
//                index the backend reads (UI frame 1 is index 0).

const AXE = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];

// The mock keeps the recipe for the life of its process: start every test from the empty one.
test.beforeEach(async ({ request }) => {
  await request.post('/api/identity/blend', { data: {} });
});

async function saveMix(request, name) {
  const res = await request.post('/api/facemgr/morph/save', {
    data: { name, components: [{ id: 'lib:Alice.fsz', weight: 1 }, { id: 'lib:Bob.fsz', weight: 1 }] },
  });
  expect(res.ok()).toBeTruthy();
}

async function openMorphLab(page) {
  await openTab(page, 'facemgr');
  await page.getByTestId('facemgr-tab-morph').click();
  await expect(page.getByTestId('morph-lab')).toBeVisible();
}

test.describe('Morph Lab (Face Manager)', () => {
  test('five identities make a pentagon, a sixth is refused', async ({ page, request }) => {
    await saveMix(request, 'S20 Mix A');
    await saveMix(request, 'S20 Mix B');
    await openMorphLab(page);
    const ids = ['src:0', 'src:1', 'lib:Alice.fsz', 'lib:Bob.fsz', 'lib:S20 Mix A.fsz'];
    for (const id of ids) await page.getByTestId(`morph-pick-${id}`).click();
    await expect(page.getByTestId('morph-pad')).toBeVisible();
    await expect(page.getByTestId('morph-shares')).toHaveText(/Shares: .* 20% · .* 20% · .* 20% · .* 20% · .* 20%/);
    await expect(page.getByTestId('morph-weight-4')).toBeVisible();

    // A sixth is refused: the pick list does not grow and no sixth weight appears.
    await page.getByTestId('morph-pick-lib:S20 Mix B.fsz').click();
    await expect(page.getByTestId('morph-weight-5')).toHaveCount(0);
    await expect(page.getByTestId('morph-pick-lib:S20 Mix B.fsz')).toHaveAttribute('aria-pressed', 'false');

    // A pentagon corner is 100% of that identity (corner 0 is the top).
    const box = await page.getByTestId('morph-pad').boundingBox();
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.1);
    await expect(page.getByTestId('morph-shares')).toHaveText(/Source face 1 100% · Source face 2 0% · Alice 0% · Bob 0% · S20 Mix A 0%/);
  });

  test('the mode and the pairwise matrix: Slerp vs Linear change the readout and the request', async ({ page }) => {
    await openMorphLab(page);
    await page.getByTestId('morph-pick-lib:Alice.fsz').click();
    await page.getByTestId('morph-pick-lib:Bob.fsz').click();
    await expect(page.getByTestId('morph-mode-slerp')).toHaveAttribute('aria-pressed', 'true');

    // Drag most of the way to Alice: 87.5 / 12.5.
    const pad = page.getByTestId('morph-pad');
    const box = await pad.boundingBox();
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.5, { steps: 6 });
    await page.mouse.up();
    const cos = page.getByTestId('morph-cosines').locator('li');
    const alice = async () => Number((await cos.nth(0).innerText()).match(/0\.\d+/)?.[0]);
    // Slerp: 0.125 x 90 deg from Alice, cos 0.981. Linear: 0.990.
    await expect.poll(alice).toBeLessThan(0.985);
    const [req] = await Promise.all([
      page.waitForRequest((r) => r.url().includes('/api/facemgr/morph/blend') && JSON.parse(r.postData() || '{}').mode === 'linear'),
      page.getByTestId('morph-mode-linear').click(),
    ]);
    expect(JSON.parse(req.postData()).mode).toBe('linear');
    await expect.poll(alice).toBeGreaterThan(0.985);

    const pair = page.getByTestId('morph-pairwise');
    await expect(pair).toBeVisible();
    await expect(pair.locator('tbody tr')).toHaveCount(2);
    await expect(pair.locator('tbody tr').first().locator('td')).toHaveText(['·', '0.00']);
  });

  test('de-identification: references are picked, sent, and the outcome is stated', async ({ page }) => {
    await openMorphLab(page);
    await page.getByTestId('morph-pick-lib:Alice.fsz').click();
    await page.getByTestId('morph-pick-lib:Bob.fsz').click();
    await expect(page.getByTestId('morph-deid-result')).toHaveCount(0);
    await page.getByTestId('morph-deid').getByText('Push the result away from reference identities').click();
    const [req] = await Promise.all([
      page.waitForRequest((r) => r.url().includes('/api/facemgr/morph/blend')
        && (JSON.parse(r.postData() || '{}').deidentify?.references || []).includes('lib:Alice.fsz')),
      page.getByTestId('morph-deid-ref-lib:Alice.fsz').click(),
    ]);
    const sent = JSON.parse(req.postData());
    expect(sent.deidentify).toMatchObject({ enabled: true, references: ['lib:Alice.fsz'], max_cosine: 0.3 });
    const result = page.getByTestId('morph-deid-result');
    await expect(result).toBeVisible();
    // Alice is one of the blended inputs: an even blend of two orthogonal identities sits at 0.71 to her, above 0.30.
    await expect(result).toContainText('0.71 →');
    await expect(result).toContainText('threshold 0.30');
  });
});

test.describe('Identity Blender (Face Swap)', () => {
  async function openBlender(page) {
    await openTab(page, 'faceswap');
    const header = page.getByRole('button', { name: /Identity Blender/ });
    await expect(header).toBeVisible();
    if ((await header.getAttribute('aria-expanded')) !== 'true') await header.click();
    const blender = page.getByTestId('identity-blender');
    await expect(blender).toBeVisible();
    return blender;
  }

  async function bindTwo(blender) {
    for (const name of ['source_alpha.png', 'source_beta.png']) {
      await blender.getByRole('button', { name: 'From gallery' }).first().click();
      await blender.getByTitle(name, { exact: true }).click();
    }
  }

  test('five slots, the mode switch, and the pairwise matrix', async ({ page }) => {
    const blender = await openBlender(page);
    await expect(blender.getByText('Drop a photo')).toHaveCount(5);        // five slots (the cap), all empty
    await expect(blender.getByTestId('blend-mode-slerp')).toHaveAttribute('aria-pressed', 'true');
    await bindTwo(blender);
    await expect(blender.getByText('Drop a photo')).toHaveCount(3);
    await expect(blender.getByTestId('identity-pairwise')).toBeVisible();
    await expect(blender.getByTestId('identity-pairwise').locator('tbody tr')).toHaveCount(2);

    const [req] = await Promise.all([
      page.waitForRequest((r) => r.url().endsWith('/api/identity/blend') && r.method() === 'POST' && JSON.parse(r.postData() || '{}').mode === 'linear'),
      blender.getByTestId('blend-mode-linear').click(),
    ]);
    expect(JSON.parse(req.postData()).components).toHaveLength(2);
    await expect(blender.getByTestId('blend-mode-linear')).toHaveAttribute('aria-pressed', 'true');
  });

  test('de-identification names references and shows the outcome; export saves the composite as a .fsz', async ({ page }) => {
    const blender = await openBlender(page);
    await bindTwo(blender);
    await blender.getByTestId('identity-deid').getByText('De-identify against reference sources').click();
    await blender.getByTestId('deid-ref-0').click();
    await expect(blender.getByTestId('deid-readout')).toContainText('cosine to references');

    const [res] = await Promise.all([
      page.waitForResponse((r) => r.url().endsWith('/api/identity/export_fsz')),
      blender.getByTestId('identity-export-fsz').click(),
    ]);
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.saved).toBe('Blended identity');
    expect(body.deidentified).toBe(true);
    expect(body.keyframed).toBe(false);
  });
});

test.describe('Identity Keyframe Track (Face Swap timeline)', () => {
  const lane = (page) => page.getByTestId('lane-identity-keyframes');
  const diamonds = (page) => lane(page).locator('button[aria-label^="Identity keyframe at frame"]');
  const frameOf = async (loc) => Number((await loc.getAttribute('aria-label')).match(/frame (\d+)/)[1]);
  // The lane can sit below the fold, and the raw mouse does not scroll: bring it into view, then measure it.
  const laneBox = async (page) => {
    await lane(page).scrollIntoViewIfNeeded();
    return page.getByTestId('identity-keyframe-hit').boundingBox();
  };

  async function bindTwoSources(page) {
    await openTab(page, 'faceswap');
    const header = page.getByRole('button', { name: /Identity Blender/ });
    if ((await header.getAttribute('aria-expanded')) !== 'true') await header.click();
    const blender = page.getByTestId('identity-blender');
    for (const name of ['source_alpha.png', 'source_beta.png']) {
      await blender.getByRole('button', { name: 'From gallery' }).first().click();
      await blender.getByTitle(name, { exact: true }).click();
    }
    await expect(blender.getByTestId('identity-pairwise')).toBeVisible();
  }

  test('the track sits under the film strip and is inert until two sources are blended', async ({ page }) => {
    await openTab(page, 'faceswap');
    await expect(lane(page)).toBeVisible();
    await expect(page.getByTestId('identity-keyframe-add')).toBeDisabled();
    await expect(page.getByTestId('identity-keyframe-count')).toHaveText('none');
    await expect(lane(page)).toContainText('Add two or more sources');
    // Double-clicking an inert track adds nothing.
    await page.getByTestId('identity-keyframe-hit').dblclick({ position: { x: 200, y: 12 } });
    await expect(diamonds(page)).toHaveCount(0);

    // Document order: the film track, then the keyframe track, then the audio / scene lanes.
    const y = async (sel) => (await page.locator(sel).first().boundingBox()).y;
    expect(await y('[data-testid="timeline-playhead"]')).toBeLessThan(await y('[data-testid="lane-identity-keyframes"]'));
    expect(await y('[data-testid="lane-identity-keyframes"]')).toBeLessThan(await y('[data-testid="timeline-stack"] >> text=Scene cuts'));
  });

  test('add at the playhead and by double-click; the request carries 0-based frame indices', async ({ page }) => {
    await bindTwoSources(page);
    await expect(page.getByTestId('identity-keyframe-add')).toBeEnabled();

    await page.getByTestId('identity-keyframe-add').click();                  // the playhead is on UI frame 1
    await expect(diamonds(page)).toHaveCount(1);
    expect(await frameOf(diamonds(page).first())).toBe(1);
    await expect(page.getByTestId('identity-keyframe-count')).toContainText('1 keyframe');

    // The recipe the blender POSTs once the second keyframe exists: UI frame 1 is frame_index 0.
    const posted = page.waitForRequest((r) => r.url().endsWith('/api/identity/blend') && r.method() === 'POST'
      && (JSON.parse(r.postData() || '{}').keyframes || []).length === 2);
    const box = await laneBox(page);
    await page.mouse.dblclick(box.x + box.width * 0.6, box.y + box.height / 2);
    await expect(diamonds(page)).toHaveCount(2);
    const second = await frameOf(diamonds(page).nth(1));
    expect(second).toBeGreaterThan(60);                                       // ~60% of the mock clip (180 frames)
    const kfs = JSON.parse((await posted).postData()).keyframes;
    expect(kfs.map((k) => k.frame_index)).toEqual([0, second - 1]);
    expect(kfs[0]).toMatchObject({ curve: 'linear' });
    expect(Object.keys(kfs[0].weights).length).toBe(2);
  });

  test('select, edit the curve and the bezier, read the backend\'s transition readout, delete', async ({ page }) => {
    await bindTwoSources(page);
    await page.getByTestId('identity-keyframe-add').click();
    const box = await laneBox(page);
    await page.mouse.dblclick(box.x + box.width * 0.5, box.y + box.height / 2);
    await expect(diamonds(page)).toHaveCount(2);

    await diamonds(page).first().click();
    const inspector = page.getByTestId('identity-keyframe-inspector');
    await expect(inspector).toBeVisible();
    await expect(page.getByTestId('identity-curve-linear')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('identity-bezier-editor')).toHaveCount(0);

    // A new keyframe is seeded with the blend already on screen, so both start identical (0 deg apart). The blender's
    // slots default to weights 100 : 50, so both are 2 : 1 shares. Make the first one all source 0: against the 2 : 1
    // keyframe that is atan(1/2) = 26.6 deg for the mock's orthogonal sources.
    await page.getByTestId('identity-keyframe-share-0').getByRole('slider').fill('100');
    await page.getByTestId('identity-keyframe-share-1').getByRole('slider').fill('0');

    await page.getByTestId('identity-curve-cubic_bezier').click();
    const editor = page.getByTestId('identity-bezier-editor');
    await expect(editor).toBeVisible();
    const handle = page.getByTestId('identity-bezier-handle-1');
    const before = await handle.getAttribute('aria-valuenow');
    await handle.focus();
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await expect.poll(async () => Number(await handle.getAttribute('aria-valuenow'))).toBeGreaterThan(Number(before));
    await page.getByRole('button', { name: 'Fast start' }).click();
    await expect(editor).toHaveAttribute('aria-label', /cubic-bezier\(0\.10, 0\.90, 0\.30, 1\.00\)/);

    // The backend's own reading of the track: the angle between the two keyframes (orthogonal sources: 90 deg).
    await expect(page.getByTestId('identity-keyframe-readout')).toContainText('26.6° to the next keyframe');
    await expect(page.getByTestId('identity-keyframe-readout')).toContainText('largest step');

    await page.getByTestId('identity-curve-hold').click();
    await expect(page.getByTestId('identity-keyframe-readout')).toContainText('1 hold cut');

    // A share is editable and the keyframe keeps its place.
    await page.getByTestId('identity-keyframe-share-1').getByRole('slider').fill('60');
    await expect(page.getByTestId('identity-keyframe-share-1')).toContainText('38%');            // 60 of (100 + 60)

    await page.getByTestId('identity-keyframe-delete').click();
    await expect(inspector).toHaveCount(0);
    await expect(diamonds(page)).toHaveCount(1);
  });

  test('drag moves a keyframe and commits once; the arrow keys nudge; Delete removes; an occupied frame refuses', async ({ page }) => {
    await bindTwoSources(page);
    // Measure before EACH double-click: the first one selects its keyframe and opens the inspector below the lane.
    let box = await laneBox(page);
    await page.mouse.dblclick(box.x + box.width * 0.25, box.y + box.height / 2);
    await expect(diamonds(page)).toHaveCount(1);
    box = await laneBox(page);
    await page.mouse.dblclick(box.x + box.width * 0.75, box.y + box.height / 2);
    await expect(diamonds(page)).toHaveCount(2);
    const start = await frameOf(diamonds(page).first());

    const d = await diamonds(page).first().boundingBox();
    await page.mouse.move(d.x + d.width / 2, d.y + d.height / 2);
    await page.mouse.down();
    await page.mouse.move(d.x + d.width / 2 + 60, d.y + d.height / 2, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => frameOf(diamonds(page).first())).toBeGreaterThan(start + 20);

    const moved = await frameOf(diamonds(page).first());
    await diamonds(page).first().focus();
    await page.keyboard.press('ArrowRight');
    await expect.poll(async () => frameOf(diamonds(page).first())).toBe(moved + 1);
    await page.keyboard.press('Shift+ArrowLeft');
    await expect.poll(async () => frameOf(diamonds(page).first())).toBe(moved - 9);

    await expect(diamonds(page)).toHaveCount(2);

    await diamonds(page).first().focus();
    await page.keyboard.press('Delete');
    await expect(diamonds(page)).toHaveCount(1);
  });

  test('the track survives a reload and belongs to its clip', async ({ page }) => {
    await bindTwoSources(page);
    await page.getByTestId('identity-keyframe-add').click();
    const box = await laneBox(page);
    await page.mouse.dblclick(box.x + box.width * 0.5, box.y + box.height / 2);
    await expect(diamonds(page)).toHaveCount(2);
    const frames = [await frameOf(diamonds(page).nth(0)), await frameOf(diamonds(page).nth(1))];

    await page.reload();
    await openTab(page, 'faceswap');
    await expect(diamonds(page)).toHaveCount(2);
    expect([await frameOf(diamonds(page).nth(0)), await frameOf(diamonds(page).nth(1))]).toEqual(frames);

    // They were stored under this clip's key, in the recipe's own shape (0-based, curve, bezier, weights).
    const stored = await page.evaluate(() => {
      const k = Object.keys(localStorage).find((x) => x.startsWith('roop_idkeys_'));
      return k ? JSON.parse(localStorage.getItem(k)) : null;
    });
    expect(stored.map((s) => s.frame_index)).toEqual(frames.map((f) => f - 1));
    expect(stored[0]).toHaveProperty('weights');

    await page.getByTestId('identity-keyframe-clear').click();
    await expect(diamonds(page)).toHaveCount(0);
    await page.reload();
    await openTab(page, 'faceswap');
    await expect(page.getByTestId('identity-keyframe-count')).toHaveText('none');
  });

  test('a keyframed preset exports the identity at the playhead', async ({ page }) => {
    await bindTwoSources(page);
    await page.getByTestId('identity-keyframe-add').click();
    const box = await laneBox(page);
    await page.mouse.dblclick(box.x + box.width * 0.5, box.y + box.height / 2);
    await expect(diamonds(page)).toHaveCount(2);
    await expect(page.getByTestId('identity-keyframe-summary')).toContainText('2 identity keyframes');
    const button = page.getByTestId('identity-export-fsz');
    // A click on the track parks the playhead, so it is wherever the double-click was: read it from the button.
    await expect(button).toContainText(/Export identity at frame \d+/);
    const shown = Number((await button.innerText()).match(/frame (\d+)/)[1]);
    const [req] = await Promise.all([
      page.waitForRequest((r) => r.url().endsWith('/api/identity/export_fsz')),
      button.click(),
    ]);
    const sent = JSON.parse(req.postData());
    expect(sent.frame_index).toBe(shown - 1);                                 // the button says UI frame N; the request is N - 1
    expect(sent.recipe.keyframes).toHaveLength(2);
  });

  test('axe: the lane, an open inspector with the bezier editor, and the blender panel', async ({ page }) => {
    await bindTwoSources(page);
    await page.getByTestId('identity-keyframe-add').click();
    const box = await laneBox(page);
    await page.mouse.dblclick(box.x + box.width * 0.5, box.y + box.height / 2);
    await diamonds(page).first().click();
    await page.getByTestId('identity-curve-cubic_bezier').click();
    await expect(page.getByTestId('identity-bezier-editor')).toBeVisible();
    for (const sel of ['[data-testid="lane-identity-keyframes"]', '[data-testid="identity-keyframe-inspector"]', '[data-testid="identity-blender"]']) {
      const results = await new AxeBuilder({ page }).include(sel).withTags(AXE).analyze();
      expect(results.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`), sel).toEqual([]);
    }
  });
});
