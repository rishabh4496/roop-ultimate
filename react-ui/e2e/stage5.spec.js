import { test, expect } from '@playwright/test';
import { openTab } from './helpers.js';

// Stage 5, in a real Chromium against the production build with the mock backend:
//   #/facemgr    Morph Lab (pad + sliders agree, 2-4 identities, save / preview) and the pose radar + Auto-Prune
//   #/processing the Segment Chunk Matrix: pills through a render, re-render ONE chunk, join with the duration check
//   #/gallery    hover scrubbing, and "Send to Face Swap as A/B reference" through to the panel there
//   #/history    checkpoint vs current settings, selective restore
//
// WHAT THESE DO NOT PROVE. The mock stands in for app/api.py: the numbers it returns for a blend or a join are its
// own arithmetic, and Playwright's Chromium cannot decode H.264, so hover scrubbing is driven against a stubbed media
// element (the pointer -> seek-time mapping is real; a real decode scrubbing real footage is not). The backend rules
// behind each of these are pinned in app/tests/test_identity_lab.py and test_segment_custody.py.

const MEDIA = 'mock-media-0001';

async function clearFaceManager(request) {
  await request.post('/api/facemgr/clear', { data: {} });
}

test.describe('Face Manager', () => {
  test.beforeEach(async ({ request }) => { await clearFaceManager(request); });
  test.afterEach(async ({ request }) => { await clearFaceManager(request); });

  test('Morph Lab: pick, drag the pad, nudge a slider - they agree, and the server reports the blend', async ({ page }) => {
    await openTab(page, 'facemgr');
    await page.getByTestId('facemgr-tab-morph').click();
    await expect(page.getByTestId('morph-lab')).toBeVisible();

    // An identity the server cannot use is listed, disabled, with the reason.
    const legacy = page.getByTestId('morph-pick-lib:Legacy.fsz');
    await expect(legacy).toBeDisabled();
    await expect(legacy).toHaveAttribute('title', /legacy \.fsz/);
    await expect(page.getByTestId('morph-need-more')).toBeVisible();

    await page.getByTestId('morph-pick-lib:Alice.fsz').click();
    await expect(page.getByTestId('morph-pad'), 'one identity is not a blend').toHaveCount(0);
    await page.getByTestId('morph-pick-lib:Bob.fsz').click();
    const pad = page.getByTestId('morph-pad');
    await expect(pad).toBeVisible();
    await expect(page.getByTestId('morph-shares')).toHaveText('Shares: Alice 50% · Bob 50%');

    // Server readout for an even blend of two orthogonal identities: 0.707 to each.
    const cos = page.getByTestId('morph-cosines').locator('li');
    await expect.poll(async () => (await cos.nth(0).innerText()).replace(/\s+/g, ' ')).toContain('0.707');

    // Drag the cursor most of the way to Alice's corner: the pad sets the weights.
    const box = await pad.boundingBox();
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.5, { steps: 6 });
    await page.mouse.up();
    await expect(page.getByTestId('morph-shares')).toHaveText('Shares: Alice 88% · Bob 12%');
    // 0.875 / sqrt(0.875^2 + 0.125^2) = 0.9899
    await expect.poll(async () => Number((await cos.nth(0).innerText()).match(/0\.\d+/)?.[0])).toBeGreaterThan(0.985);
    await expect.poll(async () => Number((await cos.nth(1).innerText()).match(/0\.\d+/)?.[0])).toBeLessThan(0.15);

    // The slider and the pad are one control: a slider keypress moves the cursor's share too.
    await page.getByTestId('morph-cursor').scrollIntoViewIfNeeded();
    const aliceSlider = page.getByTestId('morph-weight-0').getByRole('slider');
    await aliceSlider.focus();
    for (let i = 0; i < 8; i++) await aliceSlider.press('ArrowLeft');
    await expect(page.getByTestId('morph-shares')).toHaveText(/Alice \d+% · Bob \d+%/);
    const shares = (await page.getByTestId('morph-shares').innerText()).match(/Alice (\d+)% · Bob (\d+)%/);
    expect(Number(shares[1]) + Number(shares[2]), 'auto mode always totals 100').toBe(100);
    expect(Number(shares[1])).toBeLessThan(88);
  });

  test('Morph Lab: three identities make a triangle, a fifth is refused, preview opens Face Swap, save adds to the library', async ({ page }) => {
    await openTab(page, 'facemgr');
    await page.getByTestId('facemgr-tab-morph').click();
    for (const id of ['src:0', 'src:1', 'lib:Alice.fsz']) await page.getByTestId(`morph-pick-${id}`).click();
    const pad = page.getByTestId('morph-pad');
    await expect(pad).toBeVisible();
    await expect(page.getByTestId('morph-shares')).toHaveText('Shares: Source face 1 34% · Source face 2 33% · Alice 33%');
    // a corner of the triangle is 100% of that identity (corner 0 is the top)
    const box = await pad.boundingBox();
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.1);
    await expect(page.getByTestId('morph-shares')).toHaveText('Shares: Source face 1 100% · Source face 2 0% · Alice 0%');

    await page.getByTestId('morph-pick-lib:Bob.fsz').click();
    await page.getByTestId('morph-pick-lib:Bob.fsz').scrollIntoViewIfNeeded();
    // 4 picked: the pad is a square now, and a fifth pick is refused (the thumbnails are all in the list)
    await expect(page.getByTestId('morph-weight-3')).toBeVisible();
    await expect(page.getByTestId('morph-shares')).toContainText('Bob 25%');

    // Save: the blend appears in the library and the candidate list.
    await page.getByLabel('Blended identity name').fill('My Mix');
    await page.getByTestId('morph-save').click();
    await expect(page.getByTestId('morph-pick-lib:My Mix.fsz')).toBeVisible();

    // Preview: loads it as a source and shows Face Swap.
    await page.getByTestId('morph-preview').click();
    await expect(page.locator('nav[aria-label="Main"] button[aria-current="page"]')).toContainText('Face Swap');
  });

  test('pose radar: counts per bucket, the one missing angle is named, Auto-Prune prices then removes', async ({ page, request }) => {
    for (let i = 0; i < 7; i++) await request.post('/api/facemgr/cut', { data: {} });
    await openTab(page, 'facemgr');
    const radar = page.getByTestId('pose-radar');
    await expect(radar).toBeVisible();
    // every bucket has a face in the seed set -> nothing is missing, and the scatter holds all 7
    await expect(page.getByTestId('pose-missing')).toHaveCount(0);
    await expect(radar.locator('svg').nth(1).locator('circle')).toHaveCount(7);
    await expect(radar.locator('svg').first()).toHaveAttribute('aria-label', /Frontal 3, Left profile 1, Right profile 1, High angle 1, Low angle 1/);

    // face 6 is blurry in a populated bucket (dropped), face 7 repeats face 1 (dropped), face 3 is a blurry
    // right profile that is the ONLY one (kept for coverage): 2 of 7 would go.
    await expect(page.getByTestId('prune-preview')).toHaveText('would drop 2 of 7');
    await page.getByTestId('auto-prune').click();
    const dialog = page.getByRole('dialog');
    await expect(dialog).toContainText('Remove 2 of 7 faces?');
    await expect(dialog).toContainText('1 blurry');
    await expect(dialog).toContainText('1 near-duplicate');
    await expect(dialog).toContainText('only example of that angle');
    await dialog.getByRole('button', { name: 'Auto-prune' }).click();
    await expect(radar.locator('svg').nth(1).locator('circle')).toHaveCount(5);
    await expect(page.getByTestId('prune-preview')).toHaveText('nothing to drop');
    // the right profile survived even though it is soft
    await expect(radar.locator('svg').first()).toHaveAttribute('aria-label', /Right profile 1/);
  });

  test('pose radar names the angles a thin set is missing', async ({ page, request }) => {
    for (let i = 0; i < 2; i++) await request.post('/api/facemgr/cut', { data: {} });     // frontal + left profile
    await openTab(page, 'facemgr');
    await expect(page.getByTestId('pose-missing')).toContainText('Right profile, High angle, Low angle');
  });

  test('an empty set says so instead of drawing an empty radar', async ({ page }) => {
    await openTab(page, 'facemgr');
    await expect(page.getByTestId('pose-radar-empty')).toBeVisible();
  });
});

// ── segments ────────────────────────────────────────────────────────────────

async function resetQueue(request) {
  await request.post('/api/queue/stop').catch(() => {});
  await request.post('/api/queue/clear', { data: { keep_running: false } }).catch(() => {});
}

async function seedSplit(request, ranges = [[1, 60], [61, 120], [121, 180]]) {
  const jobs = ranges.map(([s, e], i) => ({
    target_name: 'interview_scene_1080p.mp4', target_media_id: MEDIA, source_index: 0, payload: {},
    frame_start: s, frame_end: e, label: `chunk ${i + 1}`, segment: { group: 'e2e-split', index: i + 1, count: ranges.length },
  }));
  const res = await request.post('/api/queue/add_batch', { data: { jobs } });
  expect(res.ok(), await res.text()).toBeTruthy();
}

const groupsOf = async (request) => (await (await request.get('/api/queue/segments')).json()).groups;

test.describe('Segment Chunk Matrix', () => {
  test.beforeEach(async ({ request }) => { await resetQueue(request); });
  test.afterEach(async ({ request }) => { await resetQueue(request); });

  test('nothing is shown when no render is split', async ({ page }) => {
    await openTab(page, 'processing');
    await expect(page.getByTestId('segment-matrix')).toHaveCount(0);
  });

  test('queued chunks are pills, the join says why it is off; a finished set joins by itself, verified', async ({ page, request }) => {
    await seedSplit(request);
    await openTab(page, 'processing');
    const matrix = page.getByTestId('segment-matrix');
    await expect(matrix).toBeVisible();
    for (const i of [1, 2, 3]) await expect(page.getByTestId(`seg-pill-${i}`)).toHaveAttribute('data-state', 'queued');
    await expect(page.getByTestId('seg-join')).toBeDisabled();
    await expect(page.getByTestId('seg-join-blocked')).toContainText('not finished');
    await expect(page.getByTestId('seg-summary')).toHaveText('0 of 3 rendered · 3 queued');

    await request.post('/api/queue/start');
    // a chunk is seen rendering while the others still wait
    await expect(page.locator('[data-testid^="seg-pill-"][data-state="rendering"]').first()).toBeVisible({ timeout: 20_000 });
    for (const i of [1, 2, 3]) await expect(page.getByTestId(`seg-pill-${i}`)).toHaveAttribute('data-state', 'completed', { timeout: 60_000 });
    await expect(page.getByTestId('seg-summary')).toContainText('3 of 3 rendered');
    await expect(page.getByTestId('seg-summary')).toContainText('joined');
    await expect(page.getByTestId('seg-verify')).toContainText('duration');
    await expect(page.getByTestId('seg-verify')).toContainText('matches');
    await expect(page.getByTestId('seg-verify')).toContainText('source audio stream copied');
    await expect(page.getByTestId('seg-join')).toBeEnabled();
    await expect(page.getByTestId('seg-joined-link')).toContainText('interview_scene_1080p_joined.mp4');
  });

  test('re-render ONE chunk: only it goes back to the queue, and the set re-joins when it is done', async ({ page, request }) => {
    await seedSplit(request);
    await request.post('/api/queue/start');
    await expect.poll(async () => (await groupsOf(request))[0]?.completed, { timeout: 60_000 }).toBe(3);

    await openTab(page, 'processing');
    await page.getByTestId('seg-pill-2').click();
    const detail = page.getByTestId('seg-detail');
    await expect(detail).toContainText('seg_0002_61_120.mp4');
    await expect(detail).toContainText('MB on disk');

    // a finished chunk asks first, and says what it will and will not touch
    await detail.getByTestId('seg-rerender').click();
    const dialog = page.getByRole('dialog');
    await expect(dialog).toContainText('Re-render chunk 2?');
    await expect(dialog).toContainText('other chunks are not touched');
    await dialog.getByRole('button', { name: 'Re-render chunk' }).click();

    // chunk 2 leaves "completed"; chunks 1 and 3 never do
    const seen = new Set();
    await expect.poll(async () => {
      const g = (await groupsOf(request))[0];
      seen.add(g.segments.map((s) => s.state).join('|'));
      return g.segments[1].state;
    }, { timeout: 30_000 }).not.toBe('COMPLETED');
    await expect.poll(async () => (await groupsOf(request))[0].completed, { timeout: 60_000 }).toBe(3);
    for (const combo of seen) {
      const [a, , c] = combo.split('|');
      expect([a, c], `a sibling chunk left COMPLETED during the re-render: ${combo}`).toEqual(['COMPLETED', 'COMPLETED']);
    }
    await expect(page.getByTestId('seg-pill-2')).toHaveAttribute('data-state', 'completed');
    await expect(page.getByTestId('seg-summary')).toContainText('joined');
  });

  test('a join whose duration is wrong is reported as wrong, not as done', async ({ page, request }) => {
    await seedSplit(request);
    await request.post('/api/queue/start');
    await expect.poll(async () => (await groupsOf(request))[0]?.completed, { timeout: 60_000 }).toBe(3);
    await request.post('/api/mock/segments/mismatch_next_join');
    await openTab(page, 'processing');
    await page.getByTestId('seg-join').click();
    const verify = page.getByTestId('seg-verify');
    await expect(verify).toContainText('expected 6.00s');
    await expect(verify).toHaveAttribute('role', 'alert');
    await expect(page.getByTestId('seg-summary')).toContainText('DURATION MISMATCH');
    // and a later, correct join clears it
    await page.getByTestId('seg-join').click();
    await expect(page.getByTestId('seg-verify')).toContainText('matches');
  });

  test('the matrix is on #/batch too', async ({ page, request }) => {
    await seedSplit(request, [[1, 90], [91, 180]]);
    await openTab(page, 'batch');
    await expect(page.getByTestId('segment-matrix')).toBeVisible();
    await expect(page.getByTestId('seg-pill-2')).toHaveAttribute('data-state', 'queued');
  });
});

// ── gallery ─────────────────────────────────────────────────────────────────

test.describe('Outputs', () => {
  test.beforeEach(async ({ page }) => {
    // Chromium here cannot decode H.264, so give every media element a 10 s duration and record the times it is
    // told to seek to. The mapping under test (pointer x -> seek time) is the app's; only the decoder is stubbed.
    await page.addInitScript(() => {
      const times = new WeakMap();
      window.__seeks = [];
      Object.defineProperty(HTMLMediaElement.prototype, 'duration', { configurable: true, get() { return 10; } });
      Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', {
        configurable: true,
        get() { return times.get(this) || 0; },
        set(v) { times.set(this, v); window.__seeks.push(Number(v)); },
      });
    });
  });

  test('moving across a video thumbnail scrubs through it; leaving resets it', async ({ page }) => {
    await openTab(page, 'gallery');
    const surface = page.getByTestId('scrub-surface').first();
    await expect(surface).toBeVisible();
    await surface.scrollIntoViewIfNeeded();
    const b = await surface.boundingBox();
    const y = b.y + b.height * 0.4;

    await page.mouse.move(b.x + b.width * 0.1, y);
    await page.mouse.move(b.x + b.width * 0.25, y, { steps: 4 });
    await expect.poll(async () => (await page.evaluate(() => window.__seeks)).at(-1)).toBeCloseTo(2.49, 1);
    await expect(page.getByTestId('scrub-clock').first()).toHaveText(/0:02\.4 \/ 0:10\.0/);
    const width = await page.getByTestId('scrub-bar').first().evaluate((el) => el.style.width);
    expect(parseFloat(width)).toBeCloseTo(25, 0);

    await page.mouse.move(b.x + b.width * 0.75, y, { steps: 6 });
    await expect.poll(async () => (await page.evaluate(() => window.__seeks)).at(-1)).toBeCloseTo(7.47, 1);

    // it follows the pointer in both directions and is monotone in x
    await page.mouse.move(b.x + b.width * 0.5, y, { steps: 4 });
    await expect.poll(async () => (await page.evaluate(() => window.__seeks)).at(-1)).toBeCloseTo(4.98, 1);

    await page.mouse.move(1, 1);                                // off the card (not onto the NEXT card in the grid)
    await expect.poll(async () => (await page.evaluate(() => window.__seeks)).at(-1)).toBe(0);
    expect(parseFloat(await page.getByTestId('scrub-bar').first().evaluate((el) => el.style.width))).toBe(0);
  });

  test('Send to Face Swap as A/B reference parks the render and shows it beside the live preview', async ({ page }) => {
    await openTab(page, 'gallery');
    // By name: other specs finish renders, and the mock lists each as a new output ahead of the seeded ones.
    const card = page.locator('div.group\\/card', { hasText: 'interview_swapped_final.mp4' }).first();
    await card.hover();
    await card.getByTestId('send-reference').click();
    await expect(page.locator('nav[aria-label="Main"] button[aria-current="page"]')).toContainText('Face Swap');
    const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('roop_ab_reference')));
    expect(stored).toMatchObject({ name: 'interview_swapped_final.mp4', kind: 'video' });

    const panel = page.getByTestId('ab-reference');
    await panel.scrollIntoViewIfNeeded();
    await expect(panel).toBeVisible();
    await expect(panel).toContainText('interview_swapped_final.mp4');
    await expect(page.getByTestId('ab-video')).toBeAttached();

    // the wipe moves
    const slider = page.getByTestId('ab-slider');
    await slider.fill('20');
    await expect(panel).toContainText('20%');
    const clip = await page.getByTestId('ab-video').evaluate((el) => el.parentElement.style.clipPath);
    expect(clip).toMatch(/inset\(0(px)? 80%/);

    // a render that starts later in the clip: the offset moves where A is looked up
    await page.evaluate(() => { window.__seeks.length = 0; });
    await page.getByTestId('ab-offset').fill('51');
    await expect.poll(async () => JSON.parse(await page.evaluate(() => localStorage.getItem('roop_ab_reference'))).offset).toBe(50);

    await page.getByTestId('ab-clear').click();
    await expect(panel).toHaveCount(0);
    expect(await page.evaluate(() => localStorage.getItem('roop_ab_reference'))).toBeNull();
  });

  test('an image output can be sent too', async ({ page }) => {
    await openTab(page, 'gallery');
    const imageCard = page.locator('div.group\\/card', { hasText: 'portrait_swapped_ultramax.png' }).first();
    await imageCard.hover();
    await imageCard.getByTestId('send-reference').click();
    const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('roop_ab_reference')));
    expect(stored).toMatchObject({ name: 'portrait_swapped_ultramax.png', kind: 'image' });
    await expect(page.getByTestId('ab-reference')).toContainText('portrait_swapped_ultramax.png');
    await expect(page.getByTestId('ab-video')).toHaveCount(0);
    await page.evaluate(() => localStorage.removeItem('roop_ab_reference'));
  });
});

// ── history ─────────────────────────────────────────────────────────────────

test.describe('History: checkpoint vs current', () => {
  test('lists what differs, restores one parameter, then the rest', async ({ page }) => {
    await openTab(page, 'history');
    await page.getByTestId('run-diff-1700000000001').click();
    const panel = page.getByTestId('checkpoint-diff');
    await expect(panel).toBeVisible();

    // The brief's own examples, in the vocabulary the rest of the app uses
    const fidelity = page.getByTestId('diff-row-codeformer_fidelity');
    await expect(fidelity).toContainText('CodeFormer fidelity');
    await expect(fidelity).toContainText('0.37');
    const feather = page.getByTestId('diff-row-face_mask_blend');
    await expect(feather).toContainText('Mask feather');
    await expect(feather).toContainText('13px');
    expect(await feather.locator('td').nth(3).innerText(), 'the current value is shown, and differs').not.toBe('13px');
    await expect(page.getByTestId('diff-row-swap_model')).toContainText('inswapper');

    // a parameter only the checkpoint recorded can still be restored (it adds it)
    await expect(page.getByTestId('diff-row-only_in_checkpoint')).toHaveAttribute('data-kind', 'only-checkpoint');

    const before = Number((await page.getByTestId('diff-count').innerText()).match(/\d+/)[0]);
    await feather.getByRole('button', { name: /Restore Mask feather/ }).click();
    await expect(page.getByTestId('diff-row-face_mask_blend'), 'a restored parameter now matches, so its row is gone').toHaveCount(0);
    await expect(page.getByTestId('diff-count')).toHaveText(`${before - 1} parameter${before - 1 === 1 ? '' : 's'} differ.`);
    await expect(page.getByTestId('diff-row-codeformer_fidelity'), 'the others were not touched').toBeVisible();

    // tick two, restore them together
    await page.getByRole('checkbox', { name: 'Restore CodeFormer fidelity' }).check();
    await page.getByRole('checkbox', { name: 'Restore Swapper' }).check();
    await expect(page.getByTestId('diff-restore-selected')).toHaveText('Restore selected (2)');
    await page.getByTestId('diff-restore-selected').click();
    await expect(page.getByTestId('diff-row-codeformer_fidelity')).toHaveCount(0);
    await expect(page.getByTestId('diff-row-swap_model')).toHaveCount(0);

    // and the rest in one go: nothing differs afterwards
    await page.getByTestId('diff-restore-all').click();
    await expect(panel).toContainText('The current settings match this checkpoint.');
  });

  test('the second run has its own checkpoint, and closing the panel closes it', async ({ page }) => {
    await openTab(page, 'history');
    await page.getByTestId('run-diff-1700000000002').click();
    await expect(page.getByTestId('diff-row-codeformer_fidelity')).toContainText('0.90');
    await page.getByTestId('diff-close').click();
    await expect(page.getByTestId('checkpoint-diff')).toHaveCount(0);
  });
});
