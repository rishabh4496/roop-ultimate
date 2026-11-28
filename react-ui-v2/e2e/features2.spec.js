// Second batch of 1.0 features brought to 2.0: split-render segments (Batch) and persistent
// projects (History), plus the extras added with them.
import { test, expect, openScreen } from './fixtures.js';

const post = (page, path, body = {}) => page.evaluate(([p, b]) => fetch(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }).then((r) => r.json()), [path, body]);
const get = (page, path) => page.evaluate((p) => fetch(p).then((r) => r.json()), path);

async function resetQueue(page) {
  await post(page, '/api/queue/stop').catch(() => {});
  await post(page, '/api/queue/clear', { keep_running: false }).catch(() => {});
}

async function seedSplit(page) {
  const st = await get(page, '/api/state');
  const t = st.targets?.[0] || {};
  const jobs = [[1, 60], [61, 120], [121, 180]].map(([s, e], i) => ({
    target_name: t.name || 'clip.mp4', target_media_id: t.media_id || t.id || 'media-0', source_index: 0, payload: {},
    frame_start: s, frame_end: e, label: `chunk ${i + 1}`, segment: { group: 'v2-e2e-split', index: i + 1, count: 3 },
  }));
  await post(page, '/api/queue/add_batch', { jobs });
}

test.describe('split renders', () => {
  test.afterEach(async ({ page }) => { await resetQueue(page); });

  test('a split shows one pill per slice; it joins once every slice is done, with the duration verified', async ({ page, v2 }) => {
    await openScreen(page, 'batch');
    await resetQueue(page);
    await seedSplit(page);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.getByTestId('batch').waitFor();
    const group = page.getByTestId('seg-group');
    await expect(group).toBeVisible();
    for (const i of [1, 2, 3]) await expect(page.getByTestId(`seg-pill-${i}`)).toHaveAttribute('data-state', 'queued');
    await expect(page.getByTestId('seg-join')).toBeDisabled();
    await expect(page.getByTestId('seg-summary')).toHaveText('0 of 3 rendered · 3 queued');

    await page.getByTestId('queue-start').click();
    for (const i of [1, 2, 3]) await expect(page.getByTestId(`seg-pill-${i}`)).toHaveAttribute('data-state', 'completed', { timeout: 60_000 });
    await expect(page.getByTestId('seg-summary')).toContainText('3 of 3 rendered');
    await expect(page.getByTestId('seg-join')).toBeEnabled({ timeout: 10_000 });
    await page.getByTestId('seg-join').click();
    await expect(page.getByTestId('seg-verify')).toContainText('matches', { timeout: 20_000 });
    await expect(page.getByTestId('seg-joined')).toBeVisible();
    expect(v2.errors).toEqual([]);
  });

  test('re-render ONE slice after a confirmation; the others stay completed', async ({ page, v2 }) => {
    await openScreen(page, 'batch');
    await resetQueue(page);
    await seedSplit(page);
    await post(page, '/api/queue/start');
    await expect.poll(async () => (await get(page, '/api/queue/segments')).groups?.[0]?.completed, { timeout: 60_000 }).toBe(3);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.getByTestId('seg-pill-2').click();
    await expect(page.getByTestId('seg-detail')).toContainText('frames 61-120');
    // Dismiss: nothing is sent.
    page.once('dialog', (d) => d.dismiss());
    await page.getByTestId('seg-rerender').click();
    await page.waitForTimeout(300);
    expect(v2.posted('/api/queue/segments/rerender')).toBe(false);
    page.once('dialog', (d) => { expect(d.message()).toContain('other slices are not touched'); d.accept(); });
    await page.getByTestId('seg-rerender').click();
    await expect.poll(() => v2.bodiesOf('/api/queue/segments/rerender')).toEqual([expect.objectContaining({ start: true })]);
    const seen = new Set();
    await expect.poll(async () => {
      const g = (await get(page, '/api/queue/segments')).groups[0];
      seen.add(g.segments.map((s) => s.state).join('|'));
      return g.completed;
    }, { timeout: 60_000 }).toBe(3);
    for (const combo of seen) {
      const [a, , c] = combo.split('|');
      expect([a, c], `a sibling slice left COMPLETED: ${combo}`).toEqual(['COMPLETED', 'COMPLETED']);
    }
    expect(v2.errors).toEqual([]);
  });
});

// The engine's shape: project_checkpoint.summarize (id, name, job_id, state, recoverable, safe_frame,
// next_frame, segments, updated_at, error, validation_errors, project_file). The mock lists none.
const PROJECTS = [
  { id: 'proj-interrupted', name: 'wedding_cut.mp4', job_id: 'j1', state: 'INTERRUPTED', recoverable: true, safe_frame: 480, next_frame: 481, segments: 4, updated_at: Math.floor(Date.now() / 1000) - 600, error: '', validation_errors: [], project_file: 'app/projects/proj-interrupted.json' },
  { id: 'proj-moved', name: 'interview.mp4', job_id: 'j2', state: 'INTERRUPTED', recoverable: false, safe_frame: 120, next_frame: 121, segments: 1, updated_at: Math.floor(Date.now() / 1000) - 7200, error: '', validation_errors: ['the swap model changed since this project started'], project_file: 'app/projects/proj-moved.json' },
  { id: 'proj-done', name: 'clip.mp4', job_id: 'j3', state: 'COMPLETED', recoverable: false, safe_frame: 300, next_frame: null, segments: 3, updated_at: Math.floor(Date.now() / 1000) - 86400 * 2, error: '', validation_errors: [], project_file: 'app/projects/proj-done.json' },
];

test('history lists projects; Resume is offered only where the engine allows it, and a refusal shows its reasons', async ({ page, v2 }) => {
  await page.route((u) => u.pathname === '/api/projects', (r) => r.fulfill({ json: { projects: PROJECTS } }));
  await page.route((u) => /^\/api\/projects\/[^/]+\/(validate|load|resume)$/.test(u.pathname), (r) => {
    const [, , , id, verb] = new URL(r.request().url()).pathname.split('/');
    if (id === 'proj-moved') return r.fulfill({ status: 409, json: { message: 'cannot resume: the swap model changed since this project started', reasons: ['the swap model changed'] } });
    if (verb === 'validate') return r.fulfill({ json: { valid: true, project: PROJECTS[0] } });
    return r.fulfill({ json: { project: PROJECTS[0], state: {} } });
  });
  await openScreen(page, 'history');
  const projects = page.getByTestId('project');
  // Resumable by default: only the one the engine calls recoverable.
  await expect(projects).toHaveCount(1);
  await expect(projects.first()).toContainText('wedding_cut.mp4');
  await expect(projects.first().getByTestId('project-resume')).toBeVisible();
  await page.getByTestId('projects-filter-all').click();
  await expect(projects).toHaveCount(3);
  await expect(projects.nth(1).getByTestId('project-resume')).toHaveCount(0);
  await expect(projects.nth(2).getByTestId('project-resume')).toHaveCount(0);

  await projects.first().getByTestId('project-validate').click();
  await expect.poll(() => v2.posted('/api/projects/proj-interrupted/validate')).toBe(true);

  // A refused load shows the server's reason on the card.
  page.once('dialog', (d) => d.accept());
  await projects.nth(1).getByTestId('project-load').click();
  await expect(projects.nth(1).getByTestId('project-why')).toContainText('swap model changed');

  // Resume asks first, then posts and re-reads the studio.
  page.once('dialog', (d) => { expect(d.message()).toContain('4 segment(s) already written'); d.accept(); });
  await projects.first().getByTestId('project-resume').click();
  await expect.poll(() => v2.posted('/api/projects/proj-interrupted/resume')).toBe(true);
  // The 409 is the test's own answer, not a 2.0 defect.
  expect(v2.errors.filter((e) => !/409/.test(e) && !/swap model changed/.test(e))).toEqual([]);
});

test('an output card runs the engine quality report on that file and shows its grade and metrics', async ({ page, v2 }) => {
  await openScreen(page, 'outputs');
  const run = page.getByTestId('quality-run').first();
  await expect(run).toBeVisible();
  await run.click();
  const report = page.getByTestId('quality-report').first();
  await expect(report).toBeVisible();
  await expect(report.getByTestId('quality-grade')).toHaveText(/[A-E]/);
  await expect(report).toContainText('Identity likeness');
  await expect(report).toContainText('Sharpness');
  const body = v2.bodiesOf('/api/quality/analyze')[0];
  expect(body.path).toMatch(/[\/]/);
  expect(v2.errors).toEqual([]);
});

test('people tools: add an angle, find angles, merge two people, scan the clip and auto-group', async ({ page, v2 }) => {
  const { readyStudio } = await import('./fixtures.js');
  await readyStudio(page, v2);
  const people = page.getByTestId('person');
  // The picked person is selected, so its tools show.
  const tools = page.getByTestId('person-tools');
  await expect(tools).toHaveCount(1);
  await tools.getByTestId('person-add-angle').click();
  await expect.poll(() => v2.posted('/api/target/add_angle')).toBe(true);
  expect(v2.bodiesOf('/api/target/add_angle')[0]).toHaveProperty('frame');
  await tools.getByTestId('person-find-angles').click();
  await expect.poll(() => v2.posted('/api/target/auto_angles')).toBe(true);

  // Two people (the added angle, regrouped as a second person), then merge the second into the first.
  await page.evaluate(async () => {
    const st = await fetch('/api/state').then((r) => r.json());
    const n = (st.target_faces || []).length;
    await fetch('/api/target/group', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ groups: Array.from({ length: n }, (_, i) => (i === n - 1 ? 1 : 0)) }) });
  });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.getByTestId('studio').waitFor();
  await expect.poll(() => people.count(), { timeout: 10_000 }).toBeGreaterThan(1);
  const before = await people.count();
  await people.nth(1).locator('button[aria-pressed]').first().click();
  await expect(people.nth(1).getByTestId('person-tools')).toBeVisible();
  page.once('dialog', (d) => d.accept());
  const merge = people.nth(1).getByTestId('person-merge');
  const firstName = (await people.first().getByTestId('person-name').textContent()).trim();
  await merge.selectOption({ label: firstName });
  await expect(people).toHaveCount(before - 1);
  expect(v2.bodiesOf('/api/target/group').at(-1)).toHaveProperty('groups');

  await page.getByTestId('people-autogroup').click();
  await expect.poll(() => v2.posted('/api/target/autocluster')).toBe(true);

  page.once('dialog', (d) => { expect(d.message()).toContain('replaces the people'); d.accept(); });
  await page.getByTestId('people-autocapture').click();
  await expect.poll(() => v2.bodiesOf('/api/target/auto_capture')).toEqual([expect.objectContaining({ replace: true })]);
  expect(v2.errors).toEqual([]);
});

test('settings → tuning: TRT cache clear, recognition apply, benchmark run + apply, auto-tune start', async ({ page, v2 }) => {
  await page.goto('#/settings?tab=tuning', { waitUntil: 'domcontentloaded' });
  await page.getByTestId('tuning').waitFor({ timeout: 20_000 });

  await expect(page.getByTestId('trt-status')).toBeVisible();
  await expect(page.getByTestId('trt-ns').first()).toBeVisible();
  page.once('dialog', (d) => d.dismiss());
  await page.getByTestId('trt-clear-all').click();
  await page.waitForTimeout(200);
  expect(v2.posted('/api/trt_cache/clear')).toBe(false);
  page.once('dialog', (d) => d.accept());
  await page.getByTestId('trt-clear-stale').click();
  await expect.poll(() => v2.bodiesOf('/api/trt_cache/clear')).toEqual([{ scope: 'stale' }]);

  const model = page.getByTestId('recog-model');
  const options = await model.locator('option').evaluateAll((os) => os.map((o) => o.value));
  await model.selectOption(options.find((v) => v !== 'default') || options[0]);
  await page.getByTestId('recog-apply').click();
  await expect.poll(() => v2.posted('/api/recognition/set')).toBe(true);

  await page.getByTestId('bench-start').click();
  await expect.poll(() => v2.posted('/api/benchmark/start')).toBe(true);
  await expect(page.getByTestId('bench-report')).toBeVisible({ timeout: 90_000 });
  const apply = page.getByTestId('bench-apply');
  if (await apply.isEnabled()) {
    await apply.click();
    await expect.poll(() => v2.posted('/api/benchmark/apply')).toBe(true);
  }

  await page.getByTestId('autotune-start').click();
  await expect.poll(() => v2.posted('/api/autotune/start')).toBe(true);
  await page.evaluate(() => fetch('/api/autotune/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {}));
  expect(v2.errors).toEqual([]);
});

test('live: start opens the camera after the engine confirms it, shows the preview, and stops', async ({ page, v2 }) => {
  await openScreen(page, 'studio');
  if ((await page.locator('[data-testid="cast"] .tile').count()) === 0) {
    const { PNG_1X1 } = await import('./fixtures.js');
    await page.locator('[data-testid="cast"] input[type=file]').setInputFiles({ name: 'face.png', mimeType: 'image/png', buffer: PNG_1X1 });
    await expect(page.locator('[data-testid="cast"] .tile').first()).toBeVisible();
  }
  await page.keyboard.press('Alt+0');
  await page.getByTestId('live').waitFor();
  await page.getByTestId('live-cam').fill('1');
  await page.getByTestId('live-start').click();
  await expect(page.getByTestId('live-stop')).toBeVisible({ timeout: 10_000 });
  expect(v2.bodiesOf('/api/livecam/start')[0]).toMatchObject({ cam_number: 1, resolution: '1280x720' });
  await expect(page.locator('[data-testid="live-preview"] img')).toHaveAttribute('src', /\/api\/livecam\/frame\?t=\d+/);
  await page.getByTestId('live-stop').click();
  await expect(page.getByTestId('live-start')).toBeVisible();
  expect(v2.posted('/api/livecam/stop')).toBe(true);
  expect(v2.errors.filter((e) => !/livecam\/frame/.test(e))).toEqual([]);
});

test('timeline: the EDL export sends each person\'s presence (In..Out, 0-based exclusive) and downloads the file', async ({ page, v2 }) => {
  const { readyStudio } = await import('./fixtures.js');
  await page.route((u) => u.pathname === '/api/target/timeline_tracks', (r) => r.fulfill({ json: {
    scene_cuts: [], presence_source: 'face_bank',
    presence: [
      { person_id: 'p-ann', name: 'Ann', intervals: [[1, 40], [100, 160]] },
      { person_id: 'p-bob', name: 'Bob', intervals: [[30, 90]] },
    ],
  } }));
  await readyStudio(page, v2);
  await expect(page.getByTestId('cutlist-export')).toContainText('2 people');
  const download = page.waitForEvent('download');
  await page.getByTestId('export-edl').click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(/\.edl$/);
  const body = v2.bodiesOf('/api/timeline/export_edl')[0];
  const segs = body.document.timeline.face_segments;
  // [1,40] -> in 0 / out 40; [30,90] -> 29/90; [100,160] -> 99/160 (clipped to the target's Out).
  expect(segs[0]).toMatchObject({ in: 0, out: 40, face_id: 'Ann' });
  expect(segs.find((s) => s.face_id === 'Bob')).toMatchObject({ in: 29, out: 90 });
  expect(body.fps).toBeGreaterThan(0);
  expect(v2.errors).toEqual([]);
});

test('command palette: Ctrl+K filters by subsequence, Enter runs, a blocked action says why', async ({ page, v2 }) => {
  await openScreen(page, 'studio');
  await page.keyboard.press('Control+k');
  const input = page.getByTestId('palette-input');
  await expect(input).toBeFocused();
  await input.fill('tun');
  await expect(page.getByTestId('palette-item').first()).toContainText('Tuning');
  await input.press('Enter');
  await expect(page.getByTestId('palette')).toHaveCount(0);
  await expect(page.getByTestId('tuning')).toBeVisible();
  // While Settings is open, a palette jump to another Settings page switches the tab.
  await page.keyboard.press('Control+k');
  await input.fill('storage');
  await input.press('Enter');
  await expect(page.getByTestId('storage')).toBeVisible();
  // Nothing is rendering, so "Stop render" is listed but refuses with its reason.
  await page.keyboard.press('Control+k');
  await input.fill('stop rend');
  await input.press('Enter');
  await expect(page.locator('.toast').last()).toContainText('no render is running');
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('palette')).toHaveCount(0);
  expect(v2.errors).toEqual([]);
});

test('render deck: a labelled time estimate, and the clip advisor returns only changes and applies them', async ({ page, v2 }) => {
  const { readyStudio } = await import('./fixtures.js');
  await readyStudio(page, v2);
  // The mock is shared by every test: put back what the advisor changes.
  const before = await page.evaluate(() => fetch('/api/settings').then((r) => r.json()));
  const restore = () => page.evaluate((b) => fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ face_detector_threshold: b.face_detector_threshold, detector_engine: b.detector_engine }) }), before).catch(() => {});
  await expect(page.getByTestId('run-estimate')).toContainText(/about .+ for \d+ frames/);
  await expect(page.getByTestId('run-estimate')).toContainText(/rough guess|learned from/);
  expect(v2.bodiesOf('/api/runtime_estimate')[0]).toHaveProperty('swap_model');
  await page.getByTestId('advisor-run').click();
  await expect(page.getByTestId('advisor-result')).toBeVisible();
  const recs = page.getByTestId('advisor-rec');
  const n = await recs.count();
  if (n > 0) {
    await page.getByTestId('advisor-apply-all').click();
    await expect.poll(() => Object.keys(Object.assign({}, ...v2.bodiesOf('/api/settings'))).length).toBeGreaterThan(0);
  }
  expect(v2.bodiesOf('/api/advisor')[0]).toHaveProperty('settings');
  // Let the debounced settings save land, then restore.
  await page.waitForTimeout(700);
  await restore();
  expect(v2.errors).toEqual([]);
});
