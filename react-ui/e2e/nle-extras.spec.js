import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openTab } from './helpers.js';

// Stage 4, Extras: the same NLE over a LOCAL video, in a real browser.
//
// Extras never uploads the clip until Apply, so everything here is built client-side: the filmstrip is
// stills seeked out of a <video>, the waveform is Web Audio's decode of the file, the frame rate is read
// from presentation timestamps. To check those against ground truth the clip has to be real, so it is
// generated with ffmpeg: 4 s, 25 fps, 320x180 (VP8 + Opus: what Playwright's Chromium can decode), with a
// 440 Hz tone gated ON for the first half of every second and silent for the second half.
// Without ffmpeg on the PATH the file skips, loudly, rather than passing on nothing.

test.use({ viewport: { width: 1600, height: 1200 } });

const FPS = 25;
const SECONDS = 4;
const FRAMES = FPS * SECONDS; // 100

let clip = null;
let skipReason = '';

test.beforeAll(() => {
  try {
    execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' });
  } catch {
    skipReason = 'ffmpeg is not on the PATH: cannot generate the fixture clip';
    return;
  }
  const dir = mkdtempSync(join(tmpdir(), 'roop-nle-extras-'));
  // The first video encoder this ffmpeg build has, of the codecs Playwright's (open-source) Chromium decodes.
  const attempts = [
    { ext: 'webm', v: ['-c:v', 'libvpx', '-b:v', '300k'], a: ['-c:a', 'libopus', '-b:a', '64k'] },
    { ext: 'webm', v: ['-c:v', 'libvpx-vp9', '-b:v', '300k', '-deadline', 'realtime', '-cpu-used', '8'], a: ['-c:a', 'libopus', '-b:a', '64k'] },
    { ext: 'webm', v: ['-c:v', 'libaom-av1', '-crf', '40', '-b:v', '0', '-cpu-used', '8'], a: ['-c:a', 'libopus', '-b:a', '64k'] },
    { ext: 'webm', v: ['-c:v', 'libsvtav1', '-crf', '40'], a: ['-c:a', 'libopus', '-b:a', '64k'] },
    { ext: 'ogv', v: ['-c:v', 'libtheora', '-q:v', '5'], a: ['-c:a', 'libvorbis', '-q:a', '3'] },
  ];
  const errors = [];
  for (const { ext, v, a } of attempts) {
    const out = join(dir, `beats.${ext}`);
    try {
      execFileSync('ffmpeg', [
        '-y', '-loglevel', 'error',
        '-f', 'lavfi', '-i', `testsrc2=size=320x180:rate=${FPS}`,
        // 440 Hz gated on for the first half of every second. Seconds 0 and 2 are full scale (the lane paints
        // those rose: clipping), seconds 1 and 3 are at 0.6 (teal): both colours and the gating are checkable.
        '-f', 'lavfi', '-i', `aevalsrc=sin(2*PI*440*t)*lt(mod(t\\,1)\\,0.5)*(0.6+0.4*lt(mod(t\\,2)\\,1)):s=48000:d=${SECONDS}`,
        '-t', String(SECONDS), '-g', '25', '-pix_fmt', 'yuv420p', ...v, ...a, out,
      ], { stdio: 'pipe' });
      if (existsSync(out)) { clip = out; break; }
    } catch (err) {
      errors.push(`${v[1]}: ${String(err.stderr || err.message).trim().split('\n').pop()}`);
    }
  }
  if (!clip) skipReason = `ffmpeg has no encoder Chromium can decode (${errors.join('; ')})`;
});

test.beforeEach(({}, testInfo) => { // eslint-disable-line no-empty-pattern
  testInfo.skip(!!skipReason, skipReason);
});

const NLE = '[data-testid="nle-timeline"]';
const colorCount = (canvas, rgb, tol = 48) => canvas.evaluate((c, [r, g, b, t]) => {
  const { data } = c.getContext('2d').getImageData(0, 0, c.width, c.height);
  let n = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] > 40 && Math.abs(data[i] - r) < t && Math.abs(data[i + 1] - g) < t && Math.abs(data[i + 2] - b) < t) n++;
  }
  return n;
}, [...rgb, tol]);

async function openExtras(page) {
  await openTab(page, 'extras');
  await page.locator('input[type="file"]').first().setInputFiles(clip);
  await page.locator(NLE).waitFor();
  await page.getByTestId('extras-timeline').scrollIntoViewIfNeeded();
  return page.locator(NLE);
}

/** The frame rate is measured while the clip plays; until then it is assumed (30) and flagged. Frame-exact checks wait for it. */
const waitFps = (page) => expect.poll(
  async () => (await page.getByTestId('nle-ruler').locator('xpath=preceding-sibling::div').first().textContent()) || '',
  { timeout: 15_000, message: 'frame rate measured from the playing video' },
).toMatch(/^Ruler\s*25 fps$/);

const readoutFrame = async (page) => Number(((await page.getByTestId('nle-readout-frame').textContent()) || '').match(/#(\d+)/)?.[1]);
const videoState = (page) => page.evaluate(() => {
  const v = document.querySelector('.extras-media-viewport video');
  return { t: v.currentTime, paused: v.paused, rate: v.playbackRate, loop: v.loop, dur: v.duration };
});
const areaBox = (page) => page.getByTestId('nle-lane-area').boundingBox();
const rulerBox = (page) => page.getByTestId('nle-ruler').boundingBox();

test.describe('Extras timeline', () => {
  test('mounts the NLE for a video, without the swap-pipeline lanes, and says what In/Out does', async ({ page }) => {
    await openExtras(page);
    await expect(page.getByTestId('extras-timeline')).toBeVisible();
    for (const id of ['nle-step-prescan', 'nle-batch-matrix-btn', 'lane-health', 'lane-kf-editor', 'nle-face-tracks']) {
      await expect(page.getByTestId(id), `${id} has no data source in Extras and must not be shown`).toHaveCount(0);
    }
    for (const id of ['lane-v1-filmstrip', 'lane-a1-waveform', 'lane-segments', 'nle-transport', 'nle-minimap', 'nle-ruler']) {
      await expect(page.getByTestId(id)).toBeVisible();
    }
    await expect(page.getByTestId('lane-segments-header')).toContainText('Markers');
    // In/Out here only loops the preview: /api/extras/apply has no trim, and the toolbar must say so
    await expect(page.getByTestId('nle-region-note')).toContainText('preview loop only');
    // the native loop attribute is gone: the timeline owns looping (the same default, whole clip)
    expect((await videoState(page)).loop).toBe(false);
  });

  test('an image gets no timeline', async ({ page }) => {
    await openTab(page, 'extras');
    const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 'base64');
    await page.locator('input[type="file"]').first().setInputFiles({ name: 'x.png', mimeType: 'image/png', buffer: png });
    await page.getByAltText('Input source').waitFor();
    await expect(page.getByTestId('extras-timeline')).toHaveCount(0);
  });

  test('frame count and frame rate come from the file: 100 frames at 25 fps, not an assumed 30', async ({ page }) => {
    await openExtras(page);
    // The rate is read from presentation timestamps as the clip plays: until then it is assumed 30 and flagged.
    await expect(page.getByTestId('nle-readout-tc'), 'flagged approximate until measured').toContainText('≈');
    await waitFps(page);
    await expect(page.getByTestId('nle-readout-frame')).toContainText(`/ ${FRAMES - 1}`);
    await expect(page.getByTestId('nle-readout-tc')).not.toContainText('≈');
  });

  test('step, click-to-seek and the speed pick drive the real <video>, frame-accurately', async ({ page }) => {
    await openExtras(page);
    await waitFps(page);
    await page.getByTestId('transport-play').click();     // autoplay was on: this pauses
    await expect.poll(async () => (await videoState(page)).paused).toBe(true);

    // seek by clicking the ruler: the video lands on the middle of that frame's interval
    const area = await areaBox(page);
    const rb = await rulerBox(page);
    // (0.503, not 0.5: that lands on a 49.5 -> 50 rounding boundary, and Chromium may truncate a fractional clientX)
    await page.mouse.click(area.x + area.width * 0.503, rb.y + rb.height - 8);
    await expect.poll(async () => Math.abs((await readoutFrame(page)) - Math.round(0.503 * (FRAMES - 1)))).toBeLessThanOrEqual(1);
    const f = await readoutFrame(page);
    await expect.poll(async () => Math.abs((await videoState(page)).t - (f + 0.5) / FPS)).toBeLessThan(0.02);

    // step: exactly one frame, and the clock follows
    await page.getByTestId('transport-step-fwd').click();
    await expect.poll(() => readoutFrame(page)).toBe(f + 1);
    await expect.poll(async () => Math.abs((await videoState(page)).t - (f + 1.5) / FPS)).toBeLessThan(0.02);
    await page.getByTestId('transport-start').click();
    await expect.poll(() => readoutFrame(page)).toBe(0);
    await page.getByTestId('transport-end').click();
    await expect.poll(() => readoutFrame(page)).toBe(FRAMES - 1);

    // speed reaches the element, and survives Play
    await page.getByTestId('transport-rate-0.5').click();
    await expect.poll(async () => (await videoState(page)).rate).toBe(0.5);
    await page.getByTestId('transport-start').click();
    await page.getByTestId('transport-play').click();
    await expect.poll(async () => (await videoState(page)).paused).toBe(false);
    expect((await videoState(page)).rate).toBe(0.5);
  });

  test('In / Out loop the preview region; Home / arrows / Space work from the focused timeline', async ({ page }) => {
    const tl = await openExtras(page);
    await waitFps(page);
    await page.getByTestId('transport-play').click();
    await expect.poll(async () => (await videoState(page)).paused).toBe(true);
    await tl.focus();
    await page.keyboard.press('Home');
    await expect.poll(() => readoutFrame(page)).toBe(0);
    for (let i = 0; i < 25; i++) await page.keyboard.press('ArrowRight');
    await expect.poll(() => readoutFrame(page)).toBe(25);
    await page.keyboard.press('i');
    for (let i = 0; i < 25; i++) await page.keyboard.press('ArrowRight');
    await expect.poll(() => readoutFrame(page)).toBe(50);
    await page.keyboard.press('o');
    await expect(page.getByTestId('nle-region-duration')).toHaveText('1s 01f');       // frames 25..50 = 26 frames
    await expect(page.getByTestId('nle-dim-before')).toBeVisible();

    // play with loop on: the playhead must stay inside [25, 50] while the clip would otherwise run to 99
    await expect(page.getByTestId('transport-loop')).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.press('Space');
    await expect.poll(async () => (await videoState(page)).paused).toBe(false);
    const seen = [];
    for (let i = 0; i < 24; i++) { seen.push(await readoutFrame(page)); await page.waitForTimeout(100); }
    await page.keyboard.press('Space');
    await expect.poll(async () => (await videoState(page)).paused).toBe(true);
    expect(Math.min(...seen), 'never plays before In').toBeGreaterThanOrEqual(25);
    expect(Math.max(...seen), 'never plays past Out + 1 frame of latency').toBeLessThanOrEqual(52);
    expect(new Set(seen).size, 'and it actually moved').toBeGreaterThan(4);
  });

  test('the waveform is the real audio: gated on/off with the beat, and full-scale peaks read as clipping', async ({ page }) => {
    await openExtras(page);
    await waitFps(page);
    const canvas = page.getByTestId('nle-waveform-canvas');
    await expect.poll(() => colorCount(canvas, [94, 234, 212], 40), { timeout: 15_000, message: 'decoded waveform (teal)' }).toBeGreaterThan(100);
    // The tone is gated ON for the first half of every second and OFF for the second half. Seconds 0 and 2 are full
    // scale (painted rose = clipping), seconds 1 and 3 are at 0.6 (teal). Columns at the middle of each half-second:
    const cols = await canvas.evaluate((c) => {
      const ctx = c.getContext('2d');
      const w = c.width; const h = c.height;
      const near = (px, rgb) => Math.abs(px[0] - rgb[0]) < 40 && Math.abs(px[1] - rgb[1]) < 40 && Math.abs(px[2] - rgb[2]) < 40;
      const col = (t) => {
        // the lane spans frames 0..99 = 0..3.96 s; scan a few px either side so a one-pixel rounding cannot miss a bar
        const x0 = Math.round((t / 3.96) * w);
        let teal = 0; let rose = 0;
        const { data } = ctx.getImageData(Math.max(0, x0 - 2), 0, 5, h);
        for (let i = 0; i < data.length; i += 4) {
          if (data[i + 3] < 40) continue;
          if (near([data[i], data[i + 1], data[i + 2]], [94, 234, 212])) teal++;
          if (near([data[i], data[i + 1], data[i + 2]], [251, 113, 133])) rose++;
        }
        return { teal, rose };
      };
      return [0.25, 0.75, 1.25, 1.75, 2.25, 2.75].map(col);
    });
    expect(cols[0].rose, 'full-scale beat at 0.25 s reads as clipping (rose)').toBeGreaterThan(5);
    expect(cols[0].teal, '...and not as a normal peak').toBe(0);
    expect(cols[1], 'silent at 0.75 s: no bars at all').toEqual({ teal: 0, rose: 0 });
    expect(cols[2].teal, 'a 0.6-scale beat at 1.25 s is a normal (teal) peak').toBeGreaterThan(5);
    expect(cols[2].rose, '...not clipping').toBe(0);
    expect(cols[3], 'silent at 1.75 s').toEqual({ teal: 0, rose: 0 });
    expect(cols[4].rose, 'full scale again at 2.25 s').toBeGreaterThan(5);
    expect(cols[5], 'silent at 2.75 s').toEqual({ teal: 0, rose: 0 });
  });

  test('the frame rate is measured even when the user pauses at once (a hidden probe, not the visible player)', async ({ page }) => {
    await openExtras(page);
    await page.getByTestId('transport-play').click();               // pause immediately: the in-player estimate cannot run
    await expect.poll(async () => (await videoState(page)).paused).toBe(true);
    await waitFps(page);                                            // ...yet 25 fps is read, within a couple of seconds at most
    await expect(page.getByTestId('nle-readout-tc')).not.toContainText('≈');
    expect((await videoState(page)).paused, 'measuring must not restart the visible player').toBe(true);
    // and the probe element is gone again: nothing is left playing off-screen
    await expect.poll(() => page.evaluate(() => document.querySelectorAll('body > video').length)).toBe(0);
  });

  test('the filmstrip holds real stills of the clip, and the scrub card shows the exact frame', async ({ page }) => {
    await openExtras(page);
    await waitFps(page);
    const film = page.getByTestId('nle-filmstrip-canvas');
    // testsrc2 is colourful: a painted still has many non-grey, non-black pixels; the placeholder is a dark box with a number
    const lit = () => film.evaluate((c) => {
      const { data } = c.getContext('2d').getImageData(0, 0, c.width, c.height);
      let n = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i + 3] > 200 && Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]) > 60) n++;
      return n;
    });
    await expect.poll(lit, { timeout: 20_000, message: 'stills decoded from the local file' }).toBeGreaterThan(2000);

    const cap = await page.getByTestId('nle-playhead-cap').boundingBox();
    const area = await areaBox(page);
    await page.mouse.move(cap.x + cap.width / 2, cap.y + cap.height / 2);
    await page.mouse.down();
    await page.mouse.move(area.x + area.width * 0.6, cap.y + 6, { steps: 6 });
    const tip = page.getByTestId('nle-scrub-tooltip');
    await expect(tip).toBeVisible();
    await expect(page.getByTestId('nle-scrub-thumb')).toHaveAttribute('data-approximate', '0', { timeout: 10_000 });
    await expect.poll(async () => Math.abs(Number(await tip.getAttribute('data-frame')) - Math.round(0.6 * (FRAMES - 1)))).toBeLessThanOrEqual(1);
    await page.mouse.up();
  });

  test('markers use the same list as the key, persist per clip, and Apply is untouched (no trim sent)', async ({ page }) => {
    const applies = [];
    page.on('request', (r) => { if (r.url().includes('/api/extras/apply')) applies.push(r.postData() || ''); });
    const tl = await openExtras(page);
    await waitFps(page);
    await page.getByTestId('transport-play').click();
    await tl.focus();
    await page.keyboard.press('Home');
    for (let i = 0; i < 10; i++) await page.keyboard.press('ArrowRight');
    await page.keyboard.press('m');
    await expect(page.getByTestId('nle-marker-flag')).toHaveCount(1);
    await page.getByTestId('nle-add-marker').click();   // same frame: refused, still one
    await expect(page.getByTestId('nle-marker-flag')).toHaveCount(1);
    const stored = await page.evaluate(() => Object.fromEntries(Object.entries(localStorage).filter(([k]) => k.startsWith('roop_markers_extras_'))));
    expect(Object.values(stored).map((v) => JSON.parse(v))).toEqual([[11]]);   // 1-based, like the workstation's list

    // Apply carries no In/Out: the region is a preview aid, and nothing pretends otherwise
    await page.getByRole('button', { name: 'Apply' }).click();
    await expect.poll(() => applies.length).toBeGreaterThan(0);
    expect(applies[0]).not.toMatch(/trim|start_frame|end_frame|frame_start|frame_end/i);
  });
});
