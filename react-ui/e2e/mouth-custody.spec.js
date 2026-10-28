import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { openTab } from './helpers.js';
import { MOCK_PORT } from './playwright.config.js';

// /__mock/* hooks are not proxied by `vite preview`; they are on the mock's own port (as compile-banner.spec.js does).
const setTargetAudio = (request, has_audio) => request.post(`http://127.0.0.1:${MOCK_PORT}/__mock/target_audio`, { data: { index: 0, has_audio } });

// Viseme Mouth Custody in a real browser, against the mock server (which applies the backend's request rules):
//
//   * the "Audio & Mouth Custody" panel on #/faceswap: three modes, the engine controls only in audio mode, an external
//     track that can be bound and removed, the +-500 ms slip with its nudges;
//   * the audio row under the filmstrip, which shows what will PLAY and carries the same slip (one state, two controls);
//   * the silent-video advisory (and the row hiding for a clip with no audio and no bound track);
//   * the request a render sends: always the mode, the bound track and the slip;
//   * the pill: the Lip-Sync figure, and the fallback stated in words;
//   * the media editor's audio half on #/extras.
//
// The mock invents telemetry and serves placeholder media; what is under test is what the shell does with it.

const AXE = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];
const WAV = Buffer.concat([Buffer.from('RIFF\x24\x00\x00\x00WAVEfmt ', 'latin1'), Buffer.alloc(40)]);

// The mock keeps whatever the UI posts to /api/settings for the life of its process, so one test's bound track or slip would
// be the next test's starting point. Put the mouth-custody keys back to their defaults before every test.
test.beforeEach(async ({ request }) => {
  await request.post('/api/settings', {
    data: { mouth_custody_mode: 'off', lipsync_enabled: false, restore_original_mouth: false, lipsync_audio_source: 'original',
            lipsync_audio_path: null, override_audio_path: null, audio_offset_ms: 0 },
  });
  await setTargetAudio(request, true);
});

async function openPanel(page) {
  await openTab(page, 'faceswap');
  const header = page.getByRole('button', { name: /Audio & Mouth Custody/ });
  await expect(header).toBeVisible();
  if ((await header.getAttribute('aria-expanded')) !== 'true') await header.click();
  const panel = page.getByTestId('mouth-custody-panel');
  await expect(panel).toBeVisible();
  return panel;
}

const mode = (panel, m) => panel.locator(`[role="radio"][data-mode="${m}"]`);

async function bindTrack(page, panel, name = 'dub.wav') {
  await panel.locator('input[type="file"]').setInputFiles({ name, mimeType: 'audio/wav', buffer: WAV });
  await expect(panel.getByTestId('mouth-custody-track')).toContainText(name);
}

test('the panel: three modes, off by default, the engine controls only in audio-driven sync', async ({ page }) => {
  const panel = await openPanel(page);
  await expect(panel.getByRole('radiogroup', { name: 'Mouth custody mode' })).toBeVisible();
  await expect(panel.locator('[role="radio"]')).toHaveCount(3);
  await expect(mode(panel, 'off')).toHaveAttribute('aria-checked', 'true');
  await expect(panel.getByText('Lip-sync Engine / Model')).toHaveCount(0);

  await mode(panel, 'audio_driven_sync').click();
  await expect(mode(panel, 'audio_driven_sync')).toHaveAttribute('aria-checked', 'true');
  await expect(mode(panel, 'off')).toHaveAttribute('aria-checked', 'false');
  await expect(panel.getByText('Lip-sync Engine / Model')).toBeVisible();
  await expect(panel.getByText('Audio/Visual Co-articulation Mode')).toBeVisible();

  await mode(panel, 'source_mouth_restore').click();
  await expect(panel.getByText('Lip-sync Engine / Model')).toHaveCount(0);
  await expect(mode(panel, 'source_mouth_restore')).toHaveAttribute('aria-checked', 'true');
});

test('an external track is bound, shown in the audio row, and removed', async ({ page }) => {
  const panel = await openPanel(page);
  await bindTrack(page, panel);
  const row = page.getByTestId('audio-lane-source');
  await expect(row).toContainText('External:');
  await expect(row).toContainText('dub.wav');
  await expect(page.getByTestId('lane-audio')).toBeVisible();

  await panel.getByRole('button', { name: 'Remove' }).click();
  await expect(panel.getByTestId('mouth-custody-track')).toHaveCount(0);
  await expect(page.getByTestId('audio-lane-source')).toContainText('Target audio');
});

test('the slip: nudges, the extremes, and one state shared by the panel and the audio row', async ({ page }) => {
  const panel = await openPanel(page);
  const panelReadout = panel.getByTestId('mouth-custody-offset');
  await expect(panelReadout).toHaveText('0 ms');
  await expect(page.getByTestId('audio-lane-offset')).toHaveText('0 ms');

  await panel.getByRole('button', { name: 'Nudge audio later by 10 milliseconds' }).click();
  await expect(panelReadout).toHaveText('+10 ms');
  await expect(page.getByTestId('audio-lane-offset')).toHaveText('+10 ms');
  await expect(panel.getByText('Audio plays 10 ms later than the video.')).toBeVisible();

  // the timeline's own nudge moves the same value
  await page.getByRole('button', { name: 'Slip audio earlier by 10 ms' }).click();
  await page.getByRole('button', { name: 'Slip audio earlier by 10 ms' }).click();
  await expect(panelReadout).toHaveText('−10 ms');
  await expect(panel.getByText('Audio plays 10 ms earlier than the video.')).toBeVisible();

  const slider = panel.getByRole('group', { name: 'Audio offset' }).getByRole('slider');
  await slider.focus();
  await slider.press('End');
  await expect(panelReadout).toHaveText('+500 ms');
  await expect(page.getByRole('button', { name: 'Slip audio later by 1 ms' })).toBeDisabled();   // clamped at the limit
  await slider.press('Home');
  await expect(panelReadout).toHaveText('−500 ms');

  await panel.getByRole('button', { name: 'Reset' }).click();
  await expect(panelReadout).toHaveText('0 ms');
  await expect(page.getByTestId('audio-lane-offset')).toHaveText('0 ms');
});

test('a silent clip: audio-driven sync says it will fall back; binding a track clears the warning; the row hides', async ({ page, request }) => {
  await setTargetAudio(request, false);
  try {
    const panel = await openPanel(page);
    await expect(page.getByTestId('lane-audio')).toHaveCount(0);           // no audio, no bound track: no audio row
    await mode(panel, 'audio_driven_sync').click();
    const note = panel.getByTestId('mouth-custody-advisory');
    await expect(note).toBeVisible();
    await expect(note).toContainText('silent');
    await expect(note).toContainText('Source mouth restore');

    await bindTrack(page, panel);
    await expect(panel.getByTestId('mouth-custody-advisory')).toHaveCount(0);
    await expect(page.getByTestId('lane-audio')).toBeVisible();            // a dub gives the silent clip an audio row
  } finally {
    await setTargetAudio(request, true);
  }
});

test('the request carries the mode, the bound track and the slip', async ({ page }) => {
  const panel = await openPanel(page);
  await mode(panel, 'audio_driven_sync').click();
  await bindTrack(page, panel, 'voice.wav');
  await panel.getByRole('button', { name: 'Nudge audio later by 10 milliseconds' }).click();

  const bodies = [];
  await page.route('**/api/preview', async (route) => {
    try { bodies.push(route.request().postDataJSON()); } catch { /* not JSON */ }
    await route.continue();
  });
  // a parameter change asks for a fresh preview; nudging again guarantees one after the route is installed
  await panel.getByRole('button', { name: 'Nudge audio later by 1 millisecond' }).click();
  await expect.poll(() => bodies.length, { timeout: 15_000 }).toBeGreaterThan(0);
  const last = bodies[bodies.length - 1];
  expect(last.mouth_custody_mode).toBe('audio_driven_sync');
  expect(last.override_audio_path).toMatch(/voice\.wav$/);
  expect(last.audio_offset_ms).toBe(11);
});

test('the pill shows the Lip-Sync figure, and says out loud when lip-sync fell back', async ({ page }) => {
  const frame = (extra) => ({
    event: 'progress', processing: true, paused: false, progress: 0.4, desc: 'Processing frame 144 / 360',
    error: '', current_frame: 144, total_frames: 360, fps: 11.8, fps_now: 14.4, frame_ms: 69, eta_s: 7,
    started_at: Date.now() / 1000 - 20, live_seq: 0, engine_state: 'IDLE', engine_label: '', engine_model: '',
    engine_precision: '', engine_since: null, lipsync_ms: null, mouth_mode: '', mouth_fallback: '', audio_offset_ms: 0,
    stage: 'STAGE_3_MAIN_SWAP', summary: {},
    stages: {
      STAGE_1_SPARSE_PRESCAN: { state: 'DONE', stage_progress_pct: 100, skipped_frames_pct: 10 },
      STAGE_2_ACTIVE_PREPASS: { state: 'DONE', stage_progress_pct: 100, skipped_frames_pct: 10 },
      STAGE_3_MAIN_SWAP: { state: 'RUNNING', stage_progress_pct: 40, skipped_frames_pct: 10 },
    },
    ...extra,
  });
  let script = [frame({ event: 'hello', mouth_mode: 'audio_driven_sync', lipsync_ms: 4.2, audio_offset_ms: 120 })];
  await page.route('**/api/progress', async (route) => {
    const res = await route.fetch();
    let body = {};
    try { body = await res.json(); } catch { body = {}; }
    await route.fulfill({ response: res, json: { ...body, processing: true } });
  });
  await page.routeWebSocket(/\/ws\/telemetry/, (ws) => {
    const send = () => ws.send(JSON.stringify(script[script.length - 1]));
    send();
    const timer = setInterval(send, 400);
    ws.onClose(() => clearInterval(timer));
  });
  await openTab(page, 'faceswap');
  await expect(page.getByTestId('pill-lipsync').first()).toHaveText('Lip-sync 4.2 ms');
  await expect(page.getByTestId('pill-mouth-fallback')).toHaveCount(0);

  script = [frame({ mouth_mode: 'audio_driven_sync', mouth_fallback: 'no usable driving audio (target has no audio stream)' })];
  const note = page.getByTestId('pill-mouth-fallback').first();
  await expect(note).toBeVisible();
  await expect(note).toContainText('Lip-sync off: no usable driving audio (target has no audio stream)');
  await expect(note).toContainText('Using Source mouth restore.');
  await expect(page.getByTestId('pill-lipsync').first()).toHaveText('');        // no figure for a stage that did not run
});

test('the panel has no accessibility violations, open, in audio mode with a track bound', async ({ page }) => {
  const panel = await openPanel(page);
  await mode(panel, 'audio_driven_sync').click();
  await bindTrack(page, panel);
  const results = await new AxeBuilder({ page }).withTags(AXE).include('[data-testid="mouth-custody-panel"]').analyze();
  expect(results.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`)).toEqual([]);
});

test('#/extras: the media editor has the audio half only - no mouth mode', async ({ page, request }) => {
  await openTab(page, 'extras');
  await page.locator('input[type="file"][accept="image/*,video/*"]').setInputFiles({
    name: 'clip.mp4', mimeType: 'video/mp4', buffer: Buffer.from('\x00\x00\x00\x18ftypisom', 'latin1'),
  });
  const panel = page.getByTestId('mouth-custody-panel');
  await expect(panel).toBeVisible();
  await expect(panel.getByRole('radiogroup')).toHaveCount(0);
  await panel.locator('input[type="file"]').setInputFiles({ name: 'dub.wav', mimeType: 'audio/wav', buffer: WAV });
  await expect(panel.getByTestId('mouth-custody-track')).toContainText('dub.wav');
  await panel.getByRole('button', { name: 'Nudge audio later by 10 milliseconds' }).click();
  await expect(panel.getByTestId('mouth-custody-offset')).toHaveText('+10 ms');

  // the multipart form carries '' for "no track" and the slip as text, never the string "null"
  const bodies = [];
  await page.route('**/api/extras/apply', async (route) => { bodies.push(route.request().postData() || ''); await route.continue(); });
  await page.getByRole('button', { name: 'Apply', exact: true }).click();
  await expect.poll(() => bodies.length).toBeGreaterThan(0);
  expect(bodies[0]).toContain('name="audio_offset_ms"');
  expect(bodies[0]).toMatch(/name="override_audio_path"\r?\n\r?\n[^\r\n]*dub\.wav/);
  expect(bodies[0]).not.toContain('null');
  void request;
});
