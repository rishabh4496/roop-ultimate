/**
 * The 3-Step Pipeline Pill (1 Pre-Scan -> 2 Range Lock -> 3 Render & Mux), driven by the frames /ws/telemetry carries.
 *
 *   * model: each backend stage state maps to a step state; Stage 4 (not in `stages`) keeps step 3 active as "Muxing"
 *     until the report arrives; a failed report fails step 3; a stage that did not run is "not needed", not done;
 *   * the culled share is shown only when the backend measured one (null is not 0); fps only beside step 3;
 *   * markup: an ordered list in step order, aria-current on the active step, a glyph AND a word per step, a polite
 *     live region that carries a CHANGE of step, nothing at all while idle;
 *   * the pill and its model make NO request: no api/useApi import, no fetch, no timer (so the 15 s API deadline
 *     cannot apply to it, however long a stage runs).
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

globalThis.window = globalThis;

const here = dirname(fileURLToPath(import.meta.url));
const src = (rel) => readFileSync(join(here, '..', 'src', rel), 'utf8');
const report = JSON.parse(readFileSync(join(here, 'pipeline-report.fixture.json'), 'utf8'));

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const P = await import('../src/components/faceswap/pillModel.js');
const { PipelinePillView } = await import('../src/components/faceswap/PipelinePill.jsx');
const store = await import('../src/store/telemetryStore.js');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const html = (el) => renderToStaticMarkup(el).replace(/<!-- -->/g, '');
const text = (h) => h.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

const S1 = 'STAGE_1_SPARSE_PRESCAN';
const S2 = 'STAGE_2_ACTIVE_PREPASS';
const S3 = 'STAGE_3_MAIN_SWAP';
const stage = (state, extra = {}) => ({ state, stage_progress_pct: state === 'DONE' ? 100 : 0, skipped_frames_pct: 46.7, ...extra });
const pipe = (a, b, c, summary = {}) => ({ stage: null, stages: { [S1]: a, [S2]: b, [S3]: c }, summary });
const states = (m) => m.steps.map((s) => s.state).join(',');

console.log('── the model ──────────────────────────────────────────────');
ok('idle: no stages -> no pill', P.pillModel({ stage: null, stages: {}, summary: {} }) === null && P.pillModel(undefined) === null);
{
  const m = P.pillModel(pipe(stage('RUNNING', { stage_progress_pct: 40 }), stage('PENDING'), stage('PENDING')));
  ok('stage 1 running: step 1 active with its percentage, the rest wait', states(m) === 'active,pending,pending' && m.steps[0].detail === '40.0%' && m.activeN === 1, states(m));
}
{
  const m = P.pillModel(pipe(stage('DONE'), stage('RUNNING', { stage_progress_pct: 12.5 }), stage('PENDING')));
  ok('stage 2 running: Range Lock is step 2', states(m) === 'done,active,pending' && m.activeN === 2 && m.steps[1].label === 'Range Lock');
}
{
  const m = P.pillModel(pipe(stage('DONE'), stage('DONE'), stage('RUNNING', { stage_progress_pct: 62 })));
  ok('stage 3 running: step 3 is Render & Mux with a percentage', states(m) === 'done,done,active' && m.steps[2].label === 'Render & Mux' && m.steps[2].detail === '62.0%');
}
{
  const m = P.pillModel(pipe(stage('DONE'), stage('DONE'), stage('DONE')), { processing: true });
  ok('stage 3 done while the job still runs: step 3 stays active as "Muxing" (stage 4 is not in `stages`)',
    states(m) === 'done,done,active' && m.steps[2].detail === 'Muxing' && /muxing audio/.test(m.announce), JSON.stringify(m.steps[2]));
}
{
  const m = P.pillModel(pipe(stage('DONE'), stage('DONE'), stage('DONE')), { processing: false });
  ok('stage 3 done and the job over (no report): done, not muxing', states(m) === 'done,done,done');
}
{
  const m = P.pillModel(pipe(stage('DONE'), stage('DONE'), stage('DONE'), report), { processing: false });
  ok('the finished report makes step 3 done', states(m) === 'done,done,done' && m.announce === 'Render finished', m.announce);
  const bad = JSON.parse(JSON.stringify(report));
  bad.ok = false;
  bad.stages.STAGE_4_FINALIZE = { ...bad.stages.STAGE_4_FINALIZE, ok: false, state: 'FAILED' };
  const mb = P.pillModel(pipe(stage('DONE'), stage('DONE'), stage('DONE'), bad), { processing: false });
  ok('a failed Stage 4 fails step 3 and says so', states(mb) === 'done,done,failed' && /failed/.test(mb.announce), mb.announce);
}
{
  const skipped = stage('SKIPPED', { skipped_frames_pct: null });
  const m = P.pillModel({ stage: null, stages: { [S1]: skipped, [S2]: { ...stage('RUNNING'), skipped_frames_pct: null }, [S3]: { ...stage('PENDING'), skipped_frames_pct: null } }, summary: {} });
  ok('a stage that did not run is "not needed", never done', m.steps[0].state === 'skipped' && m.steps[0].word === 'not needed');
  ok('no measured cull -> null, not 0', m.culled === null && P.culledText({ stages: { [S1]: skipped } }) === '');
}
{
  const m = P.pillModel(pipe(stage('RESTORED'), stage('RESTORED'), stage('RUNNING')));
  ok('a resumed stage reads as resumed from checkpoint', m.steps[0].state === 'restored' && m.steps[0].word === 'resumed from checkpoint');
}
{
  const m = P.pillModel(pipe(stage('FAILED'), stage('PENDING'), stage('PENDING')));
  ok('a failed stage announces the failing step', m.steps[0].state === 'failed' && m.announce === 'Step 1 of 3, Pre-Scan, failed', m.announce);
}
ok('culled share is formatted', P.culledText(pipe(stage('DONE'), stage('RUNNING'), stage('PENDING'))) === '46.7% culled');
ok('culled share is clamped to 0-100', P.culledPct(pipe(stage('DONE', { skipped_frames_pct: 140 }), stage('DONE'), stage('DONE'))) === 100);
{
  const m3 = P.pillModel(pipe(stage('DONE'), stage('DONE'), stage('RUNNING')));
  const m1 = P.pillModel(pipe(stage('RUNNING'), stage('PENDING'), stage('PENDING')));
  ok('fps shows beside step 3 only, current rate first', P.fpsText({ fps_now: 14.39, fps: 12 }, m3) === '14.4 fps' && P.fpsText({ fps_now: 14.39 }, m1) === '');
  ok('fps falls back to the run average, and is empty with neither', P.fpsText({ fps_now: null, fps: 12.04 }, m3) === '12.0 fps' && P.fpsText({}, m3) === '');
}

console.log('── the markup ─────────────────────────────────────────────');
const view = (p, processing = true) => html(React.createElement(PipelinePillView, { model: P.pillModel(p, { processing }), processing }));
{
  const h = view(pipe(stage('DONE'), stage('RUNNING', { stage_progress_pct: 30 }), stage('PENDING')));
  const order = ['Pre-Scan', 'Range Lock', 'Render &amp; Mux'].map((l) => h.indexOf(l));
  ok('three steps in order inside one ordered list', order.every((i) => i > 0) && order[0] < order[1] && order[1] < order[2] && /<ol [^>]*aria-label="Render pipeline, 3 steps"/.test(h), h.slice(0, 300));
  const current = h.match(/<li[^>]*aria-current="step"[^>]*>/g) || [];
  ok('exactly one step carries aria-current="step", and it is the active one (Range Lock)', current.length === 1 && /data-step="rangelock"/.test(current[0]), JSON.stringify(current));
  ok('every step prints a word for its state (not colour alone)', /done/.test(text(h)) && /running/.test(text(h)) && /waiting/.test(text(h)), text(h));
  ok('the arrows between steps are hidden from assistive tech', (h.match(/aria-hidden="true" class="text-muted text-micro select-none">→/g) || []).length === 2);
  ok('a polite live region carries the step change', /role="status" aria-live="polite"[^>]*>Step 2 of 3, Range Lock, 46\.7% of frames skipped</.test(h), h.slice(-300));
  ok('the fast numbers are hidden from the screen reader', /aria-hidden="true"[^>]*><span[^>]*data-testid="pill-culled"/.test(h), h.slice(-500));
}
ok('idle renders nothing at all', view({ stage: null, stages: {}, summary: {} }) === '');

console.log('── the telemetry path (frame -> store -> pill) ────────────');
store.setPipelineTelemetry({ stage: S3, stages: pipe(stage('DONE'), stage('DONE'), stage('RUNNING', { stage_progress_pct: 10 })).stages, summary: {} });
{
  const m = P.pillModel(store.useTelemetryStore.getState().pipeline);
  ok('a /ws/telemetry frame in the store drives the model', states(m) === 'done,done,active' && m.steps[2].detail === '10.0%');
}
store.setPipelineTelemetry({ progress: 0.5, desc: 'x' });
ok('a later frame without stage data cannot clear it', P.pillModel(store.useTelemetryStore.getState().pipeline) !== null);

console.log('── viseme mouth custody on the pill ────────────────────────');
{
  const run = (o) => ({ mouth_mode: 'audio_driven_sync', mouth_fallback: '', lipsync_ms: null, ...o });
  ok('lip-sync ms shows while audio-driven sync is timed', P.lipsyncText(run({ lipsync_ms: 4.2 })) === 'Lip-sync 4.2 ms', P.lipsyncText(run({ lipsync_ms: 4.2 })));
  ok('nothing before it has been timed', P.lipsyncText(run({ lipsync_ms: null })) === '');
  ok('nothing in other modes, even with a stale number', P.lipsyncText(run({ mouth_mode: 'source_mouth_restore', lipsync_ms: 4 })) === '' && P.lipsyncText(run({ mouth_mode: 'off', lipsync_ms: 4 })) === '');
  ok('no figure is shown while the mouth fell back', P.lipsyncText(run({ lipsync_ms: 4, mouth_fallback: 'x' })) === '');
  ok('zero is a real timing, not "missing"', P.lipsyncText(run({ lipsync_ms: 0 })) === 'Lip-sync 0.0 ms');
  ok('the fallback is stated with its reason', /Lip-sync off: no usable driving audio \(target has no audio stream\)\. Using Source mouth restore\./.test(
    P.mouthFallbackText(run({ mouth_fallback: 'no usable driving audio (target has no audio stream)' }))));
  ok('no fallback text without a fallback, or outside audio mode', P.mouthFallbackText(run({})) === '' && P.mouthFallbackText(run({ mouth_mode: 'off', mouth_fallback: 'x' })) === '');
  ok('missing run is safe', P.lipsyncText(undefined) === '' && P.mouthFallbackText(null) === '');
  const empty = store.useTelemetryStore.getState().run;
  ok('the store keeps the four new fields (FAST_FIELDS is a whitelist)', ['lipsync_ms', 'mouth_mode', 'mouth_fallback', 'audio_offset_ms'].every((k) => k in empty && store.FAST_FIELDS.includes(k)));
  store.setRunTelemetry({ lipsync_ms: 3.5, mouth_mode: 'audio_driven_sync', mouth_fallback: '', audio_offset_ms: -120 });
  const live = store.useTelemetryStore.getState().run;
  ok('a frame carrying them reaches the run slice', live.lipsync_ms === 3.5 && live.mouth_mode === 'audio_driven_sync' && live.audio_offset_ms === -120);
  ok('...and the pill text follows', P.lipsyncText(live) === 'Lip-sync 3.5 ms');
}

console.log('── it makes no request ────────────────────────────────────');
for (const rel of ['components/faceswap/PipelinePill.jsx', 'components/faceswap/pillModel.js']) {
  const code = src(rel).replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  ok(`${rel}: no api / useApi import, fetch, XHR, WebSocket or timer`,
    !/from\s+['"][^'"]*(\/api|useApi|transport)[^'"]*['"]/.test(code) && !/\b(fetch|XMLHttpRequest|WebSocket|setInterval|setTimeout)\b/.test(code), rel);
}

console.log(failures === 0 ? '\nall pipeline-pill checks passed' : `\n${failures} pipeline-pill check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
