/**
 * UI overhaul Stage 8 (mission-control screens): the pure logic behind the render queue, the rolling
 * 60-second telemetry, the thread-scaling chart, the Outputs quad compare and metadata pills, the persistent
 * checkpoint badge, and the Studio Launchpad's deep links and readiness verdict.
 *
 *   node .render-check/run-module.mjs render-ui-overhaul-stage8.mjs
 */
import {
  METRICS, pushSample, readingText, sampleOf, sparkGeometry, summarize, trimWindow, WINDOW_MS, yBounds,
} from '../src/components/telemetry/rollingSeries.js';
import {
  STATE_BADGE, clamp01, fmtEta, fmtFps, gaugeFraction, gaugeMax, jobCard, queueTotals, rackGroups, reorderIds, stateBadge,
} from '../src/components/processing/queueModel.js';
import { scalingGroups, scalingVerdict } from '../src/components/diagnostics/scalingModel.js';
import {
  clampRatio, compareSelection, fmtRenderDuration, fmtRenderFps, fmtResolution, pillsFor, quadLayout,
} from '../src/components/gallery/mediaMeta.js';
import { BADGE_TONE, checkpointBadge, parseMissing } from '../src/components/history/checkpointBadge.js';
import { MODULE_LINKS, agoLabel, readinessOf } from '../src/components/home/launchpadModel.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;

// ── rolling 60-second window ────────────────────────────────────────────────
console.log('rolling window');
{
  const now = 1_000_000;
  const samples = [
    { t: now - 61_000, gpu: 10, vram: 1, cpu: 5, frame: 40 },   // falls out
    { t: now - 59_000, gpu: 20, vram: 2, cpu: 6, frame: 41 },
    { t: now, gpu: 30, vram: 3, cpu: 7, frame: null },
  ];
  const kept = trimWindow(samples, now);
  ok('samples older than 60 s are dropped', kept.length === 2 && kept[0].t === now - 59_000);
  ok('WINDOW_MS is 60 000', WINDOW_MS === 60_000);
  const pushed = pushSample(samples, { t: now + 1000, gpu: 31, vram: 3, cpu: 7, frame: 42 }, WINDOW_MS);
  ok('pushSample drops the stale head as the window moves', pushed.every((s) => s.t >= now + 1000 - WINDOW_MS));
}

console.log('sampling honesty');
{
  const s = sampleOf({ gpu_util: undefined, vram_used: 6.5, cpu_percent: 12 }, { frame_ms: 0 }, 5);
  ok('an unreported GPU utilisation is null, not 0', s.gpu === null);
  ok('a zero frame time is "no frame", not 0 ms', s.frame === null);
  ok('VRAM passes through in GB', s.vram === 6.5);
  ok('a non-numeric CPU reading is null', sampleOf({ cpu_percent: 'n/a' }, {}, 1).cpu === null);
  ok('a live frame time is kept', sampleOf({}, { frame_ms: 41.2 }, 1).frame === 41.2);
}

console.log('summary and geometry');
{
  const samples = [
    { t: 0, gpu: 10 }, { t: 1000, gpu: null }, { t: 2000, gpu: 30 }, { t: 3000, gpu: 50 },
  ];
  const st = summarize(samples, 'gpu');
  ok('summary ignores gaps', st.count === 3 && st.min === 10 && st.max === 50);
  ok('summary mean is over real readings', near(st.mean, 30));
  ok('summary last is the newest real reading', st.last === 50);
  ok('no readings => not reported', summarize([{ t: 0, gpu: null }], 'gpu').reported === false);
  const gpu = METRICS.find((m) => m.key === 'gpu');
  ok('GPU y-axis is fixed 0..100', eq(yBounds(gpu, samples), { lo: 0, hi: 100 }));
  const vram = METRICS.find((m) => m.key === 'vram');
  ok('VRAM y-axis grows to fit the data', yBounds(vram, [{ t: 0, vram: 8 }]).hi > 8);
  const geo = sparkGeometry(samples, gpu, { now: 3000, width: 240, height: 56 });
  ok('a gap lifts the pen (two moveto commands)', (geo.d.match(/M/g) || []).length === 2);
  ok('the newest point sits at the right edge', near(geo.last.x, 240, 0.01));
  ok('readingText formats the unit', readingText(vram, 7.456) === '7.46 GB' && readingText(gpu, null) === '—');
}

// ── queue mission control ───────────────────────────────────────────────────
console.log('states');
{
  const states = ['QUEUED', 'PREPARING', 'PROCESSING', 'PAUSE_REQUESTED', 'PAUSED', 'COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED', 'RECOVERABLE'];
  ok('all ten backend states have a badge', states.every((s) => STATE_BADGE[s] && STATE_BADGE[s].tone));
  ok('no state falls back to muted by accident', states.filter((s) => STATE_BADGE[s].tone === 'muted').length === 1);
  ok('INTERRUPTED and RECOVERABLE are never styled as failures',
    STATE_BADGE.INTERRUPTED.tone !== 'bad' && STATE_BADGE.RECOVERABLE.tone !== 'bad');
  ok('FAILED is the failure tone', STATE_BADGE.FAILED.tone === 'bad');
  ok('stateBadge labels with the queue vocabulary', stateBadge('PROCESSING').label === 'Running');
}

console.log('speed and ETA');
{
  ok('fmtEta under an hour is m:ss', fmtEta(75) === '1:15');
  ok('fmtEta over an hour is h:mm:ss', fmtEta(3725) === '1:02:05');
  ok('fmtEta unknown is null', fmtEta(0) === null && fmtEta(null) === null && fmtEta(-4) === null);
  ok('fmtFps keeps one decimal below 10', fmtFps(4.56) === '4.6');
  ok('fmtFps rounds above 10', fmtFps(23.4) === '23');
  ok('fmtFps unknown is null', fmtFps(0) === null && fmtFps(undefined) === null);
  ok('gaugeMax rounds up past the reading', gaugeMax(23) === 30);
  ok('gaugeMax unknown is null', gaugeMax(0) === null);
  ok('gaugeFraction stays inside the dial', gaugeFraction(23) > 0 && gaugeFraction(23) < 1 && gaugeFraction(0) === 0);
  ok('clamp01 keeps nulls null', clamp01(null) === null && clamp01(1.4) === 1 && clamp01(-2) === 0);
}

const job = (over = {}) => ({
  id: 'j1', label: 'Clip A', target_name: 'clipA.mp4', source_name: 'Ann', state: 'QUEUED', payload: { swap_model: 'hyperswap', enhancer: 'gpen' },
  progress: { fraction: 0.4, frames_done: 40, frames_total: 100, fps: 12.5, eta_s: 30, phase: 'Swapping' }, ...over,
});

console.log('job cards');
{
  const q = jobCard(job({ state: 'PROCESSING' }), 0, { currentId: 'j1', queueRunning: true });
  ok('the running job is live (speedometer reads)', q.live === true && q.isCurrent === true);
  ok('a running job cannot be dragged or removed', q.canDrag === false && q.actions.remove === false);
  ok('a running job offers cancel, confirmed', q.actions.cancel === true && q.actions.cancelNeedsConfirm === true);
  ok('stage percent comes from the server fraction', q.pct === 40 && q.framesDone === 40 && q.framesTotal === 100);
  ok('fps and ETA come through', q.fps === 12.5 && q.etaS === 30);
  const paused = jobCard(job({ state: 'PAUSED', progress: { paused: true, fps: 9 } }), 1, { currentId: 'j1', queueRunning: true });
  ok('a paused job is not live', paused.live === false);
  const done = jobCard(job({ state: 'COMPLETED' }), 2, {});
  ok('a finished job can be retried and duplicated', done.actions.retry && done.actions.duplicate);
  ok('a finished job cannot be cancelled', done.actions.cancel === false);
  const queued = jobCard(job({ state: 'QUEUED', progress: {} }), 3, {});
  ok('a queued job reports no speed, no ETA, no bar', queued.fps === null && queued.etaS === null && queued.pct === null);
  ok('a queued job can be dragged and removed', queued.canDrag && queued.actions.remove);
  ok('a job with a project offers Load checkpoint', jobCard(job({ project_id: 'p-1' }), 0, {}).actions.loadCheckpoint === true);
  ok('a job without a project does not', jobCard(job(), 0, {}).actions.loadCheckpoint === false);
  ok('the card names its model and enhancer', q.swapModel === 'hyperswap' && q.enhancer === 'gpen');
  const fallback = jobCard({ id: 'x', state: 'QUEUED' }, 4, {});
  ok('a bare job still gets a label and a number', fallback.label === 'Job 5' && fallback.number === 5);
}

console.log('reorder and totals');
{
  const jobs = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
  ok('drop a before c moves a to front of c', eq(reorderIds(jobs, 'a', 'c', false), ['b', 'a', 'c']));
  ok('drop a after c moves a to the end', eq(reorderIds(jobs, 'a', 'c', true), ['b', 'c', 'a']));
  ok('a no-op drop returns null', reorderIds(jobs, 'b', 'b') === null && reorderIds(jobs, 'a', 'b', false) === null);
  ok('an unknown id returns null', reorderIds(jobs, 'z', 'a') === null);
  const cards = [
    jobCard(job({ state: 'PROCESSING' }), 0, { currentId: 'j1', queueRunning: true }),
    jobCard(job({ id: 'j2', state: 'QUEUED' }), 1, {}),
    jobCard(job({ id: 'j3', state: 'FAILED' }), 2, {}),
    jobCard(job({ id: 'j4', state: 'RECOVERABLE' }), 3, {}),
    jobCard(job({ id: 'j5', state: 'COMPLETED' }), 4, {}),
  ];
  const t = queueTotals(cards);
  ok('totals count running, waiting, finished and attention', t.running === 1 && t.waiting === 2 && t.done === 1 && t.attention === 1);
  ok('totals count resumable', t.resumable === 1);
}

console.log('worker rack');
{
  const groups = rackGroups([{
    group: 'g1', target_name: 'clipA.mp4', count: 3, completed: 2,
    segments: [
      { index: 2, job_id: 'j2', state: 'COMPLETED', exists: true, fraction: 1, frames: 100, start_frame: 101, end_frame: 200 },
      { index: 1, job_id: 'j1', state: 'COMPLETED', exists: true, fraction: 1, frames: 100, start_frame: 1, end_frame: 100 },
      { index: 3, job_id: 'j3', state: 'PROCESSING', exists: false, fraction: 0.5, frames: 100, start_frame: 201, end_frame: 300 },
    ],
    can_join: false, join_blocked: '1 slice(s) not finished',
    joined: { exists: false },
  }]);
  const g = groups[0];
  ok('lanes are sorted by slice index', g.lanes.map((l) => l.index).join(',') === '1,2,3');
  ok('lanes carry their job id for re-render', g.lanes[0].jobId === 'j1');
  ok('only the processing lane is active', g.lanes.filter((l) => l.active).length === 1 && g.running);
  ok('a split that is not complete is waiting, not ready', g.joinState === 'running' && g.canJoin === false);
  ok('no joined file means the remux is unknown', g.remux.status === 'unknown');
  const joined = rackGroups([{
    group: 'g2', count: 1, completed: 1,
    segments: [{ index: 1, state: 'COMPLETED', exists: true, fraction: 1 }],
    can_join: true, joined: { exists: true, url: '/outputs/x_joined.mp4', result: { source_audio: true, verify: { ok: true } } },
  }])[0];
  ok('a verified join with source audio is confirmed', joined.remux.status === 'confirmed' && joined.remux.text.includes('audio'));
  ok('a joined file is not offered a second join', joined.canJoin === false && joined.joinState === 'joined');
  const failed = rackGroups([{ group: 'g3', count: 1, segments: [], joined: { exists: true, result: { verify: { ok: false, reason: 'short by 2 s' } } } }])[0];
  ok('a failed duration check is reported as failed, not confirmed', failed.remux.status === 'failed' && failed.remux.text.includes('short by 2 s'));
}

// ── diagnostics scaling chart ───────────────────────────────────────────────
console.log('thread scaling');
{
  const result = { stages: [
    { key: 'detect', label: 'Detector', scaling: [
      { n: 4, calls_s: 80, stable: true, spread_pct: 2 },
      { n: 1, calls_s: 40, stable: true, spread_pct: 1 },
      { n: 2, calls_s: 60, stable: false, spread_pct: 14 },
    ] },
    { key: 'none', label: 'No sweep', scaling: [] },
  ] };
  const groups = scalingGroups(result);
  ok('stages with no sweep rows are dropped', groups.length === 1 && groups[0].key === 'detect');
  ok('bars are sorted by thread count', groups[0].bars.map((b) => b.n).join(',') === '1,2,4');
  ok('each bar is normalised to the stage best', near(groups[0].bars.find((b) => b.n === 1).share, 0.5) && groups[0].bars.find((b) => b.best).n === 4);
  ok('the noisy bar is flagged', groups[0].bars.find((b) => b.n === 2).stable === false);
  ok('the verdict names the best thread count', scalingVerdict(groups[0]).includes('Best at 4'));
  ok('no data gives no groups', scalingGroups({}).length === 0);
}

// ── gallery ─────────────────────────────────────────────────────────────────
console.log('compare selection');
{
  ok('one selected is not comparable', compareSelection(1).ok === false && compareSelection(1).reason.length > 0);
  ok('two to four are comparable', [2, 3, 4].every((n) => compareSelection(n).ok));
  ok('five is refused with the reason', compareSelection(5).ok === false && compareSelection(5).reason.includes('up to 4'));
}
console.log('quad layout');
{
  ok('two tiles sit in one row', eq(quadLayout(2), { cols: 2, rows: 1, count: 2 }));
  ok('three and four tiles sit in a 2x2', eq(quadLayout(3), { cols: 2, rows: 2, count: 3 }) && eq(quadLayout(4), { cols: 2, rows: 2, count: 4 }));
  ok('more than four is clamped to four', quadLayout(9).count === 4);
  ok('the playhead clamps to 0..1', clampRatio(1.7) === 1 && clampRatio(-1) === 0 && clampRatio(NaN) === 0);
}
console.log('metadata pills');
{
  ok('resolution formats as W×H', fmtResolution(1920, 1080) === '1920×1080');
  ok('resolution is null until reported', fmtResolution(0, 0) === null && fmtResolution(undefined, 720) === null);
  ok('render duration formats seconds, minutes and hours', fmtRenderDuration(41) === '41 s' && fmtRenderDuration(187) === '3m 07s' && fmtRenderDuration(3720) === '1h 02m');
  ok('render fps is one decimal and labelled as render', fmtRenderFps(12.34) === '12.3 fps');
  const entry = { fps: 9.8, duration_s: 125, settings: { swap_model: 'hyperswap' } };
  const pills = pillsFor(entry, { width: 1280, height: 720 });
  ok('all four pills appear when every value is known', pills.map((p) => p.key).join(',') === 'res,fps,model,dur');
  ok('an unrecorded run shows no pills it does not know', pillsFor(null, null).length === 0);
  ok('a run without a resolution omits only the resolution', pillsFor(entry, null).map((p) => p.key).join(',') === 'fps,model,dur');
  ok('the model pill carries the swap model', pills.find((p) => p.key === 'model').value === 'hyperswap');
}

// ── checkpoint badge ────────────────────────────────────────────────────────
console.log('checkpoint badge');
{
  ok('a recoverable record is green', checkpointBadge({ recoverable: true }).tone === 'green');
  const missing = checkpointBadge({
    recoverable: false,
    validation_errors: ['source 2 is missing: C:\\clips\\ann.mp4', 'target is missing: no path recorded'],
  });
  ok('only missing inputs are amber', missing.tone === 'amber' && missing.relink.length === 2);
  ok('a missing source keeps its index (0-based) and path', missing.relink[0].role === 'source' && missing.relink[0].index === 1 && missing.relink[0].path === 'C:\\clips\\ann.mp4');
  ok('a target with no recorded path is relinkable with an empty path', missing.relink[1].role === 'target' && missing.relink[1].path === '');
  ok('the amber label counts the paths', missing.label === '2 missing paths');
  const changed = checkpointBadge({ recoverable: false, validation_errors: ['source 1 changed (size mismatch)'] });
  ok('a changed file is red and never relinkable', changed.tone === 'red' && changed.relink.length === 0);
  const mixed = checkpointBadge({ recoverable: false, validation_errors: ['source 1 is missing: /a.mp4', 'checkpoint output x.mp4 is missing: /o/x.mp4'] });
  ok('a missing partial output is red even with a missing input', mixed.tone === 'red');
  ok('an unexplained refusal is red', checkpointBadge({ recoverable: false, validation_errors: [] }).tone === 'red');
  ok('parseMissing ignores other reasons', parseMissing('source 1 changed (size mismatch)') === null);
  ok('every tone has a class', ['green', 'amber', 'red'].every((t) => BADGE_TONE[t]));
}

// ── launchpad ──────────────────────────────────────────────────────────────
console.log('launchpad');
{
  const TABS = ['home', 'faceswap', 'batch', 'processing', 'facemgr', 'extras', 'gallery', 'history', 'settings'];
  ok('every deep link targets a real App tab', MODULE_LINKS.every((m) => TABS.includes(m.tab)));
  ok('no deep link is duplicated', new Set(MODULE_LINKS.map((m) => m.tab)).size === MODULE_LINKS.length);
  const ready = readinessOf({ 'a.onnx': { status: 'ok' }, 'b.onnx': { status: 'ok' } });
  ok('all present is ready', ready.tone === 'ready' && ready.label.includes('All 2'));
  const blocked = readinessOf({ 'a.onnx': { status: 'ok' }, 'b.onnx': { status: 'missing' }, 'c.onnx': { status: 'corrupt' } });
  ok('missing or corrupt weights are named and counted', blocked.tone === 'blocked' && blocked.blocking.length === 2);
  ok('a missing model is listed with its status', blocked.blocking.some((b) => b.file === 'b.onnx' && b.status === 'missing'));
  ok('no model report is unknown, not ready', readinessOf(null).tone === 'unknown');
  ok('one blocking model reads singular', readinessOf({ 'b.onnx': { status: 'missing' } }).label === '1 model needs attention');
  ok('two blocking models read plural', readinessOf({ 'b.onnx': { status: 'missing' }, 'c.onnx': { status: 'corrupt' } }).label === '2 models need attention');
  ok('ago labels read as minutes and days', agoLabel(1000, 1000 * 1000 + 5 * 60 * 1000) === '5 min ago');
  ok('ago of nothing is empty', agoLabel(0) === '');
}

console.log(failures === 0 ? '\nall stage 8 checks passed' : `\n${failures} stage 8 check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
if (failures) process.exitCode = 1;
