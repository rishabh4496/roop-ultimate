/**
 * The per-job pipeline report in the UI, driven by a REAL report.
 *
 * `pipeline-report.fixture.json` is the `report` of an actual render (docs/perf/stage9/verify_stage4_report.json,
 * run `cfr_3stage_trim`: a trimmed 3-stage render of the Phase 5 clip through the production path, with
 * Stage 4 verification), copied here unedited. The component is rendered to markup with react-dom/server, and
 * the store is fed the frames `/ws/telemetry` carries, so what is checked is the path a browser takes:
 *
 *   frame -> setPipelineTelemetry -> useTelemetryStore -> PipelineHud / PipelineSummaryCard -> text
 *
 *   * every number the backend measured appears, formatted, in the section it belongs to;
 *   * a metric the backend did not measure is an em dash, never "0";
 *   * a failed Stage 4 says FAILED and lists its problems; the headline says FAILED;
 *   * the speedup is always labelled an estimate;
 *   * an idle backend renders nothing; a frame without stage data cannot clear the report;
 *   * the live rows show while a run is in progress and give way to the report when it ends.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

globalThis.window = globalThis;

const here = dirname(fileURLToPath(import.meta.url));
const report = JSON.parse(readFileSync(join(here, 'pipeline-report.fixture.json'), 'utf8'));

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const helpers = await import('../src/components/faceswap/reportFormat.js');
const { PipelineHud, PipelineHudView, PipelineSummaryCard } = await import('../src/components/faceswap/PipelineReport.jsx');
const store = await import('../src/store/telemetryStore.js');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const html = (el) => renderToStaticMarkup(el).replace(/<!-- -->/g, '');
const text = (h) => h.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

const S = helpers;
const st = report.stages;

// ── the formatters ───────────────────────────────────────────────────────────
ok('null is an em dash, not zero', helpers.fmtSecs(null) === '—' && helpers.fmtCount(undefined) === '—'
  && helpers.fmtPct(null) === '—' && helpers.fmtGb(null) === '—');
ok('zero is shown as zero (a measured 0 is a finding)', helpers.fmtCount(0) === '0' && helpers.fmtSecs(0) === '0.00s');
ok('minutes are minutes', helpers.fmtSecs(125) === '2m 05s');

// ── a finished report, section by section ───────────────────────────────────
const sections = helpers.reportSections(report);
const byKey = Object.fromEntries(sections.map((s) => [s.key, s]));
const rowOf = (key, label) => (byKey[key]?.rows || []).find((r) => r.label === label)?.value;

ok('four sections, in stage order', sections.map((s) => s.key).join() === S.STAGE_ORDER.join(),
  sections.map((s) => s.key).join());
ok('Stage 1 wall clock is the backend\'s', rowOf(S.STAGE_1, 'Wall clock') === helpers.fmtSecs(st[S.STAGE_1].wall_s));
ok('Stage 1 sampled frames', rowOf(S.STAGE_1, 'Sampled frames')?.startsWith(helpers.fmtCount(st[S.STAGE_1].sampled_frames)));
ok('Stage 1 worker utilization names its pool',
  rowOf(S.STAGE_1, 'Worker pool utilization')?.includes(`${st[S.STAGE_1].workers} workers`));
ok('Stage 2 cap.set / cap.grab counts',
  rowOf(S.STAGE_2, 'cap.set() seeks') === helpers.fmtCount(st[S.STAGE_2].set_calls)
  && rowOf(S.STAGE_2, 'cap.grab() frames walked') === helpers.fmtCount(st[S.STAGE_2].grab_calls));
ok('Stage 2 verified target frames', rowOf(S.STAGE_2, 'Verified target frames') === helpers.fmtCount(st[S.STAGE_2].verified_target_frames));
ok('Stage 2 says how often the detector ran, and at what interval',
  rowOf(S.STAGE_2, 'Detector calls') === '233 of 900 frames (every 4th)', rowOf(S.STAGE_2, 'Detector calls'));
ok('Stage 3 passthrough and GPU frames',
  rowOf(S.STAGE_3, 'Zero-GPU passthrough frames') === helpers.fmtCount(st[S.STAGE_3].passthrough_frames)
  && rowOf(S.STAGE_3, 'GPU frames (swap / enhance)') === helpers.fmtCount(st[S.STAGE_3].gpu_frames));
ok('Stage 3 peak VRAM', rowOf(S.STAGE_3, 'Peak VRAM') === helpers.fmtGb(st[S.STAGE_3].peak_vram_gb));
ok('Stage 4 verification passed on a verified report', rowOf(S.STAGE_4, 'Verification') === 'passed');
ok('the speedup is marked an estimate', helpers.speedupText(report.speedup).includes('(estimate)'),
  helpers.speedupText(report.speedup));

// ── unmeasured stays unmeasured ──────────────────────────────────────────────
const sparse = JSON.parse(JSON.stringify(report));
sparse.stages[S.STAGE_1] = { state: 'SKIPPED', wall_s: null, sampled_frames: null, worker_utilization_pct: null };
sparse.stages[S.STAGE_2].grab_calls = null;
const sparseSections = Object.fromEntries(helpers.reportSections(sparse).map((s) => [s.key, s]));
ok('a skipped stage shows dashes, not zeros',
  sparseSections[S.STAGE_1].rows.filter((r) => r.label !== 'Active ranges' && r.label !== 'Frames skipped (no target)')
    .every((r) => r.value === '—'), JSON.stringify(sparseSections[S.STAGE_1].rows));
ok('a missing counter is a dash', sparseSections[S.STAGE_2].rows.find((r) => r.label === 'cap.grab() frames walked').value === '—');
{
  const restored = JSON.parse(JSON.stringify(report));
  Object.assign(restored.stages[S.STAGE_2], { detect_interval: null, detector_calls: null, scan_frames: null });
  const row = helpers.reportSections(restored).find((s) => s.key === S.STAGE_2).rows.find((r) => r.label === 'Detector calls');
  ok('a Stage 2 restored from a checkpoint (it did not run) shows a dash, not a count', row.value === '—', row.value);
  const every = JSON.parse(JSON.stringify(report));
  Object.assign(every.stages[S.STAGE_2], { detect_interval: 1, detector_calls: 900, scan_frames: 900 });
  const plain = helpers.reportSections(every).find((s) => s.key === S.STAGE_2).rows.find((r) => r.label === 'Detector calls');
  ok('at interval 1 the row does not claim an interval', plain.value === '900 of 900 frames', plain.value);
}

// ── a failed Stage 4 ─────────────────────────────────────────────────────────
const failed = JSON.parse(JSON.stringify(report));
failed.ok = false;
failed.stages[S.STAGE_4] = { ...failed.stages[S.STAGE_4], ok: false, state: 'FAILED', problems: ['the audio mux failed', 'frame count 59 != the 60 the trim implies'] };
const failedHtml = html(React.createElement(PipelineSummaryCard, { report: failed }));
ok('a failed Stage 4 says FAILED in the headline and the section', (failedHtml.match(/FAILED/g) || []).length >= 2);
ok('every problem is listed', failedHtml.includes('the audio mux failed') && failedHtml.includes('frame count 59'));

// ── the card ─────────────────────────────────────────────────────────────────
const card = html(React.createElement(PipelineSummaryCard, { report }));
const cardText = text(card);
ok('the card names every stage', S.STAGE_ORDER.every((k) => cardText.includes(S.STAGE_LABEL[k])));
ok('the card shows the frame total and the speedup estimate',
  cardText.includes(`${helpers.fmtCount(report.total_frames)} frames`) && cardText.includes('estimate'));
ok('the card carries the backend\'s caveat about the speedup', cardText.includes('Speedup:'));
ok('an empty report renders the explanation, not a table of dashes',
  text(html(React.createElement(PipelineSummaryCard, { report: {} }))).includes('No finished run'));
ok('a collapsed card (history rows) is closed',
  !/<details open/.test(html(React.createElement(PipelineSummaryCard, { report, defaultOpen: false }))));

// ── the store and the HUD ────────────────────────────────────────────────────
const view = (pipeline) => html(React.createElement(PipelineHudView, { pipeline }));
const idle = view(store.useTelemetryStore.getState().pipeline);
ok('an idle backend renders nothing', idle === '', idle);
// (zustand's useSyncExternalStore answers a server render with the store's INITIAL state, so PipelineHud itself
// is only checked for being the thin store -> view adapter; the view carries the behaviour.)
ok('PipelineHud is exported as the store adapter', typeof PipelineHud === 'function');

store.setPipelineTelemetry({ progress: 0.3, desc: 'x' });                 // a frame with no stage keys
ok('a frame without stage data leaves the store alone',
  Object.keys(store.useTelemetryStore.getState().pipeline.stages).length === 0);

const liveStages = {
  [S.STAGE_1]: { state: 'DONE', stage_progress_pct: 100, wall_s: 2.1 },
  [S.STAGE_2]: { state: 'RUNNING', stage_progress_pct: 40, wall_s: 5.5 },
  [S.STAGE_3]: { state: 'PENDING', stage_progress_pct: 0, wall_s: null },
};
store.setPipelineTelemetry({ stage: S.STAGE_2, stages: liveStages, summary: {} });
const live = view(store.useTelemetryStore.getState().pipeline);
const liveText = text(live);
ok('while a run is live the three stages show', S.STAGE_ORDER.slice(0, 3).every((k) => liveText.includes(S.STAGE_LABEL[k])), liveText);
ok('the running stage shows its progress and elapsed time', liveText.includes('40.0%') && liveText.includes('5.50s'));
ok('no finished report is shown while running', !live.includes('data-testid="pipeline-report"'));

const before = store.useTelemetryStore.getState().pipeline;
store.setPipelineTelemetry({ stage: S.STAGE_2, stages: JSON.parse(JSON.stringify(liveStages)), summary: {} });
ok('an identical frame does not replace the store object (no re-render)',
  store.useTelemetryStore.getState().pipeline === before);

store.setPipelineTelemetry({ stage: null, stages: liveStages, summary: report });
const done = view(store.useTelemetryStore.getState().pipeline);
ok('when the run ends the report replaces the live rows',
  done.includes('data-testid="pipeline-report"') && !done.includes('data-testid="pipeline-live"'));

store.setPipelineTelemetry({ progress: 1, desc: 'Done' });                 // a later plain progress frame
ok('a later frame without stage data cannot clear the report',
  !!store.useTelemetryStore.getState().pipeline.summary.stages);

// ── the per-stage HUD fields, from a fixture the REAL telemetry writers produced ─────────────────────────────
// (tools/gen_hud_fixture.py drives stage_telemetry + procmgr_runtime._prof; a renamed backend field fails here.)
{
  const hud = JSON.parse(readFileSync(join(here, 'pipeline-hud.fixture.json'), 'utf8'));
  const sum = hud.summary;
  const cardTextOf = (r) => text(html(React.createElement(PipelineSummaryCard, { report: r })));
  const secs = helpers.reportSections(sum);
  const val = (key, label) => (secs.find((x) => x.key === key)?.rows || []).find((r) => r.label === label)?.value;

  ok('Stage 1 shows keyframes per second and the scene cuts', val(S.STAGE_1, 'Keyframes sampled / s') === '10.0 /s'
    && val(S.STAGE_1, 'Scene cuts found') === '3', JSON.stringify(secs[0].rows));
  ok('Stage 2 splits flow-tracked from neural-detected frames',
    val(S.STAGE_2, 'Neural-detected frames') === '25' && val(S.STAGE_2, 'Flow-tracked frames') === '75 (75.0%)',
    JSON.stringify(secs[1].rows));
  ok('Stage 2 pre-pass speed', val(S.STAGE_2, 'Pre-pass speed') === '20.0 fps', val(S.STAGE_2, 'Pre-pass speed'));
  const cells = helpers.opsCells(sum.stages[S.STAGE_3].ops);
  const byKeyCell = Object.fromEntries(cells.map((c) => [c.key, c]));
  ok('Stage 3 ops are in pipeline order', cells.map((c) => c.key).join() === 'decode,warp,model,restore,blend,encode');
  ok('Stage 3 labels the TensorRT model column TRT and the nvenc encoder NVENC',
    byKeyCell.model.label === 'TRT' && byKeyCell.encode.label === 'NVENC', JSON.stringify(cells.map((c) => c.label)));
  ok('warp and blend are labelled with the CPU backend they really ran on',
    byKeyCell.warp.backend === 'cv2 (CPU)' && byKeyCell.blend.backend === 'cv2 (CPU)');
  ok('a column that never ran is a dash, not 0 ms', byKeyCell.restore.value === '—' && byKeyCell.restore.calls === null);
  ok('measured ops carry real milliseconds', /^\d/.test(byKeyCell.model.value) && byKeyCell.model.value.endsWith('ms'),
    byKeyCell.model.value);
  ok('the ops line is the brief\'s Decode | Warp | TRT | Blend | NVENC readout',
    /^Decode .* \| Warp .* \| TRT .* \| Restore — \| Blend .* \| NVENC /.test(helpers.opsLine(sum.stages[S.STAGE_3].ops)),
    helpers.opsLine(sum.stages[S.STAGE_3].ops));
  ok('the finished card prints each op row', cardTextOf(sum).includes('TRT per call'));
  ok('no ops block means no ops rows (a report from before this field)',
    helpers.opsCells(undefined).length === 0 && helpers.opsLine(null) === '');

  // live frame: store -> HUD
  store.setPipelineTelemetry({ stage: S.STAGE_3, stages: hud.live.stages, summary: {} });
  const liveHtml = view(store.useTelemetryStore.getState().pipeline);
  const lt = text(liveHtml);
  ok('the live HUD shows each stage rate', lt.includes('20.0 fps') && lt.includes('samples/s'), lt);
  ok('the live HUD shows Stage 1 cuts and the Stage 2 flow split',
    lt.includes('3 scene cuts') && lt.includes('75 flow') && lt.includes('25 neural'), lt);
  ok('the live HUD shows the Stage 3 ops strip', liveHtml.includes('data-testid="stage3-ops"') && lt.includes('TRT '));
  const noOps = JSON.parse(JSON.stringify(hud.live.stages));
  delete noOps[S.STAGE_3].ops;
  store.setPipelineTelemetry({ stage: S.STAGE_3, stages: noOps, summary: {} });
  ok('a Stage 3 with no ops block renders no strip',
    !view(store.useTelemetryStore.getState().pipeline).includes('data-testid="stage3-ops"'));

  // OpenVINO aux offload
  ok('no aux_offload prints nothing', helpers.auxOffloadText(null) === '' && helpers.auxOffloadText({ sessions: [] }) === ''
    && !cardTextOf(sum).includes('OpenVINO aux offload'));
  const withAux = { ...sum, aux_offload: { sessions: [
    { model_key: 'scrfd', device: 'NPU', device_name: 'Intel AI Boost', precision: 'int8', ms_per_call: 3.21 }] } };
  ok('an active OpenVINO session is named with its device and precision',
    cardTextOf(withAux).includes('scrfd on NPU (Intel AI Boost) int8 · 3.21 ms/call'), cardTextOf(withAux));
}

console.log(failures === 0 ? '\nall pipeline-report checks passed' : `\n${failures} pipeline-report check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
