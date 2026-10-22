/**
 * The "Benchmark Pipeline Providers" result in the UI, driven by a REAL `provider_matrix`.
 *
 * `provider-matrix.fixture.json` is the `result` of an actual `bench.run_benchmark('providers')` on the 4070
 * (docs/perf/providers/), copied unedited. What is checked:
 *
 *   * every measured number appears, formatted, in its row; the fastest measured arm of a stage is marked;
 *   * an arm / backend the machine cannot run shows "unavailable" WITH the backend's reason and no number;
 *   * a stage the benchmark could not time is listed, not dropped;
 *   * an empty / absent matrix renders nothing, and the benchmark state machine reads the status endpoint right.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

globalThis.window = globalThis;

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, 'provider-matrix.fixture.json'), 'utf8'));
const matrix = fixture.provider_matrix;

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const model = await import('../src/components/faceswap/providerMatrix.js');
const { ProviderMatrixView } = await import('../src/components/faceswap/ProviderMatrixView.jsx');
const { fmtMs } = await import('../src/components/faceswap/reportFormat.js');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const html = (el) => renderToStaticMarkup(el).replace(/<!-- -->/g, '');
const text = (h) => h.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

// ── the model ────────────────────────────────────────────────────────────────
const rows = model.stageRows(matrix);
ok('one table row per measured stage', rows.length === matrix.stages.length && rows.length > 0, String(rows.length));
for (const [i, r] of rows.entries()) {
  const src = matrix.stages[i];
  ok(`${src.key}: primary / CPU come straight from the backend`,
    r.primary === fmtMs(src.primary_ms) && r.cpu === (src.cpu_error ? 'error' : fmtMs(src.cpu_ms)),
    JSON.stringify(r));
}
ok('an OpenVINO number is a dash when the arm did not run (no Intel accelerator here)',
  rows.every((r) => r.openvino === '—') === matrix.stages.every((s) => s.openvino_ms == null));

const arms = model.armCards(matrix);
ok('four arms, in the backend\'s order', arms.map((a) => a.key).join() === matrix.arms.map((a) => a.key).join() && arms.length === 4);
for (const a of arms) {
  const src = matrix.arms.find((x) => x.key === a.key);
  if (src.available) ok(`${a.key}: available arm shows ms/frame`, a.available && a.value === `${fmtMs(src.serial_ms_frame)} / frame`);
  else ok(`${a.key}: unavailable arm shows its reason and no number`,
    !a.available && a.value === 'unavailable' && a.sub === src.reason && !/\d+(\.\d+)? ms/.test(a.value), a.sub);
}
ok('at most one arm is marked fastest', arms.filter((a) => a.fastest).length <= 1);
ok('the CV-CUDA / NVOFA arm is never available on this machine', arms.find((a) => a.key === 'trt_cvcuda_nvofa').available === false);

const warp = model.backendRows(matrix.warp_blend);
const flow = model.backendRows(matrix.optical_flow);
ok('CV-CUDA is unavailable with the reason, not a number',
  warp.find((b) => b.key === 'cvcuda').value === 'unavailable' && warp.find((b) => b.key === 'cvcuda').reason.length > 10);
ok('NVOFA is unavailable with the reason, not a number',
  flow.find((b) => b.key === 'nvof').value === 'unavailable' && flow.find((b) => b.key === 'nvof').reason.length > 10);
ok('the measured host backends carry milliseconds',
  warp.find((b) => b.key === 'cv2_cpu').available && flow.find((b) => b.key === 'cv').available);

// ── the markup ───────────────────────────────────────────────────────────────
const markup = html(React.createElement(ProviderMatrixView, { matrix }));
const t = text(markup);
ok('the view renders every stage name', matrix.stages.every((s) => t.includes(s.stage)), t.slice(0, 300));
ok('the view prints unavailable arms with their reasons', matrix.arms.filter((a) => !a.available).every((a) => t.includes(a.reason)));
ok('the view carries the backend\'s caveat about serial figures', t.includes('not a pipeline fps'));
ok('data-available mirrors the arm', (markup.match(/data-available="true"/g) || []).length === arms.filter((a) => a.available).length);

// A stage the per-stage measurement could not time is listed.
const withMissing = JSON.parse(JSON.stringify(matrix));
withMissing.unmeasured = [{ stage: 'Mask · XSeg', reason: 'invalid throughput: 0.0' }];
const t2 = text(html(React.createElement(ProviderMatrixView, { matrix: withMissing })));
ok('an unmeasured stage is listed with its reason', t2.includes('Not measured: Mask · XSeg') && t2.includes('invalid throughput: 0.0'), t2.slice(0, 200));
ok('a matrix with nothing unmeasured has no such list', !t.includes('Not measured'));

// An arm that errored on a stage says "error" in that cell, with the text on hover.
const errored = JSON.parse(JSON.stringify(matrix));
errored.stages[0].cpu_error = 'RuntimeException: boom';
errored.stages[0].cpu_ms = null;
ok('a stage whose CPU arm errored reads "error", not a dash or zero', model.stageRows(errored)[0].cpu === 'error'
  && model.stageRows(errored)[0].note.includes('boom'));

ok('an absent or empty matrix renders nothing', html(React.createElement(ProviderMatrixView, { matrix: null })) === ''
  && html(React.createElement(ProviderMatrixView, { matrix: {} })) === '');
ok('the model tolerates garbage', model.stageRows(undefined).length === 0 && model.armCards({}).length === 0
  && model.backendRows(null).length === 0 && model.unmeasuredRows({ unmeasured: 'x' }).length === 0);

// ── the state machine over the status endpoint ───────────────────────────────
ok('no state is idle', model.benchPhase(null) === 'idle' && model.benchPhase({}) === 'idle');
ok('running wins', model.benchPhase({ running: true, result: fixture }) === 'running');
ok('an error is failed', model.benchPhase({ running: false, error: 'RuntimeError: x' }) === 'failed');
ok('a finished matrix is done', model.benchPhase({ running: false, result: fixture }) === 'done');
ok('a result from another profile is not mistaken for ours', model.benchPhase({ running: false, result: { profile: 'quick' } }) === 'idle');
ok('a cancelled run says so', model.benchPhase({ running: false, status_msg: 'Cancelled' }) === 'cancelled');

console.log(failures === 0 ? '\nall provider-matrix checks passed' : `\n${failures} provider-matrix check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
