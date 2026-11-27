/**
 * The 3-stage pipeline controls and their live telemetry, on #/faceswap, #/settings and #/processing.
 *
 *   1  the rules       swapper list + groups, the three restorer modes and the settings each one writes (and that the
 *                      write reads back as the same mode), the pre-pass stride clamp
 *   2  the component   PipelineControls renders the four controls with the shared primitives, has no run button, and
 *                      routes each control to the right setting (driven through the element tree's own handlers)
 *   3  the call sites  Face Swap and Settings mount it; the controls it replaced are gone; Settings still binds each
 *                      key literally (test_ui_settings_catalog.py reads them) and takes only the marker props
 *   4  the telemetry   ops labels follow the backend's own words (TRT Graph / Triton Blend only when reported), Stage 2
 *                      names its flow engine only when one was reported, and the canary status says NONE rather than
 *                      PASSED when nothing was recorded
 *
 * Pure rules are imported; the component is rendered to markup with react-dom/server. Nothing here needs a backend.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.location = { origin: 'http://127.0.0.1:1', protocol: 'http:', host: '127.0.0.1:1' };

const here = dirname(fileURLToPath(import.meta.url));
const src = (rel) => readFileSync(join(here, '..', 'src', rel), 'utf8');

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const rules = await import('../src/components/faceswap/pipelineRules.js');
const PipelineControls = (await import('../src/components/faceswap/PipelineControls.jsx')).default;
const report = await import('../src/components/faceswap/reportFormat.js');
const { CanaryPrecisionStatus, PipelineHudView } = await import('../src/components/faceswap/PipelineReport.jsx');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const html = (el) => renderToStaticMarkup(el).replace(/<!-- -->/g, '');
const text = (h) => h.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

console.log('── 1. the rules ───────────────────────────────────────────');
{
  const META = ['realswap', 'inswapper', 'reswapper_256', 'alpha_256', 'hyperswap_256', 'hyperswap_512', 'ghost_1', 'simswap'];
  const groups = rules.swapperOptions(META, 'realswap');
  const flat = groups.flatMap((g) => g.options.map((o) => o.value));
  ok('every model the backend lists is selectable exactly once', META.every((m) => flat.filter((v) => v === m).length === 1), flat.join());
  ok('the five named swappers are labelled the way the brief names them',
    JSON.stringify(groups.find((g) => g.group.startsWith('Native')).options.map((o) => o.label))
      === JSON.stringify(['inswapper_128', 'ReSwapper 256', 'Alpha 256', 'HyperSwap 256', 'HyperSwap 512']));
  ok('the rest are not lost behind the short list', groups.find((g) => g.group === 'Other swappers').options.map((o) => o.value).join() === 'ghost_1,simswap');
  ok('a model the backend no longer lists is still shown (the select never goes blank)',
    rules.swapperOptions(META, 'legacy_x').flatMap((g) => g.options).some((o) => o.value === 'legacy_x'));
  ok('a backend that lists only some of the five offers only those',
    rules.swapperOptions(['realswap', 'alpha_256'], 'alpha_256').find((g) => g.group.startsWith('Native')).options.length === 1);
  ok('no metadata still yields the five (a pre-meta first paint)', rules.swapperOptions(undefined, 'realswap').some((g) => g.options.length === 5));

  const base = { swap_model: 'hyperswap_256', selected_enhancer: 'UltraMax', auto_bypass_restorer_native_swap: false, restore_ultra_frequency_blend: true };
  ok('UltraMax (the shipped restorer) is "custom", not one of the three', rules.restorerModeOf(base) === 'custom');
  ok('custom shows its own restorer as a read-only option',
    rules.restorerOptions(base).some((o) => o.value === 'custom' && o.label === 'Custom · UltraMax') && rules.restorerOptions(base).length === 4);
  for (const mode of ['bypass', 'laplacian', 'full']) {
    for (const start of [base, { ...base, selected_enhancer: 'Restore Ultra' }, { ...base, selected_enhancer: 'None' },
      { ...base, selected_enhancer: 'Restoreformer++', auto_bypass_restorer_native_swap: true }]) {
      const after = { ...start, ...rules.restorerPatch(mode) };
      ok(`mode "${mode}" reads back as "${mode}" from ${start.selected_enhancer}${start.auto_bypass_restorer_native_swap ? ' (bypass on)' : ''}`,
        rules.restorerModeOf(after) === mode, JSON.stringify(after));
    }
  }
  ok('Laplacian Split RF++ writes Restore Ultra with the frequency blend on and the bypass off',
    JSON.stringify(rules.restorerPatch('laplacian')) === JSON.stringify({ selected_enhancer: 'Restore Ultra', restore_ultra_frequency_blend: true, auto_bypass_restorer_native_swap: false }));
  ok('Full RF++ writes plain Restoreformer++ and turns the bypass off',
    JSON.stringify(rules.restorerPatch('full')) === JSON.stringify({ selected_enhancer: 'Restoreformer++', auto_bypass_restorer_native_swap: false }));
  ok('Off / Auto-Bypass only turns the bypass on - it never swaps the restorer out from under the user',
    JSON.stringify(rules.restorerPatch('bypass')) === JSON.stringify({ auto_bypass_restorer_native_swap: true }));
  ok('"custom" is never a write target', Object.keys(rules.restorerPatch('custom')).length === 0);
  ok('Restore Ultra with the frequency blend off is plain RF++', rules.restorerModeOf({ selected_enhancer: 'Restore Ultra', restore_ultra_frequency_blend: false }) === 'full');
  ok('the bypass note is honest about a 128 px swapper',
    /only fires with Alpha 256 or HyperSwap/.test(rules.restorerNote({ swap_model: 'inswapper', selected_enhancer: 'Restoreformer++', auto_bypass_restorer_native_swap: true })));
  ok('the bypass note names the 320 px rule for a swapper it applies to',
    /≤320 px/.test(rules.restorerNote({ swap_model: 'alpha_256', selected_enhancer: 'Restoreformer++', auto_bypass_restorer_native_swap: true })));

  ok('stride is clamped to 1..8', rules.clampStride(0) === 1 && rules.clampStride(99) === 8 && rules.clampStride(3.6) === 4 && rules.clampStride('x') === 1);
  ok('stride K=1 reads every frame, K>1 says it carries by optical flow',
    rules.strideText(1) === 'Detect every frame' && /every 4th frame, carry the rest by optical flow/.test(rules.strideText(4)));
}

console.log('── 2. the component ───────────────────────────────────────');
{
  const props = { values: { swap_model: 'alpha_256', selected_enhancer: 'UltraMax', prepass_detect_interval: 1, temporal_smart_skip: false },
    models: ['realswap', 'inswapper', 'reswapper_256', 'alpha_256', 'hyperswap_256', 'hyperswap_512'] };
  const calls = [];
  const el = React.createElement(PipelineControls, { ...props, onChange: (k, v) => calls.push(['one', k, v]), onChangeMany: (p) => calls.push(['many', p]) });
  const h = html(el);
  const t = text(h);
  for (const label of ['Swapper model', 'Restorer mode', 'Pre-pass optical-flow stride (K)', 'Temporal Smart-Skip (Lossless Static Reuse)']) {
    ok(`label "${label}" is rendered`, t.includes(label));
  }
  ok('the swapper select has option groups', h.includes('<optgroup label="Native 128 - 512 px">') && h.includes('<optgroup label="Other swappers">') === false);
  ok('the restorer select offers the three modes and the custom restorer',
    ['Off / Auto-Bypass ≥256px', 'Laplacian Split RF++', 'Full RF++', 'Custom · UltraMax'].every((x) => t.includes(x)));
  ok('the stride is a 1..8 range input', /<input[^>]*type="range"[^>]*min="1"[^>]*max="8"/.test(h) || /<input[^>]*min="1"[^>]*max="8"[^>]*type="range"/.test(h), h.slice(0, 400));
  ok('the smart-skip toggle is a real checkbox', /<input[^>]*type="checkbox"/.test(h));
  // The shared primitives render a "?" info badge, a MOD marker and a slider's click-to-type VALUE as buttons;
  // nothing else may be one here. The value badge is mouse-only (tabindex -1, aria-hidden) so it adds no tab stop.
  const buttons = [...h.matchAll(/<button([^>]*)>([\s\S]*?)<\/button>/g)].map((m) => ({ attrs: m[1], txt: text(m[2]).trim() }));
  const isValueBadge = (b) => /title="Click to type a value"/.test(b.attrs) && /^-?\d+(\.\d+)?$/.test(b.txt);
  ok('there is no run control in it (every button is an info badge or a slider value; the run bar stays the one Start)',
    buttons.length > 0 && buttons.every((b) => b.txt === '?' || b.txt === 'MOD' || isValueBadge(b)), JSON.stringify(buttons.map((b) => b.txt)));
  ok('a slider value badge is not a tab stop and is hidden from assistive tech (the range input announces its own value)',
    buttons.filter(isValueBadge).length > 0 && buttons.filter(isValueBadge).every((b) => /tabindex="-1"/.test(b.attrs) && /aria-hidden="true"/.test(b.attrs)));
  ok('the stride caption states K and what it means', t.includes('K = 1: Detect every frame.'));

  // Drive the handlers through the element tree PipelineControls returns (it has no hooks of its own).
  const tree = PipelineControls({ ...props, onChange: (k, v) => calls.push(['one', k, v]), onChangeMany: (p) => calls.push(['many', p]) });
  const found = [];
  const walk = (n) => {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(walk); return; }
    if (n.props && typeof n.props.label === 'string' && typeof n.props.onChange === 'function') found.push(n.props);
    if (n.props && n.props.children) walk(n.props.children);
  };
  walk(tree);
  const by = (label) => found.find((f) => f.label === label);
  ok('four controls found in the tree', found.length === 4, found.map((f) => f.label).join());
  by('Swapper model').onChange('hyperswap_512');
  by('Restorer mode').onChange('laplacian');
  by('Restorer mode').onChange('custom');
  by('Pre-pass optical-flow stride (K)').onChange(5);
  by('Pre-pass optical-flow stride (K)').onChange(40);
  by('Temporal Smart-Skip (Lossless Static Reuse)').onChange(true);
  ok('swapper writes swap_model', JSON.stringify(calls[0]) === JSON.stringify(['one', 'swap_model', 'hyperswap_512']), JSON.stringify(calls[0]));
  ok('restorer mode writes its whole patch in ONE update', calls[1][0] === 'many' && calls[1][1].selected_enhancer === 'Restore Ultra' && calls[1][1].restore_ultra_frequency_blend === true);
  ok('choosing the read-only "custom" entry writes nothing', calls.length === 5 && !calls.some((c, i) => i === 2 && c[0] === 'many'), JSON.stringify(calls));
  ok('stride writes prepass_detect_interval, clamped', calls[2][1] === 'prepass_detect_interval' && calls[2][2] === 5 && calls[3][2] === 8, JSON.stringify(calls.slice(2, 4)));
  ok('smart-skip writes temporal_smart_skip', JSON.stringify(calls[4]) === JSON.stringify(['one', 'temporal_smart_skip', true]));

  // Settings hands over bind() results; only the marker props may be used (bind's own value/onChange would override).
  const marked = React.createElement(PipelineControls, { ...props, onChange: () => {}, onChangeMany: () => {},
    bind: { swap_model: { settingKey: 'swap_model', modified: true, onReset: () => {}, value: 'WRONG', onChange: () => { throw new Error('bind.onChange used'); } } } });
  const mh = html(marked);
  ok('a bound control keeps its data-setting key and MOD marker', mh.includes('data-setting="swap_model"') && mh.includes('MOD'));
  ok("bind()'s value does not override the component's own", !mh.includes('WRONG'));
}

console.log('── 3. the call sites ──────────────────────────────────────');
{
  const fs = src('components/FaceSwap.jsx');
  const st = src('components/Settings.jsx');
  ok('Face Swap mounts the shared controls', /<PipelineControls\b/.test(fs));
  ok('Face Swap no longer has the separate swap-model select or the auto-bypass toggle',
    !/label="Swap model"/.test(fs) && !/<Toggle label="Auto-Bypass Restorer/.test(fs));
  ok('Settings mounts the shared controls', /<PipelineControls\b/.test(st));
  ok('Settings no longer has the three rows the section replaced',
    !/label="Pre-pass detection interval/.test(st) && !/label="Auto-Bypass Restorer/.test(st) && !/<Toggle label="Temporal Smart-Skip/.test(st));
  for (const k of ['swap_model', 'selected_enhancer', 'prepass_detect_interval', 'temporal_smart_skip']) {
    ok(`Settings binds ${k} with a literal key (the palette catalog test reads it)`, new RegExp(`bind(?:Toggle)?\\('${k}'`).test(st));
  }
  const pc = src('components/faceswap/PipelineControls.jsx');
  ok('the component takes only the marker props from bind()', /settingKey: b\.settingKey, modified: b\.modified, onReset: b\.onReset/.test(pc));
  ok('the component renders no button', !/<button/i.test(pc));
  ok('body text is on the 12px step (no nano / micro in the component)', !/text-(nano|micro)\b/.test(pc.replace(/className="[^"]*uppercase[^"]*"/g, '')));
}

console.log('── 4. the telemetry ───────────────────────────────────────');
{
  const cell = (ops, key) => report.opsCells(ops).find((c) => c.key === key);
  const ops = { decode: { ms: 2, calls: 10, backend: 'NVDEC' }, warp: { ms: 3, calls: 10, backend: 'cv2 (CPU)' },
    model: { ms: 9, calls: 10, backend: 'TensorrtExecutionProvider' }, restore: null,
    blend: { ms: 4, calls: 10, backend: 'cv2 (CPU)' }, encode: { ms: 1, calls: 10, backend: 'hevc_nvenc' } };
  ok('as shipped: TRT / Blend / NVENC (the blend is cv2 and is not called Triton)',
    report.opsLine(ops) === 'Decode 2.00 ms | Warp 3.00 ms | TRT 9.00 ms | Restore — | Blend 4.00 ms | NVENC 1.00 ms', report.opsLine(ops));
  const graph = { ...ops, model: { ms: 8, calls: 10, backend: 'TensorRT (CUDA graph)' }, blend: { ms: 0.2, calls: 10, backend: 'Triton (fused)' } };
  ok('a CUDA-graph replay is labelled TRT Graph and a fused kernel Triton Blend - only when the backend says so',
    cell(graph, 'model').label === 'TRT Graph' && cell(graph, 'blend').label === 'Triton Blend', report.opsLine(graph));
  ok('the one-line readout carries both', /TRT Graph .* Triton Blend .* NVENC /.test(report.opsLine(graph)));

  ok('flow engines are named from the backend', report.flowEngineLabel('nvof') === 'NVOFA' && report.flowEngineLabel('cv') === 'CPU flow'
    && report.flowEngineLabel('torch') === 'GPU flow' && report.flowEngineLabel(null) === '' && report.flowEngineLabel('mystery') === '');
  const stage = (backend) => ({ state: 'RUNNING', stage_progress_pct: 50, fps: 20, wall_s: 4, neural_detected_frames: 25, flow_tracked_frames: 75, flow_ratio_pct: 75, flow_backend: backend });
  const hud = (backend) => text(html(React.createElement(PipelineHudView, { pipeline: { stage: report.STAGE_2, stages: { [report.STAGE_2]: stage(backend) }, summary: {} } })));
  ok('Stage 2 says "NVOFA tracked" when NVOFA carried the frames', hud('nvof').includes('75.0% NVOFA tracked'), hud('nvof'));
  ok('Stage 2 says plain "flow" when no engine was reported', hud(null).includes('75.0% flow') && !hud(null).includes('NVOFA'), hud(null));

  const none = report.canaryView({}, null);
  ok('no verdicts: NONE, not PASSED', none.state === 'NONE' && none.rows.length === 0);
  const pass = report.canaryView({ configured: 'mixed', effective: 'mixed', canary_verdicts: { xseg: { passed: true, reason: '' }, w600k: { passed: true } } }, null);
  ok('all passed: PASSED with the count', pass.state === 'PASSED' && pass.passed === 2 && pass.failed === 0 && pass.configured === 'mixed');
  const fail = report.canaryView({ canary_verdicts: { xseg: { passed: true }, '1k3d68': { passed: false, reason: 'ssim 0.81 < 0.95' } } }, null);
  ok('one failed verdict: FAILED, failures listed first with the reason', fail.state === 'FAILED' && fail.rows[0].stem === '1k3d68' && fail.rows[0].reason.startsWith('ssim'));
  const unknown = report.canaryView({ canary_verdicts: { xseg: { passed: null } } }, null);
  ok('a skipped canary (passed null) is UNKNOWN and never makes a run look verified', unknown.state === 'UNKNOWN' && unknown.rows[0].result === 'UNKNOWN');
  ok('a running canary outranks the recorded verdicts',
    report.canaryView({ canary_verdicts: { xseg: { passed: true } } }, { engine_state: 'RUNNING_CANARY', engine_model: 'gpen_256', engine_precision: 'FP16' }).state === 'RUNNING');
  ok('a compiling engine is not a canary', report.canaryView({}, { engine_state: 'COMPILING_TRT' }).state === 'NONE');

  const panel = (p, run) => html(React.createElement(CanaryPrecisionStatus, { precision: p, run }));
  const ph = panel({ configured: 'mixed', effective: 'fp32', canary_verdicts: { xseg: { passed: true }, '1k3d68': { passed: false, reason: 'ssim 0.81' } } }, null);
  ok('the panel shows state, precision step-down and each model', ph.includes('data-state="FAILED"') && text(ph).includes('mixed → fp32')
    && ph.includes('data-canary="xseg"') && text(ph).includes('✗ fail · ssim 0.81') && text(ph).includes('✓ pass'), text(ph));
  ok('the panel says when nothing has been recorded', text(panel({}, null)).includes('No canary verdict recorded yet'));
  ok('a running canary names the model', text(panel({}, { engine_state: 'RUNNING_CANARY', engine_model: 'gpen_256', engine_precision: 'FP16' })).includes('gpen_256 · FP16'));

  const idle = html(React.createElement(PipelineHudView, { pipeline: { stage: null, stages: {}, summary: {} }, runtime: null, run: null }));
  ok('an idle backend with no canary data renders nothing', idle === '');
  const runningNoVerdicts = html(React.createElement(PipelineHudView, { runtime: null, run: null,
    pipeline: { stage: report.STAGE_3, stages: { [report.STAGE_3]: { state: 'RUNNING', stage_progress_pct: 5, fps: 1, wall_s: 1 } }, summary: {} } }));
  ok('a running job with no verdicts shows the canary block as NONE (not hidden, not passed)',
    runningNoVerdicts.includes('data-state="NONE"') && text(runningNoVerdicts).includes('No canary verdict recorded yet'));
  const withCanary = html(React.createElement(PipelineHudView, { pipeline: { stage: null, stages: {}, summary: {} },
    runtime: { sections: { PRECISION: { values: { canary_verdicts: { xseg: { passed: true } } } } } }, run: null }));
  ok('canary verdicts alone are enough to show the HUD (between runs)', withCanary.includes('data-testid="canary-status"'));
}

console.log(failures === 0 ? '\nall pipeline-controls checks passed' : `\n${failures} pipeline-controls check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
