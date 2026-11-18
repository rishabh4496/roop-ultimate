/**
 * UI overhaul Stage 11: the pipeline flow bar and the 3x3 parameter-sweep contact sheet.
 *
 *   node .render-check/run-module.mjs render-ui-overhaul-stage11.mjs
 *
 * What this proves, and what it does not:
 *   * flowModel     the node order, the 1-6 numbering, the signal path, the latency lookup (null, never a made-up number), every
 *                   status pill against the telemetry that raises it
 *   * sweepModel    zoom about the pointer keeps the point under it, the loupe shows ONE image point in every cell, the focus
 *                   presets centre the feature, axis values clamp and step, the warnings
 *   * slots         assigning a sweep cell writes the recipe over the live settings WITHOUT marking the slot active, and a recall
 *                   puts exactly that recipe back (the real history + slot stores)
 *   * keys          the flow / sweep actions exist, do not clash, and a digit reaches flow.solo1 as well as snapshot.slotA
 *   * markup        the server-rendered bar and sheet carry the structure the e2e specs and a screen reader rely on
 * NOT covered here (e2e/stage11-flow-sweep.spec.js drives them in a browser): the focus + stopPropagation behaviour of the key
 * handler, the pointer-driven pan / wheel / loupe, the pulse animation.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

globalThis.window = globalThis;
// src/api.js reads window.location at module scope (the sheet imports it through useApi); no request is made here.
globalThis.location = { origin: 'http://127.0.0.1:1', protocol: 'http:', host: '127.0.0.1:1' };
const here = dirname(fileURLToPath(import.meta.url));
const read = (rel) => readFileSync(join(here, '..', 'src', rel), 'utf8');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;
const tick = () => new Promise((r) => { queueMicrotask(r); });
const sleep = (ms) => new Promise((r) => { setTimeout(r, ms); });

const F = await import('../src/components/faceswap/flowModel.js');
const M = await import('../src/components/faceswap/sweepModel.js');
const { SWEEP_PARAMETERS } = await import('../src/components/faceswap/sweepParameters.js');
const { ACTIONS, ACTION_BY_ID } = await import('../src/shortcuts/actions.js');
const { allConflicts } = await import('../src/shortcuts/keymap.js');
const { actionsFor, eventMatches } = await import('../src/store/shortcutStore.js');
const { useHistoryStore, resetHistoryForTests } = await import('../src/store/historyStore.js');
const S = await import('../src/store/slotStore.js');
const C = await import('../src/history/historyChannels.js');
const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const html = (el) => renderToStaticMarkup(el).replace(/<!-- -->/g, '');

// ── the nodes ────────────────────────────────────────────────────────────────
console.log('flow nodes');
{
  ok('eight nodes in the brief\'s order', eq(F.FLOW_NODES.map((n) => n.id),
    ['source', 'detect_align', 'hyperswap', 'gaze_mouth', 'xseg_roto', 'restore_ultra', 'relight_lut', 'writer']));
  ok('keys 1-6 number the six intermediate buffers; Source and Output are unnumbered ends',
    eq(F.FLOW_NODES.filter((n) => n.n >= 1 && n.n <= 6).map((n) => n.n), [1, 2, 3, 4, 5, 6])
    && F.NODE_BY_ID.source.n === 0 && F.NODE_BY_ID.writer.n === 7);
  ok('KEY_TO_SOLO is exactly the ids the backend numbers 1-6', eq(F.KEY_TO_SOLO,
    { 1: 'detect_align', 2: 'hyperswap', 3: 'gaze_mouth', 4: 'xseg_roto', 5: 'restore_ultra', 6: 'relight_lut' }));
  ok('only Restorer and Colour have a difference pass', eq(F.FLOW_NODES.filter((n) => n.diffId).map((n) => [n.id, n.diffId]),
    [['restore_ultra', 'restore_diff'], ['relight_lut', 'relight_diff']]));
  ok('Detector cannot be bypassed; the swap, matte, restorer and colour can', !F.NODE_BY_ID.detect_align.bypassable
    && ['hyperswap', 'xseg_roto', 'restore_ultra', 'relight_lut'].every((id) => F.NODE_BY_ID[id].bypassable));
  ok('MODULAR_FLOW_NODES is the same list (the old import name keeps working)', F.MODULAR_FLOW_NODES === F.FLOW_NODES);
  ok('the four inspection passes are all registered; two are honestly unavailable',
    eq(F.BUFFER_SOLO_PASSES.map((p) => p.id), ['bisenet_parsing', 'xseg_3_mask', 'occlusion_alpha', 'depth_z_order'])
    && eq(F.BUFFER_SOLO_PASSES.filter((p) => !p.available).map((p) => p.id), ['occlusion_alpha', 'depth_z_order']));
}

console.log('solo ids');
{
  ok('a node press toggles its solo', F.nextSolo(null, 'hyperswap') === 'hyperswap' && F.nextSolo('hyperswap', 'hyperswap') === null);
  ok('another node replaces the solo', F.nextSolo('hyperswap', 'xseg_roto') === 'xseg_roto');
  ok('the difference pass is its own solo on Restorer / Colour', F.nextSolo(null, 'restore_ultra', { diff: true }) === 'restore_diff'
    && F.nextSolo('restore_diff', 'restore_ultra', { diff: true }) === null && F.nextSolo('restore_ultra', 'restore_ultra', { diff: true }) === 'restore_diff');
  ok('a node with no difference pass falls back to its plain solo', F.nextSolo(null, 'hyperswap', { diff: true }) === 'hyperswap');
  ok('the Output has nothing to solo and clears', F.nextSolo('hyperswap', 'writer') === null && F.nextSolo(null, 'nope') === null);
  ok('a difference id belongs to its node', F.nodeOfSolo('restore_diff') === 'restore_ultra' && F.nodeOfSolo('relight_diff') === 'relight_lut');
  ok('the matte passes belong to the matte node', ['bisenet_parsing', 'xseg_3_mask', 'occlusion_alpha', 'depth_z_order'].every((id) => F.nodeOfSolo(id) === 'xseg_roto'));
  ok('nothing / unknown belongs to no node', F.nodeOfSolo(null) === null && F.nodeOfSolo('warp') === null);
  ok('isDiffSolo', F.isDiffSolo('restore_diff') && F.isDiffSolo('relight_diff') && !F.isDiffSolo('restore_ultra'));
  ok('labels', F.soloLabel('hyperswap') === 'Swapper' && F.soloLabel('restore_diff') === 'Restorer difference'
    && F.soloLabel('xseg_3_mask') === 'XSeg 3 Alpha Matte' && F.soloLabel('source') === 'Source frame' && F.soloLabel(null) === '');
}

console.log('signal path');
{
  const live = F.signalPath(null);
  ok('no solo: every node and edge is lit (the whole chain feeds the viewport)', live.nodes.every((s) => s === 'lit') && live.edges.every((s) => s === 'lit'));
  const sw = F.signalPath('hyperswap');
  ok('solo the swapper: source and detector lit, swapper solo, everything after dim',
    eq(sw.nodes, ['lit', 'lit', 'solo', 'dim', 'dim', 'dim', 'dim', 'dim']), JSON.stringify(sw.nodes));
  ok('edges up to the soloed node are lit, the rest dim', eq(sw.edges, ['lit', 'lit', 'dim', 'dim', 'dim', 'dim', 'dim']), JSON.stringify(sw.edges));
  const diff = F.signalPath('restore_diff');
  ok('a difference pass lights the path to its node', diff.target === 'restore_ultra' && diff.nodes[5] === 'solo' && diff.nodes[4] === 'lit' && diff.nodes[6] === 'dim');
  const byp = F.signalPath('restore_ultra', { xseg_roto: true });
  ok('a bypassed node is off, and so are the edges around it', byp.nodes[4] === 'off' && byp.edges[3] === 'off' && byp.edges[4] === 'off' && byp.nodes[5] === 'solo');
  ok('a bypassed node outranks the solo on it', F.signalPath('xseg_roto', { xseg_roto: true }).nodes[4] === 'off');
}

// ── latency ──────────────────────────────────────────────────────────────────
console.log('latency');
{
  const swap = F.NODE_BY_ID.hyperswap;
  ok('nothing measured: null, not an invented figure', F.latencyFor({}, swap) === null && F.latencyFor(null, swap) === null);
  const dash = { dashboard: { telemetry: { latency_ms: { detect: 3.44, swap: 11.26, mask: 6.8, lipsync: null, restore: 14.5, blend: 4.1, encode: 2.04 } } } };
  ok('the dashboard\'s per-frame latency split, rounded to 0.1 ms', F.latencyFor(dash, F.NODE_BY_ID.detect_align) === 3.4 && F.latencyFor(dash, swap) === 11.3
    && F.latencyFor(dash, F.NODE_BY_ID.xseg_roto) === 6.8 && F.latencyFor(dash, F.NODE_BY_ID.restore_ultra) === 14.5
    && F.latencyFor(dash, F.NODE_BY_ID.relight_lut) === 4.1 && F.latencyFor(dash, F.NODE_BY_ID.writer) === 2);
  ok('lip-sync falls back to the run\'s own figure', F.latencyFor({ ...dash, run: { lipsync_ms: 4.26 } }, F.NODE_BY_ID.gaze_mouth) === 4.3);
  ok('stage-3 op timings are used when the dashboard has none',
    F.latencyFor({ pipeline: { stages: { STAGE_3_MAIN_SWAP: { ops: { model: { ms: 9.04 } } } } } }, swap) === 9);
  ok('Source has no latency', F.latencyFor(dash, F.NODE_BY_ID.source) === null);
  ok('negative or NaN is unmeasured', F.latencyFor({ dashboard: { telemetry: { latency_ms: { swap: -1 } } } }, swap) === null
    && F.latencyFor({ dashboard: { telemetry: { latency_ms: { swap: NaN } } } }, swap) === null);
  ok('levels: green < 5 ms, amber to 15, red above, none when unmeasured', F.latencyLevel(4.9) === 'ok' && F.latencyLevel(5) === 'warn'
    && F.latencyLevel(15) === 'warn' && F.latencyLevel(15.1) === 'hot' && F.latencyLevel(null) === 'none');
}

// ── titles ───────────────────────────────────────────────────────────────────
console.log('titles');
{
  ok('the swapper names the model in use', F.nodeTitle(F.NODE_BY_ID.hyperswap, { swap_model: 'hyperswap' }) === 'Swapper · HyperSwap'
    && F.nodeTitle(F.NODE_BY_ID.hyperswap, { swap_model: 'realswap' }) === 'Swapper · RealSwap');
  ok('the restorer names the enhancer in use', F.nodeTitle(F.NODE_BY_ID.restore_ultra, { selected_enhancer: 'restore_ultra' }) === 'Restorer · Restore Ultra'
    && F.nodeTitle(F.NODE_BY_ID.restore_ultra, { selected_enhancer: 'gpen_256_pro' }) === 'Restorer · GPEN 256 Pro'
    && F.nodeTitle(F.NODE_BY_ID.restore_ultra, { selected_enhancer: 'None' }) === 'Restorer · No restorer');
  ok('the output names the writer the job really got', F.nodeTitle(F.NODE_BY_ID.writer, {}, { io_encode: 'NVENC-Direct' }) === 'Output · NVENC-Direct'
    && F.nodeTitle(F.NODE_BY_ID.writer, {}, {}) === 'NVHardwareVideoWriter Output');
  ok('the rest keep the brief\'s names', F.nodeTitle(F.NODE_BY_ID.detect_align) === 'Detector & 5-Pt Pose' && F.nodeTitle(F.NODE_BY_ID.xseg_roto) === 'XSeg 3 / Occluder Matte'
    && F.nodeTitle(F.NODE_BY_ID.source) === 'Source / .fsz Identity' && F.nodeTitle(F.NODE_BY_ID.relight_lut) === 'Color / LUT Bake');
}

// ── status pills ─────────────────────────────────────────────────────────────
console.log('status pills');
{
  const ids = (b, node) => b[node].map((p) => p.id);
  const quiet = F.flowBadges({});
  ok('nothing reported, nothing shown', Object.values(quiet).every((l) => l.length === 0));
  ok('TensorRT wanted, CUDA running: a downgrade pill on every model node', (() => {
    const b = F.flowBadges({ provider: { requested: 'tensorrt', active: 'cuda' } });
    return ['detect_align', 'hyperswap', 'xseg_roto', 'restore_ultra'].every((n) => ids(b, n).includes('provider-downgrade'))
      && !ids(b, 'source').length && !ids(b, 'relight_lut').length && b.hyperswap[0].tone === 'warn' && /CUDA/.test(b.hyperswap[0].text);
  })());
  ok('the backend\'s own verdict and reason are shown (a TensorRT session that failed to build)', (() => {
    const b = F.flowBadges({ provider: { requested: 'auto', active: 'cuda', degraded: true, degradation_reason: 'TensorRT engine build failed (out of memory)' } });
    return b.hyperswap[0].id === 'provider-downgrade' && /out of memory/.test(b.hyperswap[0].title) && /running on CUDA/.test(b.hyperswap[0].title);
  })());
  ok('the provider name from the backend may carry ExecutionProvider', F.flowBadges({ provider: { requested: 'tensorrt', active: 'CUDAExecutionProvider' } }).hyperswap.length === 1);
  ok('the provider you asked for is not a downgrade', F.flowBadges({ provider: { requested: 'cuda', active: 'tensorrt' } }).hyperswap.length === 0
    && F.flowBadges({ provider: { requested: 'tensorrt', active: 'tensorrt', degraded: false } }).hyperswap.length === 0
    && F.flowBadges({ provider: { requested: 'auto', active: 'cuda', degraded: false } }).hyperswap.length === 0);
  ok('an unknown provider name is never called a downgrade', F.flowBadges({ provider: { requested: 'tensorrt', active: 'weird' } }).hyperswap.length === 0);
  ok('no provider status yet (meta not loaded): nothing, not a guess', F.flowBadges({ settings: { provider: 'tensorrt' } }).hyperswap.length === 0 && F.flowBadges({ provider: null }).hyperswap.length === 0);
  ok('batch > 1 shows the TensorRT batch pill on the restorer only', (() => {
    const b = F.flowBadges({ settings: { ultramax_batch_size: 4, selected_enhancer: 'ultramax' } });
    return ids(b, 'restore_ultra').join() === 'trt-batch' && b.restore_ultra[0].text === 'TRT batch ×4' && !ids(b, 'hyperswap').length;
  })());
  ok('batch 1 or no restorer: no batch pill', !ids(F.flowBadges({ settings: { ultramax_batch_size: 1, selected_enhancer: 'ultramax' } }), 'restore_ultra').includes('trt-batch')
    && !ids(F.flowBadges({ settings: { ultramax_batch_size: 4, selected_enhancer: 'None' } }), 'restore_ultra').includes('trt-batch'));
  const eng = (state, model = 'gpen_256_pro', precision = 'fp16') => F.flowBadges({ run: { engine_state: state, engine_model: model, engine_precision: precision } });
  ok('an engine build is a warning on the node of the model being built', (() => {
    const b = eng('COMPILING_TRT');
    return b.restore_ultra[0].id === 'trt-compile' && b.restore_ultra[0].text === 'TRT build' && b.restore_ultra[0].tone === 'warn'
      && /gpen_256_pro · fp16/.test(b.restore_ultra[0].title) && /not representative/.test(b.restore_ultra[0].title);
  })());
  ok('the canary that follows a build is its own pill', eng('RUNNING_CANARY').restore_ultra[0].id === 'engine-canary' && eng('RUNNING_CANARY').restore_ultra[0].text === 'engine check');
  ok('which node a build belongs to follows the model', ids(eng('COMPILING_TRT', 'inswapper_128'), 'hyperswap')[0] === 'trt-compile' && ids(eng('COMPILING_TRT', 'scrfd_10g'), 'detect_align')[0] === 'trt-compile'
    && ids(eng('COMPILING_TRT', 'xseg_3'), 'xseg_roto')[0] === 'trt-compile' && ids(eng('COMPILING_TRT', 'restoreformer'), 'restore_ultra')[0] === 'trt-compile'
    && ids(eng('COMPILING_TRT', ''), 'restore_ultra')[0] === 'trt-compile');
  ok('idle or finished: nothing', Object.values(eng('IDLE')).every((l) => l.length === 0) && Object.values(F.flowBadges({ run: {} })).every((l) => l.length === 0));
  ok('the enhancer store is NOT a source (only the Enhancer panel writes it, so it sits at its default): passing it changes nothing',
    Object.values(F.flowBadges({ enhancer: { active_execution_provider: 'cpu', compilation: { compiling: true } }, settings: { provider: 'tensorrt' } })).every((l) => l.length === 0));
  const nat = { auto_bypass_restorer_native_swap: true, selected_enhancer: 'restore_ultra', swap_model: 'hyperswap' };
  const pill = (settings, facePx) => F.flowBadges({ settings, facePx }).restore_ultra.find((p) => p.id.startsWith('native-bypass'));
  ok('the ramp is the backend\'s: 1 at 256 px, 0.5 at 320, 0 at 384, a straight line between',
    F.bypassWeight(200) === 1 && F.bypassWeight(256) === 1 && F.bypassWeight(320) === 0.5 && F.bypassWeight(384) === 0 && F.bypassWeight(500) === 0
    && near(F.bypassWeight(288), 0.75) && F.bypassWeight(null) === 0 && F.bypassWeight(NaN) === 0);
  ok('a longer ramp (the ultramax_ramp setting) stretches it', near(F.bypassWeight(384, 448), 64 / 192) && F.bypassWeight(300, 256) === 0 && F.bypassWeight(256, 256) === 1);
  ok('a 288 px face: the bypass is ACTIVE and says 75%, not a hard yes', (() => {
    const p = pill(nat, 288);
    return p.id === 'native-bypass' && p.tone === 'info' && p.text === 'native bypass 75%' && /75% DoG sharpen and 25% neural restorer/.test(p.title) && /320 px is the 50% point/.test(p.title);
  })());
  ok('at 320 px it is half, at 256 px it is all, at 384 px it is armed but not active',
    pill(nat, 320).text === 'native bypass 50%' && pill(nat, 256).text === 'native bypass 100%' && pill(nat, 384).id === 'native-bypass-armed' && pill(nat, 500).id === 'native-bypass-armed');
  ok('no face detected: armed, not active', pill(nat, null).id === 'native-bypass-armed' && /no face detected/.test(pill(nat, null).title));
  ok('the ultramax_ramp setting moves the far end of the ramp', pill({ ...nat, ultramax_ramp: 448 }, 384).text === 'native bypass 33%');
  ok('it only shows where the backend would really apply it: a native-resolution swapper AND a restorer it replaces',
    pill({ ...nat, swap_model: 'inswapper' }, 200) === undefined && pill({ ...nat, swap_model: 'simswap' }, 200) === undefined
    && pill({ ...nat, selected_enhancer: 'gpen_256' }, 200) === undefined && pill({ ...nat, selected_enhancer: 'None' }, 200) === undefined
    && pill({ ...nat, swap_model: 'realswap' }, 200) && pill({ ...nat, selected_enhancer: 'UltraMax' }, 200) && pill({ ...nat, selected_enhancer: 'codeformer' }, 200));
  ok('auto-bypass off: no bypass pill at any size', pill({ ...nat, auto_bypass_restorer_native_swap: false }, 100) === undefined);
  ok('lip-sync that fell back is a warning on Gaze & Mouth', (() => {
    const b = F.flowBadges({ run: { mouth_mode: 'audio_driven_sync', mouth_fallback: 'no audio track' } });
    return b.gaze_mouth[0].id === 'lipsync-off' && b.gaze_mouth[0].tone === 'warn' && /no audio track/.test(b.gaze_mouth[0].title);
  })());
  ok('a mouth fallback in a mode that never asked for lip-sync is not shown', !F.flowBadges({ run: { mouth_mode: 'source_mouth_restore', mouth_fallback: 'x' } }).gaze_mouth.length);
  ok('hardware I/O fallbacks warn on the Output, with a count', (() => {
    const b = F.flowBadges({ run: { io_fallbacks: 2, io_fallback: 'NVENC unavailable' } });
    return b.writer[0].text === 'HW fallback ×2' && /NVENC unavailable/.test(b.writer[0].title);
  })());
  ok('badgeCounts tallies warnings apart from info', eq(F.badgeCounts(F.flowBadges({
    settings: { ultramax_batch_size: 2, selected_enhancer: 'ultramax' }, provider: { requested: 'tensorrt', active: 'cuda' } })), { warn: 4, info: 1 }));
  ok('tallestFacePx reads the tallest box', F.tallestFacePx([[0, 0, 50, 200], [0, 0, 80, 310.5]]) === 310.5 && F.tallestFacePx([]) === null && F.tallestFacePx(null) === null
    && F.tallestFacePx([[1, 2]]) === null);
}

// ── the sweep's maths ────────────────────────────────────────────────────────
console.log('sweep view');
{
  const v0 = { zoom: 2, pan: { x: 0.1, y: -0.2 } };
  let allFixed = true;
  let exercised = 0;
  for (const [u, v, f] of [[0.5, 0.5, 1.15], [0.2, 0.7, 1.15], [0.9, 0.1, 1 / 1.15], [0.3, 0.3, 2], [0.75, 0.25, 0.8]]) {
    const before = M.imagePointAt(v0, u, v);
    const after = M.zoomAt(v0, f, { u, v });
    const pt = M.imagePointAt(after, u, v);
    // unclamped result keeps the point; a clamp can only move it when the zoom is pinned at 1 or the pan hit its limit
    const lim = Math.max(0, (after.zoom - 1) / 2);
    const clamped = Math.abs(after.pan.x) >= lim - 1e-12 || Math.abs(after.pan.y) >= lim - 1e-12;
    if (!clamped) exercised += 1;
    if (!clamped && !(near(pt.x, before.x, 1e-9) && near(pt.y, before.y, 1e-9))) allFixed = false;
  }
  ok('zooming about the pointer keeps the image point under it fixed (and the check really ran: 4+ unclamped cases)', allFixed && exercised >= 4, `exercised ${exercised}`);
  ok('zoom is clamped to 1..8', M.zoomAt({ zoom: 7.9, pan: { x: 0, y: 0 } }, 2, { u: 0.5, v: 0.5 }).zoom === 8 && M.zoomAt(M.FIT, 0.5, { u: 0.5, v: 0.5 }).zoom === 1);
  ok('at zoom 1 the pan is pinned to 0 (the picture already fits)', eq(M.clampPan({ x: 0.4, y: -0.4 }, 1), { x: 0, y: 0 }));
  ok('at zoom 3 the pan may reach +-1 (the picture overhangs by (z-1)/2)', eq(M.clampPan({ x: 5, y: -5 }, 3), { x: 1, y: -1 }));
  ok('dragging pans by the fraction moved, within the limits', eq(M.panBy({ zoom: 2, pan: { x: 0, y: 0 } }, 0.2, -0.1), { zoom: 2, pan: { x: 0.2, y: -0.1 } })
    && M.panBy({ zoom: 2, pan: { x: 0.45, y: 0 } }, 0.2, 0).pan.x === 0.5);
  ok('normalizeView repairs a bad view', eq(M.normalizeView({ zoom: 99, pan: { x: 9, y: 9 } }), { zoom: 8, pan: { x: 3.5, y: 3.5 } }) && M.normalizeView(null).zoom === 1);
  const c = M.viewCentredOn({ x: 0.5, y: 0.4 }, 2.8);
  const back = M.imagePointAt(c, 0.5, 0.5);
  ok('a focus view puts the feature in the middle of the cell', near(back.x, 0.5, 1e-9) && near(back.y, 0.4, 1e-9), JSON.stringify(back));
  const fc = { crop_rect: [100, 50, 240, 220], eyes: { left: [90, 80], right: [150, 80] }, mouth: { center: [120, 150] }, jawline: { center: [120, 200] } };
  const eyes = M.focusPoint('eyes', fc);
  ok('eye focus is the midpoint of the eyes, in crop fractions', near(eyes.x, 120 / 240) && near(eyes.y, 80 / 220));
  ok('mouth and jaw focus use their own points', near(M.focusPoint('mouth', fc).y, 150 / 220) && near(M.focusPoint('jaw', fc).y, 200 / 220));
  ok('no face coordinates: sensible defaults, no crash', M.focusPoint('eyes', null).y < 0.5 && M.focusPoint('mouth', null).y > 0.5 && M.focusPoint('jaw', undefined).y > 0.75);
  ok('Fit is zoom 1, pan 0', eq(M.focusView('fit', fc), { zoom: 1, pan: { x: 0, y: 0 } }));
  ok('Eyes / Mouth / Jaw zoom 2.8x / 3x / 2.6x', M.focusView('eyes', fc).zoom === 2.8 && M.focusView('mouth', fc).zoom === 3 && M.focusView('jaw', fc).zoom === 2.6);
}

console.log('sweep loupe');
{
  const view = { zoom: 2, pan: { x: 0.12, y: -0.05 } };
  const a = M.loupeLayout({ frameW: 300, frameH: 400, view, u: 0.4, v: 0.6, lens: 120 });
  const b = M.loupeLayout({ frameW: 600, frameH: 800, view, u: 0.4, v: 0.6, lens: 120 });
  ok('the same pointer and view show the same image point in every cell, whatever the cell size', near(a.point.x, b.point.x) && near(a.point.y, b.point.y));
  ok('the loupe magnifies 3.5x what the cell shows (so zoom 2 gives 7x the fit size)', near(a.bgW, 300 * 2 * 3.5) && near(a.bgH, 400 * 2 * 3.5) && M.LOUPE_POWER === 3.5);
  ok('the image point under the pointer sits at the lens centre', near(a.bgX + a.point.x * a.bgW, 60) && near(a.bgY + a.point.y * a.bgH, 60));
  ok('the lens is centred on the pointer', near(a.left + 60, 0.4 * 300) && near(a.top + 60, 0.6 * 400));
  const noZoom = M.loupeLayout({ frameW: 200, frameH: 200, view: M.FIT, u: 0.5, v: 0.5, lens: 100 });
  ok('at fit, the middle of the cell is the middle of the picture', near(noZoom.point.x, 0.5) && near(noZoom.point.y, 0.5) && near(noZoom.bgX, 50 - 0.5 * 700));
}

console.log('sweep axes and warnings');
{
  const w = SWEEP_PARAMETERS.find((p) => p.id === 'restore_ultra_detail_weight');
  const px = SWEEP_PARAMETERS.find((p) => p.id === 'pixel_boost');
  const sm = SWEEP_PARAMETERS.find((p) => p.id === 'swap_model');
  ok('every parameter has a three-value default', SWEEP_PARAMETERS.every((p) => p.defaultRange.length === 3));
  ok('typed values clamp to the range', M.normalizeValue(w, 1.7) === 1 && M.normalizeValue(w, -3) === 0);
  ok('typed values snap to the step without float noise', M.normalizeValue(w, 0.3) === 0.3 && M.normalizeValue(w, 0.2 + 0.1) === 0.3 && M.normalizeValue(px, 400) === 512);
  ok('junk falls back', M.normalizeValue(w, 'abc', 0.5) === 0.5 && M.normalizeValue(w, '') === w.defaultRange[0]);
  ok('a categorical value must be one of the options', M.normalizeValue(sm, 'hyperswap') === 'hyperswap' && M.normalizeValue(sm, 'nonsense') === 'inswapper');
  ok('defaultValues copies (editing a field never mutates the table)', (() => { const d = M.defaultValues(w); d[0] = 9; return w.defaultRange[0] === 0.25; })());
  ok('valuesVary', M.valuesVary([1, 2, 3]) && !M.valuesVary([1, 1, '1']));
  ok('aspect: the pixel size wins, then the crop rect, then 1', M.aspectOf(null, { w: 300, h: 200 }) === 1.5 && M.aspectOf({ face_coordinates: { crop_rect: [0, 0, 240, 120] } }, null) === 2 && M.aspectOf({}, null) === 1);
  ok('cell indexing is row-major', M.cellIndex(2, 1) === 5 && M.tileAt({ tiles: Array.from({ length: 9 }, (_, i) => ({ i })) }, 2, 1).i === 5 && M.tileAt(null, 0, 0) === null);

  const none = M.resultWarnings({ tiles: [], identical: false, axes: { x: { bound: true }, y: { bound: true } } }, { xValues: [1, 2, 3], yValues: [1, 2, 3] });
  ok('a healthy sweep says nothing', none.length === 0 && M.resultWarnings(null).length === 0);
  const dead = M.resultWarnings({ identical: true, axes: { x: { bound: false, reason: 'Needs several frames.' }, y: { bound: true } } }, { xLabel: 'Stabilization', xValues: [1, 2, 3], yValues: [1, 2, 3] });
  ok('an unbound axis is named with the backend\'s reason, and identical tiles are not double-reported', dead.length === 1 && dead[0].kind === 'unbound' && /Stabilization: Needs several frames\./.test(dead[0].text));
  const same = M.resultWarnings({ identical: true, axes: { x: { bound: true }, y: { bound: true } } }, { xValues: [1, 2, 3], yValues: [1, 2, 3] });
  ok('byte-identical tiles with bound axes are called out', same.length === 1 && same[0].kind === 'identical');
  ok('no face is called out and is not "identical"', eq(M.resultWarnings({ no_face: true, identical: true, axes: {} }, { xValues: [1, 2, 3], yValues: [1, 2, 3] }).map((x) => x.kind), ['no_face']));
  ok('two repeated-value axes say there is nothing to compare', M.resultWarnings({ identical: false, axes: {} }, { xValues: [1, 1, 1], yValues: [2, 2, 2] }).some((x) => x.kind === 'flat'));
}

// ── assigning a cell to a slot (real history + slot stores) ──────────────────
console.log('assign a cell to a slot');
{
  resetHistoryForTests();
  S.resetSlotsForTests();
  const live = { value: { restore_ultra_detail_weight: 0.75, mask_erode_dilate_radius: 0, blend_ratio: 0.8, provider: 'cuda' }, applied: [] };
  const unregister = useHistoryStore.getState().register('settings', {
    ...C.SETTINGS_CHANNEL,
    apply: (v) => { live.value = v; live.applied.push(v); queueMicrotask(() => useHistoryStore.getState().observe('settings', live.value)); },
  });
  useHistoryStore.getState().observe('settings', live.value);
  await tick();

  const recipe = { restore_ultra_detail_weight: 0.25, mask_erode_dilate_radius: -8 };
  const slot = S.assignRecipeToSlot('faceswap', 'C', recipe, { fallbackSettings: {} });
  const st = S.useSlotStore.getState();
  ok('the slot holds the live settings with the cell\'s two values written over them', slot.state.settings.restore_ultra_detail_weight === 0.25
    && slot.state.settings.mask_erode_dilate_radius === -8 && slot.state.settings.blend_ratio === 0.8 && slot.state.settings.provider === 'cuda');
  ok('assigning does NOT apply it: the live recipe is unchanged', live.value.restore_ultra_detail_weight === 0.75 && live.applied.length === 0);
  ok('assigning does NOT claim the slot is the one in force', st.active.faceswap === null);
  ok('a face crop is never mistaken for the frame on screen (no frame key)', slot.frameKey === null && st.banks.faceswap.C === slot);
  ok('the other slots are untouched', ['A', 'B', 'D'].every((id) => st.banks.faceswap[id] === null));

  const n = useHistoryStore.getState().entries.length;
  S.recallSlot('faceswap', 'C');
  await sleep(20);
  ok('recalling it applies exactly that recipe', live.value.restore_ultra_detail_weight === 0.25 && live.value.mask_erode_dilate_radius === -8 && live.value.blend_ratio === 0.8);
  ok('and leaves the machine settings alone', live.value.provider === 'cuda');
  ok('as one undoable, named entry', useHistoryStore.getState().entries.length === n + 1 && useHistoryStore.getState().entries.at(-1).label === 'Recalled Slot C');
  ok('now the slot IS the active one', S.useSlotStore.getState().active.faceswap === 'C');
  useHistoryStore.getState().undo();
  await sleep(20);
  ok('undo flips back to what you had', live.value.restore_ultra_detail_weight === 0.75);

  // the fallback: settings channel not mounted
  unregister();
  resetHistoryForTests();
  S.resetSlotsForTests();
  const bare = S.assignRecipeToSlot('faceswap', 'A', { blend_ratio: 0.3 }, { fallbackSettings: { blend_ratio: 0.8, provider: 'cuda' } });
  ok('without a settings channel the fallback settings stand in', bare.state.settings.blend_ratio === 0.3 && bare.state.settings.provider === 'cuda');
  const closed = [];
  const bmp = { width: 4, height: 4, close() { closed.push(1); } };
  S.assignRecipeToSlot('faceswap', 'B', { blend_ratio: 0.1 }, { bitmap: bmp });
  S.assignRecipeToSlot('faceswap', 'B', { blend_ratio: 0.2 });
  S.flushRetired();
  ok('replacing a slot retires the old picture exactly once', closed.length === 1);
  S.resetSlotsForTests();
  resetHistoryForTests();
}

// ── the keys ─────────────────────────────────────────────────────────────────
console.log('keys');
{
  const flow = ACTIONS.filter((a) => a.scope === 'flow');
  const sweep = ACTIONS.filter((a) => a.scope === 'sweep');
  ok('the flow bar has 1-6, 0, S, D and B', eq(flow.map((a) => a.id), ['flow.solo1', 'flow.solo2', 'flow.solo3', 'flow.solo4', 'flow.solo5', 'flow.solo6', 'flow.soloClear', 'flow.solo', 'flow.diff', 'flow.bypass']));
  ok('the sheet has 1-9, Enter, A-D, G and F', eq(sweep.map((a) => a.id), ['sweep.cell', 'sweep.apply', 'sweep.slotA', 'sweep.slotB', 'sweep.slotC', 'sweep.slotD', 'sweep.loupe', 'sweep.fit']));
  ok('sweep.cell owns all nine digits', eq(ACTION_BY_ID['sweep.cell'].keys, ['1', '2', '3', '4', '5', '6', '7', '8', '9']));
  ok('the default keymap has no clash (flow / sweep overlap only global)', allConflicts({}).length === 0, JSON.stringify(allConflicts({})));
  const ev = (key, extra = {}) => ({ key, code: /^\d$/.test(key) ? `Digit${key}` : `Key${key.toUpperCase()}`, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, ...extra });
  ok('digit 1 reaches the flow bar AND the snapshot slot (the bar stops the event; elsewhere it is slot A)',
    eq(actionsFor(ev('1')).sort(), ['flow.solo1', 'snapshot.slotA', 'sweep.cell']) && eventMatches(ev('1'), 'flow.solo1'));
  ok('digit 6 is only the flow bar and the sheet', eq(actionsFor(ev('6')).sort(), ['flow.solo6', 'sweep.cell']));
  ok('S reaches the flow bar and split view', actionsFor(ev('s')).includes('flow.solo') && actionsFor(ev('s')).includes('viewport.splitView'));
}

// ── markup ───────────────────────────────────────────────────────────────────
console.log('server-rendered bar');
{
  const { VisualPipelineNodeFlowBar } = await import('../src/components/faceswap/PipelineNodeFlow.jsx');
  const T = await import('../src/store/telemetryStore.js');
  void T;
  const downgraded = { requested: 'tensorrt', active: 'cuda', degraded: true, degradation_reason: 'TensorRT session failed to build.' };
  const base = { settings: { swap_model: 'hyperswap', selected_enhancer: 'restore_ultra', ultramax_batch_size: 4, auto_bypass_restorer_native_swap: true },
    provider: downgraded, faces: [[0, 0, 200, 288]] };
  const idle = html(React.createElement(VisualPipelineNodeFlowBar, { ...base }));
  const count = (s, re) => (s.match(re) || []).length;
  ok('a toolbar, an ordered list of eight stages and seven connectors', /role="toolbar"/.test(idle) && /<ol[^>]*aria-label="Pipeline stages, in order"/.test(idle)
    && count(idle, /data-testid="flow-node-/g) === 8 && count(idle, /data-testid="flow-edge-/g) === 7);
  ok('no solo: viewport says live, no clear button', /Viewport: live result/.test(idle) && !/flow-clear-solo/.test(idle) && count(idle, /data-state="lit"/g) >= 15);
  ok('each stage names its model', /Swapper · HyperSwap/.test(idle) && /Restorer · Restore Ultra/.test(idle) && /Detector &amp; 5-Pt Pose/.test(idle) && /XSeg 3 \/ Occluder Matte/.test(idle) && /Color \/ LUT Bake/.test(idle));
  ok('every stage has a Solo (S) pill except the Output', count(idle, /data-testid="flow-solo-/g) === 7 && /Solo \(S\)/.test(idle));
  ok('Restorer and Colour carry a difference pill; the rest do not', count(idle, /data-testid="flow-diff-/g) === 2);
  ok('bypass switches on the five bypassable stages, all on', count(idle, /data-testid="flow-bypass-/g) === 5 && count(idle, /role="switch"[^>]*aria-checked="true"/g) === 5);
  ok('latency badges say "— ms" with no telemetry, never a number', count(idle, /data-testid="flow-latency-/g) === 7 && count(idle, />— ms</g) === 7 && !/>\d+\.\dms</.test(idle));
  ok('the provider downgrade, TRT batch and native bypass pills are on the affected nodes', /flow-pill-hyperswap-provider-downgrade/.test(idle) && /flow-pill-restore_ultra-trt-batch/.test(idle)
    && /flow-pill-restore_ultra-native-bypass"/.test(idle) && /native bypass 75%/.test(idle) && /TRT batch ×4/.test(idle) && /flow-warn-count/.test(idle));
  ok('keyboard stops are the stage titles only (the inline controls are reachable by S / D / B)', count(idle, /tabindex="-1"/g) >= 14);

  const soloed = html(React.createElement(VisualPipelineNodeFlowBar, { ...base, soloStage: 'hyperswap', bypassedStages: { relight_lut: true } }));
  ok('solo: the node is marked, the path is lit up to it and dim after', /data-testid="flow-node-hyperswap"[^>]*data-state="solo"[^>]*data-solo="true"/.test(soloed)
    && /data-testid="flow-edge-1" data-state="lit"/.test(soloed) && /data-testid="flow-edge-2" data-state="dim"/.test(soloed));
  ok('solo: the label and the clear button appear', /flow-solo-label[^>]*>Swapper</.test(soloed) && /flow-clear-solo/.test(soloed) && /Soloing ✕/.test(soloed));
  ok('a bypassed stage is off and its switch reads off', /data-testid="flow-node-relight_lut"[^>]*data-state="off"[^>]*data-bypassed="true"/.test(soloed) && /role="switch"[^>]*aria-checked="false"[^>]*data-testid="flow-bypass-relight_lut"/.test(soloed.replace(/ +/g, ' ')) || /data-testid="flow-bypass-relight_lut"[^>]*aria-checked="false"/.test(soloed) || /aria-checked="false"[^>]*data-testid="flow-bypass-relight_lut"/.test(soloed));
  const diffed = html(React.createElement(VisualPipelineNodeFlowBar, { ...base, soloStage: 'restore_diff' }));
  ok('the difference pass lights the Restorer and presses its Δ, not its Solo pill', /data-testid="flow-node-restore_ultra"[^>]*data-state="solo"/.test(diffed) && /data-testid="flow-diff-restore_ultra"[^>]*aria-pressed="true"/.test(diffed.replace(/ +/g, ' '))
    || /aria-pressed="true"[^>]*data-testid="flow-diff-restore_ultra"/.test(diffed));
  const refused = html(React.createElement(VisualPipelineNodeFlowBar, { ...base, soloStage: 'depth_z_order', soloResult: { applied: false, note: 'Depth / Z-order is not built.' } }));
  ok('a solo the backend refused is said, in a status region', /role="status"[^>]*data-testid="flow-solo-refused"|data-testid="flow-solo-refused"[^>]*role="status"/.test(refused) && /Depth \/ Z-order is not built\./.test(refused));
  ok('a solo it produced is not flagged', !/flow-solo-refused/.test(html(React.createElement(VisualPipelineNodeFlowBar, { ...base, soloStage: 'hyperswap', soloResult: { applied: true, note: '' } }))));
  ok('the unavailable passes are disabled, the available ones are not', /data-testid="solo-buffer-occlusion_alpha"[^>]* disabled=""/.test(idle) && /data-testid="solo-buffer-depth_z_order"[^>]* disabled=""/.test(idle)
    && !/data-testid="solo-buffer-xseg_3_mask"[^>]* disabled=""/.test(idle) && !/data-testid="solo-buffer-bisenet_parsing"[^>]* disabled=""/.test(idle));
  ok('the key legend is on screen', /1–6 solo · 0 live · S solo · D diff · B bypass/.test(idle));
  ok('no downgrade pill once the provider is what was asked for, or while meta has not loaded',
    !/provider-downgrade/.test(html(React.createElement(VisualPipelineNodeFlowBar, { ...base, provider: { requested: 'tensorrt', active: 'tensorrt', degraded: false } })))
    && !/provider-downgrade/.test(html(React.createElement(VisualPipelineNodeFlowBar, { ...base, provider: null }))));
  ok('the pill carries the backend\'s reason', /TensorRT session failed to build\./.test(idle));
}

console.log('server-rendered contact sheet');
{
  const Modal = (await import('../src/components/faceswap/ParameterSweepModal.jsx')).default;
  // the sheet portals to document.body, which does not exist under react-dom/server: it renders nothing here, by design
  ok('closed: nothing', html(React.createElement(Modal, { isOpen: false, onClose() {} })) === '');
  const src = read('components/faceswap/ParameterSweepModal.jsx');
  ok('the panel is 90vw x 90vh', /h-\[90vh\] w-\[90vw\]/.test(src));
  ok('it is a modal dialog with a labelled title', /role="dialog"/.test(src) && /aria-modal="true"/.test(src) && /aria-labelledby="parameter-sweep-title"/.test(src));
  ok('the commit buttons carry the brief\'s words', /Apply Cell to Active Recipe/.test(src) && /Assign to Snapshot Slot/.test(src));
  ok('the sheet asks the backend with no 15 s deadline and a cancel signal', /\/api\/preview\/sweep[\s\S]{0,400}timeout: 0, signal: ctrl\.signal/.test(src));
  ok('the wheel handler is non-passive (React\'s synthetic onWheel is passive and cannot stop the page scrolling)', /addEventListener\('wheel', onWheel, \{ passive: false \}\)/.test(src));
  ok('every key it reads is a registry action, none is a literal key test (bar Esc)', /eventMatches\(e, 'sweep\.cell'\)/.test(src) && /eventMatches\(e, 'sweep\.apply'\)/.test(src)
    && (src.match(/e\.key ===/g) || []).filter(Boolean).length <= 2);
  const bar = read('components/faceswap/PipelineNodeFlow.jsx');
  ok('the flow bar reads every flow action through the registry', ['flow.solo1', 'flow.solo6', 'flow.soloClear', 'flow.solo', 'flow.diff', 'flow.bypass'].every((id) => bar.includes(`'${id}'`)) && /eventMatches\(e, 'flow\.soloClear'\)/.test(bar));
  ok('the bar stops the key it handled so 1-4 do not also hit the slots', /e\.stopPropagation\(\)/.test(bar));
  ok('the bar makes no request of its own', !/useApi|postJSON|getJSON|fetch\(/.test(bar));
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nstage 11 flow bar + contact sheet: all checks passed' : `\nstage 11 flow bar + contact sheet: ${failures} FAILED`);
