// Stage 11: the Visual Pipeline Node Flow Bar's model. Pure functions over plain values (the telemetry store's state, the
// settings, the bypass map) so the whole bar can be checked in node without a render (.render-check/render-ui-overhaul-stage11.mjs).
//
// The chain, as the brief draws it:
//
//   Source / .fsz Identity -> Detector & 5-pt Pose -> Swapper -> [Gaze & Mouth] -> XSeg 3 / Occluder Matte
//     -> Restorer -> Color / LUT Bake -> NVHardwareVideoWriter Output
//
// Six of those nodes are INTERMEDIATE buffers and carry the numbers 1-6 (the same digits `roop/preview_solo.py` and the old
// Stage 24 strip used, so every saved id and API alias still means what it did). Source and Output are the two ends: Source
// can be soloed from its pill (the untouched frame), Output is "the live result" and clears any solo.

/** @typedef {{ id:string, n:number, label:string, short:string, kind:'io'|'stage'|'sub', soloId:(string|null), diffId:(string|null),
 *   bypassable:boolean, latencyKey:(string|null), models:boolean, hint:string }} FlowNode */

/** @type {ReadonlyArray<FlowNode>} */
export const FLOW_NODES = Object.freeze([
  { id: 'source', n: 0, label: 'Source / .fsz Identity', short: 'Source', kind: 'io', soloId: 'source', diffId: null,
    bypassable: false, latencyKey: null, models: false, hint: 'The identity bank and the untouched target frame.' },
  { id: 'detect_align', n: 1, label: 'Detector & 5-Pt Pose', short: 'Detector', kind: 'stage', soloId: 'detect_align', diffId: null,
    bypassable: false, latencyKey: 'detect', models: true, hint: 'Face boxes, 5 keypoints and head pose. Cannot be bypassed: nothing is swapped without a face.' },
  { id: 'hyperswap', n: 2, label: 'Swapper', short: 'Swapper', kind: 'stage', soloId: 'hyperswap', diffId: null,
    bypassable: true, latencyKey: 'swap', models: true, hint: 'The raw swap before mask, restorer, mouth and relight. Bypass shows the original frame.' },
  { id: 'gaze_mouth', n: 3, label: 'Gaze & Mouth', short: 'Gaze/Mouth', kind: 'sub', soloId: 'gaze_mouth', diffId: null,
    bypassable: true, latencyKey: 'lipsync', models: false, hint: 'Gaze retention and mouth restore, before the restorer.' },
  { id: 'xseg_roto', n: 4, label: 'XSeg 3 / Occluder Matte', short: 'Matte', kind: 'stage', soloId: 'xseg_roto', diffId: null,
    bypassable: true, latencyKey: 'mask', models: true, hint: 'The alpha matte the swap is pasted through. Solo shows the raw matte; bypass drops the mask engines.' },
  { id: 'restore_ultra', n: 5, label: 'Restorer', short: 'Restorer', kind: 'stage', soloId: 'restore_ultra', diffId: 'restore_diff',
    bypassable: true, latencyKey: 'restore', models: true, hint: 'Neural restoration. The difference pass is the restored frame minus the unrestored one.' },
  { id: 'relight_lut', n: 6, label: 'Color / LUT Bake', short: 'Colour/LUT', kind: 'stage', soloId: 'relight_lut', diffId: 'relight_diff',
    bypassable: true, latencyKey: 'blend', models: false, hint: 'Relighting and the colour pass. The LUT is graded by the viewport in preview and baked by the writer on render.' },
  { id: 'writer', n: 7, label: 'NVHardwareVideoWriter Output', short: 'Output', kind: 'io', soloId: null, diffId: null,
    bypassable: false, latencyKey: 'encode', models: false, hint: 'The encoder. Selecting it returns the viewport to the live result.' },
]);

/** Legacy alias kept for the importers that still say MODULAR_FLOW_NODES (InteractivePreview's solo badge). */
export const MODULAR_FLOW_NODES = FLOW_NODES;

export const NODE_BY_ID = Object.freeze(Object.fromEntries(FLOW_NODES.map((n) => [n.id, n])));

/** Keys 1-6 -> the solo id of that numbered node. */
export const KEY_TO_SOLO = Object.freeze(Object.fromEntries(
  FLOW_NODES.filter((n) => n.n >= 1 && n.n <= 6).map((n) => [n.n, n.soloId])));

/**
 * The inspection passes that are not a pipeline node of their own: they hang off a node and are soloed from the pass row.
 * `available:false` ones are registered (the stage-25 check and the cheat-sheet list them) but disabled with the reason the
 * backend prints, so the button does not pretend.
 */
export const BUFFER_SOLO_PASSES = Object.freeze([
  { id: 'bisenet_parsing', key: 'bisenet_mask', alias: 'bisenet', node: 'xseg_roto', label: 'BiSeNet Parsing', shortLabel: 'BiSeNet',
    color: '#34d399', available: true,
    description: 'Parsed facial regions (eyes, brows, nose, mouth, skin) from BiSeNet, tinted over the frame' },
  { id: 'xseg_3_mask', key: 'mask', alias: 'xseg', node: 'xseg_roto', label: 'XSeg 3 Alpha Matte', shortLabel: 'XSeg 3',
    color: '#f59e0b', available: true,
    description: 'The raw XSeg 3 occlusion matte, warped back onto the frame' },
  { id: 'occlusion_alpha', key: 'occlusion_mask', alias: 'occlusion', node: 'xseg_roto', label: 'Occlusion Alpha', shortLabel: 'Occlusion',
    color: '#38bdf8', available: false,
    description: 'Foreign-object occlusion alpha. Needs the swap crop and temporal state: not available on a single preview frame' },
  { id: 'depth_z_order', key: 'depth_z_order', alias: 'depth', node: 'xseg_roto', label: 'Depth / Z-Order', shortLabel: 'Depth/Z',
    color: '#a78bfa', available: false,
    description: 'Relative depth and stacking order. Not built: nothing in the pipeline estimates it yet' },
]);

/** Which node a solo id belongs to (the signal path lights up to it). `null` for no solo / an unknown id. */
export function nodeOfSolo(soloId) {
  if (!soloId) return null;
  const direct = FLOW_NODES.find((n) => n.soloId === soloId || n.diffId === soloId || n.id === soloId);
  if (direct) return direct.id;
  const pass = BUFFER_SOLO_PASSES.find((p) => p.id === soloId || p.key === soloId || p.alias === soloId);
  return pass ? pass.node : null;
}

/** True when `soloId` is a node's difference pass. */
export const isDiffSolo = (soloId) => FLOW_NODES.some((n) => n.diffId && n.diffId === soloId);

/**
 * Toggle the solo of a node: the same id clears it. `diff` picks the node's difference pass where it has one.
 * Returns the id to send, or null to clear. A node with nothing to solo (the Output) clears.
 */
export function nextSolo(current, nodeId, { diff = false } = {}) {
  const node = NODE_BY_ID[nodeId];
  if (!node) return null;
  const target = diff ? (node.diffId || node.soloId) : node.soloId;
  if (!target) return null;
  return current === target ? null : target;
}

/**
 * How each node and each connector is drawn for the current solo: 'lit' (on the cyan signal path), 'solo' (the soloed node),
 * 'dim' (downstream of the soloed one: it does not contribute to what the viewport shows). `edges[i]` joins nodes[i] -> nodes[i+1].
 * Bypassed nodes are 'off' whatever the solo says. With no solo everything that is on is 'lit': the whole chain feeds the viewport.
 */
export function signalPath(soloId, bypassed = {}) {
  const target = nodeOfSolo(soloId);
  const at = target ? FLOW_NODES.findIndex((n) => n.id === target) : FLOW_NODES.length - 1;
  const nodes = FLOW_NODES.map((n, i) => {
    if (bypassed[n.id]) return 'off';
    if (target && i === at) return 'solo';
    return i < at || !target ? 'lit' : 'dim';
  });
  const edges = FLOW_NODES.slice(0, -1).map((_, i) => {
    if (nodes[i] === 'off' || nodes[i + 1] === 'off') return 'off';
    return i < at || !target ? 'lit' : 'dim';
  });
  return { nodes, edges, target, soloIndex: at };
}

// ── latency ────────────────────────────────────────────────────────────────

const num = (v) => (typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : null);

/**
 * Milliseconds per delivered frame for a node, from the telemetry store state; `null` when nothing has been measured.
 * The old bar fell back to constants (3.4 / 11.2 / ...) when telemetry was empty, so an idle app showed made-up
 * latencies. A badge that has no number says so.
 */
export function latencyFor(state, node) {
  const key = typeof node === 'string' ? node : node?.latencyKey;
  if (!key) return null;
  const lat = state?.dashboard?.telemetry?.latency_ms;
  const direct = num(lat?.[key]);
  if (direct != null) return Math.round(direct * 10) / 10;
  if (key === 'lipsync') {
    const v = num(state?.run?.lipsync_ms);
    return v == null ? null : Math.round(v * 10) / 10;
  }
  const ops = state?.pipeline?.stages?.STAGE_3_MAIN_SWAP?.ops;
  const opKey = { detect: 'warp', swap: 'model', restore: 'restore', blend: 'blend', mask: 'mask' }[key];
  const v = num(ops?.[opKey]?.ms);
  return v == null ? null : Math.round(v * 10) / 10;
}

/** Green < 5 ms, amber 5-15, red > 15, neutral when unmeasured. Returns a token, not classes (the view maps it to the theme). */
export function latencyLevel(ms) {
  if (ms == null) return 'none';
  if (ms < 5) return 'ok';
  if (ms <= 15) return 'warn';
  return 'hot';
}

// ── labels ─────────────────────────────────────────────────────────────────

const ENHANCER_LABELS = Object.freeze({
  restore_ultra: 'Restore Ultra', ultramax: 'UltraMax', 'restoreformer++': 'RF++', restoreformer_pp: 'RF++', rfpp: 'RF++',
  gpen_256: 'GPEN 256', gpen_256_pro: 'GPEN 256 Pro', gpen_512: 'GPEN 512', gpen: 'GPEN', codeformer: 'CodeFormer', gfpgan: 'GFPGAN',
});
const SWAP_LABELS = Object.freeze({
  hyperswap: 'HyperSwap', realswap: 'RealSwap', inswapper: 'InSwapper', ghost: 'GHOST', simswap: 'SimSwap', hififace: 'HiFiFace',
});
const humanize = (s) => String(s).replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
const isOff = (v) => v == null || v === '' || String(v).toLowerCase() === 'none';

export const enhancerLabel = (v) => (isOff(v) ? 'No restorer' : (ENHANCER_LABELS[String(v).toLowerCase()] || humanize(v)));
export const swapperLabel = (v) => (isOff(v) ? 'InSwapper' : (SWAP_LABELS[String(v).toLowerCase()] || humanize(v)));

/** The node's heading with the model the recipe actually uses ("Swapper · HyperSwap"). */
export function nodeTitle(node, settings = {}, run = {}) {
  switch (node.id) {
    case 'hyperswap': return `Swapper · ${swapperLabel(settings.swap_model)}`;
    case 'restore_ultra': return `Restorer · ${enhancerLabel(settings.selected_enhancer)}`;
    case 'writer': return run?.io_encode ? `Output · ${run.io_encode}` : node.label;
    default: return node.label;
  }
}

// ── warning / status pills ─────────────────────────────────────────────────

const PROVIDER_RANK = Object.freeze({ tensorrt: 4, cuda: 3, directml: 2, coreml: 2, openvino: 2, rocm: 2, cpu: 0 });
const providerKey = (v) => String(v || '').toLowerCase().replace(/executionprovider$/, '').replace(/[^a-z]/g, '');
const providerName = (k) => ({ tensorrt: 'TensorRT', cuda: 'CUDA', directml: 'DirectML', coreml: 'CoreML', openvino: 'OpenVINO', rocm: 'ROCm', cpu: 'CPU' }[k] || k.toUpperCase());

/** The tallest face box edge in px, from the preview's `faces` ([x1,y1,x2,y2] each), or null when no face was detected. */
export function tallestFacePx(faces) {
  let best = null;
  for (const f of Array.isArray(faces) ? faces : []) {
    if (!Array.isArray(f) || f.length < 4) continue;
    const h = Math.abs(Number(f[3]) - Number(f[1]));
    if (Number.isFinite(h) && (best == null || h > best)) best = h;
  }
  return best;
}

// The native restorer bypass, mirrored from roop/smart_restorer_bypass.py (app/tests/test_stage11_flow_solo_sweep.py pins the two
// together). It is NOT a step at 320 px: the share of the DoG-sharpened crop that replaces the neural restorer is 1.0 at or below
// RAMP_LO, 0.0 at or above RAMP_HI (the `ultramax_ramp` setting, 384 by default) and a straight line between, so 320 px is the
// 50% point. It applies only to Alpha / HyperSwap / RealSwap swappers and to the restorers below.
export const BYPASS_RAMP_LO_PX = 256;
export const BYPASS_RAMP_HI_PX = 384;
export const NATIVE_BYPASS_PX = 320;
export const isBypassSwapModel = (m) => /alpha|hyperswap|realswap/i.test(String(m ?? ''));
export const isBypassRestorer = (name) => /codeformer|gfpgan|restoreformer|restore_ultra|restore ultra|rf\+\+|ultramax/i.test(String(name ?? ''));

/** 1 = DoG only, 0 = restorer only, between = the mix; 0 for an unknown height. */
export function bypassWeight(heightPx, hi = BYPASS_RAMP_HI_PX, lo = BYPASS_RAMP_LO_PX) {
  if (!Number.isFinite(heightPx)) return 0;
  if (hi <= lo) return heightPx <= lo ? 1 : 0;
  return Math.min(1, Math.max(0, (hi - heightPx) / (hi - lo)));
}

/** The node an engine build / canary is for, from the model name the backend reports ("gpen", "scrfd", "inswapper", "xseg"...). */
export function nodeOfEngine(model) {
  const m = String(model || '').toLowerCase();
  if (/swap|simswap|ghost|hififace|ghostface/.test(m)) return 'hyperswap';
  if (/scrfd|retina|detect|det_|landmark|2d106|1k3d68|pose/.test(m)) return 'detect_align';
  if (/xseg|mask|parse|bisenet|occlu/.test(m)) return 'xseg_roto';
  return 'restore_ultra';                     // gpen / codeformer / restoreformer / ultramax / anything else is a restorer
}

/**
 * The status pills each node carries. `{ [nodeId]: [{ id, tone: 'info'|'warn'|'ok', text, title }] }`.
 *
 *   settings   the recipe (selected_enhancer, swap_model, ultramax_batch_size, auto_bypass_restorer_native_swap, ultramax_ramp)
 *   run        telemetry `run` (engine_state / engine_model / engine_precision, io_fallbacks, io_fallback, mouth_*)
 *   provider   /api/meta `provider_status`: { requested, active, degraded, degradation_reason }
 *   facePx     the tallest detected face in px (`tallestFacePx`) or null
 *
 * Every pill is derived from something the backend reported or the recipe says; none is decoration. The sources are the ones the
 * live app really fills: `meta.provider_status` (read once at start-up) and the `run.engine_*` fields on /ws/telemetry. The
 * enhancer store's `active_execution_provider` is NOT one of them - only the Enhancer panel writes it, so it sits at its default.
 */
export function flowBadges({ settings = {}, run = {}, provider = null, facePx = null } = {}) {
  const out = Object.fromEntries(FLOW_NODES.map((n) => [n.id, []]));

  // Provider downgrade: the backend says the sessions run on something other than what was asked for. Its own `degraded` verdict
  // wins (it knows about a TensorRT session that failed to build); otherwise compare the two by speed.
  const want = providerKey(provider?.requested ?? settings.provider);
  const have = providerKey(provider?.active);
  const slower = want in PROVIDER_RANK && have in PROVIDER_RANK && PROVIDER_RANK[have] < PROVIDER_RANK[want];
  if (have && (provider?.degraded === true || slower)) {
    const reason = provider?.degradation_reason || provider?.reason || '';
    const pill = {
      id: 'provider-downgrade', tone: 'warn', text: `${providerName(have)} ↓`,
      title: `Provider downgrade: ${want && want !== 'auto' ? `asked for ${providerName(want)}, ` : ''}the models are running on ${providerName(have)}.${reason ? ` ${reason}` : ''}`,
    };
    for (const n of FLOW_NODES) if (n.models) out[n.id].push(pill);
  }

  // TensorRT dynamic batching on the restorer.
  const batch = Number(settings.ultramax_batch_size);
  if (Number.isFinite(batch) && batch > 1 && !isOff(settings.selected_enhancer)) {
    out.restore_ultra.push({ id: 'trt-batch', tone: 'info', text: `TRT batch ×${batch}`,
      title: `UltraMax dynamic batching: ${batch} faces per TensorRT call.` });
  }

  // An engine being built, or verified, right now (run.engine_state, the same field the compile banner follows).
  if (run?.engine_state === 'COMPILING_TRT' || run?.engine_state === 'RUNNING_CANARY') {
    const building = run.engine_state === 'COMPILING_TRT';
    const what = [run.engine_model, run.engine_precision].filter(Boolean).join(' · ');
    out[nodeOfEngine(run.engine_model)].push({
      id: building ? 'trt-compile' : 'engine-canary', tone: 'warn', text: building ? 'TRT build' : 'engine check',
      title: `${building ? 'TensorRT is compiling an engine' : 'The fidelity canary is verifying an engine'}${what ? ` (${what})` : ''}. `
        + 'Latency is not representative until it finishes.',
    });
  }

  // Native restorer bypass (small faces on a native-resolution swapper skip the neural restorer for a DoG sharpen). Shown only when
  // the backend would really apply it: the toggle is on, the swapper is Alpha / HyperSwap / RealSwap and the restorer is one it replaces.
  if (settings.auto_bypass_restorer_native_swap && !isOff(settings.selected_enhancer)
      && isBypassSwapModel(settings.swap_model) && isBypassRestorer(settings.selected_enhancer)) {
    const ramp = Number(settings.ultramax_ramp);
    const hi = Number.isFinite(ramp) && ramp > 0 ? ramp : BYPASS_RAMP_HI_PX;
    const w = facePx == null ? 0 : bypassWeight(facePx, hi);
    const pct = Math.round(w * 100);
    if (w > 0) {
      out.restore_ultra.push({ id: 'native-bypass', tone: 'info', text: `native bypass ${pct}%`,
        title: `The tallest face is ${Math.round(facePx)} px: ${pct}% DoG sharpen and ${100 - pct}% neural restorer `
          + `(full bypass at ${BYPASS_RAMP_LO_PX} px or less, none at ${hi} px or more; ${Math.round((BYPASS_RAMP_LO_PX + hi) / 2)} px is the 50% point).` });
    } else {
      out.restore_ultra.push({ id: 'native-bypass-armed', tone: 'ok', text: 'bypass armed',
        title: `Auto-bypass is on: faces up to ${BYPASS_RAMP_LO_PX} px skip the neural restorer, fading to full restoration at ${hi} px${
          facePx != null ? ` (this face is ${Math.round(facePx)} px, so it is fully restored)` : ' (no face detected on this frame)'}.` });
    }
  }

  // Mouth custody fell back.
  if (run?.mouth_mode === 'audio_driven_sync' && run?.mouth_fallback) {
    out.gaze_mouth.push({ id: 'lipsync-off', tone: 'warn', text: 'lip-sync off',
      title: `Lip-sync off: ${run.mouth_fallback}. Source mouth restore is used.` });
  }

  // Hardware I/O fell back to a slower rung.
  const fb = Number(run?.io_fallbacks) || 0;
  if (fb > 0) {
    out.writer.push({ id: 'io-fallback', tone: 'warn', text: `HW fallback${fb > 1 ? ` ×${fb}` : ''}`,
      title: `A hardware video rung gave way to a slower one${run.io_fallback ? `: ${run.io_fallback}` : ''}.` });
  }
  return out;
}

/**
 * Whether the bar has anything to say about a node that the user must not miss (drives the dock's collapsed summary).
 * @returns {{ warn:number, info:number }}
 */
export function badgeCounts(badges) {
  let warn = 0;
  let info = 0;
  for (const list of Object.values(badges || {})) {
    for (const b of list) { if (b.tone === 'warn') warn += 1; else info += 1; }
  }
  return { warn, info };
}

/** The label the viewport's solo badge prints for any solo id. */
export function soloLabel(soloId) {
  if (!soloId) return '';
  const node = FLOW_NODES.find((n) => n.soloId === soloId);
  if (node) return nodeTitleStatic(node);
  const diffNode = FLOW_NODES.find((n) => n.diffId === soloId);
  if (diffNode) return `${diffNode.short} difference`;
  const pass = BUFFER_SOLO_PASSES.find((p) => p.id === soloId || p.key === soloId || p.alias === soloId);
  return pass ? pass.label : String(soloId);
}

const nodeTitleStatic = (n) => (n.id === 'source' ? 'Source frame' : n.label);
