/**
 * The Performance group on #/faceswap, the .fsz pose-coverage badge, and the flamegraph's derived residual lane.
 *
 *   1  performance rules   each switch reads and writes the REAL setting (tristate / mixed<->fp32 / codec pair / batch
 *                          ceiling), an unusual value reads honestly, and every key written exists in the generated
 *                          backend schema with an env mapping where the backend reads one
 *   2  the component       renders the five controls with the shared primitives, no run button, and the notes follow
 *                          the settings (FP16 on a non-TensorRT provider, batching off, a codec with no NVENC twin)
 *   3  coverage badge      unknown is not "front only", partial names the missing side, complete lights all three,
 *                          and a profile is told apart from a three-quarter view
 *   4  the call sites      Face Swap mounts the group, both .fsz galleries mount the badge, Settings binds the new key
 *                          and the catalog carries it
 *   5  flamegraph          the residual lane is end-to-end minus the measured lanes, never negative, reports
 *                          over-attribution instead of hiding it, and no alias is claimed by two lanes (no double count)
 *
 * Pure rules are imported; components are rendered to markup with react-dom/server. Nothing here needs a backend.
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
const schema = JSON.parse(readFileSync(join(here, '..', '..', 'app', 'settings.schema.json'), 'utf8')).settings;

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const rules = await import('../src/components/faceswap/performanceRules.js');
const PerformanceQuickControls = (await import('../src/components/faceswap/PerformanceQuickControls.jsx')).default;
const cov = await import('../src/components/facemgr/poseCoverage.js');
const PoseCoverageBadge = (await import('../src/components/facemgr/PoseCoverageBadge.jsx')).default;
const fg = await import('../src/components/faceswap/flamegraphModel.js');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const html = (el) => renderToStaticMarkup(el).replace(/<!-- -->/g, '');
const text = (h) => h.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

console.log('── 1. performance rules ───────────────────────────────────');
{
  ok('zero-copy reads ON for auto / on / unset, OFF only for off',
    rules.zeroCopyOn({}) && rules.zeroCopyOn({ perf_zerocopy_enhance: 'auto' }) && rules.zeroCopyOn({ perf_zerocopy_enhance: 'on' })
    && !rules.zeroCopyOn({ perf_zerocopy_enhance: 'off' }) && !rules.zeroCopyOn({ perf_zerocopy_enhance: ' OFF ' }));
  ok('turning a tristate ON writes auto (leave-alone), OFF writes off',
    rules.zeroCopyPatch(true).perf_zerocopy_enhance === 'auto' && rules.zeroCopyPatch(false).perf_zerocopy_enhance === 'off'
    && rules.decodePatch(true).perf_nvdec === 'auto' && rules.decodePatch(false).perf_nvdec === 'off');
  ok('FP16 is mixed <-> fp32 and never a third value',
    rules.fp16Patch(true).trt_precision === 'mixed' && rules.fp16Patch(false).trt_precision === 'fp32'
    && rules.fp16On({ trt_precision: 'mixed' }) && !rules.fp16On({ trt_precision: 'fp32' }) && rules.fp16On({ trt_precision: 'fp16' }));
  ok('the precision switch only applies on TensorRT',
    rules.precisionApplies({ provider: 'tensorrt' }) && !rules.precisionApplies({ provider: 'cuda' }) && !rules.precisionApplies({}));

  ok('hardware encode flips only the pairs that keep the codec family',
    rules.encodePatch({ output_video_codec: 'libx264' }, true).output_video_codec === 'h264_nvenc'
    && rules.encodePatch({ output_video_codec: 'libx265' }, true).output_video_codec === 'hevc_nvenc'
    && rules.encodePatch({ output_video_codec: 'h264_nvenc' }, false).output_video_codec === 'libx264'
    && rules.encodePatch({ output_video_codec: 'hevc_nvenc' }, false).output_video_codec === 'libx265');
  ok('a codec with no NVENC twin is never rewritten (empty patch, switch disabled)',
    Object.keys(rules.encodePatch({ output_video_codec: 'prores_ks' }, true)).length === 0
    && Object.keys(rules.encodePatch({ output_video_codec: 'h264_qsv' }, false)).length === 0
    && !rules.encodeSwitchable({ output_video_codec: 'h264_qsv' }) && !rules.encodeSwitchable({ output_video_codec: 'prores_ks' })
    && rules.encodeSwitchable({ output_video_codec: 'libx264' }) && rules.encodeSwitchable({ output_video_codec: 'hevc_nvenc' }));
  ok('QSV / AMF read as hardware encode but are not flipped', rules.encodeOn({ output_video_codec: 'h264_amf' }) && rules.encoderOf({ output_video_codec: 'h264_amf' }) === 'hardware');

  ok('micro-batch normalises to auto or a positive integer string',
    rules.batchValue({}) === 'auto' && rules.batchValue({ perf_batch_max: 'auto' }) === 'auto' && rules.batchValue({ perf_batch_max: 4 }) === '4'
    && rules.batchValue({ perf_batch_max: '0' }) === 'auto' && rules.batchValue({ perf_batch_max: 'junk' }) === 'auto');
  ok('a batch ceiling outside the short list is kept in the options, not hidden',
    rules.batchOptions({ perf_batch_max: 6 }).includes('6') && rules.batchOptions({ perf_batch_max: 4 }).length === rules.BATCH_CHOICES.length);
  ok('batch patch writes auto or an integer string', rules.batchPatch('auto').perf_batch_max === 'auto' && rules.batchPatch('8').perf_batch_max === '8');
  ok('the batch note says so when batching itself is off', /no effect/i.test(rules.batchNote({ perf_batch_swap: 'off' })));

  const keys = ['perf_zerocopy_enhance', 'perf_nvdec', 'trt_precision', 'output_video_codec', 'perf_batch_max', 'perf_batch_swap', 'perf_pinned_buffers'];
  ok('every key the rules read or write exists in the generated backend schema', keys.every((k) => schema[k]), keys.filter((k) => !schema[k]).join());
  const env = (k) => schema[k] && schema[k].env && schema[k].env.var;
  ok('the zero-copy / NVDEC / batch keys are mapped to the env vars the backend reads',
    env('perf_zerocopy_enhance') === 'ROOP_ZEROCOPY_ENHANCE' && env('perf_nvdec') === 'ROOP_NVDEC' && env('perf_batch_max') === 'ROOP_BATCH_SWAP_MAX');
  ok('a stored older config (key absent) defaults to auto in the schema', schema.perf_zerocopy_enhance.default === 'auto');
}

console.log('── 2. the component ───────────────────────────────────────');
{
  const render = (values) => html(React.createElement(PerformanceQuickControls, { values, onChangeMany: () => {} }));
  const h = render({ provider: 'tensorrt', trt_precision: 'mixed', output_video_codec: 'h264_nvenc' });
  const t = text(h);
  ok('renders the five controls', ['Zero-copy restorer buffers', 'FP16 precision', 'Hardware decode', 'Hardware encode', 'Micro-batch size'].every((s) => t.includes(s)), t.slice(0, 200));
  // The only <button>s are the shared primitives' info badges (one per control); there is no run / start action.
  const buttons = h.match(/<button[^>]*>[\s\S]*?<\/button>/g) || [];
  ok('has no run button (every button is an info badge, none says run / start / render)',
    buttons.length === 5 && buttons.every((b) => !/>\s*(run|start|render|apply)\b/i.test(text(b)) && /aria-label=/.test(b)), String(buttons.length));
  ok('the checkboxes reflect the settings (NVENC on, mixed on)', (h.match(/type="checkbox"[^>]*checked/g) || []).length >= 3);

  const cuda = text(render({ provider: 'cuda', trt_precision: 'mixed' }));
  ok('FP16 note says it has no effect off TensorRT', /no effect/i.test(cuda) && /not TensorRT/i.test(cuda));
  const trt = text(render({ provider: 'tensorrt' }));
  ok('FP16 note names the FP32-pinned models on TensorRT', /stay FP32/i.test(trt));
  const off = render({ perf_batch_swap: 'off' });
  ok('batch note follows perf_batch_swap', /no effect/i.test(text(off)));
  const prores = render({ output_video_codec: 'prores_ks' });
  ok('a non-switchable codec disables the encode switch and explains it', /disabled/.test(prores) && /no NVENC twin/i.test(text(prores)));
}

console.log('── 3. pose-coverage badge ─────────────────────────────────');
{
  const full = { left: 2, front: 3, right: 1, deep_left: 1, deep_right: 0, unsolved: 0, total: 6, missing: [], complete: true };
  const frontal = { left: 0, front: 4, right: 0, deep_left: 0, deep_right: 0, unsolved: 0, total: 4, missing: ['left', 'right'], complete: false };
  const oneSided = { left: 3, front: 1, right: 0, deep_left: 2, deep_right: 0, unsolved: 1, total: 5, missing: ['right'], complete: false };

  ok('no pose_coverage is "unknown", not front-only and not complete',
    cov.coverageBadge({}).state === 'unknown' && cov.coverageBadge({ pose_coverage: null }).state === 'unknown'
    && cov.coverageBadge(null).state === 'unknown' && cov.coverageBadge({ pose_coverage: 'x' }).state === 'unknown');
  ok('a coverage object with nothing solved is unknown', cov.coverageBadge({ pose_coverage: { left: 0, front: 0, right: 0, unsolved: 3, total: 3 } }).state === 'unknown');
  ok('complete lights all three sides', cov.coverageBadge({ pose_coverage: full }).state === 'complete' && cov.coverageBadge({ pose_coverage: full }).pips.every((p) => p.lit));
  const f = cov.coverageBadge({ pose_coverage: frontal });
  ok('front-only is partial and names both missing sides', f.state === 'partial' && /left\/right/.test(f.text) && /Missing left and right/.test(f.title), f.title);
  const o = cov.coverageBadge({ pose_coverage: oneSided });
  ok('one-sided names the missing side; profiles are told apart from three-quarter views',
    o.state === 'partial' && /No right/.test(o.text) && /2 true profiles/.test(o.title) && /1 reference with no solved pose/.test(o.title), o.title);
  ok('the deep count is not applied to the front pip', cov.coverageBadge({ pose_coverage: full }).pips.find((p) => p.side === 'front').deep === 0);

  const mk = (entry) => html(React.createElement(PoseCoverageBadge, { entry }));
  const hf = mk({ pose_coverage: frontal });
  ok('the badge renders three pips with lit state and an accessible name',
    (hf.match(/data-side=/g) || []).length === 3 && /data-state="partial"/.test(hf) && /role="img"/.test(hf)
    && /data-side="front" data-lit="true"/.test(hf) && /data-side="left" data-lit="false"/.test(hf));
  ok('an unknown badge is dashed, not amber-warning', /data-state="unknown"/.test(mk({})) && /border-dashed/.test(mk({})) && !/amber/.test(mk({})));
}

console.log('── 4. the call sites ──────────────────────────────────────');
{
  const fs = src('components/FaceSwap.jsx');
  ok('Face Swap imports and mounts the Performance group (collapsed by default)',
    /import PerformanceQuickControls from '\.\/faceswap\/PerformanceQuickControls'/.test(fs)
    && /<InspectorGroup id="performance" title="Performance"[^>]*>\s*<PerformanceQuickControls values=\{p\} onChangeMany=\{setMany\} \/>/.test(fs)
    && !/id="performance"[^>]*defaultOpen/.test(fs));
  ok('Face Swap still mounts the existing stability control and pipeline controls', /<TemporalModeControl /.test(fs) && /<PipelineControls /.test(fs));
  const iw = src('components/facemgr/IdentityWorkstation.jsx');
  const lib = src('components/faceswap/FacesetLibrary.jsx');
  ok('the #/facemgr library list mounts the badge', /import PoseCoverageBadge/.test(iw) && /<PoseCoverageBadge entry=\{e\} \/>/.test(iw));
  ok('the Face Swap / Batch library picker mounts the badge', /import PoseCoverageBadge/.test(lib) && /<PoseCoverageBadge entry=\{e\}/.test(lib));
  const st = src('components/Settings.jsx');
  ok('Settings binds the zero-copy key literally (test_ui_settings_catalog reads it)', /bind\('perf_zerocopy_enhance', 'auto'\)/.test(st));
  ok('the generated catalog carries the new key', /perf_zerocopy_enhance/.test(src('components/settingsCatalog.js')));
  ok('Settings still mounts the TensorRT cache panel (status + clear)', /<TrtCachePanel /.test(st));
}

console.log('── 5. flamegraph residual lane ────────────────────────────');
{
  const lanes = fg.TRACE_STAGES;
  const derived = lanes.filter((s) => s.derived);
  ok('exactly one derived lane, last, with no aliases', derived.length === 1 && derived[0].key === 'other' && lanes[lanes.length - 1].key === 'other' && derived[0].aliases.length === 0);
  const seen = new Map();
  const dup = [];
  for (const s of lanes) for (const a of s.aliases) { if (seen.has(a)) dup.push(`${a} in ${seen.get(a)} and ${s.key}`); seen.set(a, s.key); }
  ok('no profiler name is claimed by two lanes (no double count)', dup.length === 0, dup.join('; '));

  const a = fg.traceStageMs({ detect: 10, swap: 20, mask: 30, blend: 10, stabilize: 5, verify: 3 }, 100);
  // detect 10 + swap 20 + mask 30 + blend 10 = 70 measured. stabilize (5) and verify (3) are deliberately unmapped, so they
  // are part of the residual: 100 - 70 = 30.
  ok('residual = end-to-end minus the measured lanes (unmapped timers fall into it)',
    a.stageMs.other === 30 && a.measuredMs === 70 && a.overAttributedMs === 0, JSON.stringify(a));
  ok('a lane with no measurement stays null, not 0', a.stageMs.restore === null && a.stageMs.decode === null);
  const b = fg.traceStageMs({ swap: 80, mask: 40 }, 100);
  ok('over-attribution clamps the residual to 0 and reports the excess', b.stageMs.other === 0 && b.overAttributedMs === 20, JSON.stringify(b));
  const c = fg.traceStageMs({ swap: 10 }, null);
  ok('no end-to-end figure -> no residual (null, not a guess)', c.stageMs.other === null && c.overAttributedMs === 0);
  const d = fg.traceStageMs({ detect: 20, arcface_embedding: 4 }, 40);
  ok('the nested ArcFace timer is subtracted from detection exactly once', d.stageMs.detection === 16 && d.stageMs.embedding === 4 && d.stageMs.other === 20, JSON.stringify(d));
  ok('the derived lane is not an input: a raw "other" key is ignored', fg.traceStageMs({ other: 999, swap: 10 }, 50).stageMs.other === 40);
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
