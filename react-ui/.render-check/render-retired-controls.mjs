/**
 * Phase 12: controls that were removed stay removed, and what replaced them behaves.
 *
 *   1  Identity Blender   one attribute dial (Feature Dominance / jaw) - the only direction identity_render_validation.json
 *                         proves writable through the swapper; Age Shift, Femininity <-> Masculinity and Expression Intensity
 *                         are gone; a stale recipe (server reply, older save) loses the removed dials (normalizeBlend)
 *   2  Sparse pre-scan    no toggle in #/settings, #/faceswap or #/batch, and no stride / worker input anywhere
 *   3  Disk frames / audio  Skip audio, Keep frames, Wait before creating video and the Video method select are not in the
 *                         Face Swap panel, its defaults or the request it builds; the batch matrix sends no video_method
 *   4  Precision          the global precision select offers mixed and fp32 only (never fp16), and the Max Turbo copy
 *                         does not promise FP16
 *   5  The pill is mounted on #/processing and #/faceswap
 *
 * The source pins read the files themselves: a control that returns under another name is a different bug, but the
 * strings these controls shipped with must not reappear.
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
const blend = await import('../src/components/faceswap/identityBlend.js');
const IdentityBlender = (await import('../src/components/faceswap/IdentityBlender.jsx')).default;
const defaults = await import('../src/components/faceswap/defaults.js');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const html = (el) => renderToStaticMarkup(el).replace(/<!-- -->/g, '');

console.log('── 1. Identity Blender ────────────────────────────────────');
{
  // enabled: the section is collapsed while the blend is off, and a collapsed section would make every "not on the panel"
  // check below pass for the wrong reason - so the sentinel proves the dials' section is really in the markup.
  const h = html(React.createElement(IdentityBlender, { sourceFaces: [], sourceFacesInfo: [], value: { ...blend.EMPTY_BLEND, enabled: true }, onChange: () => {} }));
  ok('sentinel: the open section is rendered (the checks below can fail)', h.includes('Blend identities') && h.includes('Reset blender'));
  for (const gone of ['Age Shift', 'Femininity', 'Masculinity', 'Expression Intensity']) {
    ok(`"${gone}" is not on the panel`, !h.includes(gone));
  }
  ok('the jaw dial is: "Feature Dominance (Jaw)"', h.includes('Feature Dominance (Jaw)'));
  ok('two sliders remain: the jaw dial and the identity guard', (h.match(/type="range"/g) || []).length === 2, `ranges: ${(h.match(/type="range"/g) || []).length}`);
  ok('the identity guard defaults to cosine 0.80', blend.EMPTY_BLEND.min_cosine === 0.8 && h.includes('Identity guard (min cosine)'));
  ok('the multi-source blend weight sliders are the per-source ones (bound to a source, 0-100)', /Weight of/.test(src('components/faceswap/IdentityBlender.jsx')));
}
{
  ok('BLEND_DIALS is exactly [jawline]', JSON.stringify(blend.BLEND_DIALS) === '["jawline"]');
  const stale = { enabled: true, components: [{ source_id: 'a', weight: 40 }], dials: { age: 12, gender: -0.5, jawline: 0.4, expression: 1 }, min_cosine: 0.85 };
  const n = blend.normalizeBlend(stale);
  ok('a stale recipe loses age / gender / expression and keeps jawline', JSON.stringify(n.dials) === '{"jawline":0.4}' && n.min_cosine === 0.85 && n.enabled && n.components.length === 1, JSON.stringify(n));
  ok('jawline is clamped to ±1 and a non-number is 0', blend.normalizeBlend({ dials: { jawline: 7 } }).dials.jawline === 1 && blend.normalizeBlend({ dials: { jawline: 'x' } }).dials.jawline === 0);
  ok('null / garbage normalises to the empty recipe', JSON.stringify(blend.normalizeBlend(null)) === JSON.stringify(blend.normalizeBlend(blend.EMPTY_BLEND)) && blend.normalizeBlend(5).min_cosine === 0.8);
  ok('normalizeBlend returns a fresh object (the frozen default is never mutated)', blend.normalizeBlend(blend.EMPTY_BLEND) !== blend.EMPTY_BLEND && Object.isFrozen(blend.EMPTY_BLEND));
  const fs = src('components/FaceSwap.jsx');
  ok('the recipe the server returns goes through normalizeBlend', /setIdentityBlend\(normalizeBlend\(r\)\)/.test(fs));
}

console.log('── 2. sparse pre-scan ─────────────────────────────────────');
const SOURCES = ['components/Settings.jsx', 'components/FaceSwap.jsx', 'components/BatchSwap.jsx', 'components/faceswap/defaults.js',
  'components/faceswap/batchMatrix.js', 'components/settingsCatalog.js', 'components/QualityProfilesModal.jsx', 'components/settingsDiff.js'];
{
  for (const rel of SOURCES) ok(`${rel}: no sparse toggle, stride or worker input`, !/sparse_auto_range|Sparse auto-range|sparse_stride|sparse_workers|prescan_stride|Stage 1 stride|detector workers/i.test(src(rel)), rel);
}

console.log('── 3. disk frames / skip audio ────────────────────────────');
{
  const RETIRED = ['skip_audio', 'keep_frames', 'wait_after_extraction', 'video_swapping_method', 'video_method',
    'Skip audio', 'Keep frames', 'Wait before creating video', 'Video method', 'Temp Frame Format', 'temp_frame_format'];
  for (const rel of SOURCES) {
    const code = src(rel);
    const hits = RETIRED.filter((t) => code.includes(t));
    ok(`${rel}: none of the retired disk-frame / audio / video-method controls`, hits.length === 0, hits.join(', '));
  }
  for (const k of ['skip_audio', 'keep_frames', 'wait_after_extraction', 'video_swapping_method']) {
    ok(`defaults.js has no "${k}"`, !(k in defaults.FACESWAP_DEFAULTS));
  }
}

console.log('── 4. precision ───────────────────────────────────────────');
{
  const s = src('components/Settings.jsx');
  ok('the precision select offers mixed and fp32 only', /options=\{m\.trt_precisions \?\? \['mixed', 'fp32'\]\}/.test(s) && !/'fp16', 'mixed'/.test(s) && !/\['fp32', 'fp16'/.test(s));
  ok('its help text no longer says "fp16 = fastest"', !/fp16 = fastest/.test(s));
  ok('Max Turbo no longer promises "TensorRT FP16"', !/TensorRT FP16/.test(s));
}

console.log('── 5. the pill is mounted ─────────────────────────────────');
{
  for (const rel of ['components/Processing.jsx', 'components/FaceSwap.jsx']) {
    const code = src(rel);
    ok(`${rel} imports and renders <PipelinePill>`, /import PipelinePill from '\.\/faceswap\/PipelinePill'/.test(code) && /<PipelinePill\b/.test(code), rel);
  }
}

console.log(failures === 0 ? '\nall retired-control checks passed' : `\n${failures} retired-control check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
