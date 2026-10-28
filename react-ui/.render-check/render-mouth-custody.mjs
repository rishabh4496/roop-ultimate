/**
 * Mouth custody's pure logic (plain node): the legacy migration, the request fields and the slip arithmetic.
 *
 *   node .render-check/run-module.mjs render-mouth-custody.mjs
 *
 * The cases in `mouth-custody.cases.json` are also run against app/roop/mouth_custody.py by
 * app/tests/test_mouth_custody_js.py, so the two implementations cannot drift apart unnoticed.
 */
import fs from 'node:fs';
import * as M from '../src/components/faceswap/mouthCustody.js';
import { FACESWAP_DEFAULTS } from '../src/components/faceswap/defaults.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

const cases = JSON.parse(fs.readFileSync(new URL('./mouth-custody.cases.json', import.meta.url), 'utf8'));
for (const c of cases.derive) ok(`modeOf ${JSON.stringify(c.params)} -> ${c.mode}`, M.modeOf(c.params) === c.mode, `got ${M.modeOf(c.params)}`);
for (const c of cases.clamp) ok(`clampOffset(${JSON.stringify(c.in)}) -> ${c.out}`, M.clampOffset(c.in) === c.out, `got ${M.clampOffset(c.in)}`);

// withMode writes the legacy keys the mode means, and modeOf reads the result back as the same mode
for (const mode of M.MODES) {
  const p = M.withMode({ lipsync_enabled: true, restore_original_mouth: true }, mode);
  ok(`withMode(${mode}) round-trips`, M.modeOf(p) === mode);
  ok(`withMode(${mode}) legacy keys`, p.lipsync_enabled === (mode === 'audio_driven_sync') && p.restore_original_mouth === (mode === 'source_mouth_restore'));
}
ok('withMode(unknown) is off', M.modeOf(M.withMode({}, 'nope')) === 'off');

// an old recipe is not turned off by opening it - including after the defaults are merged UNDER it
ok('defaults carry no explicit mode', !('mouth_custody_mode' in FACESWAP_DEFAULTS));
ok('old lip-sync recipe over defaults stays audio-driven', M.modeOf({ ...FACESWAP_DEFAULTS, lipsync_enabled: true }) === 'audio_driven_sync');
ok('old restore recipe over defaults stays source_mouth_restore', M.modeOf({ ...FACESWAP_DEFAULTS, restore_original_mouth: true }) === 'source_mouth_restore');
ok('untouched defaults are off', M.modeOf(FACESWAP_DEFAULTS) === 'off');
ok('default slip is zero and unbound', FACESWAP_DEFAULTS.audio_offset_ms === 0 && FACESWAP_DEFAULTS.override_audio_path === null);
ok('legacy lip-sync recipe stays audio-driven', M.requestFields({ lipsync_enabled: true }).mouth_custody_mode === 'audio_driven_sync');
ok('legacy upload pair becomes the override', M.requestFields({ lipsync_enabled: true, lipsync_audio_source: 'upload', lipsync_audio_path: 'C:/t/dub.wav' }).override_audio_path === 'C:/t/dub.wav');
ok('legacy original source has no override', M.requestFields({ lipsync_enabled: true, lipsync_audio_source: 'original', lipsync_audio_path: 'C:/stale.wav' }).override_audio_path === null);
ok('requestFields always carries the mode', 'mouth_custody_mode' in M.requestFields({}));

// override
const bound = M.withOverride({ mouth_custody_mode: 'audio_driven_sync' }, 'C:/t/dub.wav');
ok('withOverride binds all three keys', bound.override_audio_path === 'C:/t/dub.wav' && bound.lipsync_audio_path === 'C:/t/dub.wav' && bound.lipsync_audio_source === 'upload');
const cleared = M.withOverride(bound, null);
ok('withOverride(null) clears them', cleared.override_audio_path === null && cleared.lipsync_audio_source === 'original');

// text
ok('offsetText +', M.offsetText(120) === '+120 ms');
ok('offsetText - uses a true minus', M.offsetText(-40) === '\u221240 ms');
ok('offsetText 0', M.offsetText(0) === '0 ms');
ok('offsetMeaning +', /later/.test(M.offsetMeaning(250)) && /250/.test(M.offsetMeaning(250)));
ok('offsetMeaning -', /earlier/.test(M.offsetMeaning(-250)));
ok('trackShiftSeconds', M.trackShiftSeconds({ audio_offset_ms: -250 }) === -0.25);

// the advisory (silent video)
ok('silent video + audio mode warns', M.advisory({ mouth_custody_mode: 'audio_driven_sync' }, { hasTargetAudio: false })?.tone === 'warn');
ok('silent video + a bound track does not warn', M.advisory(M.withOverride({ mouth_custody_mode: 'audio_driven_sync' }, 'C:/t/dub.wav'), { hasTargetAudio: false }) === null);
ok('unknown audio presence does not warn', M.advisory({ mouth_custody_mode: 'audio_driven_sync' }, { hasTargetAudio: null }) === null);
ok('other modes never warn', M.advisory({ mouth_custody_mode: 'source_mouth_restore' }, { hasTargetAudio: false }) === null);
ok('an image warns in audio mode', M.advisory({ mouth_custody_mode: 'audio_driven_sync' }, { isImage: true })?.tone === 'warn');

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
globalThis.__RENDER_CHECK_FAILURES__ = failures;
