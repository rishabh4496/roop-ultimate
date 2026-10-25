/**
 * The HUD's Video I/O line (telemetryStore.videoIoTextOf) and the fields behind it.
 *
 * The defect this pins: a hardware rung (NVDEC, NVENC) that gave way to a slower one mid-run was recorded in a list the
 * HUD never read, so the badge kept saying the render was hardware-accelerated. The backend now rewrites the backend
 * label AND counts the fallback (io_fallbacks / io_fallback in the 4 Hz frame); the line has to show both.
 *
 * Run: node .render-check/run-module.mjs video-io-hud-check.mjs
 */
import { videoIoTextOf, FAST_FIELDS, getRun } from '../src/store/telemetryStore.js';

// The store's initial `run` IS its EMPTY_RUN (not exported).
const EMPTY_RUN = getRun();

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

console.log('── the line ───────────────────────────────────────────────');
ok('idle (nothing registered) renders nothing', videoIoTextOf(EMPTY_RUN) === '' && videoIoTextOf(null) === '');
const healthy = { ...EMPTY_RUN, io_decode: 'NVDEC-DLPack', io_encode: 'NVENC-Direct', io_pcie_saved_mb: 2480, io_queue_sat: 74 };
ok('a healthy hardware run reads as before',
  videoIoTextOf(healthy) === 'Video I/O: NVDEC-DLPack → NVENC-Direct · PCIe saved ~2480 MB (est.) · decode queue 74%',
  videoIoTextOf(healthy));
ok('a healthy run carries no warning', !/FALLBACK|⚠/.test(videoIoTextOf(healthy)));

const fell = { ...healthy, io_decode: 'CPU-Fallback', io_fallbacks: 1,
  io_fallback: 'stage1-keyframes: NVDEC-DLPack -> CPU-Fallback (sparse range scan: ValueError: stride)' };
const text = videoIoTextOf(fell);
ok('a fallback names itself', /⚠ HARDWARE FALLBACK: stage1-keyframes/.test(text), text);
ok('a fallback shows the backend that now does the work, not the one that was planned',
  text.includes('CPU-Fallback') && !text.includes('NVDEC-DLPack →'), text);
ok('a single fallback has no multiplier; several do',
  !/×/.test(text) && /HARDWARE FALLBACK ×3/.test(videoIoTextOf({ ...fell, io_fallbacks: 3 })));
ok('a fallback is shown even before a backend label exists',
  /HARDWARE FALLBACK/.test(videoIoTextOf({ ...EMPTY_RUN, io_fallbacks: 1, io_fallback: 'x' })));
ok('a non-numeric count cannot raise the badge', videoIoTextOf({ ...healthy, io_fallbacks: 'abc' }) === videoIoTextOf(healthy));

console.log('── the frame carries the fields ───────────────────────────');
ok('io_fallbacks / io_fallback are fast fields (a telemetry frame may update them)',
  FAST_FIELDS.includes('io_fallbacks') && FAST_FIELDS.includes('io_fallback'));
ok('they start quiet', EMPTY_RUN.io_fallbacks === 0 && EMPTY_RUN.io_fallback === '');

console.log(failures === 0 ? '\nall video-io-hud checks passed' : `\n${failures} video-io-hud check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
