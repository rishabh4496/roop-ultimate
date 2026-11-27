/**
 * Frame transport + player pipeline checks (plain node, no browser).
 *
 * What these pin, because each is a way the fast path could "work" while
 * showing the wrong thing:
 *   * the client reads the header at the offsets the server writes
 *     (app/tests/test_frames_ws.py pins the server side of the same bytes);
 *   * the pipeline is LATEST-WINS: a burst costs one decode per layer in flight
 *     plus the newest queued, never a backlog, and never paints an older frame
 *     over a newer one;
 *   * every ImageBitmap is closed exactly once (a decoded 1080p frame is ~8 MB
 *     of memory nothing else will free);
 *   * a clear() cannot be undone by a decode that was already running;
 *   * drawing happens on the scheduler's tick, not per frame received;
 *   * the telemetry store's derived values match what the tab used to compute.
 */
import process from 'node:process';
import {
  HEADER_BYTES, KIND_END, KIND_PLAY, FLAG_ERROR, packFrameMessage, parseFrameMessage,
} from '../src/transport/frameProtocol.js';
import { FramePipeline } from '../src/components/player/framePipeline.js';
import { fitContain } from '../src/components/player/glRenderer.js';
import {
  setRunTelemetry, resetRunTelemetry, getRun, framesOf, etaMsOf, selectProg, useTelemetryStore,
} from '../src/store/telemetryStore.js';

let failures = 0;
let checks = 0;
const ok = (name, cond, detail = '') => {
  checks += 1;
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

// ── wire format ───────────────────────────────────────────────────────────
console.log('── /ws/frames header ─────────────────────────────────────');
{
  const payload = new Uint8Array([0xff, 0xd8, 1, 2, 3]).buffer;
  const buf = packFrameMessage({ kind: KIND_PLAY, stream: 7, frame: 123, width: 960, height: 540 }, payload);
  ok('header is 20 bytes', HEADER_BYTES === 20 && buf.byteLength === 25);
  const v = new DataView(buf);
  ok('little-endian at the server offsets',
    v.getUint8(0) === 1 && v.getUint8(1) === KIND_PLAY && v.getUint32(4, true) === 7
    && v.getUint32(8, true) === 123 && v.getUint32(12, true) === 960 && v.getUint32(16, true) === 540);
  const m = parseFrameMessage(buf);
  ok('round trip', m && m.stream === 7 && m.frame === 123 && m.width === 960 && m.height === 540);
  ok('payload is a separate buffer (safe to transfer)',
    m.bytes.byteLength === 5 && m.bytes !== buf && new Uint8Array(m.bytes)[0] === 0xff);
  const end = parseFrameMessage(packFrameMessage({ kind: KIND_END, flags: FLAG_ERROR, stream: 2, frame: 9 }));
  ok('END carries the error flag', end.kind === KIND_END && (end.flags & FLAG_ERROR) === 1 && end.bytes.byteLength === 0);
  ok('short or foreign messages are refused',
    parseFrameMessage(new ArrayBuffer(4)) === null
    && parseFrameMessage((() => { const b = packFrameMessage({ kind: 1 }); new DataView(b).setUint8(0, 9); return b; })()) === null);
}

// ── pipeline ─────────────────────────────────────────────────────────────
console.log('── frame pipeline ────────────────────────────────────────');
const makeRig = () => {
  const closed = [];
  const uploads = [];
  let draws = 0;
  const pendingDecodes = [];
  const ticks = [];
  const renderer = {
    setLayer: (i, bmp, { owned }) => { uploads.push(bmp.id); if (owned) bmp.close(); return [bmp.width, bmp.height]; },
    clearLayer: () => {},
    draw: () => { draws += 1; return true; },
    destroy: () => {},
  };
  const decode = (bytes) => new Promise((resolve) => {
    const id = new Uint8Array(bytes)[0];
    pendingDecodes.push(() => resolve({ id, width: 4, height: 3, close() { closed.push(id); } }));
  });
  const pipe = new FramePipeline({ renderer, decode, schedule: (cb) => ticks.push(cb) });
  const bytes = (id) => new Uint8Array([id]).buffer;
  const settle = () => new Promise((r) => setTimeout(r, 0));
  const finishDecode = async () => { pendingDecodes.shift()?.(); await settle(); await settle(); };
  const tick = () => { const t = ticks.splice(0); t.forEach((cb) => cb()); };
  return { pipe, bytes, finishDecode, tick, closed, uploads, get draws() { return draws; }, pendingDecodes };
};

{
  const r = makeRig();
  // A burst of five frames while the first decode is running.
  for (let i = 1; i <= 5; i++) r.pipe.push(0, r.bytes(i), i, null);
  ok('one decode in flight per layer', r.pendingDecodes.length === 1);
  await r.finishDecode();                // frame 1 decoded -> pending; frame 5 starts
  ok('the newest queued frame is decoded next (2..4 skipped)', r.pendingDecodes.length === 1);
  await r.finishDecode();                // frame 5 decoded -> replaces pending 1 (closed)
  ok('a superseded pending frame is closed unseen', r.closed.includes(1));
  ok('nothing drawn before the scheduler ticks', r.draws === 0);
  r.tick();
  ok('one draw per tick, showing the newest', r.draws === 1 && r.uploads.join() === '5');
  ok('the drawn bitmap is closed after upload', r.closed.includes(5));
  ok('stats count the skipped frames', r.pipe.stats.dropped >= 3, JSON.stringify(r.pipe.stats));
  r.tick();
  ok('no draw when nothing changed', r.draws === 1);
}
{
  const r = makeRig();
  r.pipe.push(0, r.bytes(9), 10, null);
  await r.finishDecode();
  r.tick();
  r.pipe.pushBitmap(0, { id: 3, width: 4, height: 3, close() { r.closed.push(3); } }, 5, null);
  ok('an OLDER frame never replaces a newer one', r.closed.includes(3) && r.uploads.join() === '9');
}
{
  const r = makeRig();
  r.pipe.push(0, r.bytes(7), 1, null);
  r.pipe.clearLayer(0);                  // clear while frame 7 is still decoding
  await r.finishDecode();
  r.tick();
  ok('a decode that started before clear() does not come back', !r.uploads.includes(7) && r.closed.includes(7));
}
{
  const r = makeRig();
  r.pipe.push(0, r.bytes(1), 1, null);
  r.pipe.push(1, r.bytes(2), 1, null);
  ok('layers decode independently', r.pendingDecodes.length === 2);
  await r.finishDecode();
  await r.finishDecode();
  r.tick();
  ok('both layers land in one draw', r.draws === 1 && r.uploads.length === 2);
  r.pipe.push(0, r.bytes(3), 2, null);
  r.pipe.dispose();
  await r.finishDecode();
  ok('a decode finishing after dispose is closed, not drawn', r.closed.includes(3) && r.uploads.length === 2);
}

// ── prefetch window ──────────────────────────────────────────────────────
console.log('── playback prefetch window ──────────────────────────────');
{
  const { nextNeededFrame } = await import('../src/components/faceswap/playbackWindow.js');
  const buffered = (lo, hi) => (f) => f >= lo && f <= hi;
  ok('asks for the first gap in the look-ahead',
    nextNeededFrame({ cur: 300, start: 1, end: 27555, ahead: 120, isLooping: true, has: buffered(300, 350) }) === 351);
  ok('a FULL look-ahead asks for nothing (not the long-evicted first frame)',
    nextNeededFrame({ cur: 300, start: 1, end: 27555, ahead: 120, isLooping: true, has: buffered(292, 420) }) === null,
    'the loop-wrap scan ran at overflow 0 and re-requested frame `start`');
  ok('near the out point, looping warms the wrap-around frames',
    nextNeededFrame({ cur: 950, start: 1, end: 1000, ahead: 120, isLooping: true, has: buffered(950, 1000) }) === 1);
  ok('...and not when looping is off',
    nextNeededFrame({ cur: 950, start: 1, end: 1000, ahead: 120, isLooping: false, has: buffered(950, 1000) }) === null);
  ok('the wrap stretch stops at the overflow',
    nextNeededFrame({ cur: 950, start: 1, end: 1000, ahead: 120, isLooping: true,
      has: (f) => (f >= 950 && f <= 1000) || (f >= 1 && f <= 71) }) === null);
}

// ── contain fit ──────────────────────────────────────────────────────────
console.log('── contain fit ───────────────────────────────────────────');
{
  const [x, y, w, h] = fitContain(1920, 1080, [0, 0, 1, 1], 1000, 1000);
  ok('16:9 in a square is letterboxed vertically', Math.abs(w - 1) < 1e-9 && Math.abs(h - 0.5625) < 1e-9 && Math.abs(y - 0.21875) < 1e-9 && Math.abs(x) < 1e-9);
  const [sx, , sw] = fitContain(1920, 1080, [0.5, 0, 0.5, 1], 2000, 562.5);
  ok('side-by-side right half starts at the midline', Math.abs(sx - 0.5) < 1e-9 && Math.abs(sw - 0.5) < 1e-9);
}

// ── telemetry store ──────────────────────────────────────────────────────
console.log('── telemetry store ───────────────────────────────────────');
{
  resetRunTelemetry();
  let notified = 0;
  const unsub = useTelemetryStore.subscribe(() => { notified += 1; });
  setRunTelemetry({ progress: 0.5, desc: 'Processing frame 10 / 40', processing: true, log: ['x'] });
  ok('fast fields are stored, structural/heavy ones are not', getRun().progress === 0.5 && getRun().processing === undefined && getRun().log === undefined);
  setRunTelemetry({ progress: 0.5, desc: 'Processing frame 10 / 40' });
  ok('an identical frame does not notify subscribers', notified === 1);
  unsub();
  ok('frames parsed from the status line when counters are absent',
    JSON.stringify(framesOf({ desc: 'Processing frame 52,450 / 88,483' })) === '{"done":52450,"total":88483}');
  ok('pushed counters win', JSON.stringify(framesOf({ current_frame: 3, total_frames: 9, desc: '1 / 2' })) === '{"done":3,"total":9}');
  ok('eta prefers the terminal eta_s', etaMsOf({ eta_s: 12, progress: 0.5 }, 1000) === 12000);
  ok('eta extrapolates without one', etaMsOf({ eta_s: null, progress: 0.5 }, 1000) === 1000);
  ok('progress is clamped', selectProg({ run: { progress: 1.4 } }) === 1 && selectProg({ run: { progress: -0.2 } }) === 0);
}

console.log(`\n${failures === 0 ? 'ALL GREEN' : 'FAILURES'}: ${checks - failures}/${checks} checks passed`);
process.exit(failures === 0 ? 0 : 1);
