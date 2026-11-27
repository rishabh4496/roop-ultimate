/**
 * Retry policy for frame requests (plain node, no browser).
 *
 * The defect this pins: a failed frame/chunk request was treated as "still
 * wanted, so ask again" -- from `finally` through a 150 ms throttle in
 * useThrottledFrameRequest, from the next animation frame in usePlaybackBuffer.
 * A backend answering 500 got ~6.5 preview requests a second from an idle tab
 * (196 in 30 s, react-ui/e2e/idle-requests.spec.js). So:
 *   * each consecutive failure waits longer on a DETERMINISTIC schedule: 250, 500, 1000 ms;
 *   * the fourth (the first attempt + 3 retries) gives up, and stays given up until the key
 *     changes or reset();
 *   * a success ends the streak; a different key starts from zero;
 *   * an AbortError is not a failure;
 *   * both hooks actually go through the policy (source-level wiring, since the
 *     hooks need a DOM to run -- the e2e suite exercises them for real).
 */
import process from 'node:process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
  RETRY_DELAYS_MS, MAX_RETRIES, MAX_FAILURES, backoffDelay, createFailureTracker, isAbort, isSuperseded, isSuperseding,
  BUSY_MIN_MS, BUSY_MAX_MS, BUSY_DEFAULT_MS, busyRetryMs, httpFailure, isBusy, busyDelay,
} from '../src/components/faceswap/retryPolicy.js';

let failures = 0;
let checks = 0;
const ok = (name, cond, detail = '') => {
  checks += 1;
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

const here = dirname(fileURLToPath(import.meta.url));
// LF-normalised: a core.autocrlf checkout has CRLF, and the wiring checks match
// multi-line text.
const src = (name) => readFileSync(join(here, '..', 'src', 'components', 'faceswap', name), 'utf8')
  .replace(/\r\n/g, '\n');

console.log('── backoff schedule ───────────────────────────────────────');
{
  ok('constants are the specified 250/500/1000 ms, 3 retries, 4 attempts',
    RETRY_DELAYS_MS.join() === '250,500,1000' && MAX_RETRIES === 3 && MAX_FAILURES === 4);
  ok('the schedule is exponential: 250, 500, 1000 ms', [1, 2, 3].map((n) => backoffDelay(n)).join() === '250,500,1000');
  ok('it is deterministic: the same count always gives the same delay',
    [1, 2, 3].every((n) => backoffDelay(n) === backoffDelay(n) && backoffDelay(n) === backoffDelay(n, () => 0) && backoffDelay(n) === backoffDelay(n, () => 1)));
  ok('a count past the schedule stays at the cap (1 s), never grows',
    [4, 5, 8, 50].every((n) => backoffDelay(n) === 1000));
  ok('nonsense counts fall back to the first delay, never NaN or negative',
    [0, -3, NaN, undefined].every((n) => backoffDelay(n) === 250));
  const total = [1, 2, 3].reduce((s, n) => s + backoffDelay(n), 0);
  ok('one request + three retries take 1.75 s of waiting in total', total === 1750, `got ${total}`);
}

console.log('── failure tracker ────────────────────────────────────────');
{
  const t = createFailureTracker();
  const seen = [];
  for (let i = 0; i < 4; i++) seen.push(t.fail('a'));
  ok('failures 1-3 carry the 250 / 500 / 1000 ms delay and are not exhausted',
    seen.slice(0, 3).every((f, i) => f.count === i + 1 && !f.exhausted && f.delay === RETRY_DELAYS_MS[i]));
  ok('the fourth consecutive failure (after 3 retries) gives up (no delay)',
    seen[3].exhausted && seen[3].delay === null && seen[3].count === 4);
  ok('given up stays given up for that key', t.exhausted('a') && t.count('a') === 4);
  ok('...and only for that key', !t.exhausted('b') && t.count('b') === 0);
  ok('a different key starts from zero (URL change)', t.fail('b').count === 1 && !t.exhausted('a'));
  t.reset();
  ok('reset() (manual retry) clears the streak', !t.exhausted('b') && t.count('b') === 0);

  const s = createFailureTracker();
  s.fail('a'); s.fail('a'); s.fail('a');
  s.ok('a');
  ok('a success ends the streak', s.count('a') === 0 && s.fail('a').count === 1);
  s.ok('other');
  ok('a success for ANOTHER key does not clear this one', s.count('a') === 1);

  // The loop the old code ran: ask, fail, ask again. With the tracker deciding,
  // an endlessly failing request is attempted exactly MAX_FAILURES times however
  // long we keep asking.
  const u = createFailureTracker();
  let attempts = 0;
  for (let i = 0; i < 10_000 && !u.exhausted('x'); i++) { attempts += 1; u.fail('x'); }
  ok('an always-failing request is attempted exactly 4 times (1 + 3 retries), not forever', attempts === 4, `attempts ${attempts}`);
}

console.log('── AbortError ─────────────────────────────────────────────');
{
  ok('a DOMException AbortError is an abort', isAbort(new DOMException('aborted', 'AbortError')));
  ok('an Error with name AbortError is an abort', isAbort(Object.assign(new Error('x'), { name: 'AbortError' })));
  ok('HTTP 500, a decode failure and null are not aborts',
    !isAbort(new Error('HTTP 500')) && !isAbort(new Error('decode failed')) && !isAbort(null) && !isAbort(undefined));
}

console.log('── 409 superseded (frame_seq_id) ──────────────────────────');
{
  ok('the backend\'s 409 is "superseded", not a failure', isSuperseded(httpFailure(409)));
  ok('...as is an abort, under the combined test', isSuperseding(new DOMException('aborted', 'AbortError')) && isSuperseding(httpFailure(409)));
  ok('500, 404, 503-busy and null are NOT superseded (they still spend the budget or wait)',
    [500, 404, 503].every((s) => !isSuperseded(httpFailure(s))) && !isSuperseded(httpFailure(503, { busy: true })) && !isSuperseded(null) && !isSuperseding(new Error('decode failed')));
}

console.log('── busy backend (pre-pass burst) ──────────────────────────');
{
  ok('a 503 hint in the body wins and is used as given',
    busyRetryMs(1500, 9) === 1500);
  ok('without a body hint the Retry-After header (seconds) is used', busyRetryMs(undefined, 2) === 2000);
  ok('no hint at all gets the default', busyRetryMs(null, null) === BUSY_DEFAULT_MS && busyRetryMs() === BUSY_DEFAULT_MS);
  ok('a hint is clamped: a tiny one cannot hammer, a huge one cannot park the requester',
    busyRetryMs(1) === BUSY_MIN_MS && busyRetryMs(10 * 60_000) === BUSY_MAX_MS && busyRetryMs(0, 600) === BUSY_MAX_MS);
  ok('garbage hints never produce NaN', [NaN, 'x', -5, Infinity].every((v) => Number.isFinite(busyRetryMs(v, v))));
  const busy = httpFailure(503, { busy: true, retryAfterMs: 1000 });
  ok('a busy refusal is recognised and carries its wait',
    isBusy(busy) && busy.status === 503 && busyDelay(busy) === 1000);
  const real = httpFailure(500);
  ok('a real HTTP error is NOT busy and has no wait', !isBusy(real) && busyDelay(real) === null && real.retryAfterMs === null);
  ok('plain errors, aborts and null are not busy',
    [new Error('decode failed'), new DOMException('aborted', 'AbortError'), null, undefined].every((e) => busyDelay(e) === null));
  // The point of it: a backend that stays busy for a minute must not burn the failure budget.
  const t = createFailureTracker();
  let waited = 0;
  for (let i = 0; i < 60; i++) {
    const wait = busyDelay(busy);
    if (wait === null) t.fail('frame');         // what the hook does for a real failure
    else waited += wait;                        // ...and for a busy one: wait, count nothing
  }
  ok('60 consecutive busy refusals leave the failure budget untouched', t.count('frame') === 0 && !t.exhausted('frame'));
  ok('...and the requester waited its told 1 s each time (about one request a second)', waited === 60_000);
  ok('BUSY bounds are ordered', BUSY_MIN_MS < BUSY_DEFAULT_MS && BUSY_DEFAULT_MS < BUSY_MAX_MS);
}

console.log('── wiring ─────────────────────────────────────────────────');
{
  const hook = src('useThrottledFrameRequest.js');
  ok('useThrottledFrameRequest counts failures through the tracker',
    hook.includes("from './retryPolicy'") && hook.includes('failuresRef.current.fail(src)'));
  ok('...never counts an abort or a superseded URL',
    hook.includes('isSuperseding(err) || ctrl.signal.aborted') && hook.includes('wantedRef.current !== src'));
  ok('...a sequenced request carries frame_seq_id on the wire only, strictly increasing, and failures stay keyed by src',
    hook.includes('const withSequence') && hook.includes('wire = withSequence(src, lastSeqRef.current)')
    && hook.includes('lastSeqRef.current + 1') && hook.includes('decodeFrame(wire,') && hook.includes('failuresRef.current.fail(src)'));
  ok('...and FaceSwap opts in, asking for the scrub proxy while dragging',
    (() => {
      const fs = readFileSync(join(here, '..', 'src', 'components', 'FaceSwap.jsx'), 'utf8').replace(/\r\n/g, '\n');
      return fs.includes('sequence: true') && fs.includes("'&width=960&scrubbing=1'");
    })());
  ok('...gives up when exhausted instead of draining again',
    hook.includes('failuresRef.current.exhausted(want)'));
  ok('...waits out the backoff in drain()', hook.includes('notBeforeRef.current - now'));
  ok('...resets on a URL change and clears the backoff timer for the old URL',
    /wanted !== wantedRef\.current[\s\S]{0,400}failuresRef\.current\.reset\(\)[\s\S]{0,200}clearTimeout\(timerRef\.current\)/.test(hook));
  ok('...a busy refusal waits as told and is not counted as a failure',
    /busyDelay\(err\)[\s\S]{0,200}notBeforeRef\.current = performance\.now\(\) \+ busyMs;[\s\S]{0,40}return;[\s\S]{0,200}failuresRef\.current\.fail\(src\)/.test(hook));
  ok('...returns error and retry', /\berror,\n\s+\/\*\*[^]*?\bretry,\n\s*\};/.test(hook));

  const play = src('usePlaybackBuffer.js');
  ok('usePlaybackBuffer routes chunk failures through the tracker',
    play.includes("from './retryPolicy'") && play.includes("chunkFailures.fail('chunk')"));
  ok('...pump() will not re-request a chunk before the backoff has elapsed',
    play.includes('performance.now() < chunkNotBefore'));
  ok('...the last failure stops playback and reports it',
    /f\.exhausted[\s\S]{0,300}setPlayError\([\s\S]{0,200}setIsPlaying\(false\)/.test(play));
  ok('...an aborted chunk is not a failure', play.includes('cancelled || ctrl.signal.aborted || isAbort(err)'));
  ok('...a busy refusal sets the same backoff gate and is not counted',
    /busyDelay\(err\)[\s\S]{0,160}chunkNotBefore = performance\.now\(\) \+ busyMs; return;[\s\S]{0,120}chunkFailures\.fail\('chunk'\)/.test(play));
  ok('...the 503 body is parsed so the hint reaches the policy', play.includes('httpFailure(r.status'));
  const worker = src('decoder.worker.js');
  const facade = src('frameDecoder.js');
  ok('the decoder worker forwards status, busy and the retry hint', /busy: res\.status === 503/.test(worker) && worker.includes('retryAfterMs: err?.retryAfterMs'));
  ok('the facade rebuilds a typed busy error (worker and main-thread paths)',
    facade.includes('httpFailure(status') && facade.includes('httpFailure(res.status'));
  ok('...a chunk that lands clears the streak', /chunkFailures\.reset\(\);\n\s+chunkNotBefore = 0;/.test(play));
}

console.log('');
if (failures) {
  console.log(`FAILED: ${failures} of ${checks}`);
  globalThis.__RENDER_CHECK_FAILURES__ = failures;
  process.exit(1);
}
console.log(`ALL GREEN: ${checks}/${checks} checks passed`);
