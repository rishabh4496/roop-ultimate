/**
 * The API client's contract, exercised for real (fetch is stubbed; nothing else is).
 *
 *   * every request has a 15 s deadline unless the call passes `timeout: 0`;
 *   * thrown errors say WHICH request and HOW it failed (status + method + path);
 *   * a caller's own abort is not a failure: it stays the AbortError it was;
 *   * failureLog reports each distinct failure once, never an abort, and rate-limits toasts;
 *   * a component's unmount aborts its reads, but not its mutations.
 *
 * What these cannot show is that every CALL SITE makes the right choice about `timeout: 0`;
 * that is scripts/api-callsites.mjs, which enumerates the sites.
 */
globalThis.window = globalThis;
globalThis.location = { origin: 'http://127.0.0.1:1', protocol: 'http:', host: '127.0.0.1:1' };

const { getJSON, postJSON, postFile, ApiError, DEFAULT_TIMEOUT_MS, timeoutFor } = await import('../src/api.js');
const { LONG_RUNNING, longRunningKey } = await import('../src/longRunning.js');
const failureLog = await import('../src/failureLog.js');
const { createUnmountScope, mergeSignals, bindApi } = await import('../src/useApi.js');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const rejects = async (p) => { try { await p; return null; } catch (e) { return e; } };

// ── fetch stub ──────────────────────────────────────────────────────────────
const json = (status, body, headers = { 'content-type': 'application/json' }) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: { 400: 'Bad Request', 404: 'Not Found', 409: 'Conflict', 500: 'Internal Server Error' }[status] || '',
  headers: { get: (k) => headers[k.toLowerCase()] ?? null },
  json: async () => { if (typeof body === 'string') throw new SyntaxError('Unexpected token'); return body; },
  text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
});
let _lastInit = null;
// The pipeline status probe (`/api/v1/pipeline/status`) is answered here, not by `impl`: a test that
// hangs every request must not also hang the probe api.js makes when a deadline fires. Default: idle.
let statusReply = () => ({ seq: 0, active: [] });
let statusCalls = 0;
const stubFetch = (impl) => {
  globalThis.fetch = (url, init) => {
    if (String(url).includes('/api/v1/pipeline/status')) {
      statusCalls += 1;
      return Promise.resolve(json(200, statusReply()));
    }
    _lastInit = init;
    return impl(url, init);
  };
};
// A fetch that never answers, but honours its abort signal like the real one: it rejects
// when the signal aborts, and at once if the signal is ALREADY aborted.
const hang = (url, init) => new Promise((_res, rej) => {
  if (init.signal?.aborted) { rej(init.signal.reason); return; }
  init.signal?.addEventListener('abort', () => rej(init.signal.reason), { once: true });
});

// Record every timer the client sets, without changing what it does.
const realSetTimeout = globalThis.setTimeout;
let timers = [];
globalThis.setTimeout = (fn, ms, ...rest) => { timers.push(ms); return realSetTimeout(fn, ms, ...rest); };
const timersSet = async (fn) => { timers = []; await fn(); return [...timers]; };

console.log('── deadlines ───────────────────────────────────────────────');
{
  ok('the default is 15 s', DEFAULT_TIMEOUT_MS === 15000);
  stubFetch(async () => json(200, { ok: true }));
  ok('a call with no `timeout` arms a 15 s timer', (await timersSet(() => getJSON('/api/state'))).includes(15000));
  ok('`timeout: 0` arms NO timer', !(await timersSet(() => getJSON('/api/x', { timeout: 0 }))).includes(0)
    && (await timersSet(() => postJSON('/api/swap', {}, { timeout: 0 }))).length === 0);
  ok('an explicit number is used as given', (await timersSet(() => getJSON('/api/x', { timeout: 8000 }))).includes(8000));

  stubFetch(hang);
  const e = await rejects(getJSON('/api/state?x=1', { timeout: 30 }));
  ok('a stalled request is cut off at its deadline', e instanceof ApiError && e.kind === 'timeout');
  ok('the timeout error names the request and the wait', /timed out after 0\.03 s/.test(e?.message) && /GET \/api\/state\)/.test(e?.message), e?.message);
  ok('the query string is not part of the path', e?.path === '/api/state');

  stubFetch((u, init) => new Promise((res, rej) => {
    realSetTimeout(() => res(json(200, { done: true })), 60);
    init.signal?.addEventListener('abort', () => rej(init.signal.reason), { once: true });
  }));
  const slow = await rejects(getJSON('/api/x', { timeout: 20 }));
  ok('a slow answer past the deadline is a timeout', slow?.kind === 'timeout');
  const patient = await getJSON('/api/x', { timeout: 0 });
  ok('`timeout: 0` waits as long as it takes', patient?.done === true);
}

console.log('── errors carry status, method and path ───────────────────');
{
  stubFetch(async () => json(409, { message: 'already processing' }));
  const e = await rejects(postJSON('/api/swap', {}, { timeout: 0 }));
  ok('an HTTP error is an ApiError', e instanceof ApiError && e.name === 'ApiError' && e instanceof Error);
  ok('it has status, method, path, kind', e?.status === 409 && e?.method === 'POST' && e?.path === '/api/swap' && e?.kind === 'http');
  ok('the message leads with the server\'s words and ends with where', e?.message === 'already processing (HTTP 409 POST /api/swap)', e?.message);
  ok('the parsed error body is kept', e?.body?.message === 'already processing' && e?.detail === 'already processing');

  stubFetch(async () => json(500, 'not json at all', { 'content-type': 'text/plain' }));
  const e2 = await rejects(getJSON('/api/state'));
  ok('a non-JSON error body falls back to the status text', e2?.message === 'Internal Server Error (HTTP 500 GET /api/state)', e2?.message);

  stubFetch(async () => json(404, { error: 'no_source' }));
  const e3 = await rejects(getJSON('/api/x'));
  ok('`{error}` is used when there is no `{message}`', e3?.detail === 'no_source');

  stubFetch(async () => { throw new TypeError('Failed to fetch'); });
  const net = await rejects(postJSON('/api/preview', {}, { timeout: 0 }));
  ok('a network failure is an ApiError with no status', net instanceof ApiError && net.kind === 'network' && net.status === 0);
  ok('and still names the request', net?.message === 'Network error: Failed to fetch (POST /api/preview)', net?.message);

  stubFetch(async () => json(200, 'oops {', { 'content-type': 'application/json' }));
  const bad = await rejects(getJSON('/api/state'));
  ok('a 2xx that is not JSON is a parse error naming the request', bad?.kind === 'parse' && /GET \/api\/state/.test(bad.message));

  stubFetch(async () => json(200, 'plain text', { 'content-type': 'text/plain' }));
  ok('a text response is returned as text', (await getJSON('/api/x')) === 'plain text');
}

console.log('── a caller\'s abort is not a failure ──────────────────────');
{
  stubFetch(hang);
  const ctrl = new AbortController();
  const p = rejects(getJSON('/api/x', { signal: ctrl.signal, timeout: 5000 }));
  ctrl.abort();
  const e = await p;
  ok('it stays an AbortError (not an ApiError)', e?.name === 'AbortError' && !(e instanceof ApiError), `${e?.name}`);

  const pre = new AbortController(); pre.abort();
  const e2 = await rejects(getJSON('/api/x', { signal: pre.signal }));
  ok('an already-aborted signal rejects at once', e2?.name === 'AbortError');

  const ctrl3 = new AbortController();
  const p3 = rejects(getJSON('/api/x', { signal: ctrl3.signal, timeout: 0 }));
  ctrl3.abort();
  ok('and so does an abort with no deadline', (await p3)?.name === 'AbortError');
  // the deadline's own timer must not outlive the request
  stubFetch(async () => json(200, { a: 1 }));
  ok('no stray timer is left behind a finished request', (await getJSON('/api/x', { timeout: 40 })).a === 1);
}

console.log('── uploads (XHR) ───────────────────────────────────────────');
{
  const made = [];
  globalThis.FormData = class { constructor() { this.parts = []; } append(k, v) { this.parts.push([k, v]); } };
  globalThis.FileList = class {};
  globalThis.XMLHttpRequest = class {
    constructor() { this.upload = {}; made.push(this); }
    open(m, u) { this.method = m; this.url = u; }
    getResponseHeader() { return 'application/json'; }
    send() {}
    abort() {}
  };
  const p = postFile('/api/extras/apply', { name: 'a' });
  const x = made.at(-1);
  ok('an upload with no `timeout` gets the 15 s deadline', x.timeout === 15000);
  x.ontimeout();
  const e = await rejects(p);
  ok('and a timeout is an ApiError naming the upload', e instanceof ApiError && e.kind === 'timeout' && /POST \/api\/extras\/apply/.test(e.message), e?.message);

  const p2 = postFile('/api/extras/apply', { name: 'a' }, undefined, { timeout: 0 });
  ok('`timeout: 0` leaves xhr.timeout unset (no limit)', made.at(-1).timeout === undefined);
  made.at(-1).status = 413; made.at(-1).statusText = 'Payload Too Large'; made.at(-1).responseText = '{"message":"too big"}';
  made.at(-1).onload();
  const e2 = await rejects(p2);
  ok('an upload HTTP error carries status and path too', e2?.status === 413 && e2?.message === 'too big (HTTP 413 POST /api/extras/apply)', e2?.message);
}

console.log('── the registry ────────────────────────────────────────────');
{
  ok('timeoutFor: a long-running endpoint opts out', timeoutFor('POST', '/api/swap') === 0);
  ok('timeoutFor: an ordinary one takes the default', timeoutFor('GET', '/api/state') === undefined && timeoutFor('POST', '/api/settings') === undefined);
  ok('timeoutFor: the verb matters', timeoutFor('GET', '/api/target/auto_angles') === undefined && timeoutFor('POST', '/api/target/auto_angles') === 0);
  ok('timeoutFor: a query string and a trailing slash are ignored', timeoutFor('POST', '/api/swap?x=1') === 0 && timeoutFor('POST', '/api/swap/') === 0);
  ok('timeoutFor: `{}` matches one segment (project verbs)',
    timeoutFor('POST', '/api/projects/abc123/load') === 0 && timeoutFor('POST', '/api/projects/abc123/resume') === 0
    && timeoutFor('POST', '/api/projects/abc123/validate') === undefined && timeoutFor('POST', '/api/projects/a/b/load') === undefined);
  ok('longRunningKey names the entry', longRunningKey('POST', '/api/projects/9/load') === 'POST /api/projects/{}/load');
  ok('every entry has a reason', Object.entries(LONG_RUNNING).every(([, why]) => typeof why === 'string' && why.length > 8));
  ok('every key is `METHOD /api/...`', Object.keys(LONG_RUNNING).every((k) => /^(GET|POST) \/api\/\S+$/.test(k)));
}

console.log('── failureLog: each distinct failure once ──────────────────');
{
  const toasts = [];
  let clock = 1_000_000;
  const warn = console.warn; let warned = [];
  console.warn = (...a) => { warned.push(a[0]); };
  failureLog.__resetForTests({ now: () => clock });
  failureLog.setFailureToast((m, t) => toasts.push([m, t]));

  const err = (status, path, detail = 'boom') => new ApiError({ detail, status, method: 'POST', path });
  ok('the first failure is reported (console + toast)', failureLog.reportFailure('Saving settings', err(500, '/api/settings')) === true
    && warned.length === 1 && toasts.length === 1);
  ok('the toast says what failed and why, as a warning', /^Saving settings failed: boom \(HTTP 500 POST \/api\/settings\)$/.test(toasts[0][0]) && toasts[0][1] === 'warning', toasts[0]?.[0]);
  ok('the same failure again is silent', failureLog.reportFailure('Saving settings', err(500, '/api/settings')) === false && warned.length === 1 && toasts.length === 1);
  ok('...but counted', Object.values(failureLog.failureCounts())[0] === 2);
  ok('a different status is a different failure', failureLog.reportFailure('Saving settings', err(409, '/api/settings')) === true);
  ok('so is a different context', failureLog.reportFailure('Saving a preset', err(409, '/api/settings')) === true && warned.length === 3);

  warned = []; toasts.length = 0;
  failureLog.__resetForTests({ now: () => clock });
  failureLog.setFailureToast((m, t) => toasts.push([m, t]));
  for (let i = 0; i < 6; i++) failureLog.reportFailure(`Thing ${i}`, err(500, `/api/t${i}`));
  ok('six distinct failures: six console lines', warned.length === 6);
  ok('...but at most 2 toasts inside the window', toasts.length === 2, `${toasts.length}`);
  ok('a suppressed toast says so in the console', warned.slice(2).every((w) => /toast suppressed/.test(w)));
  clock += 31_000;
  failureLog.reportFailure('Thing late', err(500, '/api/late'));
  ok('the budget refills after the window', toasts.length === 3);

  warned = []; toasts.length = 0;
  failureLog.reportFailure('Polling the queue', err(0, '/api/queue'), { toast: false });
  ok('`{ toast: false }` logs but never toasts', warned.length === 1 && toasts.length === 0);

  warned = [];
  const a = new AbortController(); a.abort();
  ok('an AbortError is never reported', failureLog.reportFailure('X', new DOMException('x', 'AbortError')) === false && warned.length === 0);
  const timeout = new ApiError({ detail: 'Request timed out after 15 s', method: 'GET', path: '/api/state', kind: 'timeout' });
  ok('a timeout IS reported (the user\'s request failed)', failureLog.reportFailure('Loading state', timeout) === true);

  ok('non-Error values are handled', failureLog.reportFailure('Odd', 'a string') === true && failureLog.reportFailure('Odd2', undefined) === true);
  ok('logFailure(...) returns a .catch handler that resolves', await Promise.reject(new Error('z')).catch(failureLog.logFailure('Ctx')) === undefined);
  console.warn = warn;
  failureLog.__resetForTests();
}

console.log('── useApi: unmount aborts reads, not mutations ─────────────');
{
  const scope = createUnmountScope();
  ok('a live scope is not aborted', scope.signal.aborted === false);
  const first = scope.signal;
  scope.unmount();
  ok('unmounting aborts it', first.aborted === true && first.reason?.name === 'AbortError');
  ok('the signal stays aborted: a handler that fires after unmount is cancelled', scope.signal.aborted === true);
  scope.mount();
  ok('StrictMode: mounting again gives a FRESH signal', scope.signal.aborted === false && scope.signal !== first);

  const a = new AbortController(); const b = new AbortController();
  const merged = mergeSignals(a.signal, b.signal);
  ok('mergeSignals: neither aborted -> not aborted', merged.aborted === false);
  b.abort();
  ok('mergeSignals: either one aborts it', merged.aborted === true);
  ok('mergeSignals: a single signal passes through', mergeSignals(a.signal, undefined) === a.signal && mergeSignals(undefined, b.signal) === b.signal);

  stubFetch(hang);
  const s = createUnmountScope();
  const api = bindApi(s);
  const read = rejects(api.getJSON('/api/state', { timeout: 0 }));
  const write = rejects(api.postJSON('/api/settings', { a: 1 }, { timeout: 0 }));
  const readPost = rejects(api.postJSON('/api/runtime_estimate', {}, { abortOnUnmount: true, timeout: 0 }));
  s.unmount();
  const r = await read;
  const rp = await readPost;
  // The mutation was handed no scope signal, so unmounting leaves it in flight: it neither
  // resolves nor rejects within a generous window.
  const w = await Promise.race([write, new Promise((res) => realSetTimeout(() => res('still in flight'), 80))]);
  ok('a GET is aborted on unmount, and rejects with an AbortError', r?.name === 'AbortError', `${r}`);
  ok('a POST with abortOnUnmount is aborted too', rp?.name === 'AbortError', `${rp}`);
  ok('a plain POST (a mutation) is NOT aborted', w === 'still in flight', `${w?.name || w}`);
  ok('unmount never turns into a reported failure', failureLog.reportFailure('x', r) === false);
}

console.log('── errors say how long the request lived ───────────────────');
{
  stubFetch(hang);
  const e = await rejects(getJSON('/api/state', { timeout: 40 }));
  ok('a timeout carries elapsedMs >= its deadline', e instanceof ApiError && e.elapsedMs >= 35, `${e?.elapsedMs}`);
  ok('...and its endpoint as METHOD /path', e?.endpoint === 'GET /api/state', e?.endpoint);
  ok('...and no suspension, as nothing was compiling', e?.suspendedMs === 0 && e?.status === 0);
  stubFetch(async () => json(404, { message: 'nope' }));
  const h = await rejects(getJSON('/api/missing'));
  ok('an HTTP error carries status, endpoint and elapsedMs', h?.status === 404 && h?.endpoint === 'GET /api/missing' && typeof h?.elapsedMs === 'number');
  stubFetch(async () => { throw new TypeError('Failed to fetch'); });
  const n = await rejects(getJSON('/api/state'));
  ok('a network error carries elapsedMs', n?.kind === 'network' && typeof n?.elapsedMs === 'number');
}

console.log('── the deadline is suspended while the backend compiles ───');
{
  const ps = await import('../src/pipelineStatus.js');
  const compiling = (extra = {}) => ({ seq: 1, active: [{ kind: 'STAGE_COMPILING_TRT', model: 'gpen_2048', precision: 'FP16', since: Date.now() / 1000, elapsed_s: 1, ...extra }] });
  ps.resetPipelineStatus();

  // activeStages(): only the two suspending kinds count, precision is normalised.
  const rows = ps.activeStages({ active: [
    { kind: 'STAGE_COMPILING_TRT', model: 'gpen', precision: 'fp32', since: 10, elapsed_s: 3 },
    { kind: 'STAGE_RUNNING_CANARY', model: 'xseg', precision: 'n/a' },
    { kind: 'VRAM_GOVERNOR_THROTTLE' }, { kind: 'SEGMENT_PROGRESS' },
  ] });
  ok('activeStages keeps only COMPILING_TRT and RUNNING_CANARY', rows.length === 2 && rows[0].model === 'gpen' && rows[1].model === 'xseg');
  ok('...precision is upper-cased; "n/a" is dropped', rows[0].precision === 'FP32' && rows[1].precision === null);

  // 1. Busy at the deadline: the request is NOT cut off, and finishes when the stage ends.
  statusReply = () => compiling();
  let release;
  stubFetch((u, init) => new Promise((res, rej) => {
    release = () => res(json(200, { done: true }));
    init.signal?.addEventListener('abort', () => rej(init.signal.reason), { once: true });
  }));
  statusCalls = 0;
  const waiting = getJSON('/api/state', { timeout: 40 });
  let settled = false;
  waiting.then(() => { settled = true; }, () => { settled = true; });
  await new Promise((r) => realSetTimeout(r, 200));
  ok('a request past its 40 ms deadline is still alive while STAGE_COMPILING_TRT is active', settled === false && ps.isBusy() === true);
  ok('...having asked the status channel exactly when the deadline fired', statusCalls >= 1);
  statusReply = () => ({ seq: 2, active: [] });
  await ps.probe();                         // the confirm poll would do this; do it now rather than wait 2 s
  ok('the stage ending releases the wait', ps.isBusy() === false);
  await new Promise((r) => realSetTimeout(r, 10));
  release();
  ok('...and the request then completes with its answer', (await waiting)?.done === true);

  // 2. After the stage ends the request gets a FRESH deadline, and still times out if nothing answers.
  ps.resetPipelineStatus();
  statusReply = () => compiling();
  stubFetch(hang);
  const t0 = Date.now();
  const doomed = rejects(getJSON('/api/state', { timeout: 60 }));
  await new Promise((r) => realSetTimeout(r, 150));              // suspended by now
  statusReply = () => ({ seq: 3, active: [] });
  await ps.probe();
  const err = await doomed;
  ok('after a suspension the request gets a fresh full deadline, then times out as usual', err?.kind === 'timeout' && Date.now() - t0 >= 150 + 55, `${err?.kind} after ${Date.now() - t0} ms`);
  ok('...ApiError reports elapsedMs INCLUDING the suspended time, and how much of it that was', err?.elapsedMs >= 200 && err?.suspendedMs >= 80, `elapsed ${err?.elapsedMs} suspended ${err?.suspendedMs}`);

  // 3. A status endpoint that stops answering must not hold a request forever.
  ps.resetPipelineStatus();
  statusReply = () => compiling();
  stubFetch(hang);
  const stuck = rejects(getJSON('/api/state', { timeout: 30 }));
  await new Promise((r) => realSetTimeout(r, 120));
  ok('(precondition) suspended', ps.isBusy() === true);
  const realFetch = globalThis.fetch;
  globalThis.fetch = (url, init) => (String(url).includes('/api/v1/pipeline/status') ? Promise.reject(new TypeError('Failed to fetch')) : realFetch(url, init));
  await ps.probe();
  const dead = await stuck;
  ok('when the status probe fails the claim is dropped and the request times out', dead?.kind === 'timeout', `${dead}`);

  // 4. Not busy at the deadline -> the plain old timeout, immediately.
  ps.resetPipelineStatus();
  statusReply = () => ({ seq: 0, active: [] });
  stubFetch(hang);
  const plain = await rejects(getJSON('/api/state', { timeout: 30 }));
  ok('with nothing compiling a hung request times out at its deadline as before', plain?.kind === 'timeout' && plain.suspendedMs === 0 && /timed out after 0\.03 s/.test(plain.message), plain?.message);

  // 5. A caller abort ends a suspended wait with the AbortError, not a timeout.
  ps.resetPipelineStatus();
  statusReply = () => compiling();
  stubFetch(hang);
  const ac = new AbortController();
  const aborted = rejects(getJSON('/api/state', { timeout: 30, signal: ac.signal }));
  await new Promise((r) => realSetTimeout(r, 120));
  ac.abort();
  const ae = await aborted;
  ok('aborting a suspended request rejects with AbortError', ae?.name === 'AbortError' && !(ae instanceof ApiError), `${ae}`);

  // 6. timeout:0 never asks (it has no deadline to suspend).
  ps.resetPipelineStatus();
  statusReply = () => compiling();
  stubFetch(async () => json(200, { ok: 1 }));
  statusCalls = 0;
  await getJSON('/api/x', { timeout: 0 });
  ok('`timeout: 0` makes no status request at all', statusCalls === 0);

  // 7. Staleness: a busy claim nobody re-confirmed is not believed.
  ps.resetPipelineStatus();
  ps.applySnapshot(compiling());
  ok('a fresh busy claim is believed', ps.isBusy() === true);
  const realNow = Date.now;
  Date.now = () => realNow() + ps.STALE_MS + 1;
  const believed = ps.isBusy();
  Date.now = realNow;
  ok('...and expires after STALE_MS without confirmation', believed === false);
  ps.resetPipelineStatus();
}

console.log('── useApi: view scope vs job scope ─────────────────────────');
{
  const { abortJobRequests } = await import('../src/useApi.js');
  stubFetch(hang);
  const s = createUnmountScope();
  const api = bindApi(s);
  const viewRead = rejects(api.getJSON('/api/state', { timeout: 0 }));
  const jobRead = Promise.race([rejects(api.getJSON('/api/queue', { timeout: 0, scope: 'job' })), new Promise((res) => realSetTimeout(() => res('alive'), 100))]);
  s.unmount();                                                          // the tab goes away (hash-route change)
  const v = await viewRead;
  ok('view scope: the tab\'s own read is aborted on unmount', v?.name === 'AbortError', `${v}`);
  ok('job scope: a queue read SURVIVES the tab unmounting', (await jobRead) === 'alive');

  const s2 = createUnmountScope();
  const api2 = bindApi(s2);
  const pending = rejects(api2.getJSON('/api/queue', { timeout: 0, scope: 'job' }));
  s2.unmount();
  abortJobRequests('Queue cleared');
  const j = await pending;
  ok('abortJobRequests() cancels in-flight job reads, with an AbortError', j?.name === 'AbortError' && /Queue cleared/.test(j.message || ''), `${j}`);
  stubFetch(async () => json(200, { ok: 1 }));
  ok('...and job scope works again afterwards (fresh controller)', (await bindApi(createUnmountScope()).getJSON('/api/queue', { scope: 'job' }))?.ok === 1);
}

globalThis.setTimeout = realSetTimeout;
globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\napi-client: all checks passed' : `\napi-client: ${failures} check(s) FAILED`);
