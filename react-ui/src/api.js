// Thin client for the FastAPI backend (app/api.py) proxied via Vite.
import { longRunningKey } from './longRunning';
import { suspendWhileBusy } from './pipelineStatus';
import { noteMutation } from './mutationBus';

export const API = window.location.origin;

// ── Deadlines ───────────────────────────────────────────────────────────────
// Every request has a deadline unless the call says `timeout: 0`.
//
// Fetch has no timeout of its own, so a backend that accepts the socket and then
// stalls (mid-GPU-stall, or killed between accept and response) leaves the promise
// pending forever; a polled request that never settles silently stops the poll loop,
// and a button that never answers looks exactly like a button that does nothing. The
// old default was NO deadline, opted into per call, which is how most calls ended up
// without one. It is inverted: 15 s unless the call opts OUT with `timeout: 0`, and
// the endpoints that may legitimately run longer are listed, with reasons, in
// longRunning.js (a test enumerates the call sites and fails on one that is missing
// its opt-out, and on an opt-out that is not listed).
//
//   timeout: undefined  -> DEFAULT_TIMEOUT_MS
//   timeout: 0          -> no deadline (a long-running endpoint; see longRunning.js)
//   timeout: <ms>       -> that deadline
//
// An explicit `signal` still wins: aborting it aborts the request (rejecting with the
// signal's own reason, an AbortError unless the caller chose another), and our timer is
// cleared either way so no stray abort fires after the response lands.
export const DEFAULT_TIMEOUT_MS = 15000;

/** `timeout:` for a call whose path is computed (a wrapper like `act(path)`): 0 if the
 *  registry lists it as long-running, else the default. */
export const timeoutFor = (method, path) => (longRunningKey(method, path) ? 0 : undefined);

// ── Errors ──────────────────────────────────────────────────────────────────
// Every failure thrown from here is an ApiError that says WHICH request and HOW it
// failed, so a toast reading "no source faces" or "Failed to fetch" can be traced to
// an endpoint without opening devtools:
//
//   no source faces (HTTP 400 POST /api/swap)
//   Network error: Failed to fetch (POST /api/preview)
//   Request timed out after 15 s (GET /api/state)
//
// `status` is the HTTP status, or 0 when there was none (network, timeout, unreadable
// reply); `kind` is 'http' | 'network' | 'timeout' | 'parse'; `body` is the parsed JSON
// error body when there was one. A caller aborting its own request is NOT an ApiError:
// that stays the AbortError the signal carried, because it is not a failure.
//
// `endpoint` is `METHOD /path` (the registry key shape of longRunning.js) and `elapsedMs` is how
// long the request lived before it failed, wall clock, INCLUDING any time its deadline was
// suspended for a TensorRT build (`suspendedMs` says how much of it that was).
export class ApiError extends Error {
  constructor({ detail, status = 0, method, path, kind = 'http', body = null, cause, elapsedMs = null, suspendedMs = 0 }) {
    const where = status ? `HTTP ${status} ${method} ${path}` : `${method} ${path}`;
    super(`${detail} (${where})`, cause ? { cause } : undefined);
    this.name = 'ApiError';
    this.detail = detail;
    this.status = status;
    this.method = method;
    this.path = path;
    this.endpoint = `${method} ${path}`;
    this.kind = kind;
    this.body = body;
    this.elapsedMs = elapsedMs;
    this.suspendedMs = suspendedMs;
  }
}

const bare = (path) => String(path).split('?')[0];

// A response that is not 2xx. The backend's error bodies are `{message}` (and
// sometimes `{error}`); anything else falls back to the status text.
async function httpError(res, method, path, track) {
  let body = null;
  let detail = res.statusText || `HTTP ${res.status}`;
  try {
    body = await res.json();
    detail = body.message || body.error || detail;
  } catch { /* not JSON: the status line is all there is to say */ }
  return new ApiError({ detail: String(detail), status: res.status, method, path: bare(path), body, ...track() });
}

async function handle(res, method, path, track) {
  if (!res.ok) throw await httpError(res, method, path, track);
  const ct = res.headers.get('content-type') || '';
  if (!ct.includes('application/json')) return res.text();
  try {
    return await res.json();
  } catch (cause) {
    // A 2xx that is not parseable is a server bug, and saying so beats a bare
    // "Unexpected token" surfacing three components away.
    throw new ApiError({ detail: 'Malformed JSON in the server response', status: res.status, method, path: bare(path), kind: 'parse', cause, ...track() });
  }
}

const isAbort = (e) => e && e.name === 'AbortError';

// The one place a request is made. `init` is fetch's init minus `signal`.
//
// THE DEADLINE IS SUSPENDED, NOT EXTENDED, WHILE THE BACKEND COMPILES. When the timer fires, the
// request asks pipelineStatus whether the backend announced STAGE_COMPILING_TRT or
// STAGE_RUNNING_CANARY (a cold engine build is 2-18 minutes with no frame counter):
//   * busy      -> the request is left in flight, a non-blocking banner already says why, and
//                  when the stage ends the request gets a FRESH full deadline;
//   * not busy  -> it times out exactly as before (ApiError kind 'timeout').
// A backend that dies mid-build stops answering the status probe, so the claim expires and the
// request still fails. Only finite deadlines are involved: `timeout: 0` never had one.
async function request(method, path, init, opts = {}) {
  const ms = opts.timeout === undefined ? DEFAULT_TIMEOUT_MS : opts.timeout;
  const caller = opts.signal;
  const ctrl = ms > 0 ? new AbortController() : null;
  const waiting = new AbortController();      // ends a suspended wait when the request settles
  const started = Date.now();
  let timer = null;
  let timedOut = false;
  let settled = false;
  let suspendedMs = 0;
  let onCallerAbort = null;
  const track = () => ({ elapsedMs: Date.now() - started, suspendedMs });

  const onDeadline = async () => {
    timer = null;
    const since = Date.now();
    const resumed = await suspendWhileBusy(waiting.signal);
    if (settled || caller?.aborted) return;
    if (resumed) {
      suspendedMs += Date.now() - since;
      timer = setTimeout(onDeadline, ms);
      return;
    }
    timedOut = true;
    ctrl.abort(new Error('Request timed out'));
  };

  if (ctrl) {
    timer = setTimeout(onDeadline, ms);
    onCallerAbort = () => ctrl.abort(caller.reason);
    if (caller?.aborted) onCallerAbort();
    else caller?.addEventListener('abort', onCallerAbort, { once: true });
  }

  try {
    const res = await fetch(`${API}${path}`, { ...init, signal: ctrl ? ctrl.signal : caller });
    const out = await handle(res, method, path, track);
    noteMutation(method, path);                      // after the parse: a failure here is not this request's
    return out;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (timedOut && !caller?.aborted) {
      const shown = suspendedMs ? Math.round((Date.now() - started) / 1000) : ms / 1000;
      throw new ApiError({ detail: `Request timed out after ${shown} s`, method, path: bare(path), kind: 'timeout', cause: err, ...track() });
    }
    if (isAbort(err) || caller?.aborted) throw err;           // the caller's own abort: not a failure
    if (err instanceof TypeError) {                           // fetch's "Failed to fetch" / DNS / refused / reset
      throw new ApiError({ detail: `Network error: ${err.message}`, method, path: bare(path), kind: 'network', cause: err, ...track() });
    }
    throw err;
  } finally {
    settled = true;
    waiting.abort();
    clearTimeout(timer);
    caller?.removeEventListener('abort', onCallerAbort);
  }
}

export const getJSON = (path, opts = {}) => request('GET', path, {}, opts);

// `opts.tap(body)` is called with the request body just before it is sent: a read-only look for a caller that needs to
// remember what it asked (Face Swap hands its /api/preview body to the floating preview so it can ask again).
export const postJSON = (path, body, opts = {}) => {
  if (opts.tap) opts.tap(body);
  return request('POST', path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
    // keepalive lets the request outlive a page teardown (e.g. Pinokio's
    // Run<->Dev webview reload) so a last-moment flush still reaches the server.
    keepalive: opts.keepalive,
  }, opts);
};

// Uploads go through XHR, not fetch.
//
// fetch() cannot report request-body progress — there is no event for it, and
// the streaming-request API that would allow one is not available here. So a
// 4 GB video was posted with `fetch` and the UI had nothing to show but an
// indeterminate spinner: no bytes, no percentage, no rate, no way to tell a
// slow upload from a wedged one, and no way to cancel a file dropped by
// mistake short of restarting the app. XHR still exposes `upload.onprogress`,
// which is the only reason it is preferred over fetch anywhere in 2026.
//
// onProgress receives { loaded, total, phase }:
//   phase 'upload'  — bytes still going out; `total` is 0 if not computable
//   phase 'analyse' — every byte is sent and the server is now decoding and
//                     running detection, which for a long video is the LONGER
//                     half. Without this the bar would sit at 100% for minutes
//                     and read as a hang.
//
// The deadline (xhr.timeout) is the WHOLE request, upload and analysis together, so every
// upload call passes `timeout: 0` (they are all in longRunning.js); leaving it off gives
// a 4 GB upload 15 seconds.
const xhrUpload = (path, fd, { onProgress, signal, timeout } = {}) => new Promise((resolve, reject) => {
  const method = 'POST';
  const ms = timeout === undefined ? DEFAULT_TIMEOUT_MS : timeout;
  const xhr = new XMLHttpRequest();
  const started = Date.now();
  const elapsed = () => ({ elapsedMs: Date.now() - started });
  xhr.open(method, `${API}${path}`);
  if (ms > 0) xhr.timeout = ms;

  if (onProgress) {
    xhr.upload.onprogress = (e) => onProgress({
      loaded: e.loaded,
      total: e.lengthComputable ? e.total : 0,
      phase: 'upload',
    });
    xhr.upload.onload = () => onProgress({ loaded: 0, total: 0, phase: 'analyse' });
  }

  const onAbort = () => xhr.abort();
  if (signal) {
    // Reject explicitly rather than calling xhr.abort() here: abort() on a
    // request that has been open()ed but not send()t fires no abort event, so
    // the handler below never runs and the promise would never settle — a
    // caller awaiting it would hang forever with no error to show.
    if (signal.aborted) {
      reject(new DOMException('Upload cancelled', 'AbortError'));
      return;
    }
    signal.addEventListener('abort', onAbort, { once: true });
  }
  const cleanup = () => signal?.removeEventListener('abort', onAbort);

  xhr.onload = () => {
    cleanup();
    if (xhr.status >= 200 && xhr.status < 300) {
      const ct = xhr.getResponseHeader('content-type') || '';
      if (!ct.includes('application/json')) { noteMutation(method, path); resolve(xhr.responseText); return; }
      // Mirrors handle(): a 2xx that is not parseable is a server bug.
      try { const body = JSON.parse(xhr.responseText); noteMutation(method, path); resolve(body); }
      catch (cause) {
        reject(new ApiError({ detail: 'Malformed JSON in the server response', status: xhr.status, method, path: bare(path), kind: 'parse', cause, ...elapsed() }));
      }
      return;
    }
    let body = null;
    let detail = xhr.statusText || `HTTP ${xhr.status}`;
    try {
      body = JSON.parse(xhr.responseText);
      detail = body.message || body.error || detail;
    } catch { /* not JSON: the status line is all there is to say */ }
    reject(new ApiError({ detail: String(detail), status: xhr.status, method, path: bare(path), body, ...elapsed() }));
  };
  xhr.onerror = () => {
    cleanup();
    reject(new ApiError({ detail: 'Network error during upload', method, path: bare(path), kind: 'network', ...elapsed() }));
  };
  xhr.ontimeout = () => {
    cleanup();
    reject(new ApiError({ detail: `Upload timed out after ${ms / 1000} s`, method, path: bare(path), kind: 'timeout', ...elapsed() }));
  };
  xhr.onabort = () => { cleanup(); reject(new DOMException('Upload cancelled', 'AbortError')); };

  xhr.send(fd);
});

export const postFiles = (path, files, fields, opts) => {
  const fd = new FormData();
  const list = files instanceof FileList ? Array.from(files) : [].concat(files);
  list.forEach((f) => fd.append('files', f));
  if (fields) Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
  return xhrUpload(path, fd, opts);
};

export const postFile = (path, file, fields, opts) => {
  const fd = new FormData();
  fd.append('file', file);
  if (fields) Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
  return xhrUpload(path, fd, opts);
};

export const fileUrl = (p) => {
  if (!p) return '';
  if (p.startsWith('http://') || p.startsWith('https://')) return p;
  if (p.startsWith('/')) return `${API}${p}`;
  return `${API}/api/file?path=${encodeURIComponent(p)}`;
};
