// What the backend is doing that a request cannot tell us: compiling a TensorRT engine, or
// running the fidelity canary that follows it (app/roop/pipeline_status.py, served at
// `GET /api/v1/pipeline/status` and `WS /ws/v1/pipeline/status`).
//
// WHY THE CLIENT NEEDS IT. A cold engine build takes 2-18 minutes with no frame counter. A
// request that reaches the backend in that window can be slow through no fault of its own, and
// api.js's 15 s deadline would report a healthy backend as dead. api.js asks this module at the
// moment a deadline fires: "busy" suspends the deadline until the stage ends, "not busy" lets the
// request fail as before. The same state drives the non-blocking banner (CompileBanner.jsx).
//
// This module is plain JS with no React and makes its own requests with fetch, NOT api.js:
// api.js imports it, and a status probe that obeyed the deadline it is deciding on would be
// circular.
//
// TRUST. "Busy" is a claim about the backend, so it expires. The socket only speaks when
// something changes (a 15-minute build is silent), so a busy status is trusted for STALE_MS after
// the last message that CONFIRMED it, and while anything is waiting on it a short poll keeps
// confirming. A backend that dies mid-build stops answering, the status goes stale, the
// suspended deadline resumes and the request fails with its usual timeout.

export const STATUS_PATH = '/api/v1/pipeline/status';
export const STATUS_WS_PATH = '/ws/v1/pipeline/status';

export const STAGE_COMPILING_TRT = 'STAGE_COMPILING_TRT';
export const STAGE_RUNNING_CANARY = 'STAGE_RUNNING_CANARY';
const BUSY_KINDS = new Set([STAGE_COMPILING_TRT, STAGE_RUNNING_CANARY]);

/** A busy claim older than this, unconfirmed, is not believed. */
export const STALE_MS = 6000;
/** While a deadline is suspended on a stage, re-confirm it this often. */
export const CONFIRM_POLL_MS = 2000;
const PROBE_TIMEOUT_MS = 3000;
const BACKOFF_MIN_MS = 500;
const BACKOFF_MAX_MS = 15000;

const IDLE = Object.freeze({ busy: false, active: [], connected: false });

let status = IDLE;
let confirmedAt = 0;
const listeners = new Set();
const idleWaiters = new Set();

let users = 0;
let socket = null;
let retryTimer = null;
let attempt = 0;
let pollTimer = null;

const now = () => Date.now();

/** The stages that suspend a deadline, as `{kind, model, precision, elapsed_s}` rows. */
export function activeStages(snapshot) {
  const rows = Array.isArray(snapshot?.active) ? snapshot.active : [];
  return rows
    .filter((a) => a && BUSY_KINDS.has(a.kind))
    .map((a) => ({
      kind: a.kind,
      model: a.model ? String(a.model) : 'model',
      precision: a.precision && a.precision !== 'n/a' ? String(a.precision).toUpperCase() : null,
      since: Number.isFinite(a.since) ? a.since * 1000 : null,
      elapsedS: Number.isFinite(a.elapsed_s) ? a.elapsed_s : 0,
    }));
}

function publish(next) {
  status = next;
  listeners.forEach((fn) => { try { fn(status); } catch { /* a bad listener must not stop the rest */ } });
  // Whenever the backend is NOT busy, nobody should still be waiting on it (a no-op with no waiters).
  if (!isBusy()) releaseIdleWaiters();
  ensureConfirmPoll();
}

function releaseIdleWaiters() {
  const waiting = [...idleWaiters];
  idleWaiters.clear();
  waiting.forEach((fn) => fn());
}

/** Apply a snapshot from the socket or from a probe. */
export function applySnapshot(snapshot) {
  const active = activeStages(snapshot);
  confirmedAt = now();
  publish({ busy: active.length > 0, active, connected: status.connected });
}

function setConnected(connected) {
  if (status.connected !== connected) publish({ ...status, connected });
}

/** True when the backend reported a stage and it has been confirmed recently enough to believe. */
export function isBusy() {
  return status.busy && now() - confirmedAt < STALE_MS;
}

/** The current state: a STABLE object, replaced only when something changed (useSyncExternalStore needs that). */
export const getStatus = () => status;

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** One GET of the snapshot, with its own short deadline. Resolves true/false for "busy"; a failure is "not busy". */
export async function probe() {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), PROBE_TIMEOUT_MS);
  try {
    const res = await fetch(`${window.location.origin}${STATUS_PATH}?events=0`, { signal: ctrl.signal, cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    applySnapshot(await res.json());
    return isBusy();
  } catch {
    // The status endpoint did not answer: that is no evidence of a build, and a build is the only
    // thing allowed to suspend a deadline. Drop the claim.
    if (status.busy) { confirmedAt = 0; publish({ ...status, busy: false, active: [] }); }
    return false;
  } finally {
    clearTimeout(timer);
  }
}

// ── Confirming a busy claim ─────────────────────────────────────────────────
// The socket is silent through a long build, so while ANY busy claim stands (from the socket or
// a probe) we GET the snapshot every CONFIRM_POLL_MS. That is what keeps the claim fresh, drives
// the banner's elapsed clock, and, when the backend dies mid-build, drops the claim so every
// suspended deadline resumes. It costs nothing when idle: it only runs during a stage.
function ensureConfirmPoll() {
  if (pollTimer || !status.busy) return;
  const tick = async () => {
    await probe();
    pollTimer = status.busy ? setTimeout(tick, CONFIRM_POLL_MS) : null;
  };
  pollTimer = setTimeout(tick, CONFIRM_POLL_MS);
}

/**
 * Called by api.js when a request's deadline fires. Resolves `true` once the backend is no longer
 * busy (the caller should give the request a fresh deadline) or `false` at once when it is not busy
 * to begin with (the caller times out as usual). `signal` abandons the wait.
 */
export async function suspendWhileBusy(signal) {
  if (!isBusy() && !(await probe())) return false;
  if (signal?.aborted) return false;
  await new Promise((resolve) => {
    const done = () => { idleWaiters.delete(done); signal?.removeEventListener('abort', done); resolve(); };
    idleWaiters.add(done);
    signal?.addEventListener('abort', done, { once: true });
    // The socket and the confirm poll are what flip us to idle; nothing else ends this wait.
    if (!isBusy()) done();
  });
  return !signal?.aborted;
}

// ── The socket (started once by the app shell) ──────────────────────────────
const socketUrl = () => {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}${STATUS_WS_PATH}`;
};

function connect() {
  if (users === 0 || socket || document.visibilityState === 'hidden') return;
  let ws;
  try { ws = new WebSocket(socketUrl()); } catch { scheduleRetry(); return; }
  socket = ws;
  ws.onopen = () => { attempt = 0; setConnected(true); };
  ws.onmessage = (ev) => {
    try { applySnapshot(JSON.parse(ev.data)); } catch { /* not a snapshot */ }
  };
  ws.onerror = () => { /* onclose owns the retry */ };
  ws.onclose = () => {
    if (socket === ws) socket = null;
    setConnected(false);
    scheduleRetry();
  };
}

function scheduleRetry() {
  if (users === 0 || retryTimer) return;
  const base = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS * 2 ** attempt++);
  retryTimer = setTimeout(() => { retryTimer = null; connect(); }, base * (0.5 + Math.random() * 0.5));
}

const onVisible = () => {
  if (document.visibilityState !== 'visible' || users === 0) return;
  if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
  attempt = 0;
  connect();
};

/** Keep the socket open until the returned function is called. Reference-counted: StrictMode-safe. */
export function startPipelineStatus() {
  users += 1;
  if (users === 1) {
    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('online', onVisible);
    connect();
  }
  let stopped = false;
  return () => {
    if (stopped) return;
    stopped = true;
    users -= 1;
    if (users > 0) return;
    document.removeEventListener('visibilitychange', onVisible);
    window.removeEventListener('online', onVisible);
    if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
    const ws = socket;
    socket = null;
    if (ws) { ws.onclose = null; try { ws.close(); } catch { /* already gone */ } }
    setConnected(false);
  };
}

/** Tests only. */
export function resetPipelineStatus() {
  status = IDLE;
  confirmedAt = 0;
  idleWaiters.clear();
  listeners.clear();
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
}
