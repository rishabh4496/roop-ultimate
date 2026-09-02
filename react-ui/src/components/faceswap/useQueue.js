import { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON } from '../../api';

// The batch queue, as owned by the backend (see app/routes_queue.py).
//
// This used to be a `useState([])` plus two effects that walked it, POSTing
// /api/swap job by job — which made the queue exactly as durable as the tab.
// It now holds no queue state of its own: every mutation is a request whose
// RESPONSE is the new snapshot, so what is on screen is what the server will
// actually run. That removes the whole class of bug where an optimistic local
// edit and the running batch disagree.
//
// Polling is deliberately asymmetric. While a batch runs the status changes on
// its own, so it polls; while idle nothing can change except through this hook,
// so it does not — the swap is fighting for the same GPU the UI composites on
// (see the render-lite note in FaceSwap), and a poll nobody needs is pure cost.
// A window focus or a tab becoming visible refreshes once, which covers the
// case this exists for: coming back to a browser that was closed mid-batch.

const IDLE = { jobs: [], running: false, paused: false, current: null };
const POLL_MS = 2000;

export default function useQueue({ notify } = {}) {
  const [state, setState] = useState(IDLE);
  const [busy, setBusy] = useState(false);
  // The poll must not clobber a newer snapshot returned by a mutation that
  // landed while the poll was in flight.
  const stampRef = useRef(0);

  const commit = useCallback((snap, stamp) => {
    if (!snap || typeof snap !== 'object' || !Array.isArray(snap.jobs)) return;
    if (stamp !== undefined && stamp < stampRef.current) return;
    stampRef.current = stamp !== undefined ? stamp : Date.now();
    setState({
      jobs: snap.jobs,
      running: !!snap.running,
      paused: !!snap.paused,
      current: snap.current || null,
    });
  }, []);

  const refresh = useCallback(async () => {
    try {
      const stamp = Date.now();
      commit(await getJSON('/api/queue'), stamp);
    } catch { /* backend not up yet — the next poll or action will catch up */ }
  }, [commit]);

  // `act` is every mutation: one request, commit its snapshot, surface failures.
  const act = useCallback(async (path, body, { quiet = false } = {}) => {
    setBusy(true);
    try {
      const snap = await postJSON(path, body || {});
      commit(snap, Date.now());
      return snap;
    } catch (e) {
      if (!quiet && notify) notify(e.message, 'error');
      // The server refused (a running job, an exhausted queue). Re-read rather
      // than leaving the UI showing what the user tried to do.
      refresh();
      return null;
    } finally {
      setBusy(false);
    }
  }, [commit, notify, refresh]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (!state.running) return undefined;
    const id = setInterval(() => {
      if (!document.hidden) refresh();
    }, POLL_MS);
    return () => clearInterval(id);
  }, [state.running, refresh]);

  useEffect(() => {
    const onWake = () => { if (!document.hidden) refresh(); };
    window.addEventListener('focus', onWake);
    document.addEventListener('visibilitychange', onWake);
    return () => {
      window.removeEventListener('focus', onWake);
      document.removeEventListener('visibilitychange', onWake);
    };
  }, [refresh]);

  const jobs = state.jobs;
  const pending = jobs.filter((j) => RUNNABLE_STATES.includes(jobState(j)));
  const finished = jobs.filter((j) => jobState(j) === 'COMPLETED');
  const retryable = jobs.filter((j) => RETRYABLE_STATES.includes(jobState(j)));
  const active = jobs.filter((j) => ACTIVE_STATES.includes(jobState(j)));

  return {
    ...state,
    busy,
    pending,
    finished,
    retryable,
    active,
    refresh,
    add: (job) => act('/api/queue/add', job),
    addMany: async (list) => {
      if (!list || list.length === 0) return null;
      return act('/api/queue/add_batch', { jobs: list });
    },
    addBatch: (jobs) => act('/api/queue/add_batch', { jobs }),
    join: (ids, name) => act('/api/queue/join', { ids, name }),
    remove: (id) => act('/api/queue/remove', { id }),
    clear: () => act('/api/queue/clear', {}),
    reorder: (ids) => act('/api/queue/reorder', { ids }),
    duplicate: (id) => act('/api/queue/duplicate', { id }),
    update: (id, patch) => act('/api/queue/update', { id, ...patch }),
    retry: (id) => act('/api/queue/retry', id ? { id } : {}),
    // Cooperative cancellation of ONE job. Distinct from remove (which the
    // backend refuses for the running job) and from stop (which ends the whole
    // batch): the running job is asked to stop at its own boundary and every
    // other job is left exactly where it was.
    cancel: (id) => act('/api/queue/cancel', { id }),
    start: () => act('/api/queue/start', {}),
    pause: () => act('/api/queue/pause', {}),
    resume: () => act('/api/queue/resume', {}),
    stop: () => act('/api/queue/stop', {}),
  };
}

// ── Job state ────────────────────────────────────────────────────────────────
//
// routes_queue.py emits BOTH `state` (its own ten-value vocabulary) and
// `status` (a five-value legacy projection kept for older clients). This UI
// used to read `status`, which is lossy in exactly the places a batch needs to
// be legible: PREPARING, PROCESSING, PAUSE_REQUESTED and PAUSED all arrive as
// "running"; CANCELLED and INTERRUPTED both arrive as "stopped"; RECOVERABLE
// arrives as "pending". A job the application was killed during and a job the
// user cancelled read identically, and neither says it can be resumed.
//
// `state` is therefore the field, and `status` is only the fallback for a
// backend older than schema_version 2.

const LEGACY_TO_STATE = {
  pending: 'QUEUED',
  running: 'PROCESSING',
  finished: 'COMPLETED',
  failed: 'FAILED',
  stopped: 'CANCELLED',
};

export const jobState = (job) => (
  String(job?.state || LEGACY_TO_STATE[job?.status] || 'QUEUED').toUpperCase()
);

// Mirrors routes_queue.RUNNABLE_STATES — the states queue_start will pick up.
export const RUNNABLE_STATES = ['QUEUED', 'RECOVERABLE'];
export const ACTIVE_STATES = ['PREPARING', 'PROCESSING', 'PAUSE_REQUESTED', 'PAUSED'];
export const RETRYABLE_STATES = ['FAILED', 'CANCELLED', 'INTERRUPTED'];
export const TERMINAL_STATES = ['COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED'];

// Display helpers, kept beside the states they describe so a new backend state
// cannot be added without this file being the obvious place to update.
export const QUEUE_STATE_LABEL = {
  QUEUED: 'Queued',
  PREPARING: 'Preparing',
  PROCESSING: 'Running',
  PAUSE_REQUESTED: 'Pausing',
  PAUSED: 'Paused',
  COMPLETED: 'Finished',
  FAILED: 'Failed',
  CANCELLED: 'Cancelled',
  INTERRUPTED: 'Interrupted',
  RECOVERABLE: 'Recoverable',
};

// PAUSE_REQUESTED keeps the running row's pulse so the wait for a safe
// checkpoint is visible rather than looking like the click was dropped.
// INTERRUPTED and RECOVERABLE are deliberately not styled as failures: both
// mean "this can be picked up again", which is the opposite of CANCELLED.
export const QUEUE_STATE_CLASS = {
  QUEUED: 'text-white/60 bg-white/5 border-white/5',
  PREPARING: 'text-sky-300 bg-sky-500/10 border-sky-500/30',
  PROCESSING: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30 animate-pulse shadow-[0_0_8px_rgba(234,179,8,0.2)]',
  PAUSE_REQUESTED: 'text-amber-300 bg-amber-500/10 border-amber-500/30 animate-pulse',
  PAUSED: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
  COMPLETED: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  FAILED: 'text-red-400 bg-red-500/10 border-red-500/30',
  CANCELLED: 'text-orange-300 bg-orange-500/10 border-orange-500/30',
  INTERRUPTED: 'text-violet-300 bg-violet-500/10 border-violet-500/30',
  RECOVERABLE: 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30',
};
