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
  const pending = jobs.filter((j) => !['finished', 'failed', 'stopped'].includes(j.status));
  const finished = jobs.filter((j) => j.status === 'finished');
  const retryable = jobs.filter((j) => j.status === 'failed' || j.status === 'stopped');

  return {
    ...state,
    busy,
    pending,
    finished,
    retryable,
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
    start: () => act('/api/queue/start', {}),
    pause: () => act('/api/queue/pause', {}),
    resume: () => act('/api/queue/resume', {}),
    stop: () => act('/api/queue/stop', {}),
  };
}

// Display helpers, kept beside the statuses they describe so a new backend
// status cannot be added without this file being the obvious place to update.
export const QUEUE_STATUS_LABEL = {
  pending: 'Pending',
  running: 'Running',
  finished: 'Finished',
  failed: 'Failed',
  stopped: 'Stopped',
};

export const QUEUE_STATUS_CLASS = {
  pending: 'text-white/60 bg-white/5 border-white/5',
  running: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30 animate-pulse shadow-[0_0_8px_rgba(234,179,8,0.2)]',
  paused: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
  finished: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  failed: 'text-red-400 bg-red-500/10 border-red-500/30',
  stopped: 'text-orange-300 bg-orange-500/10 border-orange-500/30',
};
