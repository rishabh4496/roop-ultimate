import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

// ── Active-job state that survives a page refresh ─────────────────────────
//
// The problem this exists for is specific to how this app is hosted. Pinokio
// RELOADS THE WEBVIEW on every tab switch (React UI <-> Terminal, Run <-> Dev),
// and a render here routinely runs for forty minutes. So the client is torn
// down and rebuilt many times during a single job, through no action of the
// user's. Anything the panel knew that the backend does not — which job this
// window started, what settings it was started with, when it started — was
// gone, and the UI came back reading "idle" over a GPU that was still rendering
// until the next poll happened to say otherwise.
//
// Two halves, and they are different in kind:
//
//   * `localStorage` holds what only THIS CLIENT knows: the job id it started,
//     the settings snapshot it started with, its own start clock. It is a
//     memory, not a source of truth, and it is never trusted on its own — a
//     remembered job can have finished, failed, or been started by a different
//     window entirely.
//   * `/api/jobs/active` is the truth. On mount we ask the server what is
//     actually running and RECONCILE: a remembered job the server confirms is
//     reattached (progress polling resumes, the run tab reopens, elapsed time
//     continues from the backend's own clock); one it does not is retired.
//
// Getting that order wrong is the whole trap. Restoring from localStorage and
// then polling gives a window that confidently shows a progress bar for a job
// that ended half an hour ago; polling and then merging gives one that briefly
// shows "idle" over a live render. So the store starts in an explicit
// `reconciling` state that is NEITHER, and no consumer may read `isActive`
// until it has resolved.

const STORAGE_KEY = 'roop_active_job';

// Bumped whenever the persisted shape changes. An older payload is dropped
// rather than migrated — the only thing lost is a reattachment, and a wrong
// reattachment is worse than none.
const VERSION = 1;

// A remembered job older than this is not worth offering to reattach: the
// server has long since finished or lost it, and the reconcile would only ever
// retire it. Generous, because renders here are measured in hours.
const MAX_AGE_MS = 12 * 60 * 60 * 1000;

export const useJobStore = create(persist((set, get) => ({
  // ── persisted ───────────────────────────────────────────────────────────
  /** Backend job id, when the backend gives us one. */
  jobId: null,
  /** Epoch ms this client believes the job started. */
  startedAt: null,
  /** The settings the job was started with — so a reattached run can show
   *  what it is actually rendering, not whatever the panel drifted to since. */
  settings: null,
  /** Free-form label for the UI (output name, target file). */
  label: '',
  /** Queue job ids this client is waiting on, in order. */
  queuedIds: [],

  // ── session-only ────────────────────────────────────────────────────────
  /** 'unknown' | 'reconciling' | 'active' | 'idle'. Starts 'unknown'. */
  phase: 'unknown',
  /** The server's own view, from the last successful reconcile. */
  server: null,
  /** True once a remembered job has been matched to a live server job. */
  reattached: false,
  lastError: '',

  // ── actions ─────────────────────────────────────────────────────────────
  beginJob: ({ jobId = null, settings = null, label = '', startedAt = null } = {}) =>
    set({
      jobId,
      settings,
      label,
      startedAt: startedAt || Date.now(),
      phase: 'active',
      reattached: false,
      lastError: '',
    }),

  /** Record queue ids so a refresh mid-batch can still say what is pending. */
  setQueuedIds: (ids) => set({ queuedIds: Array.isArray(ids) ? ids.slice(0, 200) : [] }),

  endJob: () => set({
    jobId: null, startedAt: null, settings: null, label: '',
    phase: 'idle', reattached: false,
  }),

  beginReconcile: () => set({ phase: 'reconciling' }),

  /**
   * Fold the server's answer into the store. THE ONLY place `phase` becomes
   * 'active' or 'idle' after boot.
   *
   * @param snap  { processing, job_id, started_at, label, queued, ... } or null
   *              when the server could not be reached.
   */
  reconcile: (snap) => {
    if (!snap) {
      // Unreachable is not "not running". Hold whatever we remembered and stay
      // out of 'idle' — a backend restarting under a live render must not make
      // this window declare the render over.
      set({ phase: get().jobId ? 'active' : 'idle', lastError: 'backend unreachable' });
      return;
    }
    const remembered = get().jobId;
    const running = !!snap.processing;
    const serverId = snap.job_id ?? null;

    if (!running) {
      set({
        server: snap,
        phase: 'idle',
        reattached: false,
        lastError: '',
        // Retire the memory: keeping it would offer a reattach forever.
        jobId: null, startedAt: null, label: '',
      });
      return;
    }

    // Running. Reattach when the ids agree, or when the server does not issue
    // ids at all (this backend historically did not) and we remembered one —
    // there is only ever one render in flight, so an unnamed live job IS ours.
    const matches = serverId == null || remembered == null || serverId === remembered;
    set({
      server: snap,
      phase: 'active',
      reattached: matches && remembered != null,
      lastError: '',
      jobId: serverId ?? remembered,
      // The BACKEND's clock wins. A remembered start time is this window's, and
      // this window may be minutes younger than the run.
      startedAt: snap.started_at ? snap.started_at * 1000 : (get().startedAt || Date.now()),
      label: snap.label || get().label,
      queuedIds: Array.isArray(snap.queued) ? snap.queued : get().queuedIds,
    });
  },

  setError: (msg) => set({ lastError: msg || '' }),
}), {
  name: STORAGE_KEY,
  version: VERSION,
  storage: createJSONStorage(() => {
    // A webview with storage blocked must not take the app down at import time.
    try {
      const probe = '__roop_probe__';
      window.localStorage.setItem(probe, '1');
      window.localStorage.removeItem(probe);
      return window.localStorage;
    } catch {
      const mem = new Map();
      return {
        getItem: (k) => mem.get(k) ?? null,
        setItem: (k, v) => mem.set(k, v),
        removeItem: (k) => mem.delete(k),
      };
    }
  }),
  // Persist ONLY what the server cannot tell us. `phase`, `server` and
  // `reattached` are conclusions, and a persisted conclusion is exactly the
  // stale-progress-bar bug described above.
  partialize: (s) => ({
    jobId: s.jobId,
    startedAt: s.startedAt,
    settings: s.settings,
    label: s.label,
    queuedIds: s.queuedIds,
  }),
  migrate: () => ({}),          // no migrations yet; drop anything older
  onRehydrateStorage: () => (state) => {
    if (!state) return;
    if (state.startedAt && Date.now() - state.startedAt > MAX_AGE_MS) {
      // Too old to be real. Clear it here rather than letting the reconcile do
      // it, so nothing ever renders a twelve-hour-old "in progress".
      state.jobId = null;
      state.startedAt = null;
      state.label = '';
      state.settings = null;
    }
  },
}));

/** True only once the server has actually been asked. */
export const selectIsActive = (s) => s.phase === 'active';
export const selectIsResolved = (s) => s.phase === 'active' || s.phase === 'idle';
/** A job this window did not start, that it has adopted. */
export const selectRecoveredJob = (s) =>
  (s.phase === 'active' && s.reattached ? { jobId: s.jobId, startedAt: s.startedAt, label: s.label, settings: s.settings } : null);
