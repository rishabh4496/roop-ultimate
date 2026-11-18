// The queue's job-state vocabulary, as pure data (no React, no network).
//
// Split out of useQueue.js so pure view-models and the render-checks can read the
// states without loading the API client. useQueue.js re-exports everything here, so
// existing imports from './useQueue' are unchanged.

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
