// What the Segment Chunk Matrix shows and which actions it offers. Pure; the server's `/api/queue/segments`
// is the source of truth for state, and nothing here invents a state the server did not report.

export const PILLS = Object.freeze({
  queued: { label: 'Queued', tone: 'muted' },
  rendering: { label: 'Rendering', tone: 'amber' },
  completed: { label: 'Completed', tone: 'green' },
  failed: { label: 'Failed', tone: 'red' },
});

// The queue has ten job states (routes_queue.JOB_STATES); the matrix has the four a person asks about.
const STATE_TO_PILL = {
  QUEUED: 'queued', RECOVERABLE: 'queued',
  PREPARING: 'rendering', PROCESSING: 'rendering', PAUSE_REQUESTED: 'rendering', PAUSED: 'rendering',
  COMPLETED: 'completed',
  FAILED: 'failed', CANCELLED: 'failed', INTERRUPTED: 'failed',
};

/** 'queued' | 'rendering' | 'completed' | 'failed'. An unknown state reads as queued, never as done. */
export const pillOf = (state) => STATE_TO_PILL[String(state || '').toUpperCase()] || 'queued';

/** A slice can be redone once it has finished or failed - not while it waits or runs. */
export const canRerender = (seg) => ['completed', 'failed'].includes(pillOf(seg?.state))
  || String(seg?.state).toUpperCase() === 'RECOVERABLE';

/** Slices that are done on paper but whose file is gone: the matrix must not paint those green. */
export const fileMissing = (seg) => pillOf(seg?.state) === 'completed' && !seg.exists;

export function pillCounts(group) {
  const out = { queued: 0, rendering: 0, completed: 0, failed: 0 };
  for (const s of group?.segments || []) out[pillOf(s.state)] += 1;
  return out;
}

/** The pill a slice WEARS: completed-without-a-file is shown as failed ("missing"), with the reason. */
export function wornPill(seg) {
  if (fileMissing(seg)) return { key: 'failed', label: 'Missing', note: `${seg.name} is not on disk` };
  const key = pillOf(seg?.state);
  return { key, label: PILLS[key].label, note: key === 'failed' ? (seg.error || null) : null };
}

const secs = (v) => (typeof v === 'number' ? `${v.toFixed(2)} s` : '?');

/** "3.00 s of 3.00 s" or the server's reason; null when the join was never verified. */
export function verifyLabel(verify) {
  if (!verify) return null;
  if (verify.ok === null) return { ok: null, text: 'not measured (no reference duration)' };
  if (verify.ok) return { ok: true, text: `duration ${secs(verify.joined_s)} matches ${secs(verify.expected_s)}` };
  return { ok: false, text: verify.reason || `duration ${secs(verify.joined_s)} vs ${secs(verify.expected_s)}` };
}

/** One line for the card header. */
export function groupSummary(group) {
  const c = pillCounts(group);
  const parts = [`${c.completed} of ${group.count} rendered`];
  if (c.rendering) parts.push(`${c.rendering} rendering`);
  if (c.queued) parts.push(`${c.queued} queued`);
  if (c.failed) parts.push(`${c.failed} failed`);
  const j = group.joined;
  if (j?.exists) {
    const v = verifyLabel(j.result?.verify);
    parts.push(v && v.ok === false ? 'joined - DURATION MISMATCH' : 'joined');
  }
  return parts.join(' · ');
}

/** Why the join button is off, or '' when it is on. */
export const joinDisabledReason = (group, busy = false) => {
  if (busy) return 'Joining…';
  if (group.can_join) return '';
  return group.join_blocked || 'Not every slice is finished';
};

/** Groups worth showing: any split (>= 2 slices). A lone ranged job is a trim, not a split. */
export const visibleGroups = (groups) => (groups || []).filter((g) => (g.count || 0) >= 2 || (g.segments || []).length >= 2);

/** True while anything in the set is moving, so the view should keep polling. */
export const isActive = (groups) => (groups || []).some((g) => (g.segments || []).some((s) => {
  const p = pillOf(s.state);
  return p === 'rendering' || p === 'queued';
}));
