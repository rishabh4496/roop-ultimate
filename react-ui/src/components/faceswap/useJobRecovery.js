import { useCallback, useEffect, useRef } from 'react';
import { getJSON } from '../../api';
import { useJobStore } from '../../store/jobStore';

// ── Reattach to a render that is already in flight ────────────────────────
//
// Runs once on mount, then keeps a slow heartbeat. See jobStore.js for why the
// order (ask the server FIRST, then reconcile the remembered job) is the whole
// design.
//
// `/api/jobs/active` is the endpoint this wants. It may not be there:
// `app/api.py` runs in a NON-RELOADING uvicorn thread, so a user who pulls this
// change and does not restart the backend is talking to a server that has never
// heard of the route, and gets a 404. That is not hypothetical — it is the
// documented failure mode for every route added to this app, and it presents as
// "the button does nothing".
//
// So the fallback is not defensive padding, it is the common case for one
// launch: on a 404 we synthesise the same shape from `/api/progress`, which has
// carried `processing` and `started_at` for a long time. The UI reconnects
// either way; only `job_id` and the queue tail are missing on the old path.

const HEARTBEAT_MS = 15000;

const normalise = (pr) => ({
  processing: !!pr.processing,
  job_id: pr.job_id ?? null,
  started_at: typeof pr.started_at === 'number' ? pr.started_at : null,
  label: pr.label || pr.output?.path || '',
  queued: Array.isArray(pr.queued) ? pr.queued : [],
  progress: pr,
});

export default function useJobRecovery({ onReattach } = {}) {
  const beginReconcile = useJobStore((s) => s.beginReconcile);
  const reconcile = useJobStore((s) => s.reconcile);
  // The endpoint's absence is a property of the backend, not of this render —
  // remember it so we stop paying a 404 every heartbeat.
  const legacyRef = useRef(false);
  const onReattachRef = useRef(onReattach);
  onReattachRef.current = onReattach;
  const announcedRef = useRef(false);

  const poll = useCallback(async () => {
    try {
      let snap;
      if (!legacyRef.current) {
        try {
          snap = normalise(await getJSON('/api/jobs/active', { timeout: 8000 }));
        } catch (e) {
          // Only a genuinely-missing route falls back. A timeout or a network
          // error must NOT be read as "this backend is old", or one stalled
          // request would permanently downgrade the client.
          if (!/404|not found/i.test(e?.message || '')) throw e;
          legacyRef.current = true;
        }
      }
      if (!snap) snap = normalise(await getJSON('/api/progress', { timeout: 8000 }));

      const before = useJobStore.getState();
      const hadMemory = !!before.jobId;
      reconcile(snap);
      const after = useJobStore.getState();

      // Announce ONCE per reattachment, and only when this really is a job the
      // window did not start — a job started in this session reconciles to
      // 'active' too, and telling the user we "reconnected" to their own click
      // is noise.
      if (!announcedRef.current && hadMemory && after.phase === 'active' && after.reattached) {
        announcedRef.current = true;
        onReattachRef.current?.({
          jobId: after.jobId,
          startedAt: after.startedAt,
          label: after.label,
          settings: after.settings,
          progress: snap.progress,
        });
      }
      if (after.phase !== 'active') announcedRef.current = false;
    } catch {
      reconcile(null);   // unreachable — hold, do not declare the run over
    }
  }, [reconcile]);

  useEffect(() => {
    beginReconcile();
    poll();
    // A slow heartbeat only. While a run is live, App's own 1 s progress poll is
    // what drives the bar; this exists to notice a job that STARTED elsewhere
    // (the queue advancing, a second window, a CLI run) and to retire a memory
    // whose job has ended.
    const id = setInterval(() => { if (!document.hidden) poll(); }, HEARTBEAT_MS);
    const onWake = () => { if (!document.hidden) poll(); };
    document.addEventListener('visibilitychange', onWake);
    window.addEventListener('focus', onWake);
    window.addEventListener('pageshow', onWake);
    return () => {
      clearInterval(id);
      document.removeEventListener('visibilitychange', onWake);
      window.removeEventListener('focus', onWake);
      window.removeEventListener('pageshow', onWake);
    };
  }, [beginReconcile, poll]);

  return { refresh: poll };
}
