import { useEffect, useRef, useState } from 'react';
import { getJSON } from '../../api';
import { reportFailure } from '../../failureLog';
import { TERMINAL_STATES, jobState } from '../faceswap/useQueue';

// How many queue jobs are still to run (running or waiting), for the rail's
// Processing badge. The queue itself belongs to Face Swap (faceswap/useQueue,
// mounted only on that tab), so the shell cannot read it from there; this is
// the same /api/queue snapshot Home already pulls, counted the same way
// (Home: `!TERMINAL_STATES.includes(jobState(j))`).
//
// One small GET every POLL_MS, and only while the page is visible: a hidden
// Pinokio webview should not keep asking. The count is advisory, so a failure
// keeps the last number and is logged without a toast.
const POLL_MS = 5000;

export default function useQueueSummary() {
  const [pending, setPending] = useState(0);
  // The last count handed to React: an unchanged count is not written at all, so the poll is silent on an
  // idle tab (e2e/frame-budget.spec.js holds those to <= 6 commits a second).
  const lastRef = useRef(0);

  useEffect(() => {
    let live = true;
    let timer = null;

    const pull = async () => {
      if (document.visibilityState === 'hidden') return;
      try {
        const r = await getJSON('/api/queue', { timeout: 8000 });
        const jobs = Array.isArray(r?.jobs) ? r.jobs : [];
        const n = jobs.filter((j) => !TERMINAL_STATES.includes(jobState(j))).length;
        if (live && n !== lastRef.current) { lastRef.current = n; setPending(n); }
      } catch (e) {
        reportFailure('Reading the queue for the navigation badge', e, { toast: false });
      }
    };

    pull();
    timer = setInterval(pull, POLL_MS);
    document.addEventListener('visibilitychange', pull);
    return () => {
      live = false;
      clearInterval(timer);
      document.removeEventListener('visibilitychange', pull);
    };
  }, []);

  return pending;
}
