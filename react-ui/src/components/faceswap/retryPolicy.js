// ── How long to wait before asking again, and when to stop asking ─────────
//
// Pure, so .render-check/retry-policy-check.mjs can test it without a browser.
//
// Why this exists. useThrottledFrameRequest and usePlaybackBuffer both treated a
// failed request as "the wanted thing is still not here, so ask again": the first
// re-armed itself from `finally` through a 150 ms throttle, the second from its
// next animation frame. A backend that answers 500 therefore got ~6.5 preview
// requests a second from an idle Face Swap tab (196 in 30 s, measured by
// react-ui/e2e/idle-requests.spec.js) and a playing timeline asked for the same
// chunk up to 60 times a second. A failure is information; it must make the next
// attempt LATER, and the Nth one the last.
//
//   attempt 1 fails -> wait ~0.5 s, 2 -> ~1 s, 3 -> ~2 s, 4 -> ~4 s,
//   attempt 5 fails -> give up (until the URL changes or the user retries).
//
// Each delay carries +/-25% jitter so several tabs or panels that failed together
// do not come back together, and is held inside [0.5 s, 8 s]. The 8 s cap is
// unreachable at 5 attempts (the 5th failure gives up before it would wait 8 s);
// it is there so a caller that raises `maxFailures` stays bounded.

export const BASE_DELAY_MS = 500;
export const MAX_DELAY_MS = 8000;
export const MAX_FAILURES = 5;

/** Delay before the attempt that follows the `failures`-th consecutive failure. */
export function backoffDelay(failures, rand = Math.random) {
  const n = Math.max(1, Math.floor(failures) || 1);
  const base = Math.min(MAX_DELAY_MS, BASE_DELAY_MS * 2 ** (n - 1));
  return Math.min(MAX_DELAY_MS, Math.max(BASE_DELAY_MS, base * (0.75 + rand() * 0.5)));
}

/** True for the rejection every superseded request ends in: not a failure. */
export const isAbort = (err) => err?.name === 'AbortError';

/**
 * Consecutive-failure bookkeeping for ONE thing being asked for at a time (a
 * frame URL, a playback chunk). The count belongs to a key: asking for a
 * different key starts from zero, which is what makes "the user moved on" and
 * "the user pressed retry" the only two ways out of the given-up state.
 */
export function createFailureTracker({ maxFailures = MAX_FAILURES, rand = Math.random } = {}) {
  let key = null;
  let count = 0;
  return {
    /** Record a failure of `k`. `delay` is null once the budget is spent. */
    fail(k) {
      if (k !== key) { key = k; count = 0; }
      count += 1;
      const exhausted = count >= maxFailures;
      return { count, exhausted, delay: exhausted ? null : backoffDelay(count, rand) };
    },
    /** `k` succeeded: the streak is over. */
    ok(k) { if (k === key) count = 0; },
    /** Has `k` used up its attempts? */
    exhausted(k) { return k === key && count >= maxFailures; },
    count(k) { return k === key ? count : 0; },
    reset() { key = null; count = 0; },
  };
}
