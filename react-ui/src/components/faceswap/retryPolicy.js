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
// attempt LATER, and the last one final.
//
// DETERMINISTIC, three retries (Stage 5 brief; it replaces the 0.5 s .. 8 s jittered
// five-attempt schedule of a786b33):
//
//   attempt 1 fails -> wait 250 ms, 2 -> 500 ms, 3 -> 1000 ms,
//   attempt 4 fails -> give up (until the URL changes or the user retries):
//   one request plus three retries, 1.75 s of waiting in all.
//
// No jitter. Scrubbing at 0.25x-4x produces a burst of frame requests that fail
// together when the decoder stalls; the schedule is a property the tests and the
// user can both read off the clock, and a cap of 1 s keeps "Frame unavailable - Retry"
// from arriving long after the user has stopped looking at it. (A TensorRT build is
// not a frame failure: those requests are suspended by api.js, not retried here.)

export const RETRY_DELAYS_MS = Object.freeze([250, 500, 1000]);
export const MAX_RETRIES = RETRY_DELAYS_MS.length;
/** Consecutive failures that exhaust a key: the first attempt plus MAX_RETRIES retries. */
export const MAX_FAILURES = MAX_RETRIES + 1;

/** Delay before the retry that follows the `failures`-th consecutive failure (1-based; clamped). */
export function backoffDelay(failures) {
  const n = Math.max(1, Math.floor(failures) || 1);
  return RETRY_DELAYS_MS[Math.min(n, RETRY_DELAYS_MS.length) - 1];
}

/** True for the rejection every superseded request ends in: not a failure. */
export const isAbort = (err) => err?.name === 'AbortError';

/**
 * Consecutive-failure bookkeeping for ONE thing being asked for at a time (a
 * frame URL, a playback chunk). The count belongs to a key: asking for a
 * different key starts from zero, which is what makes "the user moved on" and
 * "the user pressed retry" the only two ways out of the given-up state.
 */
export function createFailureTracker({ maxFailures = MAX_FAILURES } = {}) {
  let key = null;
  let count = 0;
  return {
    /** Record a failure of `k`. `delay` is null once the budget is spent. */
    fail(k) {
      if (k !== key) { key = k; count = 0; }
      count += 1;
      const exhausted = count >= maxFailures;
      return { count, exhausted, delay: exhausted ? null : backoffDelay(count) };
    },
    /** `k` succeeded: the streak is over. */
    ok(k) { if (k === key) count = 0; },
    /** Has `k` used up its attempts? */
    exhausted(k) { return k === key && count >= maxFailures; },
    count(k) { return k === key ? count : 0; },
    reset() { key = null; count = 0; },
  };
}

// ── A busy backend is not a failing one ───────────────────────────────────
//
// While the pipeline's pre-pass (Stage 1 sparse pre-scan, Stage 2 active-range pre-pass) has the GPU
// flat out, `/api/target/preview*` answers an UNCACHED frame with a fast
// `503 {busy: true, stage, retry_after_ms}` + `Retry-After` instead of queueing a seek behind the
// detectors (api._preview_busy_response). That is the backend saying "ask again in N ms", so it must
// not spend the failure budget above: three of those would exhaust it in 1.75 s and put "Frame
// unavailable" over a perfectly healthy render. A busy refusal waits as long as it was told to
// (clamped, so a bad hint can neither hammer nor park the requester) and costs nothing.
export const BUSY_MIN_MS = 250;
export const BUSY_MAX_MS = 5000;
export const BUSY_DEFAULT_MS = 1000;

const clampBusy = (ms) => Math.min(BUSY_MAX_MS, Math.max(BUSY_MIN_MS, ms));

/**
 * The wait a 503 asked for: the body's `retry_after_ms`, else the `Retry-After` header (seconds),
 * else a default, clamped to [BUSY_MIN_MS, BUSY_MAX_MS]. Never NaN.
 */
export function busyRetryMs(bodyMs, headerSeconds) {
  const fromBody = Number(bodyMs);
  if (Number.isFinite(fromBody) && fromBody > 0) return clampBusy(fromBody);
  const fromHeader = Number(headerSeconds);
  if (Number.isFinite(fromHeader) && fromHeader > 0) return clampBusy(fromHeader * 1000);
  return BUSY_DEFAULT_MS;
}

/** The Error a frame request rejects with for a non-2xx answer; `busy` marks "retry later". */
export function httpFailure(status, { busy = false, retryAfterMs = null } = {}) {
  return Object.assign(new Error(`HTTP ${status}`), {
    status, busy: !!busy, retryAfterMs: busy ? busyRetryMs(retryAfterMs) : null,
  });
}

/** True for a refusal that asked the client to come back later (not a failure). */
export const isBusy = (err) => !!err && err.busy === true;

/** How long to wait before asking again after a busy refusal, or null when `err` is not one. */
export function busyDelay(err) {
  return isBusy(err) ? busyRetryMs(err.retryAfterMs) : null;
}
