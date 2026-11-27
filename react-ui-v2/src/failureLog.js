// Where a swallowed failure goes instead of nowhere.
//
// The UI had ~40 `.catch(() => {})` / `catch { /* ignore */ }` around backend calls: saving
// a setting, loading a panel's data, a status poll. When one of them failed the user saw
// nothing, the console saw nothing, and the symptom was a control that quietly did not
// work. They report here now:
//
//   postJSON('/api/settings', patch).catch(logFailure('Saving settings'));
//
// WHAT IS REPORTED, AND HOW OFTEN
//   * Each DISTINCT failure once: the same context failing the same way (same endpoint,
//     status and message) is counted, not repeated. A poll that fails every second for a
//     minute is one console line, not sixty.
//   * To the console always (`console.warn`, with the error object so devtools shows the
//     stack and ApiError's status / method / path).
//   * To a toast unless the call says `{ toast: false }` (a background poll whose failure
//     the app already shows another way, like the "engine unavailable" banner) AND the
//     toast budget allows: at most TOAST_BURST toasts per TOAST_WINDOW_MS, so a backend
//     that dies and takes twelve things with it produces two toasts and twelve console
//     lines, not twelve stacked toasts.
//   * Never an abort. A request cancelled because its component unmounted, or because
//     the user cancelled it, is not a failure.
//
// WHAT STAYS SILENT ON PURPOSE. Empty catches around BROWSER APIs (`localStorage` in a
// private window, `video.play()` rejected by autoplay policy, `ws.close()` on a closed
// socket, pointer capture) are not backend failures; the browser refusing is the expected
// path there, and a toast for it would be noise. They keep their comment saying why.

const TOAST_BURST = 2;
const TOAST_WINDOW_MS = 30_000;
const MAX_REMEMBERED = 200;

const seen = new Map();          // key -> { count, first }
let toastTimes = [];
let sink = null;
let clock = () => Date.now();

/** Register the toast function (App's `notify`). Returns an unregister function. */
export function setFailureToast(fn) {
  sink = fn;
  return () => { if (sink === fn) sink = null; };
}

/** A cancelled request is expected, not a failure. */
export const isAbortError = (err) => !!err && (err.name === 'AbortError' || err.code === 20);

const messageOf = (err) => {
  if (err && typeof err.message === 'string' && err.message) return err.message;
  return String(err);
};

// ApiError carries what makes two failures "the same": the request and how it failed.
// Anything else falls back to its name and message.
export function failureKey(context, err) {
  if (err && err.name === 'ApiError') {
    return `${context}|${err.kind}|${err.status}|${err.method} ${err.path}|${err.detail}`;
  }
  return `${context}|${(err && err.name) || typeof err}|${messageOf(err)}`;
}

/**
 * Report a failure. Returns true if it was NEW (first time this context failed this way),
 * false if it was an abort or a repeat. `context` is a phrase a person can read in a
 * toast: "Saving settings", "Loading the run history".
 */
export function reportFailure(context, err, { toast = true } = {}) {
  if (isAbortError(err)) return false;
  const key = failureKey(context, err);
  const known = seen.get(key);
  if (known) { known.count += 1; return false; }

  if (seen.size >= MAX_REMEMBERED) seen.delete(seen.keys().next().value);
  seen.set(key, { count: 1, first: clock() });

  const message = messageOf(err);
  const now = clock();
  toastTimes = toastTimes.filter((t) => now - t < TOAST_WINDOW_MS);
  const canToast = toast && sink && toastTimes.length < TOAST_BURST;
  console.warn(`[roop-ui] ${context} failed: ${message}${toast && sink && !canToast ? ' (toast suppressed: rate limit)' : ''}`, err);
  if (canToast) {
    toastTimes.push(now);
    sink(`${context} failed: ${message}`.slice(0, 240), 'warning');
  }
  return true;
}

/** `.catch(logFailure('Saving settings'))`; the returned handler resolves to undefined. */
export const logFailure = (context, options) => (err) => { reportFailure(context, err, options); };

/** How many times each distinct failure has happened (for devtools: `__roopFailures()`). */
export const failureCounts = () => Object.fromEntries([...seen].map(([k, v]) => [k, v.count]));
if (typeof window !== 'undefined') window.__roopFailures = failureCounts;

/** Test hooks. */
export function __resetForTests({ now } = {}) {
  seen.clear();
  toastTimes = [];
  sink = null;
  clock = now || (() => Date.now());
}
