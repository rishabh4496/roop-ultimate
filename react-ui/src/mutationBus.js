// "A request that changes backend state just succeeded" - announced once, from the API client, to whoever cares.
//
// Why a bus and not a call from each screen: the floating preview (pip/pipStore.js) wants to know that the FACE BANK changed
// the moment a Face Manager edit lands, and Face Manager has a dozen mutating calls across four files. Asking each of them to
// remember to tell the PiP is how one of them gets forgotten; the client that sends them all already knows when one succeeds.
//
// Listeners run synchronously after the response is parsed and must not throw into the caller (a failure here must never
// turn a successful save into a rejected promise), so each is isolated.

const listeners = new Set();

/** Subscribe; returns the unsubscribe. */
export function onMutation(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Called by api.js after a non-GET request succeeds. `path` is the bare path (no query string). */
export function noteMutation(method, path) {
  if (String(method).toUpperCase() === 'GET' || listeners.size === 0) return;
  const p = String(path).split('?')[0];
  for (const fn of [...listeners]) {
    try { fn({ method, path: p }); } catch { /* a listener's bug is not the request's failure */ }
  }
}
