import { useEffect, useMemo, useRef } from 'react';
import * as client from './api';

// api.js's request functions, bound to the lifetime of the component that calls them.
//
//   const { getJSON, postJSON } = useApi();      // shadows the module-level imports
//
// WHY. A tab panel unmounts the moment you switch tabs. A request it started is still in
// flight; when it lands it calls setState on a component that is gone, and for a slow one
// (a status poll against a busy backend, a preview) it also keeps a connection and a
// handler alive for nothing. Aborting on unmount ends both.
//
// WHAT IS ABORTED. Reads: every getJSON, and any other call that passes
// `abortOnUnmount: true` (the POSTs that only compute something: a preview, an estimate).
// NOT aborted by default: mutations and uploads. Aborting a POST client-side does not undo
// it, it only stops the page learning whether it happened: a source added or a setting
// saved on the server with no UI to show it. Those complete, and their result is dropped
// by the code that already guards its own setState.
//
// An abort here is not a failure: failureLog ignores AbortError, and api.js rethrows it
// unchanged, so the `.catch(logFailure(...))` after a request that was cancelled on
// unmount is silent.

/**
 * The lifetime of one mounted component as an AbortSignal. Pure (no React) so it can be
 * tested; the hook below drives it from effects.
 *
 * React StrictMode mounts, unmounts and mounts again in development, so `mount()` after
 * an `unmount()` starts a fresh controller; `signal` is read at CALL time, never cached.
 */
export function createUnmountScope() {
  let ctrl = new AbortController();
  return {
    get signal() { return ctrl.signal; },
    mount() { if (ctrl.signal.aborted) ctrl = new AbortController(); },
    unmount() { ctrl.abort(new DOMException('Component unmounted', 'AbortError')); },
  };
}

/** Merge a caller's own signal with the scope's, so either one aborts the request. */
export function mergeSignals(a, b) {
  if (!a) return b;
  if (!b) return a;
  if (a.aborted) return a;
  if (b.aborted) return b;
  const ctrl = new AbortController();
  const forward = (from) => () => ctrl.abort(from.reason);
  a.addEventListener('abort', forward(a), { once: true });
  b.addEventListener('abort', forward(b), { once: true });
  return ctrl.signal;
}

// TWO LIFETIMES, KEPT APART.
//
//   view scope  (the default)   one per mounted component; aborted when it unmounts. Hash-route
//                               navigation (#/faceswap -> #/batch -> #/processing) unmounts the
//                               tab, so everything the tab was merely LOOKING at stops with it.
//   job scope   `scope: 'job'`  one for the whole page; aborted only by abortJobRequests(), which
//                               a deliberate action calls (clearing the queue, resetting a run).
//                               For requests that belong to the QUEUE or the RUN rather than to
//                               the tab that happened to start them: the shared queue poll, a
//                               job-status read. Leaving the tab must not cancel those, or the
//                               other tab that is rendering the same job loses its refresh.
//
// A request in job scope ignores the unmount scope entirely; a caller's own `signal` still
// applies to both.
let jobCtrl = new AbortController();

export const jobScope = {
  get signal() { return jobCtrl.signal; },
};

/** Cancel every in-flight `scope: 'job'` request, then start a fresh job scope. */
export function abortJobRequests(reason = 'Job requests cancelled') {
  const old = jobCtrl;
  jobCtrl = new AbortController();
  old.abort(new DOMException(reason, 'AbortError'));
}

/** Build the bound functions over a scope (exported for tests). */
export function bindApi(scope) {
  const signalFor = (opts, viewDefault) => {
    if (opts?.scope === 'job') return mergeSignals(opts.signal, jobScope.signal);
    return viewDefault ? mergeSignals(opts?.signal, scope.signal) : opts?.signal;
  };
  const withScope = (opts, viewDefault) => {
    const signal = signalFor(opts, viewDefault);
    return signal === opts?.signal ? opts : { ...opts, signal };
  };
  const wanted = (opts) => !!opts?.abortOnUnmount;
  return {
    getJSON: (path, opts = {}) => client.getJSON(path, withScope(opts, true)),
    postJSON: (path, body, opts = {}) => client.postJSON(path, body, withScope(opts, wanted(opts))),
    postFile: (path, file, fields, opts = {}) => client.postFile(path, file, fields, withScope(opts, wanted(opts))),
    postFiles: (path, files, fields, opts = {}) => client.postFiles(path, files, fields, withScope(opts, wanted(opts))),
  };
}

export function useApi() {
  const scope = useRef(null);
  if (scope.current === null) scope.current = createUnmountScope();
  useEffect(() => {
    const s = scope.current;
    s.mount();
    return () => s.unmount();
  }, []);
  // One stable object for the component's life, so putting `getJSON` in a dependency
  // array re-runs nothing.
  return useMemo(() => bindApi(scope.current), []);
}
