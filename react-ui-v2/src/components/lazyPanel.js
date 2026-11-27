import { createElement, lazy } from 'react';

// A code-split tab panel that can be RE-TRIED after its chunk failed to download.
//
// Plain React.lazy cannot. It memoizes the result of its first load, including a
// rejection: once `import('./Gallery')` fails (the dev server restarted, the
// connection dropped mid-navigation) that component re-throws the same error on
// every render for the life of the page. ErrorBoundary's "Retry" remounted the
// subtree and nothing happened -- no new request was even made (measured in
// e2e/error-boundary.spec.js) -- so the only way out was a full reload, while the
// panel said "Retry once it is back up".
//
// lazyPanel(load) returns a component with the same contract as lazy(load), plus:
// when `load` rejects it remembers that, and resetFailedPanels() swaps in a FRESH
// React.lazy for each failed panel so the next render runs `load()` again. Panels
// that loaded fine are never touched.
const resetters = new Set();

export function lazyPanel(load) {
  let failed = false;
  const make = () => lazy(() => load().catch((err) => {
    failed = true;
    throw err;
  }));
  let Inner = make();
  resetters.add(() => {
    if (!failed) return;
    failed = false;
    Inner = make();
  });
  // Reads `Inner` at render time, so swapping it takes effect on the next render
  // without anyone holding a stale component reference.
  const Panel = (props) => createElement(Inner, props);
  Panel.displayName = 'LazyPanel';
  return Panel;
}

/** Idempotent: safe to call from a render-phase hook and from an event handler. */
export function resetFailedPanels() {
  resetters.forEach((reset) => reset());
}
