/**
 * The one ErrorBoundary, exercised for real.
 *
 * There used to be two (a bare copy in main.jsx and components/ErrorBoundary.jsx)
 * that had drifted apart. They are one component with two variants now, and what
 * must stay true of it:
 *
 *   * variant="app" is SELF-CONTAINED. It is the last line of defence, so it
 *     must draw when the CSS bundle, the theme or the icon set is what broke:
 *     no class attributes, no CSS variables, inline styles only.
 *   * variant="panel" (the default) is the tab-panel fallback: it uses the
 *     app's classes and icons, tells a failed lazy chunk from a real error, and
 *     is cleared by a changed `resetKey`.
 *   * both keep the error visible, offer Retry (a remount) and Reload.
 *   * a render with no error is a pass-through of the children.
 *
 * React's server renderer does not run error boundaries, so a real throw cannot
 * be provoked here. Instead the REAL class is instantiated and its real static
 * hooks, `retry` and `render` are driven with a forced error state, and the
 * element it returns is rendered to markup. The e2e suite covers the live path
 * (nav-visibility is unrelated; error-boundary.spec.js breaks a lazy chunk).
 */
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import ErrorBoundary from '../src/components/ErrorBoundary.jsx';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

const here = dirname(fileURLToPath(import.meta.url));
const SRC = join(here, '..', 'src');

// A boundary instance whose setState is applied synchronously, in the state it
// would have after catching `error`.
const crashed = (props, error = new Error('kaboom: cannot read properties of undefined')) => {
  const b = new ErrorBoundary({ ...props, children: null });
  b.state = { ...b.state, ...ErrorBoundary.getDerivedStateFromError(error) };
  b.setState = (fn) => { b.state = { ...b.state, ...(typeof fn === 'function' ? fn(b.state) : fn) }; };
  return b;
};
const html = (b) => renderToStaticMarkup(b.render());

console.log('── variant="app": self-contained ───────────────────────────');
{
  const h = html(crashed({ variant: 'app' }));
  ok('shows the error message', h.includes('kaboom: cannot read properties of undefined'));
  ok('says the app crashed', h.includes('The app crashed'));
  ok('is an alert', /role="alert"/.test(h));
  ok('has NO class attributes (draws without Tailwind)', !/\sclass=/.test(h), h.match(/\sclass="[^"]*"/)?.[0]);
  ok('uses no CSS variables (draws without the theme)', !/var\(--/.test(h));
  ok('has no <svg> (draws without the icon set)', !/<svg/.test(h));
  ok('offers Retry and Reload', />Retry</.test(h) && />↺ Reload</.test(h));
  const chunk = html(crashed({ variant: 'app' }, new TypeError('Failed to fetch dynamically imported module: /assets/x.js')));
  ok('a failed chunk is called out, not blamed on a JS error', chunk.includes('The app could not be loaded') && !chunk.includes('The app crashed'));
}

console.log('── variant="panel" (the default) ───────────────────────────');
{
  const h = html(crashed({}));
  ok('is the default variant', h.includes('Something went wrong in this panel'));
  ok('shows the error message', h.includes('kaboom: cannot read properties of undefined'));
  ok('uses the app classes and icons', /\sclass=/.test(h) && /<svg/.test(h));
  ok('offers Retry and Reload app', />Retry</.test(h) && />Reload app</.test(h));
  ok('explicit panel equals the default', html(crashed({ variant: 'panel' })) === h);
  const chunk = html(crashed({}, new TypeError('Importing a module script failed.')));
  ok('a failed lazy chunk gets its own wording', chunk.includes('This panel could not be loaded') && !chunk.includes('Something went wrong'));
}

console.log('── shared state machine ────────────────────────────────────');
{
  ok('getDerivedStateFromError records the error', ErrorBoundary.getDerivedStateFromError(new Error('x')).error.message === 'x');

  const same = ErrorBoundary.getDerivedStateFromProps({ resetKey: 'faceswap' }, { key: 'faceswap', error: new Error('x') });
  ok('the same resetKey keeps the error', same === null);
  const moved = ErrorBoundary.getDerivedStateFromProps({ resetKey: 'gallery' }, { key: 'faceswap', error: new Error('x') });
  ok('a changed resetKey clears it (navigating away frees the shell)', moved && moved.error === null && moved.key === 'gallery');
  ok('no resetKey (the app root) never auto-clears',
    ErrorBoundary.getDerivedStateFromProps({}, { key: undefined, error: new Error('x') }) === null);

  const b = crashed({ variant: 'app' });
  const before = b.state.nonce;
  b.retry();
  ok('retry clears the error and bumps the nonce (a remount)', b.state.error === null && b.state.nonce === before + 1);

  // onReset: how a failed lazy chunk is made retryable (see lazyPanel.js).
  let resets = 0;
  const onReset = () => { resets += 1; };
  const r = crashed({ onReset });
  r.retry();
  ok('Retry calls onReset', resets === 1);
  ErrorBoundary.getDerivedStateFromProps({ resetKey: 'gallery', onReset }, { key: 'faceswap', error: new Error('x') });
  ok('leaving a failed panel calls onReset (before the children render again)', resets === 2);
  ErrorBoundary.getDerivedStateFromProps({ resetKey: 'gallery', onReset }, { key: 'faceswap', error: null });
  ok('a navigation with no error does not', resets === 2);
  ErrorBoundary.getDerivedStateFromProps({ resetKey: 'faceswap', onReset }, { key: 'faceswap', error: new Error('x') });
  ok('nor does a render with an unchanged key', resets === 2);
  ok('onReset is optional (the app root passes none)', (() => { crashed({ variant: 'app' }).retry(); return true; })());
  // App keys the parent by tab, so leaving a tab UNMOUNTS its boundary: that is
  // the reset that fires in the running app (e2e/error-boundary.spec.js).
  const resetsBefore = resets;
  crashed({ onReset }).componentWillUnmount();
  ok('unmounting a boundary that holds an error calls onReset', resets === resetsBefore + 1);
  const healthy = new ErrorBoundary({ onReset, children: null });
  healthy.componentWillUnmount();
  ok('unmounting a healthy boundary does not', resets === resetsBefore + 1);

  const kid = React.createElement('p', null, 'hello');
  const quiet = new ErrorBoundary({ children: kid });
  const out = quiet.render();
  ok('no error: children pass straight through', renderToStaticMarkup(out) === '<p>hello</p>');
  const remounted = new ErrorBoundary({ children: kid });
  remounted.state = { ...remounted.state, nonce: 3 };
  ok('the nonce is the Fragment key (so retry remounts the subtree)', String(remounted.render().key) === '3');
}

console.log('── there is exactly one boundary ───────────────────────────');
{
  const walk = (d) => readdirSync(d).flatMap((n) => {
    const p = join(d, n);
    return statSync(p).isDirectory() ? walk(p) : /\.(jsx?|tsx?)$/.test(p) ? [p] : [];
  });
  const owners = walk(SRC).filter((p) => /getDerivedStateFromError/.test(readFileSync(p, 'utf8')));
  ok('one implementation in src/', owners.length === 1 && owners[0].endsWith('ErrorBoundary.jsx'),
    owners.map((p) => p.replace(SRC, 'src')).join(', '));
  const main = readFileSync(join(SRC, 'main.jsx'), 'utf8');
  ok('main.jsx uses it as variant="app"', /import ErrorBoundary from '\.\/components\/ErrorBoundary/.test(main)
    && /<ErrorBoundary variant="app">/.test(main));
  ok('main.jsx keeps no private copy', !/extends\s+(React\.)?Component/.test(main));
  const app = readFileSync(join(SRC, 'App.jsx'), 'utf8');
  ok('App.jsx wraps each tab in it with the tab as resetKey and resetFailedPanels as onReset',
    /<ErrorBoundary resetKey=\{tab\} onReset=\{resetFailedPanels\}>/.test(app));
  ok('every tab panel is a lazyPanel, never a bare lazy (a bare one cannot be retried)',
    !/\blazy\(/.test(app) && (app.match(/= lazyPanel\(load\w+\)/g) || []).length === 9);
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? '\nerror-boundary: all checks passed' : `\nerror-boundary: ${failures} check(s) FAILED`);
