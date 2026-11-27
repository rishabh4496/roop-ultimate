import React from 'react';
import { Icon } from '../icons';

// The app's one error boundary. It used to be two: a bare copy in main.jsx (the
// whole app) and this one (a tab panel), which had drifted apart -- different
// logging, no retry at the top level, and a "crashed on startup" message for
// errors that happen long after startup.
//
//   variant="panel" (default)  A tab panel. Catches render errors and the
//                              rejection thrown by a lazy chunk that failed to
//                              load (a dev-server restart, a dropped connection
//                              mid-navigation); without it React 19 unmounts the
//                              whole tree on a thrown error and the app goes blank
//                              white with no way back except a manual reload.
//                              `resetKey` (the active tab id) clears the error
//                              when the user navigates elsewhere, so one broken
//                              panel never traps the whole shell.
//
//   variant="app"              The root of the tree (main.jsx). This is the last
//                              line of defence, so it is deliberately
//                              self-contained: inline styles, no Tailwind, no
//                              theme variables, no icon set. If the CSS bundle,
//                              the theme or the icon library is what broke, the
//                              crash screen must still draw.
//
// Both share the state machine below: clear-on-resetKey during render, and a
// retry that remounts the subtree. Remounting alone does NOT recover a failed lazy
// chunk -- React.lazy memoizes its rejection -- so `onReset` (App passes
// resetFailedPanels, from lazyPanel.js) is called on Retry and when resetKey moves
// off a failed panel; it makes the next render run the import() again.
const CHUNK_FAILURE = /dynamically imported module|Importing a module script failed|Failed to fetch/i;

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null, nonce: 0, key: props.resetKey };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  // Clear during render rather than in componentDidUpdate — navigating away
  // should not cost an extra render pass showing the stale error first.
  static getDerivedStateFromProps(props, state) {
    if (props.resetKey !== state.key) {
      // `onReset` (see lazyPanel) must run BEFORE the children render again, or a
      // panel whose chunk failed re-throws its memoized rejection immediately.
      // It is idempotent, which is what makes calling it from here acceptable.
      if (state.error) props.onReset?.();
      return { key: props.resetKey, error: null };
    }
    return null;
  }

  // App keys the boundary's parent by tab (`<motion.div key={tab}>`), so leaving a
  // tab UNMOUNTS its boundary rather than changing its resetKey; this is the
  // reset that actually fires there. A failed panel is released the moment the
  // user navigates off it, so coming back to it (after the server is back) loads
  // it fresh instead of re-throwing the memoized failure.
  componentWillUnmount() {
    if (this.state.error) this.props.onReset?.();
  }

  componentDidCatch(error, info) {
    // Keep the detail in the console (devtools, and the Pinokio terminal) for
    // debugging; the UI stays calm.
    console.error(`[ui] ${this.props.variant || 'panel'} crashed:`, error, info?.componentStack);
  }

  retry = () => {
    this.props.onReset?.();
    this.setState((s) => ({ error: null, nonce: s.nonce + 1 }));
  };

  render() {
    const { error, nonce } = this.state;
    if (!error) return <React.Fragment key={nonce}>{this.props.children}</React.Fragment>;

    const message = String(error?.message || error);
    const isChunk = CHUNK_FAILURE.test(message);
    return this.props.variant === 'app'
      ? <AppCrash message={message} isChunk={isChunk} onRetry={this.retry} />
      : <PanelCrash message={message} isChunk={isChunk} onRetry={this.retry} />;
  }
}

function PanelCrash({ message, isChunk, onRetry }) {
  return (
    <div role="alert" className="flex flex-col items-center justify-center h-[45vh] gap-4 text-center px-6">
      {isChunk
        ? <Icon.disconnected size={30} className="text-muted" />
        : <Icon.warning size={30} className="text-amber-400/80" />}
      <div className="text-sm font-semibold text-white/80">
        {isChunk ? 'This panel could not be loaded' : 'Something went wrong in this panel'}
      </div>
      <div className="text-xs text-muted max-w-md leading-relaxed selectable">
        {isChunk
          ? 'The UI bundle for this tab failed to download — usually the server restarted. Retry once it is back up.'
          : message}
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onRetry}
          className="px-4 py-2 rounded-xl bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-xs font-semibold border border-white/10"
        >
          Retry
        </button>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="px-4 py-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.09] text-white/70 hover:text-white text-xs font-semibold border border-white/10"
        >
          Reload app
        </button>
      </div>
    </div>
  );
}

// Inline styles only -- see the note on variant="app" above.
const APP_BUTTON = {
  border: 'none', borderRadius: 8, padding: '10px 24px', fontSize: 14,
  cursor: 'pointer', fontWeight: 600, color: '#fff',
};

function AppCrash({ message, isChunk, onRetry }) {
  return (
    <div
      role="alert"
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', minHeight: '100vh', gap: '16px',
        background: '#0a0a0f', color: '#fff', fontFamily: 'monospace',
        padding: '40px', textAlign: 'center',
      }}
    >
      <div aria-hidden="true" style={{ fontSize: 48 }}>⚡</div>
      <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f87171', margin: 0 }}>
        {isChunk ? 'The app could not be loaded' : 'The app crashed'}
      </h1>
      <p style={{ fontSize: 13, color: '#ffffff80', maxWidth: 520, margin: 0, lineHeight: 1.6 }}>
        {isChunk
          ? 'Part of the UI failed to download — usually the server restarted. Retry once it is back up.'
          : 'A JavaScript error stopped the UI. Open the browser developer console (F12) or the Pinokio terminal for the full stack trace.'}
      </p>
      <pre style={{
        background: '#1a1a2e', border: '1px solid #ffffff15', borderRadius: 8,
        padding: '12px 16px', fontSize: 12, color: '#fbbf24',
        maxWidth: 600, overflowX: 'auto', textAlign: 'left', whiteSpace: 'pre-wrap',
      }}>
        {message}
      </pre>
      <div style={{ display: 'flex', gap: 8 }}>
        <button type="button" onClick={onRetry} style={{ ...APP_BUTTON, background: '#6366f1' }}>
          Retry
        </button>
        <button type="button" onClick={() => window.location.reload()} style={{ ...APP_BUTTON, background: '#ffffff1a' }}>
          ↺ Reload
        </button>
      </div>
    </div>
  );
}
