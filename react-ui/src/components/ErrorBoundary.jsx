import React from 'react';
import { Icon } from '../icons';

// Catches render/lifecycle errors inside a tab panel — including the dynamic
// `import()` rejection thrown by a lazy chunk that failed to load (a dev-server
// restart or a dropped connection mid-navigation). Without this, React 19
// unmounts the whole tree on a thrown error and the app goes blank white with
// no way back except a manual reload.
//
// `resetKey` (the active tab id) clears the error automatically when the user
// navigates elsewhere, so a single broken panel never traps the whole shell.
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
    if (props.resetKey !== state.key) return { key: props.resetKey, error: null };
    return null;
  }

  componentDidCatch(error, info) {
    // Keep the detail in the console for debugging; the UI stays calm.
    console.error('[ui] panel crashed:', error, info?.componentStack);
  }

  retry = () => {
    // Bumping the nonce remounts the subtree from scratch. For a failed lazy
    // chunk this re-runs the import(), which succeeds once the server is back.
    this.setState((s) => ({ error: null, nonce: s.nonce + 1 }));
  };

  render() {
    const { error, nonce } = this.state;
    if (!error) return <React.Fragment key={nonce}>{this.props.children}</React.Fragment>;

    const isChunk = /dynamically imported module|Importing a module script failed|Failed to fetch/i.test(
      String(error?.message || ''),
    );
    return (
      <div role="alert" className="flex flex-col items-center justify-center h-[45vh] gap-4 text-center px-6">
        {isChunk
          ? <Icon.disconnected size={30} className="text-white/40" />
          : <Icon.warning size={30} className="text-amber-400/80" />}
        <div className="text-sm font-semibold text-white/80">
          {isChunk ? 'This panel could not be loaded' : 'Something went wrong in this panel'}
        </div>
        <div className="text-xs text-white/40 max-w-md leading-relaxed selectable">
          {isChunk
            ? 'The UI bundle for this tab failed to download — usually the server restarted. Retry once it is back up.'
            : String(error?.message || error)}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={this.retry}
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
}
