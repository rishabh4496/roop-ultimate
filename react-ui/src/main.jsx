import { StrictMode, Component } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Top-level Error Boundary — catches uncaught render/lifecycle errors in the
// entire React tree and shows a human-readable fallback instead of a blank page.
// Without this, a single bad `const` ordering or missing import leaves the user
// staring at black glass with no hint of what went wrong.
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(err) {
    return { error: err };
  }

  componentDidCatch(err, info) {
    // Log to console so devtools / Pinokio terminal show the stack.
    console.error('[ErrorBoundary]', err, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', minHeight: '100vh', gap: '16px',
        background: '#0a0a0f', color: '#fff', fontFamily: 'monospace',
        padding: '40px', textAlign: 'center',
      }}>
        <div style={{ fontSize: 48 }}>⚡</div>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f87171', margin: 0 }}>
          App crashed on startup
        </h1>
        <p style={{ fontSize: 13, color: '#ffffff80', maxWidth: 520, margin: 0, lineHeight: 1.6 }}>
          A JavaScript error prevented the UI from loading. Open the browser
          developer console (F12) or the Pinokio terminal for the full stack
          trace.
        </p>
        <pre style={{
          background: '#1a1a2e', border: '1px solid #ffffff15', borderRadius: 8,
          padding: '12px 16px', fontSize: 11, color: '#fbbf24',
          maxWidth: 600, overflowX: 'auto', textAlign: 'left', whiteSpace: 'pre-wrap',
        }}>
          {error?.message || String(error)}
        </pre>
        <button
          onClick={() => window.location.reload()}
          style={{
            background: '#6366f1', color: '#fff', border: 'none',
            borderRadius: 8, padding: '10px 24px', fontSize: 14,
            cursor: 'pointer', fontWeight: 600,
          }}
        >
          ↺ Reload
        </button>
      </div>
    );
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
