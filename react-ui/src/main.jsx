import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import TermsGate from './components/TermsGate.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'

// The top-level boundary catches uncaught render/lifecycle errors in the entire
// React tree and shows a human-readable fallback instead of a blank page. Without
// it, a single bad `const` ordering or missing import leaves the user staring at
// black glass with no hint of what went wrong. It is the SAME component App uses
// around each tab panel (variant="panel"); the app variant is self-contained
// (inline styles) so it still draws when the CSS or theme is what broke.
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary variant="app">
      {/* NOTICE.md's intended-use terms, once per install, before anything else. */}
      <TermsGate>
        <App />
      </TermsGate>
    </ErrorBoundary>
  </StrictMode>,
)
