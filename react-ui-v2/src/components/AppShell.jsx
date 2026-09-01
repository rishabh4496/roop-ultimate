import React from 'react';
import { useAppDispatch, useAppState } from '../state/appState';
import { useTheme } from '../theme/ThemeProvider';
import { ROUTES } from '../router';
import { Badge, Button } from './primitives';
import { NotificationCenter } from './NotificationCenter';

export function AppShell({ route, onNavigate, children }) {
  const { navOpen } = useAppState();
  const dispatch = useAppDispatch();
  const { theme } = useTheme();
  const navigate = (id) => { onNavigate(id); dispatch({ type: 'SET_NAV_OPEN', open: false }); };

  return <div className="v2-app-shell">
    <a className="v2-skip-link" href="#v2-main">Skip to content</a>
    <aside className={`v2-sidebar ${navOpen ? 'is-open' : ''}`} aria-label="V2 navigation">
      <div className="v2-brand"><span className="v2-brand-mark">R</span><span><strong>Roop Ultimate</strong><small>React UI 2.0 integration</small></span></div>
      <nav className="v2-nav">{ROUTES.map((item) => <button type="button" className={route.id === item.id ? 'is-active' : ''} key={item.id} onClick={() => navigate(item.id)}><span className="v2-nav-dot" aria-hidden="true" />{item.label}</button>)}</nav>
      <div className="v2-sidebar-foot"><Badge tone="accent">V2 integration</Badge><small>Backend state is authoritative. React UI 1.0 remains available.</small></div>
    </aside>
    {navOpen && <button type="button" className="v2-scrim" aria-label="Close navigation" onClick={() => dispatch({ type: 'SET_NAV_OPEN', open: false })} />}
    <div className="v2-main-column">
      <header className="v2-topbar"><Button className="v2-menu-button" size="sm" onClick={() => dispatch({ type: 'SET_NAV_OPEN', open: !navOpen })} aria-label="Toggle navigation">☰</Button><div><span className="v2-eyebrow">{route.description}</span><h1>{route.label}</h1></div><div className="v2-topbar-meta"><Badge tone="accent">V2</Badge><span className="v2-theme-indicator">{theme}</span></div></header>
      <main id="v2-main" className="v2-content">{children}</main>
    </div>
    <NotificationCenter />
  </div>;
}
