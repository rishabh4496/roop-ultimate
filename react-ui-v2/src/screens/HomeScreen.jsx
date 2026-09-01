import React from 'react';
import { useNotifications } from '../state/appState';
import { Badge, Card, Notice, Progress } from '../components/primitives';

export default function HomeScreen() {
  const { notify } = useNotifications();
  return <div className="v2-screen">
    <section className="v2-hero"><div><Badge tone="accent">Parallel foundation</Badge><h2>A calmer surface for the next generation of Roop.</h2><p>The shell, navigation, tokens, themes, state, loading, errors, and notifications are ready. Processing remains safely outside this foundation.</p></div><div className="v2-hero-orb" aria-hidden="true" /></section>
    <div className="v2-grid v2-grid-three"><Card><span className="v2-stat-label">Architecture</span><strong className="v2-stat-value">Parallel</strong><span className="v2-stat-note">V1 remains untouched</span></Card><Card><span className="v2-stat-label">Themes</span><strong className="v2-stat-value">7</strong><span className="v2-stat-note">One shared token system</span></Card><Card><span className="v2-stat-label">Connected path</span><strong className="v2-stat-value">Create</strong><span className="v2-stat-note">Verified routes only</span></Card></div>
    <div className="v2-grid v2-grid-two"><Card><div className="v2-card-heading"><div><span className="v2-eyebrow">Foundation readiness</span><h3>Core platform</h3></div><Badge tone="success">Ready</Badge></div><Progress value={100} label="Shell and state foundation" /><Progress value={100} label="Theme and token engine" /><Progress value={100} label="Error and notification surfaces" /></Card><Card><span className="v2-eyebrow">Boundary note</span><h3>Creation is deliberately narrow</h3><p className="v2-muted">The Create route uses verified source, target, preview, generation, progress, and output operations. Queue, resume, update, cleanup, and unverified controls remain unavailable.</p><button type="button" className="v2-text-button" onClick={() => notify('V2 notification system is active', 'success')}>Send a test notification →</button></Card></div>
    <Notice title="V1 safety rule">The existing `react-ui/` client and `react-ui-v1-backup/` snapshot are separate from this entry point and were not imported or overwritten.</Notice>
  </div>;
}
