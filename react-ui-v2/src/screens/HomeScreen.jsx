import React from 'react';
import { useNotifications } from '../state/appState';
import { Badge, Card, Notice, Progress } from '../components/primitives';
import { useRouter } from '../router';
import { useOperationsStatus } from '../workflow/useOperationsStatus';

function display(value) {
  return value == null || value === '' ? 'UNKNOWN' : String(value);
}

export default function HomeScreen() {
  const { notify } = useNotifications();
  const { navigate } = useRouter();
  const status = useOperationsStatus(3000);
  const runtime = status.runtime;
  const queue = runtime?.sections?.QUEUE?.values || {};
  const queueCounts = queue.state_counts || {};
  const queueLabel = queue.job_count == null ? 'UNKNOWN' : `${queue.job_count} job(s)`;
  const runtimeLabel = display(runtime?.status?.message || runtime?.status?.code);
  return <div className="v2-screen">
    <section className="v2-hero"><div><Badge tone="accent">React UI 2.0 integration</Badge><h2>A calmer surface for the next generation of Roop.</h2><p>V2 now observes the same backend-owned processing, queue, checkpoint, runtime, and storage state used by the application. Unsupported launcher operations remain clearly bounded.</p><ButtonLink onClick={() => navigate('create')}>Open creation workflow →</ButtonLink></div><div className="v2-hero-orb" aria-hidden="true" /></section>
    <div className="v2-grid v2-grid-three"><Card><span className="v2-stat-label">Runtime</span><strong className="v2-stat-value">{runtimeLabel}</strong><span className="v2-stat-note">Backend-owned status</span></Card><Card><span className="v2-stat-label">Provider</span><strong className="v2-stat-value">{display(runtime?.provider)}</strong><span className="v2-stat-note">Effective runtime value</span></Card><Card><span className="v2-stat-label">Queue</span><strong className="v2-stat-value">{queueLabel}</strong><span className="v2-stat-note">Durable server state</span></Card></div>
    <div className="v2-grid v2-grid-two"><Card><div className="v2-card-heading"><div><span className="v2-eyebrow">Integration status</span><h3>Observed backend surfaces</h3></div><Badge tone={status.error ? 'danger' : runtime ? 'success' : 'neutral'}>{status.error ? 'Partial' : runtime ? 'Connected' : 'Unknown'}</Badge></div><Progress value={runtime ? 100 : 0} label="Processing and runtime state" /><Progress value={Object.keys(queueCounts).length ? 100 : 0} label="Durable queue state" /><Progress value={status.hardware ? 100 : 0} label="Hardware evidence" /></Card><Card><span className="v2-eyebrow">Queue snapshot</span><h3>{queue.current_target || 'No active target'}</h3><p className="v2-muted">Current state: {display(queue.current_state)} · running: {queue.running == null ? 'UNKNOWN' : String(queue.running)}</p><button type="button" className="v2-text-button" onClick={() => notify('V2 notification system is active', 'success')}>Send a test notification →</button></Card></div>
    {status.error && <Notice tone="danger" title="Some backend evidence is unavailable">{status.error}. V2 keeps unknown values explicit and does not simulate processing results.</Notice>}
    <Notice title="V1 safety rule">The existing `react-ui/` client and `react-ui-v1-backup/` snapshot remain separate, available, and untouched by this V2 integration.</Notice>
  </div>;
}

function ButtonLink({ onClick, children }) {
  return <button type="button" className="v2-text-button" onClick={onClick}>{children}</button>;
}
