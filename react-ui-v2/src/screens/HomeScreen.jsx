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

  return (
    <div className="v2-screen">
      <section className="v2-hero">
        <div>
          <Badge tone="accent">React UI 2.0 Workstation</Badge>
          <h2>Roop Ultimate Professional AI Media Studio</h2>
          <p>
            Real-time multi-person face swapping, 3D head pose tracking, and high-fidelity face restoration
            powered by TensorRT and ONNX Runtime.
          </p>
          <div className="mt-3">
            <ButtonLink onClick={() => navigate('create')}>Launch Workstation Studio →</ButtonLink>
          </div>
        </div>
      </section>

      <div className="v2-grid v2-grid-three">
        <Card>
          <span className="v2-stat-label">Runtime Engine</span>
          <strong className="v2-stat-value">{runtimeLabel}</strong>
          <span className="v2-stat-note">Backend-owned session status</span>
        </Card>
        <Card>
          <span className="v2-stat-label">Execution Provider</span>
          <strong className="v2-stat-value">{display(runtime?.provider)}</strong>
          <span className="v2-stat-note">Effective accelerator</span>
        </Card>
        <Card>
          <span className="v2-stat-label">Processing Queue</span>
          <strong className="v2-stat-value">{queueLabel}</strong>
          <span className="v2-stat-note">Durable job state</span>
        </Card>
      </div>

      <div className="v2-grid v2-grid-two">
        <Card>
          <div className="v2-card-heading">
            <div>
              <span className="v2-eyebrow">Integration telemetry</span>
              <h3>Observed Backend Services</h3>
            </div>
            <Badge tone={status.error ? 'danger' : runtime ? 'success' : 'neutral'}>
              {status.error ? 'Partial' : runtime ? 'Connected' : 'Unknown'}
            </Badge>
          </div>
          <Progress value={runtime ? 100 : 0} label="Processing and runtime state" />
          <Progress value={Object.keys(queueCounts).length ? 100 : 0} label="Durable queue state" />
          <Progress value={status.hardware ? 100 : 0} label="Hardware evidence" />
        </Card>
        <Card>
          <span className="v2-eyebrow">Queue Snapshot</span>
          <h3>{queue.current_target || 'No active target in queue'}</h3>
          <p className="v2-muted">
            Current state: {display(queue.current_state)} · Running: {queue.running == null ? 'UNKNOWN' : String(queue.running)}
          </p>
          <div className="mt-3">
            <button
              type="button"
              className="v2-text-button"
              onClick={() => notify('V2 notification system is active', 'success')}
            >
              Send a test notification →
            </button>
          </div>
        </Card>
      </div>

      {status.error && (
        <Notice tone="danger" title="Some backend evidence is unavailable">
          {status.error}. V2 keeps unknown values explicit and does not simulate processing results.
        </Notice>
      )}
      <Notice title="V1 Safety Invariant">
        The existing `react-ui/` client and `react-ui-v1-backup/` snapshot remain 100% separate, operational, and untouched.
      </Notice>
    </div>
  );
}

function ButtonLink({ onClick, children }) {
  return <button type="button" className="v2-text-button" onClick={onClick}>{children}</button>;
}
