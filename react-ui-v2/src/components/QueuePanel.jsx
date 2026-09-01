import React, { useCallback } from 'react';
import { Badge, Button, Card, Progress } from './primitives';
import { JOB_STATE_LABELS } from '../workflow/useQueue';

const terminal = new Set(['COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED']);

function display(value) {
  return value == null || value === '' ? 'UNKNOWN' : String(value);
}

function stateTone(state) {
  if (state === 'COMPLETED') return 'success';
  if (state === 'FAILED') return 'danger';
  if (state === 'PROCESSING' || state === 'PREPARING' || state === 'PAUSE_REQUESTED') return 'accent';
  return 'neutral';
}

export function QueuePanel({ queue, onCancel, onRetry, onRemove, onReorder, onStart, onPause, onResume, onStop }) {
  const jobs = queue.jobs || [];
  const move = useCallback((index, delta) => {
    const next = jobs.slice();
    const target = index + delta;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target], next[index]];
    onReorder(next.map((job) => job.id));
  }, [jobs, onReorder]);
  const canRetry = jobs.some((job) => ['FAILED', 'CANCELLED', 'INTERRUPTED'].includes(job.state));

  return <Card className="v2-queue-card">
    <div className="v2-card-heading">
      <div><span className="v2-eyebrow">Render queue</span><h3>Jobs and recovery</h3></div>
      <Badge tone={queue.running ? 'accent' : 'neutral'}>{jobs.length} job{jobs.length === 1 ? '' : 's'}</Badge>
    </div>
    <div className="v2-queue-actions">
      {!queue.running && <Button size="sm" onClick={onStart} disabled={!jobs.some((job) => ['QUEUED', 'RECOVERABLE'].includes(job.state))}>Start queue</Button>}
      {queue.running && !queue.paused && <Button size="sm" onClick={onPause}>Pause</Button>}
      {queue.running && queue.paused && <Button size="sm" onClick={onResume}>Resume</Button>}
      {queue.running && <Button variant="danger" size="sm" onClick={onStop}>Stop queue</Button>}
      {canRetry && <Button size="sm" onClick={() => onRetry()}>Retry failed</Button>}
    </div>
    {!jobs.length && <div className="v2-empty-picker">Add a generation to the queue when you want to run it later or in sequence.</div>}
    <div className="v2-queue-list">
      {jobs.map((job, index) => {
        const state = job.state || 'QUEUED';
        const fraction = Number(job.progress?.fraction);
        const percent = Number.isFinite(fraction) ? Math.round(Math.max(0, Math.min(1, fraction)) * 100) : 0;
        const active = ['PREPARING', 'PROCESSING', 'PAUSE_REQUESTED', 'PAUSED'].includes(state);
        return <div className="v2-queue-row" key={job.id}>
          <div className="v2-queue-order"><strong>{job.position || index + 1}</strong><button type="button" onClick={() => move(index, -1)} disabled={index === 0}>↑</button><button type="button" onClick={() => move(index, 1)} disabled={index === jobs.length - 1}>↓</button></div>
          <div className="v2-queue-main"><div className="v2-queue-title"><strong>{job.label || job.target_name || 'Unnamed target'}</strong><Badge tone={stateTone(state)}>{JOB_STATE_LABELS[state] || state}</Badge></div><small>Source: {display(job.source_name || (Number.isInteger(job.source_index) ? `Face ${job.source_index + 1}` : null))} · {display(job.payload?.swap_model)}</small>{(active || state === 'COMPLETED') && <Progress value={percent} label={`${percent}% · ${display(job.progress?.phase)}`} />}{job.error && <div className="v2-queue-error">{job.error}</div>}</div>
          <div className="v2-queue-row-actions">{active && <Button size="sm" variant="danger" onClick={() => onCancel(job.id)}>Cancel</Button>}{['FAILED', 'CANCELLED', 'INTERRUPTED'].includes(state) && <Button size="sm" onClick={() => onRetry(job.id)}>Retry</Button>}{terminal.has(state) && state !== 'COMPLETED' && <Button size="sm" onClick={() => onRemove(job.id)}>Remove</Button>}</div>
        </div>;
      })}
    </div>
  </Card>;
}
