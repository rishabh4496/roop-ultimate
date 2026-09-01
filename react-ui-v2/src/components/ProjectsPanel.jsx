import React, { useCallback, useEffect, useState } from 'react';
import { getProjects, loadProject, resumeProject, validateProject } from '../api';
import { Badge, Button, Card, Notice } from './primitives';

const labels = { PAUSED: 'Paused', INTERRUPTED: 'Interrupted', RECOVERABLE: 'Recoverable', FAILED: 'Failed', COMPLETED: 'Completed', PROCESSING: 'Processing' };

function tone(state) {
  if (state === 'COMPLETED') return 'success';
  if (state === 'FAILED') return 'danger';
  if (state === 'PROCESSING' || state === 'PAUSED') return 'accent';
  return 'neutral';
}

export function ProjectsPanel({ workflow, notify }) {
  const [projects, setProjects] = useState([]);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    try {
      const response = await getProjects();
      setProjects(response?.projects || []);
      setError('');
    } catch (cause) { setError(cause.message); }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 3000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const act = async (kind, project) => {
    setBusy(`${kind}:${project.id}`);
    try {
      if (kind === 'load') {
        await loadProject(project.id);
        await workflow.refreshState();
      } else if (kind === 'resume') {
        await resumeProject(project.id);
      } else {
        await validateProject(project.id);
      }
      await refresh();
      notify?.(kind === 'validate' ? 'Project checkpoint is valid' : kind === 'load' ? 'Project loaded' : 'Project resumed', 'success');
    } catch (cause) {
      setError(cause.message);
      notify?.(cause.message, 'danger');
      await refresh();
    } finally { setBusy(''); }
  };

  return <Card className="v2-projects-card">
    <div className="v2-card-heading"><div><span className="v2-eyebrow">Persistent projects</span><h3>Continue safely later</h3></div><Badge tone="neutral">{projects.length} saved</Badge></div>
    <p className="v2-muted">Projects keep verified input identities, settings, runtime assumptions, and only output segments that reached a safe checkpoint.</p>
    {error && <Notice tone="danger" title="Recoverability check failed">{error}</Notice>}
    {!projects.length && <div className="v2-empty-picker">No saved processing projects yet.</div>}
    <div className="v2-project-list">{projects.map((project) => {
      const invalid = project.validation_errors?.length > 0;
      const resumable = ['PAUSED', 'INTERRUPTED', 'RECOVERABLE'].includes(project.state);
      const rowBusy = busy.endsWith(`:${project.id}`);
      return <div className="v2-project-row" key={project.id}>
        <div className="v2-project-main"><div className="v2-queue-title"><strong>{project.name || project.id}</strong><Badge tone={invalid ? 'danger' : tone(project.state)}>{invalid ? 'Cannot resume' : (labels[project.state] || project.state)}</Badge></div><small>Safe frame: {project.safe_frame ?? 'UNKNOWN'} · {project.segments || 0} committed segment(s)</small>{invalid && <div className="v2-queue-error">{project.validation_errors.join('\n')}</div>}</div>
        <div className="v2-project-actions"><Button size="sm" onClick={() => act('validate', project)} disabled={rowBusy}>Validate</Button>{!invalid && <Button size="sm" onClick={() => act('load', project)} disabled={rowBusy}>Load</Button>}{!invalid && resumable && <Button size="sm" variant="primary" onClick={() => act('resume', project)} disabled={rowBusy}>Resume</Button>}</div>
      </div>;
    })}</div>
  </Card>;
}
