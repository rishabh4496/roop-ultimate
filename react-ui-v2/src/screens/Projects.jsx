import React, { useCallback, useEffect, useState } from 'react';
import { FolderClock, Play, Upload, ShieldCheck, Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import { getJSON, postJSON, timeoutFor } from '@engine/api';
import { useStudio } from '../engine/studio';
import { useRun } from '../engine/run';
import { Panel } from '../fx/effects';
import { notify } from '../ui/controls';

// ── Projects: every render's durable record (app/projects/<id>.json) ──
// The engine writes one on EVERY /api/swap and checkpoints it as each output segment is committed,
// so an interrupted run (app closed, crash, shutdown) leaves a record with a safe frame on disk.
//   Check   - POST /validate: may this record still be resumed on this machine?
//   Load    - POST /load: put its media, sources and settings back into the Studio.
//   Resume  - POST /resume: reload it and continue after its last committed segment.
// Whether a project MAY resume is the server's call (project_checkpoint.validate): a record whose
// provider, precision, models, hardware or inputs moved is not the same render, so Resume is offered
// only on records the server calls recoverable, and its refusal reasons are shown as sent.

const RESUMABLE = ['PAUSED', 'INTERRUPTED', 'RECOVERABLE'];
const TONE = {
  PROCESSING: 'var(--accent-2)', PREPARING: 'var(--accent-2)', PAUSED: 'var(--warn)', COMPLETED: 'var(--ok)',
  FAILED: 'var(--danger)', CANCELLED: 'var(--fg-faint)', INTERRUPTED: 'var(--warn)', RECOVERABLE: 'var(--accent-2)', QUEUED: 'var(--fg-faint)',
};
const ago = (t) => {
  if (!t) return '';
  const s = Math.max(0, Math.round(Date.now() / 1000 - t));
  return s < 60 ? `${s}s ago` : s < 3600 ? `${Math.floor(s / 60)}m ago` : s < 86400 ? `${Math.floor(s / 3600)}h ago` : `${Math.floor(s / 86400)}d ago`;
};

export default function Projects() {
  const [list, setList] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState('');
  const [open, setOpen] = useState('');
  const [reasons, setReasons] = useState({});     // id -> server's refusal text
  const [filter, setFilter] = useState('resumable');
  const processing = useRun((r) => r.progress.processing);

  const refresh = useCallback(async () => {
    try { const r = await getJSON('/api/projects', { timeout: 30000 }); setList(Array.isArray(r?.projects) ? r.projects : []); setError(''); }
    catch (e) { setError(e.message); setList([]); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!processing) return undefined;
    const id = setInterval(() => { if (!document.hidden) refresh(); }, 4000);
    return () => clearInterval(id);
  }, [processing, refresh]);

  const act = async (kind, p) => {
    const label = p.name || p.id;
    if (kind === 'resume' && !window.confirm(`Resume “${label}”?\n\n${p.segments > 0
      ? `The ${p.segments} segment(s) already written stay on disk and the render continues after them.`
      : 'Nothing was committed to disk before it stopped, so its range renders from the start.'}\nThe Studio's current source, target and settings are replaced by this project's.`)) return;
    if (kind === 'load' && !window.confirm(`Load “${label}” into the Studio?\n\nIts media, sources and settings replace the current ones.`)) return;
    setBusy(`${kind}:${p.id}`);
    const path = `/api/projects/${encodeURIComponent(p.id)}/${kind}`;
    try {
      await postJSON(path, {}, { timeout: timeoutFor('POST', path) });
      setReasons((r) => ({ ...r, [p.id]: '' }));
      if (kind === 'validate') notify(`“${label}” can be resumed on this machine`);
      if (kind === 'load' || kind === 'resume') {
        // The engine replaced the media, sources and settings: re-read all of them.
        await useStudio.getState().boot().catch(() => {});
        notify(kind === 'load' ? `Loaded “${label}” into the Studio` : `Resumed “${label}”`);
      }
    } catch (e) {
      // A 409 carries the server's own reasons: "cannot resume" without why is not actionable.
      setReasons((r) => ({ ...r, [p.id]: e.message || `${kind} failed` }));
      setOpen(p.id);
      notify(e.message || `${kind} failed`, 'error');
    } finally { setBusy(''); refresh(); }
  };

  if (list === null) return null;
  const resumable = list.filter((p) => p.recoverable && RESUMABLE.includes(p.state));
  const shown = filter === 'resumable' ? resumable : list;
  return (
    <Panel title="Projects" accent={`${list.length}`} data-testid="projects"
      actions={<div className="flex gap-1" role="tablist" aria-label="Which projects">
        {[['resumable', `Resumable (${resumable.length})`], ['all', 'All']].map(([v, l]) => (
          <button key={v} type="button" role="tab" aria-selected={filter === v} className={`chip !h-7 ${filter === v ? 'on' : ''}`} onClick={() => setFilter(v)} data-testid={`projects-filter-${v}`}>{l}</button>
        ))}
      </div>}>
      <p className="mb-2 text-[11.5px] text-[var(--fg-dim)]">Every render keeps a checkpointed record. An interrupted one can be resumed after its last saved segment, or loaded back into the Studio.</p>
      {error && <div className="mb-2 text-[12px] text-[var(--danger)]" role="alert">{error}</div>}
      <div className="max-h-[420px] space-y-1.5 overflow-y-auto pr-1">
        {shown.map((p) => {
          const canResume = p.recoverable && RESUMABLE.includes(p.state);
          const why = reasons[p.id] || (p.validation_errors || []).join('; ') || p.error || '';
          const expanded = open === p.id;
          return (
            <div key={p.id} className="rounded-xl border border-[var(--rule)] bg-black/25 p-2" data-testid="project">
              <div className="flex items-center gap-2">
                <button type="button" className="btn btn-ghost btn-sm btn-icon !h-6 !w-6" aria-label={expanded ? 'Hide details' : 'Show details'} onClick={() => setOpen(expanded ? '' : p.id)}>
                  {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                </button>
                <FolderClock size={14} className="shrink-0 text-[var(--fg-faint)]" />
                <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold text-white" title={p.name || p.id}>{p.name || p.id}</span>
                <span className="badge shrink-0" style={{ color: TONE[p.state] || 'var(--fg-dim)' }} data-testid="project-state">{String(p.state || '').toLowerCase()}</span>
                <span className="hidden shrink-0 font-mono text-[10.5px] text-[var(--fg-faint)] sm:inline">{ago(p.updated_at)}</span>
              </div>
              {expanded && (
                <div className="mt-1.5 space-y-1 pl-8 font-mono text-[10.5px] text-[var(--fg-dim)]" data-testid="project-detail">
                  <div>{p.segments ?? 0} segment(s) saved{p.safe_frame != null ? ` · safe to frame ${p.safe_frame}` : ''}{p.next_frame != null ? ` · continues at ${p.next_frame}` : ''}</div>
                  {why && <div className="text-[var(--warn)]" data-testid="project-why">{why}</div>}
                </div>
              )}
              <div className="mt-1.5 flex flex-wrap gap-1.5 pl-8">
                <button type="button" className="btn btn-ghost btn-sm" disabled={!!busy} onClick={() => act('validate', p)} data-testid="project-validate">
                  {busy === `validate:${p.id}` ? <Loader2 size={12} className="animate-spin" /> : <ShieldCheck size={12} />} Check
                </button>
                <button type="button" className="btn btn-sm" disabled={!!busy || processing} onClick={() => act('load', p)} data-testid="project-load">
                  {busy === `load:${p.id}` ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />} Load
                </button>
                {canResume && (
                  <button type="button" className="btn btn-primary btn-sm" disabled={!!busy || processing} onClick={() => act('resume', p)} data-testid="project-resume">
                    {busy === `resume:${p.id}` ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />} Resume
                  </button>
                )}
              </div>
            </div>
          );
        })}
        {!shown.length && <div className="py-5 text-center text-[12px] text-[var(--fg-faint)]">{filter === 'resumable' ? 'Nothing to resume: no render was interrupted.' : 'No projects yet: one is recorded for every render.'}</div>}
      </div>
    </Panel>
  );
}
