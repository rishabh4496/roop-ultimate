import React, { useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../api';
import { Button, Section } from '../ui';
import { confirmDialog } from '../confirm';

// ── Persistent projects ──────────────────────────────────────────────────────
//
// api.py creates a project record on EVERY /api/swap (see
// `_create_processing_project`, called from the swap route) and updates its
// checkpoint as each output segment is committed. That has always happened; it
// simply had no client. A run that was interrupted — the application closed,
// the machine shut down, a crash — therefore left a durable record with a safe
// frame on disk that nothing in this UI could see or act on.
//
// Everything below is read from `project_checkpoint.summarize`, and every
// decision about whether a project may be resumed belongs to
// `project_checkpoint.validate` on the server. This panel does not decide
// recoverability, and it must never offer Resume on a record the backend has
// refused: a checkpoint whose provider, precision, model set, hardware
// signature or input files have moved is not the same render, and resuming it
// would silently splice two different pipelines into one file.

const STATE_LABEL = {
  QUEUED: 'Queued',
  PREPARING: 'Preparing',
  PROCESSING: 'Processing',
  PAUSED: 'Paused',
  COMPLETED: 'Completed',
  FAILED: 'Failed',
  CANCELLED: 'Cancelled',
  INTERRUPTED: 'Interrupted',
  RECOVERABLE: 'Recoverable',
};

const STATE_CLASS = {
  PROCESSING: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30 animate-pulse',
  PREPARING: 'text-sky-300 bg-sky-500/10 border-sky-500/30',
  PAUSED: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
  COMPLETED: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  FAILED: 'text-red-400 bg-red-500/10 border-red-500/30',
  CANCELLED: 'text-orange-300 bg-orange-500/10 border-orange-500/30',
  INTERRUPTED: 'text-violet-300 bg-violet-500/10 border-violet-500/30',
  RECOVERABLE: 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30',
  QUEUED: 'text-white/60 bg-white/5 border-white/5',
};

// States from which the backend can actually pick a project up again.
const RESUMABLE = ['PAUSED', 'INTERRUPTED', 'RECOVERABLE'];

const fmtAgo = (t) => {
  if (!t) return null;
  const s = Math.max(0, Math.round(Date.now() / 1000 - t));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};

const fmtN = (n) => (typeof n === 'number' ? n.toLocaleString() : null);

export default function ProjectsPanel({ notify, onLoaded, processing = false }) {
  const [projects, setProjects] = useState(null);   // null = not fetched yet
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState('');

  const refresh = useCallback(async () => {
    try {
      const res = await getJSON('/api/projects');
      setProjects(Array.isArray(res?.projects) ? res.projects : []);
      setError('');
    } catch (e) {
      setError(e.message || 'could not read the project list');
      setProjects([]);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Only poll while something is actually running — a project's state cannot
  // change on its own otherwise, and the render is competing for this GPU.
  useEffect(() => {
    if (!processing) return undefined;
    const id = setInterval(() => { if (!document.hidden) refresh(); }, 4000);
    return () => clearInterval(id);
  }, [processing, refresh]);

  const act = async (kind, project) => {
    if (kind === 'resume' && !(await confirmDialog({
      title: `Resume “${project.name || project.id}”?`,
      message: (project.segments > 0
        ? `The ${project.segments} segment(s) already written stay on disk and processing continues after them. `
        : 'Nothing was committed to disk before this project stopped, so its frame range is re-rendered from the start. ')
        + 'The current source, target and settings are replaced by the ones this project was started with.',
      confirmLabel: 'Resume project',
    }))) return;

    setBusy(`${kind}:${project.id}`);
    setError('');
    try {
      const res = await postJSON(`/api/projects/${encodeURIComponent(project.id)}/${kind}`, {});
      if (kind === 'validate') {
        notify?.(`“${project.name || project.id}” can be resumed safely`, 'success');
      } else if (kind === 'load') {
        notify?.('Project inputs and settings loaded into the workspace', 'success');
        await onLoaded?.(res);
      } else {
        notify?.('Project resumed', 'success');
      }
    } catch (e) {
      // The 409 body carries the server's own reasons; showing them is the
      // whole point — "cannot resume" without the reason is not actionable.
      setError(e.message || `${kind} failed`);
      setExpanded(project.id);
      notify?.(e.message || `${kind} failed`, 'error');
    } finally {
      setBusy('');
      refresh();
    }
  };

  // Nothing has ever been rendered on this install: say nothing rather than
  // showing an empty box on a screen that is already dense.
  if (projects !== null && projects.length === 0 && !error) return null;

  const resumableCount = (projects || []).filter(
    (p) => p.recoverable && RESUMABLE.includes(p.state)).length;

  return (
    <Section title="Persistent Projects">
      <div className="text-xs text-white/50 mb-3">
        {projects === null ? 'Reading saved projects…' : (
          <>
            Every render writes a project as it goes: the exact sources, target, settings, provider and
            hardware it started with, plus the last frame safely committed to disk. Close the application,
            shut the machine down, come back — a project whose inputs still match can be picked up from
            where it stopped.
            {resumableCount > 0 && (
              <span className="block mt-0.5 text-cyan-300/90">
                {resumableCount} project{resumableCount === 1 ? '' : 's'} can be resumed right now.
              </span>
            )}
          </>
        )}
      </div>

      {error && (
        <div className="mb-3 px-3 py-2 rounded-xl text-micro bg-red-500/10 border border-red-500/30 text-red-300">
          {error}
        </div>
      )}

      <ul className="space-y-1.5 max-h-72 overflow-y-auto pr-1 list-none m-0 p-0">
        {(projects || []).map((p) => {
          const st = String(p.state || 'QUEUED').toUpperCase();
          const blocked = !p.recoverable;
          const canResume = !blocked && RESUMABLE.includes(st);
          const rowBusy = busy.endsWith(`:${p.id}`);
          const open = expanded === p.id;
          return (
            <li
              key={p.id}
              className={`px-3 py-2 rounded-xl text-xs border transition-colors ${
                STATE_CLASS[st] || 'text-white bg-white/5 border-white/5'}`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <span className="font-semibold text-white block truncate">{p.name || p.id}</span>
                  <span className="opacity-75 text-micro block truncate">
                    reached frame {fmtN(p.safe_frame) ?? 'unknown'}
                    {' · '}{p.segments || 0} committed segment{p.segments === 1 ? '' : 's'}
                    {p.updated_at && <> · {fmtAgo(p.updated_at)}</>}
                  </span>
                  {/* The distinction the user has to be told, because the two
                      cases look identical otherwise. `safe_frame` is only the
                      furthest frame the previous run REACHED; the frames that
                      actually survive are the ones inside committed segments.
                      With none, there is no partial file to continue, so a
                      resume re-renders the range from its start. Saying
                      "resumes from frame 900" there would be a straight lie. */}
                  {canResume && (
                    <span className="text-micro text-cyan-300/80 block">
                      {p.segments > 0
                        ? `Continues after ${p.segments} committed segment${p.segments === 1 ? '' : 's'} — those frames are kept.`
                        : 'Nothing was committed to disk before this stopped, so resuming re-renders this range from the start.'}
                    </span>
                  )}
                  {/* `error` is the reason recorded when the record was last
                      moved to RECOVERABLE or FAILED. It is history, not a live
                      verdict — once validation passes it would sit beside a
                      Resume button contradicting it, so it is shown only while
                      it is still true of the project. */}
                  {p.error && (blocked || st === 'FAILED') && (
                    <span className="text-micro text-red-300/90 block truncate" title={p.error}>
                      {p.error}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="font-bold uppercase text-nano tracking-wider px-2 py-0.5 rounded bg-black/30 border border-white/5">
                    {blocked ? 'Cannot resume' : (STATE_LABEL[st] || st)}
                  </span>
                  <Button size="sm" variant="ghost" onClick={() => act('validate', p)} disabled={rowBusy}>
                    Check
                  </Button>
                  {!blocked && (
                    <Button size="sm" variant="ghost" onClick={() => act('load', p)} disabled={rowBusy || processing}
                      title={processing ? 'A render is in progress' : 'Put this project’s inputs and settings back into the workspace'}>
                      Load
                    </Button>
                  )}
                  {canResume && (
                    <Button size="sm" variant="primary" onClick={() => act('resume', p)} disabled={rowBusy}>
                      Resume
                    </Button>
                  )}
                </div>
              </div>

              {blocked && (
                <div className="mt-1.5">
                  <button
                    type="button"
                    onClick={() => setExpanded(open ? '' : p.id)}
                    className="text-micro text-white/50 hover:text-white underline decoration-dotted"
                  >
                    {open ? 'Hide' : `Why not? (${p.validation_errors.length})`}
                  </button>
                  {open && (
                    <ul className="mt-1 ml-3 list-disc text-micro text-white/60 space-y-0.5">
                      {p.validation_errors.map((r, i) => <li key={i}>{r}</li>)}
                    </ul>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </Section>
  );
}
