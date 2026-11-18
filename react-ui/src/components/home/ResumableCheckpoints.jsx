import React, { useCallback, useEffect, useState } from 'react';
import { timeoutFor } from '../../api';
import { useApi } from '../../useApi';
import { Section } from '../ui';
import { confirmDialog } from '../confirm';
import { Icon } from '../../icons';

// ── Resumable checkpoints: the launchpad carousel ───────────────────────────
//
// The latest project checkpoints a run can actually be picked up from, laid out
// as a horizontal carousel (scroll-snap, so a card lands whole). This is a SHORT
// VIEW over /api/projects; the full list with relink lives in History. The
// decision about what may be resumed is the backend's (`project_checkpoint.validate`),
// surfaced as `recoverable`. A record the backend has refused is never offered a
// Resume button, because resuming a checkpoint whose inputs moved would splice two
// different pipelines into one file.
//
// Two actions per card:
//   Resume            continue the render from its safe frame (the run restarts)
//   Open exact state  load the saved settings and inputs into the workstation, no render
//
// Nothing renders when there is nothing to resume: an empty strip on a screen
// people open to start work is noise.
const RESUMABLE = ['PAUSED', 'INTERRUPTED', 'RECOVERABLE'];
const SHOWN = 8;

const ago = (t) => {
  if (!t) return null;
  const s = Math.max(0, Math.round(Date.now() / 1000 - t));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};

export default function ResumableCheckpoints({ notify, setTab }) {
  const { getJSON, postJSON } = useApi();
  const [projects, setProjects] = useState([]);
  const [busy, setBusy] = useState('');

  const refresh = useCallback(async () => {
    try {
      const res = await getJSON('/api/projects', { timeout: 8000 });
      const list = Array.isArray(res?.projects) ? res.projects : [];
      setProjects(list
        .filter((p) => p.recoverable && RESUMABLE.includes(String(p.state || '').toUpperCase()))
        .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))
        .slice(0, SHOWN));
    } catch {
      setProjects([]);       // advisory: Face Swap's own panel reports a failure with its reason
    }
  }, [getJSON]);

  useEffect(() => { refresh(); }, [refresh]);

  const resume = async (p) => {
    const ok = await confirmDialog({
      title: `Resume “${p.name || p.id}”?`,
      message: (p.segments > 0
        ? `The ${p.segments} segment(s) already written stay on disk and processing continues after them. `
        : 'Nothing was committed to disk before this project stopped, so its frame range is re-rendered from the start. ')
        + 'The current source, target and settings are replaced by the ones this project was started with.',
      confirmLabel: 'Resume project',
    });
    if (!ok) return;
    setBusy(p.id);
    try {
      const path = `/api/projects/${encodeURIComponent(p.id)}/resume`;
      await postJSON(path, {}, { timeout: timeoutFor('POST', path) });
      notify('Project resumed', 'success');
    } catch (e) {
      notify(e.message || 'Resume failed', 'error');
    } finally {
      setBusy('');
      refresh();
    }
  };

  const openExact = async (p) => {
    setBusy(p.id);
    try {
      const path = `/api/projects/${encodeURIComponent(p.id)}/load`;
      await postJSON(path, {}, { timeout: timeoutFor('POST', path) });
      notify(`“${p.name || p.id}” loaded into the workstation`, 'success');
      setTab?.('faceswap');
    } catch (e) {
      notify(e.message || 'Could not load that project', 'error');
    } finally {
      setBusy('');
      refresh();
    }
  };

  if (!projects.length) return null;

  return (
    <div data-testid="resume-strip">
      <Section title="Resume where you stopped" icon={Icon.history}>
        <ul
          aria-label="Resumable projects"
          className="m-0 flex list-none gap-3 overflow-x-auto p-0 pb-2 snap-x snap-mandatory [scrollbar-width:thin]"
        >
          {projects.map((p) => (
            <li
              key={p.id}
              data-project-id={p.id}
              className="snap-start shrink-0 w-[min(20rem,85vw)] rounded-xl border border-cyan-400/25 bg-cyan-400/[0.05] p-3.5 flex flex-col gap-3"
            >
              <div className="min-w-0">
                <div className="truncate text-compact font-semibold text-white/90" title={p.name || p.id}>{p.name || p.id}</div>
                <div className="truncate text-mini text-muted">
                  frame {typeof p.safe_frame === 'number' ? p.safe_frame.toLocaleString() : '?'}
                  {' · '}{p.segments || 0} kept segment{p.segments === 1 ? '' : 's'}
                  {p.updated_at ? ` · ${ago(p.updated_at)}` : ''}
                </div>
              </div>
              <div className="mt-auto flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => resume(p)}
                  disabled={busy === p.id}
                  className="flex-1 rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-1.5 text-mini font-bold text-cyan-200 transition-colors hover:bg-cyan-400/20 disabled:opacity-50"
                >
                  {busy === p.id ? 'Working…' : 'Resume'}
                </button>
                <button
                  type="button"
                  onClick={() => openExact(p)}
                  disabled={busy === p.id}
                  title="Load the saved settings and inputs into the workstation without starting a render"
                  className="flex-1 rounded-lg border border-white/15 bg-white/5 px-3 py-1.5 text-mini font-bold text-white/85 transition-colors hover:bg-white/10 disabled:opacity-50"
                >
                  Open exact state
                </button>
              </div>
            </li>
          ))}
        </ul>
      </Section>
    </div>
  );
}
