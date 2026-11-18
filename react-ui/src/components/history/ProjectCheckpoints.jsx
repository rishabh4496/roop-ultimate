import React, { useCallback, useEffect, useState } from 'react';
import { useApi } from '../../useApi';
import { timeoutFor } from '../../api';
import { reportFailure } from '../../failureLog';
import { promptDialog } from '../confirm';
import { Section } from '../ui';
import { BADGE_TONE, checkpointBadge } from './checkpointBadge';

// Persistent project checkpoints (GET /api/projects), one card each.
//
// The badge is the backend's verdict on the record (see checkpointBadge.js). A
// green card resumes in one click. An amber card names each missing input and
// offers a relink for it: the new path is checked by the server against the
// recorded SHA-256, so pointing at a different file is refused, not silently used.
// "Resume Exact State in Workstation" loads the saved settings, inputs and runtime
// into the live studio and opens Face Swap, without starting a render.

const fmtWhen = (ts) => (ts ? new Date(Number(ts) * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—');

export default function ProjectCheckpoints({ notify, setTab }) {
  const { getJSON, postJSON } = useApi();
  const [projects, setProjects] = useState(null);
  const [busy, setBusy] = useState('');

  const read = useCallback(async () => {
    try {
      const res = await getJSON('/api/projects', { timeout: 15000 });
      setProjects(Array.isArray(res?.projects) ? res.projects : []);
    } catch (err) {
      reportFailure('Reading the saved checkpoints', err, { toast: false });
      setProjects((p) => p || []);
    }
  }, [getJSON]);

  useEffect(() => { read(); }, [read]);

  const act = async (key, path, body, okMsg) => {
    setBusy(key);
    try {
      await postJSON(path, body, { timeout: timeoutFor('POST', path) });
      if (okMsg) notify?.(okMsg, 'success');
      await read();
      return true;
    } catch (e) {
      notify?.(e.message || 'Action failed', 'error');
      await read();
      return false;
    } finally {
      setBusy('');
    }
  };

  const resumeExact = async (p) => {
    const ok = await act(`load:${p.id}`, `/api/projects/${encodeURIComponent(p.id)}/load`, {},
      `“${p.name || p.id}” loaded into the workstation`);
    if (ok) setTab?.('faceswap');
  };

  const relink = async (p, item) => {
    const answer = await promptDialog({
      title: `Relink ${item.label}`,
      message: `Enter the new full path of the ${item.label} file. The server accepts it only if it is byte-identical to the one this checkpoint recorded.`,
      defaultValue: item.path,
      confirmLabel: 'Relink',
    });
    const path = String(answer || '').trim();
    if (!path) return;
    const body = item.role === 'source' ? { role: 'source', index: item.index, path } : { role: 'target', path };
    await act(`relink:${p.id}:${item.label}`, `/api/projects/${encodeURIComponent(p.id)}/relink`, body,
      `Relinked ${item.label} for “${p.name || p.id}”`);
  };

  if (projects === null) {
    return (
      <Section title="Persistent project checkpoints">
        <p className="text-xs text-muted m-0">Reading checkpoints…</p>
      </Section>
    );
  }
  if (projects.length === 0) return null;

  return (
    <div data-testid="project-checkpoints">
    <Section title="Persistent project checkpoints">
      <p className="-mt-1 text-mini text-muted m-0">
        Every saved render project, with its inputs checked on disk. Resume restores the exact workstation state.
      </p>
      <div className="grid gap-3 grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 mt-3">
        {projects.map((p) => {
          const badge = checkpointBadge(p);
          const tone = BADGE_TONE[badge.tone];
          return (
            <article key={p.id} data-project-id={p.id} data-badge={badge.tone}
                     className={`rounded-xl border bg-white/[0.025] p-3.5 space-y-2.5 ${tone.split(' ').find((c) => c.startsWith('border-'))}`}>
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="text-sm font-bold text-white truncate" title={p.name || p.id}>{p.name || p.id}</div>
                  <div className="text-micro text-white/55 font-mono">
                    {p.state || 'SAVED'} · updated {fmtWhen(p.updated_at)}
                  </div>
                </div>
                <span className={`shrink-0 rounded-full border px-2 py-0.5 text-nano font-bold uppercase tracking-wider ${tone}`}
                      data-testid="checkpoint-badge" title={badge.detail}>
                  {badge.label}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 text-micro font-mono text-white/70">
                <div><div className="text-nano uppercase text-muted">Safe frame</div>{p.safe_frame ?? '—'}</div>
                <div><div className="text-nano uppercase text-muted">Next frame</div>{p.next_frame ?? '—'}</div>
                <div><div className="text-nano uppercase text-muted">Segments</div>{p.segments ?? 0}</div>
              </div>

              {badge.tone === 'amber' && (
                <ul className="m-0 p-0 list-none space-y-1.5" aria-label="Missing inputs">
                  {badge.relink.map((item) => (
                    <li key={item.label} className="flex flex-wrap items-center justify-between gap-2 text-micro">
                      <span className="text-amber-200 min-w-0 truncate" title={item.path || 'no path recorded'}>
                        {item.label}: {item.path || 'no path recorded'}
                      </span>
                      <button type="button" data-action="relink" disabled={!!busy}
                              onClick={() => relink(p, item)}
                              className="px-2 py-1 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 border border-amber-400/35 text-amber-100 font-bold disabled:opacity-40">
                        Relink {item.label}
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              {badge.tone === 'red' && badge.detail && (
                <p className="m-0 text-micro text-red-300/90 break-words" title={badge.reasons.join('\n')}>{badge.detail}</p>
              )}

              <div className="flex items-center justify-between gap-2 pt-1">
                <span className="text-nano text-muted">{badge.detail && badge.tone === 'green' ? badge.detail : ''}</span>
                <button type="button" data-action="resume-exact" disabled={!!busy || badge.tone !== 'green'}
                        onClick={() => resumeExact(p)}
                        title={badge.tone === 'green' ? 'Load the saved state into the workstation' : 'Fix the missing or changed inputs first'}
                        className="px-3 py-1.5 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-micro font-bold disabled:opacity-40 disabled:cursor-not-allowed">
                  {busy === `load:${p.id}` ? 'Loading…' : 'Resume Exact State in Workstation'}
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </Section>
    </div>
  );
}
