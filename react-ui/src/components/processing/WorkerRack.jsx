import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useApi } from '../../useApi';
import { timeoutFor } from '../../api';
import { reportFailure } from '../../failureLog';
import { TONE_CLASS, rackGroups } from './queueModel';

// Distributed worker rack: GOP-aligned slice rendering, one lane per slice.
//
// The backend splits a source at keyframes into numbered slices and joins them
// with a lossless concat (routes_queue._join_group), then checks the joined
// duration and whether the source audio came along. This rack shows exactly that
// state, group by group.
//
// Honest limit: the backend runs ONE job at a time on this machine, so a split's
// slices render one after another. The rack shows the single active lane and the
// queue order. It does not invent parallel lanes, and it does not show GPU
// workers the backend does not report.

const LANE_STYLE = {
  live: 'border-yellow-400/50 bg-yellow-500/10',
  ok: 'border-emerald-400/30 bg-emerald-500/10',
  bad: 'border-red-400/35 bg-red-500/10',
  stop: 'border-orange-400/30 bg-orange-500/10',
  muted: 'border-white/10 bg-white/[0.03]',
  amber: 'border-amber-400/30 bg-amber-500/10',
  sky: 'border-sky-400/30 bg-sky-500/10',
  resume: 'border-violet-400/30 bg-violet-500/10',
};

const JOIN_CHIP = {
  joined: 'text-emerald-300 border-emerald-400/30 bg-emerald-500/10',
  ready: 'text-cyan-200 border-cyan-400/30 bg-cyan-500/10',
  running: 'text-yellow-300 border-yellow-400/30 bg-yellow-500/10',
  waiting: 'text-white/60 border-white/10 bg-white/5',
};

const REMUX_CHIP = {
  confirmed: 'text-emerald-300',
  pending: 'text-amber-300',
  failed: 'text-red-300',
  unknown: 'text-white/50',
};

export default function WorkerRack({ notify, onChanged, running, jobs }) {
  const { getJSON, postJSON } = useApi();
  const [groups, setGroups] = useState(null);     // null = not read yet
  const [busy, setBusy] = useState('');
  const seq = useRef(0);

  // Re-read the split groups whenever the queue changes shape or state. The
  // signature is cheap and keeps this from polling on its own.
  const signature = (jobs || []).map((j) => `${j.id}:${j.state || j.status}`).join('|');

  const read = useCallback(async () => {
    const mine = ++seq.current;
    try {
      const res = await getJSON('/api/queue/segments', { timeout: 15000 });
      if (mine === seq.current) setGroups(res.groups || []);
    } catch (err) {
      reportFailure('Reading the render split groups', err, { toast: false });
    }
  }, [getJSON]);

  useEffect(() => { read(); }, [read, signature, running]);

  const cards = rackGroups(groups || []);
  if (groups === null || cards.length === 0) return null;

  const run = async (key, path, body, okMsg) => {
    setBusy(key);
    try {
      const res = await postJSON(path, body, { timeout: timeoutFor('POST', path) });
      if (okMsg) notify?.(okMsg(res), 'success');
      await read();
      onChanged?.();
    } catch (e) {
      notify?.(e.message || 'Action failed', 'error');
    } finally {
      setBusy('');
    }
  };

  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-3.5 space-y-3" data-testid="worker-rack" aria-label="Distributed worker rack">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <div className="text-mini uppercase tracking-[0.16em] text-accent font-semibold">Worker rack · GOP-aligned slices</div>
          <p className="text-micro text-muted m-0">
            This host renders one slice at a time. Each lane is a keyframe-aligned slice; a split joins only when every slice is finished.
          </p>
        </div>
      </div>

      {cards.map((g) => (
        <article key={g.group} className="rounded-lg border border-white/10 bg-white/[0.02] p-3 space-y-2.5" data-group={g.group}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="text-xs font-bold text-white truncate" title={g.target}>{g.target}</div>
              <div className="text-micro text-white/55 font-mono">{g.completed} / {g.count} slices finished</div>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className={`rounded-full border px-2 py-0.5 text-nano font-bold uppercase tracking-wider ${JOIN_CHIP[g.joinState]}`}>
                {g.joinState === 'joined' ? 'Joined' : g.joinState === 'ready' ? 'Ready to join' : g.joinState === 'running' ? 'Rendering' : 'Waiting'}
              </span>
              {g.canJoin && (
                <button type="button" data-action="join-group" disabled={!!busy}
                  onClick={() => run(`join:${g.group}`, '/api/queue/segments/join', { group: g.group },
                    (res) => `Joined ${g.count} slices into ${res.name || 'one clip'}`)}
                  className="px-2.5 py-1 rounded-lg bg-[var(--accent)]/20 hover:bg-[var(--accent)]/30 border border-[var(--accent)]/40 text-white text-micro font-bold disabled:opacity-40">
                  {busy === `join:${g.group}` ? 'Joining…' : '🎬 Join clips'}
                </button>
              )}
            </div>
          </div>

          {/* One lane per slice, in order. Active lanes pulse; a finished slice can be re-rendered. */}
          <div className="grid gap-1.5 grid-cols-[repeat(auto-fill,minmax(9.5rem,1fr))]" role="list" aria-label="Slices">
            {g.lanes.map((lane) => {
              const tone = lane.badge.tone;
              const cls = LANE_STYLE[tone] || LANE_STYLE.muted;
              const canRerender = ['COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED'].includes(lane.state);
              return (
                <div key={lane.index} role="listitem" data-lane={lane.index} data-state={lane.state}
                     className={`rounded-lg border p-2 ${cls} ${lane.active ? 'shadow-[0_0_14px_rgba(250,204,21,0.15)]' : ''}`}>
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-micro font-bold text-white">Slice {lane.index}</span>
                    <span className={`text-nano font-bold uppercase tracking-wider ${TONE_CLASS[tone]?.badge.split(' ')[0] || 'text-white/60'}`}>
                      {lane.badge.label}
                    </span>
                  </div>
                  <div className="text-nano font-mono text-white/50 truncate">{lane.range ? `frames ${lane.range}` : ''}</div>
                  {lane.pct !== null && lane.active && (
                    <div className="mt-1 h-1 rounded-full bg-black/40 overflow-hidden">
                      <div className="h-full bg-[var(--accent)] transition-[width] duration-500" style={{ width: `${Math.max(2, lane.pct)}%` }} />
                    </div>
                  )}
                  {lane.error && <div className="text-nano text-red-300/90 truncate mt-1" title={lane.error}>{lane.error}</div>}
                  {canRerender && (
                    <button type="button" data-action="rerender-slice" disabled={!!busy || running}
                      onClick={() => run(`slice:${lane.index}`, '/api/queue/segments/rerender', { id: lane.jobId, start: true },
                        () => `Slice ${lane.index} queued to re-render`)}
                      className="mt-1 text-nano font-bold text-white/60 hover:text-white disabled:opacity-40">
                      ↻ Re-render
                    </button>
                  )}
                </div>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 text-micro">
            <span className="text-white/55 truncate" title={g.joinHint}>{g.joinState === 'joined' ? '' : g.joinHint}</span>
            <span className={`font-semibold ${REMUX_CHIP[g.remux.status]}`} data-remux={g.remux.status}>
              Audio remux: {g.remux.text}
            </span>
          </div>
          {g.joinedUrl && (
            <a href={g.joinedUrl} download className="inline-block text-micro font-bold text-accent hover:underline">Download joined clip</a>
          )}
        </article>
      ))}
    </div>
  );
}
