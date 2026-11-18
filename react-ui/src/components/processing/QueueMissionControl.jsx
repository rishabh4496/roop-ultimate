import React, { useMemo, useState } from 'react';
import { Button } from '../ui';
import { confirmDialog } from '../confirm';
import { useApi } from '../../useApi';
import { timeoutFor } from '../../api';
import useQueue from '../faceswap/useQueue';
import { QUEUE_STATE_LABEL } from '../faceswap/useQueue';
import {
  STATE_BADGE, TONE_CLASS, fmtEta, fmtFps, gaugeFraction, gaugeMax, jobCard, queueTotals, reorderIds,
} from './queueModel';
import WorkerRack from './WorkerRack';

// Render Queue Mission Control (#/processing).
//
// The queue itself lives on the server (see useQueue.js); this view reads it and
// sends every change back, so it cannot disagree with the batch that will run.
// It adds what the plain list could not show: one card per job with a live
// speedometer and stage bar, a dense table for many jobs, all ten backend states
// with their own badge, drag and keyboard reordering, and the quick actions.

const FILTERS = ['ALL', ...Object.keys(STATE_BADGE)];

/** The needle-and-arc speedometer. Reads "—" with an empty dial when there is no fps. */
export const Speedometer = React.memo(function Speedometer({ fps, etaS, live }) {
  const max = gaugeMax(fps);
  const frac = gaugeFraction(fps);
  const eta = fmtEta(etaS);
  const value = fmtFps(fps);
  const sweep = frac * 100;
  return (
    <div className="flex items-center gap-3 shrink-0" data-testid="speedometer">
      <svg viewBox="0 0 100 60" className="h-[52px] w-[88px]" role="img"
           aria-label={value ? `${value} frames per second${eta ? `, ${eta} remaining` : ''}` : 'No speed reading yet'}>
        <path d="M 10 50 A 40 40 0 0 1 90 50" pathLength="100" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="7" strokeLinecap="round" />
        <path d="M 10 50 A 40 40 0 0 1 90 50" pathLength="100" fill="none"
              stroke={live ? '#facc15' : 'var(--accent)'} strokeWidth="7" strokeLinecap="round"
              strokeDasharray={`${sweep} 100`}
              style={{ transition: 'stroke-dasharray 500ms ease-out', filter: live ? 'drop-shadow(0 0 4px rgba(250,204,21,0.5))' : undefined }} />
        <text x="50" y="46" textAnchor="middle" className="fill-white font-mono" style={{ fontSize: 15, fontWeight: 700 }}>
          {value ?? '—'}
        </text>
        <text x="50" y="58" textAnchor="middle" className="fill-white/50" style={{ fontSize: 6 }}>
          {max ? `fps · max ${max}` : 'fps'}
        </text>
      </svg>
      <div className="min-w-0">
        <div className="text-nano font-semibold uppercase tracking-[0.14em] text-muted">ETA</div>
        <div className={`font-mono text-sm font-bold tabular-nums ${eta ? 'text-emerald-300' : 'text-muted'}`}>{eta || '—'}</div>
      </div>
    </div>
  );
});

const StateBadge = React.memo(function StateBadge({ card }) {
  const tone = TONE_CLASS[card.badge.tone] || TONE_CLASS.muted;
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-nano font-bold uppercase tracking-wider ${tone.badge}`}
          title={card.badge.hint} data-state={card.state}>
      <span className={`h-1.5 w-1.5 rounded-full bg-current ${card.badge.motion === 'pulse' ? 'animate-pulse' : ''}`} />
      {card.badge.label}
    </span>
  );
});

const StageBar = React.memo(function StageBar({ card }) {
  if (card.pct === null) {
    return <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden"><div className="h-full w-1/3 rounded-full bg-white/10" /></div>;
  }
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-nano font-mono text-white/60">
        <span className="truncate">{card.phase || (card.isActive ? 'Rendering' : '')}</span>
        <span className="tabular-nums shrink-0">
          {card.framesDone != null && card.framesTotal ? `${card.framesDone.toLocaleString()} / ${card.framesTotal.toLocaleString()} · ` : ''}{card.pct}%
        </span>
      </div>
      <div className="relative h-1.5 rounded-full bg-black/40 overflow-hidden"
           role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={card.pct}
           aria-label={`Job ${card.number} progress`}>
        <div className={`h-full rounded-full transition-[width] duration-500 ${card.live ? 'bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)]' : 'bg-[var(--accent)]/70'}`}
             style={{ width: `${Math.max(card.pct, card.isActive ? 2 : 0)}%` }} />
      </div>
    </div>
  );
});

export default function QueueMissionControl({ notify, setTab }) {
  const q = useQueue({ notify });
  const { jobs, current, running } = q;
  const { postJSON } = useApi();
  const [view, setView] = useState('cards');        // 'cards' | 'table'
  const [filter, setFilter] = useState('ALL');
  const [dragId, setDragId] = useState(null);
  const [dropId, setDropId] = useState(null);
  const [focusId, setFocusId] = useState(null);

  const cards = useMemo(() => jobs.map((job, i) => jobCard(job, i, {
    currentId: current, queueRunning: running,
  })), [jobs, current, running]);
  const totals = useMemo(() => queueTotals(cards), [cards]);
  const visible = filter === 'ALL' ? cards : cards.filter((c) => c.state === filter);
  const perState = useMemo(() => {
    const m = {};
    for (const c of cards) m[c.state] = (m[c.state] || 0) + 1;
    return m;
  }, [cards]);

  const move = (fromId, toId, after) => {
    const ids = reorderIds(q.jobs, fromId, toId, after);
    if (ids) q.reorder(ids);
  };
  // Keyboard reordering: Alt+ArrowUp / Alt+ArrowDown on a focused handle.
  const onHandleKey = (e, card) => {
    if (!(e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown'))) return;
    e.preventDefault();
    const at = cards.findIndex((c) => c.id === card.id);
    const neighbour = cards[e.key === 'ArrowUp' ? at - 1 : at + 1];
    if (!neighbour) return;
    setFocusId(card.id);
    move(card.id, neighbour.id, e.key === 'ArrowDown');
  };

  const loadCheckpoint = async (card) => {
    try {
      await postJSON(`/api/projects/${encodeURIComponent(card.projectId)}/load`, {}, { timeout: timeoutFor('POST', '/api/projects/load') });
      notify?.(`Checkpoint for “${card.label}” loaded into the studio`, 'success');
      setTab?.('faceswap');
    } catch (e) {
      notify?.(e.message || 'Could not load that checkpoint', 'error');
    }
  };

  const cancel = async (card) => {
    if (card.isCurrent && !(await confirmDialog({
      title: 'Cancel the running job?',
      message: 'It stops at its next safe checkpoint. Frames already written stay on disk, and the rest of the queue keeps going.',
      confirmLabel: 'Cancel this job',
      danger: true,
    }))) return;
    q.cancel(card.id);
  };

  const confirmClear = async () => {
    if (!(await confirmDialog({
      title: 'Clear the queue?',
      message: q.running ? 'Removes every job except the one currently rendering.' : `Removes all ${cards.length} queued jobs.`,
      confirmLabel: 'Clear queue',
      danger: true,
    }))) return;
    q.clear();
  };

  const actionRow = (card) => (
    <div className="flex flex-wrap items-center gap-1.5">
      {card.actions.loadCheckpoint && (
        <button type="button" onClick={() => loadCheckpoint(card)} data-action="load-checkpoint"
          className="px-2 py-1 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-400/25 text-cyan-200 text-micro font-bold"
          title="Load this job's saved checkpoint into the studio">Load checkpoint</button>
      )}
      {card.actions.cancel && (
        <button type="button" onClick={() => cancel(card)} data-action="cancel"
          className="px-2 py-1 rounded-lg bg-orange-500/10 hover:bg-orange-500/20 border border-orange-400/25 text-orange-200 text-micro font-bold"
          title={card.isCurrent ? 'Cancel at the next safe checkpoint' : 'Cancel this job'}>Cancel</button>
      )}
      {card.actions.duplicate && (
        <button type="button" onClick={() => q.duplicate(card.id)} data-action="duplicate"
          className="px-2 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white/80 text-micro font-bold"
          title="Queue a copy of this job">Duplicate</button>
      )}
      {card.actions.retry && (
        <button type="button" onClick={() => q.retry(card.id)} data-action="retry"
          className="px-2 py-1 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-400/25 text-emerald-200 text-micro font-bold"
          title="Run this job again">Retry</button>
      )}
      {card.actions.remove && (
        <button type="button" onClick={() => q.remove(card.id)} data-action="remove"
          className="px-2 py-1 rounded-lg text-red-300/80 hover:text-red-300 text-micro font-bold"
          aria-label={`Remove job ${card.number}`} title="Remove from the queue">Remove</button>
      )}
    </div>
  );

  const dragProps = (card) => ({
    draggable: card.canDrag,
    onDragStart: () => setDragId(card.id),
    onDragEnd: () => { setDragId(null); setDropId(null); },
    onDragOver: (e) => { if (!dragId || dragId === card.id) return; e.preventDefault(); setDropId(card.id); },
    onDrop: (e) => {
      e.preventDefault();
      const box = e.currentTarget.getBoundingClientRect();
      move(dragId, card.id, e.clientY > box.top + box.height / 2);
      setDragId(null); setDropId(null);
    },
  });

  const handle = (card) => (
    card.canDrag ? (
      <button type="button" tabIndex={0} aria-label={`Reorder job ${card.number}. Alt plus arrow keys moves it.`}
        onKeyDown={(e) => onHandleKey(e, card)} onFocus={() => setFocusId(card.id)}
        className={`cursor-grab active:cursor-grabbing select-none text-white/40 hover:text-white px-1 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] ${focusId === card.id ? 'text-white' : ''}`}
        title="Drag to reorder (or focus and press Alt+↑ / Alt+↓)">⠿</button>
    ) : <span className="px-1 text-white/15 select-none" aria-hidden title="Locked while running">⠿</span>
  );

  if (!q.jobs.length) {
    return (
      <section className="rounded-2xl glass-panel p-5 border border-white/5" aria-label="Render queue">
        <div className="text-sm font-bold text-white">Render queue</div>
        <p className="text-xs text-muted mt-1">No jobs queued. Set up a swap in Face Swap or the Batch Matrix and add it here.</p>
      </section>
    );
  }

  return (
    <section className="rounded-2xl glass-panel border border-white/5 shadow-xl p-4 sm:p-5 space-y-4" aria-label="Render queue mission control" data-testid="queue-mission-control">
      <header className="flex flex-col xl:flex-row xl:items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-mini uppercase tracking-[0.18em] text-accent font-semibold">Render queue</div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-white/70">
            <span><strong className="text-white tabular-nums">{totals.total}</strong> jobs</span>
            <span><strong className="text-yellow-300 tabular-nums">{totals.running}</strong> running</span>
            <span><strong className="text-white tabular-nums">{totals.waiting}</strong> waiting</span>
            <span><strong className="text-emerald-300 tabular-nums">{totals.done}</strong> finished</span>
            {totals.attention > 0 && <span><strong className="text-orange-300 tabular-nums">{totals.attention}</strong> need attention</span>}
            {totals.resumable > 0 && <span><strong className="text-violet-300 tabular-nums">{totals.resumable}</strong> resumable</span>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex rounded-xl border border-white/10 bg-black/30 p-0.5" role="group" aria-label="Queue layout">
            {['cards', 'table'].map((v) => (
              <button key={v} type="button" onClick={() => setView(v)} aria-pressed={view === v}
                className={`px-3 py-1 rounded-lg text-micro font-bold uppercase tracking-wider transition-colors ${view === v ? 'bg-[var(--accent)] text-white' : 'text-white/60 hover:text-white'}`}>
                {v}
              </button>
            ))}
          </div>
          {q.running ? (
            <>
              <Button size="sm" variant="secondary" onClick={q.paused ? q.resume : q.pause}>{q.paused ? '▶ Resume' : '⏸ Pause'}</Button>
              <Button size="sm" variant="stop" onClick={q.stop}>⏹ Stop</Button>
            </>
          ) : (
            <Button size="sm" variant="primary" onClick={q.start} disabled={q.pending.length === 0}>▶ Start queue</Button>
          )}
          {totals.attention > 0 && !q.running && (
            <Button size="sm" variant="secondary" onClick={() => q.retry()}>↺ Retry {q.retryable.length}</Button>
          )}
          <Button size="sm" variant="ghost" className="text-red-400" onClick={confirmClear}>Clear</Button>
        </div>
      </header>

      {/* State legend doubles as a filter: all ten backend states, with counts. */}
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by state">
        {FILTERS.map((st) => {
          const count = st === 'ALL' ? cards.length : (perState[st] || 0);
          if (st !== 'ALL' && count === 0) return null;
          const tone = st === 'ALL' ? TONE_CLASS.muted : (TONE_CLASS[STATE_BADGE[st].tone] || TONE_CLASS.muted);
          const on = filter === st;
          return (
            <button key={st} type="button" onClick={() => setFilter(st)} aria-pressed={on} data-filter={st}
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-nano font-bold uppercase tracking-wider transition-all ${tone.badge} ${on ? 'ring-2 ring-white/40' : 'opacity-80 hover:opacity-100'}`}>
              {st === 'ALL' ? 'All' : QUEUE_STATE_LABEL[st]}
              <span className="tabular-nums opacity-80">{count}</span>
            </button>
          );
        })}
      </div>

      {view === 'cards' ? (
        <ul className="grid gap-3 grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 list-none m-0 p-0" data-view="cards">
          {visible.map((card) => {
            const tone = TONE_CLASS[card.badge.tone] || TONE_CLASS.muted;
            return (
              <li key={card.id} data-job-id={card.id} data-state={card.state}
                  {...dragProps(card)}
                  className={`relative rounded-xl border bg-white/[0.025] p-3.5 space-y-3 transition-all ${tone.edge} ${card.live ? 'shadow-[0_0_18px_rgba(250,204,21,0.12)]' : ''} ${dragId === card.id ? 'opacity-40' : ''} ${dropId === card.id ? 'ring-2 ring-[var(--accent)]' : ''}`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-start gap-1.5 min-w-0">
                    {handle(card)}
                    <div className="min-w-0">
                      <div className="text-sm font-bold text-white truncate" title={card.label}>
                        <span className="text-white/65 font-mono mr-1.5">{card.number}.</span>{card.label}
                      </div>
                      <div className="text-micro text-white/55 truncate">
                        {card.source || 'Source —'} · {card.swapModel} · {card.enhancer}{card.frameRange ? ` · ${card.frameRange}` : ''}
                      </div>
                    </div>
                  </div>
                  <StateBadge card={card} />
                </div>

                <div className="flex items-center justify-between gap-3">
                  <Speedometer fps={card.fps} etaS={card.etaS} live={card.live} />
                  <div className="min-w-0 flex-1">
                    <StageBar card={card} />
                  </div>
                </div>

                {card.error && <div className="text-micro text-red-300/90 truncate" title={card.error}>{card.error}</div>}
                {card.state === 'RECOVERABLE' && <div className="text-micro text-cyan-300/90">Has a safe checkpoint: Start queue picks it up where it stopped.</div>}
                {actionRow(card)}
              </li>
            );
          })}
        </ul>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-white/10" data-view="table">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-white/[0.04] text-muted text-micro uppercase tracking-wider">
              <tr>
                <th className="py-2 px-2 w-8" aria-label="Order" />
                <th className="py-2 px-2">#</th>
                <th className="py-2 px-2">Job</th>
                <th className="py-2 px-2">State</th>
                <th className="py-2 px-2 min-w-[11rem]">Progress</th>
                <th className="py-2 px-2 text-right">FPS</th>
                <th className="py-2 px-2 text-right">ETA</th>
                <th className="py-2 px-2">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {visible.map((card) => (
                <tr key={card.id} data-job-id={card.id} data-state={card.state}
                    {...dragProps(card)}
                    className={`hover:bg-white/[0.03] ${dragId === card.id ? 'opacity-40' : ''} ${dropId === card.id ? 'ring-2 ring-inset ring-[var(--accent)]' : ''}`}>
                  <td className="py-2 px-2">{handle(card)}</td>
                  <td className="py-2 px-2 font-mono text-white/50">{card.number}</td>
                  <td className="py-2 px-2 min-w-0 max-w-[22rem]">
                    <div className="font-semibold text-white truncate" title={card.label}>{card.label}</div>
                    <div className="text-micro text-white/50 truncate">{card.source || '—'} · {card.swapModel}</div>
                  </td>
                  <td className="py-2 px-2"><StateBadge card={card} /></td>
                  <td className="py-2 px-2"><StageBar card={card} /></td>
                  <td className="py-2 px-2 text-right font-mono tabular-nums">{fmtFps(card.fps) ?? '—'}</td>
                  <td className="py-2 px-2 text-right font-mono tabular-nums">{fmtEta(card.etaS) ?? '—'}</td>
                  <td className="py-2 px-2">{actionRow(card)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {visible.length === 0 && (
        <p className="text-xs text-muted text-center py-4">No jobs in this state.</p>
      )}

      <WorkerRack notify={notify} onChanged={q.refresh} running={q.running} jobs={q.jobs} />
    </section>
  );
}
