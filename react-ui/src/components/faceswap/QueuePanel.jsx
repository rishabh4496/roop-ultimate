import React, { useState } from 'react';
import { Button, Section } from '../ui';
import { confirmDialog } from '../confirm';
import {
  ACTIVE_STATES, QUEUE_STATE_CLASS, QUEUE_STATE_LABEL, RETRYABLE_STATES,
  TERMINAL_STATES, jobState,
} from './useQueue';
import { Icon } from '../../icons';

// The batch queue view. It renders whatever /api/queue reports and sends every
// change straight back — it keeps no copy of the list, so it cannot drift from
// the batch that is actually running (see useQueue.js).
//
// Rows are reorderable by drag. The drop target is computed from the pointer's
// position over the row rather than from dragenter/dragleave bookkeeping, which
// is notoriously flaky once a row contains child elements: moving over a nested
// button fires leave on the row and the highlight drops out.

function fmtDuration(job) {
  if (!job.started) return null;
  const end = job.finished || Date.now() / 1000;
  const s = Math.max(0, Math.round(end - job.started));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
}

export default function QueuePanel({
  q,                    // the useQueue() handle
  onAddCurrent,
  onLoadJobSettings,    // put a job's settings back into the live params
  canAdd = true,
  notify,
}) {
  const [dragId, setDragId] = useState(null);
  const [dropId, setDropId] = useState(null);

  const { jobs, running, paused, current, pending, finished, retryable } = q;
  const currentJob = jobs.find((j) => j.id === current);

  const move = (fromId, toId, after) => {
    if (!fromId || fromId === toId) return;
    const ids = jobs.map((j) => j.id).filter((id) => id !== fromId);
    const at = ids.indexOf(toId);
    if (at === -1) return;
    ids.splice(after ? at + 1 : at, 0, fromId);
    q.reorder(ids);
  };

  const confirmClear = async () => {
    if (!(await confirmDialog({
      title: 'Clear the queue?',
      message: running
        ? 'Removes every job except the one currently rendering.'
        : `Removes all ${jobs.length} queued jobs.`,
      confirmLabel: 'Clear queue',
      danger: true,
    }))) return;
    q.clear();
  };

  const summary = jobs.length === 0
    ? 'No jobs queued. Set up a swap and click "Add current to queue".'
    : `${jobs.length} job${jobs.length === 1 ? '' : 's'} · ${pending.length} to run · ${finished.length} finished`
      + (retryable.length ? ` · ${retryable.length} needing attention` : '');

  return (
    <Section title="Batch Swapping Queue">
      {/* Live Active Job Render Peak & Progress Monitor */}
      {running && currentJob && (
        <div className="p-4 mb-4 rounded-2xl bg-yellow-500/10 border border-yellow-500/30 text-white space-y-2 animate-fade-in shadow-[0_0_20px_rgba(234,179,8,0.15)]">
          <div className="flex items-center justify-between">
            <span className="font-bold text-xs text-yellow-400 flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-yellow-400 animate-ping" />
              Active Render Job: {currentJob.label || currentJob.target_name}
            </span>
            <span className="text-micro font-mono text-white/70">
              Source: {currentJob.source_name || `Faceset #${(currentJob.source_index || 0) + 1}`}
            </span>
          </div>

          <div className="flex items-center gap-3 text-micro text-white/80">
            <span className="px-2 py-0.5 rounded bg-black/40 border border-white/10 font-mono text-emerald-400 font-bold">
              ⚡ Live Processing Active
            </span>
            <span>Target: <strong className="text-white">{currentJob.target_name}</strong></span>
            {currentJob.payload?.enhancer && (
              <span className="px-2 py-0.5 rounded bg-white/10 text-white/70 font-mono">
                Enhancer: {currentJob.payload.enhancer}
              </span>
            )}
          </div>
        </div>
      )}

      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-3">
        <div className="text-xs text-white/50">
          {summary}
          {jobs.length > 0 && (
            <span className="block mt-0.5 text-white/30">
              The queue lives on the server — it keeps running if you close this tab, and survives a restart.
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant="secondary" onClick={onAddCurrent} disabled={!canAdd}>
            Add current to queue
          </Button>
          {jobs.length > 0 && (
            <>
              {running ? (
                <>
                  <Button size="sm" variant="secondary" onClick={paused ? q.resume : q.pause}>
                    {paused ? '▶ Resume queue' : '⏸ Pause queue'}
                  </Button>
                  <Button size="sm" variant="stop" onClick={q.stop}>⏹ Stop queue</Button>
                </>
              ) : (
                <Button size="sm" variant="primary" onClick={q.start} disabled={pending.length === 0}>
                  ▶ Start queue
                </Button>
              )}
              {retryable.length > 0 && !running && (
                <Button size="sm" variant="secondary" onClick={() => q.retry()}>
                  ↺ Retry {retryable.length} unfinished
                </Button>
              )}
              {finished.length >= 2 && (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={async () => {
                    const finishedIds = finished.map((j) => j.id);
                    const res = await q.join(finishedIds);
                    if (res && res.path) {
                      notify?.(`Successfully joined ${res.segments} segments into ${res.name}`, 'success');
                    }
                  }}
                  title="Concatenate finished segment clips into a single video"
                >
                  🎬 Join {finished.length} Clips
                </Button>
              )}
              <Button size="sm" variant="ghost" className="text-red-400" onClick={confirmClear}>Clear</Button>
            </>
          )}
        </div>
      </div>

      {jobs.length > 0 && (
        <ul className="space-y-1.5 max-h-72 overflow-y-auto pr-1 list-none m-0 p-0">
          {jobs.map((job, idx) => {
            const isCurrent = job.id === current;
            // The backend already resolves PAUSE_REQUESTED/PAUSED onto the
            // current row, so the state it reports is the state to show — the
            // old local `paused ? 'paused'` override existed only because the
            // legacy `status` field could not express either.
            const st = jobState(job);
            const isActive = ACTIVE_STATES.includes(st);
            const editable = !isCurrent && !isActive;
            // Per-job progress, which /api/queue has always reported and this
            // panel never read.
            const frac = Number(job.progress?.fraction);
            const pct = Number.isFinite(frac)
              ? Math.round(Math.max(0, Math.min(1, frac)) * 100) : null;
            const phase = job.progress?.phase;
            const seg = (job.frame_start != null || job.frame_end != null)
              ? `frames ${job.frame_start ?? 1}–${job.frame_end ?? '∞'}`
              : null;
            const dur = fmtDuration(job);
            return (
              <li
                key={job.id}
                draggable={editable && !running}
                onDragStart={() => setDragId(job.id)}
                onDragEnd={() => { setDragId(null); setDropId(null); }}
                onDragOver={(e) => {
                  if (!dragId || dragId === job.id) return;
                  e.preventDefault();
                  setDropId(job.id);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  const box = e.currentTarget.getBoundingClientRect();
                  move(dragId, job.id, e.clientY > box.top + box.height / 2);
                  setDragId(null);
                  setDropId(null);
                }}
                className={`flex items-center justify-between px-3 py-2 rounded-xl text-xs border transition-colors ${
                  QUEUE_STATE_CLASS[st] || 'text-white bg-white/5 border-white/5'
                } ${dropId === job.id ? 'ring-2 ring-[var(--accent)]' : ''} ${
                  dragId === job.id ? 'opacity-40' : ''
                } ${editable && !running ? 'cursor-grab active:cursor-grabbing' : ''}`}
              >
                <div className="flex-1 min-w-0 pr-3">
                  <span className="font-semibold text-white block truncate">
                    {editable && !running && <span aria-hidden className="opacity-30 mr-1.5">⠿</span>}
                    {idx + 1}. {job.label || job.target_name}
                  </span>
                  <span className="opacity-75 text-micro block truncate">
                    {job.source_name || `Source ${job.source_index + 1}`}
                    {' · '}{job.payload?.swap_model || 'inswapper'}
                    {' · '}{job.payload?.enhancer || 'no enhancer'}
                    {seg && <> · {seg}</>}
                    {dur && <> · {dur}</>}
                  </span>
                  {job.error && (
                    <span className="text-micro text-red-300/90 block truncate" title={job.error}>
                      {job.error}
                    </span>
                  )}
                  {isActive && pct !== null && (
                    <div className="mt-1 flex items-center gap-2">
                      <div className="h-1 flex-1 rounded-full bg-black/40 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-300"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-nano font-mono text-white/60 shrink-0">
                        {pct}%{phase ? ` · ${phase}` : ''}
                      </span>
                    </div>
                  )}
                  {st === 'RECOVERABLE' && (
                    <span className="text-micro text-cyan-300/90 block truncate">
                      Has a safe checkpoint — “Start queue” will pick it up where it stopped.
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="font-bold uppercase text-nano tracking-wider px-2 py-0.5 rounded bg-black/30 border border-white/5">
                    {QUEUE_STATE_LABEL[st] || st}
                  </span>
                  {onLoadJobSettings && (
                    <button
                      type="button"
                      onClick={() => { onLoadJobSettings(job); notify?.(`Loaded the settings from "${job.target_name}"`); }}
                      className="text-white/40 hover:text-[var(--accent)] transition-colors"
                      title="Load this job's settings into the editor"
                      aria-label={`Load settings from job ${idx + 1}`}
                    >
                      <Icon.settings size={13} />
                    </button>
                  )}
                  {editable && (
                    <button
                      type="button"
                      onClick={() => q.duplicate(job.id)}
                      className="text-white/40 hover:text-white transition-colors"
                      title="Duplicate this job"
                      aria-label={`Duplicate job ${idx + 1}`}
                    >
                      ⧉
                    </button>
                  )}
                  {editable && TERMINAL_STATES.includes(st) && (
                    <button
                      type="button"
                      onClick={() => q.retry(job.id)}
                      className="text-white/40 hover:text-emerald-300 transition-colors"
                      title="Run this job again"
                      aria-label={`Re-run job ${idx + 1}`}
                    >
                      ↺
                    </button>
                  )}
                  {/* Cancel just this job. /api/queue/remove refuses the
                      running one and /api/queue/stop ends the whole batch, so
                      before this there was no way to drop a single job that
                      had already started without taking the queue down. */}
                  {(isActive || (!isCurrent && !TERMINAL_STATES.includes(st))) && (
                    <button
                      type="button"
                      onClick={async () => {
                        if (isCurrent && !(await confirmDialog({
                          title: 'Cancel the running job?',
                          message: 'It stops at its next safe checkpoint. Frames already written stay on disk, and the rest of the queue keeps going.',
                          confirmLabel: 'Cancel this job',
                          danger: true,
                        }))) return;
                        q.cancel(job.id);
                      }}
                      className="text-white/40 hover:text-orange-300 transition-colors"
                      title={isCurrent ? 'Cancel this job at its next safe checkpoint' : 'Cancel this job'}
                      aria-label={`Cancel job ${idx + 1}`}
                    >
                      ⃠
                    </button>
                  )}
                  {editable && (
                    <button
                      type="button"
                      onClick={() => q.remove(job.id)}
                      className="text-white/40 hover:text-red-400 font-bold transition-colors"
                      title="Remove job"
                      aria-label={`Remove job ${idx + 1}`}
                    >
                      ✕
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </Section>
  );
}
