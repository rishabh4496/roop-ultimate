import React, { useState } from 'react';
import { Play, Pause, Square, Trash2, RotateCcw, Copy, Ban, ListPlus, GripVertical, Eraser, Layers, CheckCircle2, XCircle, Clock, Loader2 } from 'lucide-react';
import useQueue, { jobState, QUEUE_STATE_LABEL, ACTIVE_STATES } from '@engine/components/faceswap/useQueue';
import { useStudio } from '../engine/studio';
import { buildQueueJob, readiness } from '../engine/payload';
import { Panel, NumberTicker, Meteors } from '../fx/effects';
import { notify } from '../ui/controls';
import Segments from './Segments';

// ── Batch: the job queue ─────────────────────────────────────────────────────
// The queue's state and actions are React UI 1.0's useQueue hook (it owns the polling and the
// engine's ten-state job vocabulary); everything you see is 2.0's. Drag a card to reorder.

const TONE = {
  QUEUED: 'badge-dim', RECOVERABLE: 'badge-warn', PREPARING: 'badge-accent', PROCESSING: 'badge-accent',
  PAUSE_REQUESTED: 'badge-warn', PAUSED: 'badge-warn', COMPLETED: 'badge-ok', FAILED: 'badge-danger',
  CANCELLED: 'badge-dim', INTERRUPTED: 'badge-warn',
};
const ICON = { COMPLETED: CheckCircle2, FAILED: XCircle, QUEUED: Clock, PROCESSING: Loader2, PREPARING: Loader2 };

function progressOf(job) {
  const p = job.progress;
  if (p == null) return null;
  if (typeof p === 'number') return p > 1 ? p / 100 : p;
  if (typeof p === 'object') {
    const d = Number(p.done ?? p.current ?? p.frames_done); const t = Number(p.total ?? p.frames_total);
    if (Number.isFinite(d) && Number.isFinite(t) && t > 0) return d / t;
    if (Number.isFinite(Number(p.fraction))) return Number(p.fraction);
  }
  return null;
}

export default function Batch() {
  const q = useQueue({ notify });
  const studio = useStudio();
  const [drag, setDrag] = useState(null);
  const jobs = q.jobs || [];
  const why = readiness(studio);

  const counts = jobs.reduce((a, j) => { const s = jobState(j); a[s] = (a[s] || 0) + 1; return a; }, {});
  const reorderTo = (fromId, toId) => {
    if (!fromId || fromId === toId) return;
    const ids = jobs.map((j) => j.id);
    const from = ids.indexOf(fromId);
    const to = ids.indexOf(toId);
    ids.splice(to, 0, ids.splice(from, 1)[0]);
    q.reorder(ids);
  };

  return (
    <div className="mx-auto grid max-w-[1400px] gap-3 lg:grid-cols-[minmax(0,1fr)_320px]" data-testid="batch">
      <Panel title="Queue" accent={`${jobs.length} job${jobs.length === 1 ? '' : 's'}`} beam={q.running && !q.paused}
        actions={<>
          {!q.running && <button type="button" className="btn btn-primary btn-sm" onClick={() => q.start()} disabled={!q.pending?.length || q.busy} data-testid="queue-start"><Play size={14} /> Run queue</button>}
          {q.running && !q.paused && <button type="button" className="btn btn-sm" onClick={() => q.pause()} data-testid="queue-pause"><Pause size={14} /> Pause</button>}
          {q.running && q.paused && <button type="button" className="btn btn-warn btn-sm" onClick={() => q.resume()} data-testid="queue-resume"><Play size={14} /> Resume</button>}
          {q.running && <button type="button" className="btn btn-danger btn-sm" onClick={() => q.stop()} data-testid="queue-stop"><Square size={13} /> Stop</button>}
          {q.retryable?.length > 0 && <button type="button" className="btn btn-ghost btn-sm" onClick={() => q.retry()}><RotateCcw size={14} /> Retry failed</button>}
          {jobs.length > 0 && !q.running && <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Clear the queue" title="Clear the queue" onClick={() => { if (window.confirm(`Remove all ${jobs.length} jobs from the queue?`)) q.clear(); }}><Eraser size={14} /></button>}
        </>}>
        {jobs.length === 0 ? (
          <div className="relative flex min-h-[260px] flex-col items-center justify-center gap-2 overflow-hidden rounded-xl text-center">
            <Meteors count={10} />
            <Layers size={30} strokeWidth={1.5} className="relative text-[var(--accent-2)]" />
            <div className="relative text-[15px] font-semibold">The queue is empty</div>
            <div className="relative max-w-sm text-[12.5px] text-[var(--fg-dim)]">Set up a swap in the Studio and press <b>Batch</b>, or add it from here. Each job freezes its settings and people when it is queued.</div>
          </div>
        ) : (
          <ol className="space-y-2">
            {jobs.map((job, i) => {
              const st = jobState(job);
              const frac = progressOf(job);
              const active = ACTIVE_STATES.includes(st);
              const Ico = ICON[st] || Clock;
              return (
                <li key={job.id} draggable={!active} onDragStart={() => setDrag(job.id)} onDragOver={(e) => e.preventDefault()}
                  onDrop={() => { reorderTo(drag, job.id); setDrag(null); }}
                  className={`tile p-3 ${active ? 'beam selected' : ''} ${drag === job.id ? 'opacity-50' : ''}`} data-testid="queue-job" data-state={st}>
                  <div className="flex items-center gap-3">
                    {!active && <GripVertical size={16} className="shrink-0 cursor-grab text-[var(--fg-faint)]" aria-hidden="true" />}
                    <span className="w-6 shrink-0 text-center font-mono text-[11px] text-[var(--fg-faint)]">{i + 1}</span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="truncate text-[13.5px] font-semibold text-white" title={job.target_name}>{job.label || job.target_name || 'job'}</span>
                        <span className={`badge ${TONE[st] || 'badge-dim'}`}><Ico size={11} className={st === 'PROCESSING' || st === 'PREPARING' ? 'animate-spin' : ''} />{QUEUE_STATE_LABEL[st] || st.toLowerCase()}</span>
                      </div>
                      <div className="mt-0.5 truncate font-mono text-[11px] text-[var(--fg-dim)]">
                        {job.source_name || 'source'} → {job.target_name}
                        {job.frame_start ? ` · frames ${job.frame_start}–${job.frame_end || 'end'}` : ''}
                      </div>
                      {job.error && <div className="mt-1 truncate text-[11.5px] text-[var(--danger)]" title={job.error}>{job.error}</div>}
                      {job.outputs?.length > 0 && <div className="mt-1 truncate font-mono text-[11px] text-[var(--ok)]" title={job.outputs.join('\n')}>→ {job.outputs[job.outputs.length - 1]}</div>}
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      {active && <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Cancel this job" title="Cancel this job (the rest continue)" onClick={() => q.cancel(job.id)}><Ban size={14} /></button>}
                      {!active && <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Duplicate" title="Duplicate" onClick={() => q.duplicate(job.id)}><Copy size={14} /></button>}
                      {['FAILED', 'CANCELLED', 'INTERRUPTED'].includes(st) && <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Retry" title="Retry" onClick={() => q.retry(job.id)}><RotateCcw size={14} /></button>}
                      {!active && <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Remove" title="Remove" onClick={() => q.remove(job.id)}><Trash2 size={14} /></button>}
                    </div>
                  </div>
                  {(active || frac != null) && <div className={`bar mt-2 ${active && frac == null ? 'indeterminate' : ''}`}><i style={{ width: `${(frac || 0) * 100}%` }} /></div>}
                </li>
              );
            })}
          </ol>
        )}
      </Panel>

      <div className="flex flex-col gap-3">
        <Panel title="Add">
          <div className="text-[12.5px] text-[var(--fg-dim)]">
            Queues what the Studio is set up to do now: <b className="text-white">{studio.sources[studio.selectedSource] ? (studio.sourcesInfo[studio.selectedSource]?.name || `Face ${studio.selectedSource + 1}`) : 'no source'}</b> → <b className="text-white">{studio.targets[studio.selectedTarget]?.name || 'no target'}</b>
          </div>
          {why && <div className="mt-2 text-[12px] text-[var(--warn)]">{why}</div>}
          <button type="button" className="btn mt-3 w-full" disabled={!!why} data-testid="batch-add-current"
            onClick={async () => { const job = buildQueueJob(studio); if (await q.add(job)) notify(`Queued “${job.target_name}”`); }}>
            <ListPlus size={15} /> Add current setup
          </button>
          <button type="button" className="btn btn-ghost mt-2 w-full" disabled={!studio.sources.length || studio.targets.length < 2}
            title="One job per target in the media bin, each with the selected source and the automatic person choice"
            onClick={async () => {
              const list = studio.targets.map((t, i) => buildQueueJob({ ...studio, selectedTarget: i, targetMediaId: t.media_id || t.id, targetPersonIds: [], targetGroups: [], faceMapping: {}, settings: { ...studio.settings, face_detection_mode: studio.settings?.face_detection_mode === 'Selected face' ? 'First found' : studio.settings?.face_detection_mode } }));
              if (await q.addMany(list)) notify(`Queued ${list.length} targets`);
            }}>
            <Layers size={15} /> Queue every target ({studio.targets.length})
          </button>
        </Panel>
        <Panel title="Totals">
          <div className="grid grid-cols-2 gap-2">
            {[['Queued', (counts.QUEUED || 0) + (counts.RECOVERABLE || 0), 'var(--fg)'], ['Running', ACTIVE_STATES.reduce((a, s) => a + (counts[s] || 0), 0), 'var(--accent-2)'],
              ['Done', counts.COMPLETED || 0, 'var(--ok)'], ['Failed', (counts.FAILED || 0) + (counts.INTERRUPTED || 0), 'var(--danger)']].map(([k, v, c]) => (
              <div key={k} className="rounded-xl border border-[var(--rule)] bg-black/40 p-2.5">
                <div className="text-[11px] font-medium text-[var(--fg-faint)]">{k}</div>
                <div className="text-2xl font-semibold" style={{ color: c }}><NumberTicker value={v} /></div>
              </div>
            ))}
          </div>
          <div className="mt-2 text-[11.5px] text-[var(--fg-faint)]">{q.running ? (q.paused ? 'Queue paused.' : 'Queue running.') : 'Queue idle.'} Queue pause holds the batch between jobs; the render's own Pause (top bar) holds the current frame.</div>
        </Panel>
        <Segments running={!!q.running} />
      </div>
    </div>
  );
}
