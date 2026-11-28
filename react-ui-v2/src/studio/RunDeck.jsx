import React from 'react';
import { Play, Pause, Square, ListPlus, Loader2, AlertTriangle, Timer, Gauge, Film } from 'lucide-react';
import { useTelemetryStore, selectRenderProgress } from '@engine/store/telemetryStore';
import { postJSON } from '@engine/api';
import { useStudio } from '../engine/studio';
import { useRun, outputLabel, outputPath } from '../engine/run';
import QualityReport from '../ui/QualityReport';
import { readiness, buildQueueJob } from '../engine/payload';
import { Panel, NumberTicker } from '../fx/effects';
import { notify } from '../ui/controls';
import Preflight from './Preflight';
import { RunEstimate, ClipAdvisor } from './Estimate';

const fmtEta = (ms) => {
  if (!ms || ms < 0) return '—';
  const s = Math.round(ms / 1000);
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m ${String(s % 60).padStart(2, '0')}s`;
};

/** Progress ring: an SVG arc in the accent gradient. */
function Ring({ fraction, size = 84 }) {
  const r = size / 2 - 6;
  const c = 2 * Math.PI * r;
  const f = fraction == null ? 0.25 : Math.max(0, Math.min(1, fraction));
  return (
    <svg width={size} height={size} className={fraction == null ? 'animate-spin' : ''} style={{ animationDuration: '2.4s' }} aria-hidden="true">
      <defs>
        <linearGradient id="ring-grad" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="var(--accent)" /><stop offset="1" stopColor="var(--accent-2)" />
        </linearGradient>
      </defs>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,.08)" strokeWidth="6" />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="url(#ring-grad)" strokeWidth="6" strokeLinecap="round"
        strokeDasharray={`${c * f} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: 'stroke-dasharray .5s var(--ease)', filter: 'drop-shadow(0 0 6px var(--accent-2))' }} />
    </svg>
  );
}

/** Pause / Resume / Stop for the running render - used by the deck and the top bar. */
export function RunControls({ compact = false }) {
  const p = useRun((r) => r.progress);
  const busy = useRun((r) => r.busy);
  const control = useRun((r) => r.control);
  if (!p.processing) return null;
  const paused = p.paused || p.pause_requested;
  const size = compact ? 'btn-sm' : '';
  return (
    <div className="flex items-center gap-1.5" data-testid={compact ? 'run-controls-compact' : 'run-controls'}>
      {paused ? (
        <button type="button" className={`btn ${size} btn-warn`} onClick={() => control('resume')} disabled={!!busy || p.stop_requested} data-testid="run-resume">
          <Play size={14} /> Resume
        </button>
      ) : (
        <button type="button" className={`btn ${size}`} onClick={() => control('pause')} disabled={!!busy || p.stop_requested} data-testid="run-pause">
          <Pause size={14} /> Pause
        </button>
      )}
      <button type="button" className={`btn ${size} btn-danger`} onClick={() => control('stop')} disabled={!!busy || p.stop_requested} data-testid="run-stop">
        <Square size={13} /> {p.stop_requested ? 'Stopping…' : 'Stop'}
      </button>
    </div>
  );
}

export default function RunDeck() {
  const studio = useStudio();
  const p = useRun((r) => r.progress);
  const busy = useRun((r) => r.busy);
  const start = useRun((r) => r.start);
  const fraction = useTelemetryStore((s) => selectRenderProgress(s).fraction);
  const label = useTelemetryStore((s) => selectRenderProgress(s).label);
  const done = useTelemetryStore((s) => selectRenderProgress(s).done);
  const total = useTelemetryStore((s) => selectRenderProgress(s).total);
  const fps = useTelemetryStore((s) => selectRenderProgress(s).fpsNow);
  const etaMs = useTelemetryStore((s) => selectRenderProgress(s).etaMs);
  const why = readiness(studio);

  const addToBatch = async () => {
    if (why) { notify(why, 'warn'); return; }
    const job = buildQueueJob(studio);
    try {
      await postJSON('/api/queue/add', job);
      notify(`Queued “${job.target_name}” — open Batch to run the queue`);
    } catch (e) { notify(e.message, 'error'); }
  };

  if (p.processing) {
    const paused = p.paused || p.pause_requested;
    return (
      <Panel title="Render" accent={p.stop_requested ? 'stopping' : paused ? 'paused' : 'live'} beam={!paused} data-testid="run-deck">
        <div className="flex items-center gap-4">
          <div className="relative grid place-items-center">
            <Ring fraction={fraction} />
            <span className="absolute font-mono text-[15px] font-semibold text-white">
              {fraction == null ? '…' : <NumberTicker value={fraction * 100} suffix="%" />}
            </span>
          </div>
          <div className="min-w-0 flex-1">
            <div className="truncate text-[15px] font-semibold text-white">{paused ? (p.paused ? 'Paused' : 'Pause requested…') : label || 'Rendering'}</div>
            <div className="mt-0.5 truncate text-[12px] text-[var(--fg-dim)]" title={p.desc}>{p.desc || '…'}</div>
            <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-[11px]">
              <span className="flex items-center gap-1 text-[var(--fg-dim)]"><Film size={12} /> {done != null ? <><NumberTicker value={done} />/{total}</> : '—'}</span>
              <span className="flex items-center gap-1 text-[var(--fg-dim)]"><Gauge size={12} /> {fps ? <NumberTicker value={fps} decimals={1} suffix=" fps" /> : '—'}</span>
              <span className="flex items-center gap-1 text-[var(--fg-dim)]"><Timer size={12} /> {fmtEta(etaMs)}</span>
            </div>
          </div>
        </div>
        <div className={`bar mt-3 ${fraction == null ? 'indeterminate' : ''}`}><i style={{ width: `${(fraction || 0) * 100}%` }} /></div>
        <div className="mt-3 flex justify-end"><RunControls /></div>
      </Panel>
    );
  }

  return (
    <Panel title="Render" data-testid="run-deck">
      {why && (
        <div className="mb-3 flex items-start gap-2 rounded-xl border border-[color-mix(in_oklab,var(--warn)_35%,transparent)] bg-[color-mix(in_oklab,var(--warn)_8%,transparent)] p-2.5 text-[12px] text-[var(--warn)]" data-testid="readiness">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" /> {why}
        </div>
      )}
      {!why && <RunEstimate />}
      {!why && <Preflight />}
      {p.error && !why && <div className="mb-3 rounded-xl border border-[var(--rule-strong)] p-2 text-[12px] text-[var(--danger)]">Last render: {p.error}</div>}
      <div className="flex gap-2">
        <button type="button" className="btn btn-primary btn-lg flex-1" onClick={start} disabled={!!busy || !!why} data-testid="run-start">
          {busy === 'start' ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />} Start render
        </button>
        <button type="button" className="btn btn-lg" onClick={addToBatch} disabled={!!why} title="Add this setup to the batch queue" data-testid="run-queue">
          <ListPlus size={16} /> Batch
        </button>
      </div>
      {outputLabel(p.output) && <div className="mt-2 truncate font-mono text-[11px] text-[var(--fg-faint)]" title={outputLabel(p.output)}>last output · {outputLabel(p.output)}</div>}
      {outputPath(p.output) && <div className="mt-2"><QualityReport path={outputPath(p.output)} /></div>}
      <ClipAdvisor />
    </Panel>
  );
}
