import React from 'react';
import { useTelemetryStore } from '../../store/telemetryStore';
import { useEngineRun } from './useEngineRun';
import {
  STAGE_ORDER, STAGE_LABEL, STAGE_1, STAGE_2, STAGE_3, auxOffloadText, canaryTone, canaryView, flowEngineLabel,
  fmtCount, fmtPct, fmtRate, fmtSecs, headline, opsCells, reportSections, stage3RateCells, stateTone,
} from './reportFormat';

// The 3-stage pipeline's live HUD and its finished report.
//
// `PipelineHud` is the Diagnostics HUD on #/processing: while a render runs it shows the three stages the
// backend reports on /ws/telemetry (state, stage-local progress, elapsed wall clock); once the render ends it
// shows the final report the same socket carries. `PipelineSummaryCard` is the finished report alone, used on
// #/settings (last run) and #/history (per run). All numbers are the backend's; nothing is computed here.

const Dot = ({ state }) => (
  <span className={`inline-block h-1.5 w-1.5 rounded-full ${
    state === 'DONE' ? 'bg-emerald-400'
      : state === 'RUNNING' ? 'bg-[var(--accent)] animate-pulse'
        : state === 'FAILED' ? 'bg-red-400'
          : state === 'RESTORED' ? 'bg-sky-300' : 'bg-white/20'}`} aria-hidden="true" />
);

function Section({ section }) {
  return (
    <div className="rounded-xl border border-white/[0.08] bg-black/40 px-3 py-2 min-w-0" data-stage={section.key}>
      <div className="flex items-center justify-between gap-2 border-b border-white/5 pb-1.5">
        <span className="text-nano font-bold tracking-[0.14em] text-white/70 uppercase truncate">{section.label}</span>
        <span className={`inline-flex items-center gap-1 text-nano font-semibold uppercase ${stateTone(section.state)}`}>
          <Dot state={section.state} />{section.state}
        </span>
      </div>
      <div className="mt-1.5 space-y-0.5">
        {section.rows.map((r, i) => (
          <div key={`${r.label}-${i}`} className="flex items-baseline justify-between gap-2 text-micro leading-relaxed">
            <span className="min-w-0 truncate text-muted">{r.label}</span>
            <span className={`min-w-0 text-right font-mono ${r.tone || 'text-white/80'}`} title={String(r.value)}>{r.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** A finished report: headline numbers, then one block per stage. `report` is the backend's summary object. */
export function PipelineSummaryCard({ report, title = 'Pipeline report', defaultOpen = true, className = '' }) {
  const head = headline(report);
  const sections = reportSections(report);
  if (!head || !sections.length) {
    return (
      <div className={`text-micro text-muted italic ${className}`}>
        No finished run to report yet — the report is built when a render ends.
      </div>
    );
  }
  return (
    <details open={defaultOpen} className={`group rounded-2xl border border-white/10 bg-white/[0.02] ${className}`} data-testid="pipeline-report">
      <summary className="cursor-pointer list-none flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2">
        <span className="text-nano font-bold uppercase tracking-[0.16em] text-white/60">{title}</span>
        <span className={`text-micro font-semibold ${head.ok ? 'text-emerald-400' : 'text-red-300'}`}>
          {head.ok ? 'verified' : 'FAILED'}
        </span>
        <span className="text-micro font-mono text-white/80" title={head.fpsTitle}>{head.frames} frames · {head.wall} · {head.fps}</span>
        <span className="text-micro font-mono text-white/60" title="Estimated from this run's own per-active-frame cost; a dense render is not run during a job">
          speedup {head.speedup}
        </span>
        <span className="text-micro font-mono text-white/60">peak VRAM {head.vram}</span>
      </summary>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 px-3 pb-3">
        {sections.map((s) => <Section key={s.key} section={s} />)}
      </div>
      {auxOffloadText(report.aux_offload) && (
        <div className="px-3 pb-2.5 text-nano text-muted" data-testid="aux-offload">
          OpenVINO aux offload: {auxOffloadText(report.aux_offload)}
        </div>
      )}
      {report.speedup && report.speedup.method && (
        <div className="px-3 pb-2.5 text-nano text-muted">
          Speedup: {report.speedup.reason || report.speedup.method}
          {report.videos_in_job > 1 ? ` · metrics are for the last of ${report.videos_in_job} videos` : ''}
        </div>
      )}
    </details>
  );
}

/** The one stage-specific line under a live stage's progress bar: what that stage is doing right now. */
function liveDetail(name, s) {
  if (name === STAGE_1) {
    return s.scene_cuts != null ? `${fmtCount(s.scene_cuts)} scene cuts` : '';
  }
  if (name === STAGE_2) {
    if (s.neural_detected_frames == null && s.flow_tracked_frames == null) return '';
    // "NVOFA tracked" only when the backend reported that engine; an unreported one stays the neutral word "flow".
    const engine = flowEngineLabel(s.flow_backend);
    return `${fmtCount(s.flow_tracked_frames)} flow · ${fmtCount(s.neural_detected_frames)} neural`
      + (s.flow_ratio_pct != null ? ` (${fmtPct(s.flow_ratio_pct)} ${engine ? `${engine} tracked` : 'flow'})` : '');
  }
  return '';
}

/** Stage 3's per-operation ms line: Decode | Warp | TRT | Blend | NVENC, each with the backend it ran on. */
function OpsStrip({ ops }) {
  const cells = opsCells(ops).filter((c) => c.key !== 'restore' || c.calls != null);
  if (!cells.length) return null;
  return (
    <div className="mt-1 flex flex-wrap gap-x-2.5 gap-y-0.5 text-nano font-mono text-white/70" data-testid="stage3-ops">
      {cells.map((c) => (
        <span key={c.key} title={c.title} className={c.calls == null ? 'text-muted' : ''}>{c.label} {c.value}</span>
      ))}
    </div>
  );
}

/** Stage 3's throughput: Effective Output FPS (frames written) | Neural inf/s | Smart-Skip %, from retired frames. */
function RatesStrip({ stage }) {
  const cells = stage3RateCells(stage);
  if (!cells.length) return null;
  return (
    <div className="mt-1 flex flex-wrap gap-x-2.5 gap-y-0.5 text-nano font-mono text-white/70" data-testid="stage3-rates">
      {cells.map((c) => <span key={c.key} title={c.title}>{c.label} {c.value}</span>)}
    </div>
  );
}

/** One live row per stage while a render runs: state, stage-local progress, elapsed time, stage rate. */
function LiveRows({ pipeline }) {
  const stages = pipeline.stages || {};
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2" data-testid="pipeline-live">
      {STAGE_ORDER.slice(0, 3).map((name) => {
        const s = stages[name];
        if (!s) return null;
        const running = s.state === 'RUNNING';
        return (
          <div key={name} className="rounded-xl border border-white/[0.08] bg-black/40 px-3 py-2 min-w-0" data-stage={name}>
            <div className="flex items-center justify-between gap-2">
              <span className="text-nano font-bold tracking-[0.14em] text-white/70 uppercase truncate">{STAGE_LABEL[name]}</span>
              <span className={`inline-flex items-center gap-1 text-nano font-semibold uppercase ${stateTone(s.state)}`}>
                <Dot state={s.state} />{s.state}
              </span>
            </div>
            <div className="mt-1.5 h-1 rounded-full bg-white/[0.06] overflow-hidden">
              <div className={`h-full rounded-full ${running ? 'bg-[var(--accent)]' : s.state === 'DONE' || s.state === 'RESTORED' ? 'bg-emerald-400/70' : 'bg-transparent'}`}
                   style={{ width: `${s.state === 'DONE' || s.state === 'RESTORED' ? 100 : Math.max(0, Math.min(100, s.stage_progress_pct || 0))}%` }} />
            </div>
            <div className="mt-1 flex justify-between text-micro font-mono text-muted">
              <span>{running ? fmtPct(s.stage_progress_pct) : s.state === 'SKIPPED' ? 'not run' : ''}</span>
              <span>{s.fps != null ? `${fmtRate(s.fps, name === STAGE_1 ? 'keyframes/s' : 'fps')} · ` : ''}{fmtSecs(s.wall_s)}</span>
            </div>
            {liveDetail(name, s) && (
              <div className="mt-0.5 text-nano font-mono text-white/70" data-testid={`${name}-detail`}>{liveDetail(name, s)}</div>
            )}
            {name === STAGE_3 && <RatesStrip stage={s} />}
            {name === STAGE_3 && <OpsStrip ops={s.ops} />}
          </div>
        );
      })}
    </div>
  );
}

/**
 * The Diagnostics HUD for #/processing. Renders nothing while the backend has reported no run at all (an idle
 * app), the live stage rows while one runs, and the finished report once `summary` arrives.
 */
export function PipelineHud({ runtime = null, className = '' }) {
  const pipeline = useTelemetryStore((s) => s.pipeline);
  const run = useEngineRun();
  return <PipelineHudView pipeline={pipeline} runtime={runtime} run={run} className={className} />;
}

/** `CanaryPrecisionStatus` wired to the live engine state, for tabs that have the runtime report but not the HUD. */
export function CanaryPrecisionPanel({ precision, className = '' }) {
  return <CanaryPrecisionStatus precision={precision} run={useEngineRun()} className={className} />;
}

/**
 * Canary precision status: the fidelity canaries' verdicts per model (roop/aux_canary.py, via the structured runtime
 * report's PRECISION section) and whether one is running right now. States and their meaning are `canaryView`'s.
 */
export function CanaryPrecisionStatus({ precision, run, className = '' }) {
  const v = canaryView(precision, run);
  return (
    <div className={`rounded-xl border border-white/[0.08] bg-black/40 px-3 py-2 min-w-0 ${className}`} data-testid="canary-status" data-state={v.state}>
      <div className="flex items-center justify-between gap-2 border-b border-white/5 pb-1.5">
        <span className="text-nano font-bold tracking-[0.14em] text-white/70 uppercase truncate">Canary precision</span>
        <span className={`text-nano font-semibold uppercase ${canaryTone(v.state)}`} data-testid="canary-state">{v.state}</span>
      </div>
      <div className="mt-1.5 space-y-0.5 text-note">
        {v.configured || v.effective ? (
          <div className="flex items-baseline justify-between gap-2">
            <span className="min-w-0 truncate text-muted">Precision (configured → effective)</span>
            <span className="min-w-0 text-right font-mono text-white/80">{v.configured || '—'} → {v.effective || '—'}</span>
          </div>
        ) : null}
        {v.live && (
          <div className="flex items-baseline justify-between gap-2" data-testid="canary-live">
            <span className="min-w-0 truncate text-muted">Verifying now</span>
            <span className="min-w-0 text-right font-mono text-[var(--accent)]">{v.live.model}{v.live.precision ? ` · ${v.live.precision}` : ''}</span>
          </div>
        )}
        {v.rows.map((r) => (
          <div key={r.stem} className="flex items-baseline justify-between gap-2" data-canary={r.stem}>
            <span className="min-w-0 truncate text-muted">{r.stem}</span>
            <span className={`min-w-0 text-right font-mono ${r.result === 'PASS' ? 'text-emerald-400' : r.result === 'FAIL' ? 'text-red-300' : 'text-muted'}`}
                  title={r.reason || undefined}>
              {r.result === 'PASS' ? '✓ pass' : r.result === 'FAIL' ? `✗ fail${r.reason ? ` · ${r.reason}` : ''}` : 'unknown'}
            </span>
          </div>
        ))}
        {!v.rows.length && !v.live && (
          <div className="text-muted italic">No canary verdict recorded yet — they are written the first time each engine is verified.</div>
        )}
      </div>
    </div>
  );
}

/** The HUD as a pure function of the store's `pipeline` slice (what the render-check drives directly). */
export function PipelineHudView({ pipeline, runtime = null, run = null, className = '' }) {
  const hasStages = pipeline && pipeline.stages && Object.keys(pipeline.stages).length > 0;
  const summary = pipeline && pipeline.summary && pipeline.summary.stages ? pipeline.summary : null;
  const precision = runtime && runtime.sections && runtime.sections.PRECISION ? runtime.sections.PRECISION.values : null;
  // The canary block is shown whenever the backend has said anything about canaries (a verdict, or one running), even
  // between runs, and during any run: "no verdict recorded yet" while a render is going is a fact worth stating, not
  // one to hide. It is the only part of the HUD that is not a statement about the current job.
  const canary = canaryView(precision, run);
  const showCanary = canary.state !== 'NONE' || hasStages || !!summary;
  if (!hasStages && !summary && !showCanary) return null;
  return (
    <div className={`flex flex-col gap-2.5 ${className}`} data-testid="pipeline-hud">
      {hasStages && !summary && <LiveRows pipeline={pipeline} />}
      {summary && <PipelineSummaryCard report={summary} />}
      {showCanary && <CanaryPrecisionStatus precision={precision} run={run} />}
    </div>
  );
}

export default PipelineHud;
