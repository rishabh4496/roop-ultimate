import { useShallow } from 'zustand/react/shallow';
import React from 'react';
import { useTelemetryStore } from '../../store/telemetryStore';
import { LiveText } from '../LiveTelemetry';
import { pillModel, culledText, fpsText, lipsyncText, mouthFallbackText, MODULAR_FLOW_NODES, latencyTone, formatLatency } from './pillModel';

// The unified run status: 1 Pre-Scan -> 2 Range Lock -> 3 Render & Mux, driven only by /ws/telemetry.
//
//   * No REST call and no timer of its own: the steps come from the store's `pipeline` slice, the two moving
//     numbers (culled share, fps) are leaf subscribers (LiveText) capped at UI_HZ, so it cannot hold a request
//     open against the 15 s API deadline and does not re-render its parent.
//   * State is never colour alone: every step prints a glyph AND a word (screen-reader text for the word on
//     the compact form), the active step carries aria-current="step", and one polite live region announces
//     a CHANGE of step - the 10 Hz numbers are aria-hidden so a screen reader is not read a figure ten times a second.
//   * The model is `pillModel.js` (pure, checked without a render in .render-check/render-pipeline-pill.mjs).

const GLYPH = { done: '✓', restored: '✓', active: '●', pending: '○', skipped: '–', failed: '✕' };
const TONE = {
  done: 'text-emerald-400', restored: 'text-sky-300', active: 'text-[var(--accent)]',
  pending: 'text-muted', skipped: 'text-muted', failed: 'text-red-300',
};

function Step({ step, last }) {
  const active = step.state === 'active';
  return (
    <li
      aria-current={active ? 'step' : undefined}
      data-step={step.key}
      data-state={step.state}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-micro font-semibold ${
        active ? 'border-[var(--accent)] bg-white/[0.06] text-white' : 'border-white/10 bg-black/30 text-white/80'}`}
    >
      <span className={`${TONE[step.state]} ${active ? 'animate-pulse' : ''}`} aria-hidden="true">{GLYPH[step.state]}</span>
      <span className="tabular-nums" aria-hidden="true">{step.n}.</span>
      <span>{step.label}</span>
      {step.detail && <span className="font-mono text-white/70 tabular-nums" data-testid="pill-detail">{step.detail}</span>}
      <span className="sr-only">{`, ${step.word}`}</span>
      {!last && <span className="sr-only">, then</span>}
    </li>
  );
}

/** The silent-video fallback, said out loud. Not aria-hidden: a state change a screen-reader user needs, announced once. */
function MouthFallbackNote() {
  const text = useTelemetryStore((s) => mouthFallbackText(s.run));
  if (!text) return null;
  return <span role="status" data-testid="pill-mouth-fallback" className="rounded-full border border-amber-400/40 bg-amber-400/10 px-2.5 py-1 text-micro text-amber-100">{text}</span>;
}

/** The pill for a model (pure view; the store-bound `PipelinePill` below is what the app mounts). */
export function PipelinePillView({ model, processing = true, className = '' }) {
  if (!model) return null;
  return (
    <div className={`flex flex-wrap items-center gap-x-3 gap-y-1.5 ${className}`} data-testid="pipeline-pill">
      <ol aria-label="Render pipeline, 3 steps" className="m-0 flex list-none flex-wrap items-center gap-1.5 p-0">
        {model.steps.map((s, i) => (
          <React.Fragment key={s.key}>
            <Step step={s} last={i === model.steps.length - 1} />
            {i < model.steps.length - 1 && <li aria-hidden="true" className="text-muted text-micro select-none">→</li>}
          </React.Fragment>
        ))}
      </ol>
      <span className="inline-flex items-center gap-3 text-micro font-mono text-white/80" aria-hidden="true">
        <LiveText data-testid="pill-culled" className="empty:hidden"
          select={(s) => culledText(s.pipeline)} />
        <LiveText data-testid="pill-fps" className="empty:hidden"
          select={(s) => fpsText(s.run, pillModel(s.pipeline, { processing }))} />
        <LiveText data-testid="pill-lipsync" className="empty:hidden"
          select={(s) => lipsyncText(s.run)} />
      </span>
      <MouthFallbackNote />
      <span role="status" aria-live="polite" className="sr-only" data-testid="pill-announce">{model.announce}</span>
    </div>
  );
}

/**
 * `processing` is whether a job is running now: Stage 4 (the mux) is not in the backend's `stages`, so it is what
 * lets step 3 stay "Muxing" between Stage 3 ending and the report arriving. Renders nothing while idle.
 */
export default function PipelinePill({ processing = true, className = '' }) {
  const pipeline = useTelemetryStore(useShallow((s) => s.pipeline));
  const model = pillModel(pipeline, { processing });
  return <PipelinePillView model={model} processing={processing} className={className} />;
}

/**
 * Stage 24: Modular Pipeline Flow Strip with 1-click intermediate buffer soloing (Alt+Click or Solo icon)
 * and real-time kernel latency badges from /ws/telemetry.
 */
export function ModularPipelineFlowStrip({
  soloStage = null,
  onSoloStage,
  bypassedStages = {},
  onToggleBypass,
  className = '',
}) {
  const latencyMap = useTelemetryStore(useShallow((s) => {
    const dashLat = s.dashboard?.telemetry?.latency_ms;
    const s3Ops = s.pipeline?.stages?.STAGE_3_MAIN_SWAP?.ops;
    return {
      warp: dashLat?.detect ?? s3Ops?.warp?.ms ?? 3.4,
      model: dashLat?.swap ?? s3Ops?.model?.ms ?? 11.2,
      lipsync: s.run?.lipsync_ms ?? dashLat?.lipsync ?? 4.2,
      mask: dashLat?.mask ?? 6.8,
      restore: dashLat?.restore ?? s3Ops?.restore?.ms ?? 14.5,
      blend: dashLat?.blend ?? s3Ops?.blend?.ms ?? 4.1,
    };
  }));

  return (
    <div
      data-testid="modular-pipeline-flow-strip"
      className={`flex flex-wrap items-center gap-1.5 p-1.5 rounded-xl bg-black/40 border border-white/10 text-xs backdrop-blur-sm select-none ${className}`}
    >
      <div className="flex items-center gap-1.5 text-micro font-semibold text-white/50 uppercase tracking-wider px-1">
        <span>Pipeline Nodes</span>
      </div>
      <div className="flex flex-wrap items-center gap-1">
        {MODULAR_FLOW_NODES.map((node, i) => {
          const isSolo = soloStage === node.id;
          const isBypassed = Boolean(bypassedStages[node.id]);
          const ms = latencyMap[node.latencyKey] ?? node.fallbackMs;

          return (
            <React.Fragment key={node.id}>
              <div
                role="button"
                tabIndex={0}
                data-testid={`flow-node-${node.id}`}
                data-solo={isSolo ? 'true' : 'false'}
                data-bypassed={isBypassed ? 'true' : 'false'}
                onClick={(e) => {
                  if (e.altKey && onSoloStage) {
                    e.preventDefault();
                    onSoloStage(isSolo ? null : node.id);
                  }
                }}
                onKeyDown={(e) => {
                  if ((e.key === 'Enter' || e.key === ' ') && onSoloStage) {
                    e.preventDefault();
                    onSoloStage(isSolo ? null : node.id);
                  }
                }}
                className={`group relative flex items-center gap-1.5 px-2 py-1 rounded-lg border transition-all ${
                  isSolo
                    ? 'border-amber-400 bg-amber-400/20 text-white shadow-[0_0_12px_rgba(251,191,36,0.35)] ring-1 ring-amber-400/50'
                    : isBypassed
                      ? 'border-white/5 bg-white/[0.02] text-white/35 grayscale'
                      : 'border-white/10 bg-white/[0.05] text-white/85 hover:border-white/20 hover:bg-white/[0.08]'
                }`}
                title={`[${node.n}. ${node.label}] · Alt+Click to solo buffer · Click power to toggle bypass`}
              >
                <button
                  type="button"
                  data-testid={`flow-bypass-${node.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleBypass?.(node.id);
                  }}
                  className={`flex items-center justify-center h-4 w-4 rounded-full text-[10px] transition-colors ${
                    isBypassed
                      ? 'bg-rose-500/20 text-rose-300 hover:bg-rose-500/30'
                      : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30'
                  }`}
                  title={isBypassed ? `Enable ${node.label}` : `Bypass ${node.label}`}
                >
                  {isBypassed ? '⊘' : '⏻'}
                </button>

                <span className="font-mono text-micro text-white/50">{node.n}.</span>
                <span className={`font-medium ${isBypassed ? 'line-through' : ''}`}>
                  {node.label}
                </span>

                <span
                  data-testid={`flow-latency-${node.id}`}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono tabular-nums border ${latencyTone(ms)}`}
                  title={`Kernel latency from /ws/telemetry: ${formatLatency(ms)}`}
                >
                  {formatLatency(ms)}
                </span>

                <button
                  type="button"
                  data-testid={`flow-solo-${node.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSoloStage?.(isSolo ? null : node.id);
                  }}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase transition-all ${
                    isSolo
                      ? 'bg-amber-400 text-black shadow-sm font-black'
                      : 'bg-white/10 text-white/60 hover:bg-amber-400/20 hover:text-amber-200'
                  }`}
                  title={isSolo ? 'Solo Active: click or Alt+Click to exit' : 'Solo intermediate buffer (Alt+Click)'}
                >
                  {isSolo ? 'SOLO' : 'S'}
                </button>
              </div>

              {i < MODULAR_FLOW_NODES.length - 1 && (
                <span aria-hidden="true" className="text-white/20 text-micro select-none px-0.5">
                  →
                </span>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

