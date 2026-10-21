import React from 'react';
import { useTelemetryStore } from '../../store/telemetryStore';
import { LiveText } from '../LiveTelemetry';
import { pillModel, culledText, fpsText } from './pillModel';

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
      </span>
      <span role="status" aria-live="polite" className="sr-only" data-testid="pill-announce">{model.announce}</span>
    </div>
  );
}

/**
 * `processing` is whether a job is running now: Stage 4 (the mux) is not in the backend's `stages`, so it is what
 * lets step 3 stay "Muxing" between Stage 3 ending and the report arriving. Renders nothing while idle.
 */
export default function PipelinePill({ processing = true, className = '' }) {
  const pipeline = useTelemetryStore((s) => s.pipeline);
  const model = pillModel(pipeline, { processing });
  return <PipelinePillView model={model} processing={processing} className={className} />;
}
