import React from 'react';
import { METRICS, WINDOW_MS, readingText, sparkGeometry, summarize } from './rollingSeries';
import { useRollingTelemetry } from './useRollingTelemetry';

// Four rolling 60-second sparklines: GPU utilisation, VRAM allocation, CPU load,
// and frame render time. Each tile says what the machine reported, with the
// window's min / mean / max beside it. A metric with no readings in the window
// shows "not reported" rather than a flat line.

function Sparkline({ metric, samples, now }) {
  const geo = sparkGeometry(samples, metric, { now, width: 240, height: 56 });
  const stats = summarize(samples, metric.key);
  const gradientId = `spark-${metric.key}`;
  const fmt = (v) => (v === null ? '—' : readingText(metric, v));
  return (
    <figure className="m-0 rounded-xl border border-white/10 bg-white/[0.02] p-3 min-w-0" data-metric={metric.key}>
      <figcaption className="flex items-baseline justify-between gap-2">
        <span className="text-micro font-semibold uppercase tracking-[0.12em] text-muted truncate">{metric.label}</span>
        <span className="font-mono text-sm font-bold text-white tabular-nums" data-testid={`spark-value-${metric.key}`}>
          {stats.reported ? fmt(stats.last) : 'not reported'}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${geo.width} ${geo.height}`} preserveAspectRatio="none" className="mt-2 h-14 w-full overflow-visible"
           role="img" aria-label={`${metric.label}, last ${WINDOW_MS / 1000} seconds: ${stats.reported ? `now ${fmt(stats.last)}, mean ${fmt(stats.mean)}, peak ${fmt(stats.max)}` : 'not reported'}`}>
        <defs>
          <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={metric.tone} stopOpacity="0.28" />
            <stop offset="100%" stopColor={metric.tone} stopOpacity="0" />
          </linearGradient>
        </defs>
        <line x1="0" x2={geo.width} y1={geo.height / 2} y2={geo.height / 2} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" />
        {geo.d && (
          <>
            <path d={geo.d} fill="none" stroke={metric.tone} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round"
                  vectorEffect="non-scaling-stroke" />
          </>
        )}
        {geo.last && (
          <circle cx={geo.last.x} cy={geo.last.y} r="2.6" fill={metric.tone} vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      <div className="mt-1.5 flex items-center justify-between font-mono text-nano text-white/50 tabular-nums">
        <span>min {fmt(stats.min)}</span>
        <span>mean {fmt(stats.mean)}</span>
        <span>peak {fmt(stats.max)}</span>
      </div>
    </figure>
  );
}

export default function TelemetrySparklines() {
  const samples = useRollingTelemetry();
  const now = samples.length ? samples[samples.length - 1].t : Date.now();
  return (
    <section aria-label="Hardware telemetry, last 60 seconds" data-testid="telemetry-sparklines" className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="m-0 text-mini font-semibold uppercase tracking-[0.16em] text-accent">Live hardware · last 60 s</h2>
        <span className="text-micro text-muted">{samples.length} samples · 1 per second</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 2xl:grid-cols-4 gap-3">
        {METRICS.map((m) => (
          <Sparkline key={m.key} metric={m} samples={samples} now={now} />
        ))}
      </div>
    </section>
  );
}
