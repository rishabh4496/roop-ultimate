import React, { useEffect, useState } from 'react';
import { useApi } from '../../useApi';
import { reportFailure } from '../../failureLog';
import { METRICS, readingText, sparkGeometry, summarize } from '../telemetry/rollingSeries';
import { useRollingTelemetry } from '../telemetry/useRollingTelemetry';
import { MODULE_LINKS, readinessOf } from './launchpadModel';

// The launchpad's status band: live hardware, model-weight readiness, and one
// click into each workstation module. Nothing here is estimated: a reading the
// machine did not report reads "not reported", and a model the check could not
// see reads "not checked".

const TONE = {
  ready: 'text-emerald-300 border-emerald-400/30 bg-emerald-500/10',
  blocked: 'text-amber-300 border-amber-400/35 bg-amber-500/10',
  unknown: 'text-white/60 border-white/10 bg-white/5',
};

function HardwareTile({ metric, samples, now }) {
  const stats = summarize(samples, metric.key);
  const geo = sparkGeometry(samples, metric, { now, width: 120, height: 28 });
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3 min-w-0" data-testid={`launch-hw-${metric.key}`}>
      <div className="text-nano font-semibold uppercase tracking-[0.12em] text-muted truncate">{metric.label}</div>
      <div className="mt-0.5 font-mono text-compact font-bold text-white tabular-nums">
        {stats.reported ? readingText(metric, stats.last) : 'not reported'}
      </div>
      <svg viewBox="0 0 120 28" preserveAspectRatio="none" className="mt-1 h-7 w-full" aria-hidden="true">
        {geo.d && <path d={geo.d} fill="none" stroke={metric.tone} strokeWidth="1.4" vectorEffect="non-scaling-stroke" />}
      </svg>
    </div>
  );
}

export default function LaunchpadBand({ setTab, queueCount = 0 }) {
  const { getJSON } = useApi();
  const samples = useRollingTelemetry();
  const now = samples.length ? samples[samples.length - 1].t : Date.now();
  const [readiness, setReadiness] = useState(null);

  useEffect(() => {
    let live = true;
    getJSON('/api/models/offline', { timeout: 15000 })
      .then((r) => { if (live) setReadiness(readinessOf(r?.models)); })
      .catch((err) => {
        reportFailure('Checking model weights', err, { toast: false });
        if (live) setReadiness(readinessOf({}));
      });
    return () => { live = false; };
  }, [getJSON]);

  const r = readiness;
  return (
    <section aria-label="Studio launchpad status" className="space-y-4" data-testid="launchpad-band">
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
        <div className="xl:col-span-7 rounded-2xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="m-0 text-mini font-semibold uppercase tracking-[0.16em] text-accent">Live hardware</h2>
            <span className="text-micro text-muted">last 60 s</span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
            {METRICS.map((m) => <HardwareTile key={m.key} metric={m} samples={samples} now={now} />)}
          </div>
        </div>

        <div className="xl:col-span-5 rounded-2xl border border-white/10 bg-white/[0.02] p-4 space-y-3" data-testid="model-readiness">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="m-0 text-mini font-semibold uppercase tracking-[0.16em] text-accent">Model weight readiness</h2>
            {r && <span className={`rounded-full border px-2 py-0.5 text-nano font-bold uppercase tracking-wider ${TONE[r.tone]}`}>{r.tone === 'ready' ? 'Ready' : r.tone === 'blocked' ? 'Attention' : 'Unknown'}</span>}
          </div>
          {!r ? (
            <p className="m-0 text-xs text-muted">Checking weights on disk…</p>
          ) : (
            <>
              <p className="m-0 text-sm font-semibold text-white/90">{r.label}</p>
              {r.total > 0 && (
                <div className="h-1.5 rounded-full bg-black/40 overflow-hidden" role="progressbar" aria-valuemin={0} aria-valuemax={r.total} aria-valuenow={r.ok} aria-label="Model weights present">
                  <div className="h-full bg-emerald-400/80" style={{ width: `${(r.ok / r.total) * 100}%` }} />
                </div>
              )}
              {r.blocking.length > 0 && (
                <ul className="m-0 p-0 list-none space-y-1 max-h-28 overflow-y-auto" aria-label="Models needing attention">
                  {r.blocking.slice(0, 8).map((b) => (
                    <li key={b.file} className="flex items-center justify-between gap-2 text-micro">
                      <span className="font-mono text-white/75 truncate" title={b.file}>{b.file}</span>
                      <span className="text-amber-300 font-bold uppercase text-nano shrink-0">{b.status}</span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      </div>

      <nav aria-label="Workstation modules">
        <ul className="m-0 p-0 list-none grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-2.5">
          {MODULE_LINKS.map((m) => {
            const count = m.tab === 'processing' ? queueCount : 0;
            return (
              <li key={m.tab}>
                <button type="button" onClick={() => setTab?.(m.tab)} data-deeplink={m.tab}
                        className="group w-full text-left rounded-xl border border-white/10 bg-white/[0.025] hover:bg-white/[0.06] hover:border-[var(--accent)]/40 p-3.5 transition-colors">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-bold text-white/90 group-hover:text-white">{m.label}</span>
                    <span className="flex items-center gap-2 text-muted group-hover:text-accent">
                      {count > 0 && <span className="rounded-full bg-[var(--accent)]/20 px-2 py-0.5 text-nano font-bold text-accent tabular-nums">{count} queued</span>}
                      <span aria-hidden="true">→</span>
                    </span>
                  </div>
                  <p className="m-0 mt-1 text-micro text-muted">{m.blurb}</p>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </section>
  );
}
