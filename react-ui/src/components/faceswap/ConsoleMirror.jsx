import React from 'react';
import { useTelemetryStore } from '../../store/telemetryStore';
import { STAGE_TONE, bottleneckText, hasJob, hitRateCells, latencyCells, queueBound, queueCells, stageRows } from './dashboardModel';

// The terminal dashboard on #/processing: the four stages, the per-frame latency split, the bottleneck and the
// optimizer hit rates - the numbers the console prints, because the backend paints the console from the same object
// it sends here (`dashboard` on /ws/telemetry, roop/terminal_dashboard.py). Nothing is computed in this file.
//
// It renders nothing until a job has run, and keeps the last job's final numbers on screen after it ends (like the
// console's closing lines). State is never colour alone: every row prints the stage name, the state word and the
// numbers; the bars are decorative (aria-hidden) beside a role="progressbar" carrying the same percentage.

const SEVERITY = { red: 'text-red-300', yellow: 'text-amber-300' };
// The verdict is words in brackets first; the tone only reinforces it (never colour alone).
const BOUND_TONE = { ok: 'text-emerald-300', warn: 'text-amber-300', muted: 'text-muted' };

function StageRow({ row }) {
  const tone = STAGE_TONE[row.id];
  const dim = row.state === 'PENDING' || row.state === 'SKIPPED';
  return (
    <li className="grid items-center gap-x-3 gap-y-0.5 py-1 grid-cols-[4.5rem_minmax(0,1fr)] lg:grid-cols-[6.5rem_minmax(8rem,14rem)_4.5rem_minmax(0,1fr)]"
      data-stage={row.id} data-state={row.state}>
      <span className={`text-micro font-bold uppercase tracking-wide ${dim ? 'text-muted' : tone.text}`}>{row.id} {row.name}</span>
      <div role="progressbar" aria-label={`${row.id} ${row.title}`} aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={row.pct == null ? undefined : Math.round(row.pct)} aria-valuetext={`${row.pctText} ${row.status}`}
        className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
        <div aria-hidden="true" className={`h-full rounded-full ${tone.bar}`} style={{ width: `${row.pct == null ? 0 : row.pct}%` }} />
      </div>
      <span className={`text-micro font-mono tabular-nums ${dim ? 'text-muted' : 'text-white/90'}`}>{row.pct == null ? '' : row.pctText}</span>
      <span className="col-span-full text-micro font-mono text-muted lg:col-span-1 min-w-0 truncate"
        title={[row.counts, row.elapsed, row.status, row.note].filter(Boolean).join('  ')}>
        {[row.counts, row.elapsed, row.status, row.note].filter(Boolean).join('  ·  ')}
      </span>
    </li>
  );
}

/** The mirror for one snapshot. Exported on its own so it can be rendered from a fixture. */
export function ConsoleMirrorView({ dash }) {
  if (!hasJob(dash)) return null;
  const rows = stageRows(dash);
  const cells = latencyCells(dash);
  const hits = hitRateCells(dash);
  const queues = queueCells(dash);
  const bound = queueBound(dash);
  const bn = dash.telemetry && dash.telemetry.bottleneck;
  const job = dash.job || {};
  return (
    <section aria-label="Console mirror" data-testid="console-mirror" data-active={dash.active ? 'true' : 'false'}
      className="rounded-2xl border border-white/10 bg-black/40 px-3 py-2.5">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-white/5 pb-1.5">
        <span className="text-nano font-bold uppercase tracking-[0.16em] text-white/70">
          Console{dash.active ? '' : ' · last run'}
        </span>
        <span className="text-micro text-muted font-mono truncate" title={job.label || ''}>
          {[job.mode && String(job.mode).toUpperCase(), job.batch_count > 1 && `queue ${job.batch_index}/${job.batch_count}`, job.label]
            .filter(Boolean).join(' · ')}
        </span>
      </div>
      <ul className="mt-1 divide-y divide-white/[0.04]" aria-label="Pipeline stages">
        {rows.map((r) => <StageRow key={r.id} row={r} />)}
      </ul>
      <div className="mt-1.5 border-t border-white/5 pt-1.5 space-y-1" data-testid="console-telemetry">
        <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-micro font-mono" aria-label="Per-frame latency in milliseconds">
          {cells.map((c) => (
            <span key={c.key} data-col={c.key} data-bottleneck={c.bottleneck ? 'true' : 'false'}
              className={c.bottleneck ? `${SEVERITY[c.severity] || 'text-amber-300'} font-bold` : 'text-white/80'}>
              {c.label} {c.ms}
            </span>
          ))}
          <span className="text-muted">ms / frame</span>
        </div>
        {queues.length > 0 && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-micro font-mono" data-testid="console-queues"
            data-bound={dash.telemetry.queues.bound || ''} aria-label="Pipeline queue fill">
            {queues.map((c) => (
              <span key={c.key} data-queue={c.key} className="inline-flex items-center gap-1.5 text-white/80"
                title={c.unit ? `${c.label} (${c.unit})` : c.label}>
                {c.label} <span className="tabular-nums">{c.value}</span>
                <span aria-hidden="true" className="inline-block h-1 w-10 overflow-hidden rounded-full bg-white/10">
                  <span className="block h-full rounded-full bg-cyan-400" style={{ width: `${c.fill == null ? 0 : Math.round(c.fill * 100)}%` }} />
                </span>
              </span>
            ))}
            <span data-testid="console-bound" title={bound.basis}
              className={`font-bold ${BOUND_TONE[bound.tone]}`}>{bound.text}</span>
          </div>
        )}
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-0.5 text-micro font-mono">
          <span className={bn ? `${SEVERITY[bn.severity] || 'text-amber-300'} font-bold` : 'text-muted'}>{bottleneckText(dash)}</span>
          {hits.map((h) => (
            <span key={h.key} className="text-white/80" data-hit={h.key}><span className="text-muted">{h.label}</span> {h.value}</span>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function ConsoleMirror() {
  const dash = useTelemetryStore((s) => s.dashboard);
  return <ConsoleMirrorView dash={dash} />;
}
