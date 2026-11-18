import React, { useMemo } from 'react';
import { scalingGroups, scalingVerdict } from './scalingModel';

/**
 * The thread / pool benchmark as horizontal bars: one block per stage, one bar
 * per pool size, each bar's length its throughput relative to that stage's best.
 * The best bar is accented; an unstable bar is hatched so a noisy number is not
 * mistaken for a clean one.
 */
export default function ThreadScalingChart({ result }) {
  const groups = useMemo(() => scalingGroups(result), [result]);
  if (!groups.length) return null;

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5 space-y-3" data-testid="thread-scaling" aria-label="Thread and pool scaling">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs font-bold text-white/85">Thread &amp; pool scaling</div>
        <div className="text-micro text-muted">Bar length = calls per second relative to each stage's best</div>
      </div>
      {groups.map((g) => (
        <div key={g.key} className="space-y-1.5" data-stage={g.key}>
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-mini font-semibold text-white/80 truncate">{g.label}</span>
            <span className="text-nano text-muted truncate">{scalingVerdict(g)}</span>
          </div>
          <ul className="m-0 p-0 list-none space-y-1">
            {g.bars.map((b) => (
              <li key={b.n} className="flex items-center gap-2 text-nano font-mono" data-threads={b.n} data-best={b.best ? 'yes' : 'no'}>
                <span className="w-16 shrink-0 text-right text-white/55 tabular-nums">{b.n} thr</span>
                <div className="relative h-3 flex-1 rounded bg-white/[0.05] overflow-hidden"
                     role="img" aria-label={`${b.n} threads: ${b.value.toFixed(1)} calls per second${b.stable ? '' : ', unstable'}`}>
                  <div className={`h-full rounded ${b.best ? 'bg-[var(--accent)]' : 'bg-cyan-400/50'}`}
                       style={{
                         width: `${Math.max(2, b.share * 100)}%`,
                         backgroundImage: b.stable ? undefined : 'repeating-linear-gradient(45deg, rgba(0,0,0,0.25) 0 4px, transparent 4px 8px)',
                       }} />
                </div>
                <span className="w-20 shrink-0 text-right text-white/80 tabular-nums">{b.value.toFixed(1)}/s</span>
                {!b.stable && <span className="text-amber-300 text-nano w-10 shrink-0">noisy</span>}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
