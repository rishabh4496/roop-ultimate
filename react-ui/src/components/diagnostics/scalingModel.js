// Thread / pool scaling from a finished benchmark (GET /api/benchmark/result).
//
// roop/bench.py sweeps each pooled stage at increasing pool sizes `n` and records
// the calls per second it reached (`calls_s`), plus whether repeated runs agreed
// (`stable`, from the run-to-run spread). This turns those rows into bars that
// can be compared: each stage normalised to its own best, so a stage that scales
// to 4 threads and one that stops at 2 read on the same scale.

/** Rows for a horizontal bar chart: one group per stage, one bar per pool size. */
export function scalingGroups(result) {
  const stages = Array.isArray(result?.stages) ? result.stages : [];
  return stages
    .map((st) => {
      const rows = Array.isArray(st.scaling) ? st.scaling : [];
      const usable = rows.filter((r) => Number.isFinite(Number(r.n)) && Number.isFinite(Number(r.calls_s)));
      if (!usable.length) return null;
      const best = Math.max(...usable.map((r) => Number(r.calls_s)));
      const bestRow = usable.find((r) => Number(r.calls_s) === best);
      return {
        key: st.key || st.label || 'stage',
        label: st.label || st.key || 'Stage',
        bestN: Number(bestRow.n),
        bars: usable
          .slice()
          .sort((a, b) => Number(a.n) - Number(b.n))
          .map((r) => {
            const value = Number(r.calls_s);
            return {
              n: Number(r.n),
              value,
              share: best > 0 ? value / best : 0,
              stable: r.stable !== false,
              spreadPct: Number.isFinite(Number(r.spread_pct)) ? Number(r.spread_pct) : null,
              best: value === best,
            };
          }),
      };
    })
    .filter(Boolean);
}

/** Plain-language verdict for one stage: where it stops paying for more threads. */
export function scalingVerdict(group) {
  const top = group.bars.find((b) => b.best);
  if (!top) return 'No scaling data';
  const noisy = group.bars.some((b) => !b.stable);
  const tail = group.bars[group.bars.length - 1];
  const beyond = tail && tail.n > top.n;
  if (noisy) return `Best at ${top.n} threads; some runs were noisy, so treat the ranking as indicative`;
  if (beyond) return `Stops paying off past ${top.n} threads (${tail.n} was ${Math.round(tail.share * 100)}% of best)`;
  return `Best at ${top.n} threads`;
}
