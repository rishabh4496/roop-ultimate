import React from 'react';
import { armCards, backendRows, matrixNote, stageRows, unmeasuredRows } from './providerMatrix';

// The provider-matrix result as markup. Pure (props in, markup out) and free of any API import, so the render
// check can drive it with a real backend result.

export function ProviderMatrixView({ matrix }) {
  const stages = stageRows(matrix);
  const arms = armCards(matrix);
  const warp = backendRows(matrix && matrix.warp_blend);
  const flow = backendRows(matrix && matrix.optical_flow);
  const missing = unmeasuredRows(matrix);
  if (!stages.length && !arms.length) return null;
  return (
    <div className="mt-3 space-y-3" data-testid="provider-matrix">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
        {arms.map((a) => (
          <div key={a.key} data-arm={a.key} data-available={a.available ? 'true' : 'false'}
               className={`rounded-xl border px-3 py-2 min-w-0 ${a.fastest ? 'border-emerald-400/40 bg-emerald-400/[0.05]' : 'border-white/[0.08] bg-black/40'}`}>
            <div className="text-nano font-bold uppercase tracking-[0.12em] text-white/70 truncate" title={a.label}>{a.label}</div>
            <div className={`mt-1 font-mono text-sm ${a.available ? 'text-white/90' : 'text-muted'}`}>{a.value}</div>
            <div className="mt-0.5 text-nano text-muted leading-snug">{a.sub}</div>
          </div>
        ))}
      </div>

      {missing.length > 0 && (
        <ul className="space-y-0.5" data-testid="provider-unmeasured">
          {missing.map((u) => (
            <li key={u.stage} className="text-micro text-amber-300">
              Not measured: {u.stage} — {u.reason}
            </li>
          ))}
        </ul>
      )}

      {stages.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-micro font-mono">
            <caption className="sr-only">Milliseconds per call for each pipeline stage, by provider</caption>
            <thead>
              <tr className="text-left text-nano uppercase tracking-wider text-muted">
                <th className="py-1 pr-3 font-semibold">Stage</th>
                <th className="py-1 pr-3 font-semibold text-right">Calls / frame</th>
                <th className="py-1 pr-3 font-semibold text-right">Primary</th>
                <th className="py-1 pr-3 font-semibold text-right">CPU</th>
                <th className="py-1 font-semibold text-right">OpenVINO</th>
              </tr>
            </thead>
            <tbody>
              {stages.map((r) => (
                <tr key={r.key} className="border-t border-white/5" data-stage-row={r.key}>
                  <td className="py-1 pr-3 text-white/80 max-w-[18rem] truncate" title={r.note || r.stage}>{r.stage}</td>
                  <td className="py-1 pr-3 text-right text-muted">{r.callsPerFrame ?? '—'}</td>
                  <td className={`py-1 pr-3 text-right ${r.fastest === 'primary' ? 'text-emerald-300' : 'text-white/80'}`} title={r.primaryProvider}>{r.primary}</td>
                  <td className={`py-1 pr-3 text-right ${r.fastest === 'cpu' ? 'text-emerald-300' : 'text-white/80'}`}>{r.cpu}</td>
                  <td className={`py-1 text-right ${r.fastest === 'openvino' ? 'text-emerald-300' : 'text-white/80'}`} title={r.openvinoDevice}>{r.openvino}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {[['Warp / blend', warp, matrix && matrix.warp_blend && matrix.warp_blend.workload],
        ['Optical flow', flow, matrix && matrix.optical_flow && matrix.optical_flow.workload]].map(([title, rows, workload]) => (
          rows.length > 0 && (
            <div key={title}>
              <div className="text-nano font-semibold uppercase tracking-wider text-muted">{title}{workload ? ` · ${workload}` : ''}</div>
              <ul className="mt-1 space-y-0.5">
                {rows.map((b) => (
                  <li key={b.key} className="flex items-baseline justify-between gap-3 text-micro">
                    <span className="text-white/80">{b.label}</span>
                    <span className={`font-mono text-right ${b.available ? 'text-white/90' : 'text-muted'}`} title={b.reason}>
                      {b.value}{!b.available && b.reason ? ` — ${b.reason}` : ''}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )
        ))}

      {matrixNote(matrix) && <p className="text-nano text-muted leading-snug">{matrixNote(matrix)}</p>}
    </div>
  );
}

export default ProviderMatrixView;
