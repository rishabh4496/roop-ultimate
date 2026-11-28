import React, { useState } from 'react';
import { RotateCcw, Merge, Loader2, Film, Download, Puzzle } from 'lucide-react';
import useSegmentGroups from '@engine/components/segments/useSegmentGroups';
import { canRerender, groupSummary, joinDisabledReason, verifyLabel, wornPill } from '@engine/components/segments/segmentModel';
import { Panel } from '../fx/effects';
import { notify } from '../ui/controls';

// ── Segments: a video rendered as N slices (N queue jobs whose files wait in .segments/<group>/) ──
// One pill per slice. A slice that finished or failed can be re-rendered on its own - the others are
// untouched - and the set is joined (stream-copy concat + source audio + a duration check) once
// every slice is complete. State, polling and actions are 1.0's useSegmentGroups / segmentModel.

const PILL_TONE = { queued: 'var(--fg-faint)', rendering: 'var(--accent-2)', completed: 'var(--ok)', failed: 'var(--danger)' };

function Group({ g, busy, rerender, join }) {
  const [sel, setSel] = useState(null);
  const seg = g.segments.find((s) => s.index === sel) || null;
  const joining = busy === `join:${g.group}`;
  const blocked = joinDisabledReason(g, joining);
  const verify = verifyLabel(g.joined?.result?.verify);
  return (
    <div className="rounded-xl border border-[var(--rule)] bg-black/25 p-2.5" data-testid="seg-group">
      <div className="flex items-start gap-2">
        <Film size={14} className="mt-0.5 shrink-0 text-[var(--accent-2)]" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[12.5px] font-semibold text-white" title={g.target_name}>{g.target_name || g.group}</div>
          <div className="font-mono text-[10.5px] text-[var(--fg-dim)]" data-testid="seg-summary">{groupSummary(g)}</div>
        </div>
      </div>
      <div className="mt-2 flex flex-wrap gap-1" role="group" aria-label="Slices">
        {g.segments.map((s) => {
          const worn = wornPill(s);
          return (
            <button key={s.job_id} type="button" onClick={() => setSel(sel === s.index ? null : s.index)} aria-pressed={sel === s.index}
              data-testid={`seg-pill-${s.index}`} data-state={worn.key}
              className={`chip !h-7 ${sel === s.index ? 'on' : ''}`} style={{ color: PILL_TONE[worn.key] }}
              title={`Slice ${s.index} · frames ${s.start_frame}-${s.end_frame} · ${worn.label}${worn.note ? ` · ${worn.note}` : ''}`}>
              {s.index} · {worn.label}{worn.key === 'rendering' && s.fraction > 0 ? ` ${Math.round(s.fraction * 100)}%` : ''}
            </button>
          );
        })}
      </div>
      {seg && (
        <div className="mt-2 flex flex-wrap items-center gap-2 rounded-lg border border-[var(--rule)] p-2 text-[11.5px]" data-testid="seg-detail">
          <span className="min-w-0 flex-1 truncate font-mono text-[var(--fg-dim)]" title={seg.name}>{seg.name} · frames {seg.start_frame}-{seg.end_frame}</span>
          {wornPill(seg).note && <span className="w-full text-[var(--danger)]">{wornPill(seg).note}</span>}
          <button type="button" className="btn btn-sm" disabled={!canRerender(seg) || !!busy} data-testid="seg-rerender"
            onClick={() => {
              if (!window.confirm(`Re-render slice ${seg.index} (frames ${seg.start_frame}-${seg.end_frame})?\n\nIt replaces ${seg.name} when it finishes. The other slices are not touched.`)) return;
              rerender(seg.job_id, `slice ${seg.index}`);
            }}>
            {busy === `rerender:${seg.job_id}` ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />} Re-render slice
          </button>
        </div>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button type="button" className="btn btn-primary btn-sm" disabled={!!blocked || !!busy} title={blocked || 'Join every slice into one video'} onClick={() => join(g.group)} data-testid="seg-join">
          {joining ? <Loader2 size={12} className="animate-spin" /> : <Merge size={12} />} {g.joined?.exists ? 'Re-join' : 'Join'}
        </button>
        {blocked && !joining && <span className="text-[11px] text-[var(--fg-faint)]">{blocked}</span>}
        {g.joined?.exists && g.joined.url && (
          <a className="btn btn-ghost btn-sm" href={g.joined.url} download data-testid="seg-joined"><Download size={12} /> {g.joined.name}</a>
        )}
      </div>
      {verify && (
        <div className="mt-1.5 text-[11px]" style={{ color: verify.ok === false ? 'var(--danger)' : verify.ok ? 'var(--ok)' : 'var(--fg-faint)' }} data-testid="seg-verify">{verify.text}</div>
      )}
    </div>
  );
}

export default function Segments({ running = false }) {
  const s = useSegmentGroups({ notify: (m, t) => notify(m, t === 'error' ? 'error' : 'success'), running });
  if (!s.loaded || !s.groups.length) return null;
  return (
    <Panel title="Split renders" accent={`${s.groups.length}`} data-testid="segments">
      <p className="mb-2 flex items-start gap-1.5 text-[11.5px] text-[var(--fg-dim)]"><Puzzle size={13} className="mt-0.5 shrink-0" /> A long video rendered in slices. Redo one bad slice without redoing the rest, then join.</p>
      <div className="space-y-2">
        {s.groups.map((g) => <Group key={g.group} g={g} busy={s.busy} rerender={s.rerender} join={s.join} />)}
      </div>
    </Panel>
  );
}
