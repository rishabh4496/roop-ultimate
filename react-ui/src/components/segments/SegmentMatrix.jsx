import React, { useState } from 'react';
import { API } from '../../api';
import { Button, Section } from '../ui';
import { confirmDialog } from '../confirm';
import useSegmentGroups from './useSegmentGroups';
import { PILLS, canRerender, groupSummary, joinDisabledReason, verifyLabel, wornPill } from './segmentModel';
import { Icon } from '../../icons';

// Segment Chunk Matrix.
//
// A video rendered in segments is N queue jobs whose finished files wait in `.segments/<split>/` until
// they are joined. This is the custody view of that set: one pill per chunk (Queued / Rendering /
// Completed / Failed, and "Missing" when a chunk finished but its file is no longer on disk), a click on
// any pill to inspect it and re-render just that chunk, and a single Join action that stitches the set
// with a lossless stream copy, carries the source's own audio stream when the chunks cover the whole
// source, and checks that the joined video lasts as long as it should.

const TONE = {
  muted: 'border-white/15 bg-white/[0.04] text-white/60',
  amber: 'border-amber-400/40 bg-amber-400/10 text-amber-300',
  green: 'border-emerald-400/40 bg-emerald-400/10 text-emerald-300',
  red: 'border-red-400/40 bg-red-400/10 text-red-300',
};

function Pill({ seg, selected, onClick }) {
  const worn = wornPill(seg);
  const tone = TONE[PILLS[worn.key].tone];
  return (
    <button
      type="button" onClick={onClick} aria-pressed={selected}
      data-testid={`seg-pill-${seg.index}`} data-state={worn.key}
      title={`Chunk ${seg.index} · frames ${seg.start_frame}-${seg.end_frame} · ${worn.label}${worn.note ? ` · ${worn.note}` : ''}`}
      className={`relative rounded-lg border px-2.5 py-1.5 text-left transition-colors cursor-pointer ${tone} ${selected ? 'ring-2 ring-[var(--accent)]' : 'hover:brightness-125'}`}
    >
      <span className="block text-nano font-semibold uppercase tracking-wider opacity-80">Chunk {seg.index}</span>
      <span className="flex items-center gap-1.5 text-mini font-bold">
        {worn.key === 'rendering' && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" aria-hidden="true" />}
        {worn.label}
        {worn.key === 'rendering' && seg.fraction > 0 && <span className="font-mono tabular-nums opacity-80">{Math.round(seg.fraction * 100)}%</span>}
      </span>
    </button>
  );
}

function Group({ g, busy, onRerender, onJoin }) {
  const [sel, setSel] = useState(null);
  const [viewMode, setViewMode] = useState('pills'); // 'pills' | 'table'
  const seg = g.segments.find((s) => s.index === sel) || null;
  const joining = busy === `join:${g.group}`;
  const blocked = joinDisabledReason(g, joining);
  const j = g.joined;
  const v = verifyLabel(j?.result?.verify);
  const failedJoin = j?.result && j.result.ok === false;

  const rerender = async (s) => {
    const done = canRerender(s) && wornPill(s).key === 'completed';
    if (done && !(await confirmDialog({
      title: `Re-render chunk ${s.index}?`,
      message: `Replaces ${s.name} when it finishes (frames ${s.start_frame}-${s.end_frame}). The other chunks are not touched, and the set is re-joined once all are complete again.`,
      confirmLabel: 'Re-render chunk',
    }))) return;
    onRerender(s.job_id, `chunk ${s.index}`);
  };

  const completedCount = g.segments.filter((s) => wornPill(s).key === 'completed').length;
  const totalChunks = g.count || g.segments.length;
  const allComplete = completedCount === totalChunks && totalChunks > 0;

  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-3.5 space-y-3" data-testid={`seg-group-${g.group}`}>
      {/* Group Header */}
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-white/5 pb-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h4 className="m-0 truncate text-sm font-bold text-white/90" title={g.target_name}>
              {g.target_name || 'Segmented render'}
            </h4>
            <span className={`px-2 py-0.5 rounded text-micro font-mono font-bold border ${
              allComplete
                ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
                : 'bg-amber-500/15 border-amber-500/30 text-amber-300'
            }`}>
              {completedCount} / {totalChunks} Chunks Ready
            </span>
          </div>
          <p className="m-0 text-mini text-muted" data-testid="seg-summary">{groupSummary(g)}</p>
        </div>
        <div className="flex items-center gap-2">
          {/* View Mode Toggle */}
          <div className="inline-flex rounded-lg border border-white/10 bg-black/40 p-0.5 text-nano font-mono">
            <button
              type="button"
              onClick={() => setViewMode('pills')}
              className={`px-2 py-0.5 rounded-md font-bold transition-colors ${
                viewMode === 'pills' ? 'bg-accent/20 text-accent border border-accent/30' : 'text-white/60 hover:text-white'
              }`}
            >
              Pills
            </button>
            <button
              type="button"
              onClick={() => setViewMode('table')}
              className={`px-2 py-0.5 rounded-md font-bold transition-colors ${
                viewMode === 'table' ? 'bg-accent/20 text-accent border border-accent/30' : 'text-white/60 hover:text-white'
              }`}
            >
              Matrix Table
            </button>
          </div>
          <span className="font-mono text-nano text-muted" title="the split's id; its chunks live in .segments/<id>/">
            {g.group.length > 22 ? `${g.group.slice(0, 20)}…` : g.group}
          </span>
        </div>
      </div>

      {/* Pills View */}
      {viewMode === 'pills' ? (
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Chunks">
          {g.segments.map((s) => (
            <Pill key={s.job_id} seg={s} selected={sel === s.index} onClick={() => setSel(sel === s.index ? null : s.index)} />
          ))}
        </div>
      ) : (
        /* Detailed Segment Matrix Table */
        <div className="overflow-x-auto rounded-lg border border-white/10 bg-black/30">
          <table className="w-full text-left font-mono text-mini">
            <thead>
              <tr className="border-b border-white/10 bg-white/[0.02] text-nano uppercase text-white/50">
                <th className="p-2">Chunk</th>
                <th className="p-2">Frame Range</th>
                <th className="p-2">Frames</th>
                <th className="p-2">Status</th>
                <th className="p-2">Progress</th>
                <th className="p-2">Disk Custody</th>
                <th className="p-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {g.segments.map((s) => {
                const worn = wornPill(s);
                const tone = TONE[PILLS[worn.key].tone];
                return (
                  <tr key={s.job_id} className={`hover:bg-white/[0.03] transition-colors ${sel === s.index ? 'bg-accent/10' : ''}`}>
                    <td className="p-2 font-bold text-white">Chunk {s.index}</td>
                    <td className="p-2 text-white/80">{s.start_frame} – {s.end_frame}</td>
                    <td className="p-2 text-white/60">{s.frames}f</td>
                    <td className="p-2">
                      <span className={`px-1.5 py-0.5 rounded text-micro border font-bold ${tone}`}>
                        {worn.label}
                      </span>
                    </td>
                    <td className="p-2">
                      {worn.key === 'rendering' ? (
                        <div className="w-20 bg-white/10 rounded-full h-1.5 overflow-hidden">
                          <div className="bg-amber-400 h-full transition-all duration-300" style={{ width: `${Math.round(s.fraction * 100)}%` }} />
                        </div>
                      ) : worn.key === 'completed' ? (
                        <span className="text-emerald-400">100%</span>
                      ) : (
                        <span className="text-white/30">—</span>
                      )}
                    </td>
                    <td className="p-2 text-micro">
                      {s.exists ? (
                        <span className="text-emerald-400/90 font-bold">{(s.bytes / 1048576).toFixed(1)} MB</span>
                      ) : (
                        <span className="text-amber-400/80">Pending</span>
                      )}
                    </td>
                    <td className="p-2 text-right">
                      <button
                        type="button"
                        onClick={() => rerender(s)}
                        disabled={!canRerender(s) || !!busy}
                        className="px-2 py-0.5 rounded text-micro font-bold border border-white/15 bg-white/5 hover:bg-white/15 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      >
                        Re-Render
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Selected Chunk Inspector */}
      {seg && (
        <div className="rounded-lg border border-white/10 bg-white/[0.03] p-2.5 flex flex-wrap items-center justify-between gap-2" data-testid="seg-detail">
          <div className="min-w-0 text-mini">
            <div className="font-mono text-white/90 truncate flex items-center gap-2">
              <span className="font-bold text-accent">Chunk {seg.index}:</span>
              <span>{seg.name}</span>
            </div>
            <div className="text-muted">
              frames {seg.start_frame}-{seg.end_frame} ({seg.frames} frames) · {seg.exists ? `${(seg.bytes / 1048576).toFixed(1)} MB on disk` : 'no file on disk'}
              {wornPill(seg).note ? ` · ${wornPill(seg).note}` : ''}
            </div>
          </div>
          <Button size="sm" variant="secondary" onClick={() => rerender(seg)} disabled={!canRerender(seg) || !!busy}
                  data-testid="seg-rerender"
                  title={canRerender(seg) ? 'Render only this chunk again' : 'A chunk can be re-rendered once it has finished or failed'}>
            {busy === `rerender:${seg.job_id}` ? 'Queuing…' : 'Re-Render Chunk'}
          </Button>
        </div>
      )}

      {/* Clip-Join Actions Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-white/5">
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant="primary" onClick={() => onJoin(g.group)} disabled={!!blocked} data-testid="seg-join"
                  title={blocked || 'Lossless concat of the chunks, the source audio stream copied when they cover the whole source, then a duration check'}>
            <span className="flex items-center gap-1.5">
              <Icon.play size={12} />
              <span>{joining ? 'Joining Chunks…' : 'Join Segments & Remux Audio Now'}</span>
            </span>
          </Button>
          {blocked && !joining && <span className="text-mini text-muted font-mono" data-testid="seg-join-blocked">{blocked}</span>}
          {j?.exists && (
            <a href={`${API}${j.url}`} target="_blank" rel="noreferrer" className="text-mini text-accent underline flex items-center gap-1 font-mono font-bold" data-testid="seg-joined-link">
              <Icon.download size={12} />
              <span>Open {j.name}</span>
            </a>
          )}
        </div>

        <div className="text-nano font-mono text-white/50">
          Custody: <span className="text-white/80">.segments/{g.group.slice(0, 14)}…</span>
        </div>
      </div>

      {/* Verification status readout */}
      {v && (
        <p className={`m-0 text-mini font-mono ${v.ok === false ? 'text-red-300' : v.ok ? 'text-emerald-300' : 'text-muted'}`} role={v.ok === false ? 'alert' : 'status'} data-testid="seg-verify">
          {v.ok === false ? '✗ ' : v.ok ? '✓ ' : ''}{v.text}
          {j.result?.source_audio && j.result.audio === 'copy' ? ' · source audio stream copied' : j.result?.source_audio ? ' · source audio re-encoded to AAC' : ' · chunk audio carried'}
        </p>
      )}
      {failedJoin && <p className="m-0 text-mini text-red-300 font-mono" role="alert" data-testid="seg-join-error">Join failed: {j.result.error}. The chunks are untouched.</p>}
    </div>
  );
}

export default function SegmentMatrix({ notify, running = false, title = 'Segment Chunk Matrix' }) {
  const { groups, loaded, busy, rerender, join } = useSegmentGroups({ notify, running });
  if (!loaded || groups.length === 0) return null; // nothing to look after: no empty box
  return (
    <Section title={title}>
      <div className="space-y-3" data-testid="segment-matrix">
        {groups.map((g) => <Group key={g.group} g={g} busy={busy} onRerender={rerender} onJoin={join} />)}
      </div>
    </Section>
  );
}
