import React from 'react';
import { Button, Section } from '../ui';
import { batchCellKey, cellSelected, estimateBatch, RECIPE_FIELDS } from './batchPlan';
import { jobState } from './useQueue';

const STATUS_TONES = {
  Planned: 'bg-accent/15 text-accent border-accent/35',
  Skipped: 'bg-white/[0.04] text-white/60 border-white/10',
  Staged: 'bg-sky-400/10 text-sky-300 border-sky-400/30',
  Queued: 'bg-amber-400/10 text-amber-300 border-amber-400/30',
  Rendering: 'bg-amber-400/10 text-amber-300 border-amber-400/30',
  Rendered: 'bg-emerald-400/10 text-emerald-300 border-emerald-400/30',
};

export function RecipeCards({ recipes, baseline, meta, setRecipes, targetIndex, setTargetIndex, targets }) {
  return (
    <Section title="Render recipes">
      <label className="flex flex-wrap items-center gap-2 text-sm text-muted">
        Comparison clip
        <select value={targetIndex} onChange={(e) => setTargetIndex(Number(e.target.value))}
          className="min-w-0 rounded-lg border border-white/15 bg-[#18181d] px-2 py-1 text-white">
          {targets.map((target, i) => <option key={target.media_id || i} value={i}>{target.name}</option>)}
        </select>
      </label>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {recipes.map((recipe) => {
          const patch = recipe.patch || {};
          const diffs = RECIPE_FIELDS.filter(([key]) => patch[key] != null && patch[key] !== baseline[key]);
          return (
            <article key={recipe.id} className="min-w-0 rounded-xl border border-white/10 bg-white/[0.03] p-3">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-white">{recipe.label}</h3>
                {recipe.id !== 'baseline' && <Button variant="ghost" onClick={() => setRecipes((list) => list.filter((item) => item.id !== recipe.id))} aria-label={`Remove ${recipe.label}`}>Remove</Button>}
              </div>
              {recipe.id === 'baseline' ? (
                <p className="mt-2 text-xs text-muted">{RECIPE_FIELDS.map(([key]) => baseline[key]).filter(Boolean).join(' · ')}</p>
              ) : (
                <>
                  <div className="mt-2 flex flex-wrap gap-1.5" aria-label={`${recipe.label} differences`}>
                    {diffs.length === 0 && <span className="text-xs text-muted">Matches baseline</span>}
                    {diffs.map(([key, label]) => (
                      <span key={key} className="rounded-md border border-accent/35 bg-accent/10 px-2 py-1 text-xs text-accent">
                        {label} · {baseline[key]} → {patch[key]}
                      </span>
                    ))}
                  </div>
                  <div className="mt-3 grid gap-2">
                    {RECIPE_FIELDS.map(([key, label, catalogue]) => (
                      <label key={key} className="grid grid-cols-[5rem_minmax(0,1fr)] items-center gap-2 text-xs text-muted">
                        {label}
                        <select aria-label={`${recipe.label} ${label}`} value={patch[key] || ''}
                          onChange={(e) => {
                            const value = e.target.value;
                            setRecipes((list) => list.map((item) => item.id === recipe.id
                              ? { ...item, patch: { ...item.patch, [key]: value || undefined } } : item));
                          }}
                          className="min-w-0 rounded border border-white/15 bg-[#18181d] px-2 py-1 text-white">
                          <option value="">Use baseline</option>
                          {[...new Set([...(meta?.[catalogue] || []), patch[key]].filter(Boolean))]
                            .map((value) => <option key={value} value={value}>{value}</option>)}
                        </select>
                      </label>
                    ))}
                  </div>
                </>
              )}
            </article>
          );
        })}
      </div>
    </Section>
  );
}

function Timeline({ columns, jobs, keyframes }) {
  const byMedia = new Map();
  const selectedRanges = new Set(jobs.map((job) => JSON.stringify([
    job.target_media_id || job.target_name, job.frame_start, job.frame_end,
  ])));
  for (const column of columns) {
    if (column.count < 2 && !(column.target.fps > 0 && column.target.frames > 1)) continue;
    const key = JSON.stringify([column.mediaKey, column.start, column.end]);
    if (!byMedia.has(column.mediaKey)) byMedia.set(column.mediaKey, new Map());
    byMedia.get(column.mediaKey).set(key, column);
  }
  return [...byMedia.entries()].map(([mediaKey, ranges]) => {
    const slices = [...ranges.values()].sort((a, b) => a.start - b.start);
    const { target } = slices[0];
    return (
      <div key={mediaKey} className="space-y-1.5" data-testid="batch-split-timeline">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 text-xs">
          <span className="break-all font-semibold text-white">{target.name}</span>
          <span className="text-muted">{slices.length} {slices.length === 1 ? 'range' : 'ranges'} · {keyframes[mediaKey] ? 'GOP aligned' : 'Duration based'}</span>
        </div>
        <div className="flex h-8 gap-px overflow-hidden rounded-lg border border-white/15 bg-black/30" aria-label={`${target.name} split timeline`}>
          {[...ranges].sort(([, a], [, b]) => a.start - b.start).map(([key, slice], i) => (
            <div key={key} style={{ flex: `${slice.end - slice.start + 1} 1 0%` }}
              className={`flex min-w-0 items-center justify-center overflow-hidden text-xs font-semibold ${selectedRanges.has(key) ? 'bg-accent/25 text-accent' : 'bg-white/10 text-muted'}`}
              title={`Chunk ${i + 1} · frames ${slice.start}–${slice.end} · ${((slice.end - slice.start + 1) / target.fps).toFixed(2)}s`}>
              {i + 1}
            </div>
          ))}
        </div>
        <p className="text-xs tabular-nums text-muted">
          {slices.map(({ start, end }) => `${start}–${end}`).join(' · ')}
        </p>
      </div>
    );
  });
}

const serverStatus = (job) => {
  const state = jobState(job);
  if (state === 'COMPLETED') return 'Rendered';
  if (['PROCESSING', 'PAUSE_REQUESTED', 'PAUSED'].includes(state)) return 'Rendering';
  if (['QUEUED', 'PENDING'].includes(state)) return 'Queued';
  return null;
};

export default function BatchFlow({
  columns, sources, selections, setSelections, jobs, stagedJobs, queueJobs, stage,
  split, setSplit, splitStatus, inspectKeyframes, onUseDuration, keyframes, recipeValues,
}) {
  const statusByCell = new Map();
  const addStatus = (job, status) => {
    if (!status) return;
    const marker = job.payload?._batch_matrix;
    if (!Array.isArray(marker?.cells)) return;
    for (const key of marker.cells) statusByCell.set(JSON.stringify([key, marker.recipe]), status);
  };
  for (const job of queueJobs) if (serverStatus(job) === 'Rendered') addStatus(job, 'Rendered');
  for (const job of stagedJobs) addStatus(job, 'Staged');
  for (const job of queueJobs) if (serverStatus(job) !== 'Rendered') addStatus(job, serverStatus(job));
  const recipeByColumn = new Map(columns.map((column) => [column.key, recipeValues(column)]));
  const hasSplittableVideo = columns.some(({ target }) => target.fps > 0 && target.frames > 1);
  const hasGopRanges = columns.some(({ mediaKey }) => keyframes[mediaKey]);
  const inspectingGop = splitStatus === 'Inspecting GOP boundaries…';

  const toggle = (pairs) => setSelections((current) => {
    const next = { ...current };
    const enabled = !pairs.every(([column, source]) => cellSelected(column, source, current));
    for (const [column, source] of pairs) next[batchCellKey(column, source)] = enabled;
    return next;
  });
  const estimate = estimateBatch(jobs);

  return (
    <Section title="Batch flow preview">
      <div className="space-y-4" data-testid="batch-flow">
        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-3">
          <label className="flex items-center gap-2 text-sm text-white">
            <input type="checkbox" checked={split.enabled} onChange={(e) => setSplit({ ...split, enabled: e.target.checked })} />
            Automatically split video
          </label>
          {split.enabled && (
            <>
              <label className="flex items-center gap-2 text-xs text-muted">
                Target duration
                <input type="number" min="1" max="3600" value={split.seconds}
                  onChange={(e) => setSplit({ ...split, seconds: e.target.value })}
                  onBlur={() => setSplit((current) => ({ ...current, seconds: Math.min(3600, Math.max(1, Number(current.seconds) || 1)) }))}
                  className="w-20 rounded border border-white/15 bg-[#18181d] px-2 py-1 text-white" />
                sec
              </label>
              <Button variant="secondary" onClick={inspectKeyframes} disabled={inspectingGop || !hasSplittableVideo}>
                Align to GOP boundaries
              </Button>
              {hasGopRanges && <Button variant="ghost" disabled={inspectingGop} onClick={onUseDuration}>Use duration boundaries</Button>}
              <span className="text-xs text-muted">Up to 32 chunks per clip</span>
              {splitStatus && <span role="status" className="text-xs text-muted">{splitStatus}</span>}
            </>
          )}
        </div>
        {split.enabled && <Timeline columns={columns} jobs={jobs} keyframes={keyframes} />}
        {sources.length > 0 && columns.length > 0 ? (
          <div className="max-h-[32rem] overflow-auto rounded-xl border border-white/10" role="region" aria-label="Batch matrix">
            <table className="w-full border-separate border-spacing-0 text-left text-xs" aria-label="Source facesets by target, segment, or recipe">
              <thead className="sticky top-0 z-20 bg-[#18181d]">
                <tr>
                  <th scope="col" className="sticky left-0 z-10 min-w-40 border-b border-white/10 bg-[#18181d] p-3 text-muted">Source facesets</th>
                  {columns.map((column) => (
                    <th key={column.key} scope="col" className="min-w-36 border-b border-white/10 p-2">
                      <button type="button" onClick={() => toggle(sources.map((source) => [column, source]))}
                        className="w-full rounded-lg px-2 py-1.5 text-left text-white hover:bg-white/10"
                        aria-label={`Toggle column ${column.title ? `${column.title} · ` : ''}${column.target.name}${column.count > 1 ? ` chunk ${column.part}` : ''}`}>
                        {column.title && !column.recipe && <span className="block text-xs text-muted">{column.title}</span>}
                        <span className="block break-all">{column.label}</span>
                        {column.count > 1 && <span className="block text-muted">Chunk {column.part} · {column.start}–{column.end}</span>}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sources.map((source) => (
                  <tr key={source.id}>
                    <th scope="row" className="sticky left-0 z-10 border-t border-white/10 bg-[#18181d] px-2 py-1">
                      <button type="button" onClick={() => toggle(columns.map((column) => [column, source]))}
                        aria-label={`Toggle row ${source.name}`} className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-white hover:bg-white/10">
                        {source.thumbnail && <img src={source.thumbnail} alt="" loading="lazy" className="size-8 shrink-0 rounded-md object-cover" />}
                        <span className="max-w-36 break-all">{source.name}</span>
                      </button>
                    </th>
                    {columns.map((column) => {
                      const key = batchCellKey(column, source);
                      const selected = cellSelected(column, source, selections);
                      const status = statusByCell.get(JSON.stringify([key, recipeByColumn.get(column.key)])) || (selected ? 'Planned' : 'Skipped');
                      return (
                        <td key={column.key} className="border-t border-white/10 p-2">
                          <button type="button" aria-pressed={selected} data-state={status.toLowerCase()}
                            aria-label={`${source.name} × ${column.label}${column.count > 1 ? ` chunk ${column.part}` : ''}: ${status}`}
                            onClick={() => toggle([[column, source]])}
                            className={`w-full rounded-lg border px-2 py-1.5 text-left font-medium hover:brightness-125 ${STATUS_TONES[status]}`}>
                            <span aria-hidden="true">{selected ? '✓' : '–'}</span> {status}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="text-sm text-muted">Load source facesets and assign target media to see their combinations.</p>}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p aria-live="polite" className="text-sm text-muted">
            <strong className="text-white">{jobs.length.toLocaleString()} {jobs.length === 1 ? 'job' : 'jobs'}</strong>
            {' · '}{estimate.totalFrames.toLocaleString()} frames · ~{estimate.timeStr} render time · ~{estimate.megabytes.toLocaleString()} MB storage
          </p>
          <Button variant="primary" disabled={!jobs.length || (split.enabled && inspectingGop)} onClick={stage}>Stage selected jobs</Button>
        </div>
        {jobs.length > 0 && <p className="text-xs text-muted">Estimates use 35 ms/frame without enhancement, 115 ms/frame with enhancement, and 0.25 MB/frame for chunks and joined outputs.</p>}
      </div>
    </Section>
  );
}
