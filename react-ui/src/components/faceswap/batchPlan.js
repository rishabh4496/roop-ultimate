import { newStagedJobId, stageOneToMany } from './batchMatrix.js';

export const RECIPE_FIELDS = [
  ['swap_model', 'Swap model', 'swap_models'],
  ['enhancer', 'Enhancer', 'enhancers'],
  ['mask_engine', 'Mask', 'mask_engines'],
];

export function splitRanges(start, end, fps, seconds, keyframes = null) {
  if (!(fps > 0 && seconds > 0)) return [[start, end]];
  const step = Math.max(1, Math.round(fps * seconds), Math.ceil((end - start + 1) / 32));
  const boundaries = [start];
  let key = 0;
  for (let desired = start + step; desired <= end; desired += step) {
    let cut = desired;
    if (keyframes) {
      while (key < keyframes.length && keyframes[key] < desired) key++;
      cut = keyframes[key];
    }
    if (cut > boundaries.at(-1) && cut <= end) boundaries.push(cut);
  }
  return boundaries.map((first, i) => [first, (boundaries[i + 1] ?? end + 1) - 1]);
}

export function buildBatchColumns({
  mode, targets, selectedTargets, matrixConfig, groups, recipes, recipeTarget,
  oneToMany, splitSeconds, keyframes,
}) {
  const targetKey = (index) => targets[index]?.media_id || targets[index]?.name;
  const entries = mode === 'grouped'
    ? groups.flatMap((group) => group.targetIndices.map((index) => ({
      ...group, index, scope: `group:${group.id}`, title: group.label, enabled: true,
    })))
    : mode === 'recipes'
      ? recipes.map((recipe) => ({
        index: recipeTarget, scope: `recipe:${recipe.id}`, title: recipe.label, recipe,
        mappings: [], enabled: true,
      }))
      : targets.map((_, index) => ({
        ...(mode === 'matrix' ? matrixConfig[index] : oneToMany),
        enabled: mode === 'matrix' ? !!matrixConfig[index]?.enabled : selectedTargets.includes(index),
        index, scope: mode,
      }));

  return entries.flatMap((entry) => {
    const target = targets[entry.index];
    if (!target) return [];
    const total = Math.max(1, target.frames || 1);
    // API target start_frame is 0-based; the queue accepts 1-based inclusive
    // ranges. Explicit per-file edits already use the queue convention.
    const start = Math.max(1, Math.min(total, entry.frameStart ?? ((target.start_frame || 0) + 1)));
    const end = Math.max(start, Math.min(total, entry.frameEnd || target.end_frame || total));
    const mediaKey = targetKey(entry.index);
    const family = JSON.stringify([entry.scope, mediaKey]);
    const wholeRangeKey = JSON.stringify([family, start, end]);
    const ranges = splitRanges(start, end, target.fps, splitSeconds, keyframes[mediaKey]);
    return ranges.map(([first, last], i) => ({
      ...entry, target, mediaKey, family, start: first, end: last, part: i + 1, count: ranges.length,
      key: JSON.stringify([family, first, last]),
      wholeRangeKey,
      label: entry.recipe ? entry.title : target.name,
    }));
  });
}

export function batchCellKey(column, source) {
  return JSON.stringify([column.key, source.id]);
}

export function cellSelected(column, source, selections) {
  const defaultSelected = column.enabled && (
    column.recipe || column.mappings?.some((mapping) => Number(mapping.sourceIdx) === source.index)
  );
  return selections[batchCellKey(column, source)]
    ?? selections[JSON.stringify([column.wholeRangeKey, source.id])]
    ?? !!defaultSelected;
}

export function planBatchJobs({ columns, sources, selections, targets, sourceFacesInfo, createJobPayload }) {
  const families = new Map();
  const planned = [];
  for (const column of columns) {
    const selected = sources.filter((source) => cellSelected(column, source, selections));
    const configured = (column.mappings || []).filter(
      (mapping) => selected.some((source) => source.index === Number(mapping.sourceIdx)),
    );
    const configuredSources = new Set(configured.map((mapping) => Number(mapping.sourceIdx)));
    const assignments = selected.filter((source) => !configuredSources.has(source.index))
      .map((source) => [{ personRank: column.mappings?.[0]?.personRank ?? 0, sourceIdx: source.index }]);
    if (configured.length) assignments.unshift(configured);

    for (const mappings of assignments) {
      const { jobs: [job] } = stageOneToMany({
        targets, selectedTargets: [column.index], mappings, swapMode: column.swapMode,
        enhancer: column.enhancer, faceDistance: column.faceDistance,
        sourceCount: sources.length, sourceFacesInfo, createJobPayload,
      });
      const cells = selected.filter((source) => mappings.some((m) => Number(m.sourceIdx) === source.index))
        .map((source) => batchCellKey(column, source));
      const payload = job.payload;
      if (column.recipe) {
        for (const [field] of RECIPE_FIELDS) {
          const value = column.recipe.patch?.[field];
          if (value != null) payload[field] = value;
        }
        payload.selected_enhancer = payload.enhancer;
        payload.auto_fallback = false;
      }
      // These keys travel with the job so status survives refreshing the page.
      // Recipe settings are included so edited variants don't inherit a render.
      payload._batch_matrix = {
        cells,
        recipe: RECIPE_FIELDS.map(([field]) => payload[field] ?? null),
      };
      Object.assign(job, {
        target_media_id: column.target.media_id,
        frame_start: column.start, frame_end: column.end, total_frames: column.end - column.start + 1,
        label: [column.title, column.target.name, job.source_name,
          column.count > 1 ? `Chunk ${column.part}/${column.count}` : null].filter(Boolean).join(' · '),
        payload,
      });
      const familyKey = JSON.stringify([column.family, mappings]);
      if (!families.has(familyKey)) families.set(familyKey, []);
      families.get(familyKey).push({ job, column });
      planned.push(job);
    }
  }

  for (const series of families.values()) {
    const [{ column }] = series;
    if (column.count < 2) continue;
    const group = newStagedJobId();
    for (const { job, column: slice } of series) {
      job.segment = { group, index: slice.part, count: slice.count };
    }
  }
  return planned;
}

export function estimateBatch(jobs) {
  let totalFrames = 0;
  let seconds = 0;
  const splits = new Map();
  for (const job of jobs) {
    const frames = job.total_frames || 1;
    totalFrames += frames;
    seconds += frames * (job.payload?.enhancer && job.payload.enhancer !== 'None' ? 0.115 : 0.035);
    if (job.segment) {
      const { group, count, index } = job.segment;
      if (!splits.has(group)) splits.set(group, { count, parts: new Map() });
      splits.get(group).parts.set(index, frames);
    }
  }
  let storedFrames = totalFrames;
  for (const { count, parts } of splits.values()) {
    if (parts.size === count) storedFrames += [...parts.values()].reduce((sum, frames) => sum + frames, 0);
  }
  const rounded = Math.round(seconds);
  const mins = Math.floor(rounded / 60);
  const secs = rounded % 60;
  return { totalFrames, timeStr: mins ? `${mins}m ${secs}s` : `${secs}s`, megabytes: storedFrames * 0.25 };
}
