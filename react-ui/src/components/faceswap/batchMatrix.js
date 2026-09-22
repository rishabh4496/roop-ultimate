// The Batch Matrix's staging logic, kept free of React so it can be run as
// code by `.render-check/batch-matrix-check.mjs` (and, through it, by
// app/tests/test_batch_matrix_queue.py) instead of being re-implemented in a
// test. BatchSwap.jsx owns the state and the toasts; everything that decides
// WHICH (target, person -> faceset) pairs become jobs lives here.
//
// All four strategies converge on one shape: a staged job is one target x one
// person->faceset mapping, and `queueRequestFromStagedJobs` turns the list
// into the body of POST /api/queue/add_batch. docs/development/
// BATCH_MATRIX_DATA_FLOW.md is the contract these functions implement.
import { FACESWAP_DEFAULTS } from './defaults.js';
import {
  normalizeSourceMapping,
  normalizeTargetSelectionState,
  SKIP,
} from './faceMapping.js';

export const DEFAULT_ENHANCER = 'Restoreformer++';
export const DEFAULT_FACE_DISTANCE = 0.75;
export const DEFAULT_SWAP_MODE = 'Selected face';

export const ERR_NO_SOURCES = 'Add a source faceset first';
export const ERR_NO_TARGETS = 'No target files available';
export const ERR_NO_SELECTED_TARGETS = 'Select at least one target file first';
export const ERR_NO_GROUP_TARGETS = 'No target files assigned to any group';
export const ERR_NO_MATRIX_FILES = 'No files enabled for matrix batch';
export const ERR_RECIPE_INPUTS = 'Requires target files and source facesets';
export const ERR_SPLIT_NO_TARGET = 'Selected target file does not exist';
export const ERR_SPLIT_SINGLE_FRAME = 'Target file is a single image or has no frames to split';

export function newStagedJobId() {
  return `staged_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
}

// The default per-target row of the per-file matrix, as refreshBackendState
// seeds it for every loaded target that has no row yet.
export function defaultMatrixRow(target, settings = {}) {
  return {
    mappings: [{ personRank: 0, sourceIdx: 0 }],
    swapMode: DEFAULT_SWAP_MODE,
    enabled: true,
    enhancer: settings.selected_enhancer || DEFAULT_ENHANCER,
    faceDistance: parseFloat(settings.max_face_distance || DEFAULT_FACE_DISTANCE),
    frameStart: target.start_frame || 1,
    frameEnd: target.end_frame || target.frames || 1,
  };
}

// One (target, mapping) pair -> the /api/swap body the queue will replay.
// `mappings` is the UI's [{personRank, sourceIdx}] list; the wire form is a
// DENSE face_mapping indexed by person rank where a missing rank is an
// explicit SKIP (-1) -- never source 0, which is a real person's identity.
export function buildBatchJobPayload({
  mappings = [],
  swapMode = DEFAULT_SWAP_MODE,
  overrides = {},
  settings = {},
  sourceCount = 0,
  sourceFacesInfo = [],
  targetGroups = [],
  autoFallback = true,
} = {}) {
  const base = { ...FACESWAP_DEFAULTS, ...settings };

  let maxRank = -1;
  mappings.forEach((m) => {
    const parsed = Number(m.personRank);
    const r = Number.isInteger(parsed) && parsed >= 0 ? parsed : -1;
    if (r > maxRank) maxRank = r;
  });

  const faceMapping = new Array(maxRank + 1).fill(SKIP);
  mappings.forEach((m) => {
    const rank = Number(m.personRank);
    if (!Number.isInteger(rank) || rank < 0) return;
    const srcIdx = Number(m.sourceIdx);
    faceMapping[rank] = Number.isInteger(srcIdx) ? srcIdx : SKIP;
  });
  const normalizedFaceMapping = normalizeSourceMapping(faceMapping, sourceCount) || [];
  const sourceMappingNames = normalizedFaceMapping.map((sourceIndex) => {
    if (sourceIndex < 0) return null;
    return sourceFacesInfo[sourceIndex]?.name || `Face ${sourceIndex + 1}`;
  });
  const sourceMappingIds = normalizedFaceMapping.map((sourceIndex) => {
    if (sourceIndex < 0) return null;
    return sourceFacesInfo[sourceIndex]?.id || `memory-slot-${sourceIndex}`;
  });

  const primarySource = Number(mappings[0]?.sourceIdx);
  const primarySourceIdx = Number.isInteger(primarySource) && primarySource >= 0
    && primarySource < sourceCount ? primarySource : SKIP;
  const personIds = Array.from(new Set(mappings
    .map((m) => Number(m.personRank))
    .filter((rank) => Number.isInteger(rank) && rank >= 0)))
    .sort((a, b) => a - b);
  const selectionState = swapMode === 'Selected people'
    ? { selection_mode: 'multi_person', person_id: null, person_ids: personIds }
    : swapMode === 'Selected face'
      ? { selection_mode: 'selected', person_id: personIds[0] ?? null, person_ids: personIds.slice(0, 1) }
      : { selection_mode: 'none', person_id: null, person_ids: [] };
  // The ranks address people on the JOB'S target, which the server resolves
  // at dispatch (_run_one -> _selection_invalidation_for_active_context).
  // `targetGroups` here is the person bank of whichever target is active in
  // the UI, so it is only a fallback range check when the caller has nothing
  // better; see batch-matrix-check.mjs "active bank" for why it must not be
  // the bank of a different target.
  const normalizedSelectionState = normalizeTargetSelectionState(selectionState, targetGroups);

  return {
    payload: {
      ...base,
      enhancer: overrides.enhancer || base.selected_enhancer || DEFAULT_ENHANCER,
      detection: swapMode || DEFAULT_SWAP_MODE,
      output_method: base.output_method || 'Images & Video',
      video_method: base.video_swapping_method || 'In-Memory processing',
      upscale: base.subsample_upscale || '128px',
      mask_engine: base.mask_engine || 'DFL XSeg',
      mask_engine_2: base.mask_engine_2 || 'None',
      clip_text: base.mask_clip_text || '',
      sam2_model_size: base.sam2_model_size || 'tiny',
      track_identities: !!base.track_identities,
      autorotate: !!base.autorotate_faces,
      face_distance: parseFloat(overrides.faceDistance ?? base.max_face_distance ?? DEFAULT_FACE_DISTANCE),
      blend_ratio: parseFloat(base.blend_ratio || 0.8),
      num_swap_steps: parseInt(base.num_swap_steps || 1, 10),
      auto_fallback: autoFallback,
      face_mapping: normalizedFaceMapping,
      source_mapping_names: sourceMappingNames,
      source_mapping_ids: sourceMappingIds,
      selected_source_name: primarySourceIdx >= 0
        ? (sourceFacesInfo[primarySourceIdx]?.name || `Face ${primarySourceIdx + 1}`)
        : null,
      selected_source_id: primarySourceIdx >= 0
        ? (sourceFacesInfo[primarySourceIdx]?.id || `memory-slot-${primarySourceIdx}`)
        : null,
      selection_state: normalizedSelectionState,
    },
    primarySourceIdx,
    mappings,
  };
}

const mappingLabel = (mappings) => mappings
  .map((m) => `P#${m.personRank + 1}➔F#${m.sourceIdx + 1}`).join(', ');

const stagedJob = ({ target, targetIndex, built, sourceFacesInfo, label,
  frameStart, frameEnd, totalFrames, fallbackName }) => ({
  id: newStagedJobId(),
  target_name: target?.name || fallbackName,
  target_index: targetIndex,
  source_index: built.primarySourceIdx,
  source_name: sourceFacesInfo[built.primarySourceIdx]?.name
    || `Faceset ${built.primarySourceIdx + 1}`,
  mappings: built.mappings,
  frame_start: frameStart ?? (target?.start_frame || 1),
  frame_end: frameEnd ?? (target?.end_frame || target?.frames || 1),
  total_frames: totalFrames ?? (target?.frames || 1),
  label,
  payload: built.payload,
});

// ── Strategy 1: one mapping applied to every selected target ─────────────
export function stageOneToMany({
  targets = [], selectedTargets = [], mappings = [], swapMode = DEFAULT_SWAP_MODE,
  enhancer, faceDistance, sourceCount = 0, sourceFacesInfo = [], createJobPayload,
} = {}) {
  if (selectedTargets.length === 0) return { jobs: [], error: ERR_NO_SELECTED_TARGETS };
  if (sourceCount === 0) return { jobs: [], error: ERR_NO_SOURCES };
  const jobs = selectedTargets.map((tIdx) => {
    const target = targets[tIdx];
    const targetName = target?.name || `Target ${tIdx + 1}`;
    const built = createJobPayload(mappings, swapMode, { enhancer, faceDistance });
    return stagedJob({
      target, targetIndex: tIdx, built, sourceFacesInfo, fallbackName: targetName,
      label: `1:M | ${targetName} (${mappingLabel(built.mappings)})`,
    });
  });
  return { jobs, error: null };
}

// ── Strategy 2: each group carries its own targets and its own mapping ───
export function stageGrouped({
  targets = [], groups = [], sourceCount = 0, sourceFacesInfo = [], createJobPayload,
} = {}) {
  if (sourceCount === 0) return { jobs: [], error: ERR_NO_SOURCES };
  const jobs = [];
  groups.forEach((grp) => {
    if (!grp.targetIndices || grp.targetIndices.length === 0) return;
    grp.targetIndices.forEach((tIdx) => {
      const target = targets[tIdx];
      if (!target) return;
      const built = createJobPayload(grp.mappings, grp.swapMode, {
        enhancer: grp.enhancer,
        faceDistance: grp.faceDistance,
      });
      jobs.push(stagedJob({
        target, targetIndex: tIdx, built, sourceFacesInfo,
        label: `${grp.label} | ${target.name} (${mappingLabel(built.mappings)})`,
      }));
    });
  });
  if (jobs.length === 0) return { jobs: [], error: ERR_NO_GROUP_TARGETS };
  return { jobs, error: null };
}

// ── Strategy 3: per-file matrix, one row per target ──────────────────────
export function stageMatrix({
  targets = [], matrixConfig = {}, sourceCount = 0, sourceFacesInfo = [], createJobPayload,
} = {}) {
  if (targets.length === 0) return { jobs: [], error: ERR_NO_TARGETS };
  if (sourceCount === 0) return { jobs: [], error: ERR_NO_SOURCES };
  const jobs = [];
  targets.forEach((target, tIdx) => {
    const cfg = matrixConfig[tIdx];
    if (!cfg || !cfg.enabled) return;
    const built = createJobPayload(cfg.mappings || [], cfg.swapMode, {
      enhancer: cfg.enhancer,
      faceDistance: cfg.faceDistance,
    });
    const fs = cfg.frameStart != null ? cfg.frameStart : target.start_frame || 1;
    const fe = cfg.frameEnd != null ? cfg.frameEnd : target.end_frame || target.frames || 1;
    jobs.push(stagedJob({
      target, targetIndex: tIdx, built, sourceFacesInfo,
      frameStart: fs, frameEnd: fe, totalFrames: Math.max(1, fe - fs + 1),
      label: `Matrix | ${target.name} (${mappingLabel(built.mappings)})`,
    }));
  });
  if (jobs.length === 0) return { jobs: [], error: ERR_NO_MATRIX_FILES };
  return { jobs, error: null };
}

// Fill matrix rows whose target filename shares a token (>= 3 chars) with a
// faceset name. First matching faceset wins; unmatched rows are left alone.
export function autoMatchMatrixConfig({ targets = [], sourceFacesInfo = [], matrixConfig = {} } = {}) {
  const updated = { ...matrixConfig };
  let matchCount = 0;
  targets.forEach((target, tIdx) => {
    const targetClean = target.name.toLowerCase().replace(/[^a-z0-9]/g, ' ');
    let bestSourceIdx = -1;
    sourceFacesInfo.forEach((srcInfo, sIdx) => {
      const srcName = (srcInfo.name || `face_${sIdx}`).toLowerCase().replace(/[^a-z0-9]/g, ' ');
      const tokens = srcName.split(/\s+/).filter((t) => t.length >= 3);
      const isMatch = tokens.some((token) => targetClean.includes(token));
      if (isMatch && bestSourceIdx === -1) bestSourceIdx = sIdx;
    });
    if (bestSourceIdx !== -1) {
      updated[tIdx] = {
        ...(updated[tIdx] || {}),
        mappings: [{ personRank: 0, sourceIdx: bestSourceIdx }],
        enabled: true,
      };
      matchCount++;
    }
  });
  return { matrixConfig: updated, matchCount };
}

// ── Strategy 4: recipes ──────────────────────────────────────────────────
export function stageCartesian({
  targets = [], sourceCount = 0, sourceFacesInfo = [], createJobPayload,
} = {}) {
  if (targets.length === 0 || sourceCount === 0) return { jobs: [], error: ERR_RECIPE_INPUTS };
  const jobs = [];
  targets.forEach((target, tIdx) => {
    for (let sIdx = 0; sIdx < sourceCount; sIdx++) {
      const built = createJobPayload([{ personRank: 0, sourceIdx: sIdx }], DEFAULT_SWAP_MODE);
      const sName = sourceFacesInfo[sIdx]?.name || `Faceset #${sIdx + 1}`;
      jobs.push({
        ...stagedJob({ target, targetIndex: tIdx, built, sourceFacesInfo,
          label: `NxM Combinatorial | ${target.name} ➔ ${sName}` }),
        source_name: sName,
      });
    }
  });
  return { jobs, error: null };
}

export function stageSequential({
  targets = [], sourceCount = 0, sourceFacesInfo = [], createJobPayload,
} = {}) {
  if (targets.length === 0 || sourceCount === 0) return { jobs: [], error: ERR_RECIPE_INPUTS };
  const jobs = targets.map((target, tIdx) => {
    const sIdx = tIdx % sourceCount;
    const sName = sourceFacesInfo[sIdx]?.name || `Faceset #${sIdx + 1}`;
    const built = createJobPayload([{ personRank: 0, sourceIdx: sIdx }], DEFAULT_SWAP_MODE);
    return {
      ...stagedJob({ target, targetIndex: tIdx, built, sourceFacesInfo,
        label: `Sequential | ${target.name} ➔ ${sName}` }),
      source_name: sName,
    };
  });
  return { jobs, error: null };
}

export function stageSegments({
  targets = [], targetIndex = 0, segmentCount = 4, sourceFacesInfo = [], createJobPayload,
} = {}) {
  const target = targets[targetIndex];
  if (!target) return { jobs: [], error: ERR_SPLIT_NO_TARGET };
  const totalFrames = target.frames || 1;
  if (totalFrames <= 1) return { jobs: [], error: ERR_SPLIT_SINGLE_FRAME };
  const segs = Math.max(2, Math.min(segmentCount, 32));
  const step = Math.ceil(totalFrames / segs);
  const jobs = [];
  for (let i = 0; i < segs; i++) {
    const fs = i * step + 1;
    const fe = Math.min((i + 1) * step, totalFrames);
    if (fs > totalFrames) break;
    const built = createJobPayload([{ personRank: 0, sourceIdx: 0 }], DEFAULT_SWAP_MODE);
    jobs.push({
      ...stagedJob({ target, targetIndex, built, sourceFacesInfo,
        frameStart: fs, frameEnd: fe, totalFrames: fe - fs + 1,
        label: `Segment ${i + 1}/${segs} (${fs}-${fe}) | ${target.name}` }),
      source_name: sourceFacesInfo[built.primarySourceIdx]?.name
        || `Faceset #${built.primarySourceIdx + 1}`,
    });
  }
  return { jobs, error: null };
}

// ── The request ──────────────────────────────────────────────────────────
// Staged jobs -> the `jobs` list of POST /api/queue/add_batch. Identical for
// every strategy: the target travels by NAME, the primary faceset by id (with
// name and gallery position as the dispatch-time fallbacks), and the per-person
// assignment inside `payload.face_mapping` / `payload.source_mapping_ids`.
export function queueRequestFromStagedJobs(stagedJobs = []) {
  return stagedJobs.map((j) => ({
    target_name: j.target_name,
    source_index: j.source_index,
    source_name: j.source_name,
    source_id: j.source_id || j.payload?.selected_source_id || null,
    payload: j.payload,
    frame_start: j.frame_start,
    frame_end: j.frame_end,
    label: j.label,
  }));
}
