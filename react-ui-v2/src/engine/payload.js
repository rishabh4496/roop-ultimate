// The /api/preview and /api/swap bodies, as one pure function of the studio state.
//
// The selection half (face_mapping, target_person_source_mapping, selection_state,
// processing_selection) is built by React UI 1.0's own pure builders - the same functions 1.0's
// Face Swap calls - so 2.0 cannot drift from the contract app/roop/processing_selection.py and
// app/roop/target_selection.py expect. That contract is where "the render swapped nothing"
// lives; it is not re-derived here.
//
// The settings half is the whole settings object (what /api/settings returns) plus the handful
// of keys the engine reads under a different name than the setting (enhancer <- selected_enhancer,
// face_distance <- max_face_distance, ...), exactly as 1.0's buildSwapPayload renames them, and
// the mouth-custody request fields. The engine reads every key with a CFG default, so a key it
// does not use is ignored.
import {
  SKIP, buildFaceMappingArray, buildTargetSelectionState, stableTargetSourceMapping,
} from '@engine/components/faceswap/faceMapping.js';
import { buildProcessingSelection } from '@engine/components/faceswap/processingSelection.js';
import { requestFields } from '@engine/components/faceswap/mouthCustody.js';

const num = (v, d) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

/** Stable id of source `i` (the gallery is positional; ids survive a reorder). */
export const sourceIdAt = (s, i) => s.sourcesInfo?.[i]?.id || `memory-slot-${i}`;
export const sourceNameAt = (s, i) => s.sourcesInfo?.[i]?.name || (s.sources?.[i] ? `Face ${i + 1}` : null);

/** The selection inputs every builder takes, from the studio state. */
function selectionInputs(s, faceSelection) {
  const sourceCount = (s.sources || []).length;
  return {
    targetGroups: s.targetGroups || [],
    faceMapping: s.faceMapping || {},
    sourceCount,
    faceSelection,
    selTargetFace: s.selTargetFace || 0,
    selectedSource: s.selectedSource || 0,
    targetPersonIds: s.targetPersonIds || [],
    selectedTargetPersonId: s.selectedTargetPersonId || null,
    sourceIdentityIds: Array.from({ length: sourceCount }, (_, i) => sourceIdAt(s, i)),
  };
}

/**
 * The /api/swap body (also what a queued job replays, and the base of a preview body).
 * `s` is the studio state: { settings, sources, sourcesInfo, selectedSource, targets, selectedTarget,
 * targetMediaId, targetGroups, targetPersonIds, faceMapping, selTargetFace, selectedTargetPersonId,
 * selectedReferenceFaceId, selectionVersion }.
 */
export function buildEnginePayload(s, { requestId = null } = {}) {
  const settings = s.settings || {};
  const mode = settings.face_detection_mode;
  const inputs = selectionInputs(s, mode);
  const faceMappingArray = buildFaceMappingArray(inputs);
  const selectionState = buildTargetSelectionState({
    ...inputs,
    targetReferenceIndex: inputs.selTargetFace,
    targetMediaIndex: s.selectedTarget ?? 0,
    selectedReferenceFaceId: s.selectedReferenceFaceId || null,
  });
  const processingSelection = buildProcessingSelection({
    targetMediaId: s.targetMediaId || null,
    targetGroups: inputs.targetGroups,
    targetPersonIds: inputs.targetPersonIds,
    faceMapping: inputs.faceMapping,
    sourceCount: inputs.sourceCount,
    sourceIdentityIds: inputs.sourceIdentityIds,
    faceSelection: mode,
    selectedTargetPersonId: inputs.selectedTargetPersonId,
    selectedReferenceFaceId: s.selectedReferenceFaceId || null,
    selectedSource: inputs.selectedSource,
    selectionState,
    selectionVersion: s.selectionVersion ?? null,
    requestId,
  });
  const sel = inputs.selectedSource;
  return {
    ...settings,
    ...requestFields(settings),
    face_mapping: faceMappingArray,
    target_person_source_mapping: stableTargetSourceMapping(inputs),
    target_person_ids: inputs.targetPersonIds,
    source_mapping_names: faceMappingArray.map((i) => (i >= 0 ? sourceNameAt(s, i) : null)),
    source_mapping_ids: faceMappingArray.map((i) => (i >= 0 ? sourceIdAt(s, i) : null)),
    selection_state: selectionState,
    processing_selection: processingSelection,
    face_detection_mode: mode,
    detection: mode,
    enhancer: settings.selected_enhancer,
    adaptive_enhancer_profile: settings.adaptive_enhancer_profile || 'BALANCED',
    upscale: settings.subsample_upscale,
    mask_engine: settings.mask_engine,
    mask_engine_2: settings.mask_engine_2,
    clip_text: settings.mask_clip_text,
    autorotate: settings.autorotate_faces,
    face_distance: num(settings.max_face_distance, 0.75),
    blend_ratio: num(settings.blend_ratio, 0.8),
    num_swap_steps: num(settings.num_swap_steps, 1),
    selected_source_name: sourceNameAt(s, sel),
    selected_source_id: sourceIdAt(s, sel),
    target_media_id: s.targetMediaId || null,
    selection_version: s.selectionVersion ?? null,
  };
}

/** The /api/preview body for `frame` (1-based). `swap: false` asks for the untouched frame + face boxes. */
export function buildPreviewPayload(s, { frame = 1, swap = true } = {}) {
  const body = buildEnginePayload(s);
  return {
    ...body,
    index: s.selectedTarget ?? 0,
    target_index: s.selectedTarget ?? 0,
    frame,
    fake_preview: !!swap,
    processing_selection: body.processing_selection,
  };
}

/** A /api/queue/add job: the swap body frozen with the names the queue shows. */
export function buildQueueJob(s, extra = {}) {
  const payload = buildEnginePayload(s);
  const sel = s.selectedSource || 0;
  return {
    target_name: s.targets?.[s.selectedTarget ?? 0]?.name || '',
    target_media_id: s.targetMediaId || null,
    source_index: sel,
    source_name: sourceNameAt(s, sel) || 'Selected face',
    source_id: sourceIdAt(s, sel),
    payload,
    processing_selection: payload.processing_selection,
    selection_version: payload.selection_version,
    request_id: payload.processing_selection?.request_id,
    ...extra,
  };
}

/**
 * Why a job cannot start yet, or null. The swap that "succeeds" with nothing to swap is this
 * project's most repeated failure, so the UI refuses it up front instead.
 */
export function readiness(s) {
  if (!(s.sources || []).length) return 'Add a source face to the cast';
  if (!(s.targets || []).length) return 'Add target media';
  const mode = s.settings?.face_detection_mode;
  if (mode === 'Selected face' && !(s.targetPersonIds || []).length && !(s.targetGroups || []).length) {
    return 'Pick the target person on the viewport (click a face)';
  }
  const mapping = buildFaceMappingArray(selectionInputs(s, mode));
  if ((mode === 'Selected face' || mode === 'Selected people') && mapping.length && mapping.every((i) => i === SKIP)) {
    return 'Every target person is set to Skip - map one to a source';
  }
  return null;
}
