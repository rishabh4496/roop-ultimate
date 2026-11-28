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
import { serialize as serializeRoto } from '@engine/components/faceswap/rotoModel.js';
import { serializePins } from '@engine/components/faceswap/identityLock.js';
import { normalizeBlend } from '@engine/components/faceswap/identityBlend.js';
import { overrideAt, patchSettings, overridePieces, patchPayload, canBypass } from '@engine/components/faceswap/timeline/overrides.js';
import { patchSettingsWithCurves, applyCurveValues } from '@engine/components/faceswap/curves/curveModel.js';
import { bakeCurvePieces, piecesHaveCurves } from '@engine/components/faceswap/curves/curvePieces.js';

/** The per-clip tool state (roto zones, pins, scene overrides, curves) of the selected target. */
export const clipKeyOf = (s) => s.targetMediaId || `legacy-index-${s.selectedTarget ?? 0}`;
export const clipFxOf = (s) => ({ roto: [], pins: [], overrides: [], curves: {}, ...((s.clipFx || {})[clipKeyOf(s)] || {}) });

/** The brush mask as ProcessMgr reads it: {"<faceset>": {exclude: dataUrl, canonical, ref_kps}}; '' when none. */
export function maskJsonOf(s) {
  const m = s.mask;
  if (!m?.dataUrl) return '';
  const entry = { exclude: m.dataUrl, canonical: false };
  if (m.refKps) entry.ref_kps = m.refKps;
  return JSON.stringify({ [String(s.selectedSource || 0)]: entry });
}

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
    // The tools. The engine CLEARS roto zones and pins on a request that does not carry them, so what
    // is sent is the whole truth for this render (the per-clip lists, never another clip's).
    imagemask: maskJsonOf(s),
    roto_polygons: serializeRoto(clipFxOf(s).roto),
    identity_pins: serializePins(clipFxOf(s).pins),
    identity_blend: normalizeBlend(s.identityBlend),
  };
}

/** The settings in force at `frame`: the clip's, with the scene override and the curves at that frame on top. */
export function settingsAtFrame(s, frame) {
  const fx = clipFxOf(s);
  return patchSettingsWithCurves(patchSettings(s.settings || {}, overrideAt(fx.overrides, frame)), fx.curves, frame);
}

/**
 * The In/Out range in the frames the UI shows: 1-based, inclusive - the frames /api/preview decodes, the
 * queue's frame_start/frame_end, and (since 11-28) the target list's start_frame/end_frame and what
 * /api/target/set_frame takes. The engine stores a 0-based seek + exclusive end (`startframe`/`endframe`)
 * and converts at the API. Before that fix it stored In raw, so In 300 rendered from 301 (measured live).
 * The one place 2.0 reads the range, so a contract change is one edit.
 */
export function inOutOf(t) {
  const total = Math.max(1, t?.frames || 1);
  const s1 = Number(t?.start_frame) || 1;
  const e1 = Number(t?.end_frame) || 0;
  const a = Math.min(total, Math.max(1, s1));
  const b = Math.min(total, Math.max(a, e1 > 0 ? e1 : total));
  return { a, b, total };
}

/** The value /api/target/set_frame takes for a UI frame `f` at `which`: the frame itself (1-based). */
export function engineFrameFor(which, f) {
  return Math.max(1, Math.round(Number(f) || 1));
}

/**
 * The render plan for the In/Out range: the pieces the scene overrides and parameter curves cut it into
 * (React UI 1.0's overridePieces + bakeCurvePieces). `has` = some piece differs from the clip's own
 * settings, so the range must render as one queued job per piece instead of one /api/swap.
 */
export function renderPlan(s) {
  const t = s.targets?.[s.selectedTarget ?? 0];
  const { a: lo, b: hi } = inOutOf(t);
  const fx = clipFxOf(s);
  const { pieces } = bakeCurvePieces(overridePieces(fx.overrides, lo, hi), fx.curves);
  const base = buildEnginePayload(s);
  const has = pieces.some((pc) => pc.override) || piecesHaveCurves(pieces, base);
  return { pieces, has, base, lo, hi };
}

/** The queue jobs for a plan: each piece's payload is the base with its override and curve values written over it. */
export function planJobs(s, plan) {
  const name = s.targets?.[s.selectedTarget ?? 0]?.name || '';
  const group = `ovr-${Date.now().toString(36)}`;
  return plan.pieces.map((pc, i) => buildQueueJob(s, {
    label: `${name} - ${pc.override ? pc.override.label : 'clip settings'}${pc.curves ? ' + curves' : ''} (${pc.start}-${pc.end})`,
    frame_start: pc.start,
    frame_end: pc.end,
    payload: applyCurveValues(patchPayload(plan.base, pc.override), pc.curves),
    ...(plan.pieces.length > 1 ? { segment: { group, index: i + 1, count: plan.pieces.length } } : {}),
  }));
}

/** Why a planned render cannot run (a bypass piece outside the targeted detection modes), or null. */
export function planBlocked(s, plan) {
  if (plan.pieces.some((pc) => pc.override?.bypass) && !canBypass(s.settings?.face_detection_mode)) {
    return 'Bypass swap needs the "Selected face" detection mode: the all-faces modes have no match threshold to close.';
  }
  return null;
}

/** The /api/preview body for `frame` (1-based). `swap: false` asks for the untouched frame + face boxes. */
export function buildPreviewPayload(s, { frame = 1, swap = true } = {}) {
  // Scrubbing shows the scene override and the curves in force at this frame.
  const body = buildEnginePayload({ ...s, settings: settingsAtFrame(s, frame) });
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
  const payload = extra.payload || buildEnginePayload(s);
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
