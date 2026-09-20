// ── The canonical processing selection (Stage 14) ─────────────────────────
// One immutable object that says WHAT a request is about: which target media,
// which target person(s), which source identity, which person->source mapping,
// which detection mode — plus a request id and a monotonic selection version.
//
// It is built ONCE per request by FaceSwap.jsx and travels unchanged through
//   /api/preview      (echoed back so a late response can be recognised),
//   /api/swap         (frozen into the run before the worker starts),
//   /api/queue/add    (frozen into the job at queue-creation time).
// The backend mirror is app/roop/processing_selection.py. Neither side ever
// re-derives target identity from array positions or from whatever the UI
// happens to show when a worker finally runs.
//
// Pure functions only: no React, no fetch — so `.render-check` can run them.

import { stableTargetSourceMapping, targetPersonRecords } from './faceMapping.js';

export const SELECTION_SCHEMA = 1;

let requestCounter = 0;
export function newRequestId() {
  requestCounter += 1;
  const rand = Math.random().toString(36).slice(2, 8);
  return `${Date.now().toString(36)}-${requestCounter.toString(36)}-${rand}`;
}

// Monotonic per session AND across reloads: a reload starts from Date.now(),
// which is larger than any version an earlier session could have written
// (unless it bumped more than once per millisecond for the whole session).
// The backend keeps the newest version per target media and ignores writes
// carrying an older one — that is what makes a late preview harmless.
export function nextSelectionVersion(previous) {
  const prev = Number.isFinite(previous) ? previous : 0;
  return Math.max(prev + 1, Date.now());
}

const text = (value) => {
  if (value === null || value === undefined || typeof value === 'boolean') return null;
  const s = String(value).trim();
  return s || null;
};

// Cheap content token for the mapping. Sorted keys so two equal mappings built
// in different insertion orders hash the same.
export function mappingVersionOf(mapping) {
  const entries = Object.entries(mapping || {})
    .filter(([, v]) => v !== null && v !== undefined && v !== '' && v !== -1)
    .map(([k, v]) => `${k}=${v}`)
    .sort();
  let h = 0;
  const s = entries.join('|');
  for (let i = 0; i < s.length; i += 1) h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0;
  return `${entries.length}:${(h >>> 0).toString(36)}`;
}

/**
 * Build the canonical selection from the current UI context.
 *
 * `selectionState` is the existing serializable target-selection contract
 * (buildTargetSelectionState) — kept inside the selection so the backend's
 * normalizer sees exactly what the client decided.
 */
export function buildProcessingSelection({
  targetMediaId = null,
  targetGroups = [],
  targetPersonIds = [],
  faceMapping = {},
  sourceCount = 0,
  sourceIdentityIds = [],
  faceSelection = null,
  selectedTargetPersonId = null,
  selectedReferenceFaceId = null,
  selectedSource = 0,
  selectionState = null,
  selectionVersion = null,
  requestId = null,
} = {}) {
  const sourceIds = Array.isArray(sourceIdentityIds) ? sourceIdentityIds : [];
  const mapping = stableTargetSourceMapping({
    targetGroups, targetPersonIds, faceMapping, sourceCount,
    sourceIdentityIds: sourceIds, faceSelection,
    selectedTargetPersonId, selectedSource,
  });
  const records = targetPersonRecords({ targetGroups, targetPersonIds });
  const knownPeople = new Set(records.map((r) => r.targetPersonId));
  const state = selectionState && typeof selectionState === 'object' ? selectionState : {};
  const mode = state.selection_mode || 'none';

  let personId = null;
  let personIds = [];
  if (mode === 'selected') {
    personId = text(state.person_id) || text(selectedTargetPersonId);
    if (personId && knownPeople.size && !knownPeople.has(personId)) personId = null;
    personIds = personId ? [personId] : [];
  } else if (mode === 'multi_person') {
    personIds = (Array.isArray(state.person_ids) ? state.person_ids : [])
      .map(text).filter((id) => id && (!knownPeople.size || knownPeople.has(id)));
  }

  const src = Number(selectedSource);
  const sourceIdentityId = Number.isInteger(src) && src >= 0 && src < sourceIds.length
    ? text(sourceIds[src]) : null;

  const referenceId = text(selectedReferenceFaceId) || text(state.target_reference_face_id);

  return {
    schema: SELECTION_SCHEMA,
    request_id: requestId || newRequestId(),
    selection_version: Number.isFinite(selectionVersion) ? selectionVersion : null,
    mapping_version: mappingVersionOf(mapping),
    target_media_id: text(targetMediaId),
    target_person_id: personId,
    target_person_ids: personIds,
    target_reference_face_id: referenceId,
    source_identity_id: sourceIdentityId,
    detection_mode: text(faceSelection),
    target_person_source_mapping: mapping,
    selection_state: { ...state },
  };
}

// The part of a selection that decides WHAT gets rendered. Two selections
// with equal identity ask for the same picture; the request id and version
// are bookkeeping and must not make otherwise-equal requests look different
// (they would defeat the preview cache).
export function selectionIdentity(selection) {
  const s = selection || {};
  const st = s.selection_state || {};
  return {
    target_media_id: s.target_media_id ?? null,
    target_person_id: s.target_person_id ?? null,
    target_person_ids: [...(s.target_person_ids || [])],
    target_reference_face_id: s.target_reference_face_id ?? null,
    source_identity_id: s.source_identity_id ?? null,
    detection_mode: s.detection_mode ?? null,
    target_person_source_mapping: { ...(s.target_person_source_mapping || {}) },
    selection_mode: st.selection_mode ?? null,
  };
}

export function sameSelectionIdentity(a, b) {
  return JSON.stringify(selectionIdentity(a)) === JSON.stringify(selectionIdentity(b));
}

/**
 * Decide what to do with a /api/preview response.
 *
 *   request  — { requestId, key, mediaId, frame, selection } captured when the
 *              request was SENT (never read from current state).
 *   response — the JSON the backend returned.
 *   wanted   — { key, mediaId, frame } of what the UI wants RIGHT NOW.
 *
 * Returns { accept, cache, reason }:
 *   accept — display it (it is the newest thing the UI wants).
 *   cache  — store it under the REQUEST's own key (the server confirmed it
 *            rendered that request), even when it is no longer wanted, so
 *            stepping back to that frame is instant. Never under the wanted
 *            key: a stale picture must not be filed under a live selection.
 */
export function classifyPreviewResponse({ request, response, wanted }) {
  const req = request || {};
  const res = response || {};
  const want = wanted || {};

  // The server must have answered THIS request. An id mismatch means a proxy,
  // a retry or a bug handed us somebody else's picture: never trust it.
  if (req.requestId && res.request_id && res.request_id !== req.requestId) {
    return { accept: false, cache: false, reason: 'request_id_mismatch' };
  }
  // The server must have rendered the media we asked for (a target can be
  // removed/replaced at the same array position while a request is in flight).
  if (req.mediaId && res.target_media_id && res.target_media_id !== req.mediaId) {
    return { accept: false, cache: false, reason: 'media_mismatch' };
  }
  // …and the person/source we asked for. The echo is the server's own reading
  // of the request. Only a DISAGREEMENT between two concrete ids counts: the
  // server legitimately answers null (plus a selection_diagnostic) when the
  // person we named no longer exists, and that answer must still be shown.
  const echo = res.processing_selection;
  if (req.selection && echo) {
    for (const field of ['target_media_id', 'target_person_id', 'source_identity_id']) {
      const asked = req.selection[field];
      const got = echo[field];
      if (asked && got && String(asked) !== String(got)) {
        return { accept: false, cache: false, reason: `${field}_mismatch` };
      }
    }
  }
  if (req.frame != null && res.frame != null && Number(res.frame) !== Number(req.frame)) {
    return { accept: false, cache: false, reason: 'frame_mismatch' };
  }
  // A valid answer to a question we are no longer asking: cache under its own
  // key, do not show it. The pending (coalesced) request will render what is
  // wanted now — or hit the cache if that turns out to be this very entry.
  if (want.key && req.key && want.key !== req.key) {
    return { accept: false, cache: true, reason: 'superseded' };
  }
  if (want.mediaId && req.mediaId && want.mediaId !== req.mediaId) {
    return { accept: false, cache: true, reason: 'superseded_media' };
  }
  if (want.frame != null && req.frame != null && Number(want.frame) !== Number(req.frame)) {
    return { accept: false, cache: true, reason: 'superseded_frame' };
  }
  return { accept: true, cache: true, reason: 'current' };
}

/**
 * Session restore: keep only identities the backend still knows about.
 * `saved` is the client's remembered context, `known` the lists the backend
 * just returned. A deleted person/angle/source can never come back from a
 * client-side memory.
 */
export function reconcileRestoredSelection({
  savedPersonId = null, savedReferenceId = null, savedMapping = {},
  personIds = [], referenceIds = [], sourceIdentityIds = null,
  serverPersonId = null, serverReferenceId = null,
} = {}) {
  const people = new Set((personIds || []).filter(Boolean).map(String));
  const refs = new Set((referenceIds || []).filter(Boolean).map(String));
  const sources = Array.isArray(sourceIdentityIds)
    ? new Set(sourceIdentityIds.filter(Boolean).map(String)) : null;

  const pick = (server, saved, known) => {
    const s = text(server);
    if (s && known.has(s)) return s;
    const c = text(saved);
    if (c && known.has(c)) return c;
    return null;
  };
  const mapping = {};
  Object.entries(savedMapping || {}).forEach(([person, source]) => {
    if (!people.has(String(person))) return;           // person no longer exists
    if (sources && !sources.has(String(source))) return; // source no longer exists
    mapping[person] = source;
  });
  return {
    selectedTargetPersonId: pick(serverPersonId, savedPersonId, people),
    selectedReferenceFaceId: pick(serverReferenceId, savedReferenceId, refs),
    faceMapping: mapping,
  };
}
