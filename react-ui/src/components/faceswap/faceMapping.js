// Target-person -> source-face mapping, shared by FaceSwap.jsx (which sends it
// to the backend as `face_mapping`) and PersonGroups.jsx (which must SHOW the
// same decision in its per-person dropdown). It lives in its own module because
// the two were computing the fallback independently and disagreed: the UI row
// said "Source 1" while the payload said "skip".
//
// Contract, matching app/api.py mapped_facesets():
//   result[rank] is the source faceset index for target person `rank`.
//   -1 means "do not swap this person" (an empty FaceSet backend-side).
//   Any index outside the loaded source gallery is an error, so it degrades to
//   -1 rather than being sent on to address a faceset that does not exist.

export const SKIP = -1;

const hasOwn = (object, key) => Object.prototype.hasOwnProperty.call(object || {}, key);

// Source-gallery indices are a separate namespace from target-person ranks.
// Normalize malformed values to the explicit skip sentinel; never coerce them
// to source 0, because source 0 is a real user's identity.
export function normalizeSourceIndex(value, sourceCount) {
  if (value === null || value === undefined || value === ''
      || typeof value === 'boolean') return SKIP;
  const n = Number(value);
  return Number.isInteger(n) && n >= 0 && n < Number(sourceCount || 0) ? n : SKIP;
}

export function normalizeSourceMapping(mapping, sourceCount) {
  if (!Array.isArray(mapping)) return null;
  return mapping.map((value) => normalizeSourceIndex(value, sourceCount));
}

export function mappingObjectFromArray(mapping) {
  const result = {};
  (Array.isArray(mapping) ? mapping : []).forEach((value, index) => {
    result[index] = value;
  });
  return result;
}

export function remapSourceMappingAfterRemoval(mapping, removedIndex) {
  const removed = Number(removedIndex);
  if (!Array.isArray(mapping) || !Number.isInteger(removed) || removed < 0) {
    return Array.isArray(mapping) ? [...mapping] : [];
  }
  return mapping.map((value) => {
    const source = Number(value);
    if (!Number.isInteger(source) || source < 0) return SKIP;
    if (source === removed) return SKIP;
    return source > removed ? source - 1 : source;
  });
}

export function remapSourceMappingAfterMove(mapping, fromIndex, toIndex) {
  const from = Number(fromIndex);
  const to = Number(toIndex);
  if (!Array.isArray(mapping) || !Number.isInteger(from) || !Number.isInteger(to)
      || from < 0 || to < 0 || from === to) {
    return Array.isArray(mapping) ? [...mapping] : [];
  }
  return mapping.map((value) => {
    const source = Number(value);
    if (!Number.isInteger(source) || source < 0) return SKIP;
    if (source === from) return to;
    if (from < to && source > from && source <= to) return source - 1;
    if (to < from && source >= to && source < from) return source + 1;
    return source;
  });
}

// The highlighted target person, from the highlighted target FACE.
// targetGroups[i] is the person rank of target face i, and is occasionally an
// array (a face banked under a person) or a numeric string out of storage.
export function selectedPersonOf(targetGroups, selTargetFace) {
  const raw = Array.isArray(targetGroups) ? targetGroups[selTargetFace] : undefined;
  if (typeof raw === 'number') return raw;
  if (Array.isArray(raw)) return Number(raw[0]);
  const n = Number(raw);
  return Number.isFinite(n) ? n : NaN;
}

export function uniquePersons(targetGroups) {
  return Array.from(new Set((Array.isArray(targetGroups) ? targetGroups : [])
    .map((x) => Array.isArray(x) ? Number(x[0]) : Number(x))
    .filter((x) => Number.isFinite(x))))
    .sort((a, b) => a - b);
}

// Stable target-person projection.  Display rank is derived from first
// appearance and is never persisted as identity.  `faceIndices` and
// `referenceFaceIds` are angle/reference namespaces, not person ids.
export function targetPersonRecords({
  targetGroups = [], targetPersonIds = [], targetReferenceFaceIds = [], targetNames = [],
} = {}) {
  const groups = Array.isArray(targetGroups) ? targetGroups : [];
  const ids = Array.isArray(targetPersonIds) ? targetPersonIds : [];
  const refs = Array.isArray(targetReferenceFaceIds) ? targetReferenceFaceIds : [];
  const order = [];
  groups.forEach((group, index) => {
    const personId = ids[index] || `legacy-target-person-${String(group)}`;
    if (!order.includes(personId)) order.push(personId);
  });
  return order.map((targetPersonId, displayRank) => {
    const faceIndices = groups.map((group, index) => {
      const id = ids[index] || `legacy-target-person-${String(group)}`;
      return id === targetPersonId ? index : null;
    }).filter((index) => index !== null);
    return {
      targetPersonId,
      target_person_id: targetPersonId,
      displayRank,
      display_rank: displayRank,
      faceIndices,
      face_indices: faceIndices,
      referenceFaceIds: faceIndices.map((index) => refs[index]).filter(Boolean),
      reference_face_ids: faceIndices.map((index) => refs[index]).filter(Boolean),
      name: Array.isArray(targetNames) ? (targetNames[displayRank] || '') : '',
    };
  });
}

export function selectedTargetPersonId({
  targetPersonIds, selectedReferenceIndex,
} = {}) {
  const index = Number(selectedReferenceIndex);
  const ids = Array.isArray(targetPersonIds) ? targetPersonIds : [];
  return Number.isInteger(index) && index >= 0 && index < ids.length
    ? ids[index] || null
    : null;
}

export function stableTargetSourceMapping({
  targetGroups, targetPersonIds, faceMapping, sourceCount, sourceIdentityIds = [],
  faceSelection, selectedTargetPersonId: selectedId, selectedSource,
} = {}) {
  const records = targetPersonRecords({ targetGroups, targetPersonIds });
  const sourceIds = Array.isArray(sourceIdentityIds) ? sourceIdentityIds : [];
  const result = {};
  records.forEach((record) => {
    const key = record.targetPersonId;
    const hasStable = hasOwn(faceMapping, key);
    const legacyRank = record.displayRank;
    const raw = hasStable ? faceMapping[key]
      : (hasOwn(faceMapping, legacyRank) ? faceMapping[legacyRank] : undefined);
    if (raw !== undefined) {
      const byIdentity = sourceIds.findIndex((id) => String(id) === String(raw));
      const sourceIndex = byIdentity >= 0
        ? byIdentity : normalizeSourceIndex(raw, sourceCount);
      if (sourceIndex >= 0 && sourceIds[sourceIndex]) result[key] = sourceIds[sourceIndex];
      return;
    }
    if (faceSelection === 'Selected face' && selectedId === key) {
      const sourceIndex = normalizeSourceIndex(selectedSource, sourceCount);
      if (sourceIndex >= 0 && sourceIds[sourceIndex]) result[key] = sourceIds[sourceIndex];
    }
  });
  return result;
}

// The source index for ONE person. Single source of truth for both the payload
// and the dropdown.
export function mapPerson({
  person,
  faceMapping,
  sourceCount,
  faceSelection,
  selectedPerson,
  selectedSource,
}) {
  const explicitRaw = faceMapping ? faceMapping[person] : undefined;
  if (hasOwn(faceMapping, person)) {
    // An explicit dropdown choice, including Skip, always wins.
    return normalizeSourceIndex(
      Array.isArray(explicitRaw) ? explicitRaw[0] : explicitRaw,
      sourceCount,
    );
  }

  // "Selected face" means the ONE highlighted target person. The old fallback
  // used the person rank as a source index, so highlighting person 1 could swap
  // person 0 instead, and two captured people with one source produced [0, 1] —
  // an out-of-range 1 that the backend turned into an empty FaceSet.
  if (faceSelection === 'Selected face') {
    if (sourceCount < 1 || person !== selectedPerson) return SKIP;
    const src = Number(selectedSource);
    if (!Number.isInteger(src) || src < 0 || src >= sourceCount) return SKIP;
    return src;
  }

  // Legacy person-rank default for the multi-source modes, clamped to the
  // gallery so an uncaptured rank is an intentional skip, not a silent no-op.
  return person >= 0 && person < sourceCount ? person : SKIP;
}

// The whole `face_mapping` array, indexed by person rank.
export function buildFaceMappingArray({
  targetGroups,
  faceMapping,
  sourceCount,
  faceSelection,
  selTargetFace,
  selectedSource,
  targetPersonIds,
  selectedTargetPersonId,
  sourceIdentityIds,
}) {
  if (Array.isArray(targetPersonIds) && targetPersonIds.length) {
    const stable = stableTargetSourceMapping({
      targetGroups, targetPersonIds, faceMapping, sourceCount, sourceIdentityIds,
      faceSelection, selectedTargetPersonId, selectedSource,
    });
    return targetPersonRecords({ targetGroups, targetPersonIds })
      .map((record) => {
        const sourceId = stable[record.targetPersonId];
        const index = Array.isArray(sourceIdentityIds)
          ? sourceIdentityIds.findIndex((id) => String(id) === String(sourceId)) : SKIP;
        return index >= 0 ? index : SKIP;
      });
  }
  const selectedPerson = selectedPersonOf(targetGroups, selTargetFace);
  return uniquePersons(targetGroups).map((person) => mapPerson({
    person,
    faceMapping,
    sourceCount,
    faceSelection,
    selectedPerson,
    selectedSource,
  }));
}

const optionalInt = (value) => {
  if (value === null || value === undefined || typeof value === 'boolean') return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  const n = Number(value);
  return Number.isInteger(n) ? n : null;
};

// UI-side normalization of the same serializable contract consumed by
// app/roop/target_selection.py. person_id/person_ids are stable target-person
// ids when the active context provides them, and legacy display ranks only for
// old contexts. They are never source-gallery, reference-angle, media,
// detection, or track indices. The extra fields remain explicit so a future
// video selector can add a stable track id without overloading person_id.
export function normalizeTargetSelectionState(selection = {}, targetGroups = []) {
  const raw = selection && typeof selection === 'object' ? selection : {};
  const mode = raw.selection_mode === 'selected' || raw.selection_mode === 'multi_person'
    ? raw.selection_mode : 'none';
  const persons = uniquePersons(targetGroups);
  let personId = optionalInt(raw.person_id);
  let personIds = Array.isArray(raw.person_ids)
    ? raw.person_ids.map(optionalInt).filter((v, i, all) => v !== null && all.indexOf(v) === i)
    : [];
  let diagnostic = null;

  if (mode === 'selected') {
    if (personId === null) diagnostic = 'selection_required';
    else personIds = [personId];
  } else if (mode === 'multi_person') {
    if (!personIds.length) diagnostic = 'selection_required';
  } else {
    personId = null;
    personIds = [];
  }

  if (!diagnostic && mode !== 'none' && personIds.some((id) => id < 0)) {
    diagnostic = 'invalid_person_id';
  }
  if (!diagnostic && mode !== 'none' && persons.length
      && personIds.some((id) => id >= persons.length)) {
    diagnostic = 'invalid_person_id';
  }
  if (diagnostic) {
    personId = null;
    personIds = [];
  }

  const result = {
    selection_mode: mode,
    person_id: personId,
    person_ids: personIds,
    target_reference_index: optionalInt(raw.target_reference_index),
    target_detection_index: optionalInt(raw.target_detection_index),
    track_id: optionalInt(raw.track_id),
    target_media_index: optionalInt(raw.target_media_index),
    valid: !diagnostic,
    diagnostic,
  };
  const referenceId = raw.target_reference_face_id || raw.selected_reference_face_id;
  if (referenceId) result.target_reference_face_id = referenceId;
  return result;
}

// Convert the UI's highlighted reference angle and mapping controls into the
// canonical target-person contract. A person's other captured angles remain
// represented by the same person_id and are pooled by ProcessMgr.
export function buildTargetSelectionState({
  faceSelection,
  targetGroups,
  faceMapping,
  sourceCount,
  selTargetFace,
  selectedSource,
  targetReferenceIndex = selTargetFace,
  targetMediaIndex = null,
  targetPersonIds = [],
  selectedTargetPersonId = null,
  selectedReferenceFaceId = null,
  sourceIdentityIds = [],
}) {
  if (Array.isArray(targetPersonIds) && targetPersonIds.length) {
    const records = targetPersonRecords({ targetGroups, targetPersonIds });
    const selectedId = selectedTargetPersonId
      || selectedTargetPersonIdOf({ targetGroups, targetPersonIds, selTargetFace });
    let selection;
    if (faceSelection === 'Selected face') {
      selection = {
        selection_mode: 'selected', person_id: selectedId,
        person_ids: selectedId ? [selectedId] : [],
      };
    } else if (faceSelection === 'Selected people') {
      const stable = stableTargetSourceMapping({
        targetGroups, targetPersonIds, faceMapping, sourceCount,
        sourceIdentityIds: sourceIdentityIds || [], faceSelection,
        selectedTargetPersonId: selectedId, selectedSource,
      });
      selection = {
        selection_mode: 'multi_person', person_id: null,
        person_ids: records.map((record) => record.targetPersonId)
          .filter((id) => hasOwn(stable, id)),
      };
    } else {
      selection = { selection_mode: 'none', person_id: null, person_ids: [] };
    }
    return {
      ...selection,
      ...(selectedReferenceFaceId
        ? { target_reference_face_id: selectedReferenceFaceId } : {}),
      target_reference_index: targetReferenceIndex,
      target_detection_index: null, track_id: null,
      target_media_index: targetMediaIndex, valid: true, diagnostic: null,
    };
  }
  const persons = uniquePersons(targetGroups);
  const highlightedRaw = selectedPersonOf(targetGroups, selTargetFace);
  const highlightedPerson = persons.indexOf(highlightedRaw);
  let selection;

  if (faceSelection === 'Selected face') {
    selection = {
      selection_mode: 'selected',
      person_id: highlightedPerson >= 0 ? highlightedPerson : null,
      person_ids: highlightedPerson >= 0 ? [highlightedPerson] : [],
    };
  } else if (faceSelection === 'Selected people') {
    const mappings = buildFaceMappingArray({
      targetGroups, faceMapping, sourceCount, faceSelection,
      selTargetFace, selectedSource,
    });
    selection = {
      selection_mode: 'multi_person',
      person_id: null,
      person_ids: mappings
        .map((source, rank) => (source >= 0 ? rank : null))
        .filter((rank) => rank !== null),
    };
  } else {
    selection = { selection_mode: 'none', person_id: null, person_ids: [] };
  }

  return normalizeTargetSelectionState({
    ...selection,
    target_reference_index: targetReferenceIndex,
    target_detection_index: null,
    track_id: null,
    target_media_index: targetMediaIndex,
  }, targetGroups);
}

function selectedTargetPersonIdOf({ targetGroups, targetPersonIds, selTargetFace }) {
  const index = Number(selTargetFace);
  return Array.isArray(targetPersonIds) && Number.isInteger(index)
    ? targetPersonIds[index] || null : selectedTargetPersonId({
      targetGroups, targetPersonIds, selectedReferenceIndex: selTargetFace,
    });
}
