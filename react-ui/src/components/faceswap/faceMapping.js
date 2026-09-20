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
  return Array.from(new Set(Array.isArray(targetGroups) ? targetGroups : []))
    .filter((x) => typeof x === 'number')
    .sort((a, b) => a - b);
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
  if (explicitRaw !== undefined && explicitRaw !== null && explicitRaw !== '') {
    // An explicit dropdown choice, including Skip, always wins.
    const explicit = Array.isArray(explicitRaw) ? Number(explicitRaw[0]) : Number(explicitRaw);
    if (!Number.isFinite(explicit)) return SKIP;
    return explicit >= 0 && explicit < sourceCount ? explicit : SKIP;
  }

  // "Selected face" means the ONE highlighted target person. The old fallback
  // used the person rank as a source index, so highlighting person 1 could swap
  // person 0 instead, and two captured people with one source produced [0, 1] —
  // an out-of-range 1 that the backend turned into an empty FaceSet.
  if (faceSelection === 'Selected face') {
    if (sourceCount < 1 || person !== selectedPerson) return SKIP;
    const src = Number(selectedSource);
    if (!Number.isFinite(src) || src < 0) return SKIP;
    return Math.min(src, sourceCount - 1);
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
}) {
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
