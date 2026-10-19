// What the render will do with each DETECTED face, and the one way to say
// "never swap this person".
//
// WHY AN EXPLICIT MARKER. `faceMapping` is {targetPersonId: sourceIdentityId}.
// A person with NO key is not "skipped": in every mode except "Selected face"
// `stableTargetSourceMapping` gives them the legacy rank default (person 0 ->
// source 0, person 1 -> source 1, ...). `PersonGroups` implemented "Ignore /
// Skip" by DELETING the key, so in "Selected people" and "All faces" the person
// it just skipped was still swapped with their default source:
//
//     faceMapping {A: srcA}, B deleted, 'Selected people'  ->  [0, 1]   (B swaps)
//     faceMapping {A: srcA, B: null},   'Selected people'  ->  [0, -1]  (B never swaps)
//
// `mapPerson`'s own comment states the intent ("An explicit dropdown choice,
// including Skip, always wins"); that only holds if Skip is an explicit entry.
// `null` is the entry the backend already treats as skip
// (`_save_target_mapping_from_payload`, `_resolve_target_person_source`).
//
// Pure functions only: `.render-check/face-roles-check.mjs` runs them under node.
import { stableTargetSourceMapping } from './faceMapping.js';

export const ROLE = Object.freeze({
  TARGET: 'target',       // the pinned person ("Selected face")
  SWAPPED: 'swapped',     // mapped to a source, will be swapped
  BYSTANDER: 'bystander', // not swapped: unmapped, excluded, or not a captured person
});

export const NEVER_SWAP = null;

const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o || {}, k);
const ALL_FACES_MODES = new Set(['All faces', 'All input faces']);

// Explicit exclusion of one person. Returns a NEW mapping.
export function excludePerson(faceMapping, personId) {
  if (personId === null || personId === undefined || personId === '') return { ...(faceMapping || {}) };
  return { ...(faceMapping || {}), [personId]: NEVER_SWAP };
}

// Undo an exclusion: drop the key so the person returns to the mode's default.
export function includePerson(faceMapping, personId) {
  const next = { ...(faceMapping || {}) };
  delete next[personId];
  return next;
}

// Is this person explicitly set to "never swap"?
export function isExcluded(faceMapping, personId, sourceCount = 0) {
  if (!hasOwn(faceMapping, personId)) return false;
  const raw = faceMapping[personId];
  if (raw === null || raw === undefined || raw === '') return true;
  const asString = String(raw).trim().toLowerCase();
  if (['skip', 'ignore', 'none', '-1'].includes(asString)) return true;
  // A number is an explicit source index: negative, or past the gallery, is a skip.
  const n = Number(raw);
  if (Number.isInteger(n)) return n < 0 || (sourceCount > 0 && n >= sourceCount);
  return false;                       // anything else is a source identity id
}

/**
 * The role of ONE detected face.
 *
 * personId        stable id of the captured person this box was matched to, or
 *                 null/undefined for a face nobody captured
 * Returns { role, excluded, canExclude, reason }
 */
export function faceRole({
  personId = null,
  faceSelection = null,
  selectedTargetPersonId = null,
  faceMapping = {},
  targetGroups = [],
  targetPersonIds = [],
  sourceCount = 0,
  sourceIdentityIds = [],
  selectedSource = 0,
} = {}) {
  const hasPerson = personId !== null && personId !== undefined && personId !== '';
  if (!hasPerson) {
    // No identity to key an exclusion on.
    if (ALL_FACES_MODES.has(faceSelection)) {
      return {
        role: ROLE.SWAPPED, excluded: false, canExclude: false,
        reason: 'Every face is swapped in this mode. Capture this face as a person to exclude it.',
      };
    }
    return {
      role: ROLE.BYSTANDER, excluded: false, canExclude: false,
      reason: 'Not a captured person, so it is never swapped.',
    };
  }

  const excluded = isExcluded(faceMapping, personId, sourceCount);
  const stable = stableTargetSourceMapping({
    targetGroups, targetPersonIds, faceMapping, sourceCount, sourceIdentityIds,
    faceSelection, selectedTargetPersonId, selectedSource,
  });
  const swaps = !excluded && hasOwn(stable, personId);
  const pinned = faceSelection === 'Selected face' && selectedTargetPersonId === personId;

  if (pinned && swaps) {
    return {
      role: ROLE.TARGET, excluded: false, canExclude: false,
      reason: 'This is the pinned target. Pin another face to change it.',
    };
  }
  if (swaps) {
    return { role: ROLE.SWAPPED, excluded: false, canExclude: true, reason: '' };
  }
  return {
    role: ROLE.BYSTANDER, excluded, canExclude: true,
    reason: excluded ? 'Excluded: this person is never swapped.'
      : 'Not mapped to a source, so it is not swapped.',
  };
}
