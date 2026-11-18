// The four roles a face can be given from the on-canvas menu (Stage 12), and what each one REALLY does.
//
// The brief named Primary Swap Target, Secondary Target, Reference Only and Exclude. The renderer has two real states for a
// person - swapped (with a source) or never swapped - and one real distinction inside "swapped": whether the person is THE
// pinned target of the "Selected face" mode. So:
//
//   primary    the pinned target. The same as clicking the box: "Selected face" mode swaps just this person.
//   secondary  also swapped, from a source, alongside the primary ("Selected people" mode).
//   exclude    never swapped, protected: the box is drawn with a crimson striped lock.
//   reference  never swapped either - the render is IDENTICAL to Exclude, because the backend has no separate "reference" state.
//              It is a marker kept in this browser (per clip) so the face can be told apart on the canvas: a tracked person
//              who is deliberately left alone, rather than one someone wants protected. The menu says so.
//
// Pure functions: .render-check/render-stage12-logic.mjs runs them under node.

export const CHOICE = Object.freeze({ PRIMARY: 'primary', SECONDARY: 'secondary', REFERENCE: 'reference', EXCLUDE: 'exclude' });

export const ROLE_CHOICES = Object.freeze([
  {
    id: CHOICE.PRIMARY, label: 'Primary Swap Target', short: 'Primary', color: '#34d399',
    hint: 'The same as clicking the box: "Selected face" mode swaps just this person.',
  },
  {
    id: CHOICE.SECONDARY, label: 'Secondary Target', short: 'Secondary', color: '#38bdf8',
    hint: 'Also swapped, from a source, alongside the others ("Selected people" mode).',
  },
  {
    id: CHOICE.REFERENCE, label: 'Reference Only', short: 'Reference', color: '#a78bfa',
    hint: 'Tracked but never swapped. Renders exactly like Exclude: the backend has no separate reference state, so this only marks the face.',
  },
  {
    id: CHOICE.EXCLUDE, label: 'Exclude (Never Swap)', short: 'Exclude', color: '#f43f5e',
    hint: 'Protected: this person is never swapped. Drawn with a crimson striped lock on the canvas.',
  },
]);

export const choiceMeta = (id) => ROLE_CHOICES.find((c) => c.id === id) || null;

/**
 * Which choice a face currently has, from faceRoles.faceRole(...) output.
 *   role    'target' | 'swapped' | 'bystander'
 *   excluded  the person is explicitly set to never swap
 *   tagged  this person carries the Reference-Only marker
 * Null for a face nobody has given a role (not captured, or captured but not mapped to a source).
 */
export function choiceOf({ role, excluded = false, personId = null, selectedTargetPersonId = null, tagged = false }) {
  if (role === 'target') return CHOICE.PRIMARY;
  if (role === 'swapped') {
    return personId !== null && personId !== undefined && personId === selectedTargetPersonId ? CHOICE.PRIMARY : CHOICE.SECONDARY;
  }
  if (excluded) return tagged ? CHOICE.REFERENCE : CHOICE.EXCLUDE;
  return null;
}

/**
 * Why a choice cannot be picked right now, in words ('' = it can). `info` is a face's role record (faceRole output plus
 * `choice`); `hasSources` says whether any source face is loaded.
 */
export function choiceBlocked(choice, info, { hasSources = true } = {}) {
  const current = info?.choice ?? null;
  if (choice === current) return 'This face already has this role';
  if (choice === CHOICE.SECONDARY) {
    if (!hasSources) return 'Add a source face to the gallery first';
    if (current === CHOICE.PRIMARY) return 'This is the primary: pick another face as primary to change it';
  }
  if ((choice === CHOICE.EXCLUDE || choice === CHOICE.REFERENCE) && info?.role === 'target') {
    return info.reason || 'This is the pinned target. Pin another face to change it.';
  }
  return '';
}

/** The little tag a face wears on the canvas: text and colour, or null for a face with no role. */
export function badgeFor(choice) {
  const m = choiceMeta(choice);
  return m ? { text: m.short, color: m.color } : null;
}

/** What "Reset Face Overrides" would clear for one face. */
export function resetSummary({ overrides = 0, pins = 0, mapped = false, tagged = false }) {
  const parts = [];
  if (overrides) parts.push(`${overrides} per-person override${overrides === 1 ? '' : 's'}`);
  if (pins) parts.push(`${pins} identity pin${pins === 1 ? '' : 's'}`);
  if (mapped) parts.push('its source / skip choice');
  if (tagged) parts.push('its Reference marker');
  return { parts, total: overrides + pins + (mapped ? 1 : 0) + (tagged ? 1 : 0), text: parts.join(', ') };
}
