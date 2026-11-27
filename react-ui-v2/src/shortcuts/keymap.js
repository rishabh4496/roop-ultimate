// Resolving "which keys does this action use right now": the defaults in actions.js, overlaid with whatever the
// user has remapped. Pure functions over an `overrides` object - { actionId: [combo, ...] } - so the same code
// serves the live handlers, the remapper's conflict check and the node tests.
//
// An override REPLACES the action's defaults entirely, and an empty array means "deliberately unbound": an action
// the user cleared must stay cleared rather than quietly growing its default back.

import { ACTIONS, ACTION_BY_ID } from './actions.js';
import { hasMod, normalizeCombo, parseCombo, reservedReason, scopesOverlap } from './combo.js';

export const STORAGE_KEY = 'roop_keybinds_v1';
export const SCHEMA = 1;

/** The combos `id` uses: its override if there is one, else its defaults. */
export function bindingsOf(id, overrides = {}) {
  const o = overrides[id];
  if (Array.isArray(o)) return o;
  return ACTION_BY_ID[id]?.keys || [];
}

/** True when the user has changed this action (including clearing it). */
export const isOverridden = (id, overrides = {}) => Array.isArray(overrides[id]);

/** combo -> [action ids] for every action, under `overrides`. Built once per change, read on every keydown. */
export function buildIndex(overrides = {}) {
  const index = new Map();
  for (const a of ACTIONS) {
    for (const combo of bindingsOf(a.id, overrides)) {
      const list = index.get(combo);
      if (list) list.push(a.id); else index.set(combo, [a.id]);
    }
  }
  return index;
}

/**
 * Other actions that would fire from `combo` alongside `id`: same combo and scopes that can be live together.
 * Returns the conflicting action objects (empty = free).
 */
export function conflictsFor(id, combo, overrides = {}) {
  const me = ACTION_BY_ID[id];
  if (!me) return [];
  const out = [];
  for (const a of ACTIONS) {
    if (a.id === id || !scopesOverlap(a.scope, me.scope)) continue;
    if (bindingsOf(a.id, overrides).includes(combo)) out.push(a);
  }
  return out;
}

/** Every clash in the effective keymap: [{ combo, actions: [id, id, ...] }]. A clean keymap returns []. */
export function allConflicts(overrides = {}) {
  const found = [];
  for (const [combo, ids] of buildIndex(overrides)) {
    if (ids.length < 2) continue;
    const clash = [];
    for (let i = 0; i < ids.length; i += 1) {
      for (let j = i + 1; j < ids.length; j += 1) {
        if (scopesOverlap(ACTION_BY_ID[ids[i]].scope, ACTION_BY_ID[ids[j]].scope)) clash.push(ids[i], ids[j]);
      }
    }
    if (clash.length) found.push({ combo, actions: [...new Set(clash)] });
  }
  return found;
}

/**
 * Can `combo` be bound to `id`? { ok: true } or { ok: false, reason, conflicts }.
 * Conflicts are reported, not forbidden: the remapper offers to replace.
 */
export function checkBinding(id, combo, overrides = {}) {
  const norm = normalizeCombo(combo);
  if (!norm) return { ok: false, reason: 'That is not a key combination' };
  const reserved = reservedReason(norm);
  if (reserved) return { ok: false, reason: `Reserved: ${reserved}` };
  const conflicts = conflictsFor(id, norm, overrides);
  if (conflicts.length) return { ok: false, reason: `Already used by ${conflicts.map((a) => a.label).join(', ')}`, conflicts, combo: norm };
  return { ok: true, combo: norm };
}

/** Whether a combo fires while a text field has focus for this action: only a Mod+ combo on an action that opts in. */
export const firesInText = (id, combo) => !!ACTION_BY_ID[id]?.modInText && hasMod(combo);

/** Keep only the overrides this build understands: known action ids, parseable reserved-free combos, no duplicates. */
export function sanitizeOverrides(raw) {
  const out = {};
  if (!raw || typeof raw !== 'object') return out;
  for (const [id, list] of Object.entries(raw)) {
    if (!ACTION_BY_ID[id] || !Array.isArray(list)) continue;
    const clean = [];
    for (const c of list) {
      const norm = normalizeCombo(c);
      if (norm && !reservedReason(norm) && !clean.includes(norm)) clean.push(norm);
    }
    // A list that HAD entries and lost every one to validation is corrupt, not "unbound": the default stands. Only a
    // deliberately empty list means the user cleared the action.
    if (list.length > 0 && clean.length === 0) continue;
    // An override identical to the defaults is not an override (it would show "modified" for nothing).
    const def = ACTION_BY_ID[id].keys;
    if (clean.length === def.length && clean.every((c, i) => c === def[i])) continue;
    out[id] = clean;
  }
  return out;
}

export function parseStored(text) {
  try {
    const v = JSON.parse(text || 'null');
    if (!v || typeof v !== 'object' || v.v !== SCHEMA) return {};
    return sanitizeOverrides(v.overrides);
  } catch {
    return {};
  }
}

export const serialize = (overrides) => JSON.stringify({ v: SCHEMA, overrides });

/** The key a combo names, ignoring modifiers - what a keyup is matched on (see viewport.peekOriginal). */
export const keyOf = (combo) => parseCombo(combo)?.key ?? null;
