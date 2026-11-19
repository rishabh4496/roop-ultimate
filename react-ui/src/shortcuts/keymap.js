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

// ── Hotkey profiles: a keymap as a file (Stage 15) ───────────────────────────
// A profile is the user's overrides plus a name, so a keymap can move between browsers, machines and people. Importing is
// REPLACE, not merge: the file is a whole keymap, and a half-merged one (some of yours, some of theirs) is the state that
// produces silent clashes.

export const PROFILE_FORMAT = 'roop-ultimate-keymap';
export const MAX_PROFILE_BYTES = 256 * 1024;

/** The object that is written to a profile file. `now` is injectable for the tests. */
export function exportProfile(overrides = {}, name = 'My keymap', now = new Date()) {
  return {
    format: PROFILE_FORMAT,
    v: SCHEMA,
    name: String(name || 'My keymap').trim().slice(0, 60) || 'My keymap',
    exportedAt: now.toISOString(),
    overrides: sanitizeOverrides(overrides),
  };
}

/**
 * Read a profile file's text. Returns { ok: true, name, overrides, ignored, total } or { ok: false, error }.
 *   ignored  how many entries this build could not use (an action id it does not have, or combos that are all invalid),
 *            so "imported 12, ignored 3" is something the UI can say instead of dropping them silently.
 * Nothing here touches storage: the caller applies `overrides`, then reads allConflicts() to report what clashes.
 */
export function parseProfile(text) {
  if (typeof text !== 'string' || !text.trim()) return { ok: false, error: 'The file is empty' };
  if (text.length > MAX_PROFILE_BYTES) return { ok: false, error: 'That file is too large to be a keymap' };
  let v;
  try { v = JSON.parse(text); } catch { return { ok: false, error: 'That is not a JSON file' }; }
  if (!v || typeof v !== 'object' || Array.isArray(v)) return { ok: false, error: 'That is not a keymap profile' };
  if (v.format !== PROFILE_FORMAT) return { ok: false, error: 'That JSON is not a Roop Ultimate keymap profile' };
  if (v.v !== SCHEMA) return { ok: false, error: `Keymap profile version ${String(v.v)} is not supported (this build reads version ${SCHEMA})` };
  if (!v.overrides || typeof v.overrides !== 'object' || Array.isArray(v.overrides)) return { ok: false, error: 'The profile has no keymap in it' };
  const entries = Object.entries(v.overrides);
  const overrides = sanitizeOverrides(v.overrides);
  // An entry that equals the defaults is correctly dropped by sanitizeOverrides and is NOT "ignored"; only an unknown action
  // or an entry that lost every combo to validation is.
  const ignored = entries.filter(([id, list]) => !ACTION_BY_ID[id] || !Array.isArray(list)
    || (list.length > 0 && !list.some((c) => { const n = normalizeCombo(c); return n && !reservedReason(n); }))).length;
  return { ok: true, name: String(v.name || 'Imported keymap').slice(0, 60), overrides, ignored, total: entries.length };
}

/** A file name for an exported profile: `roop-keymap-my-keymap.json`. */
export const profileFileName = (name) => `roop-keymap-${String(name || 'keymap').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'keymap'}.json`;

/** The key a combo names, ignoring modifiers - what a keyup is matched on (see viewport.peekOriginal). */
export const keyOf = (combo) => parseCombo(combo)?.key ?? null;
