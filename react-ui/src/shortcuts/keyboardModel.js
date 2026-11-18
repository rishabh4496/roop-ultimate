// The virtual keyboard's model: which physical key drives which workstation actions, read from the live keymap.
//
// Pure (no React, no DOM), so the render-check can pin the layout and the mapping. The keymap is the only source:
// a binding the user remaps moves its highlight with it, and nothing here is a second copy of the defaults.
//
// Key tokens are the ones combo.js produces (letters upper-case, digits as '1'..'0', named keys as 'ArrowLeft'),
// so a key on the drawn keyboard is matched against a binding by plain string equality.

import { ACTION_BY_ID } from './actions.js';
import { buildIndex } from './keymap.js';
import { parseCombo } from './combo.js';

// The drawn rows, left to right. A key is a token; `w` is its width in key units (default 1). Only the keys a
// workstation shortcut can use are drawn, plus the modifiers, so the picture stays a keyboard you can read at a glance.
export const KEY_ROWS = Object.freeze([
  Object.freeze(['Escape', '`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=']),
  Object.freeze(['Tab', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\']),
  Object.freeze(['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'"]),
  Object.freeze(['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']),
  Object.freeze(['Space', 'ArrowLeft', 'ArrowUp', 'ArrowDown', 'ArrowRight']),
]);

// Wider keys on the drawn keyboard, in key units.
export const KEY_WIDTH = Object.freeze({ Escape: 1.5, Tab: 1.5, Space: 4.5, '\\': 1.5 });

export const KEY_LABEL = Object.freeze({
  Escape: 'Esc', Space: 'Space', ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓', Tab: 'Tab', '\\': '\\',
});

/** Every key token the drawn keyboard shows. */
export const DRAWN_KEYS = Object.freeze(KEY_ROWS.flat());

/**
 * The bindings grouped by the physical key they use: { [keyToken]: [{ combo, mods, actions: [{ id, label }] }] }.
 * A combo with modifiers is listed under its base key, so `C` shows both the plain-C action and Alt+C.
 */
export function bindingsByKey(overrides = {}) {
  const index = buildIndex(overrides);
  const byKey = {};
  for (const [combo, ids] of index) {
    const parsed = parseCombo(combo);
    if (!parsed) continue;
    const actions = ids
      .map((id) => ACTION_BY_ID[id])
      .filter(Boolean)
      .map((a) => ({ id: a.id, label: a.label, scope: a.scope }));
    if (!actions.length) continue;
    (byKey[parsed.key] ||= []).push({ combo, mods: parsed.mods, actions });
  }
  for (const list of Object.values(byKey)) {
    // Plain keys first, then modifier combos in a stable order, so the tooltip reads the same every time.
    list.sort((a, b) => a.mods.length - b.mods.length || a.combo.localeCompare(b.combo));
  }
  return byKey;
}

/**
 * What one drawn key shows: whether it is bound at all, how many actions it drives, and a short line for the
 * tooltip. Unknown keys (not drawn, or nothing bound) read as unbound rather than throwing.
 */
export function keyState(token, byKey) {
  const entries = byKey[token] || [];
  const actionCount = entries.reduce((n, e) => n + e.actions.length, 0);
  return {
    token,
    label: KEY_LABEL[token] || token,
    bound: entries.length > 0,
    entries,
    actionCount,
    title: entries.length
      ? entries.map((e) => `${e.combo}: ${e.actions.map((a) => a.label).join(', ')}`).join('\n')
      : 'Not bound',
  };
}

/** How many drawn keys have at least one binding: the headline of the keyboard view. */
export function boundCount(byKey) {
  return DRAWN_KEYS.filter((t) => (byKey[t] || []).length > 0).length;
}

/** The key token for a KeyboardEvent-like object, as the keymap spells it (for the pressed-key flash). */
export function tokenOfEvent(e) {
  const key = typeof e?.key === 'string' ? e.key : '';
  if (key === ' ' || key === 'Spacebar') return 'Space';
  if (key === 'Esc') return 'Escape';
  if (/^[a-z]$/.test(key)) return key.toUpperCase();
  if (/^[A-Z]$/.test(key)) return key;
  return key;
}
