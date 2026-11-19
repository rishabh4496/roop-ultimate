// The key remapper's keyboard: a full ANSI (US) layout, and what the live keymap puts on each key of it.
//
// Pure (no React, no DOM) so .render-check/render-stage15-logic.mjs pins it. The drawn keys are the physical keys; what is
// ON them is read from the keymap (shortcuts/keymap.js), so a remap moves its highlight, a conflict lights both keys, and
// nothing here is a second copy of a default.
//
// Two ideas the older overlay keyboard (keyboardModel.js KEY_ROWS) does not have:
//   * LAYERS. A combo lives on its base key under ONE modifier set - plain, Ctrl, Alt, Shift, Ctrl+Shift... - so the keyboard
//     is shown one layer at a time. `Alt+1` is on key 1 in the Alt layer, not under the plain `1` (which is snapshot slot A).
//   * GROUPS. Every action belongs to one of six families, and each family has a colour, so "which keys are Transport" is a
//     glance, not a read.

import { ACTION_BY_ID } from './actions.js';
import { allConflicts, buildIndex } from './keymap.js';
import { parseCombo } from './combo.js';

// ── The families ─────────────────────────────────────────────────────────────
// Order is priority: a key that serves two families is coloured by the earlier one and carries a dot for the other.
export const GROUPS = Object.freeze([
  Object.freeze({ id: 'transport', label: 'Transport', color: '#38bdf8' }),
  Object.freeze({ id: 'viewport', label: 'Viewport', color: '#a78bfa' }),
  Object.freeze({ id: 'masking', label: 'Masking', color: '#34d399' }),
  Object.freeze({ id: 'soloing', label: 'Pipeline Soloing', color: '#fb923c' }),
  Object.freeze({ id: 'snapshots', label: 'Snapshots', color: '#f472b6' }),
  Object.freeze({ id: 'workspace', label: 'Workspace & Timeline', color: '#94a3b8' }),
]);
export const GROUP_BY_ID = Object.freeze(Object.fromEntries(GROUPS.map((g) => [g.id, g])));

/**
 * Which family an action is in. The flow bar and the contact sheet are Pipeline Soloing (they solo / bypass the pipeline's
 * stages); everything else follows its cheat-sheet category. An action in `Workspace` that is a transport verb by name
 * (transport.run, transport.addToQueue) stays Workspace - the family is the cheat-sheet's, not the id prefix's.
 */
export function groupOfAction(action) {
  if (!action) return 'workspace';
  if (action.scope === 'flow' || action.scope === 'sweep') return 'soloing';
  switch (action.category) {
    case 'Transport': return 'transport';
    case 'Viewport & Scopes': return 'viewport';
    case 'Mask & Roto': return 'masking';
    case 'Snapshots': return 'snapshots';
    default: return 'workspace';
  }
}

// ── The drawn keyboard ───────────────────────────────────────────────────────
// `t` token (what combo.js produces for the key), `l` label, `w` width in key units, `mod` for a modifier key, `dead` for a key
// the remapper cannot bind (it has no token the app listens for), `gap` for empty space.
const k = (t, w = 1, l) => ({ t, w, l: l ?? t });
const gap = (w) => ({ gap: w });
const dead = (id, l, w = 1) => ({ t: id, w, l, dead: true });
const mod = (t, l, w) => ({ t: `mod:${t}:${l}`, mod: t, w, l });

const fnRow = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12'];

export const ANSI_MAIN = Object.freeze([
  [k('Escape', 1, 'Esc'), gap(1), ...fnRow.slice(0, 4).map((f) => k(f)), gap(0.5), ...fnRow.slice(4, 8).map((f) => k(f)), gap(0.5), ...fnRow.slice(8).map((f) => k(f))],
  [k('`'), ...['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'].map((d) => k(d)), k('-'), k('='), k('Backspace', 2, 'Bksp')],
  [k('Tab', 1.5), ...'QWERTYUIOP'.split('').map((c) => k(c)), k('['), k(']'), k('\\', 1.5)],
  [dead('caps', 'Caps', 1.75), ...'ASDFGHJKL'.split('').map((c) => k(c)), k(';'), k("'"), k('Enter', 2.25)],
  [mod('Shift', 'Shift', 2.25), ...'ZXCVBNM'.split('').map((c) => k(c)), k(','), k('.'), k('/'), mod('Shift', 'Shift', 2.75)],
  [mod('Mod', 'Ctrl', 1.25), dead('win-l', 'Win', 1.25), mod('Alt', 'Alt', 1.25), k('Space', 6.25), mod('Alt', 'Alt', 1.25), dead('win-r', 'Win', 1.25), dead('menu', 'Menu', 1.25), mod('Mod', 'Ctrl', 1.25)],
]);

// The navigation cluster, drawn to the right of the main block, aligned to rows 1-5 of it.
export const ANSI_NAV = Object.freeze([
  [],
  [dead('ins', 'Ins'), k('Home'), k('PageUp', 1, 'PgUp')],
  [k('Delete', 1, 'Del'), k('End'), k('PageDown', 1, 'PgDn')],
  [],
  [gap(1), k('ArrowUp', 1, '↑'), gap(1)],
  [k('ArrowLeft', 1, '←'), k('ArrowDown', 1, '↓'), k('ArrowRight', 1, '→')],
]);

/** Every bindable key token on the drawn keyboard (no modifiers, no dead keys). */
export const BINDABLE_KEYS = Object.freeze(
  [...ANSI_MAIN, ...ANSI_NAV].flat().filter((key) => key.t && !key.mod && !key.dead).map((key) => key.t),
);

// ── Layers: the modifier set a combo is under ────────────────────────────────
export const LAYERS = Object.freeze([
  Object.freeze({ id: 'plain', label: 'Plain', mods: Object.freeze([]) }),
  Object.freeze({ id: 'ctrl', label: 'Ctrl', mods: Object.freeze(['Mod']) }),
  Object.freeze({ id: 'alt', label: 'Alt', mods: Object.freeze(['Alt']) }),
  Object.freeze({ id: 'shift', label: 'Shift', mods: Object.freeze(['Shift']) }),
  Object.freeze({ id: 'ctrl-shift', label: 'Ctrl+Shift', mods: Object.freeze(['Mod', 'Shift']) }),
]);
export const LAYER_BY_ID = Object.freeze(Object.fromEntries(LAYERS.map((l) => [l.id, l])));

const sameMods = (a, b) => a.length === b.length && a.every((m, i) => m === b[i]);

// A symbol the keymap spells as a CHARACTER (`?`, `+`, `{`, `<`) is a physical key with Shift held: `?` is Shift+/ . The drawn
// keyboard has the physical keys only, so those combos are placed on the key that types them, in the Shift layer.
export const SHIFTED_SYMBOL = Object.freeze({
  '?': '/', '+': '=', _: '-', '{': '[', '}': ']', '<': ',', '>': '.', '|': '\\', ':': ';', '"': "'", '~': '`',
  '!': '1', '@': '2', '#': '3', $: '4', '%': '5', '^': '6', '&': '7', '*': '8', '(': '9', ')': '0',
});

/** { key, mods } as the DRAWN keyboard places a combo: a shifted symbol moves to its physical key and gains Shift. */
export function physicalPlacement(parsed) {
  const phys = SHIFTED_SYMBOL[parsed.key];
  if (!phys) return { key: parsed.key, mods: parsed.mods };
  const mods = ['Mod', 'Alt', 'Shift'].filter((m) => m === 'Shift' || parsed.mods.includes(m));
  return { key: phys, mods };
}

/**
 * Everything the keyboard view needs, from the effective keymap (`overrides` over the defaults):
 *   byKey      { token: [{ combo, mods, actions: [{ id, label, group, scope }], conflict }] }  every combo on a base key
 *   conflicts  allConflicts(): [{ combo, actions: [id...] }] - clashes between actions that can be live together
 *   conflictCombos  Set of the clashing combos
 *   layerCount { layerId: number of keys lit in that layer }  for the layer tabs
 */
export function analyzeKeymap(overrides = {}) {
  const conflicts = allConflicts(overrides);
  const conflictCombos = new Set(conflicts.map((c) => c.combo));
  const byKey = {};
  for (const [combo, ids] of buildIndex(overrides)) {
    const parsed = parseCombo(combo);
    if (!parsed) continue;
    const actions = ids.map((id) => ACTION_BY_ID[id]).filter(Boolean)
      .map((a) => ({ id: a.id, label: a.label, group: groupOfAction(a), scope: a.scope }));
    if (!actions.length) continue;
    const at = physicalPlacement(parsed);
    (byKey[at.key] ||= []).push({ combo, mods: at.mods, actions, conflict: conflictCombos.has(combo) });
  }
  for (const list of Object.values(byKey)) list.sort((a, b) => a.mods.length - b.mods.length || a.combo.localeCompare(b.combo));
  const layerCount = {};
  for (const layer of LAYERS) {
    layerCount[layer.id] = BINDABLE_KEYS.filter((t) => (byKey[t] || []).some((e) => sameMods(e.mods, layer.mods))).length;
  }
  return { byKey, conflicts, conflictCombos, layerCount };
}

/**
 * One key as the layer shows it: the combos on it under exactly this modifier set, the families they serve (coloured by the
 * first), whether any of them clashes, and how many combos it has in OTHER layers (a hint to switch layer).
 */
export function keyView(token, analysis, layer = LAYER_BY_ID.plain) {
  const all = analysis.byKey[token] || [];
  const here = all.filter((e) => sameMods(e.mods, layer.mods));
  const groupIds = [...new Set(here.flatMap((e) => e.actions.map((a) => a.group)))]
    .sort((a, b) => GROUPS.findIndex((g) => g.id === a) - GROUPS.findIndex((g) => g.id === b));
  return {
    token,
    bound: here.length > 0,
    entries: here,
    actions: here.flatMap((e) => e.actions.map((a) => ({ ...a, combo: e.combo }))),
    groups: groupIds,
    primary: groupIds[0] || null,
    conflict: here.some((e) => e.conflict),
    elsewhere: all.length - here.length,
  };
}

/** How many modifier combos use each modifier key - lights Ctrl / Alt / Shift when any binding needs them. */
export function modifierUse(analysis) {
  const use = { Mod: 0, Alt: 0, Shift: 0 };
  for (const list of Object.values(analysis.byKey)) {
    for (const e of list) for (const m of e.mods) use[m] += 1;
  }
  return use;
}

/** Actions with a combo whose base key is `token` under ANY modifier set, for the list a clicked key opens. */
export function actionsOnKey(token, analysis) {
  const seen = new Set();
  const out = [];
  for (const e of analysis.byKey[token] || []) {
    for (const a of e.actions) {
      if (seen.has(a.id)) continue;
      seen.add(a.id);
      out.push({ ...a, combo: e.combo, conflict: e.conflict });
    }
  }
  return out;
}

/** A human line for each conflict: "Alt+C - Parameter curve editor, Compare" (labels, not ids). */
export function describeConflicts(analysis) {
  return analysis.conflicts.map((c) => ({
    combo: c.combo,
    ids: c.actions,
    labels: c.actions.map((id) => ACTION_BY_ID[id]?.label || id),
  }));
}
