import { create } from 'zustand';
import { ACTION_BY_ID } from '../shortcuts/actions.js';
import { comboFromEvent, keyToken } from '../shortcuts/combo.js';
import {
  STORAGE_KEY, bindingsOf, buildIndex, firesInText, keyOf, parseStored, sanitizeOverrides, serialize,
} from '../shortcuts/keymap.js';

// The user's key bindings and the shortcut overlay's open state.
//
// Bindings live in localStorage only (the brief's "persist custom bindings in localStorage"): they are a property of
// this browser's hands, not of the project, so they must not ride in the backend settings that sync between the two
// development machines. A second window (the pop-out reference monitor) hears about a change through the
// 'storage' event, so a remap is not stale there until a reload.
//
// Handlers are bound ONCE on window and read `eventMatches` at keydown time, so a remap takes effect on the next
// keypress without re-subscribing anything.

const readStored = () => {
  try {
    return typeof localStorage === 'undefined' ? {} : parseStored(localStorage.getItem(STORAGE_KEY));
  } catch {
    return {};                               // storage blocked: defaults, this session only
  }
};

const writeStored = (overrides) => {
  try {
    if (typeof localStorage === 'undefined') return;
    if (Object.keys(overrides).length === 0) localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, serialize(overrides));
  } catch { /* the binding lasts for this session */ }
};

let index = buildIndex({});

export const useShortcutStore = create((set, get) => {
  const initial = readStored();
  index = buildIndex(initial);
  const commit = (overrides) => {
    index = buildIndex(overrides);
    writeStored(overrides);
    set({ overrides });
  };
  return {
    overrides: initial,
    overlayOpen: false,
    // True while the remapper is waiting for a key. Handlers do not need it (the capture listener swallows the
    // event first); it is state so the overlay and the Settings tab can say "recording".
    recording: false,

    /**
     * Replace `id`'s bindings. A list equal to the defaults is stored as "not modified"; an empty list is kept, as
     * "deliberately unbound".
     */
    setBindings: (id, combos) => {
      if (!ACTION_BY_ID[id]) return;
      commit(sanitizeOverrides({ ...get().overrides, [id]: combos }));
    },
    resetBindings: (id) => {
      const next = { ...get().overrides };
      delete next[id];
      commit(next);
    },
    resetAll: () => commit({}),
    setRecording: (recording) => set({ recording: !!recording }),

    openOverlay: () => set({ overlayOpen: true }),
    closeOverlay: () => set({ overlayOpen: false }),
    toggleOverlay: () => set({ overlayOpen: !get().overlayOpen }),
  };
});

if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
  window.addEventListener('storage', (e) => {
    if (e.key !== STORAGE_KEY && e.key !== null) return;
    const overrides = readStored();
    index = buildIndex(overrides);
    useShortcutStore.setState({ overrides });
  });
}

// ── Non-React reads, for handlers bound once on window ─────────────────────

/** The combos an action is bound to right now. */
export const bindings = (id) => bindingsOf(id, useShortcutStore.getState().overrides);

// One event is tested against several actions per keydown; compute its combo once.
const comboCache = new WeakMap();
const comboOf = (e) => {
  if (!e || typeof e !== 'object') return null;
  if (comboCache.has(e)) return comboCache.get(e);
  const c = comboFromEvent(e);
  comboCache.set(e, c);
  return c;
};

/** Does this keydown fire action `id`? (Exact: the modifiers must match too.) */
export function eventMatches(e, id) {
  const c = comboOf(e);
  return c !== null && !!index.get(c)?.includes(id);
}

/** The action ids this keydown fires, in table order. */
export function actionsFor(e) {
  const c = comboOf(e);
  return c === null ? [] : [...(index.get(c) || [])];
}

/**
 * Does a text field's focus still let this keydown through to `id`? A bare key never does; a Mod+ combo does only
 * for the actions that opt in.
 */
export function eventFiresInText(e, id) {
  const c = comboOf(e);
  return c !== null && eventMatches(e, id) && firesInText(id, c);
}

/** A keyup for a hold action: matched on the key alone, because the modifier may have been released first. */
export function keyupMatches(e, id) {
  const t = keyToken(e);
  return t !== null && bindings(id).some((c) => keyOf(c) === t);
}
