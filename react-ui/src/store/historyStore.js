import { create } from 'zustand';
import * as E from '../history/historyEngine';

// Stage 23: the undo/redo history, as a store.
//
// WHAT IS TRACKED, AND WHY NOT recipeStore. The brief asked for middleware around the Zustand
// recipe & mask slices. `recipeStore.ts` has no consumer anywhere in the app (nothing imports
// useRecipeStore), so wrapping it would have produced a history of state nothing reads: it would
// have looked finished and undone nothing. The recipe the UI actually edits is the flat `settings`
// object in App.jsx, and the mask / roto / pin state is React state in FaceSwap and
// InteractivePreview. So tracking is by CHANNEL: each owner registers the value it owns with
// useHistoryChannel, and this store keeps the entries, the cursor and the transactions.
//
// An entry is the whole tracked state at a moment: { channelId: value }. A channel that is not
// mounted (FaceSwap's mask state while you are on Editor) keeps its last value in the entry, and
// undo simply skips applying it.

const CHANNELS = new Map();         // id -> { apply, equals, describe, coalesce, fromSlot, scope }
let live = {};                      // the current value of every registered channel
let pending = new Set();            // channels that changed since the last flush
let flushQueued = false;
let history = null;                 // engine history, created on the first observation
let txDepth = 0;
let txHint = null;
let txTimer = null;
let txWatchdog = null;
const appliedListeners = new Set();

const TX_SETTLE_MS = 60;            // after pointerup, let the owner's onPointerUp commit land first
const TX_MAX_MS = 20000;            // a pointerup that never arrives (released outside the window) must not wedge history

const now = () => Date.now();

const equalsMap = () => {
  const eq = {};
  for (const [id, c] of CHANNELS) if (c.equals) eq[id] = c.equals;
  return eq;
};

const publish = (set) => {
  set({
    entries: history.entries,
    index: history.index,
    canUndo: E.canUndo(history),
    canRedo: E.canRedo(history),
    headId: E.head(history).id,
  });
};

const describeAll = (before, after) => {
  const parts = [];
  for (const id of pending) {
    const c = CHANNELS.get(id);
    const b = before?.[id];
    const a = after?.[id];
    if (b === a) continue;
    const text = c?.describe ? c.describe(b, a) : null;
    if (text) parts.push(text);
  }
  if (parts.length === 0) return 'Changed';
  if (parts.length === 1) return parts[0];
  return `${parts[0]} (+${parts.length - 1} more)`;
};

export const useHistoryStore = create((set, get) => {
  // The one place a pending batch becomes an entry.
  const flush = () => {
    flushQueued = false;
    if (!history || pending.size === 0) { pending = new Set(); return; }
    if (txDepth > 0) return;                       // a drag is in flight: one entry when it ends
    const changed = pending;
    const next = { ...E.head(history).state, ...live };
    // Channels only ever enter `live` while registered, so spread order keeps unmounted ones.
    const key = [...changed].sort().join('+');
    const coalesce = [...changed].every((id) => CHANNELS.get(id)?.coalesce !== false) ? key : null;
    const label = txHint;
    txHint = null;
    const h = E.commit(history, next, {
      now: now(),
      eq: equalsMap(),
      coalesceKey: coalesce,
      label,
      describe: (before, after) => describeAll(before, after),
    });
    pending = new Set();
    if (h !== history) { history = h; publish(set); }
  };

  const queueFlush = () => {
    if (flushQueued) return;
    flushQueued = true;
    queueMicrotask(flush);
  };

  const applyState = (state, label, kind) => {
    // Bring every registered channel to `state`. `live` is updated FIRST so the owner's re-render,
    // which hands the same values back to observe(), reads as "no change" instead of a new entry.
    for (const [id, c] of CHANNELS) {
      if (!(id in state)) continue;
      if ((c.equals || Object.is)(live[id], state[id])) continue;
      live = { ...live, [id]: state[id] };
      c.apply(state[id]);
    }
    for (const fn of appliedListeners) fn({ kind, label });
  };

  return {
    entries: [],
    index: 0,
    canUndo: false,
    canRedo: false,
    headId: 0,

    /** An owner reports the current value of its channel. Cheap when nothing changed. */
    observe(id, value) {
      const c = CHANNELS.get(id);
      if (!c) return;
      if (!history) {
        live = { ...live, [id]: value };
        history = E.createHistory({ ...live }, { now: now() });
        publish(set);
        return;
      }
      if (!(id in live)) {
        // A channel arriving late (FaceSwap mounting) is not an edit: fold it into the baseline.
        live = { ...live, [id]: value };
        history = E.adopt(history, id, value);
        return;
      }
      if ((c.equals || Object.is)(live[id], value)) return;
      live = { ...live, [id]: value };
      pending.add(id);
      queueFlush();
    },

    register(id, def) {
      CHANNELS.set(id, def);
      return () => {
        if (CHANNELS.get(id) !== def) return;
        CHANNELS.delete(id);
        const { [id]: _gone, ...rest } = live;
        live = rest;
        pending.delete(id);
      };
    },

    /** Pointer-down on a slider / canvas tool: hold entries back until the gesture ends. */
    beginTransaction(hint = null) {
      if (txTimer) { clearTimeout(txTimer); txTimer = null; }
      txDepth = 1;                                 // gestures do not nest; a second down joins the first
      if (hint) txHint = hint;
      if (txWatchdog) clearTimeout(txWatchdog);
      txWatchdog = setTimeout(() => { txWatchdog = null; get().endTransaction(); }, TX_MAX_MS);
    },

    endTransaction() {
      if (txWatchdog) { clearTimeout(txWatchdog); txWatchdog = null; }
      if (txTimer) clearTimeout(txTimer);
      txTimer = setTimeout(() => {
        txTimer = null;
        txDepth = 0;
        queueFlush();
      }, TX_SETTLE_MS);
    },

    /** Name the entry the current gesture will produce, when the describers cannot ("Added Roto Polygon"). */
    labelNext(text) { txHint = text; },

    inTransaction: () => txDepth > 0,

    undo() {
      if (!history) return false;
      flush();
      const r = E.undo(history);
      if (!r) return false;
      const was = E.head(history);
      history = r.h;
      applyState(r.state, was.label, 'undo');
      publish(set);
      return true;
    },

    redo() {
      if (!history) return false;
      flush();
      const r = E.redo(history);
      if (!r) return false;
      history = r.h;
      applyState(r.state, r.entry.label, 'redo');
      publish(set);
      return true;
    },

    /** Time-travel to entry `i` of `entries` (a click in the history popover). */
    jumpTo(i) {
      if (!history) return false;
      flush();
      const r = E.jump(history, i);
      if (!r) return false;
      history = r.h;
      applyState(r.state, r.entry.label, 'jump');
      publish(set);
      return true;
    },

    /**
     * Apply a state that came from somewhere other than the history (a snapshot slot) AS an action:
     * the channels move, and one entry with `label` records it, so Ctrl+Z flips back.
     */
    restore(partial, label) {
      if (!history) return false;
      flush();
      const next = { ...live };
      for (const [id, v] of Object.entries(partial)) {
        const c = CHANNELS.get(id);
        if (!c) continue;
        next[id] = c.fromSlot ? c.fromSlot(live[id], v) : v;
      }
      applyState(next, label, 'restore');
      const h = E.commit(history, { ...E.head(history).state, ...live }, { now: now(), label, eq: equalsMap() });
      if (h !== history) { history = h; publish(set); }
      return true;
    },

    /** The current value of every mounted channel in `scopes` (for a snapshot slot). */
    snapshot(scopes = ['global']) {
      const out = {};
      for (const [id, c] of CHANNELS) {
        if (id in live && scopes.includes(c.scope || 'global')) out[id] = live[id];
      }
      return out;
    },
  };
});

/** Called after undo / redo / jump / restore with { kind, label }. Returns the unsubscribe. */
export function onHistoryApplied(fn) {
  appliedListeners.add(fn);
  return () => appliedListeners.delete(fn);
}

/** Test seam: drop every channel, entry and timer. */
export function resetHistoryForTests() {
  CHANNELS.clear();
  live = {};
  pending = new Set();
  flushQueued = false;
  history = null;
  txDepth = 0;
  txHint = null;
  if (txTimer) clearTimeout(txTimer);
  if (txWatchdog) clearTimeout(txWatchdog);
  txTimer = null;
  txWatchdog = null;
  appliedListeners.clear();
  useHistoryStore.setState({ entries: [], index: 0, canUndo: false, canRedo: false, headId: 0 });
}
