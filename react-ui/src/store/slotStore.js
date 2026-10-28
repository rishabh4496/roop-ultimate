import { create } from 'zustand';
import { useHistoryStore } from './historyStore';

// Stage 23: four instant recipe snapshot slots (A/B/C/D) per route.
//
// A slot holds the tracked recipe state (the same { channelId: value } map a history entry holds,
// restricted to the channels that route owns) and, when there was one, a decoded copy of the frame
// that recipe rendered. Recalling a slot applies the state and shows that copy at once, so two
// looks can be flipped between without waiting for a render.
//
// MEMORY. A slot's bitmap is a real allocation (a 2560 px frame is ~26 MB), so a slot owns its
// bitmap outright: whatever it is replaced by, cleared by, or dropped with, `.close()` runs.
// It does NOT run in the same tick. PreviewCanvas keeps the frame it was handed in a layer until
// its next prop update, and drawImage on a closed ImageBitmap throws, so a bitmap is first
// removed from the store (the stage re-renders without it) and closed RETIRE_MS later.

export const SLOT_IDS = ['A', 'B', 'C', 'D'];
export const BANKS = ['faceswap', 'extras'];
export const RETIRE_MS = 250;

const emptyBank = () => ({ A: null, B: null, C: null, D: null });

const retired = [];                 // bitmaps waiting for their stage to let go
let retireTimer = null;

function closeBitmap(bm) {
  try { bm?.close?.(); } catch { /* already closed */ }
}

/** Close everything that has been retired. The timer calls this; tests call it directly. */
export function flushRetired() {
  if (retireTimer) { clearTimeout(retireTimer); retireTimer = null; }
  while (retired.length) closeBitmap(retired.pop());
}

function retire(bm) {
  if (!bm) return;
  retired.push(bm);
  if (!retireTimer) retireTimer = setTimeout(flushRetired, RETIRE_MS);
}

export const useSlotStore = create((set, get) => ({
  banks: { faceswap: emptyBank(), extras: emptyBank() },
  // The slot last captured or recalled in each bank, and the history entry it matched at the time.
  // The pill shows "modified" once the head entry has moved on.
  active: { faceswap: null, extras: null },
  activeHeadId: { faceswap: 0, extras: 0 },
  // The slot compared against the live result in the split wipe.
  reference: { faceswap: null, extras: null },

  /**
   * Store a capture. `bitmap` ownership transfers to the store; the previous occupant's bitmap is
   * retired. `frameKey` ties the bitmap to a media + frame so a recall on a different frame does
   * not show a picture of some other moment.
   */
  put(bank, id, { state, bitmap = null, frameKey = null, label = '' }) {
    const prev = get().banks[bank][id];
    if (prev?.bitmap) retire(prev.bitmap);
    const slot = {
      id, state, bitmap, frameKey, label,
      ts: Date.now(),
      width: bitmap?.width || 0,
      height: bitmap?.height || 0,
    };
    set((s) => ({
      banks: { ...s.banks, [bank]: { ...s.banks[bank], [id]: slot } },
      active: { ...s.active, [bank]: id },
      activeHeadId: { ...s.activeHeadId, [bank]: useHistoryStore.getState().headId },
    }));
    return slot;
  },

  clear(bank, id) {
    const prev = get().banks[bank][id];
    if (!prev) return;
    if (prev.bitmap) retire(prev.bitmap);
    set((s) => ({
      banks: { ...s.banks, [bank]: { ...s.banks[bank], [id]: null } },
      active: { ...s.active, [bank]: s.active[bank] === id ? null : s.active[bank] },
      reference: { ...s.reference, [bank]: s.reference[bank] === id ? null : s.reference[bank] },
    }));
  },

  clearBank(bank) {
    for (const id of SLOT_IDS) get().clear(bank, id);
  },

  setReference(bank, id) {
    set((s) => ({ reference: { ...s.reference, [bank]: id } }));
  },

  /** Mark `id` as the slot now in force (after a recall's history entry exists). */
  markActive(bank, id) {
    set((s) => ({
      active: { ...s.active, [bank]: id },
      activeHeadId: { ...s.activeHeadId, [bank]: useHistoryStore.getState().headId },
    }));
  },
}));

/** Which scopes' channels travel with a bank's slots. */
export const BANK_SCOPES = { faceswap: ['global', 'faceswap'], extras: ['global', 'extras'] };

/** Capture the live recipe state of `bank` into slot `id`; `bitmap` is optional and owned from here on. */
export function captureSlot(bank, id, { bitmap = null, frameKey = null } = {}) {
  const state = useHistoryStore.getState().snapshot(BANK_SCOPES[bank]);
  return useSlotStore.getState().put(bank, id, { state, bitmap, frameKey, label: `Slot ${id}` });
}

/**
 * Apply slot `id` as one undoable action. Returns the slot, or null when it is empty. The caller
 * decides whether the slot's frame can be shown (it must match the media + frame on screen).
 */
export function recallSlot(bank, id) {
  const slot = useSlotStore.getState().banks[bank][id];
  if (!slot) return null;
  useHistoryStore.getState().restore(slot.state, `Recalled Slot ${id}`);
  useSlotStore.getState().markActive(bank, id);
  return slot;
}

/** Test seam. */
export function resetSlotsForTests() {
  flushRetired();
  for (const bank of BANKS) {
    for (const id of SLOT_IDS) {
      const s = useSlotStore.getState().banks[bank][id];
      if (s?.bitmap) closeBitmap(s.bitmap);
    }
  }
  useSlotStore.setState({
    banks: { faceswap: emptyBank(), extras: emptyBank() },
    active: { faceswap: null, extras: null },
    activeHeadId: { faceswap: 0, extras: 0 },
    reference: { faceswap: null, extras: null },
  });
}
