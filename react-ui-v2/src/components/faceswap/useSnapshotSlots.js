import { useCallback, useMemo, useRef } from 'react';
import { useSlotStore, captureSlot, recallSlot } from '../../store/slotStore';

// One bank of four snapshot slots for a route ('faceswap' | 'extras').
//
// The route supplies three things it alone knows:
//   captureFrame()   -> Promise<{ bitmap, frameKey } | null>   a decoded copy of what the stage
//                       shows now. The bitmap is handed to the slot, which closes it when done.
//   onRecalled(slot) a recall has applied the recipe; the route decides whether the slot's frame
//                       may be shown (it must be of the media + frame on screen)
//   onReference(id)  the split-wipe reference changed
//
// Returns activate() for pills AND hotkeys, so a click and the `1`..`4` keys cannot diverge.
export default function useSnapshotSlots({ bank, captureFrame, onRecalled, onReference, notify }) {
  const live = useRef({});
  live.current = { captureFrame, onRecalled, onReference, notify };
  const busy = useRef(false);

  const capture = useCallback(async (id) => {
    if (busy.current) return;
    busy.current = true;
    try {
      let cap = null;
      try { cap = await live.current.captureFrame?.(); } catch { cap = null; }
      captureSlot(bank, id, { bitmap: cap?.bitmap || null, frameKey: cap?.frameKey || null });
      live.current.notify?.(
        cap?.bitmap ? `Slot ${id} captured: recipe and frame` : `Slot ${id} captured: recipe only (no rendered frame to keep)`,
        'info',
      );
    } finally {
      busy.current = false;
    }
  }, [bank]);

  /** Empty slot or `overwrite` captures; a filled slot recalls. */
  const activate = useCallback((id, { overwrite = false } = {}) => {
    const filled = useSlotStore.getState().banks[bank][id];
    if (!filled || overwrite) return capture(id);
    const slot = recallSlot(bank, id);
    if (slot) live.current.onRecalled?.(slot);
    return Promise.resolve();
  }, [bank, capture]);

  const clear = useCallback((id) => {
    useSlotStore.getState().clear(bank, id);
    live.current.onReference?.(useSlotStore.getState().reference[bank]);
  }, [bank]);

  const setReference = useCallback((id) => {
    useSlotStore.getState().setReference(bank, id);
    live.current.onReference?.(id);
  }, [bank]);

  return useMemo(() => ({ bank, activate, capture, clear, setReference }), [bank, activate, capture, clear, setReference]);
}
