import { useHistoryStore } from '../store/historyStore';
import { eventMatches } from '../store/shortcutStore';

// Stage 23: the window-level half of the history. Two jobs:
//
//  1. TRANSACTIONS. A slider drag fires onChange dozens of times a second, a vertex drag moves a
//     polygon on every pointermove; neither may push an undo state per event. A capture-phase
//     pointerdown on a range input (or anything marked data-history-tx) begins a transaction, the
//     matching pointerup ends it, and the store makes ONE entry for the whole gesture. Doing it once
//     here, rather than adding onPointerDown/onPointerUp to every slider, covers every range input
//     in the app including ones added later. It is a capture listener on purpose: RotoLayer and the
//     stage's gesture layer call stopPropagation() on pointerdown, which would hide it from a
//     bubble-phase listener.
//
//  2. KEYS. Undo and redo (Ctrl/Cmd+Z, Ctrl/Cmd+Y / Ctrl/Cmd+Shift+Z unless remapped in Settings).

const NON_TEXT_INPUT = new Set(['range', 'checkbox', 'radio', 'button', 'submit', 'reset', 'color', 'file', 'image']);

/** True when the browser's own text undo should win. A focused range slider is NOT text. */
export function isTextEntry(el) {
  if (!el || typeof el !== 'object') return false;
  if (el.isContentEditable) return true;
  const tag = String(el.tagName || '').toLowerCase();
  if (tag === 'textarea') return true;
  if (tag === 'input') return !NON_TEXT_INPUT.has(String(el.type || 'text').toLowerCase());
  return false;
}

/**
 * 'undo' | 'redo' | null for a keydown, before any focus rules. The keys are the `edit.undo` / `edit.redo` actions
 * (shortcuts/actions.js: Ctrl/Cmd+Z, Ctrl/Cmd+Y and Ctrl/Cmd+Shift+Z by default), so a remap reaches this listener.
 */
export function historyKeyAction(e) {
  if (eventMatches(e, 'edit.undo')) return 'undo';
  if (eventMatches(e, 'edit.redo')) return 'redo';
  return null;
}

const TX_SELECTOR = 'input[type="range"], [data-history-tx]';

export function installHistoryDom(win = window, doc = document) {
  let ours = false;

  const onDown = (e) => {
    const t = e.target;
    if (!t || typeof t.closest !== 'function' || !t.closest(TX_SELECTOR)) return;
    ours = true;
    useHistoryStore.getState().beginTransaction();
  };
  const onUp = () => {
    if (!ours) return;
    ours = false;
    useHistoryStore.getState().endTransaction();
  };

  const onKey = (e) => {
    const action = historyKeyAction(e);
    if (!action) return;
    if (isTextEntry(e.target) || isTextEntry(doc.activeElement)) return;
    // A modal owns the keyboard; undoing the page underneath it is a surprise.
    if (doc.querySelector('[role="dialog"]')) return;
    e.preventDefault();
    const s = useHistoryStore.getState();
    if (s.inTransaction()) return;                  // mid-drag: the gesture is not an entry yet
    if (action === 'undo') s.undo(); else s.redo();
  };

  win.addEventListener('pointerdown', onDown, true);
  win.addEventListener('pointerup', onUp, true);
  win.addEventListener('pointercancel', onUp, true);
  win.addEventListener('blur', onUp);
  win.addEventListener('keydown', onKey);
  return () => {
    win.removeEventListener('pointerdown', onDown, true);
    win.removeEventListener('pointerup', onUp, true);
    win.removeEventListener('pointercancel', onUp, true);
    win.removeEventListener('blur', onUp);
    win.removeEventListener('keydown', onKey);
  };
}
