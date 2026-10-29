import { useCallback, useEffect, useRef, useState } from 'react';
import { SCRUB_THRESHOLD_PX, advanceScrub, beginScrub } from './scrubMath';

// Pointer handlers that turn a horizontal drag on ANY element into a stream of values. The element is a mouse
// affordance only (a label, a number badge): it is not focusable and adds no tab stop, because the control it
// scrubs already has its own keyboard route (arrow keys on the slider, the text field).
//
// DELTA TRACKING, not pointer lock. Pointer capture keeps delivering moves after the pointer leaves the element - and
// the window - for as long as the button is down, which is the property pointer lock is usually reached for. Lock would
// also hide the cursor and swallow Escape, and it cannot be exercised from a test; capture can.
//
// A press that never travels SCRUB_THRESHOLD_PX is a CLICK and is left completely alone (the badge opens its editor).
// Past it, the press becomes a drag: the click that would follow is swallowed so releasing a scrub on the badge does
// not also open the editor, and the page cursor is held at ew-resize for the whole drag so it does not flicker back to
// an arrow when the pointer outruns the label.

export function useScrub({ getValue, min, max, step, onScrub, onStart, onEnd, disabled = false }) {
  const [scrubbing, setScrubbing] = useState(false);
  const drag = useRef(null);                  // { id, x0, lastX, state, engaged }
  const swallowClick = useRef(false);
  const saved = useRef(null);                 // body styles to restore
  const cfg = useRef({});
  cfg.current = { getValue, min, max, step, onScrub, onStart, onEnd };

  const restoreBody = useCallback(() => {
    if (!saved.current || typeof document === 'undefined') return;
    document.body.style.cursor = saved.current.cursor;
    document.body.style.userSelect = saved.current.userSelect;
    saved.current = null;
  }, []);

  const finish = useCallback((e) => {
    const d = drag.current;
    if (!d || (e && e.pointerId !== undefined && e.pointerId !== d.id)) return;
    drag.current = null;
    try { e?.currentTarget?.releasePointerCapture?.(d.id); } catch { /* capture already gone */ }
    if (d.engaged) {
      swallowClick.current = true;
      restoreBody();
      setScrubbing(false);
      cfg.current.onEnd?.();
    }
  }, [restoreBody]);

  // A drag in flight when the component unmounts must not leave the page stuck on a resize cursor.
  useEffect(() => () => { drag.current = null; swallowClick.current = false; restoreBody(); }, [restoreBody]);

  const onPointerDown = useCallback((e) => {
    if (disabled || e.button !== 0) return;
    swallowClick.current = false;
    drag.current = { id: e.pointerId, x0: e.clientX, lastX: e.clientX, state: null, engaged: false };
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* a synthetic pointer cannot be captured */ }
  }, [disabled]);

  const onPointerMove = useCallback((e) => {
    const d = drag.current;
    if (!d || e.pointerId !== d.id) return;
    const c = cfg.current;
    if (!d.engaged) {
      if (Math.abs(e.clientX - d.x0) < SCRUB_THRESHOLD_PX) return;
      d.engaged = true;
      d.state = beginScrub(c.getValue(), { min: c.min, max: c.max, step: c.step });
      d.lastX = e.clientX;                    // the first counted move starts here: no jump by the threshold distance
      if (typeof document !== 'undefined') {
        saved.current = { cursor: document.body.style.cursor, userSelect: document.body.style.userSelect };
        document.body.style.cursor = 'ew-resize';
        document.body.style.userSelect = 'none';
      }
      setScrubbing(true);
      c.onStart?.();
      return;
    }
    const dx = e.clientX - d.lastX;
    d.lastX = e.clientX;
    if (dx === 0) return;
    d.state = advanceScrub(d.state, dx, { shiftKey: e.shiftKey, altKey: e.altKey });
    c.onScrub?.(d.state.value);
  }, []);

  const onClickCapture = useCallback((e) => {
    if (!swallowClick.current) return;
    swallowClick.current = false;
    e.preventDefault();
    e.stopPropagation();
  }, []);

  return {
    scrubbing,
    bind: {
      onPointerDown,
      onPointerMove,
      onPointerUp: finish,
      onPointerCancel: finish,
      onClickCapture,
      // Vertical pans still scroll the page; only the horizontal drag is ours.
      style: { touchAction: 'pan-y' },
      // An empty data-history-tx makes the history store treat the whole drag as ONE undo entry (historyDom.js).
      'data-history-tx': '',
    },
  };
}
