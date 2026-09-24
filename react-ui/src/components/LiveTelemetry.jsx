import React, { useCallback, useEffect, useLayoutEffect, useRef } from 'react';
import { UI_HZ, useTelemetryStore } from '../store/telemetryStore';
import { subscribeThrottled, useThrottledSelector } from '../store/liveSubscribe';

// ── Leaf subscribers for high-frequency telemetry ─────────────────────────
//
// The rule these exist to enforce: a number that changes several times a
// second re-renders the smallest thing that shows it, and never more than
// UI_HZ (10) times a second — never the panel it sits in.
//
//   <LiveText select={fn} />   writes textContent directly; React never
//                              re-renders it for a telemetry change at all.
//   <LiveBar select={fn} />    same, for a bar's width (0..1 -> %).
//   <LiveValue select={fn}>{(v) => ...}</LiveValue>
//                              a render prop, for markup that needs React
//                              (a component taking the value as a prop).
//   useThrottledSelector(fn)   the hook under LiveValue.
//
// Throttling is leading-and-trailing: the first change after a quiet period
// shows at once, a burst coalesces to one update per 1000/hz ms, and the LAST
// value of a burst is always shown (a trailing timer), so a run that ends
// between two ticks does not leave a stale figure on screen.

/** Render prop over useThrottledSelector. */
export function LiveValue({ select, hz = UI_HZ, equals, children }) {
  const v = useThrottledSelector(select, { hz, equals });
  return children(v);
}

/**
 * Text that follows the store. React renders the element EMPTY and never owns
 * its text node: the text is written through the ref, so a direct DOM write
 * and a React reconcile can never fight over the same node (React would
 * otherwise keep patching a text node the DOM write had already detached).
 *
 * `select` may close over props; it is re-applied after every render of the
 * parent, so a prop change shows immediately rather than at the next tick.
 */
export function LiveText({ select, hz = UI_HZ, as: Tag = 'span', ...rest }) {
  const ref = useRef(null);
  const selRef = useRef(select);
  selRef.current = select;
  const write = useCallback((state) => {
    const el = ref.current;
    if (!el) return;
    const v = selRef.current(state);
    const text = v == null ? '' : String(v);
    if (el.textContent !== text) el.textContent = text;
  }, []);
  useLayoutEffect(() => { write(useTelemetryStore.getState()); });
  useEffect(() => subscribeThrottled(write, hz), [write, hz]);
  return <Tag ref={ref} {...rest} />;
}

/**
 * A bar whose width follows the store. `select` returns 0..1; `minPct` keeps a
 * sliver visible at 0 the way the old bars did. `aria` (optional) returns the
 * progressbar attributes to keep in step for assistive tech.
 */
export function LiveBar({ select, minPct = 0, hz = UI_HZ, aria, style, ...rest }) {
  const ref = useRef(null);
  const selRef = useRef(select);
  selRef.current = select;
  const ariaRef = useRef(aria);
  ariaRef.current = aria;
  const write = useCallback((state) => {
    const el = ref.current;
    if (!el) return;
    const f = Number(selRef.current(state));
    const pct = Math.max(minPct, Number.isFinite(f) ? Math.min(1, Math.max(0, f)) * 100 : 0);
    const w = `${pct}%`;
    if (el.style.width !== w) el.style.width = w;
    const host = el.parentElement;
    const a = ariaRef.current?.(state);
    if (a && host) for (const [k, val] of Object.entries(a)) host.setAttribute(k, String(val));
  }, [minPct]);
  useLayoutEffect(() => { write(useTelemetryStore.getState()); });
  useEffect(() => subscribeThrottled(write, hz), [write, hz]);
  return <div ref={ref} style={style} {...rest} />;
}
