import { useEffect, useRef, useState } from 'react';
import { UI_HZ, useTelemetryStore } from './telemetryStore';

// Throttled subscriptions to the telemetry store — the plumbing under the
// <LiveText>/<LiveBar>/<LiveValue> components (components/LiveTelemetry.jsx),
// kept in a plain module so that file exports only components.

export const shallowEqual = (a, b) => {
  if (Object.is(a, b)) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  const ka = Object.keys(a);
  if (ka.length !== Object.keys(b).length) return false;
  return ka.every((k) => Object.is(a[k], b[k]));
};

/**
 * Subscribe `onFlush` to the store, called at most `hz` times a second.
 * Returns an unsubscribe. Used by every component below.
 */
export function subscribeThrottled(onFlush, hz = UI_HZ) {
  const gap = 1000 / Math.max(0.05, hz);
  let last = -Infinity;
  let timer = null;
  const flush = () => {
    timer = null;
    last = performance.now();
    onFlush(useTelemetryStore.getState());
  };
  const onChange = () => {
    if (timer) return;
    const wait = Math.max(0, gap - (performance.now() - last));
    timer = setTimeout(flush, wait);
  };
  const unsub = useTelemetryStore.subscribe(onChange);
  return () => {
    unsub();
    if (timer) clearTimeout(timer);
  };
}

export function useThrottledSelector(selector, { hz = UI_HZ, equals = Object.is } = {}) {
  const selRef = useRef(selector);
  selRef.current = selector;
  const eqRef = useRef(equals);
  eqRef.current = equals;
  const [value, setValue] = useState(() => selector(useTelemetryStore.getState()));
  const valueRef = useRef(value);
  useEffect(() => {
    const apply = (state) => {
      const v = selRef.current(state);
      if (!eqRef.current(v, valueRef.current)) {
        valueRef.current = v;
        setValue(v);
      }
    };
    // A change may have landed between the initial render and this effect.
    apply(useTelemetryStore.getState());
    return subscribeThrottled(apply, hz);
  }, [hz]);
  return value;
}

