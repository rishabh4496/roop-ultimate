import { useEffect, useSyncExternalStore } from 'react';
import { useTelemetryStore } from '../../store/telemetryStore';
import { useSystemTelemetryPoller } from '../faceswap/useTelemetry';
import { WINDOW_MS, pushSample, sampleOf } from './rollingSeries';

// The 60-second buffer behind the sparklines.
//
// It lives at module level, not in component state: leaving the Settings tab
// and coming back should show the last minute, not an empty graph. It samples
// once a second WHILE a consumer is mounted (ref-counted), so an idle app with
// nothing on screen does no work at all.
//
// Samples are read from the telemetry store, which is what the header pill
// reads too, so the two can never disagree about the same moment.

const TICK_MS = 1000;
let samples = [];
let listeners = new Set();
let timer = null;
let refs = 0;

const snapshot = () => samples;

function emit() {
  samples = samples.slice();
  listeners.forEach((fn) => fn());
}

function record(now = Date.now()) {
  const { system, run } = useTelemetryStore.getState();
  const next = pushSample(samples, sampleOf(system, run, now), WINDOW_MS);
  samples = next;
  emit();
}

function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function start() {
  if (timer) return;
  record();
  timer = setInterval(() => record(), TICK_MS);
}

function stop() {
  if (timer) clearInterval(timer);
  timer = null;
}

/** Test seam: the buffer as it stands, and a way to feed it a fixed clock. */
export const __rollingBuffer = {
  read: snapshot,
  reset() { samples = []; emit(); },
  record,
};

/**
 * The rolling window, re-rendering once a second while mounted. Also keeps the
 * system telemetry poll running for as long as the consumer is on screen.
 */
export function useRollingTelemetry() {
  useSystemTelemetryPoller();
  useEffect(() => {
    refs += 1;
    start();
    return () => {
      refs -= 1;
      if (refs <= 0) { refs = 0; stop(); }
    };
  }, []);
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}
