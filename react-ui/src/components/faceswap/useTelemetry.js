import { useEffect } from 'react';
import { getJSON } from '../../api';
import { setSystemTelemetry, useTelemetryStore } from '../../store/telemetryStore';

// System telemetry (GPU/VRAM/CPU/RAM/threads) for the HUDs.
//
// ONE poll for the whole app, reference-counted by the components that want
// it, writing into the telemetry store. It used to be a private setInterval +
// setState in every component that called this hook, which meant the whole
// Face Swap panel re-rendered every three seconds to move a VRAM figure that
// only the HUD strip displays. Consumers now pick what they show:
//
//   useSystemTelemetryPoller()          keep the poll running (no re-render)
//   useTelemetry()                      the object, re-rendering on change
//                                       (the legacy shape; keep it to leaves)
//   useTelemetryStore((s) => s.system?.vram_used)   one field

let refCount = 0;
let timer = null;
let currentInterval = 3000;

async function tick() {
  try {
    setSystemTelemetry(await getJSON('/api/system/telemetry', { timeout: 8000 }));
  } catch {
    // quiet fail: the HUD keeps its last reading
  }
}

function start(intervalMs) {
  currentInterval = intervalMs;
  tick();
  timer = setInterval(tick, intervalMs);
}

function stop() {
  if (timer) clearInterval(timer);
  timer = null;
}

export function useSystemTelemetryPoller(intervalMs = 3000) {
  useEffect(() => {
    refCount += 1;
    if (refCount === 1) start(intervalMs);
    else if (intervalMs < currentInterval) { stop(); start(intervalMs); }
    return () => {
      refCount -= 1;
      if (refCount === 0) stop();
    };
  }, [intervalMs]);
}

export default function useTelemetry(intervalMs = 3000) {
  useSystemTelemetryPoller(intervalMs);
  return useTelemetryStore((s) => s.system);
}
