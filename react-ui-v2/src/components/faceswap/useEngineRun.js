import { useShallow } from 'zustand/react/shallow';
import { useTelemetryStore } from '../../store/telemetryStore';

/**
 * The live engine state (is a canary running now, for which model) from the telemetry store.
 * Subscribes with atomic shallow selector to avoid redundant re-renders.
 */
export function useEngineRun() {
  return useTelemetryStore(
    useShallow((s) => ({
      engine_state: s.run?.engine_state || 'IDLE',
      engine_model: s.run?.engine_model || '',
      engine_precision: s.run?.engine_precision || '',
    }))
  );
}
