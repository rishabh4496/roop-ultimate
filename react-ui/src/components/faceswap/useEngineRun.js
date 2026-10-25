import { useMemo } from 'react';
import { useTelemetryStore } from '../../store/telemetryStore';

/**
 * The live engine state (is a canary running now, for which model) from the telemetry store. Three primitives, not
 * the whole `run` slice: that object changes on every fast telemetry tick (fps, frame counters) and selecting it
 * would re-render the caller at the socket's rate for a block that rarely changes.
 */
export function useEngineRun() {
  const engineState = useTelemetryStore((s) => s.run.engine_state);
  const engineModel = useTelemetryStore((s) => s.run.engine_model);
  const enginePrecision = useTelemetryStore((s) => s.run.engine_precision);
  return useMemo(
    () => ({ engine_state: engineState, engine_model: engineModel, engine_precision: enginePrecision }),
    [engineState, engineModel, enginePrecision],
  );
}
