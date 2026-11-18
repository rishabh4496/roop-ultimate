import React from 'react';
import TelemetrySparklines from '../telemetry/TelemetrySparklines';
import EnvironmentHealth from '../EnvironmentHealth';
import RuntimeSections from '../RuntimeSections';

// The Runtime Diagnostics tab as one multi-column dashboard.
//
// Top: the four live sparklines across the full width. Below: a two-column band
// with the standing environment health and compatibility check on the left and
// the benchmark runner (with its thread-scaling chart) on the right. The fourteen
// runtime sections fill the bottom in their own grid. On a narrow window everything
// stacks in one column; nothing is hidden at any width.

export default function DiagnosticsGrid({ notify, meta, runtime, benchmark }) {
  return (
    <div className="space-y-4" data-testid="diagnostics-grid">
      <TelemetrySparklines />
      <div className="grid grid-cols-1 2xl:grid-cols-12 gap-4 items-start">
        <div className="space-y-4 min-w-0 2xl:col-span-5">
          <EnvironmentHealth notify={notify} meta={meta} />
        </div>
        <div className="space-y-4 min-w-0 2xl:col-span-7">
          {benchmark}
        </div>
      </div>
      <RuntimeSections runtime={runtime} />
    </div>
  );
}
