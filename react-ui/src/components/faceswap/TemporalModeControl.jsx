import React from 'react';
import { TEMPORAL_MODES, temporalModeOf, temporalNote, temporalPatch } from './temporalModes';
import { Disclosure } from './InspectorGroup';

// ── Temporal stability: one choice over a dozen switches ─────────────────────
//
// Off / Smart-Skip ROI / Balanced EMA / Cinema Cut-Aware, as a segmented
// control. Picking one writes that mode's patch (faceswap/temporalModes.js); the
// highlighted segment is READ BACK from the settings, so editing any single
// parameter in the drawer below moves the control to "Custom" instead of
// leaving a mode lit that the settings no longer hold.
//
// `children` is the parameter drawer: the existing stabiliser controls, owned
// by FaceSwap so their ranges, defaults and help text stay where they were.
export default function TemporalModeControl({ settings, onApply, children }) {
  const mode = temporalModeOf(settings);

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between gap-2">
        <span id="temporal-mode-label" className="text-xs font-medium text-white/70">Stability mode</span>
        {mode === 'custom' && (
          <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-2 py-0.5 text-micro font-bold uppercase tracking-wide text-amber-300">Custom</span>
        )}
      </div>

      <div role="group" aria-labelledby="temporal-mode-label" className="grid grid-cols-2 gap-1.5">
        {TEMPORAL_MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            aria-pressed={mode === m.id}
            title={m.hint}
            onClick={() => onApply(temporalPatch(m.id))}
            className={`rounded-lg border px-2 py-1.5 text-mini font-semibold leading-tight transition-colors ${
              mode === m.id
                ? 'border-[var(--accent)]/50 bg-[var(--accent)]/15 text-white'
                : 'border-white/10 bg-white/[0.04] text-white/70 hover:border-white/25 hover:text-white'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <p className="text-mini leading-snug text-muted">{temporalNote(settings)}</p>

      {children && (
        <Disclosure id="inspector.temporal-params" title="Custom parameters">
          {children}
        </Disclosure>
      )}
    </div>
  );
}
