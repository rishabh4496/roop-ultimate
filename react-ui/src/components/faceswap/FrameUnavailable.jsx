import React from 'react';

// "Frame unavailable - Retry": the stage could not fetch the frame for the
// playhead after every retry (useThrottledFrameRequest), so whatever picture is
// up -- a swap of an earlier frame, or nothing -- is not the one asked for. Said
// out loud, with the one thing the user can do about it.
//
// Pointer and double-click events are stopped at the button: this renders over
// InteractivePreview's stage, whose container starts a pan on pointer-down and
// toggles zoom on double-click, and a click on Retry must be only a click.
export default function FrameUnavailable({ onRetry, className = '' }) {
  return (
    <div
      role="alert"
      data-testid="frame-unavailable"
      className={`flex justify-center pointer-events-none ${className}`}
    >
      <div className="pointer-events-auto inline-flex items-center gap-2.5 px-3 py-1.5 rounded-xl bg-black/85 backdrop-blur-md border border-amber-400/40 shadow-2xl text-mini font-semibold text-amber-200">
        Frame unavailable
        <span aria-hidden="true" className="text-white/30">-</span>
        <button
          type="button"
          onClick={onRetry}
          onPointerDown={(e) => e.stopPropagation()}
          onDoubleClick={(e) => e.stopPropagation()}
          className="px-2 py-0.5 rounded-lg bg-white/10 hover:bg-white/20 text-white font-bold transition-colors"
        >
          Retry
        </button>
      </div>
    </div>
  );
}
