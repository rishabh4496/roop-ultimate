import React from 'react';
import { Icon } from '../../icons';

/**
 * ProcessingDock
 * Interactive control dock positioned inside the processing stage box.
 * Gives direct control over pause/resume, desktop alerts, and cancellation.
 *
 * There was a "Power Saver / Eco Mode" button here too. It flipped a flag that
 * nothing read, and there is nothing for it to drive: the process deliberately
 * opts OUT of Windows EcoQoS and holds a wake lock for the whole run (a render
 * froze outright when the display powered off), and thread count is fixed when
 * a run starts. A control that cannot change anything is worse than no control.
 *
 * "Lite UI" is the one that does drive something, and it is not about power: it
 * drops this window's own GPU cost (backdrop blurs, looping animations) while a
 * run is in flight, because Chromium composites it on the same GPU the swap is
 * saturating. Left switchable so the difference can be watched on the FPS trend
 * rather than believed.
 */
export default function ProcessingDock({
  paused = false,
  onTogglePause,
  onCancelJob,
  desktopAlerts = false,
  onToggleDesktopAlerts,
  renderLite = true,
  onToggleRenderLite,
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-2xl border border-white/10 bg-neutral-950/70 backdrop-blur-md">
      <div className="flex items-center gap-2">
        {/* Pause / Resume Button */}
        <button
          onClick={onTogglePause}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all shadow-md active:scale-95 ${
            paused
              ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/20'
              : 'bg-amber-600 hover:bg-amber-500 text-white shadow-amber-600/20'
          }`}
          title={paused ? 'Resume execution' : 'Pause execution'}
        >
          {paused ? <Icon.play size={13} /> : <Icon.pause size={13} />}
          <span>{paused ? 'Resume Render' : 'Pause Render'}</span>
        </button>

        {/* Cancel Job Button */}
        <button
          onClick={onCancelJob}
          className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-rose-500/10 border border-rose-500/30 text-rose-300 hover:bg-rose-500/20 hover:text-rose-200 transition-all active:scale-95"
          title="Cancel current swap run"
        >
          <Icon.stop size={13} />
          <span>Cancel Job</span>
        </button>
      </div>

      <div className="flex items-center gap-2">
        {/* Lite UI — drop this window's own GPU cost while the render runs */}
        <button
          onClick={onToggleRenderLite}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
            renderLite
              ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
              : 'bg-white/5 text-neutral-400 border-white/10 hover:text-neutral-200'
          }`}
          title={renderLite
            ? 'Lite UI on — panel blurs and looping animations are off while rendering, so the browser leaves the GPU to the swap'
            : 'Lite UI off — the full interface is being composited on the same GPU as the render'}
        >
          {renderLite ? <Icon.lite size={13} /> : <Icon.full size={13} />}
          <span className="hidden sm:inline">{renderLite ? 'Lite UI' : 'Full UI'}</span>
        </button>

        {/* Desktop Alert Notification Toggle */}
        <button
          onClick={onToggleDesktopAlerts}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
            desktopAlerts
              ? 'bg-blue-500/20 text-blue-300 border-blue-500/40'
              : 'bg-white/5 text-neutral-400 border-white/10 hover:text-neutral-200'
          }`}
          title="Send desktop alert notification when job finishes"
          aria-pressed={desktopAlerts}
          aria-label="Desktop alerts when the job finishes"
        >
          <Icon.bell size={13} />
          <span className="hidden sm:inline">{desktopAlerts ? 'Alerts On' : 'Desktop Alerts'}</span>
        </button>
      </div>
    </div>
  );
}
