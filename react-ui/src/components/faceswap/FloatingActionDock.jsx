import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from '../../motion';
import { Icon } from '../../icons';
import { useTelemetryStore } from '../../store/telemetryStore';

// The badge states what /api/system/telemetry reports, not what the app hopes for. It used to
// read "GPU Turbo" with a "TensorRT FP16 + NVDEC + NVENC + Batch Swap" title on every machine,
// including CPU / DirectML ones and while the real precision was `mixed`.
function useAccelerationBadge() {
  const turbo = useTelemetryStore((s) => s.system?.turbo_active);
  const provider = useTelemetryStore((s) => s.system?.provider);
  const precision = useTelemetryStore((s) => s.system?.trt_precision);
  const nvdec = useTelemetryStore((s) => s.system?.nvdec_active);
  const nvenc = useTelemetryStore((s) => s.system?.nvenc_active);
  const batch = useTelemetryStore((s) => s.system?.batch_swap_active);
  if (!provider) return null;
  const parts = [
    provider === 'tensorrt' ? `TensorRT${precision ? ` ${precision}` : ''}` : provider,
    nvdec && 'NVDEC',
    nvenc && 'NVENC',
    batch && 'batched swap',
  ].filter(Boolean);
  return { turbo: !!turbo, label: turbo ? 'GPU Turbo' : provider, title: `Active: ${parts.join(' + ')}` };
}

/**
 * FloatingActionDock — the always-visible action bar at bottom centre: preview,
 * workspace layout, ambilight, preset studio, pop-out, drawer toggles. (Not
 * Start/Cancel: the run bar owns those. The Face Swap tab reserves room below its
 * content for this dock and a scroll-padding for focus, so it covers nothing.)
 *
 * Colour here follows the design tokens, and that is load-bearing rather than
 * tidiness. This bar is on screen on every tab, and it used to carry FOUR
 * accent hues that no theme could move — indigo for active states, blue for
 * previewing, amber for the ambilight toggle, rose for cancel — plus a
 * `from-blue-600 via-indigo-600 to-purple-600` gradient and an
 * `shadow-indigo-500/25` glow on Start Swap. Against Obsidian's crimson accent
 * that is five competing hues in one 820px strip.
 *
 * So: one accent (`.fill-accent`, `.state-accent`), `--danger` for cancel, and
 * neutrals for everything at rest. The motion is untouched — the spring hover
 * and tap are the approved "bold & cinematic" behaviour, not decoration.
 */
export default function FloatingActionDock({
  workspaceMode,
  setWorkspaceMode,
  onPreview,
  previewing,
  ambilightEnabled,
  setAmbilightEnabled,
  onOpenPopout,
  onOpenPresetStudio,
  drawers,
  setDrawers,
}) {
  const [showModeMenu, setShowModeMenu] = useState(false);
  const modeMenuRef = useRef(null);
  const accel = useAccelerationBadge();

  // The layout menu closes on Escape and on a click anywhere else, like every other popover;
  // it used to stay open until its own trigger was pressed again.
  useEffect(() => {
    if (!showModeMenu) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); setShowModeMenu(false); } };
    const onDown = (e) => { if (!modeMenuRef.current?.contains(e.target)) setShowModeMenu(false); };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('pointerdown', onDown, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('pointerdown', onDown, true);
    };
  }, [showModeMenu]);

  // Descriptions state what each mode ACTUALLY does — the table these map onto
  // lives in FaceSwap.jsx (WORKSPACE_LAYOUT). Keep the two in step.
  // The icon is its own field. It used to be an emoji glued to the front of
  // `label`, which the trigger below then had to split back apart on a space —
  // so any label whose first word was not an icon silently rendered wrong.
  const MODES = [
    { id: 'default', icon: Icon.layout, label: 'Standard Studio', desc: 'Faces, settings and timeline' },
    { id: 'cinema', icon: Icon.faceswap, label: 'Cinema Canvas', desc: 'Canvas only — panels and timeline hidden' },
    { id: 'dual', icon: Icon.faces, label: 'Dual Inspector', desc: 'Faces + parameters, no timeline' },
    { id: 'timeline', icon: Icon.film, label: 'Timeline Deck', desc: 'Canvas and timeline, panels hidden' },
  ];

  const currentMode = MODES.find((m) => m.id === workspaceMode) || MODES[0];

  return (
    // `w-max`: a fixed box at `left: 50%` shrink-wraps to HALF the viewport, so at 1280px the
    // labels broke onto two lines ("GPU / Turbo", "Pop- / out") and the dock doubled in height
    // over the content. `whitespace-nowrap` holds each label on one line; the cap keeps it on screen.
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 w-max max-w-[calc(100vw-2rem)] whitespace-nowrap pointer-events-auto transition-all duration-300">
      <div className="relative flex flex-wrap items-center justify-center gap-1 rounded-2xl border border-white/15 bg-black/85 px-2 py-2 shadow-2xl backdrop-blur-xl ring-1 ring-white/10 sm:gap-2.5 sm:rounded-full sm:px-4 sm:py-2.5">
        
        {/* There is no Start / Cancel here. The run bar at the top of the
            workspace is the ONE run control (it turns into Cancel while a job is
            in flight, and says in words why it is unavailable). This dock used to
            carry a second Start that was never disabled, so with no media loaded
            it posted a swap the run bar had correctly refused. */}

        {/* Instant Preview Button */}
        <button
          onClick={onPreview}
          disabled={previewing}
          className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-all ${
            previewing
              ? 'state-accent'
              : 'text-white/70 hover:bg-white/10 hover:text-white'
          }`}
          title="Generate instant preview frame"
          aria-label="Generate instant preview frame"
        >
          <Icon.preview size={13} />
          <span>{previewing ? 'Rendering...' : 'Preview'}</span>
        </button>

        {/* Acceleration badge: the provider the backend reports, green only when it says turbo is active. */}
        {accel && (
          <div
            className={`hidden md:flex items-center gap-1.5 rounded-full px-2.5 py-1 text-nano font-bold border select-none cursor-default ${
              accel.turbo
                ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
                : 'bg-white/5 border-white/15 text-white/70'}`}
            title={accel.title}
            data-testid="dock-acceleration"
          >
            <span className={`h-1.5 w-1.5 rounded-full ${accel.turbo ? 'bg-emerald-400 animate-pulse' : 'bg-white/40'}`} />
            <span>{accel.label}</span>
          </div>
        )}

        <div aria-hidden="true" className="hidden h-4 w-px bg-white/15 sm:block" />

        {/* Workspace Layout Switcher Menu */}
        <div className="relative" ref={modeMenuRef}>
          <button
            onClick={() => setShowModeMenu(!showModeMenu)}
            className="flex items-center gap-1.5 rounded-full bg-white/5 border border-white/10 px-3 py-1.5 text-xs font-medium text-white/80 transition-all hover:bg-white/10 hover:text-white"
            title="Switch Workspace Layout Mode"
            aria-label={`Workspace layout: ${currentMode.label}. Change layout`}
            aria-expanded={showModeMenu}
          >
            <currentMode.icon size={13} />
            <span className="hidden sm:inline">{currentMode.label}</span>
            <span className="text-micro opacity-60">▼</span>
          </button>

          <AnimatePresence>
            {showModeMenu && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                className="absolute bottom-full left-0 mb-3 w-56 rounded-2xl border border-white/15 bg-black/85 p-2 shadow-2xl backdrop-blur-2xl"
              >
                <div className="px-3 py-1.5 text-micro font-semibold uppercase tracking-[0.14em] text-muted">
                  Workspace Layout Mode
                </div>
                <div className="space-y-1">
                  {MODES.map((mode) => (
                    <button
                      key={mode.id}
                      onClick={() => {
                        setWorkspaceMode(mode.id);
                        setShowModeMenu(false);
                      }}
                      className={`w-full rounded-xl px-3 py-2 text-left transition-all ${
                        workspaceMode === mode.id
                          ? 'state-accent font-semibold'
                          : 'text-white/70 hover:bg-white/10 hover:text-white'
                      }`}
                    >
                      <div className="text-xs flex items-center gap-2"><mode.icon size={13} />{mode.label}</div>
                      <div className="text-note text-white/50 pl-[21px]">{mode.desc}</div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div aria-hidden="true" className="hidden h-4 w-px bg-white/15 sm:block" />

        {/* Ambilight Glow Toggle */}
        <button
          onClick={() => setAmbilightEnabled(!ambilightEnabled)}
          className={`flex items-center gap-1 rounded-full px-2.5 py-1.5 text-xs transition-all ${
            ambilightEnabled
              ? 'state-accent'
              : 'text-white/55 hover:bg-white/10 hover:text-white/85'
          }`}
          title="Toggle Canvas Ambilight Glow"
          aria-pressed={ambilightEnabled}
        >
          <Icon.full size={13} />
          <span className="hidden md:inline">Glow</span>
        </button>

        {/* Preset Studio Trigger */}
        {onOpenPresetStudio && (
          <button
            onClick={onOpenPresetStudio}
            className="flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs text-white/70 transition-all hover:bg-white/10 hover:text-white"
            title="Open Preset Studio & Quality Recipes"
          >
            <Icon.theme size={13} />
            <span className="hidden md:inline">Recipes</span>
          </button>
        )}

        {/* External Pop-out Window */}
        <button
          onClick={onOpenPopout}
          className="flex items-center gap-1 rounded-full px-2.5 py-1.5 text-xs text-white/55 transition-all hover:bg-white/10 hover:text-white/85"
          title="Open Detached Pop-out Preview Monitor"
        >
          <Icon.popout size={13} />
          <span className="hidden md:inline">Pop-out</span>
        </button>

        <div aria-hidden="true" className="hidden h-4 w-px bg-white/15 sm:block" />

        {/* Panel Drawer Collapse Toggles */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setDrawers((d) => ({ ...d, left: !d.left }))}
            className={`rounded-lg p-1.5 text-xs transition-all ${
              drawers.left ? 'ink-accent bg-white/10' : 'text-muted hover:text-white/70'
            }`}
            title="Toggle the settings inspector (left)"
            aria-label="Toggle the settings inspector"
            aria-pressed={drawers.left}
          >
            <Icon.panelLeft size={14} />
          </button>

          <button
            onClick={() => setDrawers((d) => ({ ...d, right: !d.right }))}
            className={`rounded-lg p-1.5 text-xs transition-all ${
              drawers.right ? 'ink-accent bg-white/10' : 'text-muted hover:text-white/70'
            }`}
            title="Toggle the media and faces column (right)"
            aria-label="Toggle the media and faces column"
            aria-pressed={drawers.right}
          >
            <Icon.panelRight size={14} />
          </button>

          <button
            onClick={() => setDrawers((d) => ({ ...d, bottom: !d.bottom }))}
            className={`rounded-lg p-1.5 text-xs transition-all ${
              drawers.bottom ? 'ink-accent bg-white/10' : 'text-muted hover:text-white/70'
            }`}
            title="Toggle Bottom Timeline & Logs Deck"
            aria-label="Toggle the timeline and logs deck"
            aria-pressed={drawers.bottom}
          >
            <Icon.panelBottom size={14} />
          </button>
        </div>

      </div>
    </div>
  );
}
