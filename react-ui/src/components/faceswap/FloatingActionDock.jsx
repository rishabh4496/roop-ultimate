import React, { useState } from 'react';
import { motion, AnimatePresence } from '../../motion';
import { Icon } from '../../icons';

/**
 * FloatingActionDock — the always-visible action bar at bottom centre: render,
 * preview, workspace layout, ambilight, preset studio, pop-out, drawer toggles.
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
  isRendering,
  onStartSwap,
  onCancelSwap,
  progress,
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
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 pointer-events-auto transition-all duration-300">
      <div className="relative flex items-center gap-2.5 rounded-full border border-white/15 bg-black/85 px-4 py-2.5 shadow-2xl backdrop-blur-xl ring-1 ring-white/10">
        
        {/* Render Button with Live Status Ring */}
        <div className="relative flex items-center">
          {isRendering ? (
            <button
              onClick={onCancelSwap}
              className="fill-danger relative flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold shadow-lg transition-all active:scale-95"
              title="Cancel current swap job"
            >
              {/* A steady dot, not `animate-ping`. The progress number next to it
                  already says the job is live, and a pulsing ring on the one
                  control the eye is drawn to is the decoration this UI dropped. */}
              <span className="h-1.5 w-1.5 rounded-full bg-white/90" />
              <span>Cancel ({Math.round(progress)}%)</span>
            </button>
          ) : (
            <button
              onClick={onStartSwap}
              className="fill-accent group relative flex items-center gap-2 overflow-hidden rounded-full px-5 py-2 text-xs font-bold shadow-lg transition-all hover:scale-105 active:scale-95"
              title="Start Face Swap processing"
            >
              <span className="text-base group-hover:scale-110 transition-transform">▶</span>
              <span>Start Swap</span>
            </button>
          )}
        </div>

        <div className="h-4 w-px bg-white/15" />

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

        <div className="h-4 w-px bg-white/15" />

        {/* Workspace Layout Switcher Menu */}
        <div className="relative">
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
                <div className="px-3 py-1.5 text-micro font-semibold uppercase tracking-[0.14em] text-white/45">
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
                      <div className="text-micro text-white/50 pl-[21px]">{mode.desc}</div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="h-4 w-px bg-white/15" />

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

        <div className="h-4 w-px bg-white/15" />

        {/* Panel Drawer Collapse Toggles */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setDrawers((d) => ({ ...d, left: !d.left }))}
            className={`rounded-lg p-1.5 text-xs transition-all ${
              drawers.left ? 'ink-accent bg-white/10' : 'text-white/40 hover:text-white/70'
            }`}
            title="Toggle Left Faces Sidebar"
            aria-label="Toggle the faces sidebar"
            aria-pressed={drawers.left}
          >
            <Icon.panelLeft size={14} />
          </button>

          <button
            onClick={() => setDrawers((d) => ({ ...d, right: !d.right }))}
            className={`rounded-lg p-1.5 text-xs transition-all ${
              drawers.right ? 'ink-accent bg-white/10' : 'text-white/40 hover:text-white/70'
            }`}
            title="Toggle Right Settings Inspector"
            aria-label="Toggle the settings inspector"
            aria-pressed={drawers.right}
          >
            <Icon.panelRight size={14} />
          </button>

          <button
            onClick={() => setDrawers((d) => ({ ...d, bottom: !d.bottom }))}
            className={`rounded-lg p-1.5 text-xs transition-all ${
              drawers.bottom ? 'ink-accent bg-white/10' : 'text-white/40 hover:text-white/70'
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
