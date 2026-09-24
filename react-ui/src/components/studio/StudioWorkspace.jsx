import React, { useState, useRef, useEffect } from 'react';
import {
  Layout, Monitor, Film, Users, Layers, Activity, RotateCcw,
  Check, SunMedium, ChevronDown
} from 'lucide-react';
import { useWorkspaceLayoutStore, WORKSPACE_PRESETS } from './workspaceLayoutStore';
import HardwareTelemetryHud from '../telemetry/HardwareTelemetryHud';
import RenderQueueDrawer from '../queue/RenderQueueDrawer';

/**
 * High-performance Studio Workspace with dockable panel management,
 * Rec.709 color-grading neutral studio mode, and localStorage persistence.
 */
export default function StudioWorkspace({
  renderPreview,
  renderTimeline,
  renderFaceBank,
  notify,
  className = '',
}) {
  // ── HOOK DECLARATIONS AT TOP ──────────────────────────────────────────────
  const {
    activePreset,
    panels,
    panelSizes,
    studioTheme,
    setPreset,
    togglePanel,
    setStudioTheme,
    resetLayout,
  } = useWorkspaceLayoutStore();

  const [presetDropdownOpen, setPresetDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setPresetDropdownOpen(false);
      }
    };
    document.addEventListener('pointerdown', handleOutsideClick);
    return () => document.removeEventListener('pointerdown', handleOutsideClick);
  }, []);

  const isRec709 = studioTheme === 'rec709_neutral';

  return (
    <div
      className={`flex flex-col h-full overflow-hidden select-none transition-colors duration-300 ${
        isRec709
          ? 'bg-[#121212] text-[#E0E0E0]'
          : 'bg-[#06070B] text-[#F4F5F7]'
      } ${className}`}
    >
      {/* ── STUDIO TOP BAR & WORKSPACE CONTROLS ───────────────────────────── */}
      <div
        className={`flex items-center justify-between px-3 h-10 border-b shrink-0 z-30 ${
          isRec709
            ? 'bg-[#181818] border-[#2A2A2A]'
            : 'bg-white/[0.02] border-white/10'
        }`}
      >
        {/* Left: Layout Presets & Mode Selector */}
        <div className="flex items-center gap-2">
          {/* Preset Dropdown */}
          <div className="relative" ref={dropdownRef}>
            <button
              type="button"
              onClick={() => setPresetDropdownOpen(!presetDropdownOpen)}
              aria-label="Select studio workspace layout preset"
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-mini font-medium transition-colors ${
                isRec709
                  ? 'bg-[#222222] border-[#333333] text-white hover:bg-[#282828]'
                  : 'bg-white/5 border-white/10 text-white/90 hover:bg-white/10'
              }`}
            >
              <Layout size={13} className={isRec709 ? 'text-gray-300' : 'text-rose-400'} aria-hidden="true" />
              <span>{WORKSPACE_PRESETS[activePreset]?.label || 'Custom Workspace'}</span>
              <ChevronDown size={12} className="text-white/40" aria-hidden="true" />
            </button>

            {presetDropdownOpen && (
              <div
                className={`absolute left-0 top-full mt-1 w-64 rounded-xl border shadow-2xl p-1.5 z-50 flex flex-col gap-0.5 ${
                  isRec709
                    ? 'bg-[#1C1C1C] border-[#333333]'
                    : 'bg-[#0D0E15] border-white/15'
                }`}
              >
                {Object.entries(WORKSPACE_PRESETS).map(([key, item]) => {
                  const isSelected = activePreset === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => {
                        setPreset(key);
                        setPresetDropdownOpen(false);
                      }}
                      aria-label={`Switch layout to ${item.label}`}
                      className={`flex flex-col text-left px-2.5 py-1.5 rounded-lg text-mini transition-colors ${
                        isSelected
                          ? isRec709
                            ? 'bg-[#2C2C2C] text-white font-medium'
                            : 'bg-rose-500/20 text-rose-300 font-medium'
                          : 'text-white/70 hover:text-white hover:bg-white/5'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span>{item.label}</span>
                        {isSelected && <Check size={12} aria-hidden="true" />}
                      </div>
                      <span className="text-nano text-white/40 mt-0.5">{item.description}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Quick Panel Dock Buttons */}
          <div className="flex items-center gap-1 ml-2 border-l border-white/10 pl-2">
            <button
              type="button"
              onClick={() => togglePanel('preview')}
              aria-label={panels.preview ? 'Hide Preview panel' : 'Show Preview panel'}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-micro font-medium transition-colors ${
                panels.preview
                  ? isRec709 ? 'bg-[#2C2C2C] text-white' : 'bg-rose-500/20 text-rose-300'
                  : 'text-white/40 hover:text-white/70'
              }`}
            >
              <Monitor size={11} aria-hidden="true" />
              <span>Preview</span>
            </button>

            <button
              type="button"
              onClick={() => togglePanel('timeline')}
              aria-label={panels.timeline ? 'Hide Timeline panel' : 'Show Timeline panel'}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-micro font-medium transition-colors ${
                panels.timeline
                  ? isRec709 ? 'bg-[#2C2C2C] text-white' : 'bg-rose-500/20 text-rose-300'
                  : 'text-white/40 hover:text-white/70'
              }`}
            >
              <Film size={11} aria-hidden="true" />
              <span>Timeline</span>
            </button>

            <button
              type="button"
              onClick={() => togglePanel('facebank')}
              aria-label={panels.facebank ? 'Hide Face Bank panel' : 'Show Face Bank panel'}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-micro font-medium transition-colors ${
                panels.facebank
                  ? isRec709 ? 'bg-[#2C2C2C] text-white' : 'bg-rose-500/20 text-rose-300'
                  : 'text-white/40 hover:text-white/70'
              }`}
            >
              <Users size={11} aria-hidden="true" />
              <span>Face Bank</span>
            </button>

            <button
              type="button"
              onClick={() => togglePanel('queue')}
              aria-label={panels.queue ? 'Hide Render Queue drawer' : 'Show Render Queue drawer'}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-micro font-medium transition-colors ${
                panels.queue
                  ? isRec709 ? 'bg-[#2C2C2C] text-white' : 'bg-rose-500/20 text-rose-300'
                  : 'text-white/40 hover:text-white/70'
              }`}
            >
              <Layers size={11} aria-hidden="true" />
              <span>Queue</span>
            </button>

            <button
              type="button"
              onClick={() => togglePanel('telemetry')}
              aria-label={panels.telemetry ? 'Hide Hardware Telemetry HUD' : 'Show Hardware Telemetry HUD'}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-micro font-medium transition-colors ${
                panels.telemetry
                  ? isRec709 ? 'bg-[#2C2C2C] text-white' : 'bg-rose-500/20 text-rose-300'
                  : 'text-white/40 hover:text-white/70'
              }`}
            >
              <Activity size={11} aria-hidden="true" />
              <span>Telemetry</span>
            </button>
          </div>
        </div>

        {/* Right: Studio Color Grading & Reset Controls */}
        <div className="flex items-center gap-2">
          {/* Rec.709 Neutral Reference Mode Toggle */}
          <button
            type="button"
            onClick={() => setStudioTheme(isRec709 ? 'obsidian' : 'rec709_neutral')}
            aria-label={isRec709 ? 'Switch to Obsidian Dark theme' : 'Switch to Rec.709 neutral color grading mode'}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-micro font-medium border transition-colors ${
              isRec709
                ? 'bg-[#2E2E2E] text-white border-[#444444] shadow-sm'
                : 'bg-white/5 hover:bg-white/10 text-white/60 hover:text-white/90 border-white/10'
            }`}
          >
            <SunMedium size={12} className={isRec709 ? 'text-amber-300' : 'text-white/40'} aria-hidden="true" />
            <span>Rec.709 Neutral Mode</span>
            {isRec709 && (
              <span className="w-2 h-2 rounded-full bg-white ml-0.5 shadow-sm" title="D65 100% White Reference" />
            )}
          </button>

          {/* Reset Workspace */}
          <button
            type="button"
            onClick={resetLayout}
            aria-label="Reset workspace to default layout"
            title="Reset Layout"
            className="p-1.5 rounded hover:bg-white/10 text-white/40 hover:text-white transition-colors"
          >
            <RotateCcw size={13} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* ── WORKSPACE BODY ────────────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* CENTER COLUMN: PREVIEW + TIMELINE */}
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* PREVIEW VIEWPORT AREA */}
          {panels.preview && (
            <div
              className={`flex-1 flex items-center justify-center relative overflow-hidden ${
                isRec709
                  ? 'border-2 border-[#2B2B2B] bg-[#141414]'
                  : 'border-b border-white/10 bg-black/40'
              }`}
            >
              {/* Rec.709 D65 Reference Notch */}
              {isRec709 && (
                <div className="absolute top-2 right-2 flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#1F1F1F] border border-[#333333] text-nano font-mono text-white/50 z-20">
                  <div className="w-2.5 h-2.5 rounded-full bg-white shadow" title="D65 White Point (6504K)" />
                  <span>Rec.709 D65 Neutral</span>
                </div>
              )}

              {/* Injected Preview Component */}
              <div className="w-full h-full">
                {renderPreview ? renderPreview({ isRec709 }) : (
                  <div className="flex items-center justify-center h-full text-white/30 text-mini">
                    <span>Cinematic Preview Viewport</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TIMELINE DECK AREA */}
          {panels.timeline && (
            <div
              style={{ height: `${panelSizes.timelineHeight}px` }}
              className={`w-full shrink-0 border-t ${
                isRec709
                  ? 'border-[#2B2B2B] bg-[#161616]'
                  : 'border-white/10 bg-[#0A0B10]'
              }`}
            >
              {renderTimeline ? renderTimeline({ isRec709 }) : (
                <div className="flex items-center justify-center h-full text-white/30 text-mini">
                  <span>Cinematic Timeline Deck</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* RIGHT / SIDE DRAWER: FACE BANK ROUTER */}
        {panels.facebank && (
          <div
            style={{ width: `${panelSizes.facebankWidth}px` }}
            className={`border-l shrink-0 flex flex-col overflow-hidden ${
              isRec709
                ? 'border-[#2B2B2B] bg-[#181818]'
                : 'border-white/10 bg-[#0A0B11]'
            }`}
          >
            {renderFaceBank ? renderFaceBank({ isRec709 }) : (
              <div className="flex items-center justify-center h-full text-white/30 text-mini">
                <span>Face Bank Identity Router</span>
              </div>
            )}
          </div>
        )}

        {/* FLOATING HARDWARE TELEMETRY HUD */}
        {panels.telemetry && (
          <HardwareTelemetryHud
            onClose={() => togglePanel('telemetry')}
          />
        )}
      </div>

      {/* ── BOTTOM DOCK: RENDER QUEUE DRAWER ──────────────────────────────── */}
      <RenderQueueDrawer
        isOpen={panels.queue}
        onToggleOpen={() => togglePanel('queue')}
        notify={notify}
      />
    </div>
  );
}
