import React from 'react';

/**
 * Pro-Workstation Floating HUD Control Bar for Media Preview.
 * Provides instant access to inspection modes, tracking toggles, loupe, brush, and zoom.
 */
export function PreviewControlsHUD({
  zoom = 1,
  onZoomIn,
  onZoomOut,
  onZoomReset,
  onZoomActual,
  showDetections = true,
  onToggleDetections,
  showLandmarks = true,
  onToggleLandmarks,
  showLabels = true,
  onToggleLabels,
  _overlayOpacity = 1.0,
  _onChangeOpacity,
  magnifierActive = false,
  onToggleMagnifier,
  maskBrushActive = false,
  onToggleMaskBrush,
  compare = false,
  onToggleCompare,
  compareMode = 'slider',
  onChangeCompareMode,
  compareDir = 'vertical',
  onToggleCompareDir,
  autoSwipe = false,
  onToggleAutoSwipe,
  isFullscreen = false,
  onToggleFullscreen,
  hasMedia = false,
  faceCount = 0,
}) {
  if (!hasMedia) return null;

  return (
    <div className="absolute inset-x-0 bottom-3 z-40 flex justify-center px-4 pointer-events-none">
      <div className="pointer-events-auto flex flex-wrap items-center gap-1 p-1.5 rounded-xl bg-[#090a0f]/90 border border-white/10 backdrop-blur-xl shadow-2xl text-white select-none">
        {/* Face Tracking & Landmarks Group */}
        <div className="flex items-center gap-0.5 pr-1.5 border-r border-white/10">
          <button
            type="button"
            onClick={onToggleDetections}
            title="Toggle Face Bounding Boxes"
            className={`px-2 py-1 rounded-lg text-[11px] font-semibold transition-colors flex items-center gap-1 ${
              showDetections
                ? 'bg-[var(--accent-primary,#38bdf8)]/20 text-[var(--accent-primary,#38bdf8)] border border-[var(--accent-primary,#38bdf8)]/40'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            <span>Face Boxes</span>
            {faceCount > 0 && (
              <span className="px-1 py-0.2 rounded bg-white/15 text-[9px] font-mono">
                {faceCount}
              </span>
            )}
          </button>

          <button
            type="button"
            onClick={onToggleLandmarks}
            title="Toggle 5-Point Landmarks & 3D Pose (Hotkey D)"
            className={`px-2 py-1 rounded-lg text-[11px] font-semibold transition-colors ${
              showLandmarks
                ? 'bg-amber-400/20 text-amber-300 border border-amber-400/40'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            Pose & Kps [D]
          </button>

          <button
            type="button"
            onClick={onToggleLabels}
            title="Toggle Person Identity Labels"
            className={`px-1.5 py-1 rounded-lg text-[11px] font-semibold transition-colors ${
              showLabels
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            Labels
          </button>
        </div>

        {/* Pro Tools Group: Loupe & Mask Brush */}
        <div className="flex items-center gap-0.5 px-1.5 border-r border-white/10">
          <button
            type="button"
            onClick={onToggleMagnifier}
            title="Toggle 3.5x Magnifying Loupe (Hotkey G)"
            className={`px-2 py-1 rounded-lg text-[11px] font-semibold transition-colors flex items-center gap-1 ${
              magnifierActive
                ? 'bg-[var(--accent-primary,#38bdf8)] text-black font-bold'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            <span>Loupe [G]</span>
          </button>

          <button
            type="button"
            onClick={onToggleMaskBrush}
            title="Toggle Face Mask Brush Tool (Hotkey B)"
            className={`px-2 py-1 rounded-lg text-[11px] font-semibold transition-colors flex items-center gap-1 ${
              maskBrushActive
                ? 'bg-pink-500 text-white font-bold'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            <span>Brush [B]</span>
          </button>
        </div>

        {/* Comparison Suite Group */}
        <div className="flex items-center gap-0.5 px-1.5 border-r border-white/10">
          <button
            type="button"
            onClick={onToggleCompare}
            title="Toggle Before/After Comparison Mode"
            className={`px-2 py-1 rounded-lg text-[11px] font-semibold transition-colors ${
              compare
                ? 'bg-emerald-500/25 text-emerald-300 border border-emerald-500/40 font-bold'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            Compare
          </button>

          {compare && (
            <>
              <button
                type="button"
                onClick={() =>
                  onChangeCompareMode?.(
                    compareMode === 'slider' ? 'blend' : compareMode === 'blend' ? 'diff' : 'slider',
                  )
                }
                title="Cycle Comparison Mode: Wipe Slider / Blend / Diff (Hotkey O)"
                className="px-1.5 py-1 rounded-lg text-[10px] font-mono uppercase bg-white/10 hover:bg-white/20 text-white/80"
              >
                {compareMode}
              </button>

              {compareMode === 'slider' && (
                <>
                  <button
                    type="button"
                    onClick={onToggleCompareDir}
                    title="Toggle Split Axis: Vertical / Horizontal (Hotkey X)"
                    className="px-1.5 py-1 rounded-lg text-[10px] font-mono bg-white/10 hover:bg-white/20 text-white/80"
                  >
                    {compareDir === 'vertical' ? 'V' : 'H'}
                  </button>

                  <button
                    type="button"
                    onClick={onToggleAutoSwipe}
                    title="Toggle Sinusoidal Auto-Swipe (Hotkey A)"
                    className={`px-1.5 py-1 rounded-lg text-[10px] font-mono transition-colors ${
                      autoSwipe ? 'bg-[var(--accent-primary,#38bdf8)] text-black font-bold' : 'bg-white/10 text-white/80'
                    }`}
                  >
                    Auto
                  </button>
                </>
              )}
            </>
          )}
        </div>

        {/* Zoom & Viewport Group */}
        <div className="flex items-center gap-1 pl-1.5">
          <button
            type="button"
            onClick={onZoomOut}
            title="Zoom Out"
            disabled={zoom <= 1}
            className="w-6 h-6 rounded flex items-center justify-center text-xs text-white/70 hover:text-white hover:bg-white/10 disabled:opacity-30"
          >
            -
          </button>

          <span className="text-[11px] font-mono text-white/80 tabular-nums px-1">
            {Math.round(zoom * 100)}%
          </span>

          <button
            type="button"
            onClick={onZoomIn}
            title="Zoom In"
            disabled={zoom >= 8}
            className="w-6 h-6 rounded flex items-center justify-center text-xs text-white/70 hover:text-white hover:bg-white/10 disabled:opacity-30"
          >
            +
          </button>

          <button
            type="button"
            onClick={onZoomReset}
            title="Fit to Canvas"
            className="px-1.5 py-0.5 rounded text-[10px] font-mono text-white/60 hover:text-white hover:bg-white/10"
          >
            Fit
          </button>

          <button
            type="button"
            onClick={onZoomActual}
            title="100% Native Pixel Scale"
            className="px-1.5 py-0.5 rounded text-[10px] font-mono text-white/60 hover:text-white hover:bg-white/10"
          >
            1:1
          </button>

          <button
            type="button"
            onClick={onToggleFullscreen}
            title="Fullscreen Mode"
            className="w-6 h-6 rounded flex items-center justify-center text-xs text-white/70 hover:text-white hover:bg-white/10"
          >
            {isFullscreen ? '⤢' : '⤡'}
          </button>
        </div>
      </div>
    </div>
  );
}
