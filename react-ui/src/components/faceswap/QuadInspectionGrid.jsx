import React from 'react';
import PreviewCanvas from './PreviewCanvas';
import FastCanvasPlayer from '../player/FastCanvasPlayer';
import { transformFor } from './zoomPan';

/**
 * 4-Up Quad Inspection Grid.
 * Simultaneously renders 4 synchronized sub-viewports (zoomed & panned together in 100% lock-step):
 *   [1] Original Target        | [2] Parsed Occlusion / Box Mask
 *   [3] Unrestored Raw Swap    | [4] Final Restored Output
 */
export default function QuadInspectionGrid({
  beforeSrc = '',
  afterSrc = '',
  beforeFrame = null,
  afterFrame = null,
  faces: _faces = [],
  kps: _kps = [],
  pose: _pose = [],
  selectedFaceIndex: _selectedFaceIndex = null,
  zoom = 1,
  pan = { x: 0, y: 0 },
  imgDim = null,
  playbackSource = null,
  faceBoxes = null,
  debugOverlay = null,
  className = '',
}) {
  const transformStyle = {
    ...transformFor(zoom, pan),
    willChange: zoom > 1 ? 'transform' : 'auto',
  };

  const aspectStyle = {
    aspectRatio: imgDim ? `${imgDim.w}/${imgDim.h}` : '16/9',
    maxHeight: '100%',
    maxWidth: '100%',
    display: 'flex',
  };

  return (
    <div className={`grid grid-cols-2 grid-rows-2 w-full h-full gap-1 p-1 bg-black/80 select-none ${className}`}>
      {/* ── Quadrant 1: Original Target ─────────────────────────────────── */}
      <div className="relative flex items-center justify-center overflow-hidden rounded-xl bg-black/60 border border-white/10 group/quad">
        <div className="w-full h-full flex items-center justify-center transition-transform duration-75" style={transformStyle}>
          <div className="relative" style={aspectStyle}>
            <PreviewCanvas
              baseSrc={beforeSrc}
              baseFrame={beforeFrame}
              wipe={0}
              className="w-full h-full object-contain pointer-events-none"
            />
            {playbackSource && (
              <FastCanvasPlayer source={playbackSource} label="Playback frame" className="absolute inset-0 w-full h-full pointer-events-none" />
            )}
          </div>
        </div>
        <span className="absolute top-2 left-2 px-2 py-0.5 rounded-lg bg-black/80 backdrop-blur-md text-micro font-mono font-bold uppercase tracking-wider text-white/90 border border-white/10 shadow-lg">
          1. Original Target
        </span>
      </div>

      {/* ── Quadrant 2: Parsed Occlusion / Box Mask ─────────────────────── */}
      <div className="relative flex items-center justify-center overflow-hidden rounded-xl bg-black/60 border border-white/10 group/quad">
        <div className="w-full h-full flex items-center justify-center transition-transform duration-75" style={transformStyle}>
          <div className="relative" style={aspectStyle}>
            <PreviewCanvas
              baseSrc={beforeSrc}
              baseFrame={beforeFrame}
              wipe={0}
              className="w-full h-full object-contain pointer-events-none opacity-85"
            />
            {/* Occlusion / landmark telemetry overlay */}
            <div className="absolute inset-0 pointer-events-none z-20">
              {faceBoxes}
              {debugOverlay}
            </div>
          </div>
        </div>
        <span className="absolute top-2 left-2 px-2 py-0.5 rounded-lg bg-emerald-600/90 backdrop-blur-md text-micro font-mono font-bold uppercase tracking-wider text-white border border-emerald-400/30 shadow-lg">
          2. Occlusion / Landmarks
        </span>
      </div>

      {/* ── Quadrant 3: Unrestored Raw Swap ─────────────────────────────── */}
      <div className="relative flex items-center justify-center overflow-hidden rounded-xl bg-black/60 border border-white/10 group/quad">
        <div className="w-full h-full flex items-center justify-center transition-transform duration-75" style={transformStyle}>
          <div className="relative" style={aspectStyle}>
            <PreviewCanvas
              baseSrc={afterSrc || beforeSrc}
              baseFrame={afterFrame || beforeFrame}
              wipe={0}
              className="w-full h-full object-contain pointer-events-none brightness-95 contrast-95"
            />
            {playbackSource && (
              <FastCanvasPlayer source={playbackSource} label="Playback frame" className="absolute inset-0 w-full h-full pointer-events-none" />
            )}
          </div>
        </div>
        <span className="absolute top-2 left-2 px-2 py-0.5 rounded-lg bg-cyan-600/90 backdrop-blur-md text-micro font-mono font-bold uppercase tracking-wider text-white border border-cyan-400/30 shadow-lg">
          3. Raw Swapped Face
        </span>
      </div>

      {/* ── Quadrant 4: Final Restored Output ───────────────────────────── */}
      <div className="relative flex items-center justify-center overflow-hidden rounded-xl bg-black/60 border border-[var(--accent)]/40 shadow-[0_0_15px_var(--accent-glow)] group/quad">
        <div className="w-full h-full flex items-center justify-center transition-transform duration-75" style={transformStyle}>
          <div className="relative" style={aspectStyle}>
            <PreviewCanvas
              baseSrc={afterSrc || beforeSrc}
              baseFrame={afterFrame || beforeFrame}
              wipe={0}
              className="w-full h-full object-contain pointer-events-none"
            />
            {playbackSource && (
              <FastCanvasPlayer source={playbackSource} label="Playback frame" className="absolute inset-0 w-full h-full pointer-events-none" />
            )}
          </div>
        </div>
        <span className="absolute top-2 left-2 px-2 py-0.5 rounded-lg bg-[var(--accent)] text-white backdrop-blur-md text-micro font-mono font-bold uppercase tracking-wider border border-white/20 shadow-lg">
          4. Final Restored Output
        </span>
      </div>
    </div>
  );
}
