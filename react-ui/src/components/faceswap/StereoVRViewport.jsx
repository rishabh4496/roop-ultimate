import React, { useEffect, useRef, useState, useCallback } from 'react';
import { GlRenderer } from '../player/glRenderer';

export const STEREO_INSPECTION_MODES = [
  { id: 'off', label: 'Standard / Off' },
  { id: 'left', label: 'Left Eye' },
  { id: 'right', label: 'Right Eye' },
  { id: 'side_by_side', label: 'Side-by-Side' },
  { id: 'anaglyph', label: 'Red-Cyan Anaglyph' },
  { id: 'disparity_diff', label: 'Stereo Disparity Diff Map' },
];

export const VR_PROJECTION_MODES = [
  { id: 'off', label: 'Off (2D Flat)' },
  { id: 'vr180', label: 'VR 180° Equirectangular' },
  { id: 'vr360', label: 'VR 360° Equirectangular' },
];

export const STEREO_LAYOUTS = [
  { id: 'sbs', label: 'Side-by-Side (SBS)' },
  { id: 'tb', label: 'Top-Bottom (Over-Under)' },
  { id: 'dual', label: 'Dual Texture (A / B)' },
];

export const DEFAULT_VR_PARAMS = {
  vrMode: 'off',
  yaw: 0.0,            // radians
  pitch: 0.0,          // radians
  fov: 90.0,           // degrees
  stereoMode: 'off',
  stereoLayout: 'sbs',
};

/**
 * Stage 21 — WebGL2 Equirectangular VR180/360 Viewport & Stereoscopic 3D Inspector.
 *
 * Implements:
 *   - Interactive Pan-Tilt-FOV drag and wheel zoom.
 *   - Equirectangular 180/360-degree spherical ray-traced projection in WebGL2.
 *   - Stereo inspection: Left Eye, Right Eye, Side-by-Side, Red-Cyan Anaglyph, and Disparity Diff Map.
 */
export default function StereoVRViewport({
  active = false,
  onClose = null,
  mediaSource = null,        // HTMLVideoElement | HTMLImageElement | ImageBitmap | Canvas | string
  compareSource = null,      // Optional second layer for Dual A/B inspection
  params = DEFAULT_VR_PARAMS,
  onChange = null,
  className = '',
}) {
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0, yaw: 0, pitch: 0 });
  const rafRef = useRef(0);

  const [vrParams, setVrParams] = useState({ ...DEFAULT_VR_PARAMS, ...params });

  // Sync internal state if external params change
  useEffect(() => {
    if (params) setVrParams((prev) => ({ ...prev, ...params }));
  }, [params]);

  const updateParam = useCallback((patch) => {
    setVrParams((prev) => {
      const next = { ...prev, ...patch };
      if (onChange) onChange(next);
      return next;
    });
  }, [onChange]);

  // Init WebGL2 GlRenderer on the dedicated canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !active) return undefined;

    const renderer = new GlRenderer(canvas);
    rendererRef.current = renderer;

    const handleResize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(rect.width * dpr));
      const h = Math.max(1, Math.round(rect.height * dpr));
      renderer.resize(w, h);
    };

    handleResize();
    let ro = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(handleResize);
      ro.observe(canvas);
    }

    return () => {
      ro?.disconnect();
      renderer.destroy();
      rendererRef.current = null;
    };
  }, [active]);

  // Render Loop
  useEffect(() => {
    if (!active) return undefined;
    const renderer = rendererRef.current;
    if (!renderer) return undefined;

    let cancelled = false;

    const loop = () => {
      if (cancelled) return;

      if (mediaSource) {
        try {
          renderer.setLayer(0, mediaSource);
          if (compareSource) {
            renderer.setLayer(1, compareSource);
          }
        } catch {
          // ignore pending texture decode
        }
      }

      renderer.setVrParams(vrParams);
      renderer.draw();

      // If video or playing, keep pumping rAF
      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [active, mediaSource, compareSource, vrParams]);

  // Mouse / Pointer Drag Handlers (Pan & Tilt)
  const handlePointerDown = (e) => {
    if (e.button !== 0) return;
    isDraggingRef.current = true;
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      yaw: vrParams.yaw,
      pitch: vrParams.pitch,
    };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (!isDraggingRef.current) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;

    // Sensitivity adjusted by FOV
    const fovFactor = (vrParams.fov || 90.0) / 90.0;
    const sens = 0.0035 * fovFactor;

    let newYaw = dragStartRef.current.yaw - dx * sens;
    // Normalize yaw to -PI..PI
    while (newYaw > Math.PI) newYaw -= 2 * Math.PI;
    while (newYaw < -Math.PI) newYaw += 2 * Math.PI;

    const maxPitch = (85.0 * Math.PI) / 180.0;
    const newPitch = Math.max(-maxPitch, Math.min(maxPitch, dragStartRef.current.pitch + dy * sens));

    updateParam({ yaw: newYaw, pitch: newPitch });
  };

  const handlePointerUp = (e) => {
    isDraggingRef.current = false;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
  };

  // Wheel Zoom (FOV)
  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 3.0 : -3.0;
    const nextFov = Math.max(30.0, Math.min(120.0, (vrParams.fov || 90.0) + delta));
    updateParam({ fov: nextFov });
  };

  // Double click resets view
  const handleDoubleClick = () => {
    updateParam({ yaw: 0.0, pitch: 0.0, fov: 90.0 });
  };

  if (!active) return null;

  const yawDeg = ((vrParams.yaw * 180) / Math.PI).toFixed(1);
  const pitchDeg = ((vrParams.pitch * 180) / Math.PI).toFixed(1);
  const fovDeg = (vrParams.fov || 90).toFixed(0);

  return (
    <div
      className={`absolute inset-0 z-40 bg-black/90 flex flex-col select-none overflow-hidden ${className}`}
      data-testid="stereo-vr-viewport"
    >
      {/* Top Floating Glass HUD Toolbar */}
      <div className="absolute top-3 left-3 right-3 z-50 flex flex-wrap items-center justify-between gap-2 p-2 rounded-xl bg-black/60 backdrop-blur-md border border-white/10 shadow-2xl text-xs text-white">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-lg bg-cyan-500/20 border border-cyan-400/40 text-cyan-300 font-bold text-micro tracking-wider uppercase">
            3D / VR Viewport
          </div>

          {/* Projection Mode */}
          <label className="flex items-center gap-1">
            <span className="text-white/60 text-micro">Projection:</span>
            <select
              value={vrParams.vrMode}
              onChange={(e) => updateParam({ vrMode: e.target.value })}
              className="bg-white/10 rounded px-1.5 py-0.5 border border-white/20 text-white text-micro focus:outline-none"
            >
              {VR_PROJECTION_MODES.map((m) => (
                <option key={m.id} value={m.id} className="bg-neutral-900 text-white">
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          {/* Stereo Inspection Mode */}
          <label className="flex items-center gap-1">
            <span className="text-white/60 text-micro">Stereo Mode:</span>
            <select
              value={vrParams.stereoMode}
              onChange={(e) => updateParam({ stereoMode: e.target.value })}
              className="bg-white/10 rounded px-1.5 py-0.5 border border-white/20 text-white text-micro focus:outline-none"
            >
              {STEREO_INSPECTION_MODES.map((m) => (
                <option key={m.id} value={m.id} className="bg-neutral-900 text-white">
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          {/* Stereo Layout */}
          <label className="flex items-center gap-1">
            <span className="text-white/60 text-micro">Layout:</span>
            <select
              value={vrParams.stereoLayout}
              onChange={(e) => updateParam({ stereoLayout: e.target.value })}
              className="bg-white/10 rounded px-1.5 py-0.5 border border-white/20 text-white text-micro focus:outline-none"
            >
              {STEREO_LAYOUTS.map((m) => (
                <option key={m.id} value={m.id} className="bg-neutral-900 text-white">
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        {/* Readouts and Actions */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-nano tabular-nums text-white/70">
            <span>Yaw: <strong className="text-white">{yawDeg}°</strong></span>
            <span>Pitch: <strong className="text-white">{pitchDeg}°</strong></span>
            <span>FOV: <strong className="text-white">{fovDeg}°</strong></span>
          </div>

          <button
            type="button"
            onClick={handleDoubleClick}
            title="Reset Pan/Tilt/FOV (or double click stage)"
            className="px-2 py-0.5 rounded text-micro bg-white/10 hover:bg-white/20 border border-white/20 text-white transition-colors"
          >
            Reset
          </button>

          {onClose && (
            <button
              type="button"
              onClick={onClose}
              title="Close 3D / VR Viewport"
              className="px-2 py-0.5 rounded text-micro bg-rose-500/30 hover:bg-rose-500/50 border border-rose-400/40 text-rose-200 transition-colors font-bold"
            >
              ✕ Exit VR
            </button>
          )}
        </div>
      </div>

      {/* Interactive WebGL2 Media Canvas */}
      <div
        className="relative flex-1 w-full h-full cursor-grab active:cursor-grabbing flex items-center justify-center overflow-hidden"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onWheel={handleWheel}
        onDoubleClick={handleDoubleClick}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-full object-contain pointer-events-none"
        />

        {/* Subtle center crosshair in VR projection modes */}
        {vrParams.vrMode !== 'off' && (
          <div className="absolute pointer-events-none w-4 h-4 border border-white/30 rounded-full flex items-center justify-center opacity-40">
            <div className="w-0.5 h-0.5 bg-white/70 rounded-full" />
          </div>
        )}

        {/* Navigation hint bar */}
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 pointer-events-none text-nano text-white/40 bg-black/40 px-3 py-1 rounded-full backdrop-blur-sm">
          Drag to Pan/Tilt · Scroll to Zoom FOV · Double-click to Reset
        </div>
      </div>
    </div>
  );
}
