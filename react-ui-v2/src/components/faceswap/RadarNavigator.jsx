import React, { useRef } from 'react';
import { clampPan } from './zoomPan';

/**
 * Mini Radar Navigator (PiP).
 * Automatically appears in the preview corner when zoomed past 1.5x,
 * showing the full frame with an interactive draggable viewport rectangle.
 */
export default function RadarNavigator({
  zoom = 1,
  pan = { x: 0, y: 0 },
  onPanChange,
  imgDim = null,
  thumbnailSrc = '',
  stageRef,
  contentRef,
  className = '',
}) {
  const radarRef = useRef(null);
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0, initialPanX: 0, initialPanY: 0 });

  // Only appear when zoomed past 1.5x
  if (zoom <= 1.5 || !imgDim || !imgDim.w || !imgDim.h) return null;

  // Viewport rectangle dimension as percentage of full frame
  const rectW = Math.min(100, Math.max(8, (1 / zoom) * 100));
  const rectH = Math.min(100, Math.max(8, (1 / zoom) * 100));

  // Compute normalized pan offsets
  // Under transform scale(z) translate(pan.x/z, pan.y/z), a pan of pan.x layout px
  // shifts the visible window by -pan.x / (contentW * zoom) in normalized space
  const stageEl = stageRef?.current;
  const contentEl = contentRef?.current || stageEl;
  const cw = contentEl?.offsetWidth || 800;
  const ch = contentEl?.offsetHeight || 450;

  const normOffsetX = cw > 0 ? -pan.x / (cw * zoom) : 0;
  const normOffsetY = ch > 0 ? -pan.y / (ch * zoom) : 0;

  const rectLeft = Math.max(0, Math.min(100 - rectW, (0.5 + normOffsetX - 0.5 / zoom) * 100));
  const rectTop = Math.max(0, Math.min(100 - rectH, (0.5 + normOffsetY - 0.5 / zoom) * 100));

  const handlePointerDown = (e) => {
    e.stopPropagation();
    e.preventDefault();
    isDraggingRef.current = true;
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      initialPanX: pan.x,
      initialPanY: pan.y,
    };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (!isDraggingRef.current || !onPanChange) return;
    e.stopPropagation();
    e.preventDefault();

    const radarEl = radarRef.current;
    if (!radarEl) return;
    const radarRect = radarEl.getBoundingClientRect();

    const deltaX = e.clientX - dragStartRef.current.x;
    const deltaY = e.clientY - dragStartRef.current.y;

    const factorX = (cw * zoom) / Math.max(1, radarRect.width);
    const factorY = (ch * zoom) / Math.max(1, radarRect.height);

    const nextPanX = dragStartRef.current.initialPanX - deltaX * factorX;
    const nextPanY = dragStartRef.current.initialPanY - deltaY * factorY;

    const clamped = clampPan({ x: nextPanX, y: nextPanY }, zoom, stageEl, contentEl);
    onPanChange(clamped);
  };

  const handlePointerUp = (e) => {
    if (isDraggingRef.current) {
      isDraggingRef.current = false;
      try { e.currentTarget.releasePointerCapture?.(e.pointerId); } catch {}
    }
  };

  const aspect = imgDim.w / imgDim.h;

  return (
    <div
      ref={radarRef}
      className={`absolute top-14 right-3 z-40 rounded-xl overflow-hidden bg-black/85 backdrop-blur-md border border-white/20 shadow-2xl p-1 select-none flex flex-col gap-1 animate-fade-in group/radar ${className}`}
      style={{ width: 116 }}
    >
      <div className="flex items-center justify-between px-1 text-nano font-mono font-bold text-white/70">
        <span className="uppercase tracking-wider text-accent">Radar</span>
        <span>{Math.round(zoom * 100)}%</span>
      </div>

      <div
        className="relative w-full rounded bg-black/70 border border-white/10 overflow-hidden cursor-crosshair"
        style={{ height: Math.round(108 / (aspect || 1.77)) }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      >
        {thumbnailSrc && (
          <img
            src={thumbnailSrc}
            alt="Navigator thumbnail"
            className="w-full h-full object-cover opacity-40 pointer-events-none"
            draggable={false}
          />
        )}

        {/* Viewport Frustum Rectangle */}
        <div
          className="absolute border-2 border-[var(--accent)] bg-[var(--accent)]/25 rounded-sm shadow-[0_0_8px_var(--accent-glow)] cursor-grab active:cursor-grabbing pointer-events-auto"
          style={{
            left: `${rectLeft}%`,
            top: `${rectTop}%`,
            width: `${rectW}%`,
            height: `${rectH}%`,
          }}
          title="Drag to pan viewport"
        />
      </div>
    </div>
  );
}
