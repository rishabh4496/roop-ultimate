import React from 'react';

/**
 * 3.5x Magnifying Glass Loupe for Inspecting Fine Pixel Details (corneas, skin pores, hair edges).
 * Follows mouse cursor in layout space without coordinate drift.
 */
export function MagnifierLoupe({
  src,
  lensPos,
  lensRadius = 90,
  zoomFactor = 3.5,
}) {
  if (!lensPos || !lensPos.visible || !src) return null;

  const diameter = lensRadius * 2;
  const { x, y, imgW, imgH, imgX, imgY } = lensPos;

  // Normalized cursor offset relative to the displayed image rectangle
  const relX = x - imgX;
  const relY = y - imgY;

  return (
    <div
      className="absolute pointer-events-none rounded-full overflow-hidden z-50 border-2 border-[var(--accent-primary,#38bdf8)] shadow-[0_0_24px_rgba(0,0,0,0.8),0_0_12px_rgba(56,189,248,0.4)]"
      style={{
        width: `${diameter}px`,
        height: `${diameter}px`,
        left: `${x - lensRadius}px`,
        top: `${y - lensRadius}px`,
        background: '#090a0f',
      }}
    >
      <div
        className="absolute"
        style={{
          width: `${imgW * zoomFactor}px`,
          height: `${imgH * zoomFactor}px`,
          left: `${lensRadius - relX * zoomFactor}px`,
          top: `${lensRadius - relY * zoomFactor}px`,
        }}
      >
        <img
          src={src}
          alt="Loupe"
          className="w-full h-full object-contain pointer-events-none"
        />
      </div>

      {/* Reticle Crosshair */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-40">
        <div className="w-full h-px bg-[var(--accent-primary,#38bdf8)]" />
        <div className="h-full w-px bg-[var(--accent-primary,#38bdf8)] absolute" />
      </div>

      {/* Loupe Badge */}
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded bg-black/80 text-[9px] font-mono font-bold text-[var(--accent-primary,#38bdf8)] pointer-events-none">
        {zoomFactor}×
      </div>
    </div>
  );
}
