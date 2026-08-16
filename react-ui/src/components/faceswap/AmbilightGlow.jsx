import React, { useEffect, useState } from 'react';

/**
 * AmbilightGlow
 * Renders a soft dynamic blurred glow behind the media preview canvas.
 * Mirrors the active previewSrc image to generate natural ambient illumination.
 */
export default function AmbilightGlow({ previewSrc, enabled = true, intensity = 0.6 }) {
  const [activeSrc, setActiveSrc] = useState(previewSrc);

  useEffect(() => {
    if (previewSrc) {
      setActiveSrc(previewSrc);
    }
  }, [previewSrc]);

  if (!enabled || !activeSrc) return null;

  return (
    <div
      className="pointer-events-none absolute -inset-8 z-0 overflow-hidden rounded-3xl transition-opacity duration-700 ease-out"
      style={{ opacity: intensity }}
      aria-hidden="true"
    >
      <img
        src={activeSrc}
        alt=""
        className="h-full w-full object-cover blur-3xl scale-125 opacity-70 transform-gpu contrast-125 saturate-150 transition-all duration-500"
        draggable={false}
      />
      {/* Overlay radial mesh gradient to soften edges and create depth */}
      <div className="absolute inset-0 bg-radial from-transparent via-neutral-950/40 to-neutral-950" />
    </div>
  );
}
