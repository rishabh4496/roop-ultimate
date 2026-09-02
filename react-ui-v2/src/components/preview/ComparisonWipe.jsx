import React, { useEffect, useRef, useState } from 'react';

/**
 * Interactive A/B Comparison Suite for Swapped vs. Original Media.
 * Supports Split Wipe (Horizontal & Vertical), 50% Alpha Blend, Difference Map, and Auto-Swipe.
 */
export function ComparisonWipe({
  beforeSrc,
  afterSrc,
  compareMode = 'slider', // 'slider' | 'blend' | 'diff'
  compareDir = 'vertical', // 'vertical' | 'horizontal'
  sliderPosition = 50,
  onSliderChange,
  isAutoSwiping = false,
  className = '',
}) {
  const containerRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  // Auto-swipe Sinusoidal Animation Loop
  useEffect(() => {
    if (!isAutoSwiping || compareMode === 'diff') return undefined;
    let animId;
    let startTime;
    let phase = 0;

    const animate = (time) => {
      if (document.hidden || document.documentElement.hasAttribute('data-render-lite')) {
        startTime = undefined;
        animId = requestAnimationFrame(animate);
        return;
      }
      if (startTime === undefined) startTime = time - phase * 1000;
      phase = (time - startTime) / 1000;
      const pos = 50 + 42 * Math.sin(phase * (Math.PI * 2 / 2.8));
      onSliderChange?.(pos);
      animId = requestAnimationFrame(animate);
    };

    animId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animId);
  }, [isAutoSwiping, compareMode, onSliderChange]);

  // Pointer Drag Handling
  const handlePointerDown = (e) => {
    if (compareMode !== 'slider') return;
    setIsDragging(true);
    updateSliderFromEvent(e);
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (!isDragging || compareMode !== 'slider') return;
    updateSliderFromEvent(e);
  };

  const handlePointerUp = () => {
    setIsDragging(false);
  };

  const updateSliderFromEvent = (e) => {
    if (!containerRef.current || !onSliderChange) return;
    const rect = containerRef.current.getBoundingClientRect();
    if (compareDir === 'horizontal') {
      const y = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
      const pct = (y / rect.height) * 100;
      onSliderChange(Math.max(0, Math.min(100, pct)));
    } else {
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      const pct = (x / rect.width) * 100;
      onSliderChange(Math.max(0, Math.min(100, pct)));
    }
  };

  if (!beforeSrc || !afterSrc) return null;

  // 1. Difference Map Mode
  if (compareMode === 'diff') {
    return (
      <div className={`relative w-full h-full ${className}`}>
        <img
          src={beforeSrc}
          alt="Original"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />
        <img
          src={afterSrc}
          alt="Swapped"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-difference"
        />
      </div>
    );
  }

  // 2. Alpha Blend Mode
  if (compareMode === 'blend') {
    const alpha = sliderPosition / 100;
    return (
      <div className={`relative w-full h-full ${className}`}>
        <img
          src={beforeSrc}
          alt="Original"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />
        <img
          src={afterSrc}
          alt="Swapped"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-75"
          style={{ opacity: alpha }}
        />
      </div>
    );
  }

  // 3. Interactive Split Wipe Mode
  const clipPath =
    compareDir === 'horizontal'
      ? `polygon(0 0, 100% 0, 100% ${sliderPosition}%, 0 ${sliderPosition}%)`
      : `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`;

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      className={`relative w-full h-full select-none cursor-ew-resize ${
        compareDir === 'horizontal' ? 'cursor-ns-resize' : 'cursor-ew-resize'
      } ${className}`}
    >
      {/* Background Layer: Swapped (After) */}
      <img
        src={afterSrc}
        alt="Swapped"
        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
      />

      {/* Foreground Clipped Layer: Original (Before) */}
      <img
        src={beforeSrc}
        alt="Original"
        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        style={{ clipPath }}
      />

      {/* Divider Line & Handle */}
      {compareDir === 'horizontal' ? (
        <div
          className="absolute inset-x-0 z-30 flex items-center justify-center pointer-events-none"
          style={{ top: `${sliderPosition}%`, transform: 'translateY(-50%)' }}
        >
          <div className="w-full h-0.5 bg-[var(--accent-primary,#38bdf8)] shadow-[0_0_8px_rgba(56,189,248,0.8)]" />
          <div className="absolute w-8 h-4 rounded-full bg-[#090a0f] border border-[var(--accent-primary,#38bdf8)] shadow-xl flex items-center justify-center text-[9px] font-mono font-bold text-[var(--accent-primary,#38bdf8)]">
            ↕
          </div>
        </div>
      ) : (
        <div
          className="absolute inset-y-0 z-30 flex items-center justify-center pointer-events-none"
          style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
        >
          <div className="h-full w-0.5 bg-[var(--accent-primary,#38bdf8)] shadow-[0_0_8px_rgba(56,189,248,0.8)]" />
          <div className="absolute w-4 h-8 rounded-full bg-[#090a0f] border border-[var(--accent-primary,#38bdf8)] shadow-xl flex items-center justify-center text-[9px] font-mono font-bold text-[var(--accent-primary,#38bdf8)]">
            ↔
          </div>
        </div>
      )}
    </div>
  );
}
