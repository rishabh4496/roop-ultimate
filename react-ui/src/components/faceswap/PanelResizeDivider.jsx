import React, { useCallback, useRef, useState } from 'react';

/**
 * Resizable panel divider for the Face Swap workstation.
 *
 * Supports horizontal dragging for panel widths (Left Inspector, Right Media Rail)
 * and vertical dragging for panel heights (Bottom Timeline Deck).
 * Proportions persist in Zustand and localStorage. Double-click resets to default.
 */
export default function PanelResizeDivider({
  direction = 'vertical', // 'vertical' (horizontal drag for width) | 'horizontal' (vertical drag for height)
  side = 'left',          // 'left' (increases width going right) | 'right' (increases width going left) | 'bottom'
  currentSize,
  onResize,
  onReset,
  min = 200,
  max = 700,
  className = '',
  title = 'Drag to resize panel (Double-click to reset)',
}) {
  const [dragging, setDragging] = useState(false);
  const startPosRef = useRef(0);
  const startSizeRef = useRef(0);

  const isVertical = direction === 'vertical';

  const handleMouseDown = useCallback((e) => {
    e.preventDefault();
    setDragging(true);
    startPosRef.current = isVertical ? e.clientX : e.clientY;
    startSizeRef.current = currentSize;

    const handleMouseMove = (moveEvent) => {
      const currentPos = isVertical ? moveEvent.clientX : moveEvent.clientY;
      const delta = currentPos - startPosRef.current;
      let newSize;

      if (isVertical) {
        if (side === 'left') {
          newSize = startSizeRef.current + delta;
        } else {
          // right rail: dragging left increases right panel width
          newSize = startSizeRef.current - delta;
        }
      } else {
        // bottom timeline: dragging up increases timeline height
        newSize = startSizeRef.current - delta;
      }

      const clamped = Math.max(min, Math.min(max, Math.round(newSize)));
      onResize(clamped);
    };

    const handleMouseUp = () => {
      setDragging(false);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [currentSize, isVertical, max, min, onResize, side]);

  return (
    <div
      role="separator"
      tabIndex={0}
      aria-orientation={isVertical ? 'vertical' : 'horizontal'}
      aria-valuenow={currentSize}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-label={title}
      title={title}
      onMouseDown={handleMouseDown}
      onDoubleClick={onReset}
      className={`group relative z-20 flex select-none transition-colors duration-150 ${
        isVertical
          ? 'w-2 cursor-col-resize items-stretch justify-center -mx-1'
          : 'h-2 cursor-row-resize flex-col items-center justify-center -my-1'
      } ${dragging ? 'bg-[var(--accent)]/30' : 'hover:bg-[var(--accent)]/20'} ${className}`}
    >
      <div
        className={`rounded-full transition-all duration-150 ${
          isVertical
            ? 'w-1 h-12 my-auto'
            : 'h-1 w-16 mx-auto'
        } ${
          dragging
            ? 'bg-[var(--accent)] shadow-[0_0_8px_var(--accent-glow)]'
            : 'bg-white/20 group-hover:bg-[var(--accent)] group-hover:shadow-[0_0_6px_var(--accent-glow)]'
        }`}
      />
    </div>
  );
}
