import React, { useRef, useState } from 'react';

/**
 * Resizable panel divider for the Face Swap workstation.
 *
 * Supports horizontal dragging for panel widths (Left Inspector, Right Media Rail)
 * and vertical dragging for panel heights (Bottom Timeline Deck).
 * Proportions persist in Zustand and localStorage. Double-click resets to default.
 */
export default function PanelResizeDivider({
  direction = 'vertical', // 'vertical' (horizontal drag for width) | 'horizontal' (vertical drag for height)
  side = 'left',          // left/top grow with the pointer; right/bottom grow against it
  currentSize,
  getSize,
  onResize,
  onReset,
  min = 200,
  max = 700,
  className = '',
  title = 'Drag to resize panel (Double-click to reset)',
  label = title,
  controls,
  valueText,
}) {
  const [dragging, setDragging] = useState(false);
  const drag = useRef(null);
  const isVertical = direction === 'vertical';
  const sign = side === 'left' || side === 'top' ? 1 : -1;
  const positionOf = (event) => isVertical ? event.clientX : event.clientY;
  const resize = (size) => onResize(Math.max(min, Math.min(max, Math.round(size))));

  const handlePointerDown = (event) => {
    if (event.button !== 0 || drag.current) return;
    const handle = event.currentTarget;
    const bounds = handle.getBoundingClientRect();
    const scale = (isVertical ? bounds.width / handle.offsetWidth : bounds.height / handle.offsetHeight) || 1;
    handle.focus({ preventScroll: true });
    handle.setPointerCapture(event.pointerId);
    drag.current = { id: event.pointerId, pos: positionOf(event), size: getSize?.() ?? currentSize, scale };
    setDragging(true);
  };
  const handlePointerMove = (event) => {
    const start = drag.current;
    if (start?.id !== event.pointerId) return;
    const delta = (positionOf(event) - start.pos) / start.scale;
    if (delta || start.moved) resize(start.size + delta * sign);
    if (delta) start.moved = true;
  };
  const endDrag = (event) => {
    if (drag.current?.id !== event.pointerId) return;
    drag.current = null;
    setDragging(false);
  };

  const handleKeyDown = (event) => {
    const back = isVertical ? 'ArrowLeft' : 'ArrowUp';
    const forward = isVertical ? 'ArrowRight' : 'ArrowDown';
    if (![back, forward, 'Home', 'End', 'Enter'].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.key === 'Enter') onReset();
    else if (event.key === 'Home') resize(min);
    else if (event.key === 'End') resize(max);
    else resize((getSize?.() ?? currentSize) + (event.key === back ? -1 : 1) * sign * (event.shiftKey ? 64 : 16));
  };

  return (
    <div
      role="separator"
      tabIndex={0}
      aria-orientation={isVertical ? 'vertical' : 'horizontal'}
      aria-valuenow={currentSize}
      aria-valuetext={valueText}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-label={label}
      aria-controls={controls}
      title={title}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onLostPointerCapture={endDrag}
      onKeyDown={handleKeyDown}
      onDoubleClick={onReset}
      className={`group relative z-20 flex touch-none select-none transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-[var(--accent)] ${
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
