import React, { useRef, useState } from 'react';
export default function PanelResizeDivider({ direction = 'vertical', side = 'left', currentSize,
  onResize, onReset, min = 200, max = 700, className = '', title = 'Resize panel' }) {
  const drag = useRef(null);
  const [dragging, setDragging] = useState(false);
  const vertical = direction === 'vertical';
  const sign = side === 'left' ? 1 : -1;
  const apply = (value) => onResize(Math.max(min, Math.min(max, Math.round(value))));
  return <div role="separator" tabIndex={0} aria-orientation={direction}
    aria-label={title} title={`${title}. Drag or use arrow keys; double-click or Home to reset.`}
    aria-valuenow={Math.round(currentSize)} aria-valuemin={min} aria-valuemax={max}
    className={`workspace-divider ${vertical ? 'vertical' : 'horizontal'} ${dragging ? 'is-dragging' : ''} ${className}`}
    onPointerDown={(e) => {
      if (e.button !== 0) return;
      e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId);
      drag.current = { position: vertical ? e.clientX : e.clientY, size: currentSize,
        zoom: parseFloat(document.documentElement.style.zoom) || 1 };
      setDragging(true);
    }}
    onPointerMove={(e) => {
      if (drag.current) apply(drag.current.size + sign * ((vertical ? e.clientX : e.clientY) - drag.current.position) / drag.current.zoom);
    }}
    onPointerUp={(e) => { drag.current = null; setDragging(false); if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId); }}
    onPointerCancel={() => { drag.current = null; setDragging(false); }}
    onLostPointerCapture={() => { drag.current = null; setDragging(false); }}
    onDoubleClick={onReset}
    onKeyDown={(e) => {
      if (e.key === 'Home' || e.key === 'Enter') { e.preventDefault(); onReset?.(); return; }
      const delta = vertical ? { ArrowLeft: -1, ArrowRight: 1 }[e.key] : { ArrowUp: -1, ArrowDown: 1 }[e.key];
      if (delta) { e.preventDefault(); apply(currentSize + delta * sign * (e.shiftKey ? 40 : 10)); }
    }} />;
}
