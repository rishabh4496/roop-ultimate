import React, {
  forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState,
} from 'react';

// ── <CompareSlider /> — the A/B curtain & 360 Vector Wipe ─────────────────
//
// An interactive divider dragged across the stage to wipe the processed output
// back to the original target frame.
// Supports both linear curtains (vertical/horizontal) and 360° Vector Split Wipe
// with an interactive rotation handle for diagonal wipes across face contours.

const clampPct = (v) => Math.max(0, Math.min(100, v));

const CompareSlider = forwardRef(function CompareSlider({
  /** Ref to a PreviewCanvas. When present the wipe is applied on the canvas. */
  canvasRef,
  direction = 'vertical',            // 'vertical' | 'horizontal'
  angle = 0,                         // 0..360 vector wipe angle in degrees
  onAngleChange,
  defaultPosition = 50,
  /** Controlled position (e.g. the auto-swipe animation). Optional. */
  position,
  onCommit,
  onChange,
  disabled = false,
  labelA = 'Original',
  labelB = 'Processed',
  accent = 'var(--accent)',
  className = 'relative w-full h-full',
  style,
  children,
  render,
}, ref) {
  const hostRef = useRef(null);
  const dividerRef = useRef(null);
  const handleRef = useRef(null);
  const rotateHandleRef = useRef(null);

  const posRef = useRef(position ?? defaultPosition);
  const angleRef = useRef(angle);
  const draggingRef = useRef(false);
  const rotatingRef = useRef(false);

  const [dragging, setDragging] = useState(false);
  const [rotating, setRotating] = useState(false);
  const [curAngle, setCurAngle] = useState(angle);
  const [, forceRenderTick] = useState(0);

  const horizontal = direction === 'horizontal';

  // Push `pct` everywhere it is displayed, without React.
  const apply = useCallback((pct, { notify = true } = {}) => {
    posRef.current = pct;
    const div = dividerRef.current;
    const a = angleRef.current || 0;

    if (div) {
      if (a !== 0) {
        // Vector split wipe geometry along arbitrary angle
        const rad = (a * Math.PI) / 180;
        const normDist = (100 - pct - 50); // -50% to +50%
        div.style.left = `calc(50% + ${normDist * Math.cos(rad)}%)`;
        div.style.top = `calc(50% + ${normDist * Math.sin(rad)}%)`;
        div.style.transform = `translate(-50%, -50%) rotate(${a + 90}deg)`;
      } else if (horizontal) {
        div.style.left = '0';
        div.style.top = `${100 - pct}%`;
        div.style.transform = '';
      } else {
        div.style.top = '0';
        div.style.left = `${100 - pct}%`;
        div.style.transform = '';
      }
    }

    const handle = handleRef.current;
    if (handle) {
      const rounded = String(Math.round(pct));
      handle.setAttribute('aria-valuenow', rounded);
      handle.setAttribute('aria-valuetext', `${rounded}% ${labelB}`);
    }

    canvasRef?.current?.setWipe?.(pct);
    if (notify && onChange) onChange(pct);
    if (render) forceRenderTick((n) => n + 1);
  }, [canvasRef, horizontal, labelB, onChange, render]);

  const applyAngle = useCallback((deg, { notify = true } = {}) => {
    angleRef.current = deg;
    setCurAngle(deg);
    canvasRef?.current?.setWipeAngle?.(deg);
    apply(posRef.current, { notify: false });
    if (notify && onAngleChange) onAngleChange(deg);
  }, [apply, canvasRef, onAngleChange]);

  // Controlled mode: an external animation (auto-swipe) drives the position.
  useEffect(() => {
    if (position === undefined || draggingRef.current) return;
    apply(clampPct(position), { notify: false });
  }, [position, apply]);

  useEffect(() => {
    angleRef.current = angle;
    setCurAngle(angle);
    apply(posRef.current, { notify: false });
  }, [angle, apply]);

  // Re-apply after a direction change
  useEffect(() => {
    const div = dividerRef.current;
    if (div && angleRef.current === 0) {
      div.style.top = '';
      div.style.left = '';
      div.style.transform = '';
    }
    apply(posRef.current, { notify: false });
  }, [direction, apply]);

  const pctFromEvent = useCallback((e) => {
    const host = hostRef.current;
    if (!host) return posRef.current;
    const rect = host.getBoundingClientRect();
    const x = e.clientX ?? e.touches?.[0]?.clientX;
    const y = e.clientY ?? e.touches?.[0]?.clientY;

    const a = angleRef.current || 0;
    if (a !== 0) {
      // Vector wipe projection along the normal angle
      if (x === undefined || y === undefined || !rect.width || !rect.height) return posRef.current;
      const rad = (a * Math.PI) / 180;
      const nx = Math.cos(rad);
      const ny = Math.sin(rad);
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (x - cx) / rect.width;
      const dy = (y - cy) / rect.height;
      const proj = dx * nx + dy * ny; // -0.5 to +0.5
      return clampPct((0.5 - proj) * 100);
    }

    if (horizontal) {
      if (y === undefined || !rect.height) return posRef.current;
      return clampPct(100 - ((y - rect.top) / rect.height) * 100);
    }
    if (x === undefined || !rect.width) return posRef.current;
    return clampPct(100 - ((x - rect.left) / rect.width) * 100);
  }, [horizontal]);

  const beginDrag = useCallback((e) => {
    if (disabled || rotatingRef.current) return;
    e.stopPropagation();
    e.preventDefault();
    draggingRef.current = true;
    setDragging(true);
    try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* no capture */ }
    apply(pctFromEvent(e));
  }, [apply, disabled, pctFromEvent]);

  const moveDrag = useCallback((e) => {
    if (!draggingRef.current) return;
    e.preventDefault();
    apply(pctFromEvent(e));
  }, [apply, pctFromEvent]);

  const endDrag = useCallback((e) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    setDragging(false);
    try { e?.currentTarget?.releasePointerCapture?.(e.pointerId); } catch { /* ignore */ }
    onCommit?.(posRef.current);
  }, [onCommit]);

  // Rotation Handle Dragging (0-360 degrees)
  const beginRotate = useCallback((e) => {
    if (disabled) return;
    e.stopPropagation();
    e.preventDefault();
    rotatingRef.current = true;
    setRotating(true);
    try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* no capture */ }
  }, [disabled]);

  const moveRotate = useCallback((e) => {
    if (!rotatingRef.current) return;
    e.preventDefault();
    const handle = handleRef.current;
    if (!handle) return;
    const hr = handle.getBoundingClientRect();
    const hx = hr.left + hr.width / 2;
    const hy = hr.top + hr.height / 2;
    const cx = e.clientX ?? e.touches?.[0]?.clientX ?? hx;
    const cy = e.clientY ?? e.touches?.[0]?.clientY ?? hy;
    const rad = Math.atan2(cy - hy, cx - hx);
    let deg = Math.round((rad * 180) / Math.PI);
    if (deg < 0) deg += 360;
    applyAngle(deg);
  }, [applyAngle]);

  const endRotate = useCallback((e) => {
    if (!rotatingRef.current) return;
    rotatingRef.current = false;
    setRotating(false);
    try { e?.currentTarget?.releasePointerCapture?.(e.pointerId); } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (!dragging && !rotating) return undefined;
    const stop = () => {
      if (draggingRef.current) {
        draggingRef.current = false;
        setDragging(false);
        onCommit?.(posRef.current);
      }
      if (rotatingRef.current) {
        rotatingRef.current = false;
        setRotating(false);
      }
    };
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);
    window.addEventListener('blur', stop);
    return () => {
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
      window.removeEventListener('blur', stop);
    };
  }, [dragging, rotating, onCommit]);

  const onKeyDown = useCallback((e) => {
    if (disabled) return;
    const step = e.shiftKey ? 10 : 2;
    let next = posRef.current;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next -= step;
    else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next += step;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = 100;
    else if (e.key === 'Enter' || e.key === ' ') next = 50;
    else return;
    e.preventDefault();
    e.stopPropagation();
    apply(clampPct(next));
    onCommit?.(posRef.current);
  }, [apply, disabled, onCommit]);

  useImperativeHandle(ref, () => ({
    set: (pct) => apply(clampPct(pct)),
    setAngle: (deg) => applyAngle(deg),
    get: () => posRef.current,
    getAngle: () => angleRef.current,
    reset: () => { apply(50); applyAngle(0); onCommit?.(50); },
  }), [apply, applyAngle, onCommit]);

  const initial = posRef.current;
  const isRotated = curAngle !== 0;

  return (
    <div
      ref={hostRef}
      className={className}
      style={{ touchAction: 'none', ...style }}
    >
      {render ? render(initial) : children}

      {!disabled && (
        <div
          ref={dividerRef}
          className="absolute z-40 pointer-events-none"
          style={isRotated ? {
            left: '50%',
            top: '50%',
            width: '200vmax',
            height: 0,
            transformOrigin: 'center center',
            transform: `translate(-50%, -50%) rotate(${curAngle + 90}deg)`,
          } : (horizontal
            ? { left: 0, right: 0, top: `${100 - initial}%`, height: 0 }
            : { top: 0, bottom: 0, left: `${100 - initial}%`, width: 0 }
          )}
        >
          {/* Split Line */}
          <div
            className="absolute"
            style={isRotated ? {
              left: 0, right: 0, top: -1, height: 2,
              background: `linear-gradient(90deg, transparent 5%, ${accent} 25%, #fff 50%, ${accent} 75%, transparent 95%)`,
              boxShadow: '0 0 14px rgba(0,0,0,0.8)',
            } : (horizontal
              ? {
                left: 0, right: 0, top: -1, height: 2,
                background: `linear-gradient(90deg, ${accent}, #fff, ${accent})`,
                boxShadow: '0 0 12px rgba(0,0,0,0.55)',
              }
              : {
                top: 0, bottom: 0, left: -1, width: 2,
                background: `linear-gradient(180deg, ${accent}, #fff, ${accent})`,
                boxShadow: '0 0 12px rgba(0,0,0,0.55)',
              }
            )}
          />

          {/* Central Grab & Rotation Control */}
          <div
            ref={handleRef}
            role="slider"
            tabIndex={0}
            aria-label={`Compare ${labelA} against ${labelB}`}
            aria-orientation={horizontal ? 'vertical' : 'horizontal'}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(initial)}
            aria-valuetext={`${Math.round(initial)}% ${labelB}`}
            onPointerDown={beginDrag}
            onPointerMove={moveDrag}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onKeyDown={onKeyDown}
            onDoubleClick={(e) => { e.stopPropagation(); apply(50); applyAngle(0); onCommit?.(50); }}
            title={`Drag to wipe · Rotate knob for 360° diagonal split · double-click to reset`}
            className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9
              rounded-full bg-black/85 backdrop-blur-md border border-white/30
              flex items-center justify-center pointer-events-auto
              shadow-[0_4px_24px_rgba(0,0,0,0.85)] outline-none
              focus-visible:ring-2 focus-visible:ring-[var(--accent)]
              ${dragging || rotating ? 'scale-115' : 'transition-transform duration-200 hover:scale-110'}
              ${isRotated ? 'cursor-move' : (horizontal ? 'cursor-ns-resize' : 'cursor-ew-resize')}`}
            style={isRotated
              ? { left: '50%', top: 0, transform: `translate(-50%, -50%) rotate(${-(curAngle + 90)}deg)` }
              : (horizontal ? { left: '50%', top: 0 } : { top: '50%', left: 0 })}
          >
            {/* Arrows */}
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={accent}
              strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"
              style={horizontal ? { transform: 'rotate(90deg)', marginTop: -2 } : { marginRight: 2 }}>
              <polyline points="15 18 9 12 15 6" />
            </svg>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={accent}
              strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"
              style={horizontal ? { transform: 'rotate(-90deg)', marginBottom: -2 } : { transform: 'rotate(180deg)', marginLeft: 2 }}>
              <polyline points="15 18 9 12 15 6" />
            </svg>

            {/* Orbiting 360 Vector Rotation Handle */}
            <div
              ref={rotateHandleRef}
              onPointerDown={beginRotate}
              onPointerMove={moveRotate}
              onPointerUp={endRotate}
              onPointerCancel={endRotate}
              className={`absolute -right-3 -top-3 w-5 h-5 rounded-full bg-[var(--accent)] border border-white/60
                shadow-[0_0_10px_var(--accent-glow)] flex items-center justify-center cursor-crosshair
                hover:scale-125 transition-transform ${rotating ? 'scale-125 ring-2 ring-white' : ''}`}
              title={`Rotate split angle: ${curAngle}° (drag around)`}
            >
              <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3">
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
              </svg>
            </div>

            {isRotated && (
              <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded bg-black/90 backdrop-blur border border-white/20 text-nano font-mono font-bold text-white whitespace-nowrap shadow-md pointer-events-none">
                {curAngle}°
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

export default CompareSlider;
