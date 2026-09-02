import React, {
  forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState,
} from 'react';

// ── <CompareSlider /> — the A/B curtain ───────────────────────────────────
//
// A vertical (or horizontal) divider dragged across the stage to wipe the
// processed output back to the original target frame.
//
// The design point is that DRAGGING IT MUST NOT RE-RENDER REACT. The obvious
// implementation — `position` in useState, `clipPath` in the style prop — puts
// a full commit of the Face Swap panel between the pointer moving and the pixel
// moving, which is what made the old slider feel like it was catching. Here the
// position lives in a ref: the divider is moved by writing `style.left` on its
// own node, and the picture is cut by calling `setWipe` on the canvas handle,
// which repaints on its own rAF. Neither touches the React tree.
//
// `onCommit` fires on release only, for the callers that DO want the settled
// value in state (persisting it, syncing a second view). Nothing needs it while
// the pointer is down.
//
// Works over any child — the canvas is the fast path, but a `render` callback
// receives the live percentage for callers that must cut something else.

const clampPct = (v) => Math.max(0, Math.min(100, v));

const CompareSlider = forwardRef(function CompareSlider({
  /** Ref to a PreviewCanvas. When present the wipe is applied on the canvas. */
  canvasRef,
  direction = 'vertical',            // 'vertical' = a left/right curtain
  defaultPosition = 50,
  /** Controlled position (e.g. the auto-swipe animation). Optional. */
  position,
  onCommit,
  onChange,
  disabled = false,
  labelA = 'Original',
  labelB = 'Processed',
  accent = 'var(--accent)',
  // Layout is the caller's. Overlaying an existing stage needs `absolute
  // inset-0 pointer-events-none`, and Tailwind cannot express that as an
  // OVERRIDE of a hardcoded `relative w-full h-full` — position utilities all
  // have the same specificity, so which one wins is stylesheet order, not the
  // order they appear in the attribute. Replacing the default is the only way
  // that is actually deterministic.
  className = 'relative w-full h-full',
  style,
  children,
  render,
}, ref) {
  const hostRef = useRef(null);
  const dividerRef = useRef(null);
  const handleRef = useRef(null);
  const posRef = useRef(position ?? defaultPosition);
  const draggingRef = useRef(false);
  // Only for the caller-visible chrome (the handle's cursor, the label fade).
  // Deliberately NOT the position — see the note above.
  const [dragging, setDragging] = useState(false);
  const [, forceRenderTick] = useState(0);

  const horizontal = direction === 'horizontal';

  // Push `pct` everywhere it is displayed, without React.
  const apply = useCallback((pct, { notify = true } = {}) => {
    posRef.current = pct;
    const div = dividerRef.current;
    if (div) {
      if (horizontal) div.style.top = `${100 - pct}%`;
      else div.style.left = `${100 - pct}%`;
    }
    // The rendered `aria-valuenow` is a render-time snapshot, and dragging
    // deliberately does not re-render — so without this the value announced to
    // a screen reader freezes at wherever the last commit left it while the
    // curtain visibly moves. Written on the node, like the divider's position.
    const handle = handleRef.current;
    if (handle) {
      const rounded = String(Math.round(pct));
      handle.setAttribute('aria-valuenow', rounded);
      handle.setAttribute('aria-valuetext', `${rounded}% ${labelB}`);
    }
    // The canvas draws the cut itself: one clip rect inside its existing paint,
    // versus a CSS clip-path that forces the compositor to re-rasterise a layer
    // the size of the stage on every pointer move.
    canvasRef?.current?.setWipe?.(pct);
    if (notify && onChange) onChange(pct);
    // `render` consumers need a commit to see the new value; they opt into that
    // cost by passing the prop at all.
    if (render) forceRenderTick((n) => n + 1);
  }, [canvasRef, horizontal, labelB, onChange, render]);

  // Controlled mode: an external animation (auto-swipe) drives the position.
  useEffect(() => {
    if (position === undefined || draggingRef.current) return;
    apply(clampPct(position), { notify: false });
  }, [position, apply]);

  // Re-apply after a direction change — the divider node's inline style is
  // written on the OTHER axis, and the stale one would leave it parked.
  useEffect(() => {
    const div = dividerRef.current;
    if (div) { div.style.top = ''; div.style.left = ''; }
    apply(posRef.current, { notify: false });
  }, [direction, apply]);

  const pctFromEvent = useCallback((e) => {
    const host = hostRef.current;
    if (!host) return posRef.current;
    const rect = host.getBoundingClientRect();
    const x = e.clientX ?? e.touches?.[0]?.clientX;
    const y = e.clientY ?? e.touches?.[0]?.clientY;
    if (horizontal) {
      if (y === undefined || !rect.height) return posRef.current;
      return clampPct(100 - ((y - rect.top) / rect.height) * 100);
    }
    if (x === undefined || !rect.width) return posRef.current;
    return clampPct(100 - ((x - rect.left) / rect.width) * 100);
  }, [horizontal]);

  const beginDrag = useCallback((e) => {
    if (disabled) return;
    e.stopPropagation();          // the stage below this pans on pointer-down
    e.preventDefault();
    draggingRef.current = true;
    setDragging(true);
    // Pointer capture means a drag that leaves the stage — or the window —
    // still tracks, and the release always lands on us. Without it, letting go
    // outside left the curtain stuck to the pointer.
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

  // A pointer released while the window was not focused (alt-tab mid-drag)
  // never delivers pointerup to us, so the curtain would stay latched.
  useEffect(() => {
    if (!dragging) return undefined;
    const stop = () => { if (draggingRef.current) { draggingRef.current = false; setDragging(false); onCommit?.(posRef.current); } };
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);
    window.addEventListener('blur', stop);
    return () => {
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
      window.removeEventListener('blur', stop);
    };
  }, [dragging, onCommit]);

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
    get: () => posRef.current,
    reset: () => { apply(50); onCommit?.(50); },
  }), [apply, onCommit]);

  const initial = posRef.current;

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
          style={horizontal
            ? { left: 0, right: 0, top: `${100 - initial}%`, height: 0 }
            : { top: 0, bottom: 0, left: `${100 - initial}%`, width: 0 }}
        >
          {/* The line. Its own node so the handle below can be hit-tested
              independently of it. */}
          <div
            className="absolute"
            style={horizontal
              ? {
                left: 0, right: 0, top: -1, height: 2,
                background: `linear-gradient(90deg, ${accent}, #fff, ${accent})`,
                boxShadow: '0 0 12px rgba(0,0,0,0.55)',
              }
              : {
                top: 0, bottom: 0, left: -1, width: 2,
                background: `linear-gradient(180deg, ${accent}, #fff, ${accent})`,
                boxShadow: '0 0 12px rgba(0,0,0,0.55)',
              }}
          />
          {/* The grab handle. */}
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
            onDoubleClick={(e) => { e.stopPropagation(); apply(50); onCommit?.(50); }}
            title={`Drag to wipe between ${labelA} and ${labelB} · double-click to centre · arrow keys to nudge`}
            className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9
              rounded-full bg-black/80 backdrop-blur-md border border-white/30
              flex items-center justify-center pointer-events-auto
              shadow-[0_4px_20px_rgba(0,0,0,0.7)] outline-none
              focus-visible:ring-2 focus-visible:ring-[var(--accent)]
              ${dragging ? 'scale-110' : 'transition-transform duration-200 hover:scale-110'}
              ${horizontal ? 'cursor-ns-resize' : 'cursor-ew-resize'}`}
            style={horizontal
              ? { left: '50%', top: 0 }
              : { top: '50%', left: 0 }}
          >
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
          </div>
        </div>
      )}
    </div>
  );
});

export default CompareSlider;
