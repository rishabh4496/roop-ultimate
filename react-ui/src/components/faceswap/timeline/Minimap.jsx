import React, { useEffect, useRef } from 'react';
import { useLaneCanvas } from './useLaneCanvas';
import { STATE_COLORS } from './lanes';
import { clamp, centerView, minSpan, normalizeView, panView, placeView, viewSpan, zoomView } from './viewMath';

// The always-visible overview bar above the ruler: the whole clip at a glance (render state,
// scene cuts, who is on screen) with the visible window drawn on it. Drag the lit window to pan,
// drag either handle to zoom that edge, click anywhere else to centre the window there.
//
// It is the one lane that never zooms with the view - it always shows frames 1..maxFrames - so it
// answers the question a zoomed track cannot: where in the clip am I?

export const MINIMAP_H = 34;

export default function Minimap({
  maxFrames, view, setView, frame, startFrame, endFrame, spans, cuts, presence, trackW,
}) {
  const v = normalizeView(view, maxFrames);
  const span = Math.max(1, maxFrames - 1);
  const pct = (f) => ((clamp(f, 1, maxFrames) - 1) / span) * 100;
  const hostRef = useRef(null);
  const dragRef = useRef(null);

  const canvasRef = useLaneCanvas((ctx, { w, h, theme }) => {
    const px = (f) => ((f - 1) / span) * w;
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, w, h);
    // render state along the bottom, 4 px
    for (const s of spans) {
      const x0 = Math.floor(px(s.start - 0.5));
      const x1 = Math.ceil(px(s.end + 0.5));
      ctx.globalAlpha = s.state === 'dirty' ? 0.5 : 0.9;
      ctx.fillStyle = STATE_COLORS[s.state];
      ctx.fillRect(Math.max(0, x0), h - 4, Math.max(1, x1 - Math.max(0, x0)), 4);
    }
    ctx.globalAlpha = 1;
    // who is on screen: one 3 px line per person above it
    presence.slice(0, 6).forEach((row, i) => {
      ctx.fillStyle = row.color;
      ctx.globalAlpha = 0.85;
      const y = h - 10 - i * 4;
      for (const [s, e] of row.intervals) {
        const x0 = Math.floor(px(s - 0.5));
        ctx.fillRect(Math.max(0, x0), y, Math.max(1, Math.ceil(px(e + 0.5)) - Math.max(0, x0)), 3);
      }
    });
    ctx.globalAlpha = 1;
    // cuts: ticks, or a density wash when they are closer than a pixel
    const bins = new Uint16Array(Math.max(1, w));
    for (const c of cuts) bins[clamp(Math.floor(px(c)), 0, w - 1)]++;
    ctx.fillStyle = '#fb7a3c';
    for (let x = 0; x < w; x++) {
      if (!bins[x]) continue;
      ctx.globalAlpha = clamp(0.35 + bins[x] * 0.15, 0.35, 1);
      ctx.fillRect(x, 0, 1, 7);
    }
    ctx.globalAlpha = 1;
    // In/Out
    if (startFrame > 1 || endFrame < maxFrames) {
      ctx.fillStyle = theme.accent;
      ctx.globalAlpha = 0.14;
      ctx.fillRect(px(startFrame), 0, Math.max(1, px(endFrame) - px(startFrame)), h);
      ctx.globalAlpha = 1;
    }
  }, [spans, cuts, presence, startFrame, endFrame, maxFrames, trackW]);

  const frameAt = (clientX) => {
    const rect = hostRef.current?.getBoundingClientRect();
    if (!rect || !rect.width) return 1;
    return clamp(Math.round(1 + ((clientX - rect.left) / rect.width) * span), 1, maxFrames);
  };

  // The drag handlers need the CURRENT window but must not re-subscribe on every playback tick
  // (auto-follow moves the window while playing), so the latest values go through a ref.
  const liveRef = useRef(null);
  liveRef.current = { v, setView, maxFrames, trackW, frameAt };
  useEffect(() => {
    const onMove = (e) => {
      const d = dragRef.current;
      const L = liveRef.current;
      if (!d || !L?.setView) return;
      const f = L.frameAt(e.clientX);
      const floor = minSpan(L.trackW, L.maxFrames);
      if (d.mode === 'move') L.setView(placeView(f - d.grab, viewSpan(L.v), L.maxFrames));
      else if (d.mode === 'left') L.setView({ start: clamp(f, 1, L.v.end - floor), end: L.v.end });
      else L.setView({ start: L.v.start, end: clamp(f, L.v.start + floor, L.maxFrames) });
    };
    const onUp = () => { dragRef.current = null; };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }, []);

  const begin = (mode, grab = 0) => (e) => {
    e.stopPropagation();
    e.preventDefault();
    dragRef.current = { mode, grab };
  };

  const onHostDown = (e) => {
    // Anywhere that is not the window or a handle: centre the window there, then keep dragging it.
    const f = frameAt(e.clientX);
    setView(centerView(v, maxFrames, f));
    dragRef.current = { mode: 'move', grab: Math.round(viewSpan(v) / 2) };
  };

  const onKey = (e) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); setView(panView(v, maxFrames, -Math.round(viewSpan(v) * 0.2))); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); setView(panView(v, maxFrames, Math.round(viewSpan(v) * 0.2))); }
    else if (e.key === '+' || e.key === '=') { e.preventDefault(); setView(zoomView(v, maxFrames, trackW, 0.7, (v.start + v.end) / 2)); }
    else if (e.key === '-' || e.key === '_') { e.preventDefault(); setView(zoomView(v, maxFrames, trackW, 1.4, (v.start + v.end) / 2)); }
  };

  const winLeft = pct(v.start);
  const winWidth = Math.max(1.2, pct(v.end) - pct(v.start));

  return (
    <div
      ref={hostRef}
      data-testid="timeline-minimap"
      role="slider"
      aria-valuenow={v.start}
      aria-valuemin={1}
      aria-valuemax={maxFrames}
      tabIndex={0}
      aria-label="Timeline overview. Drag the window to scroll, drag its handles to zoom. Arrow keys scroll, plus and minus zoom."
      onPointerDown={onHostDown}
      onKeyDown={onKey}
      className="relative w-full rounded-lg border border-[var(--border-color)] overflow-hidden cursor-pointer outline-none focus-visible:border-[var(--accent)]"
      style={{ height: MINIMAP_H }}
      title="Whole clip - drag the lit window to scroll, its handles to zoom"
    >
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full block pointer-events-none" />
      <span className="absolute inset-y-0 w-px bg-white/80 pointer-events-none" style={{ left: `${pct(frame)}%` }} />
      <div
        className="absolute inset-y-0 border-y border-white/55 bg-white/[0.10] cursor-grab active:cursor-grabbing"
        style={{ left: `${winLeft}%`, width: `${winWidth}%` }}
        onPointerDown={(e) => begin('move', frameAt(e.clientX) - v.start)(e)}
        data-testid="timeline-minimap-window"
      >
        <span
          className="absolute inset-y-0 -left-[5px] w-[10px] cursor-ew-resize grid place-items-center rounded-[3px] bg-white/80 shadow-[0_0_0_1px_rgba(0,0,0,0.5)]"
          onPointerDown={begin('left')}
          data-testid="timeline-minimap-handle-left"
        >
          <span className="h-3 w-px bg-black/40" />
        </span>
        <span
          className="absolute inset-y-0 -right-[5px] w-[10px] cursor-ew-resize grid place-items-center rounded-[3px] bg-white/80 shadow-[0_0_0_1px_rgba(0,0,0,0.5)]"
          onPointerDown={begin('right')}
          data-testid="timeline-minimap-handle-right"
        >
          <span className="h-3 w-px bg-black/40" />
        </span>
      </div>
    </div>
  );
}
