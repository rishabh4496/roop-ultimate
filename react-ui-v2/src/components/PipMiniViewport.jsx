import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Icon } from '../icons';
import { usePipStore } from '../store/pipStore';
import {
  PIP_BAR, PIP_DEFAULT, PIP_MARGIN, clampPosition, cornerPositions, pictureHeight, pipVisible, resizeTo, snapCorner,
} from '../pip/pipModel';
import { uiScale } from './faceswap/zoomPan';

// The floating mini-viewport (280x158 by default, resizable): Face Swap's current picture, kept in a corner of the screen
// while you work somewhere else. Where the picture comes from, and how it is renewed, is store/pipStore.js; where the window
// may sit and when it shows is pip/pipModel.js. This file is only the window.
//
// COORDINATES. The app applies its UI zoom as CSS `zoom` on <html>. Pointer positions and window.innerWidth are in visual
// pixels; `left` / `top` / `width` are in layout pixels, which the zoom scales. Everything the model sees is layout pixels:
// each pointer delta is divided by the window's own scale (uiScale) before it is used, so the window follows the pointer at
// every zoom instead of drifting by the zoom ratio.
//
// DOCKING. A window dropped within 72 px of a corner snaps to it and stays anchored to that corner when resized or when the
// browser window changes size; one dropped elsewhere floats where it was left (and is pulled back on screen if the window
// shrinks under it).
//
// The controls are real buttons and the move/resize handles take arrow keys, so nothing here is reachable by mouse alone.
// They exist only while the window is on screen, so they add no tab stop to a tab that is not showing it.

const NUDGE = 16;
const PIP_DEFAULT_AR = PIP_DEFAULT.w / PIP_DEFAULT.h;
const iconBtn = 'grid h-6 w-6 place-items-center rounded-md text-white/65 hover:bg-white/15 hover:text-white apple-transition';

export default function PipMiniViewport({ tab, onReturn }) {
  const enabled = usePipStore((s) => s.enabled);
  const frame = usePipStore((s) => s.frame);
  const stageVisible = usePipStore((s) => s.stageVisible);
  const w = usePipStore((s) => s.w);
  const corner = usePipStore((s) => s.corner);
  const sx = usePipStore((s) => s.x);
  const sy = usePipStore((s) => s.y);
  const context = usePipStore((s) => s.context);
  const refreshing = usePipStore((s) => s.refreshing);
  const refreshError = usePipStore((s) => s.refreshError);
  const faceswapMounted = usePipStore((s) => s.faceswapMounted);
  const { setGeometry, dock, cycleCorner, setEnabled, refresh } = usePipStore.getState();   // actions are stable

  const visible = pipVisible({ enabled, hasFrame: !!frame, tab, stageVisible });
  const rootRef = useRef(null);
  const canvasRef = useRef(null);
  const [vp, setVp] = useState({ vw: 1280, vh: 720, scale: 1 });
  const [live, setLive] = useState(null);                 // { x, y } or { w } while a drag is in flight (layout px)

  const aspect = frame ? frame.w / frame.h : PIP_DEFAULT_AR;
  const width = live?.w ?? w;
  const height = pictureHeight(width, aspect);

  // The viewport in LAYOUT pixels, re-measured on resize (and when the window first appears).
  const measure = useCallback(() => {
    const scale = uiScale(rootRef.current) || 1;
    setVp({ vw: window.innerWidth / scale, vh: window.innerHeight / scale, scale });
  }, []);
  useLayoutEffect(() => { if (visible) measure(); }, [visible, measure]);
  useEffect(() => {
    if (!visible) return undefined;
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [visible, measure]);

  // Where the window is: docked -> that corner; otherwise its saved/dragged position, kept on screen.
  const size = { w: width, h: height };
  const corners = cornerPositions(size, vp);
  const free = sx !== null && sy !== null ? clampPosition({ x: sx, y: sy }, size, vp) : corners['bottom-right'];
  const pos = live && live.x !== undefined ? clampPosition(live, size, vp) : (corner && corners[corner] ? corners[corner] : free);

  // Paint the frame. The canvas backing store is the on-screen size (capped), not the frame's, so a 4K frame is not held at
  // 4K in a 280 px window.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!visible || !canvas || !frame?.bitmap) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = Math.max(1, Math.round(width * vp.scale * dpr));
    const ch = Math.max(1, Math.round(height * vp.scale * dpr));
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    try {
      ctx.drawImage(frame.bitmap, 0, 0, cw, ch);
    } catch { /* the bitmap was released between the store update and this paint: the next frame repaints */ }
  }, [visible, frame, width, height, vp.scale]);

  // ── drag (move) ─────────────────────────────────────────────────────────
  const dragRef = useRef(null);
  const onMoveDown = (e) => {
    if (e.button !== 0 || e.target.closest('button')) return;
    const scale = uiScale(rootRef.current) || 1;
    dragRef.current = { id: e.pointerId, px: e.clientX, py: e.clientY, x: pos.x, y: pos.y, scale };
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
  };
  const onMoveMove = (e) => {
    const d = dragRef.current;
    if (!d || e.pointerId !== d.id) return;
    setLive({ x: d.x + (e.clientX - d.px) / d.scale, y: d.y + (e.clientY - d.py) / d.scale });
  };
  const onMoveUp = (e) => {
    const d = dragRef.current;
    if (!d || e.pointerId !== d.id) return;
    dragRef.current = null;
    if (live && live.x !== undefined) {
      const p = clampPosition(live, size, vp);
      const snap = snapCorner(p, size, vp);
      if (snap) dock(snap); else setGeometry({ corner: null, x: Math.round(p.x), y: Math.round(p.y) });
    }
    setLive(null);
  };

  // ── drag (resize), from the corner opposite the docked one ──────────────────
  // A window anchored bottom-right grows up and to the left, so its grip is the top-left corner; a free window grows from
  // its bottom-right. `dir` is +1 when moving the grip right makes the window wider, -1 when it makes it narrower.
  const gripLeft = !!corner && corner.endsWith('right');
  const gripTop = !!corner && corner.startsWith('bottom');
  const dir = gripLeft ? -1 : 1;
  const resizeRef = useRef(null);
  const onResizeDown = (e) => {
    if (e.button !== 0) return;
    const scale = uiScale(rootRef.current) || 1;
    resizeRef.current = { id: e.pointerId, px: e.clientX, w: width, scale };
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
    e.preventDefault();
  };
  const onResizeMove = (e) => {
    const r = resizeRef.current;
    if (!r || e.pointerId !== r.id) return;
    setLive({ w: resizeTo(r.w + dir * (e.clientX - r.px) / r.scale, aspect, pos, vp).w });
  };
  const onResizeUp = (e) => {
    const r = resizeRef.current;
    if (!r || e.pointerId !== r.id) return;
    resizeRef.current = null;
    if (live?.w) setGeometry({ w: live.w });
    setLive(null);
  };

  const nudge = (e) => {
    const step = e.shiftKey ? NUDGE * 4 : NUDGE;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
    if (!d) return;
    e.preventDefault();
    const p = clampPosition({ x: pos.x + d[0], y: pos.y + d[1] }, size, vp);
    const snap = snapCorner(p, size, vp, PIP_MARGIN + 1);
    if (snap) dock(snap); else setGeometry({ corner: null, x: Math.round(p.x), y: Math.round(p.y) });
  };
  const resizeKey = (e) => {
    const step = e.shiftKey ? NUDGE * 4 : NUDGE;
    const grow = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
    if (!grow) return;
    e.preventDefault();
    setGeometry({ w: resizeTo(width + grow * step, aspect, pos, vp).w });
  };

  if (!visible) return null;

  const caption = context.frame ? `Frame ${context.frame}${context.total > 1 ? ` / ${context.total}` : ''}` : 'Preview';
  const gripPos = `${gripTop ? 'top-0' : 'bottom-0'} ${gripLeft ? 'left-0' : 'right-0'}`;
  const gripCursor = gripLeft === gripTop ? 'cursor-nwse-resize' : 'cursor-nesw-resize';

  return (
    <div
      ref={rootRef}
      role="region"
      aria-label="Floating preview"
      data-testid="pip-window"
      data-corner={corner || 'free'}
      className="fixed z-[55] select-none overflow-hidden rounded-xl border border-white/15 bg-black shadow-2xl shadow-black/60"
      style={{ left: pos.x, top: pos.y, width, height: height + PIP_BAR }}
    >
      <div
        data-testid="pip-bar"
        className="flex items-center gap-1 px-1.5 bg-[#10131c]/95 cursor-move touch-none"
        style={{ height: PIP_BAR }}
        onPointerDown={onMoveDown}
        onPointerMove={onMoveMove}
        onPointerUp={onMoveUp}
        onPointerCancel={onMoveUp}
      >
        <button
          type="button"
          onKeyDown={nudge}
          aria-label="Move the floating preview (arrow keys; Shift for bigger steps)"
          title="Drag the bar to move, or focus this and use the arrow keys"
          className={iconBtn}
        >
          <Icon.pip size={13} />
        </button>
        <span className="min-w-0 flex-1 truncate text-nano font-semibold uppercase tracking-wider text-white/60" data-testid="pip-caption">{caption}</span>
        <button
          type="button"
          onClick={() => refresh()}
          disabled={faceswapMounted || refreshing}
          aria-label="Re-render the floating preview"
          title="Re-render this frame with the source that is selected now"
          className={`${iconBtn} disabled:opacity-40`}
          data-testid="pip-refresh"
        >
          <Icon.refresh size={12} className={refreshing ? 'animate-spin' : ''} />
        </button>
        <button type="button" onClick={cycleCorner} aria-label="Dock to the next corner" title="Dock to the next corner" className={iconBtn} data-testid="pip-dock">
          <Icon.panelRight size={12} />
        </button>
        <button type="button" onClick={onReturn} aria-label="Back to Face Swap" title="Back to Face Swap (or scroll the preview stage into view)" className={iconBtn} data-testid="pip-return">
          <Icon.popout size={12} />
        </button>
        <button type="button" onClick={() => setEnabled(false)} aria-label="Close the floating preview" title="Close (turn it back on from the header)" className={iconBtn} data-testid="pip-close">
          <Icon.close size={12} />
        </button>
      </div>

      <div className="relative" style={{ height }}>
        <canvas
          ref={canvasRef}
          data-testid="pip-canvas"
          role="img"
          aria-label="Floating preview of the current frame"
          className="block w-full h-full"
        />
        {refreshError && (
          <div role="status" data-testid="pip-error" className="absolute inset-x-0 bottom-0 bg-black/75 px-2 py-1 text-nano text-amber-300 truncate" title={refreshError}>
            Could not refresh: {refreshError}
          </div>
        )}
        <button
          type="button"
          onPointerDown={onResizeDown}
          onPointerMove={onResizeMove}
          onPointerUp={onResizeUp}
          onPointerCancel={onResizeUp}
          onKeyDown={resizeKey}
          aria-label="Resize the floating preview (arrow keys; Shift for bigger steps)"
          title="Drag to resize; the picture keeps its shape"
          data-testid="pip-resize"
          className={`absolute ${gripPos} h-4 w-4 ${gripCursor} touch-none opacity-60 hover:opacity-100 focus-visible:opacity-100`}
        >
          <span aria-hidden="true" className={`absolute ${gripPos} h-2.5 w-2.5 border-white/80 ${gripTop ? 'border-t-2' : 'border-b-2'} ${gripLeft ? 'border-l-2' : 'border-r-2'}`} />
        </button>
      </div>
    </div>
  );
}
