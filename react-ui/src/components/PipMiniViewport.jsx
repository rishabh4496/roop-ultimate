import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Icon } from '../icons';
import { usePipStore } from '../store/pipStore';
import {
  PIP_BAR, PIP_DEFAULT, PIP_MARGIN, clampPosition, cornerPositions, pickPicture, pictureHeight, pipVisible, resizeTo, snapCorner,
} from '../pip/pipModel';
import { usePipLiveFeed } from '../pip/usePipLiveFeed';
import { uiScale } from './faceswap/zoomPan';
import { createViewportMatrix } from './faceswap/viewportMatrix';

// The floating mini-viewport (280x158 by default, resizable): a glass window that keeps the picture in a corner of the screen
// while you work somewhere else - Face Manager, Batch, Processing, Settings, or Face Swap itself once its stage has scrolled
// away. Where the picture comes from, and how it is renewed, is store/pipStore.js; where the window may sit and when it shows is
// pip/pipModel.js. This file is only the window.
//
// WHAT IT SHOWS. While a render runs: the frames the render publishes (pip/usePipLiveFeed.js - the same channel as the
// Processing tab's live view). Otherwise Face Swap's last "after" picture, and - when the stage also gave its "before" - an A/B
// control: After -> Wipe (a draggable divider, before on the left) -> Before.
//
// PLAY / PAUSE. One button, three transports (pip/pipModel.transportKind): during a render it freezes THIS window on the frame it
// shows (the render is untouched); with Face Swap mounted it is Face Swap's own Play/Pause; with Face Swap unmounted the window
// steps through the clip itself, one preview request at a time.
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
// The controls are real buttons and the move/resize/wipe handles take arrow keys, so nothing here is reachable by mouse alone.
// They exist only while the window is on screen, so they add no tab stop to a tab that is not showing it.

const NUDGE = 16;
const PIP_DEFAULT_AR = PIP_DEFAULT.w / PIP_DEFAULT.h;
const iconBtn = 'grid h-6 w-6 place-items-center rounded-md text-white/65 hover:bg-white/15 hover:text-white apple-transition';
const chipBtn = 'grid h-5 min-w-[1.25rem] place-items-center rounded px-1 text-white/80 hover:bg-white/20 hover:text-white apple-transition';
const AB_LABEL = { after: 'After', wipe: 'A | B', before: 'Before' };

export default function PipMiniViewport({ tab, onReturn }) {
  const enabled = usePipStore((s) => s.enabled);
  const frame = usePipStore((s) => s.frame);
  const base = usePipStore((s) => s.base);
  const liveFrame = usePipStore((s) => s.liveFrame);
  const running = usePipStore((s) => s.running);
  const paused = usePipStore((s) => s.paused);
  const playing = usePipStore((s) => s.playing);
  const transport = usePipStore((s) => s.transport);
  const hasRequest = usePipStore((s) => !!s.lastRequest);
  const view = usePipStore((s) => s.view);
  const wipe = usePipStore((s) => s.wipe);
  const stageVisible = usePipStore((s) => s.stageVisible);
  const w = usePipStore((s) => s.w);
  const corner = usePipStore((s) => s.corner);
  const sx = usePipStore((s) => s.x);
  const sy = usePipStore((s) => s.y);
  const context = usePipStore((s) => s.context);
  const refreshing = usePipStore((s) => s.refreshing);
  const refreshError = usePipStore((s) => s.refreshError);
  const faceswapMounted = usePipStore((s) => s.faceswapMounted);
  const { setGeometry, dock, cycleCorner, setEnabled, refresh, togglePlay, cycleView, setWipe } = usePipStore.getState();   // actions are stable

  const visible = pipVisible({ enabled, hasFrame: !!frame || !!liveFrame, tab, stageVisible });
  // The feed must start BEFORE the first live frame exists (that frame is what makes the window visible), so it keys on the
  // window being wanted here, not on there being something to show yet.
  const wanted = pipVisible({ enabled, hasFrame: true, tab, stageVisible });
  usePipLiveFeed(wanted && running && !paused);

  const rootRef = useRef(null);
  const canvasRef = useRef(null);
  const pictureRef = useRef(null);
  const [vp, setVp] = useState({ vw: 1280, vh: 720, scale: 1 });
  const [live, setLive] = useState(null);                 // { x, y } or { w } while a drag is in flight (layout px)

  const picked = pickPicture({ running, hasLive: !!liveFrame, hasAfter: !!frame, hasBase: !!base, view });
  const shown = picked.kind === 'live' ? liveFrame : frame;
  const aspect = shown ? shown.w / shown.h : PIP_DEFAULT_AR;
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

  // Unified ViewportMatrix for floating PiP mini-viewport (Stage 26)
  const pipMatrix = useMemo(() => {
    return createViewportMatrix({
      containerWidth: width,
      containerHeight: height,
      mediaWidth: shown?.w || PIP_DEFAULT.w,
      mediaHeight: shown?.h || PIP_DEFAULT.h,
      zoom: 1.0,
      pan: { x: 0, y: 0 },
      objectFit: 'contain',
    });
  }, [width, height, shown?.w, shown?.h]);

  // Paint the picture. The canvas backing store is the on-screen size (capped), not the frame's, so a 4K frame is not held at
  // 4K in a 280 px window. A wipe draws the "before" across the whole canvas and the "after" over everything right of the divider.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!visible || !canvas || !shown?.bitmap) return;
    canvas.viewportMatrix = pipMatrix;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = Math.max(1, Math.round(width * vp.scale * dpr));
    const ch = Math.max(1, Math.round(height * vp.scale * dpr));
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    try {
      if (picked.kind === 'before') {
        ctx.drawImage(base.bitmap, 0, 0, cw, ch);
      } else if (picked.kind === 'wipe') {
        const at = Math.round(cw * (wipe / 100));
        ctx.drawImage(base.bitmap, 0, 0, cw, ch);
        ctx.save();
        ctx.beginPath();
        ctx.rect(at, 0, cw - at, ch);
        ctx.clip();
        ctx.drawImage(shown.bitmap, 0, 0, cw, ch);
        ctx.restore();
      } else {
        ctx.drawImage(shown.bitmap, 0, 0, cw, ch);
      }
    } catch { /* a bitmap was released between the store update and this paint: the next frame repaints */ }
  }, [visible, shown, base, picked.kind, wipe, width, height, vp.scale, pipMatrix]);

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

  // ── the A/B wipe divider ────────────────────────────────────────────────
  const wipeDrag = useRef(null);
  const wipeFromPointer = (e) => {
    const rect = pictureRef.current?.getBoundingClientRect();
    if (rect && rect.width > 0) setWipe(((e.clientX - rect.left) / rect.width) * 100);
  };
  const onWipeDown = (e) => {
    if (e.button !== 0) return;
    wipeDrag.current = e.pointerId;
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
    wipeFromPointer(e);
    e.preventDefault();
  };
  const onWipeMove = (e) => { if (wipeDrag.current === e.pointerId) wipeFromPointer(e); };
  const onWipeUp = (e) => { if (wipeDrag.current === e.pointerId) wipeDrag.current = null; };
  const onWipeKey = (e) => {
    const step = e.shiftKey ? 10 : 2;
    const d = { ArrowLeft: -step, ArrowRight: step, Home: -100, End: 100 }[e.key];
    if (d === undefined) return;
    e.preventDefault();
    setWipe(wipe + d);
  };

  if (!visible) return null;

  const kind = picked.kind;
  const isLive = kind === 'live';
  const frozen = isLive && paused;
  const caption = isLive
    ? (liveFrame?.frame ? `Render · frame ${liveFrame.frame}` : 'Render')
    : (context.frame ? `Frame ${context.frame}${context.total > 1 ? ` / ${context.total}` : ''}` : 'Preview');
  // Is the transport doing anything? (A still has nothing to play: the button says so rather than going quiet.)
  const canMounted = !!transport;
  const canStep = hasRequest && context.total > 1 && (context.frame ?? 0) < context.total;
  const playable = isLive || (faceswapMounted ? canMounted : canStep || playing);
  const isPlaying = isLive ? !paused : faceswapMounted ? !!transport?.playing : playing;
  const playLabel = isLive
    ? (paused ? 'Resume the live view' : 'Pause the live view (the render keeps going)')
    : (isPlaying ? 'Pause playback' : 'Play the clip in the floating preview');
  const gripPos = `${gripTop ? 'top-0' : 'bottom-0'} ${gripLeft ? 'left-0' : 'right-0'}`;
  const gripCursor = gripLeft === gripTop ? 'cursor-nwse-resize' : 'cursor-nesw-resize';
  const badge = isLive ? (frozen ? 'Paused' : 'Live') : AB_LABEL[kind] || 'After';

  return (
    <div
      ref={rootRef}
      role="region"
      aria-label="Floating preview"
      data-testid="pip-window"
      data-corner={corner || 'free'}
      data-mode={kind}
      className="pip-glass fixed z-[55] select-none overflow-hidden rounded-xl"
      style={{ left: pos.x, top: pos.y, width, height: height + PIP_BAR }}
    >
      <div
        data-testid="pip-bar"
        className="pip-bar flex items-center gap-1 px-1.5 cursor-move touch-none"
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
        <span className="min-w-0 flex-1 truncate text-nano font-semibold uppercase tracking-wider text-white/70" data-testid="pip-caption">{caption}</span>
        <button
          type="button"
          onClick={() => refresh()}
          disabled={faceswapMounted || refreshing || running}
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

      <div ref={pictureRef} className="relative bg-black" style={{ height }}>
        <canvas
          ref={canvasRef}
          data-testid="pip-canvas"
          role="img"
          aria-label="Floating preview of the current frame"
          className="block w-full h-full"
        />
        <span data-testid="pip-badge" className={`pip-badge ${isLive ? (frozen ? 'is-paused' : 'is-live') : ''}`}>{badge}</span>

        {kind === 'wipe' && (
          <button
            type="button"
            role="slider"
            aria-label="A/B wipe position (before on the left, after on the right)"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(wipe)}
            aria-valuetext={`${Math.round(wipe)}% before`}
            data-testid="pip-wipe"
            className="pip-wipe-handle"
            style={{ left: `${wipe}%` }}
            onPointerDown={onWipeDown}
            onPointerMove={onWipeMove}
            onPointerUp={onWipeUp}
            onPointerCancel={onWipeUp}
            onKeyDown={onWipeKey}
          />
        )}

        {/* Transport: Play/Pause and the A/B switch. Bottom-centre - the resize grip sits in a corner, the error across the foot. */}
        <div
          role="group"
          aria-label="Floating preview transport"
          data-testid="pip-transport"
          className="absolute bottom-1 left-1/2 flex -translate-x-1/2 items-center gap-0.5 rounded-full border border-white/15 bg-black/55 px-1 py-0.5 backdrop-blur-sm"
        >
          <button
            type="button"
            onClick={() => togglePlay()}
            disabled={!playable}
            aria-pressed={isPlaying}
            aria-label={playLabel}
            title={playable ? playLabel : 'Nothing to play: a single frame, or nothing to step from yet'}
            data-testid="pip-play"
            className={`${chipBtn} disabled:opacity-35`}
          >
            {isPlaying ? <Icon.pause size={11} /> : <Icon.play size={11} />}
          </button>
          <button
            type="button"
            onClick={() => cycleView()}
            disabled={!picked.ab}
            aria-label={`Before / after: showing ${AB_LABEL[picked.ab ? view : 'after']}. Press to change`}
            title={picked.ab ? 'Cycle After → A | B wipe → Before' : (isLive ? 'A render frame has no "before"' : 'The stage has not offered its "before" picture yet')}
            data-testid="pip-ab"
            className={`${chipBtn} text-nano font-bold disabled:opacity-35`}
          >
            <Icon.compare size={11} />
            <span className="sr-only">{AB_LABEL[picked.ab ? view : 'after']}</span>
          </button>
        </div>

        {refreshError && (
          <div role="status" data-testid="pip-error" className="absolute inset-x-0 top-0 bg-black/75 px-2 py-1 text-nano text-amber-300 truncate" title={refreshError}>
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
