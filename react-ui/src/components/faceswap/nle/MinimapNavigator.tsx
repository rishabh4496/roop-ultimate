/**
 * MinimapNavigator: the overview bar at the top of the timeline.
 *
 * Shows the WHOLE clip (face coverage, track spans, markers, In/Out, playhead) with a window over the
 * part the lanes below display. Drag the window to pan, drag either edge to zoom, click outside it to
 * recentre, wheel over it to zoom. Arrow keys pan the window when it has focus (Home / End jump to the
 * clip's ends), so it is operable without a pointer.
 */

import React, { useLayoutEffect, useRef, useState } from 'react';
import { HEALTH_COLORS, type HealthSegment } from './faceHealth';
import { dragMinimap, hitMinimap, type MinimapMode } from './minimapModel';
import { prepareCanvas, useElementSize } from './useElementSize';

export interface MinimapTrack {
  trackId: number;
  color: string;
  ranges: ReadonlyArray<readonly [number, number]>;
}

export interface MinimapNavigatorProps {
  maxFrames: number;
  view: readonly [number, number];
  onViewChange: (view: [number, number]) => void;
  frame: number;
  trimIn: number;
  trimOut: number;
  health: ReadonlyArray<HealthSegment>;
  tracks: ReadonlyArray<MinimapTrack>;
  markers: ReadonlyArray<{ frame: number; color?: string }>;
  headerWidth: number;
  height?: number;
}

const CURSORS: Record<MinimapMode, string> = { move: 'grab', 'resize-left': 'ew-resize', 'resize-right': 'ew-resize', jump: 'pointer' };

function MinimapNavigatorImpl({ maxFrames, view, onViewChange, frame, trimIn, trimOut, health, tracks, markers, headerWidth, height = 34 }: MinimapNavigatorProps) {
  const areaRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ mode: MinimapMode; startRatio: number; startView: [number, number] } | null>(null);
  const { width } = useElementSize(areaRef);
  const [cursor, setCursor] = useState<string>('pointer');
  const last = Math.max(1, maxFrames - 1);
  const pct = (f: number) => `${(f / last) * 100}%`;

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0) return;
    const ctx = prepareCanvas(canvas, width, height);
    if (!ctx) return;
    const x = (f: number) => (f / last) * width;
    // coverage strip along the bottom
    const stripH = 8;
    for (const s of health) {
      ctx.fillStyle = HEALTH_COLORS[s.state];
      ctx.globalAlpha = s.state === 'unscanned' ? 1 : 0.9;
      const left = x(s.start);
      const w = Math.max(s.state === 'risk' || s.state === 'lost' ? 2 : 0, x(s.end + 1) - left);
      ctx.fillRect(left, height - stripH, w, stripH);
    }
    ctx.globalAlpha = 1;
    // one thin line per track above it
    const top = 4;
    const rowH = Math.max(2, Math.min(5, Math.floor((height - stripH - top - 4) / Math.max(1, tracks.length))));
    tracks.forEach((t, i) => {
      ctx.fillStyle = t.color;
      ctx.globalAlpha = 0.85;
      for (const [a, b] of t.ranges) ctx.fillRect(x(a), top + i * (rowH + 1), Math.max(1.5, x(b + 1) - x(a)), rowH);
    });
    ctx.globalAlpha = 1;
    // markers
    for (const m of markers) {
      ctx.fillStyle = m.color || '#38bdf8';
      ctx.fillRect(Math.round(x(m.frame)) - 1, 0, 2, 6);
    }
  }, [health, tracks, markers, width, height, last]);

  const ratioOf = (clientX: number) => {
    const rect = areaRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return 0;
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || maxFrames < 2) return;
    e.preventDefault();
    e.currentTarget.focus({ preventScroll: true }); // preventDefault stops focus moving: take it, so the arrow keys pan this window
    const ratio = ratioOf(e.clientX);
    let mode = hitMinimap(ratio, view, maxFrames, width || 1);
    let startView: [number, number] = [view[0], view[1]];
    const recentred = mode === 'jump';
    if (recentred) {
      // a click outside the window recentres it there; the rest of the drag then pans that window
      startView = dragMinimap({ mode, startRatio: ratio, startView, ratio, maxFrames });
      onViewChange(startView);
      mode = 'move';
    }
    dragRef.current = { mode, startRatio: ratio, startView };
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setCursor(recentred ? 'grabbing' : CURSORS[mode]);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const ratio = ratioOf(e.clientX);
    const drag = dragRef.current;
    if (!drag) {
      setCursor(CURSORS[hitMinimap(ratio, view, maxFrames, width || 1)]);
      return;
    }
    onViewChange(dragMinimap({ ...drag, ratio, maxFrames }));
  };

  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setCursor(CURSORS[hitMinimap(ratioOf(e.clientX), view, maxFrames, width || 1)]);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const span = view[1] - view[0];
    let next: [number, number] | null = null;
    if (e.key === 'ArrowLeft') next = [Math.max(0, view[0] - Math.ceil(span / 4)), 0];
    else if (e.key === 'ArrowRight') next = [Math.min(last - span, view[0] + Math.ceil(span / 4)), 0];
    else if (e.key === 'Home') next = [0, 0];
    else if (e.key === 'End') next = [last - span, 0];
    if (!next) return;
    e.preventDefault();
    e.stopPropagation(); // arrows step frames globally; on the focused navigator they pan the window
    const start = Math.max(0, next[0]);
    onViewChange([start, start + span]);
  };

  const fullyZoomedOut = view[0] <= 0 && view[1] >= last;

  return (
    <div className="flex w-full bg-black/40 border-b border-white/10" data-testid="nle-minimap">
      <div className="shrink-0 flex items-center px-3 border-r border-white/10 text-mini font-semibold text-neutral-400 uppercase tracking-wider bg-[var(--surface-2,#1a1d26)]" style={{ width: headerWidth }}>
        Overview
      </div>
      <div
        ref={areaRef}
        className="relative flex-1 touch-none outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--accent-cyan,#00F0FF)]"
        style={{ height, cursor }}
        role="slider"
        tabIndex={0}
        aria-label="Timeline overview. Drag the window to pan, its edges to zoom"
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={view[0]}
        aria-valuetext={`Showing frames ${view[0]} to ${view[1]} of ${last}`}
        data-testid="nle-minimap-area"
        data-view-start={view[0]}
        data-view-end={view[1]}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onKeyDown={onKeyDown}
      >
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" aria-hidden="true" />
        {/* the clip outside the window, dimmed */}
        <div className="absolute inset-y-0 left-0 bg-black/55 pointer-events-none" style={{ width: pct(view[0]) }} />
        <div className="absolute inset-y-0 right-0 bg-black/55 pointer-events-none" style={{ width: `${100 - (view[1] / last) * 100}%` }} />
        {/* In / Out region */}
        <div className="absolute bottom-0 h-[3px] bg-amber-400/80 pointer-events-none" style={{ left: pct(trimIn), width: `${((Math.max(trimOut, trimIn) - trimIn) / last) * 100}%` }} />
        {/* the window */}
        {!fullyZoomedOut && (
          <div
            className="absolute inset-y-0 border border-[var(--accent-cyan,#00F0FF)]/80 bg-[var(--accent-cyan,#00F0FF)]/10 pointer-events-none"
            style={{ left: pct(view[0]), width: `${((view[1] - view[0]) / last) * 100}%`, minWidth: 6 }}
            data-testid="nle-minimap-window"
          >
            <span className="absolute inset-y-1 -left-[3px] w-[5px] rounded-sm bg-[var(--accent-cyan,#00F0FF)]" />
            <span className="absolute inset-y-1 -right-[3px] w-[5px] rounded-sm bg-[var(--accent-cyan,#00F0FF)]" />
          </div>
        )}
        <div className="absolute inset-y-0 w-px bg-[var(--accent-cyan,#00F0FF)] pointer-events-none" style={{ left: pct(frame) }} />
      </div>
    </div>
  );
}

export const MinimapNavigator = React.memo(MinimapNavigatorImpl);
export default MinimapNavigator;
