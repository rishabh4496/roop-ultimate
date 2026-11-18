/**
 * FaceHealthLane: the face tracking / confidence heatmap, drawn on a canvas.
 *
 * Emerald = a face track covers the frame, amber = a pose / occlusion warning was measured there,
 * rose = no track (a dropout). A clip with no scan is a flat hatched strip, never rose. A click on an
 * amber or rose run jumps to its first frame; a click on emerald scrubs like any other lane (the
 * parent's `onScrubDown`). See faceHealth.ts for what is measured vs. inferred.
 */

import React, { useLayoutEffect, useMemo, useRef } from 'react';
import { HEALTH_COLORS, HEALTH_LABELS, type HealthSegment, problemTargetNear, segmentAt } from './faceHealth';
import { clientXToFrame } from './nleGeometry';
import { prepareCanvas, useElementSize } from './useElementSize';

export interface FaceHealthLaneProps {
  segments: ReadonlyArray<HealthSegment>;
  viewStart: number;
  viewEnd: number;
  maxFrames: number;
  onJump: (frame: number) => void;
  /** pointer down on a non-problem area: the lane scrubs like the others */
  onScrubDown: (e: React.PointerEvent<HTMLDivElement>) => void;
  disabled?: boolean;
}

const MIN_PROBLEM_PX = 3;

function FaceHealthLaneImpl({ segments, viewStart, viewEnd, maxFrames, onJump, onScrubDown, disabled }: FaceHealthLaneProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width, height } = useElementSize(wrapRef);
  const span = Math.max(1, viewEnd - viewStart);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0 || height <= 0) return;
    const ctx = prepareCanvas(canvas, width, height);
    if (!ctx) return;
    const pxPerFrame = width / span;
    for (const seg of segments) {
      if (seg.end < viewStart - 1 || seg.start > viewEnd + 1) continue;
      const x0 = (seg.start - viewStart) * pxPerFrame;
      let w = (seg.end - seg.start + 1) * pxPerFrame;
      const problem = seg.state === 'risk' || seg.state === 'lost';
      if (problem && w < MIN_PROBLEM_PX) w = MIN_PROBLEM_PX;
      const left = Math.max(0, x0);
      const right = Math.min(width, x0 + w);
      if (right <= left) continue;
      ctx.fillStyle = HEALTH_COLORS[seg.state];
      ctx.fillRect(left, 0, right - left, height);
      if (seg.state === 'unscanned') {
        // diagonal hatch: "no data", visibly different from a measured state
        ctx.save();
        ctx.beginPath();
        ctx.rect(left, 0, right - left, height);
        ctx.clip();
        ctx.strokeStyle = 'rgba(255,255,255,0.10)';
        ctx.lineWidth = 1;
        for (let hx = left - height; hx < right; hx += 8) {
          ctx.beginPath();
          ctx.moveTo(hx, height);
          ctx.lineTo(hx + height, 0);
          ctx.stroke();
        }
        ctx.restore();
      }
    }
  }, [segments, viewStart, viewEnd, span, width, height]);

  const frameAt = (clientX: number) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    return clientXToFrame(clientX, rect ? { left: rect.left, width: rect.width } : null, viewStart, viewEnd, maxFrames);
  };
  const tolFrames = () => Math.ceil(6 / Math.max(1e-6, width / span));

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const target = disabled ? null : problemTargetNear(segments, frameAt(e.clientX), tolFrames());
    if (target !== null) {
      e.preventDefault();
      e.stopPropagation();
      onJump(target);
      return;
    }
    onScrubDown(e);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const f = frameAt(e.clientX);
    const seg = segmentAt(segments, f);
    const el = wrapRef.current;
    if (!el || !seg) return;
    const hit = problemTargetNear(segments, f, tolFrames());
    const label = HEALTH_LABELS[seg.state];
    el.title = `${label}: frames ${seg.start}–${seg.end}${hit !== null ? ' · click to jump' : ''}`;
    el.style.cursor = hit !== null ? 'pointer' : 'col-resize';
  };

  const summary = useMemo(() => segments.filter((s) => s.state === 'risk' || s.state === 'lost').length, [segments]);

  return (
    <div
      ref={wrapRef}
      className="absolute inset-0"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      data-testid="nle-health-surface"
      data-problem-runs={summary}
    >
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" data-testid="nle-health-canvas" aria-hidden="true" />
    </div>
  );
}

export const FaceHealthLane = React.memo(FaceHealthLaneImpl);
export default FaceHealthLane;
