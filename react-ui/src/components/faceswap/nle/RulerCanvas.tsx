/**
 * RulerCanvas: the precision ruler. SMPTE timecode on the first line of every major tick and the exact
 * frame number (`#00412`) on the second; tick density follows pixels per frame (see rulerModel.ts).
 *
 * It redraws only when the visible window, frame rate or size changes. The playhead is a separate
 * overlay, so playback does not repaint the ruler.
 */

import React, { useLayoutEffect, useMemo, useRef } from 'react';
import { buildRuler } from './rulerModel';
import { prepareCanvas, useElementSize } from './useElementSize';

export interface RulerCanvasProps {
  viewStart: number;
  viewEnd: number;
  fps: number;
  height?: number;
  className?: string;
  /** colour of the second (frame-number) line; defaults to the theme's cyan token */
  accent?: string;
}

export const RULER_HEIGHT = 40;

function RulerCanvasImpl({ viewStart, viewEnd, fps, height = RULER_HEIGHT, className = '', accent }: RulerCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrapRef);

  const layout = useMemo(
    () => buildRuler({ viewStart, viewEnd, widthPx: Math.max(1, width), fps }),
    [viewStart, viewEnd, width, fps],
  );

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0) return;
    const ctx = prepareCanvas(canvas, width, height);
    if (!ctx) return;
    const style = getComputedStyle(canvas);
    const ink = style.color || '#d4d4d8';
    const cyan = accent || style.getPropertyValue('--accent-cyan').trim() || '#00F0FF';

    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.textBaseline = 'top';

    // baseline
    ctx.fillStyle = 'rgba(255,255,255,0.14)';
    ctx.fillRect(0, height - 1, width, 1);

    for (const tick of layout.ticks) {
      const x = Math.round(tick.ratio * width) + 0.5;
      const tall = tick.major ? 14 : 6;
      ctx.fillStyle = tick.major ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.26)';
      ctx.fillRect(x - 0.5, height - 1 - tall, 1, tall);
      if (!tick.major) continue;
      ctx.fillStyle = ink;
      ctx.fillText(tick.timecode, x + 4, 4);
      ctx.fillStyle = cyan;
      ctx.globalAlpha = 0.85;
      ctx.fillText(tick.frameLabel, x + 4, 16);
      ctx.globalAlpha = 1;
    }
  }, [layout, width, height, accent]);

  return (
    <div ref={wrapRef} className={`relative w-full ${className}`} style={{ height }}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full text-neutral-300"
        data-testid="nle-ruler-canvas"
        data-minor-step={layout.minorStep}
        data-major-step={layout.majorStep}
        data-major-count={layout.ticks.filter((t) => t.major).length}
        data-first-label={layout.ticks.find((t) => t.major)?.frameLabel ?? ''}
        aria-hidden="true"
      />
    </div>
  );
}

export const RulerCanvas = React.memo(RulerCanvasImpl);
export default RulerCanvas;
