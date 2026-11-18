/**
 * FilmstripCanvas: the V1 lane's stills on ONE canvas.
 *
 * The old lane mounted a <canvas> per slot with an inline ref callback that redrew on every render,
 * so each playhead move repainted every still. Here the strip is a single canvas that repaints when
 * the slots, the stills (`version`) or the size change.
 */

import React, { useLayoutEffect, useRef } from 'react';
import { prepareCanvas, useElementSize } from './useElementSize';

/** What the canvas needs of a slot: the server hook's and the browser hook's slots both fit. */
export interface FilmstripSlot {
  frame: number;
  leftPct: number;
  widthPct: number;
  stride?: number;
}

export interface FilmstripCanvasProps {
  slots: ReadonlyArray<FilmstripSlot>;
  /** nearest cached still for a 0-based frame (exact hit preferred) */
  getThumbnail: (frame: number, reach?: number) => ImageBitmap | null;
  /** bumps whenever a still arrives, so the strip repaints */
  version: number;
  className?: string;
}

function FilmstripCanvasImpl({ slots, getThumbnail, version, className = '' }: FilmstripCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width, height } = useElementSize(wrapRef);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0 || height <= 0) return;
    const ctx = prepareCanvas(canvas, width, height);
    if (!ctx) return;
    ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
    ctx.textBaseline = 'middle';

    for (const slot of slots) {
      const x0 = Math.round((slot.leftPct / 100) * width);
      const x1 = Math.round(((slot.leftPct + slot.widthPct) / 100) * width);
      const w = Math.max(1, x1 - x0);
      // Exact still if it has arrived; otherwise the nearest one within a stride, so a zoom change
      // shows a (slightly early/late) picture instead of a blank box while the exact one loads.
      const bmp = getThumbnail(slot.frame, Math.max(1, slot.stride ?? 1));
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, 0, w, height);
      ctx.clip();
      if (bmp) {
        // object-fit: cover
        const scale = Math.max(w / bmp.width, height / bmp.height);
        const dw = bmp.width * scale;
        const dh = bmp.height * scale;
        ctx.drawImage(bmp, x0 + (w - dw) / 2, (height - dh) / 2, dw, dh);
      } else {
        ctx.fillStyle = 'rgba(255,255,255,0.04)';
        ctx.fillRect(x0, 0, w, height);
        ctx.fillStyle = 'rgba(255,255,255,0.38)';
        ctx.fillText(`#${String(slot.frame).padStart(5, '0')}`, x0 + 6, height / 2);
      }
      ctx.restore();
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.fillRect(x1 - 1, 0, 1, height);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `version` is the change signal for getThumbnail's cache
  }, [slots, version, width, height, getThumbnail]);

  return (
    <div ref={wrapRef} className={`absolute inset-0 ${className}`}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        data-testid="nle-filmstrip-canvas"
        data-slot-count={slots.length}
        data-slot-stride={slots[0]?.stride ?? 0}
        aria-hidden="true"
      />
    </div>
  );
}

export const FilmstripCanvas = React.memo(FilmstripCanvasImpl);
export default FilmstripCanvas;
