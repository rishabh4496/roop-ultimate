/**
 * ScrubTooltip: the floating card above the playhead while scrubbing (or hovering the ruler):
 * a still of the frame under the cursor, its exact frame number, SMPTE timecode and seconds.
 *
 * Rendered through a portal with `position: fixed` because the timeline root is `overflow-hidden` and
 * the card has to rise ABOVE it; inside the root it would be clipped to nothing.
 */

import React, { useLayoutEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { formatFrameLabel } from './rulerModel';

export interface ScrubTooltipProps {
  visible: boolean;
  /** viewport x the card centres on (the playhead / pointer) */
  clientX: number;
  /** viewport y of the top of the timeline: the card sits above it */
  anchorTop: number;
  frame: number;
  timecode: string;
  seconds: string;
  bitmap: ImageBitmap | null;
  /** true while `bitmap` is a neighbouring still, not the exact frame */
  approximate?: boolean;
}

const CARD_W = 184;
const MARGIN = 8;

function ScrubTooltipImpl({ visible, clientX, anchorTop, frame, timecode, seconds, bitmap, approximate }: ScrubTooltipProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const picW = CARD_W - 8;
  const picH = bitmap ? Math.max(56, Math.min(150, Math.round((picW * bitmap.height) / bitmap.width))) : Math.round((picW * 9) / 16);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !visible) return;
    canvas.width = picW * 2;
    canvas.height = picH * 2;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (bitmap) ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  }, [bitmap, visible, picW, picH]);

  if (!visible || typeof document === 'undefined') return null;
  const vw = typeof window !== 'undefined' ? window.innerWidth : 1280;
  const vh = typeof window !== 'undefined' ? window.innerHeight : 720;
  const left = Math.max(MARGIN, Math.min(vw - CARD_W - MARGIN, clientX - CARD_W / 2));

  return createPortal(
    <div
      role="status"
      aria-label={`Frame ${frame}, ${timecode}`}
      data-testid="nle-scrub-tooltip"
      data-frame={frame}
      className="pointer-events-none fixed z-[2000] rounded-lg border border-[var(--accent-cyan,#00F0FF)]/50 bg-black/92 p-1 shadow-2xl"
      style={{ left, bottom: Math.max(MARGIN, vh - anchorTop + 12), width: CARD_W }}
    >
      <canvas ref={canvasRef} className="block rounded bg-black" style={{ width: picW, height: picH }} data-testid="nle-scrub-thumb" data-approximate={approximate ? '1' : '0'} />
      <div className="flex items-baseline justify-between gap-2 px-1 pt-1 font-mono tabular-nums">
        <span className="text-plain font-bold text-[var(--accent-cyan,#00F0FF)]" data-testid="nle-scrub-frame">{formatFrameLabel(frame)}</span>
        <span className="text-mini text-neutral-300">{seconds}</span>
      </div>
      <div className="px-1 pb-0.5 text-mini font-mono tabular-nums text-white" data-testid="nle-scrub-timecode">{timecode}</div>
    </div>,
    document.body,
  );
}

export const ScrubTooltip = React.memo(ScrubTooltipImpl);
export default ScrubTooltip;
