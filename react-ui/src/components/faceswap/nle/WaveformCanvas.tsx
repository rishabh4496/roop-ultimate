/**
 * WaveformCanvas: the A1 lane's peaks. High contrast on purpose (bright body on near-black, clipped
 * peaks in rose) so a dialogue beat is visible at a glance and trim points can be aligned to it.
 *
 * Max-pools the peaks that fall under each pixel column (the old drawing sampled one bucket per bar
 * and skipped the rest, so a zoomed-out view dropped transients). It repaints only when the peaks,
 * the visible time window or the size change, not on every playhead move, which the previous inline
 * ref-callback did (a fresh callback each render re-ran the whole draw 25-60 times a second).
 */

import React, { useLayoutEffect, useRef } from 'react';
import { prepareCanvas, useElementSize } from './useElementSize';

export interface WaveformCanvasProps {
  /** magnitudes 0..255 spread evenly over `duration` seconds */
  peaks: ArrayLike<number>;
  duration: number;
  /** visible window in seconds */
  tStart: number;
  tEnd: number;
  muted?: boolean;
  className?: string;
}

const CLIP_AT = 0.92;

function WaveformCanvasImpl({ peaks, duration, tStart, tEnd, muted = false, className = '' }: WaveformCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width, height } = useElementSize(wrapRef);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0 || height <= 0) return;
    const ctx = prepareCanvas(canvas, width, height);
    if (!ctx) return;

    const mid = height / 2;
    ctx.fillStyle = 'rgba(255,255,255,0.10)';
    ctx.fillRect(0, Math.floor(mid), width, 1);

    const n = peaks.length;
    const dur = Math.max(0.001, duration);
    if (n === 0) return;
    const perSec = n / dur;
    const w0 = Math.max(0, tStart) * perSec;
    const w1 = Math.min(dur, Math.max(tEnd, tStart + 1e-3)) * perSec;
    const perPx = (w1 - w0) / width;

    const body = muted ? 'rgba(161,161,170,0.45)' : '#5eead4';
    const hot = muted ? 'rgba(161,161,170,0.45)' : '#fb7185';
    for (let x = 0; x < width; x++) {
      const a = Math.floor(w0 + x * perPx);
      const b = Math.max(a + 1, Math.ceil(w0 + (x + 1) * perPx));
      let peak = 0;
      for (let i = a; i < b && i < n; i++) if (peaks[i] > peak) peak = peaks[i];
      const norm = Math.min(1, peak / 255);
      if (norm <= 0) continue;
      const h = Math.max(1, norm * (mid - 1));
      ctx.fillStyle = norm >= CLIP_AT ? hot : body;
      ctx.fillRect(x, mid - h, 1, h * 2);
    }
  }, [peaks, duration, tStart, tEnd, muted, width, height]);

  return (
    <div ref={wrapRef} className={`absolute inset-0 ${className}`}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" data-testid="nle-waveform-canvas" aria-hidden="true" />
    </div>
  );
}

export const WaveformCanvas = React.memo(WaveformCanvasImpl);
export default WaveformCanvas;
