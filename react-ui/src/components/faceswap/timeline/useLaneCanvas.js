import { useCallback, useEffect, useRef } from 'react';
import { readTheme } from './lanes';

/**
 * A <canvas> that repaints itself when `deps` change and when it is resized, at most once per
 * animation frame however many changes arrive.
 *
 * `draw(ctx, {w, h, theme})` paints in CSS pixels (the context is pre-scaled by the device pixel
 * ratio, capped at 2 so a 4K display does not quadruple the fill cost of a lane). Painting is
 * skipped while the canvas has no size, and the backing store is only reallocated when the size
 * actually changes - resizing a canvas clears it and is the expensive part.
 */
export function useLaneCanvas(draw, deps) {
  const ref = useRef(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });
  const rafRef = useRef(0);

  const paint = useCallback(() => {
    rafRef.current = 0;
    const canvas = ref.current;
    const { w, h, dpr } = sizeRef.current;
    if (!canvas || !w || !h) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    drawRef.current(ctx, { w, h, dpr, theme: readTheme(canvas) });
  }, []);

  const schedule = useCallback(() => {
    if (!rafRef.current) rafRef.current = requestAnimationFrame(paint);
  }, [paint]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const measure = () => {
      const rect = canvas.getBoundingClientRect();
      const w = Math.round(rect.width);
      const h = Math.round(rect.height);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const bw = Math.round(w * dpr);
      const bh = Math.round(h * dpr);
      if (canvas.width !== bw) canvas.width = bw;
      if (canvas.height !== bh) canvas.height = bh;
      sizeRef.current = { w, h, dpr };
      schedule();
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(canvas);
    return () => {
      ro.disconnect();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    };
  }, [schedule]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { schedule(); }, deps);

  return ref;
}
