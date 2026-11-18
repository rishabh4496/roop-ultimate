/**
 * useScrubThumbnail.ts: the exact still for the frame under the scrub tooltip.
 *
 * The filmstrip only holds stills on its stride grid, so mid-drag the tooltip first shows the nearest
 * one it has (instant, slightly off) and this hook then replaces it with the real frame. Requests are
 * trailing-throttled and each new one aborts the previous, so a fast drag asks for the frames it
 * LANDS on, not every frame it passes through.
 *
 * `fetchFrame` is injected: the server path decodes via /api/target/preview, the Extras path seeks a
 * hidden <video>. Both resolve to an ImageBitmap (or null when the frame cannot be read, in which case
 * the nearest still stays on screen).
 */

import { useEffect, useRef, useState } from 'react';

export type FrameFetcher = (frame: number, signal: AbortSignal) => Promise<ImageBitmap | null>;

const THROTTLE_MS = 90;
const CACHE_MAX = 48;

export function useScrubThumbnail(active: boolean, frame: number | null, fetchFrame: FrameFetcher | null, resetKey: string) {
  const cacheRef = useRef<Map<number, ImageBitmap>>(new Map());
  const [exact, setExact] = useState<{ frame: number; bitmap: ImageBitmap } | null>(null);

  // A different clip invalidates every still.
  useEffect(() => {
    const cache = cacheRef.current;
    return () => {
      for (const bmp of cache.values()) bmp.close?.();
      cache.clear();
    };
  }, [resetKey]);

  useEffect(() => {
    if (!active || frame === null || !fetchFrame) return undefined;
    const cached = cacheRef.current.get(frame);
    if (cached) {
      setExact({ frame, bitmap: cached });
      return undefined;
    }
    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const bmp = await fetchFrame(frame, ctrl.signal);
        if (!bmp || ctrl.signal.aborted) {
          bmp?.close?.();
          return;
        }
        const cache = cacheRef.current;
        cache.set(frame, bmp);
        while (cache.size > CACHE_MAX) {
          const oldest = cache.keys().next().value;
          if (oldest === undefined) break;
          cache.get(oldest)?.close?.();
          cache.delete(oldest);
        }
        setExact({ frame, bitmap: bmp });
      } catch {
        /* aborted or unreadable: the nearest filmstrip still stays up */
      }
    }, THROTTLE_MS);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [active, frame, fetchFrame]);

  // Only hand back a still that belongs to the frame being asked about.
  return exact && exact.frame === frame ? exact.bitmap : null;
}

/** Server fetcher: one JPEG from /api/target/preview (1-based UI frame, small width). */
export function serverFrameFetcher(api: string, targetIndex: number): FrameFetcher {
  return async (frame, signal) => {
    const res = await fetch(`${api}/api/target/preview?index=${targetIndex}&frame=${Math.round(frame) + 1}&width=240`, { signal });
    if (!res.ok || !(res.headers.get('content-type') || '').startsWith('image/')) return null;
    return createImageBitmap(await res.blob());
  };
}
