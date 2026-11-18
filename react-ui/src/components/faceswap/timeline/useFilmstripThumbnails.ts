/**
 * useFilmstripThumbnails.ts: LRU tile cache and request scheduler for V1 Video Filmstrip Lane.
 * 100% Offline, frame-bounded stills for smooth scrubbing.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { API } from '../../../api';
import { filmstripSlots, toServerFrame } from '../nle/filmstripModel';

export interface ThumbnailSlot {
  slotIndex: number;
  frame: number; // exact integer frame_idx (0-based)
  stride?: number;
  leftPct: number;
  widthPct: number;
}

export interface UseFilmstripOptions {
  targetIndex?: number | null;
  targetKey?: string;
  viewStart: number;
  viewEnd: number;
  maxFrames: number;
  trackWidth?: number;
  enabled?: boolean;
}

export function useFilmstripThumbnails({
  targetIndex = null,
  targetKey = '',
  viewStart,
  viewEnd,
  maxFrames,
  trackWidth = 800,
  enabled = true,
}: UseFilmstripOptions) {
  const cacheRef = useRef<Map<number, ImageBitmap>>(new Map());
  const inflightRef = useRef<{ ctrl: AbortController; frames: Set<number> } | null>(null);
  const failedRef = useRef<Map<number, number>>(new Map());
  const [version, setVersion] = useState(0);

  // Slots sit on a stride grid (see nle/filmstripModel.ts): density follows pixels, and panning
  // reuses every still already cached instead of re-requesting the whole strip.
  const slots = useMemo<ThumbnailSlot[]>(
    () => filmstripSlots({ viewStart, viewEnd, widthPx: trackWidth, maxFrames }),
    [trackWidth, viewStart, viewEnd, maxFrames],
  );

  // Clear cache when targetKey changes
  const prevKeyRef = useRef(targetKey);
  useEffect(() => {
    if (prevKeyRef.current !== targetKey) {
      for (const bmp of cacheRef.current.values()) {
        try {
          bmp.close?.();
        } catch {
          // ignore
        }
      }
      cacheRef.current.clear();
      failedRef.current.clear();
      inflightRef.current?.ctrl.abort();
      inflightRef.current = null;
      prevKeyRef.current = targetKey;
      setVersion((v) => v + 1);
    }
  }, [targetKey]);

  // Fetch missing frames
  useEffect(() => {
    if (!enabled || targetIndex === null || targetIndex < 0 || slots.length === 0) {
      return;
    }

    const now = Date.now();
    const wanted: number[] = [];

    for (const s of slots) {
      if (
        cacheRef.current.has(s.frame) ||
        (failedRef.current.get(s.frame) || 0) > now
      ) {
        continue;
      }
      wanted.push(s.frame);
    }

    if (wanted.length === 0) return;

    const timer = setTimeout(async () => {
      inflightRef.current?.ctrl.abort();

      const batch = [...new Set(wanted)].sort((a, b) => a - b).slice(0, 16);
      const ctrl = new AbortController();
      inflightRef.current = { ctrl, frames: new Set(batch) };

      try {
        // The endpoint takes 1-based UI frames; `batch` is 0-based and ascending, which is also the
        // order the response comes back in, so the length-prefixed stills below line up with it.
        const url = `${API}/api/target/preview_grid?index=${targetIndex}&frames=${batch.map(toServerFrame).join(',')}&width=160`;
        const res = await fetch(url, { signal: ctrl.signal });

        if (!res.ok) throw new Error('Failed to fetch preview grid');

        const buf = new Uint8Array(await res.arrayBuffer());
        let off = 0;
        let batchIdx = 0;
        const decoded: Array<Promise<{ frame: number; bmp: ImageBitmap } | null>> = [];

        while (off + 4 <= buf.length && batchIdx < batch.length) {
          const len =
            ((buf[off] << 24) |
              (buf[off + 1] << 16) |
              (buf[off + 2] << 8) |
              buf[off + 3]) >>>
            0;
          off += 4;
          if (off + len > buf.length) break;

          const frame = batch[batchIdx++];
          if (len === 0) {
            failedRef.current.set(frame, Date.now() + 15000);
          } else {
            const blob = new Blob([buf.slice(off, off + len)], { type: 'image/jpeg' });
            decoded.push(
              createImageBitmap(blob)
                .then((bmp) => ({ frame, bmp }))
                .catch(() => null)
            );
          }
          off += len;
        }

        const hits = await Promise.all(decoded);
        for (const hit of hits) {
          if (!hit) continue;
          cacheRef.current.set(hit.frame, hit.bmp);
        }
        // Anything asked for that did not arrive (an undecodable still, a short response) sits out a
        // cool-down. Without this the `version` bump below re-runs the effect, which asks for the same
        // missing stills again, forever: measured as an unbounded stream of identical requests.
        const missingUntil = Date.now() + 15000;
        for (const f of batch) if (!cacheRef.current.has(f)) failedRef.current.set(f, missingUntil);

        // Bound LRU to 150 stills
        while (cacheRef.current.size > 150) {
          const oldestKey = cacheRef.current.keys().next().value;
          if (oldestKey !== undefined) {
            cacheRef.current.get(oldestKey)?.close?.();
            cacheRef.current.delete(oldestKey);
          }
        }

        if (inflightRef.current?.ctrl === ctrl) {
          inflightRef.current = null;
        }
        setVersion((v) => v + 1);
      } catch (err: any) {
        if (err?.name !== 'AbortError') {
          for (const f of batch) {
            failedRef.current.set(f, Date.now() + 15000);
          }
        }
        if (inflightRef.current?.ctrl === ctrl) {
          inflightRef.current = null;
        }
      }
    }, 120);

    return () => clearTimeout(timer);
    // `version` is a dependency on purpose: one request carries at most 16 stills, so a wide strip
    // needs the effect to run again once a batch lands to ask for the rest. It terminates: stills
    // that arrived are cached and ones that failed sit out a cool-down.
  }, [slots, enabled, targetIndex, targetKey, version]);

  /**
   * Return the cached still for `frame` or nearest within reach frames.
   */
  const getThumbnail = useCallback(
    (frame: number, reach = 10): ImageBitmap | null => {
      const exact = cacheRef.current.get(frame);
      if (exact) return exact;

      if (reach <= 0) return null;

      let bestBmp: ImageBitmap | null = null;
      let minDiff = reach + 1;

      for (const [f, bmp] of cacheRef.current.entries()) {
        const diff = Math.abs(f - frame);
        if (diff < minDiff) {
          minDiff = diff;
          bestBmp = bmp;
        }
      }

      return bestBmp;
    },
    []
  );

  return {
    slots,
    getThumbnail,
    cacheSize: cacheRef.current.size,
    version,
  };
}
