import { useEffect, useRef, useState } from 'react';

/**
 * Load `url` into a displayable src, ONE request at a time, latest wins.
 *
 * Why this exists: pointing an <img> straight at a URL that changes as fast as
 * the pointer moves issues a request per intermediate value. Every one of those
 * is a video seek on the server — 40-200 ms on long-GOP footage, serialised
 * behind a single decoder — so a two-second drag queued dozens of decodes and
 * the frame you actually stopped on came out at the BACK of that queue. The
 * preview lagged the playhead by seconds, and got worse the longer you dragged.
 *
 * The fix is not to throttle (which just picks a different arbitrary number of
 * wasted decodes) but to let the server's own speed set the rate: keep exactly
 * one request in flight, remember the latest URL asked for, and start it the
 * moment the current one lands. Intermediate positions the user swept past are
 * never requested at all, the frame settled on is always requested, and the
 * queue can never build up.
 *
 * Each completed load IS shown — the requests are strictly ordered, so whatever
 * lands is newer than what is on screen, which is what makes the drag read as
 * continuous rather than as a jump at the end. Between loads the previously
 * loaded frame stays up rather than blanking.
 */
export default function useSequentialImage(url) {
  const [src, setSrc] = useState('');
  const wantedRef = useRef(url);
  const loadingRef = useRef(false);
  const aliveRef = useRef(true);

  // Re-arm on mount, not just disarm on unmount: StrictMode mounts, cleans up
  // and mounts again with these same refs, so a cleanup-only version would latch
  // `alive` false on the second mount and never show a frame again in dev.
  useEffect(() => {
    aliveRef.current = true;
    return () => { aliveRef.current = false; };
  }, []);

  useEffect(() => {
    wantedRef.current = url || '';
    if (!url) {
      setSrc('');
      return;
    }
    if (loadingRef.current) return;   // the drain below will pick this up

    const start = (u) => {
      loadingRef.current = true;
      const img = new Image();
      // The <img> the caller renders points at this same URL, so it comes out of
      // the browser's cache rather than off the wire a second time; the endpoint
      // answers any revalidation from a stat, without decoding anything.
      const done = () => {
        loadingRef.current = false;
        if (!aliveRef.current) return;
        const next = wantedRef.current;
        if (next && next !== u) start(next);
      };
      img.onload = () => { if (aliveRef.current) setSrc(u); done(); };
      img.onerror = done;
      img.src = u;
    };
    start(url);
  }, [url]);

  return src;
}
