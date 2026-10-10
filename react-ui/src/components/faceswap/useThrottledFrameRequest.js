import { useCallback, useEffect, useRef, useState } from 'react';
import { decodeFrame, releaseFrame } from './frameDecoder';
import { createFailureTracker, isAbort } from './retryPolicy';

// ── One frame request at a time, throttled, and cancelled the moment it is
//    superseded ────────────────────────────────────────────────────────────
//
// This replaces useSequentialImage, which was right about the SHAPE of the
// problem and could not act on it. Its notes are worth keeping:
//
//   pointing an <img> straight at a URL that changes as fast as the pointer
//   moves issues a request per intermediate value, and every one of those is a
//   video seek on the server — 40-200 ms on long-GOP footage, serialised behind
//   a single decoder — so a two-second drag queued dozens of decodes and the
//   frame you actually stopped on came out at the BACK of that queue.
//
// It solved that with single-flight + latest-wins, using `new Image()`. Two
// things that cannot do:
//
//   * an <img> load CANNOT BE CANCELLED. "Latest wins" only decided which
//     result to ignore; the request the user had already dragged past still ran
//     to completion, still occupied the server's one decoder, and still paid a
//     full JPEG decode on the main thread. The frame you stopped on waited
//     behind work whose result was thrown away.
//   * with no throttle, a fast decoder is asked for a frame per pointer sample
//     (~120/s on a high-rate mouse). Single-flight caps the QUEUE at one, not
//     the RATE.
//
// So: a 150 ms throttle (leading edge, so the first move is instant, plus a
// trailing edge, so the frame you settle on is always requested), and an
// AbortController per request that is aborted the instant a newer frame is
// wanted — which cancels the fetch AND the decode, in the worker, together.
//
// 150 ms is chosen against the backend, not by feel: a seek on this pipeline is
// a flat ~125-180 ms whatever the distance (see the filmstrip note in
// FaceSwap), so a shorter period cannot produce more frames — it only produces
// more cancelled work. The throttle is bypassed entirely when the URL settles
// (a keyboard step, a click on the track), because there is nothing to coalesce.
//
// ── A failed request is not a reason to ask again at once ─────────────────
//
// `drain` re-runs from every request's `finally`, and "the wanted URL is not the
// frame on screen" stays true after a FAILURE — so with a backend answering 500
// this hook was a request loop, one every throttle period: 196 in 30 s from an
// idle tab (react-ui/e2e/idle-requests.spec.js). Failures are now counted per
// URL (retryPolicy.js): each pushes the next attempt out exponentially with
// jitter (0.5 s .. 8 s), the fifth consecutive one gives up, and the hook
// returns `error` and `retry` so the stage can say so instead of silently
// holding a frame that is not the one asked for. The count resets when the URL
// (or `enabled`) changes and on `retry()`. An AbortError is the normal end of a
// superseded request and is never counted.

const DEFAULT_THROTTLE_MS = 150;

export default function useThrottledFrameRequest(url, {
  throttleMs = DEFAULT_THROTTLE_MS,
  /** Pass false to hold the current frame and issue nothing (e.g. tab hidden). */
  enabled = true,
} = {}) {
  // The decoded frame is NOT state — handing an ImageBitmap to React would put
  // it in the render path. It goes to the canvas through this ref-like box,
  // and `version` is the only thing that re-renders, so consumers can react to
  // "a new frame exists" without the frame itself moving through a commit.
  const frameRef = useRef({ img: null, src: '' });
  const [version, setVersion] = useState(0);
  const [pending, setPending] = useState(false);
  // Set once the wanted URL has failed its whole retry budget; cleared by a URL
  // change or retry(). While retries are still being spent it stays null, so a
  // single blip never flashes an error over a working preview.
  const [error, setError] = useState(null);

  const wantedRef = useRef('');
  const inFlightRef = useRef(null);     // { ctrl, src }
  const lastStartRef = useRef(0);
  const timerRef = useRef(null);
  const aliveRef = useRef(true);
  const throttleRef = useRef(throttleMs);
  throttleRef.current = throttleMs;
  const failuresRef = useRef(null);
  if (failuresRef.current === null) failuresRef.current = createFailureTracker();
  // Earliest time the next attempt may start, when the last one failed.
  const notBeforeRef = useRef(0);
  const drainRef = useRef(() => {});

  // Re-arm on mount, not just disarm on unmount: StrictMode mounts, cleans up
  // and mounts again with these same refs, so a cleanup-only version would
  // latch `alive` false on the second mount and never show a frame again in dev.
  useEffect(() => {
    aliveRef.current = true;
    return () => { aliveRef.current = false; };
  }, []);

  const start = useCallback((src) => {
    lastStartRef.current = performance.now();
    const ctrl = new AbortController();
    inFlightRef.current = { ctrl, src };
    setPending(true);

    decodeFrame(src, { signal: ctrl.signal }).then((img) => {
      if (!aliveRef.current || ctrl.signal.aborted) { releaseFrame(img); return; }
      failuresRef.current.ok(src);
      notBeforeRef.current = 0;
      // Release the frame being replaced. A 4K ImageBitmap is ~33 MB of
      // resident memory that nothing collects until it is closed, so a scrub
      // that skipped this would grow the heap by a frame per tick.
      releaseFrame(frameRef.current.img);
      frameRef.current = { img, src };
      setVersion((v) => v + 1);
      setError(null);
    }).catch((err) => {
      // AbortError is the normal outcome for every frame the user swept past —
      // not a failure, and not counted. A genuine failure leaves the previous
      // frame up rather than blanking, and schedules the next attempt later.
      if (isAbort(err) || ctrl.signal.aborted || !aliveRef.current) return;
      if (wantedRef.current !== src) return;      // the user has moved on
      const f = failuresRef.current.fail(src);
      if (f.exhausted) {
        setError(Object.assign(new Error('Frame unavailable'), {
          attempts: f.count, url: src, cause: err,
        }));
      } else {
        notBeforeRef.current = performance.now() + f.delay;
      }
    }).finally(() => {
      if (inFlightRef.current?.ctrl === ctrl) {
        inFlightRef.current = null;
        if (aliveRef.current) setPending(false);
      }
      drainRef.current();
    });
  }, []);

  // Issue the newest wanted URL, respecting the throttle period and, after a
  // failure, the backoff.
  const drain = useCallback(() => {
    if (!aliveRef.current) return;
    const want = wantedRef.current;
    if (!want || want === frameRef.current.src) return;
    if (failuresRef.current.exhausted(want)) return;  // given up; retry() re-arms
    if (inFlightRef.current) return;                 // finally() re-drains
    const now = performance.now();
    const wait = Math.max(
      throttleRef.current - (now - lastStartRef.current),
      notBeforeRef.current - now,
    );
    if (wait > 0) {
      if (timerRef.current) return;                  // trailing edge already armed
      timerRef.current = setTimeout(() => { timerRef.current = null; drain(); }, wait);
      return;
    }
    start(want);
  }, [start]);
  drainRef.current = drain;

  useEffect(() => {
    const wanted = enabled ? (url || '') : '';
    if (wanted !== wantedRef.current) {
      // A different URL (or the hook switched off/on) is a fresh start: the old
      // one's failures say nothing about this one, and a backoff timer armed for
      // the old URL must not hold the new one back.
      failuresRef.current.reset();
      notBeforeRef.current = 0;
      if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
      setError(null);
    }
    wantedRef.current = wanted;

    if (!wantedRef.current) {
      // Nothing wanted any more: cancel what is running rather than letting it
      // finish into a frame nobody will look at.
      inFlightRef.current?.ctrl.abort();
      inFlightRef.current = null;
      return undefined;
    }

    // THE ABORT THIS HOOK EXISTS FOR: a request for a frame the pointer has
    // already left is dead work on a serialised decoder. Kill it now, do not
    // wait for it to land and then discard the result.
    const running = inFlightRef.current;
    if (running && running.src !== wantedRef.current) {
      running.ctrl.abort();
      inFlightRef.current = null;
    }
    drain();
    return undefined;
  }, [url, enabled, throttleMs, drain]);

  /** Manual retry: forget the failures and ask again straight away. */
  const retry = useCallback(() => {
    failuresRef.current.reset();
    notBeforeRef.current = 0;
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    setError(null);
    drainRef.current();
  }, []);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    inFlightRef.current?.ctrl.abort();
    inFlightRef.current = null;
    releaseFrame(frameRef.current.img);
    frameRef.current = { img: null, src: '' };
  }, []);

  return {
    /** The decoded frame, or null before the first one lands. */
    frame: frameRef.current.img,
    /** Which URL that frame is of — the caller's staleness check. */
    frameSrc: frameRef.current.src,
    /** Bumps whenever `frame` changes; useful as a render key. */
    version,
    /** True while a request is on the wire. */
    pending,
    /**
     * null, or an Error ({attempts, url}) once the wanted URL failed five times
     * in a row. `frame` is then NOT the frame that was asked for.
     */
    error,
    /** Clear the failure streak and request the wanted URL again now. */
    retry,
  };
}
