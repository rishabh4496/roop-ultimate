import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../lib/api";
import type { PreviewResult, RenderParams } from "../lib/types";

export const SCRUB_DEBOUNCE_MS = 100;

export interface PreviewState {
  /** The latest completed preview (its object URL is owned by this hook). */
  result: PreviewResult | null;
  /** A request is in flight. */
  busy: boolean;
  error: string | null;
  /** Ask for `frame`; debounced, and any earlier request still in flight is aborted. */
  request: (frame: number) => void;
  /** Render the last requested frame again now (e.g. the "Preview frame" button). */
  refresh: () => void;
  clear: () => void;
}

/**
 * Single-frame GPU previews for scrubbing and live parameter tuning.
 *
 * Every request waits `debounceMs` of quiet, then aborts the previous fetch
 * (AbortController) so a fast scrub never queues stale frames behind the one
 * the user stopped on. While `live` is on, a parameter change re-renders the
 * current frame. Object URLs are revoked as they are replaced and on unmount.
 */
export function usePreview(
  params: RenderParams,
  { live, debounceMs = SCRUB_DEBOUNCE_MS }: { live: boolean; debounceMs?: number },
): PreviewState {
  const [result, setResult] = useState<PreviewResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const frame = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const inflight = useRef<AbortController | null>(null);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const replace = useCallback((next: PreviewResult | null) => {
    setResult((old) => {
      if (old && old.url !== next?.url) URL.revokeObjectURL(old.url);
      return next;
    });
  }, []);

  const run = useCallback(async () => {
    inflight.current?.abort();
    const controller = new AbortController();
    inflight.current = controller;
    setBusy(true);
    try {
      const next = await api.preview(frame.current, paramsRef.current, "swapped", controller.signal);
      if (controller.signal.aborted) {
        URL.revokeObjectURL(next.url);
        return;
      }
      setError(null);
      replace(next);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setError(err instanceof ApiError ? err.detail : (err as Error).message);
    } finally {
      if (inflight.current === controller) {
        inflight.current = null;
        setBusy(false);
      }
    }
  }, [replace]);

  const request = useCallback(
    (f: number) => {
      frame.current = f;
      clearTimeout(timer.current);
      timer.current = setTimeout(() => void run(), debounceMs);
    },
    [debounceMs, run],
  );

  const refresh = useCallback(() => {
    clearTimeout(timer.current);
    void run();
  }, [run]);

  const clear = useCallback(() => {
    clearTimeout(timer.current);
    inflight.current?.abort();
    replace(null);
  }, [replace]);

  // Live tuning: a parameter change re-renders the frame on screen.
  const hasResult = result !== null;
  useEffect(() => {
    if (live && hasResult) request(frame.current);
  }, [params]);

  useEffect(
    () => () => {
      clearTimeout(timer.current);
      inflight.current?.abort();
      setResult((old) => {
        if (old) URL.revokeObjectURL(old.url);
        return null;
      });
    },
    [],
  );

  return { result, busy, error, request, refresh, clear };
}
