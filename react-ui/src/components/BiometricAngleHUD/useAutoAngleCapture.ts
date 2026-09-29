// State + transport for the Biometric Angle HUD.
//
// SCAN. One /ws/angle-scan socket per scan: `start` streams progress and ends
// with exactly one of result / error / cancelled, after which the socket is
// closed. Closing it (unmount, a newer scan, Cancel) is also what stops the
// server: it cancels the scanner so a closed tab never leaves the detector busy.
//
// AUTO. A scan starts by itself when the selected target PERSON (or the target
// media) changes, after a short settle so clicking through people does not
// launch a scan per click. It does NOT start when the server already holds a
// session for that person and media: a Pinokio tab switch reloads this app, and
// re-scanning a whole clip on every reload would make the feature unusable. The
// stored session is fetched first (rehydration) and decides. An auto attempt is
// made at most once per person/media pair; Rescan is always available.
//
// THRESHOLDS. The sliders re-gate the candidates the server already measured
// (POST /api/angle-scan/thresholds): no decode, no rescan. Requests are
// debounced and sequenced, so a slower older response never overwrites a newer
// one.
import { useCallback, useEffect, useReducer, useRef } from 'react';
import { getJSON, postJSON } from '../../api.js';
import {
  hudReducer, initialHudState, parseServerEvent, sessionMatches, toFrameIdx,
  type AngleBinName, type AngleSession, type BinEntry, type HudState,
} from './angleHudModel';

const AUTO_SETTLE_MS = 600;
const THRESHOLD_DEBOUNCE_MS = 250;

export interface AutoAngleCaptureOptions {
  targetPersonId: string | null | undefined;
  targetMediaId: string | null | undefined;
  targetIndex: number;
  isVideo: boolean;
  autoScan: boolean;
}

export interface OverrideResult {
  warnings: string[];
  similarity: number | null;
  entry: BinEntry | null;           // the bin as the server now holds it
}

export interface AutoAngleCapture extends HudState {
  startScan: () => void;
  cancelScan: () => void;
  setMinIod: (px: number) => void;
  setBlurFrac: (frac: number) => void;
  selectReferenceBin: (bin: AngleBinName) => void;
  beginOverride: (bin: AngleBinName) => void;
  cancelOverride: () => void;
  assignOverride: (bin: AngleBinName, timelineFrame: number) => Promise<OverrideResult | null>;
  clearOverride: (bin: AngleBinName) => Promise<void>;
  applyToPerson: (bins?: AngleBinName[]) => Promise<Record<string, unknown> | null>;
  dismissMessage: () => void;
  sessionIsCurrent: boolean;
}

const socketUrl = (): string => {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/angle-scan`;
};

const errorOf = (err: unknown): string => (err instanceof Error ? err.message : String(err));

export default function useAutoAngleCapture(opts: AutoAngleCaptureOptions): AutoAngleCapture {
  const { targetPersonId, targetMediaId, targetIndex, isVideo, autoScan } = opts;
  const [state, dispatch] = useReducer(hudReducer, initialHudState);

  const socketRef = useRef<WebSocket | null>(null);
  const sessionRef = useRef<AngleSession | null>(null);
  const hydratedRef = useRef(false);
  const autoKeyRef = useRef<string | null>(null);
  const thresholdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const thresholdSeq = useRef(0);
  const pendingThresholds = useRef<{ min_iod?: number; blur_frac?: number }>({});
  const targetRef = useRef({ targetPersonId, targetMediaId, targetIndex });
  targetRef.current = { targetPersonId, targetMediaId, targetIndex };

  useEffect(() => { sessionRef.current = state.session; }, [state.session]);

  const closeSocket = useCallback((): void => {
    const ws = socketRef.current;
    socketRef.current = null;
    if (ws && ws.readyState <= 1) {
      try { ws.close(); } catch { /* already closing */ }
    }
  }, []);

  const startScan = useCallback((): void => {
    const { targetPersonId: person, targetMediaId: media, targetIndex: index } = targetRef.current;
    if (!person) {
      dispatch({ type: 'error', error: { code: 'bad_request', message: 'Select a target person first.' } });
      return;
    }
    closeSocket();
    dispatch({ type: 'scan/start' });
    let ws: WebSocket;
    try {
      ws = new WebSocket(socketUrl());
    } catch (err) {
      dispatch({ type: 'scan/lost', message: `Could not open the scan connection: ${errorOf(err)}` });
      return;
    }
    socketRef.current = ws;
    let finished = false;
    ws.onopen = () => {
      ws.send(JSON.stringify({
        op: 'start', target_person_id: person, target_media_id: media ?? undefined, index,
      }));
    };
    ws.onmessage = (msg: MessageEvent) => {
      if (socketRef.current !== ws) return;                  // superseded
      const ev = parseServerEvent(typeof msg.data === 'string' ? msg.data : '');
      if (!ev) return;
      dispatch({ type: 'server', event: ev });
      if (ev.event === 'result' || ev.event === 'error' || ev.event === 'cancelled') {
        finished = true;
        closeSocket();
      }
    };
    ws.onclose = () => {
      if (!finished && socketRef.current === ws) {
        socketRef.current = null;
        dispatch({ type: 'scan/lost', message: 'The scan connection closed before the scan finished '
          + '(the backend may have restarted).' });
      }
    };
  }, [closeSocket]);

  const cancelScan = useCallback((): void => {
    const ws = socketRef.current;
    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ op: 'cancel' }));             // server answers "cancelled"
    } else {
      closeSocket();
      dispatch({ type: 'server', event: { event: 'cancelled' } });
    }
  }, [closeSocket]);

  // Rehydrate once: the server keeps the last session across page reloads.
  useEffect(() => {
    let live = true;
    getJSON('/api/angle-scan/session', { timeout: 8000 })
      .then((res: { session?: AngleSession | null }) => {
        if (live && res?.session) dispatch({ type: 'session/set', session: res.session });
      })
      .catch(() => { /* backend without the route, or down: nothing to restore */ })
      .finally(() => { if (live) hydratedRef.current = true; });
    return () => { live = false; };
  }, []);

  // Auto-trigger on a person / media change.
  useEffect(() => {
    if (!autoScan || !isVideo || !targetPersonId) return undefined;
    const key = `${targetMediaId ?? ''}|${targetPersonId}`;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const tryStart = (): void => {
      if (cancelled) return;
      if (!hydratedRef.current) {                           // wait for the stored session
        timer = setTimeout(tryStart, 150);
        return;
      }
      if (autoKeyRef.current === key) return;
      autoKeyRef.current = key;
      if (sessionMatches(sessionRef.current, targetPersonId, targetMediaId)) return;
      startScan();
    };
    timer = setTimeout(tryStart, AUTO_SETTLE_MS);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [autoScan, isVideo, targetPersonId, targetMediaId, startScan]);

  // Leaving the page stops the scan server-side.
  useEffect(() => () => {
    closeSocket();
    if (thresholdTimer.current) clearTimeout(thresholdTimer.current);
  }, [closeSocket]);

  const flushThresholds = useCallback((): void => {
    const body = pendingThresholds.current;
    pendingThresholds.current = {};
    const seq = ++thresholdSeq.current;
    dispatch({ type: 'request/pending', pending: true });
    postJSON('/api/angle-scan/thresholds', body)
      .then((res: { session?: AngleSession }) => {
        if (seq === thresholdSeq.current && res?.session) dispatch({ type: 'session/set', session: res.session });
      })
      .catch((err: unknown) => {
        if (seq === thresholdSeq.current) dispatch({ type: 'error', error: { code: 'thresholds', message: errorOf(err) } });
      })
      .finally(() => { if (seq === thresholdSeq.current) dispatch({ type: 'request/pending', pending: false }); });
  }, []);

  const queueThreshold = useCallback((patch: { min_iod?: number; blur_frac?: number }): void => {
    pendingThresholds.current = { ...pendingThresholds.current, ...patch };
    if (!sessionRef.current) return;                        // flushed when a session arrives
    if (thresholdTimer.current) clearTimeout(thresholdTimer.current);
    thresholdTimer.current = setTimeout(flushThresholds, THRESHOLD_DEBOUNCE_MS);
  }, [flushThresholds]);

  // A slider moved while no session existed takes effect on the next result.
  useEffect(() => {
    if (state.session && Object.keys(pendingThresholds.current).length) flushThresholds();
  }, [state.session, flushThresholds]);

  const setMinIod = useCallback((px: number): void => {
    dispatch({ type: 'thresholds/set', minIod: px });
    queueThreshold({ min_iod: px });
  }, [queueThreshold]);

  const setBlurFrac = useCallback((frac: number): void => {
    dispatch({ type: 'thresholds/set', blurFrac: frac });
    queueThreshold({ blur_frac: frac });
  }, [queueThreshold]);

  const selectReferenceBin = useCallback((bin: AngleBinName): void => {
    dispatch({ type: 'reference/select', bin });
  }, []);

  const beginOverride = useCallback((bin: AngleBinName): void => dispatch({ type: 'override/begin', bin }), []);
  const cancelOverride = useCallback((): void => dispatch({ type: 'override/end' }), []);

  const assignOverride = useCallback(async (bin: AngleBinName, timelineFrame: number): Promise<OverrideResult | null> => {
    dispatch({ type: 'request/pending', pending: true });
    try {
      const res = await postJSON('/api/angle-scan/override', { bin, frame_idx: toFrameIdx(timelineFrame) });
      dispatch({ type: 'session/set', session: res.session });
      dispatch({ type: 'override/end' });
      dispatch({ type: 'reference/select', bin });
      const entry = (res.session?.bins || []).find((b: BinEntry) => b.bin === bin) || null;
      return { warnings: res.warnings || [], similarity: res.similarity ?? null, entry };
    } catch (err) {
      dispatch({ type: 'error', error: { code: 'override', message: errorOf(err) } });
      return null;
    } finally {
      dispatch({ type: 'request/pending', pending: false });
    }
  }, []);

  const clearOverride = useCallback(async (bin: AngleBinName): Promise<void> => {
    dispatch({ type: 'request/pending', pending: true });
    try {
      const res = await postJSON('/api/angle-scan/override/clear', { bin });
      dispatch({ type: 'session/set', session: res.session });
    } catch (err) {
      dispatch({ type: 'error', error: { code: 'override', message: errorOf(err) } });
    } finally {
      dispatch({ type: 'request/pending', pending: false });
    }
  }, []);

  const applyToPerson = useCallback(async (bins?: AngleBinName[]): Promise<Record<string, unknown> | null> => {
    dispatch({ type: 'request/pending', pending: true });
    try {
      return await postJSON('/api/angle-scan/apply', bins ? { bins } : {});
    } catch (err) {
      dispatch({ type: 'error', error: { code: 'apply', message: errorOf(err) } });
      return null;
    } finally {
      dispatch({ type: 'request/pending', pending: false });
    }
  }, []);

  const dismissMessage = useCallback((): void => {
    dispatch({ type: 'error', error: null });
    dispatch({ type: 'notice', notice: null });
  }, []);

  return {
    ...state,
    startScan,
    cancelScan,
    setMinIod,
    setBlurFrac,
    selectReferenceBin,
    beginOverride,
    cancelOverride,
    assignOverride,
    clearOverride,
    applyToPerson,
    dismissMessage,
    sessionIsCurrent: sessionMatches(state.session, targetPersonId, targetMediaId),
  };
}
