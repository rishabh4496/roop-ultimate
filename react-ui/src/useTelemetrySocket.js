// Live telemetry over WebSocket, with polling as the fallback.
//
// WHY
// The client used to learn about a running render only by polling
// `GET /api/progress` once a second. Measured on this repo
// (app/tests/probe_progress_cost.py) that response is 17.4 KB mid-render --
// it carries the 250-line rolling log, the parts snapshot and the whole
// nested `runtime` block -- against a 108-byte telemetry frame. That is 161x
// the bytes to deliver the handful of numbers that actually changed, ~60 MB
// over a one-hour render, and it still only samples an fps counter once a
// second.
//
// DESIGN RULES
//  1. The socket is an ACCELERATOR, never a requirement. `/api/progress`
//     remains the source of truth, and whenever the socket is not connected
//     the caller keeps polling exactly as before. A proxy that blocks
//     WebSocket upgrades therefore degrades to today's behaviour instead of
//     breaking the UI.
//  2. Reconnect with exponential backoff and jitter. A backend restart mid
//     render must not leave a dead socket and a frozen progress bar, and a
//     tight retry loop against a down server is its own denial of service.
//  3. Never reconnect while the tab is hidden. Browsers throttle background
//     timers hard; retrying into that just burns the backoff budget so the
//     socket is at its slowest retry exactly when the tab comes back. We
//     reconnect immediately on becoming visible instead.
import { useCallback, useEffect, useRef, useState } from 'react';

// Backoff schedule. Starts fast because the overwhelmingly common cause is a
// backend that is still coming up (the launcher starts the UI and the API
// together), and caps so a genuinely dead server is retried at a sane rate.
const BACKOFF_MIN_MS = 500;
const BACKOFF_MAX_MS = 15000;

// If nothing arrives for this long the connection is treated as dead even
// though the socket still claims to be open. The server heartbeats every 15 s,
// so silence well past that means a half-open TCP connection -- the failure
// that leaves a progress bar frozen with no error anywhere.
const SILENCE_TIMEOUT_MS = 45000;

const socketUrl = () => {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/telemetry`;
};

/**
 * Subscribe to live run telemetry.
 *
 * @param {(frame: object) => void} onFrame called with each telemetry frame.
 * @param {boolean} enabled set false to tear the socket down entirely.
 * @returns {{connected: boolean, lastFrameAt: number}} transport state, so the
 *          caller can keep polling while `connected` is false.
 */
export default function useTelemetrySocket(onFrame, enabled = true) {
  const [connected, setConnected] = useState(false);
  const [lastFrameAt, setLastFrameAt] = useState(0);

  const socketRef = useRef(null);
  const retryRef = useRef(null);
  const attemptRef = useRef(0);
  const silenceRef = useRef(null);
  const closedByUsRef = useRef(false);

  // Keep the callback in a ref so a caller passing an inline arrow does not
  // tear down and rebuild the socket on every render.
  const onFrameRef = useRef(onFrame);
  useEffect(() => { onFrameRef.current = onFrame; }, [onFrame]);

  const clearTimers = useCallback(() => {
    if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null; }
    if (silenceRef.current) { clearTimeout(silenceRef.current); silenceRef.current = null; }
  }, []);

  const connect = useCallback(() => {
    if (!enabled) return;
    if (document.visibilityState === 'hidden') return;
    // readyState 0/1 = CONNECTING/OPEN: already have a usable socket.
    if (socketRef.current && socketRef.current.readyState <= 1) return;

    let ws;
    try {
      ws = new WebSocket(socketUrl());
    } catch {
      // Constructor throws on a malformed URL or a blocked scheme; treat it as
      // a failed attempt so the backoff path handles it uniformly.
      scheduleRetry();
      return;
    }
    socketRef.current = ws;
    closedByUsRef.current = false;

    const armSilenceTimer = () => {
      if (silenceRef.current) clearTimeout(silenceRef.current);
      silenceRef.current = setTimeout(() => {
        // Force a reconnect: close() triggers onclose, which schedules the
        // retry. A half-open socket never fires onclose on its own.
        try { ws.close(); } catch { /* already gone */ }
      }, SILENCE_TIMEOUT_MS);
    };

    ws.onopen = () => {
      attemptRef.current = 0;
      setConnected(true);
      armSilenceTimer();
    };

    ws.onmessage = (event) => {
      armSilenceTimer();
      let frame;
      try {
        frame = JSON.parse(event.data);
      } catch {
        return; // ignore anything that is not a frame
      }
      // `pong` is liveness only and carries no state; arming the silence timer
      // above is its entire purpose.
      if (frame.event === 'pong') return;
      setLastFrameAt(Date.now());
      try {
        onFrameRef.current?.(frame);
      } catch {
        // A rendering fault in the consumer must not kill the transport.
      }
    };

    ws.onerror = () => {
      // Always followed by onclose, which owns the retry.
    };

    ws.onclose = () => {
      setConnected(false);
      if (silenceRef.current) { clearTimeout(silenceRef.current); silenceRef.current = null; }
      if (socketRef.current === ws) socketRef.current = null;
      if (!closedByUsRef.current) scheduleRetry();
    };

    function scheduleRetry() {
      if (!enabled || closedByUsRef.current) return;
      if (retryRef.current) return;
      const attempt = attemptRef.current++;
      const base = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS * 2 ** attempt);
      // Jitter so several webviews (Pinokio reloads them on tab switches) do
      // not resynchronise into a thundering herd against a restarting backend.
      const delay = base * (0.5 + Math.random() * 0.5);
      retryRef.current = setTimeout(() => {
        retryRef.current = null;
        connect();
      }, delay);
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      closedByUsRef.current = true;
      clearTimers();
      if (socketRef.current) {
        try { socketRef.current.close(); } catch { /* ignore */ }
        socketRef.current = null;
      }
      setConnected(false);
      return undefined;
    }

    connect();

    // Reconnect the instant the tab is looked at again. Pinokio reloads or
    // hides this webview on every tab switch, and a socket that died while
    // hidden would otherwise wait out a long backoff in full view of the user.
    const onVisible = () => {
      if (document.visibilityState === 'visible') {
        if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null; }
        attemptRef.current = 0;
        connect();
      }
    };
    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('focus', onVisible);
    window.addEventListener('online', onVisible);

    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('focus', onVisible);
      window.removeEventListener('online', onVisible);
      closedByUsRef.current = true;
      clearTimers();
      if (socketRef.current) {
        try { socketRef.current.close(); } catch { /* ignore */ }
        socketRef.current = null;
      }
    };
  }, [enabled, connect, clearTimers]);

  return { connected, lastFrameAt };
}
