import { useEffect, useRef, useState } from "react";
import { telemetryUrl } from "../lib/api";
import type { Telemetry } from "../lib/types";

export interface TelemetrySample {
  time: number;
  vramUsedMb: number | null;
  temperatureC: number | null;
  utilizationPct: number | null;
  fps: number | null;
  latencyMs: number | null;
}

export interface TelemetryState {
  connected: boolean;
  latest: Telemetry | null;
  history: TelemetrySample[];
}

/** 60 s of history at the server's 4 Hz. */
export const HISTORY_LENGTH = 240;

export function toSample(t: Telemetry): TelemetrySample {
  const state = t.render?.state ?? t.job?.state;
  const rendering = state === "rendering";
  const fps = t.render?.fps ?? t.job?.fps ?? null;
  return {
    time: t.time,
    vramUsedMb: t.gpu?.vram_used_mb ?? null,
    temperatureC: t.gpu?.temperature_c ?? null,
    utilizationPct: t.gpu?.utilization_pct ?? null,
    fps: rendering ? fps : null,
    latencyMs: rendering && t.job ? t.job.latency_ms : null,
  };
}

/**
 * Subscribe to /ws/telemetry. Reconnects with exponential back-off (1 s -> 10 s)
 * and keeps the last HISTORY_LENGTH samples for the HUD graphs.
 */
export function useTelemetry(url: string = telemetryUrl()): TelemetryState {
  const [state, setState] = useState<TelemetryState>({ connected: false, latest: null, history: [] });
  const retry = useRef(1000);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let closed = false;

    const connect = () => {
      socket = new WebSocket(url);
      socket.onopen = () => {
        retry.current = 1000;
        setState((s) => ({ ...s, connected: true }));
      };
      socket.onmessage = (event: MessageEvent<string>) => {
        let message: Telemetry;
        try {
          message = JSON.parse(event.data) as Telemetry;
        } catch {
          return;
        }
        if (message.type !== "telemetry") return;
        setState((s) => ({
          connected: true,
          latest: message,
          history: [...s.history, toSample(message)].slice(-HISTORY_LENGTH),
        }));
      };
      socket.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (!closed) {
          timer = setTimeout(connect, retry.current);
          retry.current = Math.min(retry.current * 2, 10_000);
        }
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
      socket?.close();
    };
  }, [url]);

  return state;
}
