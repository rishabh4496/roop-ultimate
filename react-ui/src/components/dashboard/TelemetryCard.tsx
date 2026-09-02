import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Activity,
  Cpu,
  Gauge,
  Zap,
  Clock,
  HardDrive,
  Layers,
  Wifi,
  WifiOff,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  ArrowRight,
} from 'lucide-react';

import {
  TelemetryPacket,
  PIPELINE_STAGES,
  PipelineStage,
  TelemetryRingBuffer,
  generateSparklinePaths,
  formatEta,
} from './telemetryBuffer';

export type { TelemetryPacket, PipelineStage };

export interface TelemetryCardProps {
  /** WebSocket endpoint for real-time telemetry */
  wsUrl?: string;
  /** Fixed capacity of the circular history buffer (default: 60 points = 30s @ 500ms) */
  bufferCapacity?: number;
  /** Force simulation mode for local testing without backend */
  simulate?: boolean;
  /** Automatically fallback to simulation if WebSocket fails to connect */
  fallbackToSimulation?: boolean;
  /** Custom CSS class names */
  className?: string;
  /** Optional callback fired when a new packet arrives */
  onPacket?: (packet: TelemetryPacket) => void;
}

export function TelemetryCard({
  wsUrl,
  bufferCapacity = 60,
  simulate = false,
  fallbackToSimulation = true,
  className = '',
  onPacket,
}: TelemetryCardProps) {
  // Store incoming telemetry in a fixed-size circular buffer using React ref to prevent global re-renders
  const ringBufferRef = useRef<TelemetryRingBuffer>(new TelemetryRingBuffer(bufferCapacity));

  // Minimum required state: current packet & connection status
  const [packet, setPacket] = useState<TelemetryPacket>({
    fps: 0,
    vram_used: 0,
    vram_total: 12.0,
    p_core_util: 0,
    eta_seconds: 0,
    stage: 'Decoding',
  });

  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'simulated' | 'disconnected'>('connecting');
  const [, setTick] = useState<number>(0);

  // Ingest packet into circular buffer and update component state
  const handleIncomingPacket = useRef((p: TelemetryPacket) => {
    ringBufferRef.current.push(p);
    setPacket(p);
    setTick((t) => (t + 1) % 10000);
    onPacket?.(p);
  });

  useEffect(() => {
    handleIncomingPacket.current = (p: TelemetryPacket) => {
      ringBufferRef.current.push(p);
      setPacket(p);
      setTick((t) => (t + 1) % 10000);
      onPacket?.(p);
    };
  }, [onPacket]);

  // ── WebSocket Connection & Fallback Simulation ─────────────────────────
  useEffect(() => {
    let ws: WebSocket | null = null;
    let simInterval: number | null = null;
    let reconnectTimeout: number | null = null;
    let isDisposed = false;

    // Simulation Generator for zero-dependency local testing
    const startSimulation = () => {
      setConnectionStatus('simulated');
      let currentStageIdx = 0;
      let mockEta = 184;

      simInterval = window.setInterval(() => {
        if (isDisposed) return;
        mockEta = Math.max(0, mockEta - 0.5);

        // Realistic telemetry packet with jitter and stage advancement
        if (Math.random() < 0.08) {
          currentStageIdx = (currentStageIdx + 1) % PIPELINE_STAGES.length;
        }

        const simPacket: TelemetryPacket = {
          fps: parseFloat((28.5 + (Math.random() * 3.8 - 1.9)).toFixed(1)),
          vram_used: parseFloat((9.15 + Math.sin(Date.now() / 4000) * 0.45).toFixed(2)),
          vram_total: 12.0,
          p_core_util: Math.round(58 + Math.random() * 18),
          eta_seconds: Math.round(mockEta),
          stage: PIPELINE_STAGES[currentStageIdx],
          timestamp: Date.now(),
        };

        handleIncomingPacket.current(simPacket);
      }, 500);
    };

    if (simulate) {
      startSimulation();
      return () => {
        isDisposed = true;
        if (simInterval) clearInterval(simInterval);
      };
    }

    // Determine target WebSocket URL
    const targetWsUrl =
      wsUrl ||
      (() => {
        if (typeof window !== 'undefined' && window.location) {
          const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
          return `${proto}//${window.location.host}/api/telemetry`;
        }
        return 'ws://localhost:8000/api/telemetry';
      })();

    const connectWebSocket = () => {
      if (isDisposed) return;
      setConnectionStatus('connecting');

      try {
        ws = new WebSocket(targetWsUrl);

        ws.onopen = () => {
          if (isDisposed) return;
          setConnectionStatus('connected');
          if (simInterval) {
            clearInterval(simInterval);
            simInterval = null;
          }
        };

        ws.onmessage = (event) => {
          if (isDisposed) return;
          try {
            const data: TelemetryPacket = JSON.parse(event.data);
            handleIncomingPacket.current(data);
          } catch {
            // Drop malformed frame without throwing
          }
        };

        ws.onerror = () => {
          if (isDisposed) return;
          if (fallbackToSimulation && !simInterval) {
            startSimulation();
          }
        };

        ws.onclose = () => {
          if (isDisposed) return;
          setConnectionStatus((prev) => (prev === 'simulated' ? 'simulated' : 'disconnected'));
          if (fallbackToSimulation && !simInterval) {
            startSimulation();
          } else if (!simulate) {
            reconnectTimeout = window.setTimeout(connectWebSocket, 3000);
          }
        };
      } catch {
        if (fallbackToSimulation && !simInterval) {
          startSimulation();
        }
      }
    };

    connectWebSocket();

    return () => {
      isDisposed = true;
      if (ws) ws.close();
      if (simInterval) clearInterval(simInterval);
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [fallbackToSimulation, simulate, wsUrl]);

  // ── Micro-Sparklines & Metric Computations ──────────────────────────────
  const { fpsHistory, vramHistory, smoothedFps } = useMemo(() => {
    const fpsHist = ringBufferRef.current.getFpsHistory();
    const vramHist = ringBufferRef.current.getVramHistory();

    // Smoothed FPS via trailing moving average of the last 10 points
    let smooth = packet.fps;
    if (fpsHist.length > 0) {
      const windowSize = Math.min(fpsHist.length, 10);
      const recent = fpsHist.slice(fpsHist.length - windowSize);
      const sum = recent.reduce((a, b) => a + b, 0);
      smooth = parseFloat((sum / windowSize).toFixed(1));
    }

    return { fpsHistory: fpsHist, vramHistory: vramHist, smoothedFps: smooth };
  }, [packet.fps]);

  // Generate lightweight SVG paths
  const fpsSparkline = useMemo(
    () => generateSparklinePaths(fpsHistory, 120, 36, 3),
    [fpsHistory]
  );
  const vramSparkline = useMemo(
    () => generateSparklinePaths(vramHistory, 120, 36, 3),
    [vramHistory]
  );

  // VRAM Normalized Math (detects MB vs GB automatically)
  const isVramInMb = packet.vram_total > 64;
  const vramTotalGb = isVramInMb ? packet.vram_total / 1024 : packet.vram_total;
  const vramUsedGb = isVramInMb ? packet.vram_used / 1024 : packet.vram_used;
  const vramPercent = vramTotalGb > 0 ? Math.min(100, Math.max(0, (vramUsedGb / vramTotalGb) * 100)) : 0;

  // VRAM Alert Thresholds: >85% turns amber, >95% turns red
  const vramTier = vramPercent >= 95 ? 'critical' : vramPercent >= 85 ? 'warning' : 'optimal';

  // Active Pipeline Stage index
  const activeStageIndex = PIPELINE_STAGES.indexOf(packet.stage as PipelineStage);

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-white/10 bg-neutral-950/80 p-5 text-white shadow-2xl backdrop-blur-xl transition-all ${className}`}
    >
      {/* ── Top Bar: Telemetry Header & WebSocket Link Status ───────────── */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-3.5">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            <Activity className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-xs font-bold uppercase tracking-wider text-white">Real-Time Telemetry</h2>
            <p className="text-[10px] text-white/40">500ms Edge Stream &bull; 30s Circular Buffer (60 pts)</p>
          </div>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] font-medium">
          {connectionStatus === 'connected' ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
              </span>
              <Wifi className="h-3 w-3 text-emerald-400" />
              <span className="font-mono text-[10px] text-emerald-300">LIVE (WS)</span>
            </>
          ) : connectionStatus === 'simulated' ? (
            <>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
              <Zap className="h-3 w-3 text-cyan-400" />
              <span className="font-mono text-[10px] text-cyan-300">SIMULATED</span>
            </>
          ) : connectionStatus === 'connecting' ? (
            <>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-400 animate-ping" />
              <Wifi className="h-3 w-3 text-amber-400" />
              <span className="font-mono text-[10px] text-amber-300">CONNECTING</span>
            </>
          ) : (
            <>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-rose-500" />
              <WifiOff className="h-3 w-3 text-rose-400" />
              <span className="font-mono text-[10px] text-rose-300">OFFLINE</span>
            </>
          )}
        </div>
      </div>

      {/* ── Status Grid Layout (4 Metrics) ───────────────────────────────── */}
      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* ── Metric 1: Smoothed Throughput & ETA ────────────────────────── */}
        <div className="flex flex-col justify-between rounded-xl border border-white/10 bg-neutral-900/50 p-3.5 transition-all hover:border-white/20">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-white/50 flex items-center gap-1.5">
              <Gauge className="h-3.5 w-3.5 text-emerald-400" />
              Throughput
            </span>
            <div className="flex items-center gap-1 text-[11px] font-mono text-emerald-300 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
              <Clock className="h-3 w-3" />
              <span>ETA {formatEta(packet.eta_seconds)}</span>
            </div>
          </div>

          <div className="my-2 flex items-baseline justify-between">
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono text-2xl font-black tracking-tight text-white">
                {smoothedFps.toFixed(1)}
              </span>
              <span className="text-xs font-bold text-emerald-400">FPS</span>
            </div>

            {/* SVG Micro-Sparkline for FPS */}
            <div className="relative h-9 w-[120px] overflow-hidden">
              {fpsSparkline.linePath ? (
                <svg viewBox="0 0 120 36" className="h-full w-full overflow-visible">
                  <defs>
                    <linearGradient id="fpsGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
                    </linearGradient>
                  </defs>
                  <path d={fpsSparkline.areaPath} fill="url(#fpsGradient)" />
                  <path
                    d={fpsSparkline.linePath}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  {/* Latest value glowing pulse dot */}
                  <circle
                    cx={fpsSparkline.latestX}
                    cy={fpsSparkline.latestY}
                    r="2.5"
                    fill="#10b981"
                    className="animate-pulse"
                  />
                </svg>
              ) : (
                <div className="h-full w-full flex items-center justify-center text-[10px] text-white/30 font-mono">
                  Ingesting...
                </div>
              )}
            </div>
          </div>

          <div className="text-[10px] text-white/40 flex justify-between items-center">
            <span>Instant: {packet.fps.toFixed(1)} FPS</span>
            <span>Last 30s</span>
          </div>
        </div>

        {/* ── Metric 2: VRAM Gauge ────────────────────────────────────────── */}
        <div
          className={`flex flex-col justify-between rounded-xl border p-3.5 transition-all ${
            vramTier === 'critical'
              ? 'border-rose-500/60 bg-rose-950/20 shadow-lg shadow-rose-950/20'
              : vramTier === 'warning'
              ? 'border-amber-500/50 bg-amber-950/20'
              : 'border-white/10 bg-neutral-900/50 hover:border-white/20'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-white/50 flex items-center gap-1.5">
              <HardDrive className="h-3.5 w-3.5 text-cyan-400" />
              VRAM Capacity
            </span>
            <div
              className={`flex items-center gap-1 text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                vramTier === 'critical'
                  ? 'border-rose-500/40 bg-rose-500/20 text-rose-300 animate-pulse'
                  : vramTier === 'warning'
                  ? 'border-amber-500/40 bg-amber-500/20 text-amber-300'
                  : 'border-cyan-500/30 bg-cyan-500/10 text-cyan-300'
              }`}
            >
              {vramTier === 'critical' ? (
                <AlertOctagon className="h-3 w-3" />
              ) : vramTier === 'warning' ? (
                <AlertTriangle className="h-3 w-3" />
              ) : null}
              <span>{vramPercent.toFixed(1)}%</span>
            </div>
          </div>

          <div className="my-2 flex items-baseline justify-between">
            <div className="flex items-baseline gap-1">
              <span className="font-mono text-2xl font-black tracking-tight text-white">
                {vramUsedGb.toFixed(2)}
              </span>
              <span className="text-xs text-white/40 font-mono">/ {vramTotalGb.toFixed(1)} GB</span>
            </div>

            {/* SVG Micro-Sparkline for VRAM */}
            <div className="relative h-9 w-[120px] overflow-hidden">
              {vramSparkline.linePath ? (
                <svg viewBox="0 0 120 36" className="h-full w-full overflow-visible">
                  <defs>
                    <linearGradient id="vramGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="0%"
                        stopColor={vramTier === 'critical' ? '#f43f5e' : vramTier === 'warning' ? '#f59e0b' : '#06b6d4'}
                        stopOpacity="0.4"
                      />
                      <stop
                        offset="100%"
                        stopColor={vramTier === 'critical' ? '#f43f5e' : vramTier === 'warning' ? '#f59e0b' : '#06b6d4'}
                        stopOpacity="0.0"
                      />
                    </linearGradient>
                  </defs>
                  <path d={vramSparkline.areaPath} fill="url(#vramGradient)" />
                  <path
                    d={vramSparkline.linePath}
                    fill="none"
                    stroke={vramTier === 'critical' ? '#f43f5e' : vramTier === 'warning' ? '#f59e0b' : '#06b6d4'}
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <circle
                    cx={vramSparkline.latestX}
                    cy={vramSparkline.latestY}
                    r="2.5"
                    fill={vramTier === 'critical' ? '#f43f5e' : vramTier === 'warning' ? '#f59e0b' : '#06b6d4'}
                    className="animate-pulse"
                  />
                </svg>
              ) : (
                <div className="h-full w-full flex items-center justify-center text-[10px] text-white/30 font-mono">
                  Ingesting...
                </div>
              )}
            </div>
          </div>

          {/* VRAM Progress Bar with dynamic alerts */}
          <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className={`h-full transition-all duration-300 ${
                vramTier === 'critical'
                  ? 'bg-gradient-to-r from-rose-500 to-red-600'
                  : vramTier === 'warning'
                  ? 'bg-gradient-to-r from-amber-500 to-orange-500'
                  : 'bg-gradient-to-r from-cyan-500 to-teal-400'
              }`}
              style={{ width: `${vramPercent}%` }}
            />
          </div>
        </div>

        {/* ── Metric 3: Pipeline Stage Stepper Pill ────────────────────────── */}
        <div className="flex flex-col justify-between rounded-xl border border-white/10 bg-neutral-900/50 p-3.5 transition-all hover:border-white/20">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-white/50 flex items-center gap-1.5">
              <Layers className="h-3.5 w-3.5 text-violet-400" />
              Pipeline Execution
            </span>
            <span className="rounded bg-violet-500/15 border border-violet-500/30 px-2 py-0.5 font-mono text-[10px] font-bold text-violet-300">
              {packet.stage || 'Idle'}
            </span>
          </div>

          {/* Active Step Indicator Pill Strip */}
          <div className="my-2 flex items-center gap-1">
            {PIPELINE_STAGES.map((st, idx) => {
              const isPast = activeStageIndex > idx;
              const isCurrent = activeStageIndex === idx;

              return (
                <React.Fragment key={st}>
                  <div
                    className={`flex flex-1 flex-col items-center justify-center rounded-lg py-1 px-0.5 text-center transition-all ${
                      isCurrent
                        ? 'border border-violet-400 bg-violet-500/25 text-white shadow-md shadow-violet-950/40'
                        : isPast
                        ? 'border border-white/10 bg-white/10 text-emerald-300'
                        : 'border border-white/5 bg-white/2 text-white/30'
                    }`}
                    title={`Stage: ${st} (${isCurrent ? 'Active' : isPast ? 'Done' : 'Upcoming'})`}
                  >
                    <div className="flex items-center gap-0.5">
                      {isPast && <CheckCircle2 className="h-2.5 w-2.5 shrink-0 text-emerald-400" />}
                      <span className="truncate text-[9px] font-mono font-semibold">
                        {st.slice(0, 4)}
                      </span>
                    </div>
                  </div>
                  {idx < PIPELINE_STAGES.length - 1 && (
                    <ArrowRight className="h-2.5 w-2.5 shrink-0 text-white/20" />
                  )}
                </React.Fragment>
              );
            })}
          </div>

          <div className="text-[10px] text-white/40 flex justify-between items-center">
            <span>Step {activeStageIndex >= 0 ? activeStageIndex + 1 : 1} of 5</span>
            <span className="capitalize">{packet.stage} Engine</span>
          </div>
        </div>

        {/* ── Metric 4: P-Core Load ───────────────────────────────────────── */}
        <div className="flex flex-col justify-between rounded-xl border border-white/10 bg-neutral-900/50 p-3.5 transition-all hover:border-white/20">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-white/50 flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-amber-400" />
              P-Core Saturation
            </span>
            <span
              className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-bold border ${
                packet.p_core_util >= 90
                  ? 'border-rose-500/40 bg-rose-500/20 text-rose-300'
                  : packet.p_core_util >= 70
                  ? 'border-amber-500/40 bg-amber-500/20 text-amber-300'
                  : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
              }`}
            >
              {packet.p_core_util}%
            </span>
          </div>

          <div className="my-2 flex items-baseline justify-between">
            <div className="flex items-baseline gap-1">
              <span className="font-mono text-2xl font-black tracking-tight text-white">
                {packet.p_core_util}
              </span>
              <span className="text-xs text-white/40 font-mono">%</span>
            </div>

            {/* Segmented Core Utilization Bars */}
            <div className="flex items-end gap-1 h-8">
              {[20, 40, 60, 80, 100].map((threshold) => {
                const isActive = packet.p_core_util >= threshold;
                return (
                  <div
                    key={threshold}
                    className={`w-2 rounded-t transition-all duration-300 ${
                      isActive
                        ? threshold >= 80
                          ? 'bg-rose-500'
                          : threshold >= 60
                          ? 'bg-amber-400'
                          : 'bg-emerald-400'
                        : 'bg-white/10'
                    }`}
                    style={{ height: `${threshold * 0.3}px` }}
                  />
                );
              })}
            </div>
          </div>

          {/* Load Progress Bar */}
          <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className={`h-full transition-all duration-300 ${
                packet.p_core_util >= 90
                  ? 'bg-gradient-to-r from-amber-500 to-rose-600'
                  : packet.p_core_util >= 70
                  ? 'bg-gradient-to-r from-emerald-500 to-amber-500'
                  : 'bg-gradient-to-r from-teal-500 to-emerald-400'
              }`}
              style={{ width: `${Math.min(100, Math.max(0, packet.p_core_util))}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default TelemetryCard;
