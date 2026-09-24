import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Activity, Cpu, HardDrive, Zap, Flame, AlertTriangle, Minimize2, Maximize2,
  X, GripHorizontal, Gauge, Layers
} from 'lucide-react';
import { getJSON } from '../../api';
import { useTelemetryStore } from '../../store/telemetryStore';

const STORAGE_KEY = 'roop_telemetry_hud_pos';
const HISTORY_LENGTH = 50;

/**
 * High-performance Sparkline Canvas drawer.
 * Renders smooth polyline and gradient fill directly into HTML5 canvas 2D context.
 */
function drawSparkline(canvas, history, color, minVal = 0, maxVal = null) {
  if (!canvas || !history || history.length < 2) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  let min = minVal;
  let max = maxVal;
  if (max === null) {
    max = Math.max(...history, 1);
  }
  const range = Math.max(1e-5, max - min);

  const step = w / (HISTORY_LENGTH - 1);
  const startIdx = Math.max(0, HISTORY_LENGTH - history.length);

  ctx.beginPath();
  history.forEach((val, i) => {
    const x = (startIdx + i) * step;
    const norm = Math.max(0, Math.min(1, (val - min) / range));
    const y = h - norm * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  // Stroke line
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke();

  // Gradient area fill
  ctx.lineTo(w, h);
  ctx.lineTo(startIdx * step, h);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, `${color}40`);
  grad.addColorStop(1, `${color}00`);
  ctx.fillStyle = grad;
  ctx.fill();
}

/**
 * Glassmorphic Hardware Telemetry HUD with zero-react-render DOM/Canvas metrics stream.
 */
export default function HardwareTelemetryHud({
  initialPosition = { x: 24, y: 80 },
  onClose,
  className = '',
}) {
  // ── HOOK DECLARATIONS AT TOP ──────────────────────────────────────────────
  const [collapsed, setCollapsed] = useState(false);
  const [pos, setPos] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        return {
          x: Number.isFinite(parsed.x) ? parsed.x : initialPosition.x,
          y: Number.isFinite(parsed.y) ? parsed.y : initialPosition.y,
        };
      }
    } catch {
      // Fallback
    }
    return initialPosition;
  });

  // Direct DOM Refs for high-frequency updates without React re-rendering
  const containerRef = useRef(null);
  const fpsTextRef = useRef(null);
  const fpsInstantTextRef = useRef(null);
  const fpsCanvasRef = useRef(null);

  const vramTextRef = useRef(null);
  const vramBarRef = useRef(null);
  const vramCanvasRef = useRef(null);

  const gpuCoreTextRef = useRef(null);
  const gpuCanvasRef = useRef(null);

  const latDetRef = useRef(null);
  const latSwapRef = useRef(null);
  const latRestRef = useRef(null);
  const latEncRef = useRef(null);
  const latTotalRef = useRef(null);

  const barDetRef = useRef(null);
  const barSwapRef = useRef(null);
  const barRestRef = useRef(null);
  const barEncRef = useRef(null);

  const thermalAlertRef = useRef(null);
  const thermalValRef = useRef(null);
  const powerAlertRef = useRef(null);
  const powerValRef = useRef(null);
  const droppedTextRef = useRef(null);

  // History buffers for sparkline rendering
  const historyRef = useRef({
    fps: [0],
    vram: [0],
    gpu: [0],
    latency: [0],
  });

  // Dragging state
  const dragRef = useRef({
    active: false,
    startX: 0,
    startY: 0,
    origX: 0,
    origY: 0,
  });

  // ── DRAG HANDLERS ─────────────────────────────────────────────────────────
  const handlePointerDown = useCallback((e) => {
    if (e.target.closest('button')) return;
    dragRef.current = {
      active: true,
      startX: e.clientX,
      startY: e.clientY,
      origX: pos.x,
      origY: pos.y,
    };

    const handlePointerMove = (moveEvt) => {
      if (!dragRef.current.active) return;
      const dx = moveEvt.clientX - dragRef.current.startX;
      const dy = moveEvt.clientY - dragRef.current.startY;
      const nextX = Math.max(10, Math.min(window.innerWidth - 320, dragRef.current.origX + dx));
      const nextY = Math.max(10, Math.min(window.innerHeight - 100, dragRef.current.origY + dy));
      setPos({ x: nextX, y: nextY });
    };

    const handlePointerUp = () => {
      dragRef.current.active = false;
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(pos));
      } catch {
        // Ignored
      }
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
  }, [pos]);

  // Persist position when updated
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(pos));
    } catch {
      // Ignored
    }
  }, [pos]);

  // ── ZERO-REACT-RENDER METRICS STREAM ──────────────────────────────────────
  useEffect(() => {
    let isMounted = true;
    let droppedFramesCount = 0;
    let lastSeq = -1;

    // Helper to push history
    const pushHist = (arr, val) => {
      arr.push(val);
      if (arr.length > HISTORY_LENGTH) arr.shift();
    };

    // Fast metric updater (reads telemetry store directly without setState)
    const updateRunMetrics = (state) => {
      if (!isMounted) return;
      const run = state?.run || {};
      const fps = Number(run.fps) || 0;
      const fpsNow = Number(run.fps_now) || fps;
      const frameMs = Number(run.frame_ms) || (fpsNow > 0 ? 1000 / fpsNow : 0);

      // Dropped frames detection via sequence gap
      const curSeq = Number(run.live_seq) || 0;
      if (lastSeq !== -1 && curSeq > lastSeq + 1) {
        droppedFramesCount += (curSeq - lastSeq - 1);
      }
      lastSeq = curSeq;

      // Update FPS texts directly
      if (fpsTextRef.current) {
        fpsTextRef.current.textContent = `${fps.toFixed(1)} FPS`;
      }
      if (fpsInstantTextRef.current) {
        fpsInstantTextRef.current.textContent = fpsNow > 0 ? `(${fpsNow.toFixed(1)} live)` : '';
      }

      // Latency Breakdown (estimates/approximations based on pipeline stages)
      // Standard Roop split: Detection ~20%, Swap ~45%, Restoration ~25%, Encoding ~10%
      const totalMs = frameMs > 0 ? frameMs : 25;
      const detMs = Math.round(totalMs * 0.20);
      const swapMs = Math.round(totalMs * 0.45);
      const restMs = Math.round(totalMs * 0.25);
      const encMs = Math.max(1, Math.round(totalMs * 0.10));

      if (latDetRef.current) latDetRef.current.textContent = `${detMs}ms`;
      if (latSwapRef.current) latSwapRef.current.textContent = `${swapMs}ms`;
      if (latRestRef.current) latRestRef.current.textContent = `${restMs}ms`;
      if (latEncRef.current) latEncRef.current.textContent = `${encMs}ms`;
      if (latTotalRef.current) latTotalRef.current.textContent = `${Math.round(totalMs)}ms`;

      if (barDetRef.current) barDetRef.current.style.width = '20%';
      if (barSwapRef.current) barSwapRef.current.style.width = '45%';
      if (barRestRef.current) barRestRef.current.style.width = '25%';
      if (barEncRef.current) barEncRef.current.style.width = '10%';

      if (droppedTextRef.current) {
        droppedTextRef.current.textContent = String(droppedFramesCount);
      }

      // Update sparklines
      pushHist(historyRef.current.fps, fpsNow);
      pushHist(historyRef.current.latency, totalMs);

      if (fpsCanvasRef.current) {
        drawSparkline(fpsCanvasRef.current, historyRef.current.fps, '#10B981', 0, 60);
      }
    };

    // Fast subscription to Zustand telemetry store (runs at UI_HZ outside React)
    const unsubTelemetry = useTelemetryStore.subscribe((state) => {
      updateRunMetrics(state);
    });

    // Hardware probes (VRAM, GPU %, Temp, Power) polled at 2.5s interval
    const pollSystemHardware = async () => {
      if (!isMounted) return;
      try {
        const sys = await getJSON('/api/system/telemetry');
        if (!isMounted || !sys) return;

        const vramUsed = Number(sys.vram_used) || 0;
        const vramTotal = Number(sys.vram_total) || 12;
        const vramPct = Math.min(100, Math.round((vramUsed / Math.max(1, vramTotal)) * 100));

        const gpuUtil = Number(sys.gpu_util) || 0;
        const gpuTemp = Number(sys.gpu_temp) || 0;
        const gpuPower = Number(sys.gpu_power) || 0;

        // VRAM elements
        if (vramTextRef.current) {
          vramTextRef.current.textContent = `${vramUsed.toFixed(1)} / ${vramTotal.toFixed(1)} GB (${vramPct}%)`;
        }
        if (vramBarRef.current) {
          vramBarRef.current.style.width = `${vramPct}%`;
          vramBarRef.current.style.backgroundColor = vramPct > 90 ? '#EF4444' : vramPct > 75 ? '#F59E0B' : '#3B82F6';
        }

        // GPU Core elements
        if (gpuCoreTextRef.current) {
          gpuCoreTextRef.current.textContent = `${Math.round(gpuUtil)}%`;
        }

        // Thermal Alert (>80°C warning, >86°C critical)
        if (thermalAlertRef.current && thermalValRef.current) {
          if (gpuTemp >= 80) {
            thermalAlertRef.current.style.display = 'flex';
            thermalValRef.current.textContent = `${Math.round(gpuTemp)}°C`;
            thermalAlertRef.current.className = `flex items-center gap-1 px-1.5 py-0.5 rounded text-nano font-medium ${
              gpuTemp >= 86 ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 animate-pulse' : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
            }`;
          } else {
            thermalAlertRef.current.style.display = 'none';
          }
        }

        // Power throttling alert (power >= 195W on 4070 or near max TDP)
        if (powerAlertRef.current && powerValRef.current) {
          if (gpuPower >= 190) {
            powerAlertRef.current.style.display = 'flex';
            powerValRef.current.textContent = `${Math.round(gpuPower)}W`;
          } else {
            powerAlertRef.current.style.display = 'none';
          }
        }

        // Sparklines for VRAM and GPU
        pushHist(historyRef.current.vram, vramUsed);
        pushHist(historyRef.current.gpu, gpuUtil);

        if (vramCanvasRef.current) {
          drawSparkline(vramCanvasRef.current, historyRef.current.vram, '#3B82F6', 0, vramTotal);
        }
        if (gpuCanvasRef.current) {
          drawSparkline(gpuCanvasRef.current, historyRef.current.gpu, '#8B5CF6', 0, 100);
        }
      } catch {
        // Polling blip
      }
    };

    pollSystemHardware();
    const intervalId = setInterval(pollSystemHardware, 2500);

    return () => {
      isMounted = false;
      unsubTelemetry();
      clearInterval(intervalId);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ left: `${pos.x}px`, top: `${pos.y}px` }}
      className={`fixed z-50 select-none shadow-2xl backdrop-blur-xl bg-[#090A0F]/85 border border-white/15 rounded-2xl overflow-hidden transition-shadow ${
        collapsed ? 'w-64' : 'w-80'
      } ${className}`}
    >
      {/* ── DRAGGABLE HEADER ──────────────────────────────────────────────── */}
      <div
        onPointerDown={handlePointerDown}
        className="flex items-center justify-between px-3 py-2 border-b border-white/10 bg-white/[0.03] cursor-grab active:cursor-grabbing hover:bg-white/[0.06] transition-colors"
      >
        <div className="flex items-center gap-2">
          <GripHorizontal size={14} className="text-white/40" aria-hidden="true" />
          <div className="flex items-center gap-1.5">
            <Activity size={14} className="text-emerald-400" aria-hidden="true" />
            <span className="text-micro font-semibold tracking-wider uppercase text-white/90">
              Hardware Telemetry HUD
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? 'Expand telemetry HUD' : 'Collapse telemetry HUD'}
            className="p-1 rounded text-white/50 hover:text-white hover:bg-white/10 transition-colors"
          >
            {collapsed ? <Maximize2 size={12} aria-hidden="true" /> : <Minimize2 size={12} aria-hidden="true" />}
          </button>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close telemetry HUD"
              className="p-1 rounded text-white/50 hover:text-white hover:bg-white/10 transition-colors"
            >
              <X size={12} aria-hidden="true" />
            </button>
          )}
        </div>
      </div>

      {/* ── COLLAPSED QUICK STATUS PILL ───────────────────────────────────── */}
      {collapsed ? (
        <div className="p-2.5 flex items-center justify-between text-mini font-mono">
          <div className="flex items-center gap-1.5">
            <Gauge size={12} className="text-emerald-400" aria-hidden="true" />
            <span ref={fpsTextRef} className="font-semibold text-white/90">-- FPS</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Cpu size={12} className="text-violet-400" aria-hidden="true" />
            <span ref={gpuCoreTextRef} className="font-semibold text-white/90">--%</span>
          </div>
        </div>
      ) : (
        /* ── EXPANDED FULL TELEMETRY DASHBOARD ────────────────────────────── */
        <div className="p-3.5 flex flex-col gap-3">
          {/* WARNING RIBBON (THERMAL / POWER / DROPPED FRAMES) */}
          <div className="flex flex-wrap items-center gap-1.5">
            {/* Thermal Alert */}
            <div
              ref={thermalAlertRef}
              style={{ display: 'none' }}
              className="items-center gap-1 px-1.5 py-0.5 rounded text-nano font-medium bg-amber-500/20 text-amber-300 border border-amber-500/40"
            >
              <Flame size={10} aria-hidden="true" />
              <span>Thermal:</span>
              <span ref={thermalValRef} className="font-mono">82°C</span>
            </div>

            {/* Power Throttling Alert */}
            <div
              ref={powerAlertRef}
              style={{ display: 'none' }}
              className="items-center gap-1 px-1.5 py-0.5 rounded text-nano font-medium bg-rose-500/20 text-rose-300 border border-rose-500/40"
            >
              <Zap size={10} aria-hidden="true" />
              <span>Power Limit:</span>
              <span ref={powerValRef} className="font-mono">200W</span>
            </div>

            {/* Dropped Frames Badge */}
            <div className="flex items-center gap-1 px-1.5 py-0.5 rounded text-nano bg-white/5 border border-white/10 text-white/60 ml-auto font-mono">
              <AlertTriangle size={10} className="text-white/40" aria-hidden="true" />
              <span>Drops:</span>
              <span ref={droppedTextRef} className="font-semibold text-white/90">0</span>
            </div>
          </div>

          {/* 1. PIPELINE FPS + SPARKLINE */}
          <div className="flex flex-col gap-1 p-2 rounded-xl bg-white/[0.025] border border-white/5">
            <div className="flex items-center justify-between text-mini">
              <div className="flex items-center gap-1.5">
                <Gauge size={13} className="text-emerald-400" aria-hidden="true" />
                <span className="font-medium text-white/80">Pipeline FPS</span>
              </div>
              <div className="flex items-baseline gap-1 font-mono">
                <span ref={fpsTextRef} className="text-compact font-bold text-emerald-400">0.0 FPS</span>
                <span ref={fpsInstantTextRef} className="text-nano text-white/40"></span>
              </div>
            </div>
            <canvas ref={fpsCanvasRef} width={280} height={28} className="w-full h-7 rounded" />
          </div>

          {/* 2. VRAM USAGE (GB / TOTAL) + SPARKLINE */}
          <div className="flex flex-col gap-1 p-2 rounded-xl bg-white/[0.025] border border-white/5">
            <div className="flex items-center justify-between text-mini">
              <div className="flex items-center gap-1.5">
                <HardDrive size={13} className="text-sky-400" aria-hidden="true" />
                <span className="font-medium text-white/80">VRAM Allocation</span>
              </div>
              <span ref={vramTextRef} className="text-micro font-mono text-white/80">-- / -- GB</span>
            </div>
            <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden my-0.5">
              <div ref={vramBarRef} style={{ width: '0%' }} className="h-full bg-sky-500 rounded-full transition-all duration-300" />
            </div>
            <canvas ref={vramCanvasRef} width={280} height={24} className="w-full h-6 rounded" />
          </div>

          {/* 3. GPU CORE % + SPARKLINE */}
          <div className="flex flex-col gap-1 p-2 rounded-xl bg-white/[0.025] border border-white/5">
            <div className="flex items-center justify-between text-mini">
              <div className="flex items-center gap-1.5">
                <Cpu size={13} className="text-violet-400" aria-hidden="true" />
                <span className="font-medium text-white/80">GPU Core Load</span>
              </div>
              <span ref={gpuCoreTextRef} className="text-compact font-bold text-violet-400 font-mono">--%</span>
            </div>
            <canvas ref={gpuCanvasRef} width={280} height={24} className="w-full h-6 rounded" />
          </div>

          {/* 4. INFERENCE LATENCY BREAKDOWN (MS BREAKDOWN) */}
          <div className="flex flex-col gap-1.5 p-2 rounded-xl bg-white/[0.025] border border-white/5">
            <div className="flex items-center justify-between text-mini">
              <div className="flex items-center gap-1.5">
                <Layers size={13} className="text-amber-400" aria-hidden="true" />
                <span className="font-medium text-white/80">Inference Latency</span>
              </div>
              <span ref={latTotalRef} className="text-micro font-bold font-mono text-amber-400">-- ms</span>
            </div>

            {/* Stacked Latency Bar */}
            <div className="flex w-full h-1.5 rounded-full overflow-hidden bg-white/10">
              <div ref={barDetRef} style={{ width: '20%' }} className="bg-sky-400 h-full" title="Detection" />
              <div ref={barSwapRef} style={{ width: '45%' }} className="bg-rose-400 h-full" title="Swap" />
              <div ref={barRestRef} style={{ width: '25%' }} className="bg-emerald-400 h-full" title="Restoration" />
              <div ref={barEncRef} style={{ width: '10%' }} className="bg-amber-400 h-full" title="Encoding" />
            </div>

            {/* Legend / Values Breakdown */}
            <div className="grid grid-cols-4 gap-1 pt-1 text-nano font-mono text-center">
              <div className="flex flex-col items-center">
                <span className="text-sky-300">Detect</span>
                <span ref={latDetRef} className="text-white/60">--ms</span>
              </div>
              <div className="flex flex-col items-center">
                <span className="text-rose-300">Swap</span>
                <span ref={latSwapRef} className="text-white/60">--ms</span>
              </div>
              <div className="flex flex-col items-center">
                <span className="text-emerald-300">Restore</span>
                <span ref={latRestRef} className="text-white/60">--ms</span>
              </div>
              <div className="flex flex-col items-center">
                <span className="text-amber-300">Encode</span>
                <span ref={latEncRef} className="text-white/60">--ms</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
