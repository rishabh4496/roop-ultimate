import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Activity,
  Cpu,
  Gauge,
  Zap,
  Sliders,
  Sparkles,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Eye,
  Smile,
  Layers,
  Flame,
  HardDrive,
  Workflow,
  Radio,
} from 'lucide-react';

export type PrecisionMode = 'hybrid_fp16_fp32' | 'strict_fp32';
export type ExecutionProvider = 'tensorrt' | 'cuda' | 'cpu';
export type EngineStatus = 'idle' | 'compiling' | 'ready' | 'failed';

export interface UltraMaxSettings {
  skin_texture_strength: number; // 0.0 - 2.0
  eye_clarity: number;           // 0.0 - 2.0
  teeth_detail_weight: number;   // 0.0 - 1.0
  lip_detail_weight: number;     // 0.0 - 1.0
  blend_ratio: number;           // 0.0 - 1.0
  batch_size: number;            // 1 - 8
  precision_mode: PrecisionMode;
  provider: ExecutionProvider;
}

export interface EngineCompilationStatus {
  compiling: boolean;
  progress: number; // 0.0 - 1.0
  stage: string;    // 'idle' | 'optimizing_graph' | 'building_engine' | 'canary_verification' | 'ready' | 'failed'
  status: EngineStatus;
  message?: string;
  engine_path?: string;
  compilation_time_s?: number;
}

export interface PipelineQueueDepths {
  decode: number;
  preprocess: number;
  detect_track: number;
  swap: number;
  enhance: number;
  composite: number;
  encode: number;
  [key: string]: number;
}

export interface TelemetryData {
  event?: string;
  timestamp: number;
  instantaneous_fps: number;
  average_fps: number;
  fps_now?: number;
  fps?: number;
  cpu_load_percentage: number;
  system_ram_mb: number;
  system_ram_total_mb: number;
  system_ram_allocation?: number;
  gpu_vram_used_mb: number;
  gpu_vram_total_mb: number;
  gpu_vram_usage?: number;
  active_execution_provider: string;
  active_provider?: string;
  pipeline_queue_depths: PipelineQueueDepths;
  compilation: EngineCompilationStatus;
  settings?: UltraMaxSettings;
}

export interface EnhancerSettingsProps {
  initialSettings?: Partial<UltraMaxSettings>;
  onSettingsChange?: (settings: UltraMaxSettings) => void;
  wsUrl?: string;
  className?: string;
}

const DEFAULT_SETTINGS: UltraMaxSettings = {
  skin_texture_strength: 1.0,
  eye_clarity: 1.0,
  teeth_detail_weight: 0.5,
  lip_detail_weight: 0.5,
  blend_ratio: 0.75,
  batch_size: 1,
  precision_mode: 'hybrid_fp16_fp32',
  provider: 'tensorrt',
};

const DEFAULT_COMPILATION: EngineCompilationStatus = {
  compiling: false,
  progress: 1.0,
  stage: 'ready',
  status: 'ready',
  message: 'TensorRT engine ready',
};

const DEFAULT_TELEMETRY: TelemetryData = {
  timestamp: Date.now() / 1000,
  instantaneous_fps: 0,
  average_fps: 0,
  cpu_load_percentage: 0,
  system_ram_mb: 0,
  system_ram_total_mb: 16384,
  gpu_vram_used_mb: 0,
  gpu_vram_total_mb: 12288,
  active_execution_provider: 'tensorrt',
  pipeline_queue_depths: {
    decode: 0,
    preprocess: 0,
    detect_track: 0,
    swap: 0,
    enhance: 0,
    composite: 0,
    encode: 0,
  },
  compilation: DEFAULT_COMPILATION,
};

// ── Debounced Slider Subcomponent ──────────────────────────────────────────

interface DebouncedSliderProps {
  label: string;
  info: string;
  value: number;
  defaultValue: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  icon?: React.ReactNode;
  onChange: (value: number) => void;
}

function DebouncedSlider({
  label,
  info,
  value,
  defaultValue,
  min,
  max,
  step = 0.01,
  unit = '',
  icon,
  onChange,
}: DebouncedSliderProps) {
  const [localValue, setLocalValue] = useState<number>(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const commit = useCallback((val: number) => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    onChangeRef.current(val);
  }, []);

  const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setLocalValue(val);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      commit(val);
    }, 100);
  };

  const handleBlur = () => {
    if (timerRef.current) commit(localValue);
  };

  const handleReset = () => {
    setLocalValue(defaultValue);
    commit(defaultValue);
  };

  const isModified = Math.abs(localValue - defaultValue) > 1e-4;

  return (
    <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-slate-900/40 border border-white/5 hover:border-white/10 transition-colors">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 font-medium text-slate-200">
          {icon}
          <span>{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-cyan-400 font-semibold">
            {localValue.toFixed(step >= 1 ? 0 : 2)}
            {unit}
          </span>
          {isModified && (
            <button
              type="button"
              onClick={handleReset}
              className="text-micro text-slate-400 hover:text-cyan-400 underline transition-colors cursor-pointer"
              title="Reset to default"
            >
              reset
            </button>
          )}
        </div>
      </div>

      <p className="text-mini text-slate-400 leading-tight">{info}</p>

      <div className="flex items-center gap-2 pt-1">
        <span className="text-micro font-mono text-slate-500 w-6">{min}</span>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={localValue}
          onChange={handleInput}
          onBlur={handleBlur}
          onPointerUp={handleBlur}
          className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
        />
        <span className="text-micro font-mono text-slate-500 w-6 text-right">{max}</span>
      </div>
    </div>
  );
}

// ── Main EnhancerSettings Component ────────────────────────────────────────

export default function EnhancerSettings({
  initialSettings,
  onSettingsChange,
  wsUrl,
  className = '',
}: EnhancerSettingsProps) {
  const [settings, setSettings] = useState<UltraMaxSettings>(() => ({
    ...DEFAULT_SETTINGS,
    ...initialSettings,
  }));

  const [telemetry, setTelemetry] = useState<TelemetryData>(DEFAULT_TELEMETRY);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [isCompiling, setIsCompiling] = useState<boolean>(false);
  const [compileError, setCompileError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Derive target WebSocket URL
  const targetWsUrl = useMemo(() => {
    if (wsUrl) return wsUrl;
    if (typeof window !== 'undefined' && window.location) {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host || '127.0.0.1:8001';
      return `${proto}//${host}/ws/enhancer/telemetry`;
    }
    return 'ws://127.0.0.1:8001/ws/enhancer/telemetry';
  }, [wsUrl]);

  // Sync settings update upward and to backend API
  const syncSettingsToBackend = useCallback(async (newSettings: UltraMaxSettings) => {
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            action: 'set_settings',
            settings: newSettings,
          })
        );
      }
      await fetch('/api/enhancer/ultramax/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSettings),
      });
    } catch {
      // Degrades gracefully, local state already updated
    }
  }, []);

  const updateParam = useCallback(
    <K extends keyof UltraMaxSettings>(key: K, val: UltraMaxSettings[K]) => {
      setSettings((prev) => {
        const next = { ...prev, [key]: val };
        onSettingsChange?.(next);
        syncSettingsToBackend(next);
        return next;
      });
    },
    [onSettingsChange, syncSettingsToBackend]
  );

  // Connect WebSocket telemetry stream
  useEffect(() => {
    let unmounted = false;

    function connectWs() {
      if (unmounted) return;
      try {
        const ws = new WebSocket(targetWsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!unmounted) setWsConnected(true);
        };

        ws.onmessage = (event) => {
          if (unmounted) return;
          try {
            const data = JSON.parse(event.data);
            if (data.event === 'telemetry' || data.event === 'hello') {
              setTelemetry(data);
              if (data.compilation) {
                setIsCompiling(Boolean(data.compilation.compiling));
              }
            } else if (data.event === 'settings_updated' && data.settings) {
              setSettings((prev) => ({ ...prev, ...data.settings }));
            }
          } catch {
            // Ignored
          }
        };

        ws.onclose = () => {
          if (!unmounted) {
            setWsConnected(false);
            reconnectTimeoutRef.current = setTimeout(connectWs, 2000);
          }
        };

        ws.onerror = () => {
          ws.close();
        };
      } catch {
        if (!unmounted) {
          reconnectTimeoutRef.current = setTimeout(connectWs, 3000);
        }
      }
    }

    connectWs();

    // Fallback polling if WebSocket is not connecting
    const pollInterval = setInterval(async () => {
      if (!wsConnected && !unmounted) {
        try {
          const res = await fetch('/api/enhancer/ultramax/telemetry');
          if (res.ok) {
            const data: TelemetryData = await res.json();
            setTelemetry(data);
            if (data.compilation) {
              setIsCompiling(Boolean(data.compilation.compiling));
            }
          }
        } catch {
          // Ignored
        }
      }
    }, 1000);

    return () => {
      unmounted = true;
      clearInterval(pollInterval);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [targetWsUrl, wsConnected]);

  // Initial settings load from REST
  useEffect(() => {
    let unmounted = false;
    async function loadSettings() {
      try {
        const res = await fetch('/api/enhancer/ultramax/settings');
        if (res.ok) {
          const data: UltraMaxSettings = await res.json();
          if (!unmounted) {
            setSettings((prev) => ({ ...prev, ...data }));
          }
        }
      } catch {
        // Ignored
      }
    }
    loadSettings();
    return () => {
      unmounted = true;
    };
  }, []);

  // Trigger TensorRT compilation
  const handleTriggerCompilation = async () => {
    setIsCompiling(true);
    setCompileError(null);
    try {
      const res = await fetch('/api/enhancer/ultramax/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          precision_mode: settings.precision_mode,
          provider: settings.provider,
        }),
      });
      if (!res.ok) {
        const err = await res.text();
        setCompileError(err || 'Failed to start compilation');
        setIsCompiling(false);
      }
    } catch (err: unknown) {
      setCompileError(err instanceof Error ? err.message : 'Compilation request failed');
      setIsCompiling(false);
    }
  };

  const vramPercent =
    telemetry.gpu_vram_total_mb > 0
      ? Math.min(100, Math.round((telemetry.gpu_vram_used_mb / telemetry.gpu_vram_total_mb) * 100))
      : 0;

  const ramPercent =
    telemetry.system_ram_total_mb > 0
      ? Math.min(100, Math.round((telemetry.system_ram_mb / telemetry.system_ram_total_mb) * 100))
      : 0;

  const comp = telemetry.compilation || DEFAULT_COMPILATION;

  return (
    <div
      className={`flex flex-col gap-5 p-5 rounded-2xl bg-slate-950/80 border border-white/10 backdrop-blur-md shadow-2xl text-slate-100 ${className}`}
      data-testid="enhancer-settings-panel"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/5 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Sparkles className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <h2 className="text-base font-semibold tracking-wide text-white flex items-center gap-2">
              UltraMax Restorer Engine
              <span className="text-micro font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300">
                Stage 5
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Anatomical feature controls, TensorRT engine profiles, and real-time telemetry
            </p>
          </div>
        </div>

        {/* Live WebSocket Status Pill */}
        <div className="flex items-center gap-2 text-xs">
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-mini font-medium border ${
              wsConnected
                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
            }`}
          >
            <Radio className={`w-3 h-3 ${wsConnected ? 'animate-pulse' : ''}`} />
            <span>{wsConnected ? 'Telemetry: 250ms Live' : 'Telemetry: Fallback Polling'}</span>
          </div>
        </div>
      </div>

      {/* Grid: Settings on left, Telemetry Dashboard on right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: Sliders & Engine Configuration (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
              Anatomical Region Controls
            </h3>
            <button
              type="button"
              onClick={() => {
                setSettings(DEFAULT_SETTINGS);
                syncSettingsToBackend(DEFAULT_SETTINGS);
                onSettingsChange?.(DEFAULT_SETTINGS);
              }}
              className="text-mini text-slate-400 hover:text-white flex items-center gap-1 cursor-pointer transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              Reset All
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Skin Texture */}
            <DebouncedSlider
              label="Skin Texture Strength"
              info="Frequency-split dermal pore and skin micro-texture reinjection"
              value={settings.skin_texture_strength}
              defaultValue={1.0}
              min={0.0}
              max={2.0}
              step={0.05}
              icon={<Sparkles className="w-3.5 h-3.5 text-amber-400" />}
              onChange={(v) => updateParam('skin_texture_strength', v)}
            />

            {/* Eye Clarity */}
            <DebouncedSlider
              label="Eye Clarity"
              info="Iris, corneal limbus, and lash micro-sharpening weight"
              value={settings.eye_clarity}
              defaultValue={1.0}
              min={0.0}
              max={2.0}
              step={0.05}
              icon={<Eye className="w-3.5 h-3.5 text-cyan-400" />}
              onChange={(v) => updateParam('eye_clarity', v)}
            />

            {/* Teeth Detail */}
            <DebouncedSlider
              label="Teeth Detail Weight"
              info="Dental enamel structure preservation without edge duplication"
              value={settings.teeth_detail_weight}
              defaultValue={0.5}
              min={0.0}
              max={1.0}
              step={0.05}
              icon={<Smile className="w-3.5 h-3.5 text-emerald-400" />}
              onChange={(v) => updateParam('teeth_detail_weight', v)}
            />

            {/* Lip Detail */}
            <DebouncedSlider
              label="Lip Detail Weight"
              info="Vermilion border and lip crease texture fidelity"
              value={settings.lip_detail_weight}
              defaultValue={0.5}
              min={0.0}
              max={1.0}
              step={0.05}
              icon={<Sparkles className="w-3.5 h-3.5 text-pink-400" />}
              onChange={(v) => updateParam('lip_detail_weight', v)}
            />

            {/* Global Blend Ratio */}
            <DebouncedSlider
              label="Restoration Blend Ratio"
              info="Master alpha blend between input swap plate and restored output"
              value={settings.blend_ratio}
              defaultValue={0.75}
              min={0.0}
              max={1.0}
              step={0.05}
              icon={<Layers className="w-3.5 h-3.5 text-purple-400" />}
              onChange={(v) => updateParam('blend_ratio', v)}
            />

            {/* Batch Size */}
            <DebouncedSlider
              label="Face Batch Size"
              info="Inference batch grouping for frames with multiple faces"
              value={settings.batch_size}
              defaultValue={1}
              min={1}
              max={8}
              step={1}
              unit=" faces"
              icon={<Layers className="w-3.5 h-3.5 text-blue-400" />}
              onChange={(v) => updateParam('batch_size', Math.round(v))}
            />
          </div>

          {/* Engine Settings Box */}
          <div className="p-3.5 rounded-xl bg-slate-900/40 border border-white/5 flex flex-col gap-3">
            <h4 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Engine Precision & Execution Target
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Precision Mode */}
              <div className="flex flex-col gap-1">
                <span className="text-mini font-medium text-slate-400">Precision Mode</span>
                <div className="grid grid-cols-2 gap-1.5 bg-slate-950 p-1 rounded-lg border border-white/5">
                  <button
                    type="button"
                    onClick={() => updateParam('precision_mode', 'hybrid_fp16_fp32')}
                    className={`text-xs py-1.5 px-2 rounded-md font-medium transition-colors cursor-pointer text-center ${
                      settings.precision_mode === 'hybrid_fp16_fp32'
                        ? 'bg-cyan-500 text-slate-950 shadow-sm'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Hybrid FP16/32
                  </button>
                  <button
                    type="button"
                    onClick={() => updateParam('precision_mode', 'strict_fp32')}
                    className={`text-xs py-1.5 px-2 rounded-md font-medium transition-colors cursor-pointer text-center ${
                      settings.precision_mode === 'strict_fp32'
                        ? 'bg-cyan-500 text-slate-950 shadow-sm'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Strict FP32
                  </button>
                </div>
              </div>

              {/* Execution Provider */}
              <div className="flex flex-col gap-1">
                <span className="text-mini font-medium text-slate-400">Execution Provider</span>
                <div className="grid grid-cols-3 gap-1 bg-slate-950 p-1 rounded-lg border border-white/5">
                  {(['tensorrt', 'cuda', 'cpu'] as ExecutionProvider[]).map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => updateParam('provider', p)}
                      className={`text-xs py-1.5 px-1.5 rounded-md font-medium uppercase tracking-wider transition-colors cursor-pointer text-center ${
                        settings.provider === p
                          ? 'bg-cyan-500 text-slate-950 shadow-sm'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Telemetry Dashboard & Compilation Progress (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            Live Hardware Telemetry
          </h3>

          {/* Metric Cards Grid */}
          <div className="grid grid-cols-2 gap-2.5">
            {/* FPS Card */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 flex flex-col gap-1">
              <div className="flex items-center justify-between text-mini text-slate-400">
                <span className="flex items-center gap-1">
                  <Gauge className="w-3.5 h-3.5 text-cyan-400" />
                  Speed (FPS)
                </span>
                <span className="font-mono text-micro text-slate-500">
                  avg {telemetry.average_fps.toFixed(1)}
                </span>
              </div>
              <div className="text-2xl font-bold font-mono text-cyan-400 flex items-baseline gap-1">
                {(telemetry.instantaneous_fps || telemetry.fps_now || 0).toFixed(1)}
                <span className="text-xs text-slate-500 font-normal">fps</span>
              </div>
              <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-400 transition-all duration-300"
                  style={{
                    width: `${Math.min(100, (telemetry.instantaneous_fps / 60) * 100)}%`,
                  }}
                />
              </div>
            </div>

            {/* GPU VRAM Card */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 flex flex-col gap-1">
              <div className="flex items-center justify-between text-mini text-slate-400">
                <span className="flex items-center gap-1">
                  <Flame className="w-3.5 h-3.5 text-amber-400" />
                  GPU VRAM
                </span>
                <span className="font-mono text-micro text-slate-500">{vramPercent}%</span>
              </div>
              <div className="text-lg font-bold font-mono text-amber-400 flex items-baseline gap-1">
                {(telemetry.gpu_vram_used_mb / 1024).toFixed(1)}
                <span className="text-xs text-slate-500 font-normal">
                  / {(telemetry.gpu_vram_total_mb / 1024).toFixed(1)} GB
                </span>
              </div>
              <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${
                    vramPercent > 85 ? 'bg-rose-500' : vramPercent > 70 ? 'bg-amber-400' : 'bg-emerald-400'
                  }`}
                  style={{ width: `${vramPercent}%` }}
                />
              </div>
            </div>

            {/* CPU Load Card */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 flex flex-col gap-1">
              <div className="flex items-center justify-between text-mini text-slate-400">
                <span className="flex items-center gap-1">
                  <Cpu className="w-3.5 h-3.5 text-purple-400" />
                  CPU Load
                </span>
                <span className="font-mono text-micro text-slate-500">
                  {telemetry.cpu_load_percentage.toFixed(0)}%
                </span>
              </div>
              <div className="text-lg font-bold font-mono text-purple-300">
                {telemetry.cpu_load_percentage.toFixed(1)}%
              </div>
              <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-400 transition-all duration-300"
                  style={{ width: `${Math.min(100, telemetry.cpu_load_percentage)}%` }}
                />
              </div>
            </div>

            {/* System RAM Card */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 flex flex-col gap-1">
              <div className="flex items-center justify-between text-mini text-slate-400">
                <span className="flex items-center gap-1">
                  <HardDrive className="w-3.5 h-3.5 text-blue-400" />
                  System RAM
                </span>
                <span className="font-mono text-micro text-slate-500">{ramPercent}%</span>
              </div>
              <div className="text-lg font-bold font-mono text-blue-300 flex items-baseline gap-1">
                {(telemetry.system_ram_mb / 1024).toFixed(1)}
                <span className="text-xs text-slate-500 font-normal">
                  / {(telemetry.system_ram_total_mb / 1024).toFixed(0)} GB
                </span>
              </div>
              <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-400 transition-all duration-300"
                  style={{ width: `${ramPercent}%` }}
                />
              </div>
            </div>
          </div>

          {/* Pipeline Stage Queue Depths */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-white/5 flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="flex items-center gap-1 font-medium">
                <Workflow className="w-3.5 h-3.5 text-cyan-400" />
                Pipeline Stage Queue Depths
              </span>
              <span className="text-micro font-mono text-slate-500">Bounded (In-Flight)</span>
            </div>

            <div className="grid grid-cols-4 gap-1.5 text-micro font-mono pt-1">
              {Object.entries(telemetry.pipeline_queue_depths || {}).map(([stage, depth]) => (
                <div
                  key={stage}
                  className="p-1.5 rounded-lg bg-slate-950/60 border border-white/5 flex flex-col items-center justify-center"
                >
                  <span className="text-slate-400 truncate w-full text-center">{stage}</span>
                  <span
                    className={`font-bold ${
                      depth > 4 ? 'text-rose-400' : depth > 2 ? 'text-amber-400' : 'text-cyan-400'
                    }`}
                  >
                    {depth}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* TensorRT Engine Compilation Card */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-white/5 flex flex-col gap-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
                <Zap className="w-3.5 h-3.5 text-cyan-400" />
                TensorRT Compilation Status
              </div>
              <div className="flex items-center gap-1">
                {comp.status === 'ready' && (
                  <span className="flex items-center gap-1 text-micro text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-500/20">
                    <CheckCircle2 className="w-3 h-3" /> Ready
                  </span>
                )}
                {comp.status === 'compiling' && (
                  <span className="flex items-center gap-1 text-micro text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-md border border-cyan-500/20 animate-pulse">
                    <RefreshCw className="w-3 h-3 animate-spin" /> Compiling
                  </span>
                )}
                {comp.status === 'failed' && (
                  <span className="flex items-center gap-1 text-micro text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded-md border border-rose-500/20">
                    <AlertCircle className="w-3 h-3" /> Failed
                  </span>
                )}
              </div>
            </div>

            {/* Compilation Progress Bar */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-mini text-slate-400">
                <span className="truncate">
                  {comp.stage === 'optimizing_graph' && 'Optimizing computational graph...'}
                  {comp.stage === 'building_engine' && 'Compiling TensorRT engine plan...'}
                  {comp.stage === 'canary_verification' && 'Startup canary & numerical stability check...'}
                  {comp.stage === 'ready' && 'Engine optimized and verified'}
                  {comp.stage === 'failed' && 'Engine compilation encountered an error'}
                  {comp.stage === 'idle' && 'No compilation in progress'}
                </span>
                <span className="font-mono text-cyan-400">
                  {Math.round((comp.progress || 0) * 100)}%
                </span>
              </div>

              <div className="h-2 bg-slate-950 rounded-full overflow-hidden border border-white/5 relative">
                <div
                  className={`h-full transition-all duration-300 ${
                    comp.status === 'failed'
                      ? 'bg-rose-500'
                      : comp.status === 'compiling'
                      ? 'bg-gradient-to-r from-cyan-500 to-blue-500 animate-pulse'
                      : 'bg-emerald-500'
                  }`}
                  style={{ width: `${Math.max(5, Math.round((comp.progress || 0) * 100))}%` }}
                />
              </div>

              {comp.message && <p className="text-micro text-slate-500 italic truncate">{comp.message}</p>}
              {compileError && <p className="text-micro text-rose-400">{compileError}</p>}
            </div>

            <button
              type="button"
              disabled={isCompiling || comp.status === 'compiling'}
              onClick={handleTriggerCompilation}
              className={`w-full py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                isCompiling || comp.status === 'compiling'
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-md hover:shadow-cyan-500/20 active:scale-[0.99]'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              {isCompiling || comp.status === 'compiling'
                ? 'Building TensorRT Engine...'
                : 'Compile / Warmup TensorRT Engine'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
