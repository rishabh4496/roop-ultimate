import React, { useState, useEffect, useCallback } from 'react';
import {
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
  Workflow,
  Radio,
} from 'lucide-react';
import { useTelemetryStore, setEnhancerCompilation } from '../store/telemetryStore';

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

export interface EnhancerSettingsProps {
  initialSettings?: Partial<UltraMaxSettings>;
  onSettingsChange?: (settings: UltraMaxSettings) => void;
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

// ── Debounced Slider Subcomponent ──────────────────────────────────────────

interface DebouncedSliderProps {
  label: string;
  info: string;
  value: number;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  icon?: React.ReactNode;
  onChange: (val: number) => void;
}

function DebouncedSlider({
  label,
  info,
  value,
  defaultValue,
  min,
  max,
  step,
  unit = '',
  icon,
  onChange,
}: DebouncedSliderProps) {
  const [localVal, setLocalVal] = useState<number>(value);

  useEffect(() => {
    setLocalVal(value);
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseFloat(e.target.value);
    setLocalVal(next);
    onChange(next);
  };

  const isModified = Math.abs(localVal - defaultValue) > 0.0001;

  return (
    <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-slate-900/50 border border-white/5 hover:border-white/10 transition-colors">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 font-medium text-slate-200">
          {icon}
          {label}
        </span>
        <div className="flex items-center gap-1.5">
          {isModified && (
            <button
              type="button"
              aria-label={`Reset ${label} to default`}
              onClick={() => {
                setLocalVal(defaultValue);
                onChange(defaultValue);
              }}
              className="text-micro text-amber-400/80 hover:text-amber-300 transition-colors cursor-pointer"
              title="Reset to default"
            >
              MOD
            </button>
          )}
          <span className="font-mono text-cyan-400 font-semibold">
            {localVal.toFixed(step < 0.1 ? 2 : 1)}
            {unit}
          </span>
        </div>
      </div>

      <p className="text-micro text-muted line-clamp-1">{info}</p>

      <div className="flex items-center gap-2 pt-1">
        <span className="text-micro font-mono text-muted w-6">{min}</span>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={localVal}
          onChange={handleChange}
          aria-label={label}
          className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400 focus:outline-none"
        />
        <span className="text-micro font-mono text-muted w-6 text-right">{max}</span>
      </div>
    </div>
  );
}

// ── Main EnhancerSettings Component ────────────────────────────────────────

export default function EnhancerSettings({
  initialSettings,
  onSettingsChange,
  className = '',
}: EnhancerSettingsProps) {
  const [settings, setSettings] = useState<UltraMaxSettings>(() => ({
    ...DEFAULT_SETTINGS,
    ...initialSettings,
  }));

  // Centralized telemetry from singleton store — zero redundant WebSockets
  const runTele = useTelemetryStore((s: any) => s.run) as Record<string, any>;
  const enhancerStore = useTelemetryStore((s: any) => s.enhancer) as Record<string, any>;

  const [isCompiling, setIsCompiling] = useState<boolean>(false);
  const [compileError, setCompileError] = useState<string | null>(null);

  // Sync settings update upward and to backend API
  const syncSettingsToBackend = useCallback(async (newSettings: UltraMaxSettings) => {
    try {
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

  // On mount, fetch current settings and compilation status once
  useEffect(() => {
    let active = true;
    fetch('/api/enhancer/ultramax/settings')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (active && data) {
          setSettings((prev) => ({ ...prev, ...data }));
        }
      })
      .catch(() => {});

    fetch('/api/enhancer/ultramax/status')
      .then((res) => (res.ok ? res.json() : null))
      .then((status) => {
        if (active && status) {
          setEnhancerCompilation(status);
          if (status.compiling) setIsCompiling(true);
        }
      })
      .catch(() => {});

    return () => { active = false; };
  }, []);

  // Trigger TensorRT Compilation
  const handleTriggerCompilation = useCallback(async () => {
    try {
      setIsCompiling(true);
      setCompileError(null);
      setEnhancerCompilation({
        compiling: true,
        progress: 0.05,
        stage: 'optimizing_graph',
        status: 'compiling',
        message: 'Compilation triggered via workstation UI...',
      });

      const res = await fetch('/api/enhancer/ultramax/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          precision_mode: settings.precision_mode,
          provider: settings.provider,
        }),
      });

      if (!res.ok) {
        throw new Error(`Compilation trigger failed: ${res.statusText}`);
      }

      const status: EngineCompilationStatus = await res.json();
      setEnhancerCompilation(status);
    } catch (err: any) {
      setCompileError(err?.message || 'Engine compilation failed to start');
      setEnhancerCompilation({
        compiling: false,
        status: 'failed',
        stage: 'failed',
        message: err?.message || 'Compilation error',
      });
    } finally {
      setIsCompiling(false);
    }
  }, [settings.precision_mode, settings.provider]);

  // Derived Telemetry Values. (VRAM / CPU / RAM are not here: the header's telemetry pill is the one place
  // the app shows hardware, and subscribing to the 3 s system sample made this panel re-render on an idle tab.)
  const comp = enhancerStore?.compilation || DEFAULT_COMPILATION;
  const fpsValue = runTele.fps || runTele.fps_now || 0;

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

        {/* Live Status Pill */}
        <div className="flex items-center gap-2 text-xs">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-mini font-medium border bg-emerald-500/10 border-emerald-500/20 text-emerald-400">
            <Radio className="w-3 h-3 animate-pulse" />
            <span>Shared Telemetry Service</span>
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

          {/* Engine Execution Options */}
          <div className="p-3.5 rounded-xl bg-slate-900/40 border border-white/5 flex flex-col gap-3">
            <h4 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-cyan-400" />
              Engine Precision & Execution Provider
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label htmlFor="enhancer-precision-mode" className="text-xs text-slate-400 mb-1.5 block">Precision Mode</label>
                <select
                  id="enhancer-precision-mode"
                  value={settings.precision_mode}
                  onChange={(e) => updateParam('precision_mode', e.target.value as PrecisionMode)}
                  className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-white/10 text-white text-xs focus:outline-none focus:border-cyan-400"
                >
                  <option value="hybrid_fp16_fp32">Hybrid FP16/FP32 (Balanced Speed & Precision)</option>
                  <option value="strict_fp32">Strict FP32 (Full Precision Reference)</option>
                </select>
              </div>

              <div>
                <label htmlFor="enhancer-execution-provider" className="text-xs text-slate-400 mb-1.5 block">Execution Provider</label>
                <select
                  id="enhancer-execution-provider"
                  value={settings.provider}
                  onChange={(e) => updateParam('provider', e.target.value as ExecutionProvider)}
                  className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-white/10 text-white text-xs focus:outline-none focus:border-cyan-400"
                >
                  <option value="tensorrt">NVIDIA TensorRT (Hardware Accelerated Plan)</option>
                  <option value="cuda">CUDA (Direct Execution)</option>
                  <option value="cpu">CPU Fallback</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Live Telemetry & Compilation Status (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Gauge className="w-3.5 h-3.5 text-cyan-400" />
            Live Throughput & Queue Telemetry
          </h3>

          <div className="grid grid-cols-1 gap-2.5">
            {/* Speed / FPS Card */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 flex flex-col gap-1">
              <div className="flex items-center justify-between text-mini text-slate-400">
                <span className="flex items-center gap-1">
                  <Gauge className="w-3.5 h-3.5 text-cyan-400" />
                  Speed (FPS)
                </span>
                <span className="font-mono text-micro text-muted">
                  avg {fpsValue.toFixed(1)}
                </span>
              </div>
              <div className="text-2xl font-bold font-mono text-cyan-400 flex items-baseline gap-1">
                {fpsValue.toFixed(1)}
                <span className="text-xs text-muted font-normal">fps</span>
              </div>
              <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-400 transition-all duration-300"
                  style={{
                    width: `${Math.min(100, (fpsValue / 60) * 100)}%`,
                  }}
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
              <span className="text-micro font-mono text-muted">Bounded (In-Flight)</span>
            </div>

            <div className="grid grid-cols-4 gap-1.5 text-micro font-mono pt-1">
              {Object.entries(enhancerStore?.pipeline_queue_depths || { decode: 0, swap: 0, enhance: 0, encode: 0 }).map(([stage, depth]) => (
                <div
                  key={stage}
                  className="p-1.5 rounded-lg bg-slate-950/60 border border-white/5 flex flex-col items-center justify-center"
                >
                  <span className="text-slate-400 truncate w-full text-center">{stage}</span>
                  <span
                    className={`font-bold ${
                      (depth as number) > 4 ? 'text-rose-400' : (depth as number) > 2 ? 'text-amber-400' : 'text-cyan-400'
                    }`}
                  >
                    {depth as number}
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

              {comp.message && <p className="text-micro text-muted italic truncate">{comp.message}</p>}
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
