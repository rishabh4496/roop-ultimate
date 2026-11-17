import React, { useMemo, useState, useEffect, useRef } from 'react';
import { useTelemetryStore } from '../../store/telemetryStore';
import { Icon } from '../../icons';
import {
  FLAMEGRAPH_STAGES,
  MODEL_PASS_STAGES,
  DEFAULT_LATENCY_MS,
  DEFAULT_MODEL_LATENCY_MS,
  calcPercentile,
} from './flamegraphModel';

/**
 * Live Per-Frame Bottleneck Flamegraph & Telemetry Export (#/processing).
 *
 * Horizontal Stacked Frame-Time Flamegraph Bar (100% = total ms/frame):
 *   Decode / Stabilizer (Cyan) | Detect / SCRFD-gnkps (Purple) |
 *   Swap / HyperSwap (Emerald) | Mask / BiSeNet+XSeg (Blue) |
 *   Restore / UltraMax (Amber) | Blend / Laplacian Blend (Pink) | Encode / NVENC (Slate).
 *
 * Supports both high-level pipeline stages and detailed per-stage model timings:
 *   (SCRFD-gnkps, Stabilizer, BiSeNet+XSeg, HyperSwap, LivePortrait, UltraMax, Laplacian Blend).
 *
 * Hovering over any segment shows Min / Avg / P95 / Max (ms) and residency.
 * Completed runs offer 1-click Download Job Telemetry & Recipe Report (.json).
 */
export default function FrameFlamegraph({
  progress = {},
  settings = {},
  telemetry = null,
  completed = false,
  className = '',
}) {
  const dash = useTelemetryStore((s) => s.dashboard);
  const latencyMs = dash?.telemetry?.latency_ms;
  const modelLatencyMs = dash?.telemetry?.model_latency_ms;

  const [viewMode, setViewMode] = useState('pipeline'); // 'pipeline' | 'models'

  const historyRef = useRef({
    decode: [],
    detect: [],
    swap: [],
    mask: [],
    restore: [],
    blend: [],
    encode: [],
  });

  const modelHistoryRef = useRef({
    'SCRFD-gnkps': [],
    'Stabilizer': [],
    'BiSeNet+XSeg': [],
    'HyperSwap': [],
    'LivePortrait': [],
    'UltraMax': [],
    'Laplacian Blend': [],
  });

  const [hoveredStage, setHoveredStage] = useState(null);

  // Accumulate timing samples
  useEffect(() => {
    if (!latencyMs) return;
    for (const stage of FLAMEGRAPH_STAGES) {
      const val = latencyMs[stage.key];
      if (typeof val === 'number' && Number.isFinite(val) && val > 0) {
        const hist = historyRef.current[stage.key];
        hist.push(val);
        if (hist.length > 200) hist.shift();
      }
    }
  }, [latencyMs]);

  // Accumulate model timing samples
  useEffect(() => {
    for (const m of MODEL_PASS_STAGES) {
      const rawVal = modelLatencyMs?.[m.modelKey] ?? latencyMs?.[m.stageKey];
      if (typeof rawVal === 'number' && Number.isFinite(rawVal) && rawVal > 0) {
        const hist = modelHistoryRef.current[m.modelKey];
        if (hist) {
          hist.push(rawVal);
          if (hist.length > 200) hist.shift();
        }
      }
    }
  }, [latencyMs, modelLatencyMs]);

  // Compute live / windowed metrics per stage
  const stageStats = useMemo(() => {
    return FLAMEGRAPH_STAGES.map((s) => {
      const current = typeof latencyMs?.[s.key] === 'number' && Number.isFinite(latencyMs[s.key])
        ? latencyMs[s.key]
        : DEFAULT_LATENCY_MS[s.key];

      const history = historyRef.current[s.key];
      const samples = history.length > 0 ? history : [current];
      const sorted = [...samples].sort((a, b) => a - b);
      const min = sorted[0];
      const max = sorted[sorted.length - 1];
      const sum = sorted.reduce((acc, v) => acc + v, 0);
      const avg = sum / sorted.length;
      const p95 = calcPercentile(sorted, 0.95);

      return {
        ...s,
        current: Number(current.toFixed(1)),
        min: Number(min.toFixed(1)),
        avg: Number(avg.toFixed(1)),
        p95: Number(p95.toFixed(1)),
        max: Number(max.toFixed(1)),
      };
    });
  }, [latencyMs]);

  // Compute live / windowed metrics per model pass
  const modelStats = useMemo(() => {
    return MODEL_PASS_STAGES.map((m) => {
      const rawVal = modelLatencyMs?.[m.modelKey] ?? latencyMs?.[m.stageKey];
      const current = typeof rawVal === 'number' && Number.isFinite(rawVal)
        ? rawVal
        : DEFAULT_MODEL_LATENCY_MS[m.modelKey] || 0;

      const history = modelHistoryRef.current[m.modelKey];
      const samples = history && history.length > 0 ? history : [current];
      const sorted = [...samples].sort((a, b) => a - b);
      const min = sorted[0];
      const max = sorted[sorted.length - 1];
      const sum = sorted.reduce((acc, v) => acc + v, 0);
      const avg = sum / sorted.length;
      const p95 = calcPercentile(sorted, 0.95);

      return {
        ...m,
        current: Number(current.toFixed(1)),
        min: Number(min.toFixed(1)),
        avg: Number(avg.toFixed(1)),
        p95: Number(p95.toFixed(1)),
        max: Number(max.toFixed(1)),
      };
    });
  }, [latencyMs, modelLatencyMs]);

  const totalMs = useMemo(() => {
    return Number(stageStats.reduce((acc, s) => acc + s.current, 0).toFixed(1));
  }, [stageStats]);

  const totalModelMs = useMemo(() => {
    return Number(modelStats.reduce((acc, m) => acc + m.current, 0).toFixed(1));
  }, [modelStats]);

  const stagesWithShare = useMemo(() => {
    const total = Math.max(0.1, totalMs);
    return stageStats.map((s) => ({
      ...s,
      sharePct: Number(((s.current / total) * 100).toFixed(1)),
    }));
  }, [stageStats, totalMs]);

  const modelsWithShare = useMemo(() => {
    const total = Math.max(0.1, totalModelMs);
    return modelStats.map((m) => ({
      ...m,
      sharePct: Number(((m.current / total) * 100).toFixed(1)),
    }));
  }, [modelStats, totalModelMs]);

  // Active items based on viewMode
  const activeItems = viewMode === 'models' ? modelsWithShare : stagesWithShare;

  // Bottleneck
  const bottleneck = useMemo(() => {
    let top = stagesWithShare[0];
    for (const s of stagesWithShare) {
      if (s.sharePct > top.sharePct) top = s;
    }
    return top;
  }, [stagesWithShare]);

  const effectiveFps = totalMs > 0 ? (1000 / totalMs).toFixed(1) : '—';

  // Export Telemetry & Recipe Report
  const downloadReport = () => {
    const reportData = {
      report_type: 'Roop Ultimate Job Telemetry & Recipe Report',
      schema_version: 'v2.7-stage27',
      exported_at: new Date().toISOString(),
      job_summary: {
        duration_s: progress.duration_s || progress.elapsed_s || null,
        total_frames: progress.total_frames || null,
        current_frame: progress.current_frame || null,
        average_fps: progress.fps || Number(effectiveFps) || null,
        status: completed ? 'COMPLETED' : progress.error ? 'FAILED' : 'STOPPED',
      },
      recipe_configuration: {
        swapper_model: settings.swap_model || 'realswap',
        selected_enhancer: settings.selected_enhancer || 'UltraMax',
        trt_precision: settings.trt_precision || 'mixed',
        provider: settings.provider || 'cuda',
        video_encoder: settings.output_video_codec || settings.video_encoder || 'h264_nvenc',
        threads: settings.max_threads || 10,
        face_detection_mode: settings.face_detection_mode || 'many',
        face_detector_size: settings.face_detector_size || 512,
        temporal_detection: !!settings.temporal_detection,
        track_identities: !!settings.track_identities,
        restore_ultra_frequency_blend: !!settings.restore_ultra_frequency_blend,
        ultramax_batch_size: settings.ultramax_batch_size || 1,
        pixel_boost: settings.pixel_boost || 256,
      },
      frame_flamegraph_telemetry: {
        total_frame_ms: totalMs,
        effective_fps: Number(effectiveFps),
        bottleneck_stage: bottleneck.label,
        bottleneck_model: bottleneck.model || 'Unknown',
        bottleneck_share_pct: bottleneck.sharePct,
        stages: stagesWithShare.reduce((acc, s) => {
          acc[s.key] = {
            label: s.label,
            model: s.model,
            current_ms: s.current,
            share_pct: s.sharePct,
            min_ms: s.min,
            avg_ms: s.avg,
            p95_ms: s.p95,
            max_ms: s.max,
            residency: s.residency,
          };
          return acc;
        }, {}),
        model_passes: modelsWithShare.reduce((acc, m) => {
          acc[m.modelKey] = {
            model: m.modelKey,
            category: m.category,
            stage: m.stageKey,
            current_ms: m.current,
            share_pct: m.sharePct,
            min_ms: m.min,
            avg_ms: m.avg,
            p95_ms: m.p95,
            max_ms: m.max,
            residency: m.residency,
          };
          return acc;
        }, {}),
      },
      hardware_environment: {
        gpu_util: telemetry?.gpu_util ?? null,
        vram_used_gb: telemetry?.vram_used ?? null,
        vram_total_gb: telemetry?.vram_total ?? null,
        cpu_percent: telemetry?.cpu_percent ?? null,
        active_pools: telemetry?.pools ?? null,
      },
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `job_telemetry_recipe_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div
      role="region"
      aria-label="Frame Bottleneck Flamegraph"
      data-testid="frame-flamegraph"
      className={`rounded-2xl border border-white/10 bg-black/45 p-4 shadow-xl backdrop-blur-md space-y-3 select-none ${className}`}
    >
      {/* ── Header ── */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="grid place-items-center h-6 w-6 rounded-md bg-[var(--accent)]/15 text-accent border border-[var(--accent)]/30">
            <Icon.meter size={14} />
          </span>
          <div>
            <div className="text-xs font-bold uppercase tracking-wider text-white/90 flex items-center gap-2 flex-wrap">
              <span>Frame Bottleneck Flamegraph</span>
              <span className="font-mono text-micro text-accent font-semibold px-1.5 py-0.5 rounded bg-accent/10 border border-accent/20">
                100% = {viewMode === 'models' ? totalModelMs : totalMs} ms/frame
              </span>
              {/* View Mode Toggle */}
              <div className="inline-flex rounded-lg border border-white/10 bg-black/40 p-0.5 text-nano font-mono">
                <button
                  type="button"
                  onClick={() => setViewMode('pipeline')}
                  className={`px-2 py-0.5 rounded-md font-bold transition-colors ${
                    viewMode === 'pipeline'
                      ? 'bg-accent/20 text-accent border border-accent/30'
                      : 'text-white/60 hover:text-white'
                  }`}
                >
                  Pipeline Stages
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('models')}
                  className={`px-2 py-0.5 rounded-md font-bold transition-colors ${
                    viewMode === 'models'
                      ? 'bg-accent/20 text-accent border border-accent/30'
                      : 'text-white/60 hover:text-white'
                  }`}
                >
                  Model Timings (7 Models)
                </button>
              </div>
            </div>
            <div className="text-micro text-muted font-mono">
              ~{effectiveFps} fps pipeline ceiling &bull; Bottleneck:{' '}
              <span className="font-bold text-amber-300">
                {bottleneck.label} ({bottleneck.model ? `${bottleneck.model} · ` : ''}{bottleneck.sharePct}%)
              </span>
            </div>
          </div>
        </div>

        {/* Completed Job Export Button */}
        {completed && (
          <button
            type="button"
            onClick={downloadReport}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold bg-white/10 hover:bg-white/20 text-white border border-white/15 transition-all shadow-sm cursor-pointer"
            title="Download full JSON report with flamegraph timings, recipe parameters, and hardware telemetry"
          >
            <Icon.download size={14} className="text-accent" />
            <span>Download Job Telemetry &amp; Recipe Report (.json)</span>
          </button>
        )}
      </div>

      {/* ── Horizontal Stacked Flamegraph Bar ── */}
      <div className="relative pt-1 pb-1">
        <div
          role="group"
          aria-label={viewMode === 'models' ? 'Per-model latency split' : 'Stacked frame latency split'}
          className="relative h-7 w-full overflow-hidden rounded-xl bg-white/5 p-0.5 flex items-stretch border border-white/10 shadow-inner"
        >
          {activeItems.map((item) => {
            const isHovered = (hoveredStage?.key || hoveredStage?.modelKey) === (item.key || item.modelKey);
            const displayLabel = item.modelKey || item.label;
            return (
              <div
                key={item.key || item.modelKey}
                onMouseEnter={() => setHoveredStage(item)}
                onMouseLeave={() => setHoveredStage(null)}
                style={{
                  width: `${Math.max(2.5, item.sharePct)}%`,
                  backgroundColor: item.color,
                }}
                className={`relative h-full transition-all duration-200 cursor-pointer flex items-center justify-center overflow-hidden border-r border-black/20 last:border-r-0 ${
                  isHovered ? 'brightness-125 z-10 scale-y-105 shadow-lg' : 'hover:brightness-110 opacity-90'
                }`}
                title={`${displayLabel}: ${item.current}ms (${item.sharePct}%) - ${item.residency}`}
              >
                {/* Truncated label inside bar if width allows */}
                {item.sharePct >= 8 && (
                  <span className="text-micro font-mono font-bold text-black drop-shadow-sm truncate px-1">
                    {displayLabel.split('/')[0].split('-')[0]} {item.current}ms
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Interactive Detail Cards / Hover Tooltip ── */}
      {hoveredStage ? (
        <div className="p-2.5 rounded-xl bg-white/5 border border-white/15 animate-in fade-in duration-150">
          <div className="flex items-center justify-between gap-2 border-b border-white/10 pb-1.5 mb-2 font-mono text-xs flex-wrap">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full shrink-0" style={{ backgroundColor: hoveredStage.color }} />
              <span className="font-bold text-white text-sm">
                {hoveredStage.label}
                {hoveredStage.model && hoveredStage.model !== hoveredStage.label && (
                  <span className="ml-1 text-white/70 font-semibold">({hoveredStage.model})</span>
                )}
              </span>
              <span className="text-white/50 text-xs">({hoveredStage.sharePct}% share)</span>
            </div>
            <div className="flex items-center gap-2">
              {hoveredStage.category && (
                <span className="px-2 py-0.5 rounded text-micro bg-black/40 text-white/70 border border-white/5">
                  {hoveredStage.category}
                </span>
              )}
              <div className="px-2 py-0.5 rounded text-micro font-semibold bg-white/10 text-white/90">
                Residency: <span className="text-accent font-bold">{hoveredStage.residency}</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-2 font-mono text-center">
            <div className="p-1.5 rounded-lg bg-black/40 border border-white/5">
              <div className="text-micro text-muted uppercase">Min</div>
              <div className="text-xs font-bold text-emerald-400">{hoveredStage.min} ms</div>
            </div>
            <div className="p-1.5 rounded-lg bg-black/40 border border-white/5">
              <div className="text-micro text-muted uppercase">Avg</div>
              <div className="text-xs font-bold text-white">{hoveredStage.avg} ms</div>
            </div>
            <div className="p-1.5 rounded-lg bg-black/40 border border-white/5">
              <div className="text-micro text-muted uppercase">P95</div>
              <div className="text-xs font-bold text-amber-300">{hoveredStage.p95} ms</div>
            </div>
            <div className="p-1.5 rounded-lg bg-black/40 border border-white/5">
              <div className="text-micro text-muted uppercase">Max</div>
              <div className="text-xs font-bold text-red-400">{hoveredStage.max} ms</div>
            </div>
          </div>
        </div>
      ) : (
        /* Legend Strip when not hovering */
        <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-nano font-mono pt-1">
          {activeItems.map((item) => (
            <div
              key={item.key || item.modelKey}
              onMouseEnter={() => setHoveredStage(item)}
              className="flex items-center gap-1.5 text-white/70 hover:text-white cursor-pointer transition-colors"
            >
              <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
              <span>{item.modelKey || item.label}:</span>
              <span className="text-white font-bold">{item.current}ms</span>
              <span className="text-white/60">({item.sharePct}%)</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Sub-panel: Per-Stage Model Timing Matrix (SCRFD-gnkps, Stabilizer, BiSeNet+XSeg, HyperSwap, LivePortrait, UltraMax, Laplacian Blend) ── */}
      <div className="pt-2 border-t border-white/5">
        <div className="flex items-center justify-between text-nano font-mono text-muted mb-1.5">
          <span className="uppercase font-bold tracking-wider text-white/70">Per-Stage Model Timings:</span>
          <span>7 Hardware-Resident Passes</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-1.5 font-mono text-micro">
          {modelsWithShare.map((m) => (
            <div
              key={m.modelKey}
              onMouseEnter={() => setHoveredStage(m)}
              onMouseLeave={() => setHoveredStage(null)}
              className="p-1.5 rounded-lg bg-white/[0.03] border border-white/5 hover:border-white/20 transition-all cursor-pointer"
            >
              <div className="flex items-center gap-1 truncate text-white/80 font-bold">
                <span className="h-1.5 w-1.5 rounded-full shrink-0" style={{ backgroundColor: m.color }} />
                <span className="truncate">{m.modelKey}</span>
              </div>
              <div className="flex items-baseline justify-between mt-0.5">
                <span className="text-white font-semibold">{m.current} ms</span>
                <span className="text-white/50 text-nano">{m.sharePct}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
