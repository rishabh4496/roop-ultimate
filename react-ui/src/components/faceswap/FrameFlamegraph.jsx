import React, { useMemo, useState, useEffect, useRef } from 'react';
import { useTelemetryStore } from '../../store/telemetryStore';
import { Icon } from '../../icons';
import { FLAMEGRAPH_STAGES, DEFAULT_LATENCY_MS, calcPercentile } from './flamegraphModel';

/**
 * Live Per-Frame Bottleneck Flamegraph & Telemetry Export (#/processing).
 *
 * Horizontal Stacked Frame-Time Flamegraph Bar (100% = total ms/frame):
 *   Decode (Cyan) | Detect/Align (Purple) | Swap (Emerald) | Mask/Roto (Blue) |
 *   Restore Ultra (Amber) | Relight/LUT (Pink) | Encode (Slate).
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

  const historyRef = useRef({
    decode: [],
    detect: [],
    swap: [],
    mask: [],
    restore: [],
    blend: [],
    encode: [],
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

  const totalMs = useMemo(() => {
    return Number(stageStats.reduce((acc, s) => acc + s.current, 0).toFixed(1));
  }, [stageStats]);

  const stagesWithShare = useMemo(() => {
    const total = Math.max(0.1, totalMs);
    return stageStats.map((s) => ({
      ...s,
      sharePct: Number(((s.current / total) * 100).toFixed(1)),
    }));
  }, [stageStats, totalMs]);

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
        selected_enhancer: settings.selected_enhancer || 'Restore Ultra',
        trt_precision: settings.trt_precision || 'mixed',
        provider: settings.provider || 'cuda',
        video_encoder: settings.output_video_codec || settings.video_encoder || 'h264_nvenc',
        threads: settings.max_threads || 10,
        face_detection_mode: settings.face_detection_mode || 'many',
        face_detector_size: settings.face_detector_size || 512,
        temporal_detection: !!settings.temporal_detection,
        track_identities: !!settings.track_identities,
        restore_ultra_frequency_blend: !!settings.restore_ultra_frequency_blend,
      },
      frame_flamegraph_telemetry: {
        total_frame_ms: totalMs,
        effective_fps: Number(effectiveFps),
        bottleneck_stage: bottleneck.label,
        bottleneck_share_pct: bottleneck.sharePct,
        stages: stagesWithShare.reduce((acc, s) => {
          acc[s.key] = {
            label: s.label,
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
            <div className="text-xs font-bold uppercase tracking-wider text-white/90 flex items-center gap-2">
              <span>Frame Bottleneck Flamegraph</span>
              <span className="font-mono text-micro text-accent font-semibold px-1.5 py-0.5 rounded bg-accent/10 border border-accent/20">
                100% = {totalMs} ms/frame
              </span>
            </div>
            <div className="text-micro text-muted font-mono">
              ~{effectiveFps} fps pipeline ceiling &bull; Bottleneck:{' '}
              <span className="font-bold text-amber-300">
                {bottleneck.label} ({bottleneck.sharePct}%)
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
          aria-label="Stacked frame latency split"
          className="relative h-7 w-full overflow-hidden rounded-xl bg-white/5 p-0.5 flex items-stretch border border-white/10 shadow-inner"
        >
          {stagesWithShare.map((stage) => {
            const isHovered = hoveredStage?.key === stage.key;
            return (
              <div
                key={stage.key}
                onMouseEnter={() => setHoveredStage(stage)}
                onMouseLeave={() => setHoveredStage(null)}
                style={{
                  width: `${Math.max(2.5, stage.sharePct)}%`,
                  backgroundColor: stage.color,
                }}
                className={`relative h-full transition-all duration-200 cursor-pointer flex items-center justify-center overflow-hidden border-r border-black/20 last:border-r-0 ${
                  isHovered ? 'brightness-125 z-10 scale-y-105 shadow-lg' : 'hover:brightness-110 opacity-90'
                }`}
                title={`${stage.label}: ${stage.current}ms (${stage.sharePct}%)`}
              >
                {/* Truncated label inside bar if width allows */}
                {stage.sharePct >= 7 && (
                  <span className="text-micro font-mono font-bold text-black drop-shadow-sm truncate px-1">
                    {stage.label.split('/')[0]} {stage.current}ms
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
          <div className="flex items-center justify-between gap-2 border-b border-white/10 pb-1.5 mb-2 font-mono text-xs">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full shrink-0" style={{ backgroundColor: hoveredStage.color }} />
              <span className="font-bold text-white text-sm">{hoveredStage.label}</span>
              <span className="text-white/50 text-xs">({hoveredStage.sharePct}% share)</span>
            </div>
            <div className="px-2 py-0.5 rounded text-micro font-semibold bg-white/10 text-white/90">
              Residency: <span className="text-accent font-bold">{hoveredStage.residency}</span>
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
          {stagesWithShare.map((s) => (
            <div
              key={s.key}
              onMouseEnter={() => setHoveredStage(s)}
              className="flex items-center gap-1.5 text-white/70 hover:text-white cursor-pointer transition-colors"
            >
              <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: s.color }} />
              <span>{s.label}:</span>
              <span className="text-white font-bold">{s.current}ms</span>
              <span className="text-white/40">({s.sharePct}%)</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
