import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Icon } from '../../icons';
import { useTelemetryStore } from '../../store/telemetryStore';
import { useFrameTrace } from '../../store/frameTraceStore';
import { TRACE_STAGES } from './flamegraphModel';

const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
const fmtMs = (n) => Number.isFinite(n) ? `${n.toFixed(2)} ms` : '—';
const LIMIT = 480;

function measuredTotal(sample) {
  return TRACE_STAGES.reduce((sum, stage) => sum + (sample.stageMs[stage.key] ?? 0), 0);
}

function percentile(values, p) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * p)];
}

/** Canvas profiler for measured Stage 3 telemetry; a trace point stays linked to its source frame. */
export default function FrameFlamegraph({
  progress = {},
  settings = {},
  telemetry = null,
  completed = false,
  className = '',
  compact = false,
  onSeek = null,
  currentFrame = null,
  frameOffset = 0,
}) {
  const trace = useFrameTrace();
  const system = useTelemetryStore((s) => s.system);
  const displayStages = useMemo(() => {
    const maskEngines = [settings?.mask_engine, settings?.mask_engine_2]
      .filter((name) => name && name !== 'None');
    const maskLabel = maskEngines.some((name) => /xseg[\s_-]?3/i.test(String(name)))
      ? 'XSeg 3 Mask'
      : maskEngines.length ? `Mask · ${maskEngines.join(' + ')}` : 'Mask Engine';
    return TRACE_STAGES.map((stage) => stage.key === 'mask' ? { ...stage, label: maskLabel } : stage);
  }, [settings?.mask_engine, settings?.mask_engine_2]);
  const canvasRef = useRef(null);
  const hostRef = useRef(null);
  const [size, setSize] = useState({ width: 0, height: compact ? 112 : 174 });
  const [visibleCount, setVisibleCount] = useState(compact ? 72 : 144);
  const [hoverIndex, setHoverIndex] = useState(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setSize((old) => ({ ...old, width: entry.contentRect.width }));
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setSize((old) => ({ ...old, height: compact ? 112 : 174 }));
    setVisibleCount(compact ? 72 : 144);
  }, [compact]);

  const visible = useMemo(() => trace.slice(-visibleCount), [trace, visibleCount]);
  const stageStats = useMemo(() => displayStages.map((stage) => {
    const values = trace.map((sample) => sample.stageMs[stage.key]).filter(Number.isFinite);
    const current = [...trace].reverse().find((sample) => sample.stageMs[stage.key] !== null)?.stageMs[stage.key];
    return {
      ...stage,
      current: Number.isFinite(current) ? current : null,
      avg: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
      p95: percentile(values, 0.95),
    };
  }), [trace, displayStages]);

  const latest = trace[trace.length - 1] || null;
  const bottleneck = latest
    ? stageStats.reduce((top, stage) => (stage.current != null && (!top || stage.current > top.current) ? stage : top), null)
    : null;
  const totals = visible.map(measuredTotal);
  const scaleMax = Math.max(1, percentile(totals, 0.95) || 1, ...totals) * 1.08;
  const latencyValues = trace.map((sample) => sample.frameMs).filter((value) => value != null && value > 0);
  const latencyMedian = percentile(latencyValues, 0.5);
  const slowFrames = latencyMedian == null ? [] : visible.filter((sample) => sample.frameMs > latencyMedian * 1.35);
  const hovered = hoverIndex == null ? null : visible[hoverIndex] || null;
  const focusSample = hovered || latest;
  const focusFrame = focusSample ? focusSample.frame + Number(frameOffset || 0) : null;
  const currentAbsoluteFrame = Number(currentFrame ?? progress.current_frame);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size.width || !size.height) return;
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(size.width * ratio);
    canvas.height = Math.round(size.height * ratio);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, size.width, size.height);

    const left = compact ? 34 : 42;
    const right = 8;
    const top = 8;
    const bottom = compact ? 18 : 23;
    const plotW = Math.max(1, size.width - left - right);
    const plotH = Math.max(1, size.height - top - bottom);
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i += 1) {
      const y = top + plotH - (plotH * i) / 4;
      const ms = (scaleMax * i) / 4;
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + plotW, y); ctx.stroke();
      ctx.fillStyle = 'rgba(203,213,225,0.64)';
      ctx.fillText(ms >= 100 ? `${Math.round(ms)}` : ms.toFixed(0), left - 6, y);
    }

    if (!visible.length) {
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(203,213,225,0.58)';
      ctx.fillText('Waiting for measured per-stage samples', left + plotW / 2, top + plotH / 2);
      return;
    }

    const slot = plotW / visible.length;
    const barW = Math.max(1, Math.min(10, slot * 0.72));
    visible.forEach((sample, index) => {
      const x = left + index * slot + (slot - barW) / 2;
      let y = top + plotH;
      for (const stage of displayStages) {
        const ms = sample.stageMs[stage.key];
        if (ms == null || ms <= 0) continue;
        const h = Math.max(0.75, (ms / scaleMax) * plotH);
        y -= h;
        ctx.fillStyle = stage.color;
        ctx.globalAlpha = hoverIndex === index ? 1 : 0.88;
        ctx.fillRect(x, Math.max(top, y), barW, Math.min(h, plotH));
      }
      ctx.globalAlpha = 1;
      const endToEndSpike = latencyMedian != null && sample.frameMs > latencyMedian * 1.35;
      if (endToEndSpike) {
        ctx.fillStyle = '#fb7185';
        ctx.fillRect(x - 0.5, top - 3, barW + 1, 3);
      }
      if (hoverIndex === index) {
        ctx.strokeStyle = '#f8fafc';
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 1, top, barW + 2, plotH);
      }
      const shownFrame = sample.frame + Number(frameOffset || 0);
      if (Number.isFinite(currentAbsoluteFrame) && shownFrame === currentAbsoluteFrame) {
        ctx.strokeStyle = 'rgba(255,255,255,0.85)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x + barW / 2, top); ctx.lineTo(x + barW / 2, top + plotH); ctx.stroke();
      }
    });
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = 'rgba(203,213,225,0.6)';
    const firstFrame = visible[0].frame + Number(frameOffset || 0);
    const lastFrame = visible[visible.length - 1].frame + Number(frameOffset || 0);
    ctx.fillText(`f ${firstFrame}`, left, size.height - 3);
    ctx.textAlign = 'right';
    ctx.fillText(`f ${lastFrame}`, left + plotW, size.height - 3);
  }, [visible, size, compact, hoverIndex, scaleMax, latencyMedian, frameOffset, currentAbsoluteFrame, displayStages]);

  const sampleAtPointer = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const left = compact ? 34 : 42;
    const slot = (rect.width - left - 8) / Math.max(1, visible.length);
    return clamp(Math.floor((event.clientX - rect.left - left) / slot), 0, Math.max(0, visible.length - 1));
  };

  const downloadReport = () => {
    const report = {
      report_type: 'Roop Ultimate Frame Trace',
      schema_version: 'v2.8-frame-trace',
      exported_at: new Date().toISOString(),
      job: {
        duration_s: progress.duration_s || progress.elapsed_s || null,
        total_frames: progress.total_frames || null,
        current_frame: progress.current_frame || null,
        status: completed ? 'COMPLETED' : progress.error ? 'FAILED' : 'RUNNING',
      },
      recipe: {
        swapper_model: settings.swap_model || null,
        selected_enhancer: settings.selected_enhancer || null,
        provider: settings.provider || null,
        video_encoder: settings.output_video_codec || settings.video_encoder || null,
      },
      hardware: telemetry ?? system ?? null,
      basis: latest?.sampleWindowS
        ? `Backend stage worker-time, sampled over ${latest.sampleWindowS}s; stage values may overlap across worker threads.`
        : 'Backend stage worker-time samples; stage values may overlap across worker threads.',
      samples: trace,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `frame_trace_${Date.now()}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <section
      ref={hostRef}
      aria-label="Per-frame bottleneck flamegraph and frame trace profiler"
      data-testid="frame-flamegraph"
      className={`rounded-xl border border-white/10 bg-black/40 ${compact ? 'p-2.5 space-y-2' : 'p-4 space-y-3'} ${className}`}
    >
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2 text-xs font-bold uppercase tracking-wider text-white/90">
            <Icon.meter size={14} className="text-cyan-300" />
            <span>{compact ? 'Frame Trace Inspector' : 'Per-Frame Bottleneck Flamegraph'}</span>
            {bottleneck && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-400/40 bg-rose-500/15 px-2 py-0.5 text-micro normal-case tracking-normal text-rose-100 shadow-[0_0_14px_rgba(244,63,94,0.22)] animate-pulse">
                <span className="h-1.5 w-1.5 rounded-full bg-rose-300" />
                Bottleneck · {bottleneck.label} · {fmtMs(bottleneck.current)}
              </span>
            )}
          </div>
          <p className="mt-1 text-micro font-mono text-white/50">
            {trace.length
              ? `${trace.length} measured samples · ${latest?.sampleWindowS ? `${latest.sampleWindowS}s backend window` : 'backend timing window'} · worker time can overlap`
              : 'No measured stage samples yet; values are never estimated here.'}
            {latest?.frameMs != null ? ` · end-to-end ${fmtMs(latest.frameMs)}` : ''}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <button type="button" onClick={() => setVisibleCount((n) => clamp(Math.round(n / 1.4), 24, LIMIT))}
            disabled={visibleCount <= 24} className="rounded-md border border-white/10 px-2 py-1 text-micro text-white/70 hover:bg-white/10 disabled:opacity-40" aria-label="Zoom in on frame trace">−</button>
          <span className="min-w-12 text-center font-mono text-micro text-white/50">{visible.length} samples</span>
          <button type="button" onClick={() => setVisibleCount((n) => clamp(Math.round(n * 1.4), 24, LIMIT))}
            disabled={visibleCount >= Math.max(24, trace.length)} className="rounded-md border border-white/10 px-2 py-1 text-micro text-white/70 hover:bg-white/10 disabled:opacity-40" aria-label="Zoom out on frame trace">+</button>
          {completed && !compact && (
            <button type="button" onClick={downloadReport} className="ml-1 inline-flex items-center gap-1 rounded-md border border-white/10 px-2 py-1 text-micro text-white/70 hover:bg-white/10">
              <Icon.download size={12} /> Export
            </button>
          )}
        </div>
      </header>

      <div className="relative overflow-hidden rounded-lg border border-white/10 bg-[#07090d]" onWheel={(event) => {
        if (Math.abs(event.deltaY) < Math.abs(event.deltaX)) return;
        event.preventDefault();
        setVisibleCount((n) => clamp(Math.round(n * (event.deltaY < 0 ? 0.82 : 1.22)), 24, LIMIT));
      }}>
        <canvas
          ref={canvasRef}
          className={`block w-full ${onSeek ? 'cursor-crosshair' : 'cursor-default'}`}
          style={{ height: size.height }}
          role="img"
          aria-label="Stacked frame time breakdown. Click a sample to move the timeline playhead."
          onPointerMove={(event) => setHoverIndex(sampleAtPointer(event))}
          onPointerLeave={() => setHoverIndex(null)}
          onClick={(event) => {
            if (!onSeek || !visible.length) return;
            const sample = visible[sampleAtPointer(event)];
            if (sample) onSeek(clamp(sample.frame + Number(frameOffset || 0), 1, Number(progress.total_frames) + Number(frameOffset || 0) || Number.MAX_SAFE_INTEGER));
          }}
          title={onSeek ? 'Click a frame sample to seek the timeline' : 'Frame samples are visible in the Processing HUD'}
        />
        {focusSample && hovered && (
          <div className="pointer-events-none absolute right-2 top-2 z-10 max-w-[min(420px,90%)] rounded-lg border border-white/15 bg-[#090b10]/95 p-2.5 shadow-xl backdrop-blur">
            <div className="mb-1 flex items-center justify-between gap-3 font-mono text-micro text-white/80">
              <span>Frame {focusFrame}</span>
              <span>{focusSample.frameMs != null ? `end-to-end ${fmtMs(focusSample.frameMs)}` : 'stage timing sample'}</span>
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
              {displayStages.filter((stage) => focusSample.stageMs[stage.key] != null).map((stage) => (
                <span key={stage.key} className="flex items-center justify-between gap-2 font-mono text-nano text-white/65">
                  <span className="flex min-w-0 items-center gap-1 truncate"><i className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: stage.color }} />{stage.label}</span>
                  <b className="shrink-0 text-white/90">{fmtMs(focusSample.stageMs[stage.key])}</b>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {displayStages.map((stage) => (
          <span key={stage.key} className="inline-flex items-center gap-1 font-mono text-nano text-white/60">
            <i className="h-1.5 w-1.5 rounded-full" style={{ background: stage.color }} />{stage.label}
          </span>
        ))}
        {slowFrames.length > 0 && <span className="font-mono text-nano text-rose-300">{slowFrames.length} slow samples</span>}
      </div>

      <div className={`grid gap-1.5 ${compact ? 'grid-cols-2 sm:grid-cols-4' : 'grid-cols-2 sm:grid-cols-4 xl:grid-cols-8'}`}>
        {stageStats.map((stage) => (
          <div key={stage.key} className="min-w-0 rounded-lg border border-white/[0.06] bg-white/[0.025] px-2 py-1.5" title={`${stage.label}: observed backend measurements`}>
            <div className="flex min-w-0 items-center gap-1 truncate text-micro font-semibold text-white/65">
              <i className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: stage.color }} />{stage.label}
            </div>
            <div className="mt-0.5 font-mono text-xs font-bold tabular-nums text-white/90">{fmtMs(stage.current)}</div>
            {!compact && <div className="font-mono text-nano text-white/40">p95 {fmtMs(stage.p95)}</div>}
          </div>
        ))}
      </div>
    </section>
  );
}
