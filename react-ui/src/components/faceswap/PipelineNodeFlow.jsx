import React from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useTelemetryStore } from '../../store/telemetryStore';
import { MODULAR_FLOW_NODES, latencyTone, formatLatency } from './pillModel';

/**
 * Stage 24: Intermediate buffer solo passes for modular inspection.
 * Provides 1-click soloing of BiSeNet Parsing, XSeg 3 Mask, Occlusion Alpha, and Depth/Z-Order passes.
 */
export const BUFFER_SOLO_PASSES = Object.freeze([
  {
    id: 'bisenet_parsing',
    key: 'bisenet_mask',
    alias: 'bisenet',
    label: 'BiSeNet Parsing',
    shortLabel: 'BiSeNet',
    color: '#34d399',
    description: '1-click solo parsed facial regions (eyes, eyebrows, nose, mouth, skin) from BiSeNet',
  },
  {
    id: 'xseg_3_mask',
    key: 'mask',
    alias: 'xseg',
    label: 'XSeg 3 Mask',
    shortLabel: 'XSeg 3',
    color: '#f59e0b',
    description: '1-click solo DeepFaceLab XSeg-3 occlusion matte boundary pass',
  },
  {
    id: 'occlusion_alpha',
    key: 'occlusion_mask',
    alias: 'occlusion',
    label: 'Occlusion Alpha',
    shortLabel: 'Occlusion',
    color: '#38bdf8',
    description: '1-click solo foreign object occlusion alpha transparency matte',
  },
  {
    id: 'depth_z_order',
    key: 'depth_z_order',
    alias: 'depth',
    label: 'Depth/Z-Order',
    shortLabel: 'Depth/Z',
    color: '#a78bfa',
    description: '1-click solo estimated relative depth and occlusion stacking order map',
  },
]);

/**
 * Stage 24: Visual Pipeline Node Flow Bar with 1-click buffer solo buttons
 * and kernel latency badges from /ws/telemetry.
 */
export function VisualPipelineNodeFlowBar({
  soloStage = null,
  onSoloStage,
  bypassedStages = {},
  onToggleBypass,
  className = '',
}) {
  const latencyMap = useTelemetryStore(useShallow((s) => {
    const dashLat = s.dashboard?.telemetry?.latency_ms;
    const s3Ops = s.pipeline?.stages?.STAGE_3_MAIN_SWAP?.ops;
    return {
      warp: dashLat?.detect ?? s3Ops?.warp?.ms ?? 3.4,
      model: dashLat?.swap ?? s3Ops?.model?.ms ?? 11.2,
      lipsync: s.run?.lipsync_ms ?? dashLat?.lipsync ?? 4.2,
      mask: dashLat?.mask ?? 6.8,
      restore: dashLat?.restore ?? s3Ops?.restore?.ms ?? 14.5,
      blend: dashLat?.blend ?? s3Ops?.blend?.ms ?? 4.1,
    };
  }));

  const isPassActive = (pass) => soloStage === pass.id || soloStage === pass.key || soloStage === pass.alias;

  return (
    <div
      data-testid="visual-pipeline-node-flow-bar"
      className={`flex flex-col gap-2 p-2 rounded-xl bg-black/45 border border-white/10 text-xs backdrop-blur-sm select-none ${className}`}
    >
      {/* Top Strip: Modular Pipeline Flow Nodes */}
      <div className="flex flex-wrap items-center justify-between gap-1.5 px-1">
        <div className="flex items-center gap-1.5 text-micro font-semibold text-white/50 uppercase tracking-wider">
          <span>Pipeline Nodes</span>
          <span className="text-note text-white/40 font-normal">· Alt+Click or [S] to solo intermediate buffer · ⏻ bypass</span>
        </div>
        {soloStage && (
          <button
            type="button"
            data-testid="flow-clear-solo"
            onClick={() => onSoloStage?.(null)}
            className="px-2 py-0.5 rounded text-micro font-bold text-amber-300 bg-amber-400/15 border border-amber-400/30 hover:bg-amber-400/25 transition-all"
            title="Clear active buffer solo"
          >
            Clear Solo ({soloStage}) ✕
          </button>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-1">
        {MODULAR_FLOW_NODES.map((node, i) => {
          const isSolo = soloStage === node.id || soloStage === node.key;
          const isBypassed = Boolean(bypassedStages[node.id]);
          const ms = latencyMap[node.latencyKey] ?? node.fallbackMs;

          return (
            <React.Fragment key={node.id}>
              {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-noninteractive-element-interactions */}
              <div
                role="group"
                aria-label={`${node.n}. ${node.label}`}
                data-testid={`flow-node-${node.id}`}
                data-solo={isSolo ? 'true' : 'false'}
                data-bypassed={isBypassed ? 'true' : 'false'}
                onClick={(e) => {
                  if (e.altKey && onSoloStage) {
                    e.preventDefault();
                    onSoloStage(isSolo ? null : node.id);
                  }
                }}
                className={`group relative flex items-center gap-1.5 px-2 py-1 rounded-lg border transition-all ${
                  isSolo
                    ? 'border-amber-400 bg-amber-400/20 text-white shadow-[0_0_12px_rgba(251,191,36,0.35)] ring-1 ring-amber-400/50'
                    : isBypassed
                      ? 'border-white/5 bg-white/[0.02] text-white/35 grayscale'
                      : 'border-white/10 bg-white/[0.05] text-white/85 hover:border-white/20 hover:bg-white/[0.08]'
                }`}
                title={`[${node.n}. ${node.label}] · Alt+Click to solo buffer · Click power to toggle bypass`}
              >
                <button
                  type="button"
                  data-testid={`flow-bypass-${node.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleBypass?.(node.id);
                  }}
                  className={`flex items-center justify-center h-4 w-4 rounded-full text-micro transition-colors ${
                    isBypassed
                      ? 'bg-rose-500/20 text-rose-300 hover:bg-rose-500/30'
                      : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30'
                  }`}
                  aria-label={isBypassed ? `Enable ${node.label}` : `Bypass ${node.label}`}
                  title={isBypassed ? `Enable ${node.label}` : `Bypass ${node.label}`}
                >
                  {isBypassed ? '⊘' : '⏻'}
                </button>

                <span className="font-mono text-micro text-white/50">{node.n}.</span>
                <span className={`font-medium ${isBypassed ? 'line-through' : ''}`}>
                  {node.label}
                </span>

                <span
                  data-testid={`flow-latency-${node.id}`}
                  className={`px-1.5 py-0.5 rounded text-micro font-mono tabular-nums border ${latencyTone(ms)}`}
                  title={`Kernel latency from /ws/telemetry: ${formatLatency(ms)}`}
                >
                  {formatLatency(ms)}
                </span>

                <button
                  type="button"
                  data-testid={`flow-solo-${node.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSoloStage?.(isSolo ? null : node.id);
                  }}
                  className={`px-1.5 py-0.5 rounded text-micro font-bold uppercase transition-all ${
                    isSolo
                      ? 'bg-amber-400 text-black shadow-sm font-black'
                      : 'bg-white/10 text-white/60 hover:bg-amber-400/20 hover:text-amber-200'
                  }`}
                  title={isSolo ? 'Solo Active: click to exit' : 'Solo intermediate buffer (Alt+Click)'}
                >
                  {isSolo ? 'SOLO' : 'S'}
                </button>
              </div>

              {i < MODULAR_FLOW_NODES.length - 1 && (
                <span aria-hidden="true" className="text-white/20 text-micro select-none px-0.5">
                  →
                </span>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* 1-Click Buffer Solo Bar: BiSeNet Parsing, XSeg 3 Mask, Occlusion Alpha, Depth/Z-Order */}
      <div
        data-testid="buffer-solo-strip"
        className="flex flex-wrap items-center gap-1.5 pt-1 border-t border-white/10"
      >
        <span className="text-micro font-bold uppercase tracking-wider text-[var(--accent)] px-1">
          Buffer Solo:
        </span>
        <div className="flex flex-wrap items-center gap-1" role="toolbar" aria-label="Intermediate buffer solo buttons">
          {BUFFER_SOLO_PASSES.map((pass) => {
            const active = isPassActive(pass);
            return (
              <button
                key={pass.id}
                type="button"
                data-testid={`solo-buffer-${pass.id}`}
                aria-pressed={active}
                onClick={() => onSoloStage?.(active ? null : pass.id)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-micro font-semibold transition-all ${
                  active
                    ? 'border-amber-400 bg-amber-400/25 text-white shadow-[0_0_10px_rgba(251,191,36,0.3)] ring-1 ring-amber-400/40'
                    : 'border-white/10 bg-white/[0.04] text-white/75 hover:border-white/25 hover:bg-white/[0.08] hover:text-white'
                }`}
                title={`${pass.description} · Click to ${active ? 'clear solo' : 'solo pass'}`}
              >
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ background: pass.color, opacity: active ? 1 : 0.6 }}
                />
                <span>{pass.label}</span>
                {active && (
                  <span className="px-1 py-0.2 rounded bg-amber-400 text-black text-nano font-black uppercase">
                    SOLO
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default VisualPipelineNodeFlowBar;
