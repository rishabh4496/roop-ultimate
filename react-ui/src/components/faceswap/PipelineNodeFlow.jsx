import React, { useCallback, useMemo, useRef, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useTelemetryStore } from '../../store/telemetryStore';
import { eventMatches } from '../../store/shortcutStore';
import { LiveValue } from '../LiveTelemetry';
import {
  FLOW_NODES, BUFFER_SOLO_PASSES, KEY_TO_SOLO, signalPath, nextSolo, isDiffSolo, latencyFor, latencyLevel,
  flowBadges, badgeCounts, nodeTitle, tallestFacePx, soloLabel,
} from './flowModel';
import { formatLatency } from './pillModel';

export { BUFFER_SOLO_PASSES };

// Stage 11: the Visual Pipeline Node Flow Bar.
//
//   Source -> Detector -> Swapper -> [Gaze & Mouth] -> XSeg 3 matte -> Restorer -> Colour / LUT -> Output
//
// One horizontal strip docked under the viewport. Each node is a real stage of /api/preview:
//   * Solo (S) / keys 1-6   shows that stage's intermediate buffer in the viewport (`solo_stage`), and a cyan signal path
//                           lights up to it; everything downstream is dimmed because it no longer feeds what you see.
//   * Δ                     (Restorer, Colour) the difference pass: what that stage changed, as a heat-map.
//   * bypass switch         switches the stage OFF in the preview (`bypass_stages`), server-side: the picture changes.
//   * latency badge         ms per delivered frame from /ws/telemetry, a leaf subscriber so a tick re-renders one badge.
//   * status pills          provider downgrade, TensorRT batching / engine build, the native <=320 px restorer bypass,
//                           lip-sync fallback, hardware I/O fallback - each derived from something the backend reported.
//
// Keys act while focus is inside the bar (hover a node and it takes focus when nothing else has it): registry scope 'flow',
// ids flow.solo1..6 / flow.soloClear / flow.solo / flow.diff / flow.bypass (shortcuts/actions.js), remappable in Settings.
// The handler stops the event, so 1-4 stay the snapshot slots everywhere else.

const SOLO_KEY_ACTIONS = Object.freeze([
  ['flow.solo1', 1], ['flow.solo2', 2], ['flow.solo3', 3], ['flow.solo4', 4], ['flow.solo5', 5], ['flow.solo6', 6],
]);

const LEVEL_CLASS = Object.freeze({
  ok: 'text-emerald-300 border-emerald-400/30 bg-emerald-500/10',
  warn: 'text-amber-300 border-amber-400/30 bg-amber-500/10',
  hot: 'text-rose-300 border-rose-400/35 bg-rose-500/10',
  none: 'text-white/70 border-white/15 bg-white/[0.04]',
});
const PILL_CLASS = Object.freeze({
  info: 'text-cyan-200 border-cyan-400/35 bg-cyan-400/10',
  warn: 'text-amber-200 border-amber-400/40 bg-amber-400/10',
  ok: 'text-emerald-200 border-emerald-400/30 bg-emerald-500/10',
});

const STORAGE_KEY = 'roop_flow_bar_collapsed_v1';
// The user's choice wins; with none, a short window starts collapsed (the Face Swap deck is ~190 px tall at 720p and the bar must
// not take the stage's room).
const readCollapsed = () => {
  let stored = null;
  try { stored = globalThis.localStorage?.getItem(STORAGE_KEY); } catch { stored = null; }
  if (stored === '1') return true;
  if (stored === '0') return false;
  return typeof window !== 'undefined' && Number(window.innerHeight) > 0 && window.innerHeight < 820;
};
const writeCollapsed = (v) => {
  try { globalThis.localStorage?.setItem(STORAGE_KEY, v ? '1' : '0'); } catch { /* private window: the bar still works */ }
};

/** One latency badge. A leaf subscriber: a telemetry tick re-renders this span, never the bar. */
function LatencyBadge({ node }) {
  if (!node.latencyKey) return null;
  return (
    <LiveValue select={(s) => latencyFor(s, node)}>
      {(ms) => (
        <span
          data-testid={`flow-latency-${node.id}`}
          data-ms={ms == null ? '' : String(ms)}
          className={`rounded border px-1.5 py-px font-mono text-micro tabular-nums ${LEVEL_CLASS[latencyLevel(ms)]}`}
          title={ms == null
            ? `${node.short}: nothing measured yet. Latency appears while a render runs.`
            : `${node.short}: ${formatLatency(ms)} per delivered frame (thread time, from /ws/telemetry)`}
        >
          {ms == null ? '— ms' : formatLatency(ms)}
        </span>
      )}
    </LiveValue>
  );
}

function Pill({ pill, nodeId }) {
  return (
    <span
      data-testid={`flow-pill-${nodeId}-${pill.id}`}
      className={`inline-flex items-center rounded-full border px-1.5 py-px text-nano font-bold leading-4 ${PILL_CLASS[pill.tone] || PILL_CLASS.info}`}
      title={pill.title}
    >
      {pill.tone === 'warn' && <span aria-hidden="true" className="mr-0.5">⚠</span>}
      {pill.text}
      <span className="sr-only">{`. ${pill.title}`}</span>
    </span>
  );
}

function FlowNode({
  node, state, title, pills, soloStage, bypassed, onPress, onSolo, onDiff, onBypass,
}) {
  const isSoloNode = state === 'solo';
  const diffActive = isSoloNode && isDiffSolo(soloStage);
  const stateWord = { solo: 'soloed', lit: 'feeds the viewport', dim: 'downstream of the solo', off: 'bypassed' }[state];
  return (
    <li
      data-flow-node={node.id}
      data-testid={`flow-node-${node.id}`}
      data-state={state}
      data-solo={isSoloNode ? 'true' : 'false'}
      data-bypassed={bypassed ? 'true' : 'false'}
      className={`flow-node group/node flex min-w-[196px] max-w-[260px] shrink-0 flex-col gap-1 rounded-lg border bg-white/[0.04] px-1.5 py-1 ${
        node.kind === 'sub' ? 'min-w-[176px]' : ''} ${node.kind === 'io' ? 'border-white/10' : 'border-white/15'}`}
    >
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          data-testid={`flow-main-${node.id}`}
          onClick={() => onPress(node)}
          aria-pressed={node.soloId || node.id === 'writer' ? isSoloNode : undefined}
          aria-label={`${node.n >= 1 && node.n <= 6 ? `${node.n}. ` : ''}${title}, ${stateWord}${
            node.n >= 1 && node.n <= 6 ? `. Press ${node.n} to solo` : ''}`}
          title={`${title}. ${node.hint}`}
          className="flex min-w-0 flex-1 items-center gap-1.5 rounded text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
        >
          <span
            aria-hidden="true"
            className={`grid h-4 w-4 shrink-0 place-items-center rounded-full font-mono text-nano font-bold ${
              isSoloNode ? 'bg-cyan-300 text-black' : 'bg-white/10 text-white/80'}`}
          >
            {node.n >= 1 && node.n <= 6 ? node.n : node.id === 'source' ? '▸' : '■'}
          </span>
          <span className={`truncate text-mini font-semibold ${bypassed ? 'line-through text-white/55' : 'text-white'}`}>{title}</span>
        </button>
        <LatencyBadge node={node} />
        {node.soloId && (
          <button
            type="button"
            tabIndex={-1}
            data-testid={`flow-solo-${node.id}`}
            aria-pressed={isSoloNode && !diffActive}
            onClick={(e) => { e.stopPropagation(); onSolo(node); }}
            className={`rounded px-1.5 py-px text-micro font-bold transition-colors ${
              isSoloNode && !diffActive
                ? 'bg-cyan-300 text-black shadow-[0_0_10px_rgba(34,211,238,0.6)]'
                : 'bg-white/10 text-white/85 hover:bg-cyan-400/20 hover:text-white'}`}
            title={isSoloNode && !diffActive ? 'Soloing: click to return to the live result' : `Solo (S) this buffer${node.n ? ` · key ${node.n}` : ''}`}
          >
            {isSoloNode && !diffActive ? 'Soloing ✕' : 'Solo (S)'}
          </button>
        )}
        {node.diffId && (
          <button
            type="button"
            tabIndex={-1}
            data-testid={`flow-diff-${node.id}`}
            aria-pressed={diffActive}
            onClick={(e) => { e.stopPropagation(); onDiff(node); }}
            className={`rounded px-1.5 py-px text-micro font-bold transition-colors ${
              diffActive ? 'bg-orange-300 text-black' : 'bg-white/10 text-white/85 hover:bg-orange-400/20 hover:text-white'}`}
            title={`${node.short} difference pass (D): what this stage changed, as a heat-map`}
          >
            Δ
          </button>
        )}
        {node.bypassable && (
          <button
            type="button"
            tabIndex={-1}
            role="switch"
            aria-checked={!bypassed}
            data-testid={`flow-bypass-${node.id}`}
            onClick={(e) => { e.stopPropagation(); onBypass(node); }}
            className={`relative h-4 w-7 shrink-0 rounded-full border transition-colors ${
              bypassed ? 'border-rose-400/50 bg-rose-500/25' : 'border-emerald-400/50 bg-emerald-500/30'}`}
            aria-label={`${bypassed ? 'Enable' : 'Bypass'} ${title}`}
            title={bypassed ? `${title} is bypassed in the preview: click to enable (B)` : `${title} is on: click to bypass it in the preview (B)`}
          >
            <span
              aria-hidden="true"
              className={`absolute top-px h-3 w-3 rounded-full bg-white shadow transition-all ${bypassed ? 'left-px' : 'left-[13px]'}`}
            />
          </button>
        )}
      </div>
      {pills.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          {pills.map((p) => <Pill key={p.id} pill={p} nodeId={node.id} />)}
        </div>
      )}
    </li>
  );
}

/**
 * Props
 *   soloStage        the solo id in force (or null)               onSoloStage(id|null)
 *   bypassedStages   { [nodeId]: true }                           onToggleBypass(nodeId)
 *   settings         the recipe: swap_model, selected_enhancer, ultramax_batch_size, auto_bypass_restorer_native_swap, ultramax_ramp
 *   provider         /api/meta `provider_status` ({ requested, active, degraded, degradation_reason }): the downgrade pill's source
 *   faces            the preview's detected boxes ([x1,y1,x2,y2]) - the native-bypass pill reads the tallest
 *   soloResult       { applied, note } from the last preview response, or null: a solo the backend refused is said, not shown
 */
export function VisualPipelineNodeFlowBar({
  soloStage = null,
  onSoloStage,
  bypassedStages = {},
  onToggleBypass,
  settings = {},
  provider = null,
  faces = [],
  soloResult = null,
  className = '',
}) {
  const [collapsed, setCollapsed] = useState(readCollapsed);
  const rootRef = useRef(null);

  // Primitives only, from the fields the live app really fills (/ws/telemetry `run`), so a tick re-renders the bar only when one changes.
  const live = useTelemetryStore(useShallow((s) => ({
    engineState: s.run?.engine_state || 'IDLE',
    engineModel: s.run?.engine_model || '',
    enginePrecision: s.run?.engine_precision || '',
    ioFallbacks: Number(s.run?.io_fallbacks) || 0,
    ioFallback: s.run?.io_fallback || '',
    ioEncode: s.run?.io_encode || '',
    mouthMode: s.run?.mouth_mode || '',
    mouthFallback: s.run?.mouth_fallback || '',
  })));

  const facePx = useMemo(() => tallestFacePx(faces), [faces]);
  const badges = useMemo(() => flowBadges({
    settings,
    run: {
      engine_state: live.engineState, engine_model: live.engineModel, engine_precision: live.enginePrecision,
      io_fallbacks: live.ioFallbacks, io_fallback: live.ioFallback, mouth_mode: live.mouthMode, mouth_fallback: live.mouthFallback,
    },
    provider,
    facePx,
  }), [settings, live, provider, facePx]);
  const counts = badgeCounts(badges);
  const path = useMemo(() => signalPath(soloStage, bypassedStages), [soloStage, bypassedStages]);

  const solo = useCallback((id) => onSoloStage?.(id), [onSoloStage]);

  // A press on a node's title: the Output returns to live, every other stage toggles its solo.
  const press = useCallback((node) => {
    if (node.id === 'writer') { solo(null); return; }
    if (node.soloId) solo(nextSolo(soloStage, node.id));
  }, [solo, soloStage]);
  const soloNode = useCallback((node) => solo(nextSolo(soloStage, node.id)), [solo, soloStage]);
  const diffNode = useCallback((node) => solo(nextSolo(soloStage, node.id, { diff: true })), [solo, soloStage]);
  const bypassNode = useCallback((node) => { if (node.bypassable) onToggleBypass?.(node.id); }, [onToggleBypass]);

  const onKeyDown = useCallback((e) => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target?.tagName) || e.isComposing) return;
    let handled = false;
    for (const [action, n] of SOLO_KEY_ACTIONS) {
      if (eventMatches(e, action)) { solo(nextSolo(soloStage, FLOW_NODES.find((x) => x.n === n).id)); handled = true; break; }
    }
    if (!handled && eventMatches(e, 'flow.soloClear')) { solo(null); handled = true; }
    if (!handled) {
      const id = e.target?.closest?.('[data-flow-node]')?.getAttribute('data-flow-node');
      const node = id ? FLOW_NODES.find((x) => x.id === id) : null;
      if (node) {
        if (eventMatches(e, 'flow.solo')) { if (node.soloId) solo(nextSolo(soloStage, node.id)); handled = true; }
        else if (eventMatches(e, 'flow.diff')) { if (node.diffId) solo(nextSolo(soloStage, node.id, { diff: true })); handled = true; }
        else if (eventMatches(e, 'flow.bypass')) { bypassNode(node); handled = true; }
      }
    }
    if (handled) {
      e.preventDefault();
      e.stopPropagation();          // keep 1-4 from also hitting the snapshot slots, S from toggling split view
    }
  }, [solo, soloStage, bypassNode]);

  // Hovering a node gives it keyboard focus, so "hover a node and press 4 / S / B" works - but ONLY when nothing else has focus or
  // focus is already inside the bar. A number typed into a slider, or a focused button elsewhere, is never stolen.
  const onPointerOver = useCallback((e) => {
    const node = e.target?.closest?.('[data-flow-node]');
    if (!node) return;
    const active = document.activeElement;
    const free = !active || active === document.body || active === document.documentElement;
    if (!free && !rootRef.current?.contains(active)) return;
    const main = node.querySelector('[data-testid^="flow-main-"]');
    if (main && main !== active) main.focus({ preventScroll: true });
  }, []);

  const toggleCollapsed = () => setCollapsed((c) => { writeCollapsed(!c); return !c; });
  const refused = soloStage && soloResult && soloResult.applied === false;

  return (
    <div
      ref={rootRef}
      role="toolbar"
      data-testid="visual-pipeline-node-flow-bar"
      aria-label="Pipeline flow: solo an intermediate buffer, bypass a stage"
      aria-orientation="horizontal"
      tabIndex={-1}
      onKeyDown={onKeyDown}
      onPointerOver={onPointerOver}
      className={`rounded-xl border border-white/10 bg-black/45 px-2 py-1 backdrop-blur-sm ${className}`}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-1">
        <button
          type="button"
          data-testid="flow-collapse"
          aria-expanded={!collapsed}
          onClick={toggleCollapsed}
          className="flex items-center gap-1.5 rounded text-micro font-bold uppercase tracking-wider text-white/80 hover:text-white"
          title={collapsed ? 'Show the pipeline flow' : 'Collapse the pipeline flow'}
        >
          <span aria-hidden="true">{collapsed ? '▸' : '▾'}</span>
          Pipeline flow
        </button>
        <span className="text-micro text-white/60">
          {soloStage
            ? <>Viewport: <strong data-testid="flow-solo-label" className="text-cyan-200">{soloLabel(soloStage)}</strong></>
            : 'Viewport: live result'}
        </span>
        {counts.warn > 0 && (
          <span data-testid="flow-warn-count" className="rounded-full border border-amber-400/40 bg-amber-400/10 px-1.5 text-nano font-bold text-amber-200">
            ⚠ {counts.warn}
          </span>
        )}
        {!collapsed && (
          <div data-testid="buffer-solo-strip" role="toolbar" aria-label="Inspection passes" className="flex flex-wrap items-center gap-1">
            <span className="text-micro font-bold uppercase tracking-wider text-white/70">Passes</span>
            {BUFFER_SOLO_PASSES.map((pass) => {
              const active = soloStage === pass.id || soloStage === pass.key || soloStage === pass.alias;
              return (
                <button
                  key={pass.id}
                  type="button"
                  data-testid={`solo-buffer-${pass.id}`}
                  aria-pressed={active}
                  aria-disabled={!pass.available}
                  disabled={!pass.available}
                  onClick={() => solo(active ? null : pass.id)}
                  title={pass.description}
                  className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-px text-micro font-semibold transition-colors ${
                    active ? 'border-cyan-300 bg-cyan-400/20 text-white shadow-[0_0_10px_rgba(34,211,238,0.35)]'
                      : pass.available ? 'border-white/15 bg-white/[0.04] text-white/85 hover:border-white/30 hover:text-white'
                        : 'cursor-not-allowed border-dashed border-white/10 bg-transparent text-white/45'}`}
                >
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: pass.color, opacity: pass.available ? 1 : 0.4 }} />
                  {pass.shortLabel}
                  {!pass.available && <span className="text-nano uppercase">n/a</span>}
                </button>
              );
            })}
          </div>
        )}
        <span className="flex-1" />
        <span className="hidden text-micro text-white/50 sm:inline">1–6 solo · 0 live · S solo · D diff · B bypass</span>
        {soloStage && (
          <button
            type="button"
            data-testid="flow-clear-solo"
            onClick={() => solo(null)}
            className="rounded border border-cyan-400/40 bg-cyan-400/10 px-2 py-px text-micro font-bold text-cyan-100 hover:bg-cyan-400/20"
            title="Return the viewport to the live result (0 or Esc)"
          >
            Back to live ✕
          </button>
        )}
      </div>

      {refused && (
        <div role="status" data-testid="flow-solo-refused" className="mx-1 mt-1 rounded-lg border border-amber-400/40 bg-amber-400/10 px-2 py-1 text-micro text-amber-100">
          <strong>{soloLabel(soloStage)}:</strong> {soloResult.note || 'this buffer could not be produced; the viewport shows the live result.'}
        </div>
      )}

      {!collapsed && (
        <>
          <ol
            aria-label="Pipeline stages, in order"
            className="m-0 mt-1.5 flex list-none items-stretch gap-0 overflow-x-auto p-1 pb-1.5"
            data-testid="flow-rail"
          >
            {FLOW_NODES.map((node, i) => (
              <React.Fragment key={node.id}>
                <FlowNode
                  node={node}
                  state={path.nodes[i]}
                  title={nodeTitle(node, settings, { io_encode: live.ioEncode })}
                  pills={badges[node.id]}
                  soloStage={soloStage}
                  bypassed={Boolean(bypassedStages[node.id])}
                  onPress={press}
                  onSolo={soloNode}
                  onDiff={diffNode}
                  onBypass={bypassNode}
                />
                {i < FLOW_NODES.length - 1 && (
                  <li aria-hidden="true" role="presentation" className="flex flex-1 items-center">
                    <span className="flow-edge" data-testid={`flow-edge-${i}`} data-state={path.edges[i]} />
                  </li>
                )}
              </React.Fragment>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}

export default VisualPipelineNodeFlowBar;
export { KEY_TO_SOLO };
