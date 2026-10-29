import React, { useEffect, useMemo, useRef, useState } from 'react';
import NumericInput from '../../NumericInput';
import { useLaneCanvas } from '../timeline/useLaneCanvas';
import { frameAtX, normalizeView, viewSpan } from '../timeline/viewMath';
import {
  CURVE_PARAMS, DEFAULT_INTERP, INTERPS, INTERP_LABELS, clampValue, curveIneffective, keyFrameBounds, makeEvaluator, paramById,
} from './curveModel';

// The Curve Editor: one parameter's value against the frame, on the timeline's own x-axis.
//
// It sits in the lane stack, so it shares the stack's width and the timeline's view window: zoom or scroll the
// timeline and the graph follows, key for key. The line is drawn on a canvas (it can be thousands of pixels of
// samples), the keyframes are real buttons over it (at most MAX_KEYS = 128, so they are keyboard- and
// screen-reader-reachable, which a canvas is not). Positions are percentages of the graph in x and fixed layout
// pixels in y, so the app-level CSS zoom cannot offset them; pointer positions are turned into frames and values
// from ratios of the graph's own rect for the same reason.
//
// Gestures: click the graph to park the playhead; double-click to key the value under the pointer; drag a key (a
// ghost follows the pointer and the move commits on release, so one drag is one undo step); with a key focused,
// Left / Right nudge a frame (Shift: ten), Up / Down nudge the value (Shift: ten steps), Delete removes it.

const EMPTY = Object.freeze([]);
const GRAPH_H = 168;
const PAD_Y = 14;
const HANDLE = 12;
const smallBtn = 'rounded-md px-1.5 py-0.5 text-note font-semibold text-[var(--text-muted)] hover:bg-white/[0.06] hover:text-[var(--text-main)] disabled:opacity-40 transition-colors';

const yOf = (param, v) => PAD_Y + (1 - (v - param.min) / (param.max - param.min)) * (GRAPH_H - 2 * PAD_Y);
const valueAtY = (param, yPx) => param.min + (1 - (yPx - PAD_Y) / (GRAPH_H - 2 * PAD_Y)) * (param.max - param.min);
const fmtValue = (param, v) => `${Number(v).toFixed(param.digits)}${param.unit}`;

/** A segment-shape glyph for the interpolation buttons. */
const InterpGlyph = ({ kind }) => {
  const d = { hold: 'M2 14 H10 V4 H18', linear: 'M2 14 L18 4', ease: 'M2 14 C9 14 11 4 18 4', smooth: 'M2 14 C7 14 7 9 10 9 C13 9 13 4 18 4' }[kind];
  return (
    <svg width="20" height="18" viewBox="0 0 20 18" aria-hidden="true">
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
};

export default function CurveEditor({
  view, maxFrames, frame, tracks, enabled, onEnabledChange, settings, api, onSeek, onClose, morph = null, planNote = '',
}) {
  const v = normalizeView(view, maxFrames);
  const span = viewSpan(v);
  const [paramId, setParamId] = useState(() => CURVE_PARAMS.find((p) => tracks?.[p.id]?.length)?.id || CURVE_PARAMS[0].id);
  const param = paramById(paramId);
  const list = tracks?.[paramId] || EMPTY;
  const [sel, setSel] = useState(null);                       // the selected key's frame
  const [ghost, setGhost] = useState(null);                   // { frame, toFrame, toValue } while a key is dragged
  const graphRef = useRef(null);
  const dragRef = useRef(null);
  const selKey = list.find((k) => k.frame === sel) || null;
  const selIsLast = selKey && list[list.length - 1] === selKey;

  // A selection that no longer exists (undo, another parameter, a removed key) is dropped, not left dangling.
  useEffect(() => { if (sel !== null && !list.some((k) => k.frame === sel)) setSel(null); }, [list, sel]);

  const baseValue = Number.isFinite(Number(settings?.[param.key])) ? Number(settings[param.key]) : param.base;
  // What is drawn: the list, with the dragged key at the pointer. `orig` is the key's frame before the drag - the React
  // key - so the button under the pointer is never remounted mid-drag (which would drop its pointer capture).
  const shown = useMemo(() => list.map((k) => (ghost && k.frame === ghost.frame
    ? { ...k, orig: k.frame, frame: ghost.toFrame, value: ghost.toValue } : { ...k, orig: k.frame })), [list, ghost]);
  const evalShown = useMemo(() => makeEvaluator(shown), [shown]);

  const canvasRef = useLaneCanvas((ctx, { w, h, theme }) => {
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.textBaseline = 'middle';
    // Grid: five value lines with their labels.
    for (let i = 0; i <= 4; i++) {
      const val = param.min + ((param.max - param.min) * i) / 4;
      const y = yOf(param, val);
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke();
      ctx.fillStyle = theme.muted;
      ctx.fillText(fmtValue(param, val), 4, Math.min(h - 6, Math.max(6, y - 6)));
    }
    // The clip's own setting: where the parameter sits when no curve is driving it.
    const by = yOf(param, clampValue(param, baseValue));
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = theme.muted;
    ctx.beginPath(); ctx.moveTo(0, by + 0.5); ctx.lineTo(w, by + 0.5); ctx.stroke();
    ctx.restore();
    if (!shown.length) return;
    // The curve: one sample per 2 px (a step shows as a near-vertical edge), faint outside the first / last key,
    // where the value is only held.
    const first = shown[0].frame;
    const last = shown[shown.length - 1].frame;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    let prevOut = null;
    let path = null;
    let lx = 0;
    let ly = 0;
    const flush = (out) => {
      if (path) { ctx.strokeStyle = param.color; ctx.globalAlpha = out ? 0.4 : 1; ctx.stroke(path); }
    };
    for (let x = 0; x <= w; x += 2) {
      const f = v.start + (x / Math.max(1, w)) * span;
      const out = f < first || f > last;
      const y = yOf(param, evalShown(f));
      if (out !== prevOut) {
        flush(prevOut);
        path = new Path2D();
        path.moveTo(prevOut === null ? x : lx, prevOut === null ? y : ly);      // join onto the previous stretch
        path.lineTo(x, y);
        prevOut = out;
      } else path.lineTo(x, y);
      lx = x; ly = y;
    }
    flush(prevOut);
    ctx.globalAlpha = 1;
  }, [v.start, v.end, param, shown, evalShown, baseValue, enabled]);

  const pct = (f) => ((f - v.start) / span) * 100;
  const inView = (f) => f >= v.start - 1 && f <= v.end + 1;

  const pointer = (e) => {
    const r = graphRef.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return null;
    return {
      frame: frameAtX(e.clientX - r.left, r.width, v, maxFrames),
      value: clampValue(param, valueAtY(param, ((e.clientY - r.top) / r.height) * GRAPH_H)),
    };
  };

  // A first key keeps the value the slider has, so adding one changes nothing by itself.
  const keyHere = (f, value) => {
    const out = api.add(paramId, f, value === undefined && !list.length ? clampValue(param, baseValue) : value);
    if (out?.created) setSel(out.frame);
    else if (out?.full) return false;
    return out;
  };

  const onGraphClick = (e) => { const p = pointer(e); if (p) onSeek?.(p.frame); };
  const onGraphDouble = (e) => {
    const p = pointer(e);
    if (p) keyHere(p.frame, p.value);
  };

  const down = (k) => (e) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    dragRef.current = { frame: k.frame, moved: false, x: e.clientX, y: e.clientY };
    setSel(k.frame);
    onSeek?.(k.frame);
  };
  const drag = (e) => {
    const d = dragRef.current;
    if (!d || !(e.buttons & 1)) return;
    if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 3) return;      // a click is not a drag
    d.moved = true;
    const p = pointer(e);
    const b = keyFrameBounds(list, d.frame, maxFrames);
    if (!p || !b) return;
    setGhost({ frame: d.frame, toFrame: Math.max(b[0], Math.min(b[1], p.frame)), toValue: p.value });
  };
  // The final position comes from the release event itself, not from the last ghost: a quick flick can deliver the
  // release before React has re-rendered with the newest ghost.
  const up = (e) => {
    const d = dragRef.current;
    dragRef.current = null;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setGhost(null);
    if (!d?.moved) return;
    const p = pointer(e);
    const b = keyFrameBounds(list, d.frame, maxFrames);
    if (!p || !b) return;
    const to = Math.max(b[0], Math.min(b[1], p.frame));
    if (api.move(paramId, d.frame, to, p.value)) setSel(to);
  };

  const nudge = (k) => (e) => {
    const big = e.shiftKey ? 10 : 1;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      const to = k.frame + (e.key === 'ArrowLeft' ? -big : big);
      if (api.move(paramId, k.frame, to, k.value)) {
        const b = keyFrameBounds(list, k.frame, maxFrames);
        const landed = b ? Math.max(b[0], Math.min(b[1], to)) : to;
        setSel(landed);
        requestAnimationFrame(() => graphRef.current?.querySelector(`[data-testid="curve-key-${landed}"]`)?.focus());
      }
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      api.setValue(paramId, k.frame, k.value + (e.key === 'ArrowUp' ? 1 : -1) * big * param.quantum);
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      api.remove(paramId, k.frame);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setSel(k.frame);
      onSeek?.(k.frame);
    }
  };


  const ineffective = list.length && enabled ? curveIneffective(param, settings) : null;
  const atPlayhead = list.some((k) => k.frame === frame);

  return (
    <div className="space-y-1.5" data-testid="curve-editor" role="group" aria-label="Parameter curve editor">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 pt-1.5">
        <span className="text-nano font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Curve editor</span>
        <div className="flex flex-wrap items-center gap-0.5" role="tablist" aria-label="Parameter">
          {CURVE_PARAMS.map((p) => {
            const on = p.id === paramId;
            const n = tracks?.[p.id]?.length || 0;
            return (
              <button key={p.id} type="button" role="tab" aria-selected={on} data-testid={`curve-param-${p.id}`}
                      onClick={() => { setParamId(p.id); setSel(null); setGhost(null); }}
                      className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-note font-semibold transition-colors ${
                        on ? 'bg-[var(--accent)]/15 text-accent' : 'text-[var(--text-muted)] hover:bg-white/[0.06] hover:text-[var(--text-main)]'}`}>
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: p.color, opacity: n ? 1 : 0.35 }} />
                {p.label}
                {n > 0 && <span className="font-mono tabular-nums opacity-70">{n}</span>}
              </button>
            );
          })}
        </div>
        <span className="flex-1" />
        <label className="inline-flex items-center gap-1 text-note text-[var(--text-muted)]"
               title="Off: the curves stay on the timeline, but the preview and the render use the clip's own settings">
          <input type="checkbox" checked={enabled} data-testid="curve-enabled" aria-label="Apply parameter curves"
                 onChange={(e) => onEnabledChange?.(e.target.checked)} className="accent-[var(--accent)]" />
          Apply
        </label>
        <button type="button" className={smallBtn} data-testid="curve-add-key" disabled={atPlayhead}
                onClick={() => keyHere(frame, undefined)}
                title={atPlayhead ? 'There is already a keyframe at the playhead' : `Key ${param.label} at the playhead (it keeps the value the curve - or the slider - has there)`}>
          + Key at playhead
        </button>
        <button type="button" className={smallBtn} data-testid="curve-clear" disabled={!list.length}
                onClick={() => { api.clearParam(paramId); setSel(null); }} title={`Remove every ${param.label} keyframe on this clip`}>
          Clear curve
        </button>
        <button type="button" className={smallBtn} data-testid="curve-close" onClick={onClose} title="Close the curve editor (C)">Close</button>
      </div>

      {/* Interpolation of the segment AFTER the selected key, and its exact value */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-note text-[var(--text-muted)]">
          {selKey ? `Key at frame ${selKey.frame} - the segment after it:` : 'Select a key to set how the curve leaves it:'}
        </span>
        <div className="inline-flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5" role="group" aria-label="Interpolation">
          {INTERPS.map((kind) => (
            <button key={kind} type="button" data-testid={`curve-interp-${kind}`}
                    disabled={!selKey || selIsLast} aria-pressed={selKey?.interp === kind}
                    onClick={() => api.interp(paramId, selKey.frame, kind)}
                    title={selIsLast ? 'The last key has no segment after it' : INTERP_LABELS[kind]}
                    aria-label={INTERP_LABELS[kind]}
                    className={`grid h-6 w-7 place-items-center rounded-md transition-colors disabled:opacity-30 ${
                      selKey?.interp === kind ? 'bg-[var(--accent)] text-white' : 'text-[var(--text-muted)] hover:bg-white/[0.06] hover:text-[var(--text-main)]'}`}>
              <InterpGlyph kind={kind} />
            </button>
          ))}
        </div>
        {selKey && !selIsLast && <span className="text-note text-[var(--text-muted)]">{INTERP_LABELS[selKey.interp]}</span>}
        <span className="flex-1" />
        {selKey && (
          <NumericInput
            label="Value"
            testId="curve-value"
            ariaLabel={`${param.label} value at the selected key`}
            value={selKey.value}
            min={param.min}
            max={param.max}
            step={param.quantum}
            unit={param.unit}
            onCommit={(n) => api.setValue(paramId, selKey.frame, n)}
            className="gap-1.5 text-note text-[var(--text-muted)]"
            inputClassName="w-[8ch] rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] py-0.5 text-center font-mono text-mini tabular-nums text-[var(--text-main)] outline-none focus:border-[var(--accent)]"
          />
        )}
      </div>

      <div
        ref={graphRef}
        data-testid="curve-graph"
        className="relative w-full rounded-md border border-[var(--border-color)] bg-[var(--input-bg)]"
        style={{ height: GRAPH_H, opacity: enabled ? 1 : 0.55 }}
      >
        {/* Not clipped: the x-axis must be exactly the lanes' (frame 1 is the left edge), so a key on the first or
            last frame sits ON the edge and its handle has to overhang it rather than lose half to overflow. */}
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full rounded-md" aria-hidden="true" />
        {/* The pointer layer: click parks the playhead, double-click keys the value under the pointer. A mouse
            affordance only, so it is presentation; the keyboard routes are "+ Key at playhead" and the key buttons
            themselves (focus one: arrows nudge, Delete removes). */}
        <div role="presentation" data-testid="curve-hit" className="absolute inset-0" style={{ cursor: 'copy' }}
             onClick={onGraphClick} onDoubleClick={onGraphDouble} />
        {!list.length && (
          <span className="pointer-events-none absolute inset-x-0 top-2 px-3 text-center text-note text-[var(--text-muted)]">
            No keyframes: {param.label} follows the slider ({fmtValue(param, baseValue)}). Double-click to key a value here.
          </span>
        )}
        {shown.map((k) => {
          if (!inView(k.frame)) return null;
          const on = sel === k.orig;
          return (
            <button key={k.orig} type="button"
                    data-testid={`curve-key-${k.frame}`}
                    aria-label={`${param.label} keyframe at frame ${k.frame}, value ${fmtValue(param, k.value)}, ${INTERP_LABELS[k.interp || DEFAULT_INTERP]} to the next`}
                    aria-pressed={!!on}
                    title={`Frame ${k.frame} - ${fmtValue(param, k.value)}. Drag to move, arrows nudge, Delete removes.`}
                    onPointerDown={down(k)} onPointerMove={drag} onPointerUp={up}
                    onPointerCancel={() => { dragRef.current = null; setGhost(null); }}
                    onKeyDown={nudge(k)}
                    onClick={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}
                    className={`absolute touch-none rounded-full border outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] ${
                      on ? 'border-white' : 'border-black/60 hover:border-white'}`}
                    style={{
                      left: `${pct(k.frame)}%`, top: yOf(param, k.value), width: HANDLE, height: HANDLE,
                      transform: 'translate(-50%, -50%)', background: param.color, cursor: 'grab',
                      boxShadow: on ? `0 0 0 3px ${param.color}55` : 'none',
                    }} />
          );
        })}
      </div>

      {ineffective && <p className="text-note text-amber-300" role="status" data-testid="curve-ineffective">{ineffective}</p>}
      {param.identity && morph && (
        <div className="flex flex-wrap items-center gap-2 text-note text-[var(--text-muted)]" data-testid="curve-morph">
          <span className="min-w-0 flex-1">{morph.note}</span>
          <button type="button" className={smallBtn} data-testid="curve-morph-apply" disabled={!morph.ready || !list.length}
                  onClick={morph.onApply} title={morph.title}>
            Write to Identity Keyframes
          </button>
        </div>
      )}
      {!param.identity && planNote && list.length > 0 && enabled && (
        <p className="text-note text-[var(--text-muted)]" data-testid="curve-plan">{planNote}</p>
      )}
      <p className="text-note text-[var(--text-muted)]">
        The preview follows the curve exactly. Dashed line: the clip&apos;s own {param.label} setting, used wherever there is no curve.
      </p>
    </div>
  );
}
