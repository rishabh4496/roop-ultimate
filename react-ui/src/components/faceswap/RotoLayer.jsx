import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import * as R from './rotoModel';

// The Pen tool (P): vector exclusion / inclusion zones drawn over the preview.
//
// RotoLayer is the SVG layer. It always DRAWS the clip's zones, and only takes pointer events while the
// tool is active. While active it owns the pointer completely (it stops propagation so the stage's pan /
// brush gesture never sees it) and edits a LOCAL draft during a drag, committing once on pointer-up: the
// parent's preview refreshes on every committed change, and a drag would otherwise re-request the preview
// on every mouse move.
//
// Coordinates are frame pixels. The SVG fills the same box the face overlay does (preserveAspectRatio
// none), so a pointer maps linearly: (client - rect) / rect * size, which is already correct under the
// stage's CSS zoom and pan because getBoundingClientRect reports the transformed box.

const { MODE_COLOR, rangeKind, rangeFor } = R;
const HIT_PX = 9;                 // pick tolerance in SCREEN pixels
const DRAG_PX = 3;                // a press that moves less than this is a click, not a handle drag

const replaceZone = (zones, next) => zones.map((z) => (z.id === next.id ? next : z));

export default function RotoLayer({
  imgDim, zones, onChange, selectedId, onSelect, active, frame, faces = [], newMode = 'exclude', zoom = 1,
}) {
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const [draft, setDraft] = useState(null);       // zones while a drag is in flight
  const [cursor, setCursor] = useState(null);     // media-px pointer, for the open zone's rubber band
  const [selPoint, setSelPoint] = useState(-1);
  const [px, setPx] = useState(1);                // media pixels per screen pixel
  const shown = draft || zones;
  const liveRef = useRef({});
  liveRef.current = { shown, zones, onChange, onSelect, selectedId, selPoint, active };

  useLayoutEffect(() => {
    const r = svgRef.current?.getBoundingClientRect();
    if (r && r.width && imgDim?.w) setPx(imgDim.w / r.width);
  }, [imgDim, zoom, active, zones.length]);

  const toMedia = useCallback((e) => {
    const r = svgRef.current.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / Math.max(r.width, 1)) * imgDim.w,
      y: ((e.clientY - r.top) / Math.max(r.height, 1)) * imgDim.h,
      scale: imgDim.w / Math.max(r.width, 1),
    };
  }, [imgDim]);

  const commit = useCallback((next) => {
    setDraft(null);
    liveRef.current.onChange(next);
  }, []);

  const onPointerDown = (e) => {
    if (!active || e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    svgRef.current?.focus({ preventScroll: true });   // so Enter / Escape / Delete reach this layer
    const m = toMedia(e);
    const tol = HIT_PX * m.scale;
    const zs = shown;
    const sel = zs.find((z) => z.id === selectedId) || null;
    const begin = (d) => { dragRef.current = d; try { svgRef.current.setPointerCapture(e.pointerId); } catch { /* synthetic event */ } };

    if (sel) {
      const hh = R.hitHandle(sel, m.x, m.y, tol);
      if (hh) { begin({ kind: 'handle', id: sel.id, ...hh }); return; }
      const pi = R.hitPoint(sel, m.x, m.y, tol);
      if (pi >= 0) {
        if (!sel.closed && pi === 0 && sel.points.length >= 3) { commit(replaceZone(zs, R.closeZone(sel))); setSelPoint(-1); return; }
        setSelPoint(pi);
        begin({ kind: 'point', id: sel.id, i: pi });
        return;
      }
      if (!sel.closed) {
        const next = R.addPoint(sel, m.x, m.y);
        setDraft(replaceZone(zs, next));
        setSelPoint(next.points.length - 1);
        begin({ kind: 'new', id: sel.id, i: next.points.length - 1, ox: m.x, oy: m.y });
        return;
      }
      const hs = R.hitSegment(sel, m.x, m.y, tol);
      if (hs) {
        const next = R.insertPoint(sel, hs.seg, hs.t);
        const at = hs.seg + 1;
        setDraft(replaceZone(zs, next));
        setSelPoint(at);
        begin({ kind: 'point', id: sel.id, i: at });
        return;
      }
    }
    const hit = [...zs].reverse().find((z) => z.closed && R.activeAt(z, frame)
      && (R.pointInPolygon(R.flatten(z.points), m.x, m.y) || R.hitSegment(z, m.x, m.y, tol)));
    if (hit) { onSelect(hit.id); setSelPoint(-1); return; }
    const created = R.addPoint(R.newZone({ mode: newMode }), m.x, m.y);
    onChange([...zs, created]);
    onSelect(created.id);
    setSelPoint(0);
  };

  const onPointerMove = (e) => {
    if (!active) return;
    const m = toMedia(e);
    setCursor({ x: m.x, y: m.y });
    const d = dragRef.current;
    if (!d) return;
    const base = liveRef.current.shown;
    const z = base.find((q) => q.id === d.id);
    if (!z) return;
    let next = z;
    if (d.kind === 'point') next = R.movePoint(z, d.i, m.x, m.y);
    else if (d.kind === 'handle') {
      const p = z.points[d.i];
      next = R.setHandle(z, d.i, d.which, m.x - p.x, m.y - p.y, !e.altKey);
    } else if (d.kind === 'new') {
      // press-and-drag after placing a point pulls out a smooth Bezier handle
      if (Math.hypot(m.x - d.ox, m.y - d.oy) >= DRAG_PX * m.scale) next = R.setHandle(z, d.i, 'hout', m.x - d.ox, m.y - d.oy, true);
    }
    if (next !== z) setDraft(replaceZone(base, next));
  };

  const endDrag = (e) => {
    const d = dragRef.current;
    dragRef.current = null;
    try { svgRef.current.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
    if (!d) return;
    const L = liveRef.current;
    commit(L.shown);
  };

  // Keys belong to the LAYER, not to `window`. FaceSwap.jsx already binds Enter and Escape on window, so a
  // window listener here fired both on every press (test_ui_shortcut_keys pins that no two components claim
  // a key). The layer is focusable while the tool is on and takes focus on every press and when the tool
  // opens; stopPropagation keeps the same key from reaching FaceSwap's own handler.
  const onKeyDown = (e) => {
    if (!active) return;
    const L = liveRef.current;
    const sel = L.shown.find((z) => z.id === L.selectedId);
    if (!sel) return;
    if (e.key === 'Enter') {
      if (!sel.closed && sel.points.length >= 3) {
        e.preventDefault(); e.stopPropagation();
        L.onChange(replaceZone(L.shown, R.closeZone(sel))); setSelPoint(-1);
      }
    } else if (e.key === 'Escape') {
      if (!sel.closed) {
        e.preventDefault(); e.stopPropagation();
        L.onChange(L.shown.filter((z) => z.id !== sel.id)); L.onSelect(null); setSelPoint(-1);
      }
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault(); e.stopPropagation();
      if (L.selPoint >= 0 && sel.points.length > 3) {
        L.onChange(replaceZone(L.shown, R.removePoint(sel, L.selPoint)));
      } else {
        L.onChange(L.shown.filter((z) => z.id !== sel.id));
        L.onSelect(null);
      }
      setSelPoint(-1);
    }
  };

  useEffect(() => {
    if (active) svgRef.current?.focus({ preventScroll: true });
  }, [active]);

  const rendered = useMemo(() => shown.filter((z) => z.enabled !== false).map((z) => {
    const live = R.activeAt(z, frame);
    const { outline, tracked } = z.closed ? R.placedOutline(z, faces) : { outline: R.flatten(z.points, R.BEZIER_STEPS, false), tracked: false };
    return { z, live, outline, tracked };
  }), [shown, frame, faces]);

  if (!imgDim) return null;
  const k = px;                                   // media px per screen px, to size knobs on screen
  const sw = (n) => n * k;                        // a screen-pixel width in media units (no vector-effect needed)
  const sel = shown.find((z) => z.id === selectedId) || null;

  // A drawing surface is pointer-driven by nature, so it is not a native control. It is still operable
  // without a pointer where that is possible: it takes focus, Enter / Escape / Delete work on it, and the
  // toolbar's Close / Delete / Done are real buttons. Scoped to this element only.
  /* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */
  return (
    <svg
      ref={svgRef}
      data-testid="roto-layer"
      data-active={active ? 'true' : 'false'}
      className="absolute inset-0 w-full h-full z-[36]"
      viewBox={`0 0 ${imgDim.w} ${imgDim.h}`}
      preserveAspectRatio="none"
      style={{ pointerEvents: active ? 'auto' : 'none', cursor: active ? 'crosshair' : 'default', touchAction: 'none', outline: 'none' }}
      role="application"
      aria-label="Pen tool canvas. Click to place points, drag to curve, Enter to close, Escape to cancel, Delete to remove."
      tabIndex={active ? 0 : -1}
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onPointerLeave={() => setCursor(null)}
      aria-hidden={active ? undefined : true}
    >
      {rendered.map(({ z, live, outline, tracked }) => {
        if (outline.length < 2) return null;
        const color = MODE_COLOR[z.mode] || MODE_COLOR.exclude;
        const d = outline.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ') + (z.closed ? ' Z' : '');
        const isSel = z.id === selectedId;
        return (
          <g key={z.id} data-zone-id={z.id} data-mode={z.mode} data-closed={z.closed ? 'true' : 'false'}
             data-live={live ? 'true' : 'false'} data-tracked={tracked ? 'true' : 'false'} opacity={live ? 1 : 0.4}>
            <path d={d} fill={z.closed ? color : 'none'} fillOpacity={0.16} stroke={color}
                  strokeWidth={sw(isSel ? 2.4 : 1.6)} strokeDasharray={live ? undefined : `${sw(5)} ${sw(4)}`}
                  strokeLinejoin="round" />
            {z.closed && z.feather > 0 && (
              // The soft edge: the matte ramps over about 1.3 x the feather around the outline.
              <path d={d} fill="none" stroke={color} strokeOpacity={0.12} strokeWidth={z.feather * 1.28}
                    strokeLinejoin="round" data-testid="roto-feather-band" />
            )}
            {z.pin && (
              <circle cx={outline[0][0]} cy={outline[0][1]} r={sw(4)} fill={tracked ? '#38bdf8' : 'none'}
                      stroke="#38bdf8" strokeWidth={sw(1.5)} data-testid="roto-pin-marker" />
            )}
          </g>
        );
      })}

      {active && sel && !sel.closed && cursor && sel.points.length > 0 && (
        <line x1={sel.points[sel.points.length - 1].x} y1={sel.points[sel.points.length - 1].y}
              x2={cursor.x} y2={cursor.y} stroke={MODE_COLOR[sel.mode]} strokeWidth={sw(1.2)}
              strokeDasharray={`${sw(4)} ${sw(3)}`} opacity={0.8} />
      )}

      {active && sel && sel.points.map((p, i) => (
        <g key={i} data-testid="roto-knob" data-point={i}>
          {['hin', 'hout'].map((w) => p[w] && (
            <g key={w}>
              <line x1={p.x} y1={p.y} x2={p.x + p[w][0]} y2={p.y + p[w][1]} stroke="#cbd5e1" strokeWidth={sw(1)} opacity={0.8} />
              <rect x={p.x + p[w][0] - sw(3)} y={p.y + p[w][1] - sw(3)} width={sw(6)} height={sw(6)} fill="#0f172a" stroke="#cbd5e1" strokeWidth={sw(1)} />
            </g>
          ))}
          <circle cx={p.x} cy={p.y} r={sw(i === 0 && !sel.closed && sel.points.length >= 3 ? 7 : 5)}
                  fill={i === selPoint ? '#fde047' : '#0f172a'} stroke="#fff" strokeWidth={sw(1.5)} />
        </g>
      ))}
    </svg>
  );
  /* eslint-enable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */
}

// ── toolbar ─────────────────────────────────────────────────────────────────

const RANGES = [
  ['clip', 'Whole clip'],
  ['frame', 'This frame'],
  ['from', 'From here on'],
  ['until', 'Up to here'],
];

export function RotoToolbar({
  zones, selectedId, onSelect, onChange, newMode, setNewMode, frame, maxFrames, faces = [], onDone,
}) {
  const sel = zones.find((z) => z.id === selectedId) || null;
  const update = (fn) => { if (sel) onChange(zones.map((z) => (z.id === sel.id ? fn(z) : z))); };

  // The face a new pin would follow: the one whose box centre is nearest the zone's centre.
  const pinTarget = useMemo(() => {
    if (!sel || !faces.length || !sel.points.length) return null;
    const cx = sel.points.reduce((a, p) => a + p.x, 0) / sel.points.length;
    const cy = sel.points.reduce((a, p) => a + p.y, 0) / sel.points.length;
    return faces.reduce((best, f) => {
      const bx = f.bbox ? (f.bbox[0] + f.bbox[2]) / 2 : f.nose[0];
      const by = f.bbox ? (f.bbox[1] + f.bbox[3]) / 2 : f.nose[1];
      const d = Math.hypot(bx - cx, by - cy);
      return !best || d < best.d ? { f, d } : best;
    }, null)?.f || null;
  }, [sel, faces]);

  const btn = 'px-2 py-1 rounded-lg text-micro font-bold transition-colors';
  // Click-through HUD: the wrapper passes the pointer to the picture, only the pills catch it. A solid
  // bar wrapped to four rows in a 420 px stage and covered the top of the frame it was meant to draw on.
  const pill = 'pointer-events-auto flex items-center gap-1.5 px-2 py-1 rounded-xl bg-black/85 backdrop-blur-md border border-white/15 shadow-2xl';
  return (
    <div data-testid="roto-toolbar"
         className="absolute top-2 left-1/2 -translate-x-1/2 z-50 flex max-w-[98%] flex-wrap items-start justify-center gap-1.5 pointer-events-none">
      <div className={pill}>
        <span className="text-nano font-mono text-white/60 uppercase font-bold"
              title="Click to place points - drag after a click to curve - click the first point or press Enter to close - Esc cancels - Delete removes - Alt while dragging a handle breaks it">
          Pen
        </span>
        {['exclude', 'include'].map((m) => {
          const on = (sel ? sel.mode : newMode) === m;
          return (
            <button key={m} type="button" aria-pressed={on} data-testid={`roto-mode-${m}`}
                    onClick={() => { setNewMode(m); update((z) => R.setMode(z, m)); }}
                    className={`${btn} ${on ? (m === 'exclude' ? 'bg-rose-700 text-white' : 'bg-emerald-500 text-black') : 'bg-white/10 text-white/70'}`}
                    title={m === 'exclude' ? 'Exclusion zone: the original pixels are restored inside it (microphone, hand, watermark)'
                      : 'Inclusion zone: the swap is forced inside it, cancelling the occluder mask and any exclusion it overlaps'}>
              {m === 'exclude' ? 'Exclude' : 'Include'}
            </button>
          );
        })}
        {zones.length > 0 && (
          <span className="flex items-center gap-1 pl-1.5 border-l border-white/10" role="group" aria-label="Zones">
            {zones.slice(0, 6).map((z, i) => (
              <button key={z.id} type="button" aria-pressed={z.id === selectedId} data-testid="roto-chip"
                      aria-label={R.describe(z, i)} onClick={() => onSelect(z.id)} title={R.describe(z, i)}
                      className={`rounded-md px-1.5 py-0.5 text-nano font-bold text-white ${z.id === selectedId ? 'ring-1 ring-white' : ''} ${z.mode === 'include' ? 'bg-emerald-800' : 'bg-rose-800'}`}>
                {i + 1}{z.pin ? '📌' : ''}{z.closed ? '' : '…'}
              </button>
            ))}
          </span>
        )}
        <button type="button" onClick={onDone} data-testid="roto-done"
                className={`${btn} bg-white/10 hover:bg-white/20 text-white/80 hover:text-white`} title="Leave the Pen tool (P)">
          Done
        </button>
      </div>

      {sel && (
        <div className={pill}>
          <label className="flex items-center gap-1">
            <span className="text-nano font-mono text-white/60 uppercase font-bold">Feather</span>
            <input type="range" min={0} max={120} step={1} value={sel.feather} data-testid="roto-feather"
                   aria-label="Feather (px)"
                   onChange={(e) => update((z) => R.setFeather(z, Number(e.target.value)))}
                   className="w-16 h-1 accent-rose-400 cursor-pointer" />
            <span className="text-micro font-mono font-bold text-white/80 tabular-nums w-8">{Math.round(sel.feather)}px</span>
          </label>
          <label className="flex items-center gap-1">
            <span className="text-nano font-mono text-white/60 uppercase font-bold">Frames</span>
            <select value={rangeKind(sel)} data-testid="roto-range" aria-label="Frames the zone applies to"
                    onChange={(e) => update((z) => R.setRange(z, rangeFor(e.target.value, frame, maxFrames)))}
                    className="rounded-md bg-white/10 px-1 py-0.5 text-micro font-bold text-white">
              {RANGES.map(([v, l]) => <option key={v} value={v} className="text-black">{l}</option>)}
            </select>
          </label>
          <button type="button" aria-pressed={!!sel.pin} data-testid="roto-pin"
                  disabled={!sel.pin && !pinTarget}
                  onClick={() => update((z) => (z.pin ? R.unpinZone(z) : R.pinZone(z, pinTarget && { nose: pinTarget.nose, chin: pinTarget.chin }, pinTarget?.source_index)))}
                  className={`${btn} ${sel.pin ? 'bg-sky-500 text-black' : 'bg-white/10 text-white/80'} disabled:opacity-40`}
                  title={sel.pin ? 'Stop following the face'
                    : pinTarget ? "2-point landmark pin: the zone follows this face's nose and chin through the shot"
                      : 'No face with landmarks on this frame to pin to'}>
            {sel.pin ? '📌 Pinned' : '📌 Pin to face'}
          </button>
          {!sel.closed && (
            <button type="button" data-testid="roto-close" disabled={sel.points.length < 3}
                    onClick={() => update((z) => R.closeZone(z))}
                    className={`${btn} bg-[var(--accent)] text-white disabled:opacity-40`} title="Close the shape (Enter)">
              Close
            </button>
          )}
          <button type="button" data-testid="roto-delete"
                  onClick={() => { onChange(zones.filter((z) => z.id !== sel.id)); onSelect(null); }}
                  className={`${btn} bg-white/10 hover:bg-white/20 text-white/80 hover:text-white`} title="Delete this zone (Delete)">
            Delete
          </button>
        </div>
      )}
    </div>
  );
}
