import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  clampPan, panAnchoredAt, panCenteringAt, transformFor, uiScale, wheelZoom,
} from './zoomPan';

const ZOOM_MAX = 8;

/**
 * Generic side-by-side comparison grid with a single shared zoom/pan, used by
 * both the Enhancer grid and the Mask-engine grid. Each cell shows one variant's
 * rendered preview (or a live "Rendering…" spinner while it's in flight).
 *
 * Props:
 *   items          string[]  — variant labels to show, one cell each
 *   previews       {label: dataURL}
 *   times          {label: "1.23s" | "Cached"}
 *   timers         {label: "0.4s"}  — live elapsed while rendering
 *   gridColsClass  tailwind grid-cols class
 *   emptyHint      optional string shown center when there are no items
 */
export default function CompareGrid({ items, previews, times, timers, gridColsClass, emptyHint }) {
  // Zoom/pan is shared by every cell, so the same region is magnified in all
  // variants at once — that's what makes fine differences readable.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const containerRef = useRef(null);
  const cellRef = useRef(null);          // first cell — the geometry pan is clamped against

  // The wheel listener below is attached once and never re-bound, and a drag
  // reads the pan it started from, so both need the live values rather than the
  // ones captured when they were created.
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  useEffect(() => { zoomRef.current = zoom; }, [zoom]);
  useEffect(() => { panRef.current = pan; }, [pan]);

  const reset = useCallback(() => { setZoom(1); setPan({ x: 0, y: 0 }); }, []);

  // Any change to what is being compared invalidates the framing — the old pan
  // was chosen against a different image.
  const itemKey = items.join('|');
  useEffect(() => { reset(); }, [itemKey, reset]);

  /** The cell the pointer is over, so wheel/double-click anchor to ITS centre. */
  const stageUnder = (target) =>
    (target?.closest?.('[data-cmp-cell]')) || cellRef.current || containerRef.current;

  /** What is actually being looked at inside a cell. The cell is a grid box of
   *  whatever shape the layout gives it and the <img> is `object-contain`
   *  inside it, so clamping the pan against the CELL counts the letterbox bars
   *  as picture and lets the image be dragged out under the rounded edge. Null
   *  while the cell is still rendering, which falls back to the cell. */
  const contentIn = (stage) => stage?.querySelector?.('img') || null;

  // Wheel zoom must be a NATIVE non-passive listener. React attaches `wheel` at
  // the root as passive, so the preventDefault() this used to call from onWheel
  // was a no-op: every notch zoomed the grid AND scrolled the page behind it.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const onWheel = (e) => {
      // Ctrl/Cmd + wheel is the APP zoom, claimed on window in App.jsx. An
      // element listener runs before the event reaches window, so without this
      // one gesture zoomed the grid AND the whole UI at once.
      if (e.ctrlKey || e.metaKey) return;
      const next = wheelZoom(e.deltaY, zoomRef.current, ZOOM_MAX);
      if (next === null) return;    // nothing left to zoom — let the page scroll
      e.preventDefault();
      const stage = stageUnder(e.target);
      if (next === 1) { setZoom(1); setPan({ x: 0, y: 0 }); return; }
      const p = panAnchoredAt({ x: e.clientX, y: e.clientY }, stage,
                              zoomRef.current, next, panRef.current);
      setZoom(next);
      setPan(clampPan(p, next, stage, contentIn(stage)));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // Derived from the refs, not from inside a setZoom updater: an updater must
  // be pure, and React calls it twice in StrictMode.
  const zoomBy = useCallback((factor) => {
    const nz = Math.min(Math.max(1, zoomRef.current * factor), ZOOM_MAX);
    setZoom(nz);
    setPan(nz === 1 ? { x: 0, y: 0 }
                    : clampPan(panRef.current, nz, cellRef.current, contentIn(cellRef.current)));
  }, []);

  // `+` / `-` are documented in the shortcuts HUD under "Compare & Zoom", but
  // they were only ever implemented on InteractivePreview — which is precisely
  // the component a comparison grid REPLACES, so in compare mode the documented
  // keys did nothing.
  //
  // Bound on the grid ELEMENT (which is focusable, and takes focus on the first
  // click or scroll), not on window. Every other keydown handler in this app is
  // global, and InteractivePreview already owns bare `+`/`-` there — a second
  // window listener for the same key does not override the first, it runs
  // alongside it, so the two would double-step each other's zoom the moment
  // both were ever mounted together. Scoping to the element also means the keys
  // only act on the surface you are actually looking at.
  const handleKeyDown = (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;   // Ctrl +/- is the app zoom
    if (e.key === '=' || e.key === '+') { e.preventDefault(); zoomBy(1.4); }
    else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomBy(1 / 1.4); }
    else if (e.key === '0') { e.preventDefault(); reset(); }
  };

  // Panning uses pointer CAPTURE, so a drag that leaves the grid — trivially
  // easy once zoomed — keeps tracking instead of stopping dead at the border
  // and needing a fresh grab.
  const dragRef = useRef(null);

  const handlePointerDown = (e) => {
    // Arm the +/- keys. Skipped when the press landed on a real control, or
    // clicking Expand/Reset would move focus off the button that was just
    // pressed and its focus ring would land on the grid instead.
    if (!e.target?.closest?.('button, a, input, textarea, select')) {
      containerRef.current?.focus?.({ preventScroll: true });
    }
    if (zoomRef.current <= 1 || e.button > 0) return;
    const stage = stageUnder(e.target);
    dragRef.current = {
      stage,
      scale: uiScale(stage),
      ox: e.clientX,
      oy: e.clientY,
      pan: panRef.current,
      content: contentIn(stage),
    };
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setIsPanning(true);
  };

  const handlePointerMove = (e) => {
    const d = dragRef.current;
    if (!d) return;
    // Divide by the app zoom: clientX deltas are visual pixels, translate() is
    // layout pixels, and without this the image lags or outruns the cursor by
    // the app-zoom ratio at any setting other than 100%.
    const next = {
      x: d.pan.x + (e.clientX - d.ox) / d.scale,
      y: d.pan.y + (e.clientY - d.oy) / d.scale,
    };
    setPan(clampPan(next, zoomRef.current, d.stage, d.content));
  };

  const endPan = (e) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    // pointercancel means the browser has ALREADY released the capture, and
    // releasing an id it no longer holds throws NotFoundError — which would
    // escape as an unhandled error out of an event handler.
    try { e.currentTarget.releasePointerCapture?.(e.pointerId); } catch { /* already released */ }
    setIsPanning(false);
  };

  // Double click toggles fit ↔ 2.5x centred on the clicked spot of that cell.
  const handleDoubleClick = (e) => {
    if (zoomRef.current > 1) { reset(); return; }
    const z = 2.5;
    setZoom(z);
    const stage = stageUnder(e.target);
    setPan(panCenteringAt({ x: e.clientX, y: e.clientY }, stage, z, contentIn(stage)));
  };

  const transformStyle = transformFor(zoom, pan);

  // Layout. The grid used to be `aspect-video max-h-[45vh]`, which on a 2x2
  // comparison left each variant about 200px tall — too small to judge the
  // pixel-level differences the grid exists to show. Height is now driven by
  // the ROW count (a 2x2 needs twice the box of a single row to give each cell
  // the same size) and the rows are explicit `1fr`, since auto rows let a slow
  // cell that is still rendering collapse to its spinner.
  const cols = /grid-cols-1(?!\d)/.test(gridColsClass || '') ? 1 : 2;
  const rows = Math.max(1, Math.ceil(Math.max(items.length, 1) / cols));
  const height = useMemo(() => {
    if (expanded) return rows > 1 ? 'min(88vh, 1240px)' : 'min(82vh, 920px)';
    return rows > 1 ? 'clamp(420px, 74vh, 1000px)' : 'clamp(300px, 56vh, 720px)';
  }, [expanded, rows]);

  return (
    <div
      ref={containerRef}
      className={`relative grid ${gridColsClass} gap-3 rounded-2xl overflow-hidden bg-black/45 border border-white/5 p-2 outline-none focus-visible:ring-1 focus-visible:ring-[var(--accent)]/40`}
      tabIndex={0}
      role="group"
      aria-label="Comparison grid — scroll to zoom all variants together"
      onKeyDown={handleKeyDown}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endPan}
      onPointerCancel={endPan}
      style={{
        height,
        gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
        touchAction: 'none',
        cursor: zoom > 1 ? (isPanning ? 'grabbing' : 'grab') : 'zoom-in',
      }}
    >
      {items.length === 0 && emptyHint && (
        <div className="col-span-full flex items-center justify-center text-xs text-white/40 font-semibold p-6 text-center">
          {emptyHint}
        </div>
      )}
      {items.map((label, i) => (
        <div key={label} data-cmp-cell
             ref={i === 0 ? cellRef : undefined}
             className="relative min-h-0 min-w-0 rounded-xl overflow-hidden bg-black/50 border border-white/5 flex items-center justify-center"
             onDoubleClick={handleDoubleClick}>
          {previews[label] ? (
            <div className="w-full h-full flex items-center justify-center transition-transform duration-75 select-none" style={transformStyle}>
              <img src={previews[label]} alt={label} className="max-w-full max-h-full object-contain pointer-events-none" draggable={false} />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-2 text-white/40 text-xs px-3 text-center">
              <span className="h-4 w-4 rounded-full border-2 border-white/20 border-t-[var(--accent)] animate-spin" />
              <span className="font-semibold">Rendering {label}…</span>
              {timers[label] && (
                <span className="text-micro text-white/45 font-mono">
                  Elapsed: {timers[label]}
                </span>
              )}
            </div>
          )}
          <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur border border-white/10 text-micro font-bold text-white uppercase pointer-events-none max-w-[calc(100%-1rem)] truncate">{label}</span>
          {times[label] && (
            <span className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur border border-white/10 text-micro font-bold text-white/60 font-mono pointer-events-none">
              ⏱️ {times[label]}
            </span>
          )}
        </div>
      ))}
      <div className="absolute top-3 right-3 z-30 flex items-center gap-1.5">
        {zoom > 1 ? (
          <button
            type="button"
            onClick={reset}
            className="px-2.5 py-1 rounded-full bg-black/70 backdrop-blur border border-white/10 text-mini font-bold text-white/80 hover:text-white hover:border-white/30 transition-all"
            title="Reset zoom (or double-click)"
          >
            {zoom.toFixed(1)}× — Reset
          </button>
        ) : items.length > 0 && (
          <span className="px-2.5 py-1 rounded-full bg-black/50 backdrop-blur text-micro font-semibold text-white/45 pointer-events-none select-none">
            Scroll or double-click to zoom all
          </span>
        )}
        {items.length > 0 && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="px-2.5 py-1 rounded-full bg-black/70 backdrop-blur border border-white/10 text-mini font-bold text-white/70 hover:text-white hover:border-white/30 transition-all"
            title={expanded ? 'Shrink the comparison box' : 'Expand the comparison box to fill the screen'}
            aria-pressed={expanded}
          >
            {expanded ? 'Shrink' : 'Expand'}
          </button>
        )}
      </div>
    </div>
  );
}
