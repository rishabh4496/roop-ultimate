import React, { useEffect, useMemo, useRef, useState } from 'react';
import CompareGrid from './CompareGrid';
import { useSlotStore } from '../../store/slotStore';
import { SLOT_ORDER, buildQuad, deltaSummary, diffTable, pickBaseline } from './slotDelta';

// ── Quad-Slot Split Compare (Stage 12) ──────────────────────────────────────
//
// The viewport splits into four quadrants - Slot A, B, C, D - each showing the picture that slot KEPT when it was
// captured. Scroll zooms, drag pans, and all four move together (CompareGrid's one shared zoom / pan), so the same
// pixel of the same face is under the same corner of every quadrant.
//
// Each quadrant says exactly how its recipe differs from the baseline slot (the sliders and toggles, with their old and
// new values, then any curve / brush / mask / roto / pin differences), and a table under the grid lists every setting
// that is not the same in all four.
//
// WHAT THIS IS NOT. It does not render anything: it shows the four frames the slots already hold. A slot captured with
// no rendered preview on screen has no picture ("recipe only"), and a slot captured on another frame shows that
// frame's picture, labelled with its frame number, so a difference in the picture is never confused with a difference
// in the recipe. The earlier 4-Up grid (QuadInspectionGrid) drew one frame four times; this one never does.

const MAX_TABLE_ROWS = 14;

/** A slot's kept bitmap as an object URL (the grid draws <img>), converted once per slot and revoked with it. */
function bitmapToUrl(bm) {
  return new Promise((resolve) => {
    try {
      const c = document.createElement('canvas');
      c.width = bm.width;
      c.height = bm.height;
      c.getContext('2d').drawImage(bm, 0, 0);
      c.toBlob((b) => resolve(b ? URL.createObjectURL(b) : null), 'image/png');
    } catch { resolve(null); }                       // a bitmap closed under us: the quadrant shows "no picture"
  });
}

function useSlotPictureUrls(slots) {
  const cache = useRef(new Map());                    // slot object -> object URL ('' while converting)
  const alive = useRef(true);
  const [tick, setTick] = useState(0);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);

  useEffect(() => {
    const live = new Set();
    for (const id of SLOT_ORDER) {
      const s = slots[id];
      if (!s?.bitmap) continue;
      live.add(s);
      if (cache.current.has(s)) continue;
      cache.current.set(s, '');
      bitmapToUrl(s.bitmap).then((u) => {
        if (!u) return;
        if (!alive.current || !cache.current.has(s)) { URL.revokeObjectURL(u); return; }   // the slot went away meanwhile
        cache.current.set(s, u);
        setTick((n) => n + 1);
      });
    }
    let dropped = false;
    for (const [s, u] of cache.current) {
      if (live.has(s)) continue;
      if (u) URL.revokeObjectURL(u);
      cache.current.delete(s);
      dropped = true;
    }
    if (dropped) setTick((n) => n + 1);
  }, [slots]);

  useEffect(() => () => {
    for (const u of cache.current.values()) if (u) URL.revokeObjectURL(u);
    cache.current.clear();
  }, []);

  // `tick` is what makes a finished conversion show: the cache itself is a ref.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => Object.fromEntries(SLOT_ORDER.map((id) => [id, (slots[id] && cache.current.get(slots[id])) || null])), [slots, tick]);
}

const stop = (e) => e.stopPropagation();           // CompareGrid pans on any pointerdown: a button must not start a pan

export default function QuadSlotCompare({ bank = 'faceswap', frameKey = null, onClose, onRecall, onCapture }) {
  const slots = useSlotStore((s) => s.banks[bank]);
  const [baseReq, setBaseReq] = useState(null);
  const urls = useSlotPictureUrls(slots);
  const baseId = pickBaseline(slots, baseReq);
  const quad = useMemo(() => buildQuad(slots, baseReq, frameKey), [slots, baseReq, frameKey]);
  const byId = useMemo(() => Object.fromEntries(quad.map((q) => [q.id, q])), [quad]);
  const table = useMemo(() => diffTable(slots, baseReq), [slots, baseReq]);

  const placeholder = (id) => {
    const q = byId[id];
    if (!q) return null;
    if (q.empty) {
      return (
        <div className="flex flex-col items-center gap-2 px-3 text-center text-xs text-muted" data-testid={`quad-empty-${id}`}>
          <span className="font-semibold">Slot {id} is empty</span>
          {onCapture && (
            <button type="button" onPointerDown={stop} onClick={() => onCapture(id)} data-testid={`quad-capture-${id}`}
                    className="rounded-full border border-white/15 bg-white/[0.06] px-3 py-1 text-mini font-bold text-white/85 hover:bg-white/10">
              Capture the live look here
            </button>
          )}
        </div>
      );
    }
    return (
      <div className="flex flex-col items-center gap-1 px-3 text-center text-xs text-muted" data-testid={`quad-nopicture-${id}`}>
        <span className="font-semibold">Recipe only</span>
        <span>Slot {id} was captured with no rendered frame on screen, so it kept the settings but no picture.</span>
      </div>
    );
  };

  const caption = (id) => {
    const q = byId[id];
    if (!q || q.empty) return `Slot ${id}`;
    return `Slot ${id}${q.clock ? ` · ${q.clock}` : ''}${q.frame ? ` · frame ${q.frame}` : ''}`;
  };

  const overlay = (id) => {
    const q = byId[id];
    if (!q) return null;
    const sum = deltaSummary(q, baseId);
    return (
      <div className="pointer-events-none absolute left-2 top-2 z-10 flex max-w-[calc(100%-1rem)] flex-col items-start gap-1" data-testid={`quad-info-${id}`}>
        <div className="pointer-events-auto flex flex-wrap items-center gap-1">
          {q.baseline
            ? <span className="rounded bg-cyan-500/30 px-1.5 py-0.5 text-micro font-bold uppercase tracking-wide text-cyan-100" data-testid={`quad-baseline-${id}`}>Baseline</span>
            : !q.empty && (
              <button type="button" onPointerDown={stop} onClick={() => setBaseReq(id)} data-testid={`quad-set-baseline-${id}`}
                      className="rounded border border-white/15 bg-black/60 px-1.5 py-0.5 text-micro font-semibold text-white/80 hover:bg-black/80">
                Use as baseline
              </button>
            )}
          {!q.empty && onRecall && (
            <button type="button" onPointerDown={stop} onClick={() => onRecall(id)} data-testid={`quad-recall-${id}`}
                    title="Apply this slot's recipe to the editor and go back to the single viewport"
                    className="rounded border border-white/15 bg-black/60 px-1.5 py-0.5 text-micro font-semibold text-white/80 hover:bg-black/80">
              Recall
            </button>
          )}
          {q.otherFrame && (
            <span className="rounded bg-amber-500/25 px-1.5 py-0.5 text-micro font-semibold text-amber-200" title="This slot's picture is of a different frame from the one on the timeline">
              Another frame
            </span>
          )}
        </div>
        {!q.empty && (
          <div className={`rounded-lg border px-2 py-1 text-micro leading-snug backdrop-blur ${
            sum.names.length ? 'border-amber-400/40 bg-black/70 text-amber-100' : 'border-white/10 bg-black/60 text-white/70'}`}>
            <div className="font-bold">{sum.headline}</div>
            {sum.names.map((n) => <div key={n} className="truncate" title={n}>{n}</div>)}
            {sum.more > 0 && <div className="text-white/55">+{sum.more} more (see the table)</div>}
          </div>
        )}
      </div>
    );
  };

  const filled = SLOT_ORDER.filter((id) => slots[id]).length;
  return (
    <section className="space-y-2" data-testid="quad-slot-compare" aria-label="Quad slot compare">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <h3 className="text-mini font-bold uppercase tracking-[0.14em] text-white/80">Quad slot compare</h3>
        <span className="text-micro text-white/55">
          {filled < 2 ? 'Capture at least two slots to compare them.' : `Differences are measured against Slot ${baseId}. Scroll zooms and drag pans all four together.`}
        </span>
        <span className="flex-1" />
        <button type="button" onClick={onClose} data-testid="quad-close"
                className="rounded-lg border border-white/15 bg-white/[0.05] px-2.5 py-1 text-mini font-semibold text-white/85 hover:bg-white/10">
          Back to the viewport
        </button>
      </div>

      <CompareGrid
        items={SLOT_ORDER}
        previews={urls}
        times={{}}
        timers={{}}
        gridColsClass="grid-cols-2"
        cellPlaceholder={placeholder}
        cellLabel={caption}
        cellOverlay={overlay}
      />

      {table.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-white/10 bg-white/[0.03]" data-testid="quad-diff-table">
          <table className="w-full min-w-[420px] text-left text-micro">
            <caption className="px-3 pt-2 text-left text-micro font-bold uppercase tracking-wide text-white/60">
              Settings that differ ({table.length}) - highlighted values differ from Slot {baseId}
            </caption>
            <thead>
              <tr className="text-white/55">
                <th scope="col" className="px-3 py-1 font-semibold">Setting</th>
                {SLOT_ORDER.map((id) => <th key={id} scope="col" className="px-2 py-1 font-semibold">Slot {id}{id === baseId ? ' (base)' : ''}</th>)}
              </tr>
            </thead>
            <tbody>
              {table.slice(0, MAX_TABLE_ROWS).map((r) => (
                <tr key={r.key} className="border-t border-white/[0.06]" data-testid={`quad-row-${r.key}`}>
                  <th scope="row" className="px-3 py-1 font-medium text-white/80">{r.label}</th>
                  {SLOT_ORDER.map((id) => (
                    <td key={id} className={`px-2 py-1 font-mono tabular-nums ${r.differs.has(id) ? 'font-bold text-amber-300' : 'text-white/65'}`}
                        data-differs={r.differs.has(id) ? '1' : '0'}>
                      {r.values[id] || '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {table.length > MAX_TABLE_ROWS && <p className="px-3 pb-2 text-micro text-white/55">+{table.length - MAX_TABLE_ROWS} more settings differ.</p>}
        </div>
      )}
    </section>
  );
}
