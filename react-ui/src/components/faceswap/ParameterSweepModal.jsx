import React, { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useApi } from '../../useApi';
import { eventMatches } from '../../store/shortcutStore';
import { useSlotStore } from '../../store/slotStore';
import { SWEEP_PARAMETERS } from './sweepParameters';
import {
  FIT, LOUPE_POWER, SLOT_ORDER, aspectOf, defaultValues, focusView, loupeLayout, normalizeValue, normalizeView, panBy,
  recipeText, resultWarnings, tileAt, zoomAt,
} from './sweepModel';

// Stage 11: the 3x3 Parameter Sweep contact sheet, edge to edge (90vw x 90vh).
//
//   * Pick Parameter X (columns) and Parameter Y (rows) with their min / centre / max values; nine live swaps of THIS frame come
//     back (the backend detects and aligns once and re-runs the swap per cell: /api/preview/sweep).
//   * ONE view serves all nine cells: wheel zooms about the pointer, drag pans, Fit / Eyes / Mouth / Jaw jump - every cell follows.
//   * ONE loupe serves all nine: the 3.5x lens follows the pointer in every cell at once, over the same image point, so a pore, a
//     lash or a seam can be compared across the whole matrix without moving the mouse.
//   * "Apply Cell to Active Recipe" and "Assign to Snapshot Slot A/B/C/D" commit the selected cell's coordinates and pulse it.
//
// An axis the backend does not wire to anything (or a sweep whose nine pictures are byte-identical) is SAID, in a banner: a sheet
// of nine copies of one picture is not nine results. See roop/sweep_bindings.py.
//
// Keys (shortcuts/actions.js, scope 'sweep', remappable): 1-9 apply that cell, Enter applies the selected one, A-D assign it to a
// slot, G loupe, F fit, Esc closes.

const LOUPE_LENS_MIN = 72;
const LOUPE_LENS_MAX = 210;
const SLOT_KEY_ACTIONS = Object.freeze([['sweep.slotA', 'A'], ['sweep.slotB', 'B'], ['sweep.slotC', 'C'], ['sweep.slotD', 'D']]);

const paramById = (id, fallbackIndex = 0) => SWEEP_PARAMETERS.find((p) => p.id === id) || SWEEP_PARAMETERS[fallbackIndex];
const cssUrl = (u) => `url("${String(u).replace(/"/g, '%22')}")`;
const pos = (i) => ({ col: i % 3, row: Math.floor(i / 3) });

// ── one axis: parameter + min / centre / max ────────────────────────────────

function ValueField({ def, value, label, tone, onCommit }) {
  const [draft, setDraft] = useState(String(value));
  const [editing, setEditing] = useState(false);
  // The committed value flows back into the box only while you are NOT typing in it: re-syncing mid-edit turns "0." into "0"
  // (the number was already committed as 0) and the next digit lands in the wrong place.
  useEffect(() => { if (!editing) setDraft(String(value)); }, [value, editing]);
  if (def.type === 'categorical') {
    return (
      <label className="flex flex-col gap-0.5">
        <span className="text-nano font-bold uppercase tracking-wide text-white/60">{label}</span>
        <select
          value={value}
          onChange={(e) => onCommit(e.target.value)}
          className="rounded-md border border-white/20 bg-black/60 px-1.5 py-1 text-micro text-white"
        >
          {def.options.map((opt) => <option key={opt} value={opt} className="bg-[#121420]">{opt}</option>)}
        </select>
      </label>
    );
  }
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-nano font-bold uppercase tracking-wide text-white/60">{label}</span>
      <input
        type="number"
        inputMode="decimal"
        step={def.step}
        min={def.min}
        max={def.max}
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value);
          const n = parseFloat(e.target.value);
          if (Number.isFinite(n)) onCommit(n, { typing: true });
        }}
        onFocus={() => setEditing(true)}
        onBlur={() => { const v = normalizeValue(def, draft, value); setDraft(String(v)); onCommit(v); setEditing(false); }}
        onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }}
        className={`w-[4.6rem] rounded-md border bg-black/60 px-1.5 py-1 text-center font-mono text-micro text-white tabular-nums focus:outline-none focus:ring-1 ${tone}`}
        aria-label={`${label} value`}
      />
    </label>
  );
}

const AxisEditor = memo(function AxisEditor({ axis, title, accent, paramId, values, otherId, onParam, onValues }) {
  const def = paramById(paramId, axis === 'x' ? 0 : 6);
  const ring = axis === 'x' ? 'border-cyan-400/40 focus:ring-cyan-300' : 'border-amber-400/40 focus:ring-amber-300';
  return (
    <fieldset className="flex flex-wrap items-end gap-x-2 gap-y-1 rounded-xl border border-white/10 bg-white/[0.03] px-2.5 py-1.5" data-testid={`sweep-axis-${axis}`}>
      <legend className={`px-1 text-micro font-bold uppercase tracking-wider ${accent}`}>{title}</legend>
      <label className="flex flex-col gap-0.5">
        <span className="text-nano font-bold uppercase tracking-wide text-white/60">Parameter</span>
        <select
          value={paramId}
          data-testid={`sweep-param-${axis}`}
          onChange={(e) => onParam(e.target.value)}
          className="max-w-[13rem] rounded-md border border-white/20 bg-black/60 px-1.5 py-1 text-micro text-white"
        >
          {SWEEP_PARAMETERS.map((p) => (
            <option key={p.id} value={p.id} disabled={p.id === otherId} className="bg-[#121420]">{p.label}</option>
          ))}
        </select>
      </label>
      {['Min', 'Center', 'Max'].map((label, i) => (
        <ValueField
          key={label}
          def={def}
          label={label}
          tone={ring}
          value={values[i]}
          onCommit={(v) => onValues(i, v)}
        />
      ))}
      <button
        type="button"
        onClick={() => onValues(null, null)}
        className="mb-px rounded-md border border-white/15 px-1.5 py-1 text-micro text-white/80 hover:border-white/35 hover:text-white"
        title={`Reset ${def.label} to its default min / centre / max`}
      >
        ↺
      </button>
    </fieldset>
  );
});

// ── one cell ────────────────────────────────────────────────────────────────

const Cell = memo(function Cell({
  index, tile, xLabel, yLabel, view, aspect, selected, committed, onSelect, onApply, registerFrame, registerLens, busy,
}) {
  const { col, row } = pos(index);
  return (
    <div
      data-cell={index}
      data-testid={`sweep-tile-${col}-${row}`}
      data-selected={selected ? 'true' : 'false'}
      data-committed={committed ? 'true' : 'false'}
      role="group"
      aria-label={`Cell ${index + 1}`}
      className={`sweep-cell group relative min-h-0 min-w-0 overflow-hidden rounded-xl border bg-black/80 shadow-md transition-colors ${
        selected ? 'border-cyan-300 ring-1 ring-cyan-300/70' : 'border-white/10 hover:border-white/35'} ${busy ? 'opacity-60' : ''}`}
      style={{ containerType: 'size' }}
    >
      {/* The cell's one focus stop and its pan surface: focus or click selects it, double-click applies it. */}
      <button
        type="button"
        data-pan-surface
        data-testid={`sweep-select-${col}-${row}`}
        aria-pressed={selected}
        aria-label={tile
          ? `Cell ${index + 1}. ${recipeText(xLabel, yLabel, tile)}${committed ? `. Committed ${committed.kind === 'slot' ? `to slot ${committed.slot}` : 'to the active recipe'}` : ''}`
          : `Cell ${index + 1}, empty`}
        onFocus={() => onSelect(index)}
        onDoubleClick={() => onApply(index)}
        style={{ cursor: 'inherit' }}
        className="absolute inset-0 z-[1] rounded-xl bg-transparent focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-cyan-300"
      />
      <div className="absolute inset-0 grid place-items-center">
        <div
          ref={(el) => registerFrame(index, el)}
          className="relative overflow-hidden"
          style={{
            width: `min(100cqw, calc(100cqh * ${aspect}))`,
            height: `min(100cqh, calc(100cqw / ${aspect}))`,
          }}
        >
          {tile?.image ? (
            <img
              src={tile.image}
              alt=""
              draggable={false}
              className="pointer-events-none block h-full w-full select-none object-fill"
              style={{ transform: `translate(${view.pan.x * 100}%, ${view.pan.y * 100}%) scale(${view.zoom})`, transformOrigin: '50% 50%' }}
            />
          ) : (
            <div className="grid h-full w-full place-items-center text-micro text-white/55">Cell {index + 1}</div>
          )}
          <div ref={(el) => registerLens(index, el)} className="sweep-loupe" style={{ display: 'none' }} aria-hidden="true" />
        </div>
      </div>

      {committed && <span key={committed.nonce} className="sweep-pulse-ring" aria-hidden="true" />}

      <div className="pointer-events-none absolute left-1.5 top-1.5 flex items-center gap-1">
        <span className="rounded bg-black/75 px-1.5 py-0.5 font-mono text-micro font-bold text-white">#{index + 1}</span>
        {committed && (
          <span data-testid={`sweep-committed-${index}`} className="rounded bg-cyan-300 px-1.5 py-0.5 text-micro font-bold text-black">
            {committed.kind === 'slot' ? `→ Slot ${committed.slot}` : '✓ Applied'}
          </span>
        )}
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-1 bg-gradient-to-t from-black/90 via-black/70 to-transparent p-1.5">
        <div className="min-w-0 font-mono text-micro leading-tight text-white/85">
          <div className="truncate">{xLabel}: {tile ? tile.x_value : '·'}</div>
          <div className="truncate">{yLabel}: {tile ? tile.y_value : '·'}</div>
        </div>
        <button
          type="button"
          tabIndex={-1}
          data-testid={`sweep-apply-${col}-${row}`}
          disabled={!tile}
          onClick={(e) => { e.stopPropagation(); onApply(index); }}
          className="pointer-events-auto relative z-[2] rounded bg-[var(--accent)] px-2 py-0.5 text-micro font-bold text-black opacity-0 shadow transition-opacity focus:opacity-100 group-hover:opacity-100 disabled:hidden"
          title={`Apply cell ${index + 1} to the active recipe (key ${index + 1})`}
        >
          Apply
        </button>
      </div>
    </div>
  );
});

// ── the sheet ───────────────────────────────────────────────────────────────

export default function ParameterSweepModal({
  isOpen,
  onClose,
  targetIndex = 0,
  frame = 1,
  currentSettings = {},
  onApplyRecipe,
  onAssignSlot,
  slotBank = 'faceswap',
  notify,
}) {
  const { postJSON } = useApi();

  const [xParamId, setXParamId] = useState('restore_ultra_detail_weight');
  const [xValues, setXValues] = useState(() => defaultValues(paramById('restore_ultra_detail_weight')));
  const [yParamId, setYParamId] = useState('mask_erode_dilate_radius');
  const [yValues, setYValues] = useState(() => defaultValues(paramById('mask_erode_dilate_radius')));

  const [loading, setLoading] = useState(false);
  const [sweepResult, setSweepResult] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(4);
  const [committed, setCommitted] = useState({});          // { [cellIndex]: { kind, slot?, nonce } }
  const [status, setStatus] = useState('');
  const [view, setView] = useState(FIT);
  const [loupeOn, setLoupeOn] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [natural, setNatural] = useState(null);

  const xDef = paramById(xParamId, 0);
  const yDef = paramById(yParamId, 6);

  const slots = useSlotStore((s) => s.banks[slotBank]);

  // Refs the pointer handlers read, so a pan or a loupe move never re-subscribes anything.
  const viewRef = useRef(view);
  viewRef.current = view;
  const loupeOnRef = useRef(loupeOn);
  loupeOnRef.current = loupeOn;
  const frameEls = useRef([]);
  const lensEls = useRef([]);
  const probeRef = useRef(null);
  const rafRef = useRef(0);
  const gridRef = useRef(null);
  const dragRef = useRef(null);
  const abortRef = useRef(null);
  const contextRef = useRef('');
  const nonceRef = useRef(0);

  const registerFrame = useCallback((i, el) => { frameEls.current[i] = el; }, []);
  const registerLens = useCallback((i, el) => { lensEls.current[i] = el; }, []);

  // ── run ───────────────────────────────────────────────────────────────────
  const runSweep = useCallback(async () => {
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    try {
      const res = await postJSON('/api/preview/sweep', {
        ...currentSettings,
        index: targetIndex,
        frame,
        x_param: xParamId,
        x_values: xValues,
        y_param: yParamId,
        y_values: yValues,
      }, { timeout: 0, signal: ctrl.signal, abortOnUnmount: true });
      if (ctrl.signal.aborted) return;
      if (res?.error || !Array.isArray(res?.tiles)) throw new Error(res?.message || res?.error || 'The sweep returned no tiles');
      setSweepResult(res);
      setCommitted({});
      setStatus('');
      setNatural(null);
      setView(FIT);
    } catch (err) {
      if (err?.name === 'AbortError' || ctrl.signal.aborted) return;
      setError(err?.message || 'Failed to generate the 3×3 parameter sweep');
    } finally {
      if (abortRef.current === ctrl) { abortRef.current = null; setLoading(false); }
    }
  }, [postJSON, currentSettings, targetIndex, frame, xParamId, xValues, yParamId, yValues]);
  const runRef = useRef(runSweep);
  runRef.current = runSweep;

  // Open: sweep THIS frame. Reopened on another frame or target: the old sheet is of something else, so it is cleared and re-run.
  useEffect(() => {
    if (!isOpen) {
      if (abortRef.current) { abortRef.current.abort(); abortRef.current = null; setLoading(false); }
      return;
    }
    const key = `${targetIndex}|${frame}`;
    if (contextRef.current !== key) {
      contextRef.current = key;
      setSweepResult(null);
      setCommitted({});
      runRef.current();
    }
  }, [isOpen, targetIndex, frame]);

  // ── axes ──────────────────────────────────────────────────────────────────
  const changeParam = useCallback((axis, id) => {
    const def = paramById(id);
    if (axis === 'x') { setXParamId(id); setXValues(defaultValues(def)); } else { setYParamId(id); setYValues(defaultValues(def)); }
  }, []);
  const changeValues = useCallback((axis, i, v) => {
    const def = paramById(axis === 'x' ? xParamId : yParamId);
    const set = axis === 'x' ? setXValues : setYValues;
    if (i == null) { set(defaultValues(def)); return; }
    set((cur) => { const next = [...cur]; next[i] = v; return next; });
  }, [xParamId, yParamId]);
  const onXParam = useCallback((id) => changeParam('x', id), [changeParam]);
  const onYParam = useCallback((id) => changeParam('y', id), [changeParam]);
  const onXValues = useCallback((i, v) => changeValues('x', i, v), [changeValues]);
  const onYValues = useCallback((i, v) => changeValues('y', i, v), [changeValues]);

  // ── the shared view and the loupe ─────────────────────────────────────────
  const paintLoupe = useCallback(() => {
    rafRef.current = 0;
    const probe = probeRef.current;
    for (let i = 0; i < 9; i += 1) {
      const lens = lensEls.current[i];
      const box = frameEls.current[i];
      if (!lens) continue;
      if (!probe || !loupeOnRef.current || !box || !box.clientWidth) { lens.style.display = 'none'; continue; }
      const w = box.clientWidth;
      const h = box.clientHeight;
      const size = Math.round(Math.min(LOUPE_LENS_MAX, Math.max(LOUPE_LENS_MIN, Math.min(w, h) * 0.4)));
      const lay = loupeLayout({ frameW: w, frameH: h, view: viewRef.current, u: probe.u, v: probe.v, lens: size, power: LOUPE_POWER });
      const st = lens.style;
      st.display = 'block';
      st.width = `${size}px`;
      st.height = `${size}px`;
      st.left = `${Math.round(lay.left)}px`;
      st.top = `${Math.round(lay.top)}px`;
      st.backgroundSize = `${lay.bgW}px ${lay.bgH}px`;
      st.backgroundPosition = `${lay.bgX}px ${lay.bgY}px`;
    }
  }, []);
  const schedulePaint = useCallback(() => {
    if (!rafRef.current) rafRef.current = requestAnimationFrame(paintLoupe);
  }, [paintLoupe]);

  // The lenses show their own tile's picture; set once per result, not per pointer move.
  useLayoutEffect(() => {
    for (let i = 0; i < 9; i += 1) {
      const lens = lensEls.current[i];
      const tile = sweepResult?.tiles?.[i];
      if (lens) { lens.style.backgroundImage = tile?.image ? cssUrl(tile.image) : 'none'; lens.style.display = 'none'; }
    }
  }, [sweepResult]);
  useEffect(() => { schedulePaint(); }, [view, loupeOn, schedulePaint]);
  useEffect(() => () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); }, []);

  // Position of the pointer inside the picture box of the cell under it: (u, v) in 0..1, or null off the pictures.
  const probeFromEvent = useCallback((e) => {
    const cellEl = e.target?.closest?.('[data-cell]');
    if (!cellEl) return null;
    const i = Number(cellEl.getAttribute('data-cell'));
    const box = frameEls.current[i];
    if (!box) return null;
    const r = box.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    const u = (e.clientX - r.left) / r.width;
    const v = (e.clientY - r.top) / r.height;
    return { i, u: Math.min(1, Math.max(0, u)), v: Math.min(1, Math.max(0, v)), inside: u >= 0 && u <= 1 && v >= 0 && v <= 1, w: r.width, h: r.height };
  }, []);

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    // A press on a control inside a cell (its Apply button) is a click, not the start of a pan: capturing the pointer would
    // retarget the click to the grid and the button would never fire.
    if (e.target?.closest?.('button:not([data-pan-surface]), input, select, a')) return;
    const p = probeFromEvent(e);
    if (!p) return;
    setSelected(p.i);
    dragRef.current = { x: e.clientX, y: e.clientY, w: p.w, h: p.h, view: viewRef.current };
    setDragging(true);
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e) => {
    const d = dragRef.current;
    if (d) {
      const next = panBy(d.view, (e.clientX - d.x) / d.w, (e.clientY - d.y) / d.h);
      setView(next);
    }
    const p = probeFromEvent(e);
    probeRef.current = p && p.inside ? { u: p.u, v: p.v } : null;
    schedulePaint();
  };
  const endDrag = (e) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    setDragging(false);
    e.currentTarget.releasePointerCapture?.(e.pointerId);
  };
  const onPointerLeave = () => { if (!dragRef.current) { probeRef.current = null; schedulePaint(); } };

  // Wheel must be NON-passive to stop the page behind scrolling: React's synthetic onWheel is passive.
  useEffect(() => {
    const el = gridRef.current;
    if (!isOpen || !el) return undefined;
    const onWheel = (e) => {
      const p = probeFromEvent(e);
      if (!p) return;
      e.preventDefault();
      setView((v) => zoomAt(v, e.deltaY < 0 ? 1.15 : 1 / 1.15, { u: p.u, v: p.v }));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [isOpen, probeFromEvent, sweepResult]);

  const focusOn = (which) => setView(normalizeView(focusView(which, sweepResult?.face_coordinates)));

  // ── committing a cell ─────────────────────────────────────────────────────
  const commit = useCallback((index, kind, slot) => {
    const tile = sweepResult?.tiles?.[index];
    if (!tile) return;
    nonceRef.current += 1;
    const nonce = nonceRef.current;
    const text = recipeText(xDef.label, yDef.label, tile);
    if (kind === 'slot') {
      if (!onAssignSlot) return;
      onAssignSlot(slot, tile.recipe, { tile, index });
      setStatus(`Cell #${index + 1} (${text}) assigned to Slot ${slot}. Not applied: the live recipe is unchanged.`);
    } else {
      onApplyRecipe?.(tile.recipe);
      notify?.(`Applied cell #${index + 1}: ${text}`, 'info');
      setStatus(`Cell #${index + 1} applied to the active recipe: ${text}.`);
    }
    setSelected(index);
    setCommitted((c) => ({ ...c, [index]: { kind, slot, nonce } }));
  }, [sweepResult, xDef.label, yDef.label, onApplyRecipe, onAssignSlot, notify]);
  const applyCell = useCallback((index) => commit(index, 'recipe'), [commit]);
  const selectCell = useCallback((i) => setSelected(i), []);

  // ── keys (registry actions, scope 'sweep') ────────────────────────────────
  useEffect(() => {
    if (!isOpen) return undefined;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); return; }
      const tag = e.target?.tagName;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || e.isComposing) return;
      if (eventMatches(e, 'sweep.cell')) {
        const n = parseInt(e.key, 10);
        if (n >= 1 && n <= 9) { e.preventDefault(); commit(n - 1, 'recipe'); }
        return;
      }
      if (eventMatches(e, 'sweep.apply')) {
        // Enter on a focused button activates that button - except a cell's own select button, whose job IS to be applied.
        if (['BUTTON', 'A'].includes(tag) && !e.target.closest?.('[data-pan-surface]')) return;
        e.preventDefault();
        commit(selected, 'recipe');
        return;
      }
      for (const [action, slot] of SLOT_KEY_ACTIONS) {
        if (eventMatches(e, action)) { e.preventDefault(); commit(selected, 'slot', slot); return; }
      }
      if (eventMatches(e, 'sweep.loupe')) { e.preventDefault(); setLoupeOn((v) => !v); return; }
      if (eventMatches(e, 'sweep.fit')) { e.preventDefault(); setView(FIT); }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen, onClose, commit, selected]);

  // Focus the dialog when it opens so the keys above reach it, and give focus back on close.
  const panelRef = useRef(null);
  useEffect(() => {
    if (!isOpen) return undefined;
    const before = document.activeElement;
    panelRef.current?.focus({ preventScroll: true });
    return () => { try { before?.focus?.({ preventScroll: true }); } catch { /* the opener is gone */ } };
  }, [isOpen]);

  const aspect = useMemo(() => aspectOf(sweepResult, natural), [sweepResult, natural]);
  const warnings = useMemo(() => resultWarnings(sweepResult, {
    xLabel: xDef.label, yLabel: yDef.label, xValues, yValues,
  }), [sweepResult, xDef.label, yDef.label, xValues, yValues]);
  const tilesStale = Boolean(sweepResult)
    && (sweepResult.x_param !== xParamId || sweepResult.y_param !== yParamId
      || JSON.stringify(sweepResult.x_values) !== JSON.stringify(xValues) || JSON.stringify(sweepResult.y_values) !== JSON.stringify(yValues));

  // The tile's real pixel size gives the box its aspect; read it once from the first picture.
  useEffect(() => {
    const src = sweepResult?.tiles?.[0]?.image;
    if (!src) return undefined;
    let live = true;
    const img = new Image();
    img.onload = () => { if (live && img.naturalWidth) setNatural({ w: img.naturalWidth, h: img.naturalHeight }); };
    img.src = src;
    return () => { live = false; };
  }, [sweepResult]);

  if (!isOpen) return null;

  const selTile = sweepResult?.tiles?.[selected] || null;
  const canGenerate = !loading;
  const xShown = sweepResult?.x_values || xValues;
  const yShown = sweepResult?.y_values || yValues;

  return createPortal(
    <div
      role="presentation"
      className="fixed inset-0 z-[120] grid place-items-center bg-black/85 p-[5vh_5vw] backdrop-blur-md select-none"
      data-testid="parameter-sweep-overlay"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="parameter-sweep-title"
        tabIndex={-1}
        data-testid="parameter-sweep-modal"
        className="flex h-[90vh] w-[90vw] flex-col overflow-hidden rounded-2xl border border-white/15 bg-[#0d1019] shadow-2xl outline-none"
      >
        {/* header */}
        <div className="flex items-center justify-between gap-3 border-b border-white/10 bg-white/[0.03] px-4 py-2">
          <div className="flex min-w-0 items-center gap-3">
            <span className="grid h-7 w-9 shrink-0 place-items-center rounded-lg border border-cyan-300/40 bg-cyan-300/15 text-sm font-bold text-cyan-200">3×3</span>
            <div className="min-w-0">
              <h2 id="parameter-sweep-title" className="truncate text-sm font-bold tracking-wide text-white">Parameter sweep · 3x3 Contact Sheet</h2>
              <p className="truncate text-micro text-white/60">
                Frame {frame} · one detection, nine swaps · wheel zooms, drag pans, the loupe follows the pointer in all nine cells
              </p>
            </div>
          </div>
          <button type="button" onClick={onClose} aria-label="Close contact sheet" title="Close (Esc)"
            className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-white/70 hover:bg-white/10 hover:text-white">✕</button>
        </div>

        {/* axes + view controls */}
        <div className="flex flex-wrap items-end gap-2 border-b border-white/10 bg-white/[0.015] px-3 py-2">
          <AxisEditor axis="x" title="Parameter X · columns" accent="text-cyan-200" paramId={xParamId} values={xValues}
            otherId={yParamId} onParam={onXParam} onValues={onXValues} />
          <AxisEditor axis="y" title="Parameter Y · rows" accent="text-amber-200" paramId={yParamId} values={yValues}
            otherId={xParamId} onParam={onYParam} onValues={onYValues} />
          <button
            type="button"
            data-testid="sweep-generate"
            onClick={runSweep}
            disabled={!canGenerate}
            className="rounded-lg bg-[var(--accent)] px-3 py-2 text-xs font-bold text-black shadow-md transition hover:brightness-110 active:scale-95 disabled:opacity-50"
          >
            {loading ? 'Rendering 9 cells…' : tilesStale ? 'Regenerate 3×3' : 'Generate 3×3'}
          </button>
          <span className="flex-1" />
          <div className="flex items-center gap-1.5" role="toolbar" aria-label="Synchronized view">
            <span className="font-mono text-micro tabular-nums text-white/70" data-testid="sweep-zoom">{Math.round(view.zoom * 100)}%</span>
            <div className="flex items-center rounded-lg border border-white/15 bg-black/60 p-0.5">
              {[['fit', 'Fit', 'text-white'], ['eyes', 'Eyes', 'text-cyan-200'], ['mouth', 'Mouth', 'text-emerald-200'], ['jaw', 'Jaw', 'text-amber-200']].map(([k, label, tone]) => (
                <button key={k} type="button" data-testid={`sweep-focus-${k}`} onClick={() => focusOn(k)}
                  className={`rounded px-2 py-0.5 text-micro font-semibold hover:bg-white/10 ${tone}`}
                  title={k === 'fit' ? 'Fit all nine cells (F)' : `Zoom all nine cells onto the ${label.toLowerCase()}`}>
                  {label}
                </button>
              ))}
            </div>
            <button
              type="button"
              data-testid="sweep-loupe-toggle"
              aria-pressed={loupeOn}
              onClick={() => setLoupeOn((v) => !v)}
              className={`rounded-lg border px-2 py-1 text-micro font-bold ${loupeOn ? 'border-cyan-300 bg-cyan-300/20 text-cyan-100' : 'border-white/15 bg-black/60 text-white/75 hover:text-white'}`}
              title="Synchronized 3.5× loupe across all nine cells (G)"
            >
              Loupe {LOUPE_POWER}×
            </button>
          </div>
        </div>

        {/* warnings */}
        {(error || warnings.length > 0 || tilesStale) && (
          <div className="space-y-1 px-3 pt-2" data-testid="sweep-warnings">
            {error && <div role="alert" className="rounded-lg border border-rose-400/40 bg-rose-500/15 px-3 py-1.5 text-xs text-rose-100">{error}</div>}
            {warnings.map((w) => (
              <div key={w.kind + w.text} role="status" data-warning={w.kind}
                className="rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-1.5 text-xs text-amber-100">⚠ {w.text}</div>
            ))}
            {tilesStale && !loading && (
              <div role="status" className="rounded-lg border border-white/15 bg-white/[0.04] px-3 py-1 text-micro text-white/75">
                The axes changed since these nine were rendered. Press Regenerate to update them.
              </div>
            )}
          </div>
        )}

        {/* the matrix */}
        <div className="relative min-h-0 flex-1 p-3">
          {loading && !sweepResult && (
            <div className="absolute inset-0 z-20 grid place-items-center bg-black/55 backdrop-blur-sm" role="status">
              <div className="flex flex-col items-center gap-2 text-center">
                <span className="h-8 w-8 animate-spin rounded-full border-2 border-[var(--accent)] border-t-transparent" />
                <p className="text-sm font-semibold text-white">Rendering nine variations of frame {frame}…</p>
                <p className="text-micro text-white/65">The face is detected and aligned once; the swap runs nine times.</p>
              </div>
            </div>
          )}
          <div
            ref={gridRef}
            role="presentation"
            data-testid="sweep-grid"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onPointerLeave={onPointerLeave}
            className={`grid h-full w-full gap-2 ${dragging ? 'cursor-grabbing' : 'cursor-grab'}`}
            style={{ gridTemplateColumns: '4.25rem repeat(3, minmax(0, 1fr))', gridTemplateRows: 'auto repeat(3, minmax(0, 1fr))', touchAction: 'none' }}
          >
            <div />
            {xShown.map((xv, c) => (
              <div key={c} className="truncate rounded-lg border border-cyan-300/25 bg-cyan-300/10 px-2 py-1 text-center text-micro font-bold text-cyan-100" title={`${xDef.label} = ${xv}`}>
                X{c + 1} · {xDef.label} = <span className="font-mono">{String(xv)}</span>
              </div>
            ))}
            {[0, 1, 2].map((r) => (
              <React.Fragment key={r}>
                <div className="grid place-content-center rounded-lg border border-amber-300/25 bg-amber-300/10 px-1 text-center" title={`${yDef.label} = ${yShown[r]}`}>
                  <span className="text-micro font-bold text-amber-100">Y{r + 1}</span>
                  <span className="break-all font-mono text-micro text-white/85">{String(yShown[r])}</span>
                </div>
                {[0, 1, 2].map((c) => {
                  const i = r * 3 + c;
                  return (
                    <Cell
                      key={c}
                      index={i}
                      tile={tileAt(sweepResult, c, r)}
                      xLabel={xDef.label}
                      yLabel={yDef.label}
                      view={view}
                      aspect={aspect}
                      selected={selected === i}
                      committed={committed[i] || null}
                      onSelect={selectCell}
                      onApply={applyCell}
                      registerFrame={registerFrame}
                      registerLens={registerLens}
                      busy={loading && !!sweepResult}
                    />
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* selected cell: commit it */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-white/10 bg-white/[0.03] px-4 py-2.5" data-testid="sweep-actions">
          <div className="min-w-0 text-xs text-white/80" aria-live="polite">
            {selTile
              ? <>Selected <strong className="text-cyan-200">cell #{selected + 1}</strong> · <span className="font-mono">{recipeText(xDef.label, yDef.label, selTile)}</span></>
              : 'No cell selected yet'}
          </div>
          <button
            type="button"
            data-testid="sweep-apply-selected"
            disabled={!selTile || !onApplyRecipe}
            onClick={() => commit(selected, 'recipe')}
            className="rounded-lg bg-[var(--accent)] px-3 py-1.5 text-xs font-bold text-black shadow-md transition hover:brightness-110 active:scale-95 disabled:opacity-40"
            title="Write this cell's two parameter values into the active recipe (Enter)"
          >
            Apply Cell to Active Recipe
          </button>
          {onAssignSlot && (
            <div className="flex items-center gap-1.5" role="group" aria-label="Assign the selected cell to a snapshot slot">
              <span className="text-micro font-bold uppercase tracking-wide text-white/70">Assign to Snapshot Slot</span>
              {SLOT_ORDER.map((id) => {
                const filled = Boolean(slots?.[id]);
                return (
                  <button
                    key={id}
                    type="button"
                    data-testid={`sweep-slot-${id}`}
                    disabled={!selTile}
                    onClick={() => commit(selected, 'slot', id)}
                    className="relative rounded-lg border border-white/20 bg-white/[0.06] px-2.5 py-1 text-xs font-bold text-white hover:border-cyan-300/70 hover:bg-cyan-300/15 disabled:opacity-40"
                    title={`${filled ? 'Replace' : 'Fill'} Slot ${id} with this cell's recipe (key ${id})${filled ? ' · the slot already holds a recipe' : ''}`}
                  >
                    {id}
                    {filled && <span aria-hidden="true" className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-cyan-300" />}
                    {filled && <span className="sr-only"> (filled)</span>}
                  </button>
                );
              })}
            </div>
          )}
          <span className="flex-1" />
          <span data-testid="sweep-status" role="status" aria-live="polite" className="min-w-0 max-w-[48%] truncate text-micro text-cyan-100" title={status}>{status}</span>
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-white/10 bg-white/[0.02] px-4 py-1.5 text-micro text-white/60">
          <span>Click selects · double-click or Enter applies · 1–9 apply that cell · A–D assign to a slot</span>
          <span>G loupe · F fit · Esc close</span>
        </div>
      </div>
    </div>,
    document.body,
  );
}
