import React, { useEffect, useMemo, useRef, useState } from 'react';
import NumericInput from '../../NumericInput';
import { useLaneCanvas } from '../timeline/useLaneCanvas';
import { frameAtX, normalizeView, viewSpan } from '../timeline/viewMath';
import {
  ALL_CURVE_PARAMS, DEFAULT_INTERP, INTERPS, INTERP_LABELS, SNAP_FRAME_STEPS, SNAP_VALUE_DIVS,
  clampValue, curveIneffective, handleFromPoint, handlePoint, keysInBox, makeEvaluator, moveKeys, normValue, paramById,
  segmentHandles, selKey, setSegmentBezier, shiftBounds, snapFrame, snapValue, tangents,
} from './curveModel';

// The Curve Editor drawer: parameter values against the frame, on the timeline's own x-axis.
//
// It sits in the lane stack, so it shares the stack's width and the timeline's view window: zoom or scroll the
// timeline and the graph follows, key for key. The lines are drawn on a canvas (they can be thousands of pixels of
// samples); the keyframes and tangent handles are real buttons over it (at most MAX_KEYS = 128 a curve), so they are
// keyboard- and screen-reader-reachable, which a canvas is not. Positions are percentages of the graph in x and fixed
// layout pixels in y, so the app-level CSS zoom cannot offset them; pointer positions are turned into frames and values
// from ratios of the graph's own rect for the same reason.
//
// STAGE 12 (the drawer):
//   Overlay       every curve that has keys is drawn at once, colour-coded, each on its own 0..100% scale (their units
//                 differ). One is ACTIVE: drawn bold, its axis labels shown, its keys the large handles. A chip's eye
//                 hides / shows a curve; clicking a dot of another curve makes it the active one.
//   Handles       the segment on either side of the selected key shows two Bezier tangent handles. Dragging one turns the
//                 segment into Custom Bezier; the move commits on release (one undo step).
//   Grid          a value grid (the range in 4 / 10 / 20 steps) and a frame grid (5 / 10 / 25) that drags snap to. Hold
//                 Alt to drag free.
//   Selection     drag on empty graph to box-select keys (Shift adds); Shift-click toggles one; Ctrl/Cmd+A selects every
//                 visible key. A selected group drags, nudges, deletes and re-interpolates as one (one undo step).
//
// Gestures kept from Stage 25: click the graph to park the playhead; double-click to key the value under the pointer;
// with a key focused, Left / Right nudge a frame (Shift: ten), Up / Down nudge the value (Shift: ten steps), Delete removes.

const EMPTY = Object.freeze([]);
const H_COMPACT = 168;
const H_EXPANDED = 340;
const PAD_Y = 14;
const HANDLE = 12;
const DOT = 8;
const TANGENT = 10;
const smallBtn = 'rounded-md px-1.5 py-0.5 text-note font-semibold text-[var(--text-muted)] hover:bg-white/[0.06] hover:text-[var(--text-main)] disabled:opacity-40 transition-colors';
const selectCls = 'rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] px-1 py-0.5 text-note text-[var(--text-main)] outline-none focus:border-[var(--accent)]';

const fmtValue = (param, v) => `${Number(v).toFixed(param.digits)}${param.unit}`;
const lsGet = (k, d) => { try { const v = window.localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } };
const lsSet = (k, v) => { try { window.localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable: the choice lasts for this session */ } };

/** A segment-shape glyph for the interpolation buttons. */
const InterpGlyph = ({ kind }) => {
  const d = {
    hold: 'M2 14 H10 V4 H18', linear: 'M2 14 L18 4', ease: 'M2 14 C9 14 11 4 18 4',
    smooth: 'M2 14 C7 14 7 9 10 9 C13 9 13 4 18 4', bezier: 'M2 14 C2 3 18 17 18 4',
  }[kind];
  return (
    <svg width="20" height="18" viewBox="0 0 20 18" aria-hidden="true">
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      {kind === 'bezier' && <><circle cx="2" cy="3" r="1.6" fill="currentColor" /><circle cx="18" cy="17" r="1.6" fill="currentColor" /></>}
    </svg>
  );
};

export default function CurveEditor({
  view, maxFrames, frame, tracks, enabled, onEnabledChange, settings, api, onSeek, onClose, morph = null, planNote = '',
}) {
  const v = normalizeView(view, maxFrames);
  const span = viewSpan(v);
  const [paramId, setParamId] = useState(() => ALL_CURVE_PARAMS.find((p) => tracks?.[p.id]?.length)?.id || ALL_CURVE_PARAMS[0].id);
  const param = paramById(paramId);
  const list = tracks?.[paramId] || EMPTY;
  const [hidden, setHidden] = useState(() => new Set());       // curves the user hid from the overlay
  const [sel, setSel] = useState(() => new Set());             // selected keys: "paramId:frame"
  const [primary, setPrimary] = useState(null);                // the key the handles, value box and interp row speak about
  const [ghost, setGhost] = useState(null);                    // a drag in flight: { kind: 'keys', dFrame, dNorm } | { kind: 'handle', paramId, frame, bz }
  const [marquee, setMarquee] = useState(null);                // { x0, y0, x1, y1 } in graph px
  const [expanded, setExpanded] = useState(() => lsGet('roop_curve_drawer_expanded', false) === true);
  const [snap, setSnap] = useState(() => {
    const s = lsGet('roop_curve_snap', null);
    return { value: SNAP_VALUE_DIVS.includes(s?.value) ? s.value : 0, frame: SNAP_FRAME_STEPS.includes(s?.frame) ? s.frame : 1 };
  });
  const graphRef = useRef(null);
  const dragRef = useRef(null);
  const marqueeRef = useRef(null);
  const suppressClick = useRef(false);
  const H = expanded ? H_EXPANDED : H_COMPACT;

  const yOfNorm = (n) => PAD_Y + (1 - n) * (H - 2 * PAD_Y);
  const yOf = (p, val) => yOfNorm(normValue(p, val));
  const normAtY = (yPx) => Math.max(0, Math.min(1, 1 - (yPx - PAD_Y) / (H - 2 * PAD_Y)));
  const valueAtNorm = (p, n) => p.min + n * (p.max - p.min);

  const primaryParsed = useMemo(() => {
    if (!primary) return null;
    const at = primary.lastIndexOf(':');
    return { paramId: primary.slice(0, at), frame: Number(primary.slice(at + 1)) };
  }, [primary]);
  const primaryKey = primaryParsed ? (tracks?.[primaryParsed.paramId] || EMPTY).find((k) => k.frame === primaryParsed.frame) || null : null;
  const primaryParam = primaryParsed ? paramById(primaryParsed.paramId) : null;

  // A selection that no longer exists (undo, a removed key, a cleared curve) is dropped, not left dangling.
  useEffect(() => {
    setSel((cur) => {
      let drop = false;
      const next = new Set();
      for (const s of cur) {
        const at = s.lastIndexOf(':');
        const ok = (tracks?.[s.slice(0, at)] || EMPTY).some((k) => k.frame === Number(s.slice(at + 1)));
        if (ok) next.add(s); else drop = true;
      }
      return drop ? next : cur;
    });
    setPrimary((cur) => {
      if (!cur) return cur;
      const at = cur.lastIndexOf(':');
      return (tracks?.[cur.slice(0, at)] || EMPTY).some((k) => k.frame === Number(cur.slice(at + 1))) ? cur : null;
    });
  }, [tracks]);

  // The overlay: every curve with keys, plus the active one, minus the ones the user hid.
  const visibleIds = useMemo(() => ALL_CURVE_PARAMS
    .filter((p) => p.id === paramId || (tracks?.[p.id]?.length && !hidden.has(p.id)))
    .map((p) => p.id), [paramId, tracks, hidden]);

  // What is DRAWN: the tracks with the drag in flight applied. Nothing is committed until release.
  const shown = useMemo(() => {
    if (!ghost) return tracks || {};
    if (ghost.kind === 'keys') return moveKeys(tracks || {}, ghost.sel, ghost.dFrame, ghost.dNorm, maxFrames);
    if (ghost.kind === 'handle') {
      const cur = tracks?.[ghost.paramId] || EMPTY;
      const next = setSegmentBezier(cur, ghost.frame, ghost.bz);
      return next === cur ? tracks : { ...tracks, [ghost.paramId]: next };
    }
    return tracks || {};
  }, [tracks, ghost, maxFrames]);
  const evals = useMemo(() => Object.fromEntries(visibleIds.map((id) => [id, makeEvaluator(shown[id] || EMPTY)])), [visibleIds, shown]);
  const shownList = shown[paramId] || EMPTY;

  const baseValue = Number.isFinite(Number(settings?.[param.key])) ? Number(settings[param.key]) : param.base;
  const gridDiv = snap.value > 0 ? snap.value : 4;

  const canvasRef = useLaneCanvas((ctx, { w, h, theme }) => {
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.textBaseline = 'middle';
    // Value grid. With snapping on these ARE the lines a drag lands on, so they are drawn stronger.
    for (let i = 0; i <= gridDiv; i++) {
      const n = i / gridDiv;
      const y = yOfNorm(n);
      ctx.strokeStyle = theme.border;
      ctx.globalAlpha = snap.value > 0 ? 1 : 0.7;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke();
      ctx.globalAlpha = 1;
      if (i % (gridDiv >= 10 ? 5 : 1) === 0) {
        ctx.fillStyle = theme.muted;
        ctx.fillText(fmtValue(param, valueAtNorm(param, n)), 4, Math.min(h - 6, Math.max(6, y - 6)));
      }
    }
    // Frame grid, when snapping to it.
    if (snap.frame > 1) {
      ctx.strokeStyle = theme.border;
      ctx.globalAlpha = 0.5;
      const first = Math.ceil(v.start / snap.frame) * snap.frame;
      for (let f = first; f <= v.end; f += snap.frame) {
        const x = ((f - v.start) / span) * w;
        ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, h); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    // The clip's own setting for the active curve: where the parameter sits when no curve is driving it.
    const by = yOf(param, clampValue(param, baseValue));
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = theme.muted;
    ctx.beginPath(); ctx.moveTo(0, by + 0.5); ctx.lineTo(w, by + 0.5); ctx.stroke();
    ctx.restore();
    // Every visible curve, the active one last (on top) and bold.
    const order = [...visibleIds.filter((id) => id !== paramId), paramId];
    for (const id of order) {
      const p = paramById(id);
      const keys = shown[id] || EMPTY;
      if (!p || !keys.length) continue;
      const active = id === paramId;
      const ev = evals[id];
      const first = keys[0].frame;
      const last = keys[keys.length - 1].frame;
      ctx.lineWidth = active ? 2.2 : 1.4;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = p.color;
      let prevOut = null;
      let path = null;
      let lx = 0;
      let ly = 0;
      const flush = (out) => {
        if (!path) return;
        ctx.globalAlpha = (out ? 0.35 : 1) * (active ? 1 : 0.55);
        ctx.stroke(path);
      };
      for (let x = 0; x <= w; x += 2) {
        const f = v.start + (x / Math.max(1, w)) * span;
        const out = f < first || f > last;
        const y = yOf(p, ev(f));
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
    }
  }, [v.start, v.end, param, paramId, shown, evals, visibleIds, baseValue, enabled, snap.value, snap.frame, gridDiv, H]);

  const pct = (f) => ((f - v.start) / span) * 100;
  const inView = (f) => f >= v.start - 1 && f <= v.end + 1;

  const rectOf = () => graphRef.current?.getBoundingClientRect() || null;
  // Pointer -> frame and 0..1 position (the active curve's value follows from it).
  const pointer = (e, forParam = param) => {
    const r = rectOf();
    if (!r || !r.width || !r.height) return null;
    const n = normAtY(((e.clientY - r.top) / r.height) * H);
    return { frame: frameAtX(e.clientX - r.left, r.width, v, maxFrames), norm: n, value: clampValue(forParam, valueAtNorm(forParam, n)) };
  };
  const snapped = (e, p, pt) => (e.altKey ? { frame: pt.frame, value: pt.value }
    : { frame: snapFrame(pt.frame, snap.frame, maxFrames), value: snapValue(p, pt.value, snap.value) });

  // A first key keeps the value the slider has, so adding one changes nothing by itself.
  const keyHere = (f, value) => {
    const out = api.add(paramId, f, value === undefined && !list.length ? clampValue(param, baseValue) : value);
    if (out?.created) { const k = selKey(paramId, out.frame); setSel(new Set([k])); setPrimary(k); }
    else if (out?.full) return false;
    return out;
  };

  const onGraphClick = (e) => {
    if (suppressClick.current) { suppressClick.current = false; return; }
    const pt = pointer(e);
    if (pt) onSeek?.(pt.frame);
    if (!e.shiftKey) { setSel(new Set()); setPrimary(null); }
  };
  const onGraphDouble = (e) => {
    const pt = pointer(e);
    if (!pt) return;
    const s = snapped(e, param, pt);
    keyHere(s.frame, s.value);
  };

  // ── box selection ──────────────────────────────────────────────────────────
  const hitDown = (e) => {
    if (e.button !== 0) return;
    const r = rectOf();
    if (!r) return;
    marqueeRef.current = { x0: e.clientX - r.left, y0: e.clientY - r.top, additive: e.shiftKey, moved: false, id: e.pointerId };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const hitMove = (e) => {
    const m = marqueeRef.current;
    const r = rectOf();
    if (!m || !r || !(e.buttons & 1)) return;
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    if (!m.moved && Math.hypot(x - m.x0, y - m.y0) < 4) return;       // a click is not a box
    m.moved = true;
    setMarquee({ x0: m.x0, y0: m.y0, x1: x, y1: y });
  };
  const hitUp = (e) => {
    const m = marqueeRef.current;
    marqueeRef.current = null;
    try { e.currentTarget.releasePointerCapture?.(e.pointerId); } catch { /* already released */ }
    setMarquee(null);
    const r = rectOf();
    if (!m?.moved || !r) return;
    suppressClick.current = true;                                      // the click that follows a box is not a seek
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const f0 = frameAtX(Math.min(m.x0, x), r.width, v, maxFrames);
    const f1 = frameAtX(Math.max(m.x0, x), r.width, v, maxFrames);
    const n0 = normAtY((Math.max(m.y0, y) / r.height) * H);
    const n1 = normAtY((Math.min(m.y0, y) / r.height) * H);
    const hit = keysInBox(tracks, visibleIds, { f0, f1, n0, n1 });
    const next = m.additive ? new Set([...sel, ...hit]) : hit;
    setSel(next);
    setPrimary(next.size ? [...next][0] : null);
    // A box that caught only other curves' keys makes the first of those curves the active one.
    const ids = [...next];
    if (ids.length && !ids.some((s) => s.startsWith(`${paramId}:`))) setParamId(ids[0].slice(0, ids[0].lastIndexOf(':')));
  };

  // ── key drag (one key or the whole selection) ──────────────────────────────
  const down = (k, pid) => (e) => {
    e.stopPropagation();
    const id = selKey(pid, k.frame);
    if (e.shiftKey) {                                                  // Shift-click toggles; it does not start a drag
      setSel((cur) => { const n = new Set(cur); if (n.has(id)) n.delete(id); else n.add(id); return n; });
      setPrimary(id);
      return;
    }
    e.currentTarget.setPointerCapture?.(e.pointerId);
    const group = sel.has(id) ? sel : new Set([id]);
    if (!sel.has(id)) setSel(group);
    setPrimary(id);
    // Not setParamId here: another curve's dot and the active curve's key are different buttons, and swapping them
    // mid-press would drop the pointer capture. A plain press switches curves on release; a drag never does.
    dragRef.current = { anchor: { pid, frame: k.frame, value: k.value }, group, moved: false, x: e.clientX, y: e.clientY };
    onSeek?.(k.frame);
  };
  const dragDelta = (e, d) => {
    const ap = paramById(d.anchor.pid);
    const pt = pointer(e, ap);
    if (!pt) return null;
    const s = snapped(e, ap, pt);
    return { dFrame: s.frame - d.anchor.frame, dNorm: normValue(ap, s.value) - normValue(ap, d.anchor.value) };
  };
  const drag = (e) => {
    const d = dragRef.current;
    if (!d || !(e.buttons & 1)) return;
    if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 3) return;      // a click is not a drag
    d.moved = true;
    const dl = dragDelta(e, d);
    if (dl) setGhost({ kind: 'keys', sel: d.group, ...dl });
  };
  // The final position comes from the release event itself, not from the last ghost: a quick flick can deliver the
  // release before React has re-rendered with the newest ghost.
  const up = (e) => {
    const d = dragRef.current;
    dragRef.current = null;
    try { e.currentTarget.releasePointerCapture?.(e.pointerId); } catch { /* already released */ }
    setGhost(null);
    if (d && !d.moved && d.anchor && d.anchor.pid !== paramId) setParamId(d.anchor.pid);
    if (!d?.moved) return;
    const dl = dragDelta(e, d);
    if (!dl) return;
    const b = shiftBounds(tracks, d.group, maxFrames);
    const landed = Math.max(b.min, Math.min(b.max, dl.dFrame));
    if (api.moveMany(d.group, dl.dFrame, dl.dNorm)) {
      const remap = (s) => { const at = s.lastIndexOf(':'); return selKey(s.slice(0, at), Number(s.slice(at + 1)) + landed); };
      setSel(new Set([...d.group].map(remap)));
      setPrimary((cur) => (cur ? remap(cur) : cur));
    }
  };

  // ── tangent handles ────────────────────────────────────────────────────────
  const handleSegs = useMemo(() => {
    if (!primaryParsed || primaryParsed.paramId !== paramId) return EMPTY;
    const i = shownList.findIndex((k) => k.frame === primaryParsed.frame);
    if (i < 0) return EMPTY;
    const tans = tangents(shownList);
    const out = [];
    for (const s of [i - 1, i]) {
      if (s < 0) continue;
      const bz = segmentHandles(shownList, s, tans);
      if (bz) out.push({ s, a: shownList[s], b: shownList[s + 1], bz });
    }
    return out;
  }, [primaryParsed, paramId, shownList]);

  const hDown = (seg, which) => (e) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    dragRef.current = { handle: { frame: seg.a.frame, which, a: seg.a, b: seg.b, bz: seg.bz }, moved: false, x: e.clientX, y: e.clientY };
  };
  const handleTarget = (e, h) => {
    const pt = pointer(e);
    if (!h || !pt) return null;
    let [hx, hy] = handleFromPoint(h.a, h.b, pt.frame, pt.value);
    if (!e.altKey && snap.value > 0) {                                  // the unit square snaps to the value grid's step
      hx = Math.round(hx * snap.value) / snap.value;
      hy = Math.round(hy * snap.value) / snap.value;
    }
    return h.which === 1 ? [hx, hy, h.bz[2], h.bz[3]] : [h.bz[0], h.bz[1], hx, hy];
  };
  const hMove = (e) => {
    const d = dragRef.current;
    if (!d?.handle || !(e.buttons & 1)) return;
    if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 2) return;
    d.moved = true;
    const bz = handleTarget(e, d.handle);
    if (bz) setGhost({ kind: 'handle', paramId, frame: d.handle.frame, bz });
  };
  const hUp = (e) => {
    const d = dragRef.current;
    dragRef.current = null;
    try { e.currentTarget.releasePointerCapture?.(e.pointerId); } catch { /* already released */ }
    if (!d?.handle) return;
    const bz = d.moved ? handleTarget(e, d.handle) : null;
    setGhost(null);
    if (bz) api.setBezier(paramId, d.handle.frame, bz);
  };

  // ── keyboard ───────────────────────────────────────────────────────────────
  const nudge = (k, pid) => (e) => {
    const p = paramById(pid);
    const id = selKey(pid, k.frame);
    const group = sel.has(id) && sel.size > 1 ? sel : null;
    const big = e.shiftKey ? 10 : 1;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      const dir = e.key === 'ArrowLeft' ? -1 : 1;
      if (group) {
        const b = shiftBounds(tracks, group, maxFrames);
        const landed = Math.max(b.min, Math.min(b.max, dir * big));
        if (api.moveMany(group, dir * big, 0)) {
          const remap = (s) => { const at = s.lastIndexOf(':'); return selKey(s.slice(0, at), Number(s.slice(at + 1)) + landed); };
          setSel(new Set([...group].map(remap)));
          setPrimary((cur) => (cur ? remap(cur) : cur));
        }
        return;
      }
      const to = k.frame + dir * big;
      if (api.move(pid, k.frame, to, k.value)) {
        const bnds = (tracks?.[pid] || EMPTY);
        const i = bnds.findIndex((x) => x.frame === k.frame);
        const lo = i > 0 ? bnds[i - 1].frame + 1 : 1;
        const hi = i < bnds.length - 1 ? bnds[i + 1].frame - 1 : Math.max(1, maxFrames);
        const landed = Math.max(lo, Math.min(hi, to));
        const nid = selKey(pid, landed);
        setSel(new Set([nid])); setPrimary(nid);
        requestAnimationFrame(() => graphRef.current?.querySelector(`[data-testid="curve-key-${landed}"]`)?.focus());
      }
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      const dir = e.key === 'ArrowUp' ? 1 : -1;
      if (group) api.moveMany(group, 0, (dir * big * p.quantum) / (p.max - p.min));
      else api.setValue(pid, k.frame, k.value + dir * big * p.quantum);
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      if (group) { api.removeMany(group); setSel(new Set()); setPrimary(null); } else api.remove(pid, k.frame);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setSel(new Set([id])); setPrimary(id);
      onSeek?.(k.frame);
    }
  };
  const selectAll = () => {
    const all = new Set();
    for (const id of visibleIds) for (const k of tracks?.[id] || EMPTY) all.add(selKey(id, k.frame));
    setSel(all);
    setPrimary(all.size ? [...all][0] : null);
  };
  const onEditorKeyDown = (e) => {
    const typing = e.target?.tagName === 'INPUT' || e.target?.tagName === 'SELECT' || e.target?.isContentEditable;
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'a' && !typing) { e.preventDefault(); selectAll(); }
    else if (e.key === 'Escape' && sel.size && !typing) { e.preventDefault(); e.stopPropagation(); setSel(new Set()); setPrimary(null); }
  };

  const setSnapPart = (part) => setSnap((cur) => { const next = { ...cur, ...part }; lsSet('roop_curve_snap', next); return next; });
  const toggleExpanded = () => setExpanded((x) => { lsSet('roop_curve_drawer_expanded', !x); return !x; });
  const toggleHidden = (id) => setHidden((cur) => { const n = new Set(cur); if (n.has(id)) n.delete(id); else n.add(id); return n; });

  // The interpolation row speaks about the group when several keys are selected, else about the one key.
  const group = sel.size > 1 ? sel : null;
  const singleKey = !group && primaryKey ? primaryKey : null;
  const singleList = singleKey ? (tracks?.[primaryParsed.paramId] || EMPTY) : EMPTY;
  const singleIsLast = singleKey && singleList[singleList.length - 1] === singleKey;
  const groupInterps = group ? new Set([...group].map((s) => {
    const at = s.lastIndexOf(':');
    return (tracks?.[s.slice(0, at)] || EMPTY).find((k) => k.frame === Number(s.slice(at + 1)))?.interp;
  })) : null;
  const interpPressed = (kind) => (group ? groupInterps.size === 1 && groupInterps.has(kind) : singleKey?.interp === kind);
  const setInterpTo = (kind) => {
    if (group) api.interpMany(group, kind);
    else if (singleKey) api.interp(primaryParsed.paramId, singleKey.frame, kind);
  };

  const ineffective = list.length && enabled ? curveIneffective(param, settings) : null;
  const atPlayhead = list.some((k) => k.frame === frame);
  const selectedHere = (pid, f) => sel.has(selKey(pid, f));

  return (
    // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions -- the group only catches keys bubbling up from its own buttons (Ctrl+A, Escape)
    <div className="space-y-1.5" data-testid="curve-editor" role="group" aria-label="Parameter curve editor" onKeyDown={onEditorKeyDown}>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 pt-1.5">
        <span className="text-nano font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Curve editor</span>
        <div className="flex flex-wrap items-center gap-0.5" role="tablist" aria-label="Parameter">
          {ALL_CURVE_PARAMS.map((p) => {
            const on = p.id === paramId;
            const n = tracks?.[p.id]?.length || 0;
            const shownInOverlay = visibleIds.includes(p.id);
            return (
              <span key={p.id} className={`inline-flex items-center rounded-md ${on ? 'bg-[var(--accent)]/15' : ''}`}>
                <button type="button" role="tab" aria-selected={on} data-testid={`curve-param-${p.id}`}
                        onClick={() => { setParamId(p.id); setGhost(null); }}
                        className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-note font-semibold transition-colors ${
                          on ? 'text-accent' : 'text-[var(--text-muted)] hover:bg-white/[0.06] hover:text-[var(--text-main)]'}`}>
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: p.color, opacity: n ? 1 : 0.35 }} />
                  {p.label}
                  {n > 0 && <span className="font-mono tabular-nums opacity-70">{n}</span>}
                </button>
                {n > 0 && !on && (
                  <button type="button" data-testid={`curve-eye-${p.id}`} aria-pressed={shownInOverlay}
                          aria-label={`${shownInOverlay ? 'Hide' : 'Show'} ${p.label} in the overlay`}
                          title={`${shownInOverlay ? 'Hide' : 'Show'} the ${p.label} curve in the overlay`}
                          onClick={() => toggleHidden(p.id)}
                          className={`mr-0.5 grid h-4 w-4 place-items-center rounded text-nano leading-none ${shownInOverlay ? 'text-[var(--text-main)]' : 'text-[var(--text-muted)] opacity-50'}`}>
                    {shownInOverlay ? '◉' : '○'}
                  </button>
                )}
              </span>
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
                onClick={() => { api.clearParam(paramId); }} title={`Remove every ${param.label} keyframe on this clip`}>
          Clear curve
        </button>
        <button type="button" className={smallBtn} data-testid="curve-expand" aria-pressed={expanded} onClick={toggleExpanded}
                title={expanded ? 'Shrink the curve drawer' : 'Expand the curve drawer for finer work'}>
          {expanded ? 'Collapse' : 'Expand'}
        </button>
        <button type="button" className={smallBtn} data-testid="curve-close" onClick={onClose} title="Close the curve editor (C)">Close</button>
      </div>

      {/* Interpolation of the segment AFTER the selected key(s), the snap grids, and the exact value */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-note text-[var(--text-muted)]" data-testid="curve-sel-text">
          {group ? `${group.size} keys selected - set the segment after each:`
            : singleKey ? `Key at frame ${singleKey.frame} - the segment after it:` : 'Select a key to set how the curve leaves it:'}
        </span>
        <div className="inline-flex items-center gap-0.5 rounded-lg border border-[var(--border-color)] bg-[var(--surface-2)] p-0.5" role="group" aria-label="Interpolation">
          {INTERPS.map((kind) => (
            <button key={kind} type="button" data-testid={`curve-interp-${kind}`}
                    disabled={!group && (!singleKey || singleIsLast)} aria-pressed={!!interpPressed(kind)}
                    onClick={() => setInterpTo(kind)}
                    title={singleIsLast ? 'The last key has no segment after it' : INTERP_LABELS[kind]}
                    aria-label={INTERP_LABELS[kind]}
                    className={`grid h-6 w-7 place-items-center rounded-md transition-colors disabled:opacity-30 ${
                      interpPressed(kind) ? 'bg-[var(--accent)] text-white' : 'text-[var(--text-muted)] hover:bg-white/[0.06] hover:text-[var(--text-main)]'}`}>
              <InterpGlyph kind={kind} />
            </button>
          ))}
        </div>
        {singleKey && !singleIsLast && <span className="text-note text-[var(--text-muted)]">{INTERP_LABELS[singleKey.interp || DEFAULT_INTERP]}</span>}
        <span className="flex-1" />
        <label className="inline-flex items-center gap-1 text-note text-[var(--text-muted)]" title="Drags land on this many equal steps of the curve's range. Hold Alt to drag free.">
          Snap value
          <select className={selectCls} data-testid="curve-snap-value" value={snap.value} onChange={(e) => setSnapPart({ value: Number(e.target.value) })} aria-label="Value grid snapping">
            {SNAP_VALUE_DIVS.map((d) => <option key={d} value={d}>{d === 0 ? 'Off' : `1/${d}`}</option>)}
          </select>
        </label>
        <label className="inline-flex items-center gap-1 text-note text-[var(--text-muted)]" title="Drags land on multiples of this many frames. Hold Alt to drag free.">
          frame
          <select className={selectCls} data-testid="curve-snap-frame" value={snap.frame} onChange={(e) => setSnapPart({ frame: Number(e.target.value) })} aria-label="Frame grid snapping">
            {SNAP_FRAME_STEPS.map((d) => <option key={d} value={d}>{d === 1 ? 'Off' : `${d} f`}</option>)}
          </select>
        </label>
        <button type="button" className={smallBtn} data-testid="curve-select-all" onClick={selectAll} title="Select every key of the visible curves (Ctrl/Cmd+A)">Select all</button>
        {singleKey && primaryParam && (
          <NumericInput
            label="Value"
            testId="curve-value"
            ariaLabel={`${primaryParam.label} value at the selected key`}
            value={singleKey.value}
            min={primaryParam.min}
            max={primaryParam.max}
            step={primaryParam.quantum}
            unit={primaryParam.unit}
            onCommit={(n) => api.setValue(primaryParsed.paramId, singleKey.frame, n)}
            className="gap-1.5 text-note text-[var(--text-muted)]"
            inputClassName="w-[8ch] rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] py-0.5 text-center font-mono text-mini tabular-nums text-[var(--text-main)] outline-none focus:border-[var(--accent)]"
          />
        )}
      </div>

      <div
        ref={graphRef}
        data-testid="curve-graph"
        className="relative w-full rounded-md border border-[var(--border-color)] bg-[var(--input-bg)]"
        style={{ height: H, opacity: enabled ? 1 : 0.55 }}
      >
        {/* Not clipped: the x-axis must be exactly the lanes' (frame 1 is the left edge), so a key on the first or
            last frame sits ON the edge and its handle has to overhang it rather than lose half to overflow. */}
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full rounded-md" aria-hidden="true" />
        {/* The pointer layer: click parks the playhead, double-click keys the value under the pointer, a drag draws a
            selection box. A mouse affordance only, so it is presentation; the keyboard routes are "+ Key at playhead",
            "Select all" and the key buttons themselves (focus one: arrows nudge, Delete removes). */}
        <div role="presentation" data-testid="curve-hit" className="absolute inset-0 touch-none" style={{ cursor: 'crosshair' }}
             onClick={onGraphClick} onDoubleClick={onGraphDouble}
             onPointerDown={hitDown} onPointerMove={hitMove} onPointerUp={hitUp}
             onPointerCancel={() => { marqueeRef.current = null; setMarquee(null); }} />
        {marquee && (
          <div data-testid="curve-marquee" aria-hidden="true" className="pointer-events-none absolute border border-dashed border-[var(--accent)] bg-[var(--accent)]/10"
               style={{ left: Math.min(marquee.x0, marquee.x1), top: Math.min(marquee.y0, marquee.y1), width: Math.abs(marquee.x1 - marquee.x0), height: Math.abs(marquee.y1 - marquee.y0) }} />
        )}
        {!list.length && !visibleIds.some((id) => tracks?.[id]?.length) && (
          <span className="pointer-events-none absolute inset-x-0 top-2 px-3 text-center text-note text-[var(--text-muted)]">
            No keyframes: {param.label} follows the slider ({fmtValue(param, baseValue)}). Double-click to key a value here.
          </span>
        )}

        {/* Tangent handles of the segments beside the selected key */}
        {handleSegs.length > 0 && (
          <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible" aria-hidden="true">
            {handleSegs.map(({ s, a, b, bz }) => {
              const p1 = handlePoint(a, b, bz[0], bz[1]);
              const p2 = handlePoint(a, b, bz[2], bz[3]);
              return (
                <g key={s} stroke={param.color} strokeWidth="1" strokeDasharray="3 3" opacity="0.8">
                  <line x1={`${pct(a.frame)}%`} y1={yOf(param, a.value)} x2={`${pct(p1.frame)}%`} y2={yOf(param, p1.value)} />
                  <line x1={`${pct(b.frame)}%`} y1={yOf(param, b.value)} x2={`${pct(p2.frame)}%`} y2={yOf(param, p2.value)} />
                </g>
              );
            })}
          </svg>
        )}
        {handleSegs.map(({ s, a, b, bz }) => [1, 2].map((which) => {
          const pt = handlePoint(a, b, bz[which === 1 ? 0 : 2], bz[which === 1 ? 1 : 3]);
          return (
            <button key={`${s}-${which}`} type="button"
                    data-testid={`curve-handle-${a.frame}-${which}`}
                    aria-label={`Bezier handle ${which} of the segment from frame ${a.frame} to frame ${b.frame}`}
                    title="Drag to shape the segment (Alt: no snapping)"
                    onPointerDown={hDown({ a, b, bz }, which)} onPointerMove={hMove} onPointerUp={hUp}
                    onPointerCancel={() => { dragRef.current = null; setGhost(null); }}
                    onClick={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}
                    className="absolute touch-none outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
                    style={{
                      left: `${pct(pt.frame)}%`, top: yOf(param, pt.value), width: TANGENT, height: TANGENT,
                      transform: 'translate(-50%, -50%) rotate(45deg)', background: 'var(--card-bg)', border: `2px solid ${param.color}`, cursor: 'grab',
                    }} />
          );
        }))}

        {/* The other visible curves' keys: small dots. Press one to make its curve the active one. */}
        {visibleIds.filter((id) => id !== paramId).flatMap((id) => {
          const p = paramById(id);
          const origList = tracks?.[id] || EMPTY;
          return (shown[id] || EMPTY).map((k, i) => ({ k, orig: origList[i] || k })).filter(({ k }) => inView(k.frame)).map(({ k, orig }) => (
            <button key={`${id}-${orig.frame}`} type="button" data-testid={`curve-dot-${id}-${k.frame}`}
                    aria-label={`${p.label} keyframe at frame ${k.frame}, value ${fmtValue(p, k.value)}. Press to edit this curve`}
                    title={`${p.label} - frame ${k.frame}, ${fmtValue(p, k.value)}`}
                    onPointerDown={down(orig, id)} onPointerMove={drag} onPointerUp={up}
                    onPointerCancel={() => { dragRef.current = null; setGhost(null); }}
                    onKeyDown={nudge(orig, id)}
                    onClick={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}
                    className={`absolute touch-none rounded-full outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] ${selectedHere(id, orig.frame) ? 'border-2 border-white' : 'border border-black/50'}`}
                    style={{ left: `${pct(k.frame)}%`, top: yOf(p, k.value), width: DOT, height: DOT, transform: 'translate(-50%, -50%)', background: p.color, opacity: 0.85, cursor: 'pointer' }} />
          ));
        })}

        {/* The active curve's keys: real buttons (focus one: arrows nudge, Delete removes) */}
        {shownList.map((k, i) => {
          if (!inView(k.frame)) return null;
          // `list[i]` is the same key before the drag in flight moved it (a group move keeps the keys' order and count):
          // its ORIGINAL frame is what selection and React keys are about, so the button under the pointer is never remounted.
          const orig = list[i] || k;
          const id = selKey(paramId, orig.frame);
          const on = sel.has(id);
          return (
            <button key={id} type="button"
                    data-testid={`curve-key-${k.frame}`}
                    aria-label={`${param.label} keyframe at frame ${k.frame}, value ${fmtValue(param, k.value)}, ${INTERP_LABELS[k.interp || DEFAULT_INTERP]} to the next`}
                    aria-pressed={!!on}
                    title={`Frame ${k.frame} - ${fmtValue(param, k.value)}. Drag to move (Shift-click adds to a selection), arrows nudge, Delete removes.`}
                    onPointerDown={down(orig, paramId)}
                    onPointerMove={drag} onPointerUp={up}
                    onPointerCancel={() => { dragRef.current = null; setGhost(null); }}
                    onKeyDown={nudge(orig, paramId)}
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
        The preview follows the curves exactly. Dashed line: the clip&apos;s own {param.label} setting, used wherever there is no curve.
        Every curve is drawn on its own 0-100% scale; axis labels are for {param.label}.
      </p>
    </div>
  );
}
