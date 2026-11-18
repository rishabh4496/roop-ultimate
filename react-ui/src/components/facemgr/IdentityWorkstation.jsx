import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getJSON, postJSON, postFile, fileUrl } from '../../api';
import { Button } from '../ui';
import CoverageRadar from './CoverageRadar';
import {
  applyClick, clampThumb, facePose, filterLibrary, fmtBytes, libraryTags, marqueeSelect, mergeGroups,
  outliersBelow, qualityFromWidth, ranksOf, rectFromPoints, resolveHero, splitGroup,
} from './identityModel';

// Identity Workstation: a full-viewport, three-pane view of every identity the app knows.
//   left   .fsz archive library (search, tag filter, size badges, load / export)
//   centre responsive grid of faces, either the Face Manager's working faceset or the captured target people
//   right  inspector: head-angle coverage radar + quality + per-selection facts
//
// Selection is shared by the grid and the radar. Shift+Click selects a range, Ctrl/Cmd+Click toggles, dragging on empty
// space draws a marquee. Cluster edits (merge / split / rename) are POSTed to /api/target/group and /api/target/name and
// the server's reply replaces local state; this file never keeps a client-side copy of the grouping that could drift.

const HERO_KEY = 'roop_identity_heroes';
const THUMB_KEY = 'roop_identity_thumb';
const readLocal = (k, fallback) => { try { const v = localStorage.getItem(k); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } };
const writeLocal = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage blocked: preference lasts this session */ } };

const isTyping = (el) => !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable);
const TAGS = ['single', 'multi-angle', 'rich', 'large'];

export default function IdentityWorkstation({ notify, setTab, facesetState, onFacesetChange }) {
  // ── mode: 'people' (captured target clusters) | 'faceset' (the Face Manager's working set)
  const [mode, setMode] = useState('faceset');
  const [thumb, setThumb] = useState(() => clampThumb(readLocal(THUMB_KEY, 112)));
  const [threshold, setThreshold] = useState(0);
  const [busy, setBusy] = useState(false);
  const [heroes, setHeroes] = useState(() => readLocal(HERO_KEY, {}));

  // library
  const [entries, setEntries] = useState([]);
  const [query, setQuery] = useState('');
  const [tag, setTag] = useState(null);
  const [activeEntry, setActiveEntry] = useState(null);

  // target people
  const [target, setTarget] = useState(null);

  // selection (face indices of the current mode)
  const [selection, setSelection] = useState({ sel: [], anchor: null });
  const sel = selection.sel;

  const gridRef = useRef(null);
  const [marquee, setMarquee] = useState(null);

  const run = useCallback(async (fn) => {
    setBusy(true);
    try { return await fn(); } catch (e) { notify?.(e.message || String(e), 'error'); return undefined; } finally { setBusy(false); }
  }, [notify]);

  // ── library ──
  const loadLibrary = useCallback(async () => {
    try { const r = await getJSON('/api/faceset/library'); setEntries(Array.isArray(r?.entries) ? r.entries : []); }
    catch { setEntries([]); }
  }, []);
  useEffect(() => { loadLibrary(); }, [loadLibrary]);

  const visibleEntries = useMemo(() => filterLibrary(entries, query, tag), [entries, query, tag]);
  const totalBytes = useMemo(() => entries.reduce((a, e) => a + (Number(e.size) || 0), 0), [entries]);

  const openEntry = (entry) => run(async () => {
    // Pull the archive back through the Face Manager's own loader so the grid, the radar and the builder all see it.
    const blob = await (await fetch(fileUrl(entry.path))).blob();
    const res = await postFile('/api/facemgr/faceset', new File([blob], entry.filename), undefined, { timeout: 0 });
    onFacesetChange?.(res);
    setActiveEntry(entry.filename);
    setMode('faceset');
    setSelection({ sel: [], anchor: null });
    notify?.(`Opened ${entry.name}`);
  });
  const loadAsSource = (entry) => run(async () => {
    await postJSON('/api/faceset/library/load', { filename: entry.filename });
    notify?.(`${entry.name} added to Face Swap sources`);
  });
  const importFsz = (e) => {
    const f = e.target.files?.[0];
    e.target.value = '';
    if (!f) return;
    run(async () => {
      const res = await postFile('/api/faceset/library/import', f, undefined, { timeout: 0 });
      setEntries(res.entries || []);
      notify?.(`Imported ${f.name}`);
    });
  };

  // ── target people ──
  const refreshTarget = useCallback(async () => {
    try { setTarget(await postJSON('/api/target/context', {})); } catch { setTarget(null); }
  }, []);
  useEffect(() => { if (mode === 'people') refreshTarget(); }, [mode, refreshTarget]);

  const people = useMemo(() => {
    const faces = target?.target_faces || [];
    const groups = (target?.target_groups || []).map((g) => (Array.isArray(g) ? (g[0] ?? 0) : Number(g) || 0));
    const byRank = new Map();
    faces.forEach((_s, i) => {
      const r = groups[i] ?? 0;
      if (!byRank.has(r)) byRank.set(r, []);
      byRank.get(r).push(i);
    });
    return [...byRank.keys()].sort((a, b) => a - b).map((rank, n) => {
      const indices = byRank.get(rank);
      const rec = (target?.target_persons || [])[n] || {};
      const id = rec.target_person_id || `rank:${rank}`;
      return { rank, n, id, indices, name: (target?.target_names || [])[n] || rec.name || '', hero: resolveHero(heroes, id, indices) };
    });
  }, [target, heroes]);

  // ── what the centre + inspector draw, for either mode ──
  const items = useMemo(() => {
    if (mode === 'faceset') {
      const { faces = [], scores = [], meta = [] } = facesetState || {};
      return faces.map((src, i) => {
        const m = meta[i] || {};
        const pose = facePose(m);
        return { index: i, src, quality: typeof scores[i] === 'number' ? scores[i] : null, sharpness: typeof m.sharpness === 'number' ? m.sharpness : null, ...pose, px: m.face_px ?? null };
      });
    }
    const faces = target?.target_faces || [];
    const info = target?.target_faces_info || [];
    return faces.map((src, i) => {
      const pose = facePose({ pose: info[i]?.pose });
      return { index: i, src, quality: qualityFromWidth(info[i]?.width), sharpness: null, ...pose, px: info[i]?.width ?? null };
    });
  }, [mode, facesetState, target]);

  const order = useMemo(() => items.map((it) => it.index), [items]);
  const outliers = useMemo(() => (threshold > 0 ? outliersBelow(items.map((it) => it.quality), threshold) : []), [items, threshold]);
  const outlierSet = useMemo(() => new Set(outliers), [outliers]);

  // Drop selection entries that no longer exist after an edit.
  useEffect(() => {
    setSelection((s) => {
      const keep = s.sel.filter((i) => i < items.length);
      return keep.length === s.sel.length ? s : { sel: keep, anchor: keep.includes(s.anchor) ? s.anchor : null };
    });
  }, [items.length]);

  const selectedPerson = useMemo(() => {
    if (mode !== 'people' || !sel.length) return null;
    const ranks = ranksOf((target?.target_groups || []).map((g) => (Array.isArray(g) ? g[0] : g)), sel);
    return ranks.length === 1 ? people.find((p) => p.rank === ranks[0]) || null : null;
  }, [mode, sel, target, people]);

  // The radar plots the selection's person / the whole faceset; with a multi-person selection it plots the union.
  const radarItems = useMemo(() => {
    if (mode === 'faceset') return items;
    if (selectedPerson) return items.filter((it) => selectedPerson.indices.includes(it.index));
    if (sel.length) return items.filter((it) => sel.includes(it.index));
    return items;
  }, [mode, items, selectedPerson, sel]);
  const radarPoints = useMemo(
    () => radarItems.map((it) => ({ key: it.index, yaw: it.yaw, pitch: it.pitch, sharpness: it.sharpness, approx: it.approx })),
    [radarItems],
  );

  const stats = useMemo(() => {
    const q = radarItems.map((it) => it.quality).filter((v) => typeof v === 'number');
    return {
      count: radarItems.length,
      avg: q.length ? q.reduce((a, b) => a + b, 0) / q.length : null,
      min: q.length ? Math.min(...q) : null,
      max: q.length ? Math.max(...q) : null,
    };
  }, [radarItems]);

  // ── selection handlers ──
  const clickFace = (index, e) => {
    e.stopPropagation();
    setSelection((s) => applyClick(s, index, { shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey }, order));
  };
  const selectAll = () => setSelection({ sel: [...order], anchor: order[0] ?? null });
  const clearSel = () => setSelection({ sel: [], anchor: null });

  // marquee: drag on the grid background (a tile click stops propagation, so it never starts one)
  const dragRef = useRef(null);
  const onGridDown = (e) => {
    if (e.button !== 0 || e.target.closest('[data-face-index],button,input')) return;
    const host = gridRef.current;
    if (!host) return;
    const box = host.getBoundingClientRect();
    dragRef.current = { x0: e.clientX, y0: e.clientY, box, prior: e.shiftKey || e.ctrlKey || e.metaKey ? sel : [], additive: e.shiftKey || e.ctrlKey || e.metaKey, moved: false };
    host.setPointerCapture?.(e.pointerId);
  };
  const onGridMove = (e) => {
    const d = dragRef.current;
    if (!d) return;
    if (!d.moved && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < 4) return;
    d.moved = true;
    const rect = rectFromPoints({ x: d.x0, y: d.y0 }, { x: e.clientX, y: e.clientY });
    const tiles = [...gridRef.current.querySelectorAll('[data-face-index]')].map((el) => {
      const r = el.getBoundingClientRect();
      return { index: Number(el.getAttribute('data-face-index')), rect: { left: r.left, top: r.top, right: r.right, bottom: r.bottom } };
    });
    setSelection({ sel: marqueeSelect(rect, tiles, d.prior, d.additive), anchor: null });
    setMarquee({ left: rect.left - d.box.left + gridRef.current.scrollLeft, top: rect.top - d.box.top + gridRef.current.scrollTop, width: rect.right - rect.left, height: rect.bottom - rect.top });
  };
  const onGridUp = (e) => {
    const d = dragRef.current;
    dragRef.current = null;
    setMarquee(null);
    gridRef.current?.releasePointerCapture?.(e.pointerId);
    if (d && !d.moved && !d.additive) clearSel(); // a plain click on empty space deselects
  };

  // ── actions ──
  const groupsNow = () => (target?.target_groups || []).map((g) => (Array.isArray(g) ? (g[0] ?? 0) : Number(g) || 0));
  const postGroups = (groups, msg) => run(async () => {
    const res = await postJSON('/api/target/group', { groups, target_media_id: target?.target_media_id });
    setTarget(res);
    notify?.(msg);
  });
  const mergeSelected = () => {
    const ranks = ranksOf(groupsNow(), sel);
    if (ranks.length < 2) return;
    postGroups(mergeGroups(groupsNow(), ranks, ranks[0]), `Merged ${ranks.length} identities`);
    clearSel();
  };
  const splitSelected = () => {
    if (!sel.length) return;
    const { groups } = splitGroup(groupsNow(), sel);
    postGroups(groups, `Split ${sel.length} face${sel.length === 1 ? '' : 's'} into a new identity`);
    clearSel();
  };
  const renamePerson = (p, name) => {
    if (name === p.name) return;
    run(async () => {
      const res = await postJSON('/api/target/name', { person: p.rank, target_person_id: p.id.startsWith('rank:') ? undefined : p.id, name, target_media_id: target?.target_media_id });
      setTarget(res);
    });
  };
  const setHero = () => {
    if (!selectedPerson || sel.length !== 1) return;
    const next = { ...heroes, [selectedPerson.id]: sel[0] };
    setHeroes(next);
    writeLocal(HERO_KEY, next);
    notify?.('Hero thumbnail set');
  };
  const excludeIndices = (indices) => {
    if (!indices.length) return;
    const desc = [...indices].sort((a, b) => b - a); // highest first so earlier indices stay valid
    run(async () => {
      if (mode === 'faceset') {
        let res;
        for (const i of desc) res = await postJSON('/api/facemgr/remove', { index: i });
        if (res) onFacesetChange?.(res);
      } else {
        let res;
        for (const i of desc) res = await postJSON('/api/target/remove_face', { index: i, target_media_id: target?.target_media_id });
        if (res) setTarget(res);
      }
      clearSel();
      notify?.(`Excluded ${desc.length} face${desc.length === 1 ? '' : 's'}`);
    });
  };

  // keyboard: Delete excludes, Esc clears, Ctrl+A selects all. Ignored while typing (rename field, search box).
  useEffect(() => {
    const onKey = (e) => {
      if (isTyping(e.target)) return;
      if (e.key === 'Escape' && sel.length) { clearSel(); }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && sel.length) { e.preventDefault(); excludeIndices(sel); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a' && order.length) { e.preventDefault(); selectAll(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const setThumbSize = (v) => { const c = clampThumb(v); setThumb(c); writeLocal(THUMB_KEY, c); };

  const canMerge = mode === 'people' && ranksOf(groupsNow(), sel).length >= 2;
  const canSplit = mode === 'people' && sel.length > 0 && people.length > 0;
  const activeLib = entries.find((e) => e.filename === activeEntry);

  const renderTile = (it) => {
    const on = sel.includes(it.index);
    const low = outlierSet.has(it.index);
    return (
      <button key={it.index} type="button" data-face-index={it.index} data-selected={on ? 'true' : 'false'} data-outlier={low ? 'true' : 'false'}
              onClick={(e) => clickFace(it.index, e)} aria-pressed={on}
              aria-label={`Face ${it.index + 1}${it.quality !== null ? `, quality ${Math.round(it.quality * 100)}%` : ''}`}
              style={{ width: thumb, height: thumb }}
              className={`relative shrink-0 rounded-lg overflow-hidden border transition-[opacity,box-shadow] ${on ? 'ring-2 ring-[var(--accent)] border-[var(--accent)]/60' : 'border-white/10 hover:border-white/30'} ${low ? 'opacity-35 grayscale' : ''}`}>
        <img src={it.src} alt="" draggable={false} className="w-full h-full object-cover pointer-events-none" />
        {it.quality !== null && thumb >= 80 && (
          <span className="absolute top-1 left-1 px-1 rounded text-nano font-bold tabular-nums bg-black/65 text-white/90">{Math.round(it.quality * 100)}%</span>
        )}
        {mode === 'people' && people.some((p) => p.hero === it.index) && (
          <span className="absolute top-1 right-1 px-1 rounded text-nano font-bold bg-amber-500/80 text-black" title="Hero thumbnail">★</span>
        )}
        {low && thumb >= 96 && <span className="absolute bottom-1 inset-x-1 text-center text-nano font-bold uppercase text-red-300 bg-red-950/70 rounded">low</span>}
      </button>
    );
  };

  return (
    <div className="grid h-full min-h-0 min-w-0 overflow-hidden bg-[var(--bg-void)]" data-testid="identity-workstation"
         style={{ gridTemplateColumns: 'minmax(220px, 280px) minmax(0, 1fr) minmax(260px, 340px)' }}>
      {/* ───── Left: .fsz archive library ───── */}
      <aside className="flex min-h-0 flex-col border-r border-white/10 bg-[var(--bg-panel)]" aria-label="Faceset library" data-testid="ws-library">
        <div className="p-2 space-y-2 border-b border-white/10">
          <div className="flex items-center justify-between text-mini">
            <span className="font-bold uppercase tracking-wider text-nano text-white/80">.fsz library</span>
            <span className="text-muted tabular-nums" data-testid="ws-library-total">{entries.length} · {fmtBytes(totalBytes)}</span>
          </div>
          <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search identities…" aria-label="Search identities"
                 data-testid="ws-search" className="w-full px-2 py-1.5 rounded-md glass-input text-sm text-white focus:outline-none" />
          <div className="flex flex-wrap gap-1" role="group" aria-label="Tag filter">
            {TAGS.map((t) => (
              <button key={t} type="button" aria-pressed={tag === t} data-testid={`ws-tag-${t}`} onClick={() => setTag(tag === t ? null : t)}
                      className={`px-2 py-0.5 rounded-full border text-nano font-semibold ${tag === t ? 'border-[var(--accent)] bg-[var(--accent)]/20 text-white' : 'border-white/10 text-white/60 hover:text-white'}`}>
                {t}
              </button>
            ))}
          </div>
        </div>
        <ul className="m-0 p-1 list-none min-h-0 flex-1 overflow-y-auto" data-testid="ws-library-list">
          {visibleEntries.length === 0 && (
            <li className="p-4 text-center text-mini text-muted" data-testid="ws-library-empty">{entries.length ? 'No identity matches.' : 'No saved facesets yet.'}</li>
          )}
          {visibleEntries.map((e) => (
            <li key={e.filename}>
              <div data-testid="ws-entry" data-active={activeEntry === e.filename ? 'true' : 'false'}
                   className={`flex items-center gap-2 rounded-lg p-1.5 ${activeEntry === e.filename ? 'bg-[var(--accent)]/15' : 'hover:bg-white/[0.05]'}`}>
                <button type="button" onClick={() => setActiveEntry(e.filename)} onDoubleClick={() => openEntry(e)}
                        className="flex min-w-0 flex-1 items-center gap-2 text-left" aria-label={`Select ${e.name}`}>
                  {e.thumb ? <img src={e.thumb} alt="" className="h-10 w-10 rounded-md object-cover shrink-0" /> : <span className="h-10 w-10 rounded-md bg-white/10 shrink-0" />}
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-semibold text-white/90">{e.name}</span>
                    <span className="flex flex-wrap items-center gap-1 text-nano text-muted">
                      <span>{e.faces} face{e.faces === 1 ? '' : 's'}</span>
                      <span className="px-1 rounded bg-white/10 text-white/70 tabular-nums" data-testid="ws-size">{fmtBytes(e.size)}</span>
                      {libraryTags(e).map((t) => <span key={t} className="text-white/70">#{t}</span>)}
                    </span>
                  </span>
                </button>
              </div>
              {activeEntry === e.filename && (
                <div className="flex flex-wrap gap-1 px-1.5 pb-1.5">
                  <Button size="sm" variant="primary" onClick={() => openEntry(e)} disabled={busy} data-testid="ws-open">Open</Button>
                  <Button size="sm" variant="secondary" onClick={() => loadAsSource(e)} disabled={busy}>Use as source</Button>
                  <a href={fileUrl(e.path)} download={e.filename} data-testid="ws-export"
                     className="inline-flex items-center rounded-md border border-white/15 px-2 text-mini text-white/80 hover:bg-white/10">Export</a>
                </div>
              )}
            </li>
          ))}
        </ul>
        <div className="flex gap-2 border-t border-white/10 p-2">
          <label className="flex-1 cursor-pointer text-center rounded-md border border-white/15 px-2 py-1.5 text-mini text-white/80 hover:bg-white/10 focus-within:outline focus-within:outline-2 focus-within:outline-[var(--accent)]">
            Import .fsz
            <input type="file" accept=".fsz" onChange={importFsz} className="sr-only" data-testid="ws-import" />
          </label>
          <Button size="sm" variant="secondary" onClick={loadLibrary} disabled={busy}>Refresh</Button>
        </div>
      </aside>

      {/* ───── Centre: grid ───── */}
      <section className="flex min-h-0 min-w-0 flex-col" aria-label="Identity grid" data-testid="ws-stage">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-white/10 bg-[var(--bg-panel)] px-3 py-2">
          <div role="tablist" aria-label="Grid source" className="flex gap-1 rounded-lg border border-white/10 p-0.5">
            {[['faceset', 'Faceset'], ['people', 'Target people']].map(([id, label]) => (
              <button key={id} type="button" role="tab" aria-selected={mode === id} data-testid={`ws-mode-${id}`}
                      onClick={() => { setMode(id); clearSel(); }}
                      className={`rounded-md px-2.5 py-1 text-mini font-semibold ${mode === id ? 'bg-[var(--accent)]/20 text-white' : 'text-white/60 hover:text-white'}`}>{label}</button>
            ))}
          </div>

          <label className="flex items-center gap-2 text-mini text-white/70">
            <span>Size</span>
            <input type="range" min={64} max={240} step={8} value={thumb} onChange={(e) => setThumbSize(e.target.value)}
                   aria-label="Thumbnail size" data-testid="ws-thumb-slider" className="w-28" />
            <span className="tabular-nums w-10" data-testid="ws-thumb-value">{thumb}px</span>
          </label>

          <label className="flex items-center gap-2 text-mini text-white/70" data-testid="ws-quality-bar">
            <span>Quality ≥</span>
            <input type="range" min={0} max={1} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))}
                   aria-label="Quality threshold" data-testid="ws-threshold" className="w-28" />
            <span className="tabular-nums w-9">{Math.round(threshold * 100)}%</span>
          </label>
          {threshold > 0 && (
            <>
              <span className="text-mini text-amber-300" data-testid="ws-outlier-count">{outliers.length} below</span>
              <Button size="sm" variant="secondary" disabled={!outliers.length} onClick={() => setSelection({ sel: outliers, anchor: null })} data-testid="ws-select-outliers">Select outliers</Button>
              <Button size="sm" variant="stop" disabled={!outliers.length || busy} onClick={() => excludeIndices(outliers)} data-testid="ws-exclude-outliers">Exclude outliers</Button>
            </>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 border-b border-white/10 px-3 py-1.5 text-mini" data-testid="ws-actions">
          <span className="text-muted tabular-nums" data-testid="ws-selected-count">{sel.length} selected</span>
          <Button size="sm" variant="secondary" onClick={selectAll} disabled={!order.length}>Select all</Button>
          <Button size="sm" variant="secondary" onClick={clearSel} disabled={!sel.length}>Clear</Button>
          <span className="text-white/20" aria-hidden="true">|</span>
          <Button size="sm" variant="secondary" onClick={mergeSelected} disabled={!canMerge || busy} data-testid="ws-merge"
                  title={mode === 'people' ? 'Merge the identities the selected faces belong to' : 'Switch to Target people to merge identities'}>Merge identities</Button>
          <Button size="sm" variant="secondary" onClick={splitSelected} disabled={!canSplit || busy} data-testid="ws-split"
                  title={mode === 'people' ? 'Move the selected faces into a new identity' : 'Switch to Target people to split identities'}>Split to new identity</Button>
          <Button size="sm" variant="secondary" onClick={setHero} disabled={!(selectedPerson && sel.length === 1)} data-testid="ws-hero">★ Set hero</Button>
          <Button size="sm" variant="stop" onClick={() => excludeIndices(sel)} disabled={!sel.length || busy} data-testid="ws-exclude">Exclude</Button>
          <span className="ml-auto text-nano text-muted hidden xl:inline">Shift = range · Ctrl = toggle · drag = marquee · Del = exclude</span>
        </div>

        <div ref={gridRef} className="relative min-h-0 flex-1 overflow-auto p-3 select-none" data-testid="ws-grid"
             onPointerDown={onGridDown} onPointerMove={onGridMove} onPointerUp={onGridUp} onPointerCancel={onGridUp}>
          {mode === 'faceset' && (items.length === 0 ? (
            <Empty testId="ws-empty-faceset" title="No faces in the working faceset"
                   body="Open a saved .fsz from the library, or add images in the Faceset builder." />
          ) : (
            <div className="flex flex-wrap gap-2" data-testid="ws-faces">{items.map(renderTile)}</div>
          ))}

          {mode === 'people' && (people.length === 0 ? (
            <Empty testId="ws-empty-people" title="No target people captured"
                   body="Load a target video in Face Swap and capture its faces; the clusters appear here."
                   action={setTab ? <Button size="sm" variant="secondary" onClick={() => setTab('faceswap')}>Open Face Swap</Button> : null} />
          ) : (
            <div className="space-y-4">
              {people.map((p) => (
                <div key={p.id} data-testid="ws-person" data-rank={p.rank}>
                  <div className="mb-1.5 flex items-center gap-2">
                    <PersonName person={p} onCommit={(name) => renamePerson(p, name)} />
                    <span className="text-mini text-muted tabular-nums">{p.indices.length} face{p.indices.length === 1 ? '' : 's'}</span>
                    <button type="button" className="text-mini text-accent hover:underline" onClick={() => setSelection({ sel: p.indices, anchor: p.indices[0] })}>select all</button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {p.indices.map((i) => renderTile(items[i]))}
                  </div>
                </div>
              ))}
            </div>
          ))}
          {marquee && <div aria-hidden="true" data-testid="ws-marquee" className="pointer-events-none absolute border border-[var(--accent)] bg-[var(--accent)]/15"
                           style={{ left: marquee.left, top: marquee.top, width: marquee.width, height: marquee.height }} />}
        </div>
      </section>

      {/* ───── Right: inspector ───── */}
      <aside className="flex min-h-0 flex-col overflow-y-auto border-l border-white/10 bg-[var(--bg-panel)] p-3 space-y-4" aria-label="Identity inspector" data-testid="ws-inspector">
        <div>
          <div className="m-0 text-sm font-bold text-white/90 truncate" data-testid="ws-inspector-title">
            {mode === 'people' ? (selectedPerson ? (selectedPerson.name || `Person ${selectedPerson.n + 1}`) : sel.length ? `${sel.length} faces selected` : 'All target people')
              : (activeLib ? activeLib.name : 'Working faceset')}
          </div>
          <p className="m-0 mt-0.5 text-nano text-muted">
            {mode === 'people' ? 'Captured target cluster' : activeLib ? `${activeLib.filename} · ${fmtBytes(activeLib.size)}` : 'Faces being built in the Face Manager'}
          </p>
        </div>

        <CoverageRadar points={radarPoints} selected={sel}
                       onSelect={(i, e) => setSelection((s) => applyClick(s, i, { shift: e?.shiftKey, ctrl: e?.ctrlKey || e?.metaKey }, order))} />

        <dl className="m-0 grid grid-cols-2 gap-x-3 gap-y-1.5 text-mini" data-testid="ws-stats">
          <dt className="text-muted">Faces</dt><dd className="m-0 text-right tabular-nums" data-testid="ws-stat-count">{stats.count}</dd>
          <dt className="text-muted">Avg quality</dt><dd className="m-0 text-right tabular-nums">{stats.avg === null ? '—' : `${Math.round(stats.avg * 100)}%`}</dd>
          <dt className="text-muted">Best / worst</dt><dd className="m-0 text-right tabular-nums">{stats.max === null ? '—' : `${Math.round(stats.max * 100)}% / ${Math.round(stats.min * 100)}%`}</dd>
          {mode === 'people' && selectedPerson && (<><dt className="text-muted">Hero</dt><dd className="m-0 text-right tabular-nums">face {(selectedPerson.hero ?? 0) + 1}</dd></>)}
        </dl>

        {sel.length === 1 && items[sel[0]] && (
          <div className="rounded-lg border border-white/10 bg-black/20 p-2 text-mini space-y-1" data-testid="ws-face-detail">
            <img src={items[sel[0]].src} alt="" className="w-full max-h-40 object-contain rounded" />
            <div className="flex justify-between"><span className="text-muted">Yaw / pitch</span>
              <span className="tabular-nums">{Math.round(items[sel[0]].yaw)}° / {Math.round(items[sel[0]].pitch)}°{items[sel[0]].approx ? ' ≈' : ''}</span></div>
            <div className="flex justify-between"><span className="text-muted">Quality</span>
              <span className="tabular-nums">{items[sel[0]].quality === null ? '—' : `${Math.round(items[sel[0]].quality * 100)}%`}</span></div>
            {items[sel[0]].px != null && <div className="flex justify-between"><span className="text-muted">Size</span><span className="tabular-nums">{items[sel[0]].px}px</span></div>}
          </div>
        )}
      </aside>
    </div>
  );
}

function Empty({ title, body, action, testId }) {
  return (
    <div className="grid h-full min-h-[200px] place-items-center text-center" data-testid={testId}>
      <div className="max-w-sm space-y-2">
        <div className="text-sm font-semibold text-white/80">{title}</div>
        <p className="m-0 text-mini text-muted">{body}</p>
        {action}
      </div>
    </div>
  );
}

// Inline rename: commits on Enter or blur, Escape reverts. Keyed on the committed name so a server reply resets it.
function PersonName({ person, onCommit }) {
  const [v, setV] = useState(person.name);
  useEffect(() => { setV(person.name); }, [person.name]);
  return (
    <input type="text" value={v} onChange={(e) => setV(e.target.value)} placeholder={`Person ${person.n + 1}`} aria-label={`Name for person ${person.n + 1}`}
           data-testid="ws-person-name" onPointerDown={(e) => e.stopPropagation()}
           onBlur={() => onCommit(v.trim())}
           onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); else if (e.key === 'Escape') { setV(person.name); e.currentTarget.blur(); } }}
           className="w-44 rounded-md bg-transparent px-1.5 py-0.5 text-sm font-bold text-white/90 border border-transparent hover:border-white/20 focus:border-[var(--accent)] focus:outline-none" />
  );
}
