import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  UserPlus, Trash2, Library, FolderInput, Film, Image as ImageIcon, X, ScanFace, Users, Eraser,
  ArrowRight, Cpu, Wand2, Layers, Crosshair, Clapperboard, Check, Link2,
} from 'lucide-react';
import { getJSON, postJSON } from '@engine/api';
import { swapperOptions } from '@engine/components/faceswap/pipelineRules.js';
import { useStudio } from '../engine/studio';
import { useRun } from '../engine/run';
import { sourceNameAt } from '../engine/payload';
import { Panel, PipelineBeams } from '../fx/effects';
import { Select, Slider } from '../ui/controls';
import { peopleOf, hueOf, mappedSource, frameUrl, isSelectedPerson } from './people';
import { notify } from '../ui/controls';

// ── Cast: the source faces ───────────────────────────────────────────────────
export function CastPanel({ onOpenLibrary }) {
  const sources = useStudio((s) => s.sources);
  const info = useStudio((s) => s.sourcesInfo);
  const selected = useStudio((s) => s.selectedSource);
  const busy = useStudio((s) => s.busy === 'sources');
  const input = useRef(null);
  const [over, setOver] = useState(false);
  const add = (files) => useStudio.getState().addSources(Array.from(files || []).filter((f) => /^image\//.test(f.type) || /\.(fsz|jpe?g|png|webp|bmp)$/i.test(f.name)));
  return (
    <Panel title="Cast" accent={`${sources.length} source${sources.length === 1 ? '' : 's'}`} data-testid="cast"
      actions={<>
        <button type="button" className="btn btn-ghost btn-sm btn-icon" title="Load from the faceset library" aria-label="Faceset library" onClick={onOpenLibrary}><Library size={15} /></button>
        {sources.length > 0 && <button type="button" className="btn btn-ghost btn-sm btn-icon" title="Clear the cast" aria-label="Clear the cast" onClick={() => useStudio.getState().clearSources()}><Eraser size={15} /></button>}
      </>}>
      <div className={`grid grid-cols-3 gap-2 ${over ? 'rounded-xl ring-2 ring-[var(--accent-2)]' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}>
        {sources.map((src, i) => (
          <div key={info[i]?.id || i} className={`tile group aspect-square ${i === selected ? 'selected' : ''}`}>
            <button type="button" className="absolute inset-0" onClick={() => useStudio.getState().selectSource(i)}
              aria-pressed={i === selected} aria-label={`Use ${sourceNameAt({ sources, sourcesInfo: info }, i)} as the source`}>
              <img src={src} alt="" className="h-full w-full object-cover" />
            </button>
            <span className="pointer-events-none absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-black/90 to-transparent px-1.5 pb-1 pt-4 text-[10.5px] font-medium text-white">
              {info[i]?.name || `Face ${i + 1}`}
            </span>
            {i === selected && <span className="pointer-events-none absolute left-1 top-1 grid h-4 w-4 place-items-center rounded-full bg-[var(--accent)] text-white"><Check size={11} /></span>}
            <button type="button" className="absolute right-1 top-1 grid h-5 w-5 place-items-center rounded-md bg-black/70 text-white opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100"
              aria-label={`Remove ${info[i]?.name || `face ${i + 1}`}`} onClick={() => useStudio.getState().removeSource(i)}><X size={12} /></button>
          </div>
        ))}
        <button type="button" className="dropzone aspect-square text-[11px]" onClick={() => input.current?.click()} disabled={busy} data-testid="cast-add">
          <UserPlus size={18} className="text-[var(--accent-2)]" />
          {busy ? 'Adding…' : 'Add face'}
        </button>
      </div>
      <input ref={input} type="file" multiple accept="image/*,.fsz" className="hidden" onChange={(e) => { add(e.target.files); e.target.value = ''; }} />
    </Panel>
  );
}

// ── Media bin: the targets ───────────────────────────────────────────────────
export function TargetsPanel() {
  const targets = useStudio((s) => s.targets);
  const selected = useStudio((s) => s.selectedTarget);
  const busy = useStudio((s) => s.busy === 'targets');
  const input = useRef(null);
  const [path, setPath] = useState('');
  return (
    <Panel title="Media" accent={`${targets.length}`} data-testid="targets"
      actions={<button type="button" className="btn btn-ghost btn-sm" onClick={() => input.current?.click()} disabled={busy}><FolderInput size={14} /> {busy ? 'Adding…' : 'Add'}</button>}>
      <div className="max-h-[260px] space-y-1.5 overflow-y-auto pr-1">
        {targets.map((t, i) => {
          const video = (t.frames || 0) > 1;
          return (
            <div key={t.media_id || t.id || i} className={`tile group flex items-center gap-2.5 p-1.5 ${i === selected ? 'selected' : ''}`}>
              <button type="button" className="flex min-w-0 flex-1 items-center gap-2.5 text-left" onClick={() => useStudio.getState().selectTarget(i)} aria-pressed={i === selected}>
                <span className="relative h-11 w-[72px] shrink-0 overflow-hidden rounded-lg bg-black">
                  <img src={frameUrl(i, Math.max(1, t.start_frame || 1), 160)} alt="" loading="lazy" className="h-full w-full object-cover" />
                  <span className="absolute bottom-0.5 right-0.5 rounded bg-black/75 px-1 font-mono text-[9.5px] text-white">{video ? `${t.frames}f` : 'img'}</span>
                </span>
                <span className="min-w-0">
                  <span className="block truncate text-[12.5px] font-semibold text-white" title={t.name}>{t.name}</span>
                  <span className="flex items-center gap-1 font-mono text-[10.5px] text-[var(--fg-faint)]">
                    {video ? <Film size={11} /> : <ImageIcon size={11} />}
                    {video ? `${t.fps ? Number(t.fps).toFixed(t.fps % 1 ? 2 : 0) : '?'} fps` : 'still'}
                    {t.color_badge && <> · {t.color_badge}</>}
                  </span>
                </span>
              </button>
              <button type="button" className="btn btn-ghost btn-sm btn-icon opacity-0 group-hover:opacity-100 focus:opacity-100" aria-label={`Remove ${t.name}`} onClick={() => useStudio.getState().removeTarget(i)}><Trash2 size={14} /></button>
            </div>
          );
        })}
        {!targets.length && <div className="py-4 text-center text-[12px] text-[var(--fg-faint)]">Drop media on the viewport, or add it here.</div>}
      </div>
      <form className="mt-2 flex gap-1.5" onSubmit={(e) => { e.preventDefault(); useStudio.getState().addTargetPaths(path.split(/\r?\n|;/)); setPath(''); }}>
        <label className="sr-only" htmlFor="target-path">Add by path</label>
        <input id="target-path" className="input !h-8 text-[12px]" placeholder="or paste a file path — no copy, no wait" value={path} onChange={(e) => setPath(e.target.value)} />
        <button type="submit" className="btn btn-sm btn-icon" aria-label="Add path" disabled={!path.trim()}><Link2 size={14} /></button>
      </form>
      <input ref={input} type="file" multiple accept="video/*,image/*" className="hidden" onChange={(e) => { useStudio.getState().addTargets(Array.from(e.target.files || [])); e.target.value = ''; }} />
    </Panel>
  );
}

// ── People: who is swapped, and into whom ────────────────────────────────────
export function PeoplePanel() {
  const s = useStudio();
  const people = useMemo(() => peopleOf(s), [s.targetPersonIds, s.targetGroups, s.targetNames, s.targetFaces]); // eslint-disable-line react-hooks/exhaustive-deps
  const mode = s.settings?.face_detection_mode;
  const modes = s.meta?.face_detection_modes || ['Selected face', 'All faces', 'First found'];
  return (
    <Panel title="People" accent={people.length ? `${people.length}` : undefined} data-testid="people"
      actions={<>
        <button type="button" className="btn btn-ghost btn-sm" title="Make every face on this frame a target person" onClick={() => s.captureFace({ all: true })}><Users size={14} /> All</button>
        {people.length > 0 && <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Forget the target people" title="Forget the target people" onClick={() => s.clearPeople()}><Eraser size={14} /></button>}
      </>}>
      <Select label="Who gets swapped" value={mode} options={[...modes, ...(modes.includes('Selected people') ? [] : ['Selected people'])]} onChange={(v) => s.setSetting('face_detection_mode', v)} />
      <div className="mt-3 space-y-2">
        {people.map((p) => {
          const src = mappedSource(s, p.id);
          const sel = isSelectedPerson(s, p);
          const value = src == null ? 'auto' : String(src);
          return (
            <div key={p.id} className={`tile flex items-center gap-2 p-2 ${sel ? 'selected' : ''}`} style={sel ? { borderColor: hueOf(p.rank) } : undefined}>
              <button type="button" onClick={() => s.selectPerson(p.id, p.faceIndex)} aria-pressed={sel} className="flex min-w-0 items-center gap-2 text-left">
                <span className="relative h-10 w-10 shrink-0 overflow-hidden rounded-full border-2" style={{ borderColor: hueOf(p.rank) }}>
                  {p.thumb ? <img src={p.thumb} alt="" className="h-full w-full object-cover" /> : <ScanFace size={18} className="m-auto mt-2" />}
                </span>
                <span className="min-w-0">
                  <span className="block truncate text-[12.5px] font-semibold text-white">{p.name}</span>
                  <span className="block font-mono text-[10px]" style={{ color: hueOf(p.rank) }}>{sel ? 'selected' : 'click to select'}</span>
                </span>
              </button>
              <ArrowRight size={14} className="shrink-0 text-[var(--fg-faint)]" />
              <label className="sr-only" htmlFor={`map-${p.id}`}>{p.name} becomes</label>
              <select id={`map-${p.id}`} className="select !h-8 min-w-0 flex-1 text-[12px]" value={value}
                onChange={(e) => s.mapPerson(p.id, e.target.value === 'auto' ? undefined : Number(e.target.value))}>
                <option value="auto">{mode === 'Selected face' ? (sel ? `Selected source (${sourceNameAt(s, s.selectedSource) || '—'})` : 'Skip (not selected)') : 'By order'}</option>
                {s.sources.map((_, i) => <option key={i} value={i}>{sourceNameAt(s, i)}</option>)}
                <option value="-1">Leave untouched</option>
              </select>
            </div>
          );
        })}
        {!people.length && (
          <div className="rounded-xl border border-dashed border-[var(--rule-strong)] p-3 text-[12px] text-[var(--fg-dim)]">
            <Crosshair size={14} className="mr-1 inline text-[var(--accent-2)]" />
            Click a face on the viewport to make it a target person{mode && mode !== 'Selected face' && mode !== 'Selected people' ? ` — or keep “${mode}”, which needs no picking` : ''}.
          </div>
        )}
      </div>
    </Panel>
  );
}

// ── Pipeline: the models, as a flowing diagram + selectors ───────────────────
export function PipelinePanel() {
  const meta = useStudio((s) => s.meta);
  const settings = useStudio((s) => s.settings) || {};
  const set = useStudio((s) => s.setSetting);
  const live = useRun((r) => r.progress.processing && !r.progress.paused);
  const nodes = [
    { id: 'detect', label: 'Detect', icon: Crosshair, value: settings.detector_engine },
    { id: 'swap', label: 'Swap', icon: Wand2, value: settings.swap_model },
    { id: 'mask', label: 'Mask', icon: Layers, value: settings.mask_engine },
    { id: 'restore', label: 'Restore', icon: Cpu, value: settings.selected_enhancer },
    { id: 'out', label: 'Output', icon: Clapperboard, value: settings.video_codec || settings.output_method || 'file' },
  ];
  return (
    <Panel title="Pipeline" accent={live ? 'live' : undefined} beam={live} data-testid="pipeline">
      <PipelineBeams nodes={nodes} live={live} className="mb-3 px-1" />
      <div className="grid grid-cols-2 gap-2.5">
        <Select label="Swapper" value={settings.swap_model} options={swapperOptions(meta?.swap_models, settings.swap_model)} onChange={(v) => set('swap_model', v)} />
        <Select label="Restorer" value={settings.selected_enhancer} options={meta?.enhancers || []} onChange={(v) => set('selected_enhancer', v)} />
        <Select label="Mask" value={settings.mask_engine} options={meta?.mask_engines || []} onChange={(v) => set('mask_engine', v)} />
        <Select label="Detector" value={settings.detector_engine} options={meta?.detector_engines || []} onChange={(v) => set('detector_engine', v)} />
        <Select label="Swap resolution" value={settings.subsample_upscale} options={meta?.upscale || []} onChange={(v) => set('subsample_upscale', v)} />
        <Select label="Colour match" value={settings.color_transfer_mode} options={meta?.color_transfer_modes || []} onChange={(v) => set('color_transfer_mode', v)} />
      </div>
    </Panel>
  );
}

// ── Look: the sliders people actually reach for ──────────────────────────────
export function LookPanel() {
  const settings = useStudio((s) => s.settings) || {};
  const set = useStudio((s) => s.setSetting);
  const pct = (v) => `${Math.round(v * 100)}%`;
  return (
    <Panel title="Look" data-testid="look">
      <div className="space-y-2.5">
        <Slider label="Blend" value={settings.blend_ratio ?? 0.8} min={0} max={1} step={0.01} format={pct} onChange={(v) => set('blend_ratio', v)} />
        <Slider label="Identity distance" hint="How unlike the source a face may be and still swap" value={settings.max_face_distance ?? 0.75} min={0.2} max={1.5} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set('max_face_distance', v)} />
        <Slider label="Mask feather" value={settings.face_mask_blend ?? 25} min={0} max={100} step={1} onChange={(v) => set('face_mask_blend', v)} />
        <Slider label="Sharpen" value={settings.merger_sharpen ?? 0} min={0} max={1} step={0.01} format={pct} onChange={(v) => set('merger_sharpen', v)} />
        <Slider label="Detection threshold" value={settings.face_detector_threshold ?? 0.5} min={0.1} max={0.95} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set('face_detector_threshold', v)} />
      </div>
    </Panel>
  );
}

// ── Library drawer content (also the Library screen's data) ──────────────────
export function useFacesetLibrary() {
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState(null);
  const load = async () => {
    try { const r = await getJSON('/api/faceset/library'); setEntries(r.entries || []); setError(null); }
    catch (e) { setError(e.message); setEntries([]); }
  };
  useEffect(() => { load(); }, []);
  const act = async (path, body, msg) => {
    try {
      const r = await postJSON(path, body, { timeout: 0 });
      if (r?.entries) setEntries(r.entries); else await load();
      if (msg) notify(typeof msg === 'function' ? msg(r) : msg);
      return r;
    } catch (e) { notify(e.message, 'error'); return null; }
  };
  return {
    entries, error, reload: load,
    loadInto: async (e) => { const r = await act('/api/faceset/library/load', { filename: e.filename }, `Loaded “${e.name}” into the cast`); await useStudio.getState().refresh(); return r; },
    save: (name) => act('/api/faceset/library/save', { name }, (r) => `Saved “${r?.saved || name}” to the library`),
    remove: (e) => act('/api/faceset/library/delete', { filename: e.filename }, `Deleted “${e.name}”`),
    rename: (e, name) => act('/api/faceset/library/rename', { filename: e.filename, name }, `Renamed to “${name}”`),
  };
}
