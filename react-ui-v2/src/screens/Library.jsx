import React, { useRef, useState } from 'react';
import { Download, Trash2, Pencil, Upload, Save, Search, Library as LibraryIcon, Users, Check, X } from 'lucide-react';
import { postFile } from '@engine/api';
import { useFacesetLibrary } from '../studio/Panels';
import { useStudio } from '../engine/studio';
import { sourceNameAt } from '../engine/payload';
import { Panel, Meteors, SplitText } from '../fx/effects';
import { notify } from '../ui/controls';

const kb = (n) => (n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round((n || 0) / 1024))} KB`);
const ago = (t) => {
  if (!t) return '';
  const s = Date.now() / 1000 - t;
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
};

export default function Library() {
  const lib = useFacesetLibrary();
  const studio = useStudio();
  const [q, setQ] = useState('');
  const [renaming, setRenaming] = useState(null);
  const [newName, setNewName] = useState('');
  const [saveName, setSaveName] = useState('');
  const importInput = useRef(null);
  const entries = (lib.entries || []).filter((e) => !q || String(e.name).toLowerCase().includes(q.toLowerCase()));
  const selName = sourceNameAt(studio, studio.selectedSource);

  return (
    <div className="mx-auto max-w-[1500px] space-y-3" data-testid="library">
      <div className="flex flex-wrap items-end gap-3">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight"><SplitText text="Faceset library" className="gradient-text" /></h1>
          <p className="text-[13px] text-[var(--fg-dim)]">Saved identities. Load one into the cast in a click — embeddings and geometry are cached, so it is ready to swap at once.</p>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <label className="relative">
            <span className="sr-only">Search the library</span>
            <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--fg-faint)]" />
            <input className="input !h-9 w-56 pl-8" placeholder="Search facesets" value={q} onChange={(e) => setQ(e.target.value)} />
          </label>
          <button type="button" className="btn" onClick={() => importInput.current?.click()}><Upload size={15} /> Import .fsz</button>
          <input ref={importInput} type="file" accept=".fsz" className="hidden" onChange={async (e) => {
            const f = e.target.files?.[0]; e.target.value = '';
            if (!f) return;
            try { await postFile('/api/faceset/library/import', f, undefined, { timeout: 0 }); await lib.reload(); notify(`Imported “${f.name}”`); } catch (err) { notify(err.message, 'error'); }
          }} />
        </div>
      </div>

      <Panel title="Save from the cast" bodyClass="p-3">
        <form className="flex flex-wrap items-end gap-2" onSubmit={async (e) => { e.preventDefault(); if (await lib.save(saveName.trim() || selName || 'faceset')) setSaveName(''); }}>
          <div className="min-w-[200px] flex-1">
            <label className="field-label" htmlFor="save-name">Name</label>
            <input id="save-name" className="input" placeholder={selName || 'Add a face to the cast first'} value={saveName} onChange={(e) => setSaveName(e.target.value)} />
          </div>
          <button type="submit" className="btn btn-primary" disabled={!studio.sources.length}><Save size={15} /> Save “{selName || '—'}”</button>
        </form>
      </Panel>

      {lib.entries === null ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 2xl:grid-cols-6">
          {Array.from({ length: 6 }, (_, i) => <div key={i} className="tile aspect-[4/5] animate-pulse" />)}
        </div>
      ) : entries.length === 0 ? (
        <div className="panel relative flex min-h-[280px] flex-col items-center justify-center gap-2 overflow-hidden text-center">
          <Meteors count={12} />
          <LibraryIcon size={30} strokeWidth={1.5} className="relative text-[var(--accent-2)]" />
          <div className="relative text-[15px] font-semibold">{q ? 'Nothing matches' : 'No saved facesets yet'}</div>
          <div className="relative text-[12.5px] text-[var(--fg-dim)]">{lib.error || 'Save a cast face above, or import a .fsz file.'}</div>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 2xl:grid-cols-6" data-testid="library-grid">
          {entries.map((e) => (
            <article key={e.filename} className="tile group flex flex-col">
              <div className="relative aspect-square overflow-hidden bg-black">
                {e.thumb ? <img src={e.thumb} alt="" className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105" /> : <Users size={32} className="m-auto mt-[40%] text-[var(--fg-faint)]" />}
                <span className="absolute left-2 top-2 badge badge-dim bg-black/60">{e.faces ?? '?'} face{e.faces === 1 ? '' : 's'}</span>
                <div className="absolute inset-x-2 bottom-2 flex translate-y-2 gap-1.5 opacity-0 transition-all duration-300 group-hover:translate-y-0 group-hover:opacity-100 focus-within:translate-y-0 focus-within:opacity-100">
                  <button type="button" className="btn btn-primary btn-sm flex-1" onClick={() => lib.loadInto(e)} data-testid="library-load"><Download size={13} /> Load</button>
                  <button type="button" className="btn btn-sm btn-icon bg-black/70" aria-label={`Rename ${e.name}`} onClick={() => { setRenaming(e.filename); setNewName(e.name); }}><Pencil size={13} /></button>
                  <button type="button" className="btn btn-danger btn-sm btn-icon" aria-label={`Delete ${e.name}`}
                    onClick={() => { if (window.confirm(`Delete “${e.name}” from the library? This removes the file on disk.`)) lib.remove(e); }}><Trash2 size={13} /></button>
                </div>
              </div>
              <div className="p-2.5">
                {renaming === e.filename ? (
                  <form className="flex gap-1" onSubmit={async (ev) => { ev.preventDefault(); await lib.rename(e, newName.trim()); setRenaming(null); }}>
                    <input className="input !h-7 text-[12px]" value={newName} autoFocus onChange={(ev) => setNewName(ev.target.value)} aria-label="New name" />
                    <button type="submit" className="btn btn-sm btn-icon !h-7" aria-label="Save name"><Check size={13} /></button>
                    <button type="button" className="btn btn-ghost btn-sm btn-icon !h-7" aria-label="Cancel" onClick={() => setRenaming(null)}><X size={13} /></button>
                  </form>
                ) : (
                  <div className="truncate text-[13px] font-semibold text-white" title={e.name}>{e.name}</div>
                )}
                <div className="mt-0.5 font-mono text-[10.5px] text-[var(--fg-faint)]">{kb(e.size)} · {ago(e.mtime)}</div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
