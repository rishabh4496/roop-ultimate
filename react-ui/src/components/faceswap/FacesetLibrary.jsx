import React, { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON, postFile, fileUrl } from '../../api';
import { Button } from '../ui';
import { confirmDialog, promptDialog } from '../confirm';
import { Icon } from '../../icons';

// Persistent, named .fsz facesets on disk. Save the selected source faceset here
// once and reload it any time without re-uploading. Point the library folder
// (Settings → "Faceset library folder") at OneDrive/Dropbox/Google Drive to sync
// facesets across devices.
export default function FacesetLibrary({ canSave, onLoaded, notify }) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState([]);
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState(null); // filename being renamed
  const [renameVal, setRenameVal] = useState('');
  const [filter, setFilter] = useState('');
  const [pickerOpen, setPickerOpen] = useState(false);
  const [selected, setSelected] = useState(null); // last-loaded entry (for the closed box)
  const importRef = useRef(null);
  const pickerRef = useRef(null);

  const q = filter.trim().toLowerCase();
  const shown = q ? entries.filter((e) => e.name.toLowerCase().includes(q)) : entries;

  const refresh = useCallback(async () => {
    try {
      const r = await getJSON('/api/faceset/library');
      setEntries(r.entries || []);
    } catch (e) { notify?.(e.message, 'error'); }
  }, [notify]);

  useEffect(() => { if (open) refresh(); }, [open, refresh]);

  // Close the dropdown when clicking outside of it.
  useEffect(() => {
    if (!pickerOpen) return undefined;
    const onDown = (ev) => { if (pickerRef.current && !pickerRef.current.contains(ev.target)) setPickerOpen(false); };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [pickerOpen]);

  const saveCurrent = async () => {
    const name = await promptDialog({ title: 'Save faceset', message: 'Name this faceset', placeholder: 'e.g. Scarlett — hero angles', confirmLabel: 'Save' });
    if (name == null) return;
    setBusy(true);
    try {
      const r = await postJSON('/api/faceset/library/save', { name });
      setEntries(r.entries || []);
      notify?.(`Saved “${r.saved}” to library`);
    } catch (e) { notify?.(e.message, 'error'); } finally { setBusy(false); }
  };

  const load = async (entry) => {
    if (busy) return; // rows stay clickable while busy — don't double-load
    setBusy(true);
    setPickerOpen(false);
    try {
      const r = await postJSON('/api/faceset/library/load', { filename: entry.filename });
      setSelected(entry);
      onLoaded?.(r);
      notify?.(`Loaded “${entry.name}” into source faces`);
    } catch (e) { notify?.(e.message, 'error'); } finally { setBusy(false); }
  };

  const del = async (entry) => {
    if (!(await confirmDialog({ title: 'Delete faceset?', message: `Delete “${entry.name}” from the library? This removes the file on disk.`, confirmLabel: 'Delete', danger: true }))) return;
    try {
      const r = await postJSON('/api/faceset/library/delete', { filename: entry.filename });
      setEntries(r.entries || []);
      if (selected?.filename === entry.filename) setSelected(null);
    } catch (e) { notify?.(e.message, 'error'); }
  };

  // Rename lifecycle guard: closing the input (Escape, or the unmount-blur that
  // follows Enter/Escape) must not re-commit or commit a cancelled rename.
  const renameDone = useRef(false);
  const beginRename = (entry) => { renameDone.current = false; setRenaming(entry.filename); setRenameVal(entry.name); };
  const cancelRename = () => { renameDone.current = true; setRenaming(null); };
  const commitRename = async (entry) => {
    if (renameDone.current) return;
    renameDone.current = true;
    const newName = renameVal.trim();
    if (!newName || newName === entry.name) { setRenaming(null); return; } // no-op
    try {
      const r = await postJSON('/api/faceset/library/rename', { filename: entry.filename, name: newName });
      setEntries(r.entries || []);
      if (selected?.filename === entry.filename) setSelected(null);
    } catch (e) { notify?.(e.message, 'error'); } finally { setRenaming(null); }
  };

  const onImport = async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = '';
    setBusy(true);
    try {
      let r;
      for (const f of files) r = await postFile('/api/faceset/library/import', f);
      if (r) setEntries(r.entries || []);
      if (files.length) notify?.(`Imported ${files.length} faceset(s)`);
    } catch (err) { notify?.(err.message, 'error'); } finally { setBusy(false); }
  };

  const openFolder = async () => {
    try { await postJSON('/api/faceset/library/open', {}); } catch (e) { notify?.(e.message, 'error'); }
  };

  const rebuildThumbs = async () => {
    setBusy(true);
    try {
      const r = await postJSON('/api/faceset/library/rebuild_thumbs', {});
      setEntries(r.entries || []);
      notify?.(`Rebuilt ${r.rebuilt} thumbnail(s) — picked the most frontal face`);
    } catch (e) { notify?.(e.message, 'error'); } finally { setBusy(false); }
  };

  const Thumb = ({ e, size }) => (
    <span className={`shrink-0 ${size} rounded-md overflow-hidden bg-black/40 border border-white/10`}>
      {e?.thumb
        ? <img src={e.thumb} alt={e.name} className="w-full h-full object-cover" draggable={false} />
        : <span className="flex items-center justify-center w-full h-full text-white/20"><Icon.faces size={14} /></span>}
    </span>
  );

  return (
    <div className={`rounded-xl bg-black/45 border border-white/5 ${pickerOpen ? 'relative z-20' : ''}`}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 text-left select-none rounded-t-xl hover:bg-white/[0.02] transition-colors"
      >
        <span className="font-semibold text-micro uppercase tracking-[0.14em] text-white/50">
          Faceset library
        </span>
        <span className="flex items-center gap-2">
          {entries.length > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-white/[0.04] text-micro text-white/50 border border-white/5">
              {entries.length}
            </span>
          )}
          <span className="text-white/40 text-xs">{open ? '▲' : '▼'}</span>
        </span>
      </button>

      {open && (
        <div className="px-3.5 pb-3.5 space-y-3">
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="primary" disabled={!canSave || busy} onClick={saveCurrent}>
              Save selected face
            </Button>
            <Button size="sm" variant="secondary" onClick={() => importRef.current?.click()}>⬆ Import .fsz</Button>
            <Button size="sm" variant="secondary" onClick={openFolder}>Open folder</Button>
            <Button size="sm" variant="secondary" disabled={busy || entries.length === 0} onClick={rebuildThumbs} title="Regenerate previews, picking the most frontal face in each set">Fix thumbnails</Button>
            <Button size="sm" variant="secondary" onClick={refresh}>↻</Button>
            <input ref={importRef} type="file" accept=".fsz" multiple className="hidden" onChange={onImport} />
          </div>

          {entries.length === 0 ? (
            <p className="text-mini text-white/45 leading-relaxed">
              No saved facesets yet. Select a source face above and hit
              <span className="text-white/55"> Save selected face</span> to keep it here — it survives
              restarts, so you never re-upload. Set the folder to a cloud drive in Settings to sync across devices.
            </p>
          ) : (
            <div>
              <div className="text-micro uppercase tracking-[0.12em] text-white/45 mb-1.5">Load a faceset</div>
              <div className="relative" ref={pickerRef}>
                {/* Closed dropdown box (mirrors the Select control) */}
                <button
                  type="button"
                  onClick={() => setPickerOpen((v) => !v)}
                  className="w-full flex items-center justify-between gap-2 bg-black/40 border border-white/10 hover:border-white/20 rounded-lg px-2.5 py-2 text-left transition-colors"
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <Thumb e={selected} size="w-6 h-6" />
                    <span className={`truncate text-note ${selected ? 'text-white/85' : 'text-white/45'}`}>
                      {selected ? selected.name : 'Select a faceset…'}
                    </span>
                  </span>
                  <span className={`text-white/40 text-xs shrink-0 transition-transform ${pickerOpen ? 'rotate-180' : ''}`}>⌄</span>
                </button>

                {/* Open menu: absolute under the trigger. The library card gets
                    relative z-20 while open so this paints above the Input-faces
                    gallery that follows it in the same Section. */}
                {pickerOpen && (
                  <div className="absolute z-50 left-0 right-0 mt-1 rounded-lg bg-[#181016] border border-white/10 shadow-2xl shadow-black/70 overflow-hidden">
                    {entries.length > 6 && (
                      <div className="p-1.5 border-b border-white/5">
                        <input
                          autoFocus
                          value={filter}
                          onChange={(ev) => setFilter(ev.target.value)}
                          placeholder={`Search ${entries.length} facesets…`}
                          className="w-full bg-black/40 border border-white/10 focus:border-[var(--accent)]/40 rounded-md px-2.5 py-1.5 text-mini text-white/80 placeholder-white/25 outline-none"
                        />
                      </div>
                    )}
                    <div className="max-h-60 overflow-y-auto py-1 [scrollbar-width:thin]">
                      {shown.length === 0 ? (
                        <p className="text-mini text-white/45 py-2 text-center">No match for “{filter}”.</p>
                      ) : shown.map((e) => (
                        <div
                          key={e.filename}
                          className={`group flex items-center gap-2.5 px-2 py-1.5 mx-1 rounded-md cursor-pointer transition-colors ${selected?.filename === e.filename ? 'bg-[var(--accent)]/10' : 'hover:bg-white/[0.06]'}`}
                          onClick={() => { if (renaming !== e.filename) load(e); }}
                        >
                          <Thumb e={e} size="w-8 h-8" />
                          {renaming === e.filename ? (
                            <input
                              autoFocus
                              value={renameVal}
                              onClick={(ev) => ev.stopPropagation()}
                              onChange={(ev) => setRenameVal(ev.target.value)}
                              onBlur={() => commitRename(e)}
                              onKeyDown={(ev) => { ev.stopPropagation(); if (ev.key === 'Enter') commitRename(e); if (ev.key === 'Escape') cancelRename(); }}
                              className="flex-1 min-w-0 bg-black/50 border border-[var(--accent)]/40 rounded px-1.5 py-0.5 text-mini text-white/90 outline-none"
                            />
                          ) : (
                            <span className="flex-1 min-w-0 truncate text-note text-white/80" title={e.name}>
                              {e.name}
                              {e.faces > 1 && <span className="text-white/35"> · {e.faces} faces</span>}
                            </span>
                          )}
                          {/* focus-within as well as group-hover — these are
                              real tab stops, and opacity-0 alone left a
                              keyboard user focused on invisible controls. */}
                          <span className="flex items-center gap-1.5 text-mini text-white/45 shrink-0 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                            <button type="button" className="hover:text-white/80 transition-colors" title="Rename" aria-label={`Rename faceset ${e.name}`} onClick={(ev) => { ev.stopPropagation(); beginRename(e); }}><Icon.rename size={12} /></button>
                            <a className="hover:text-white/80 transition-colors" title="Export .fsz" aria-label={`Export faceset ${e.name}`} href={fileUrl(e.path)} download={e.filename} onClick={(ev) => ev.stopPropagation()}><Icon.download size={12} /></a>
                            <button type="button" className="hover:text-[var(--accent)] transition-colors" title="Delete" aria-label={`Delete faceset ${e.name}`} onClick={(ev) => { ev.stopPropagation(); del(e); }}><Icon.trash size={12} /></button>
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
