import React, { useCallback, useEffect, useRef, useState } from 'react';
import { facesetAdapter } from '../../adapters/facesetAdapter';

/**
 * Faceset Archive (.fsz) Library Modal for React UI 2.0.
 * Allows managing persistent faceset archives on disk with save, load, rename, delete, import, and reveal.
 */
export function FacesetLibraryModal({
  isOpen = false,
  onClose,
  canSave = false,
  onLoadedFaceset,
  notify,
}) {
  const [entries, setEntries] = useState([]);
  const [busy, setBusy] = useState(false);
  const [search, setSearch] = useState('');
  const [renamingFile, setRenamingFile] = useState(null);
  const [renameVal, setRenameVal] = useState('');
  const [saveName, setSaveName] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const importInputRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const res = await facesetAdapter.getLibrary();
      setEntries(res.entries || []);
    } catch (e) {
      notify?.(e.message, 'danger');
    }
  }, [notify]);

  useEffect(() => {
    if (isOpen) {
      refresh();
      setIsSaving(false);
      setSaveName('');
    }
  }, [isOpen, refresh]);

  if (!isOpen) return null;

  const filteredEntries = search.trim()
    ? entries.filter((e) => e.name.toLowerCase().includes(search.trim().toLowerCase()))
    : entries;

  const handleSave = async () => {
    const name = saveName.trim();
    if (!name) return;
    setBusy(true);
    try {
      const res = await facesetAdapter.saveToLibrary(name);
      setEntries(res.entries || []);
      notify?.(`Saved "${res.saved || name}" to library`, 'success');
      setIsSaving(false);
      setSaveName('');
    } catch (e) {
      notify?.(e.message, 'danger');
    } finally {
      setBusy(false);
    }
  };

  const handleLoad = async (entry) => {
    setBusy(true);
    try {
      const res = await facesetAdapter.loadFromLibrary(entry.filename);
      onLoadedFaceset?.(res);
      notify?.(`Loaded "${entry.name}" into source faces`, 'success');
      onClose?.();
    } catch (e) {
      notify?.(e.message, 'danger');
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (entry) => {
    if (!window.confirm(`Delete "${entry.name}" from disk?`)) return;
    try {
      const res = await facesetAdapter.deleteFromLibrary(entry.filename);
      setEntries(res.entries || []);
      notify?.(`Deleted "${entry.name}"`, 'info');
    } catch (e) {
      notify?.(e.message, 'danger');
    }
  };

  const handleCommitRename = async (entry) => {
    const newName = renameVal.trim();
    if (!newName || newName === entry.name) {
      setRenamingFile(null);
      return;
    }
    try {
      const res = await facesetAdapter.renameInLibrary(entry.filename, newName);
      setEntries(res.entries || []);
      notify?.(`Renamed to "${newName}"`, 'success');
    } catch (e) {
      notify?.(e.message, 'danger');
    } finally {
      setRenamingFile(null);
    }
  };

  const handleImport = async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = '';
    if (!files.length) return;
    setBusy(true);
    try {
      let r;
      for (const f of files) {
        r = await facesetAdapter.importFaceset(f);
      }
      if (r) setEntries(r.entries || []);
      notify?.(`Imported ${files.length} faceset(s)`, 'success');
    } catch (err) {
      notify?.(err.message, 'danger');
    } finally {
      setBusy(false);
    }
  };

  const handleReveal = async () => {
    try {
      await facesetAdapter.revealLibrary();
    } catch (e) {
      notify?.(e.message, 'danger');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div className="relative w-full max-w-2xl max-h-[85vh] flex flex-col rounded-2xl bg-[#0e1118] border border-white/15 shadow-2xl text-white overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
          <div>
            <span className="text-[10px] font-mono uppercase tracking-widest text-[var(--accent-primary,#38bdf8)] font-bold">
              Archive Manager
            </span>
            <h2 className="text-base font-bold text-white">Faceset Library (.fsz)</h2>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleReveal}
              title="Reveal library folder in Explorer"
              className="px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-mono text-white/80 transition-colors"
            >
              Open Folder
            </button>
            <button
              type="button"
              onClick={() => importInputRef.current?.click()}
              title="Import .fsz archive files"
              className="px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-mono text-white/80 transition-colors"
            >
              Import .fsz
            </button>
            <button
              type="button"
              onClick={onClose}
              className="w-8 h-8 rounded-lg flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 text-base"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Hidden Import Input */}
        <input
          ref={importInputRef}
          type="file"
          multiple
          accept=".fsz"
          onChange={handleImport}
          className="hidden"
        />

        {/* Search & Save Bar */}
        <div className="flex items-center gap-3 px-5 py-3 border-b border-white/5 bg-black/30">
          <input
            type="text"
            placeholder="Search saved facesets..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-white placeholder-white/40 outline-none focus:border-[var(--accent-primary,#38bdf8)]"
          />

          {canSave && !isSaving && (
            <button
              type="button"
              onClick={() => setIsSaving(true)}
              className="px-3 py-1.5 rounded-lg bg-[var(--accent-primary,#38bdf8)] text-black font-bold text-xs hover:brightness-110 transition-all shrink-0"
            >
              + Save Current Faces
            </button>
          )}
        </div>

        {/* Inline Save Prompt */}
        {isSaving && (
          <div className="flex items-center gap-2 px-5 py-3 bg-[var(--accent-primary,#38bdf8)]/10 border-b border-[var(--accent-primary,#38bdf8)]/20">
            <input
              type="text"
              autoFocus
              placeholder="e.g. Scarlett — Hero Angles"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSave();
                if (e.key === 'Escape') setIsSaving(false);
              }}
              className="flex-1 px-3 py-1.5 rounded-lg bg-black/80 border border-[var(--accent-primary,#38bdf8)] text-xs text-white outline-none"
            />
            <button
              type="button"
              onClick={handleSave}
              disabled={busy || !saveName.trim()}
              className="px-3 py-1.5 rounded-lg bg-[var(--accent-primary,#38bdf8)] text-black font-bold text-xs disabled:opacity-50"
            >
              Save .fsz
            </button>
            <button
              type="button"
              onClick={() => setIsSaving(false)}
              className="px-2.5 py-1.5 rounded-lg bg-white/10 text-white text-xs hover:bg-white/20"
            >
              Cancel
            </button>
          </div>
        )}

        {/* Entries Grid */}
        <div className="flex-1 overflow-y-auto p-5 grid grid-cols-1 sm:grid-cols-2 gap-3 min-h-[260px]">
          {filteredEntries.length > 0 ? (
            filteredEntries.map((entry) => {
              const isRenaming = renamingFile === entry.filename;

              return (
                <div
                  key={`fsz-${entry.filename}`}
                  className="group flex items-center justify-between p-3 rounded-xl bg-white/[0.03] border border-white/10 hover:border-white/20 hover:bg-white/[0.05] transition-colors"
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0 pr-2">
                    {/* Thumbnail */}
                    <div className="w-12 h-12 rounded-lg bg-black/60 overflow-hidden shrink-0 border border-white/10 flex items-center justify-center">
                      {entry.thumb ? (
                        <img
                          src={entry.thumb}
                          alt={entry.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-white/20 text-xs font-mono">FSZ</span>
                      )}
                    </div>

                    {/* Metadata */}
                    <div className="flex flex-col flex-1 min-w-0">
                      {isRenaming ? (
                        <input
                          type="text"
                          value={renameVal}
                          autoFocus
                          onChange={(e) => setRenameVal(e.target.value)}
                          onBlur={() => handleCommitRename(entry)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleCommitRename(entry);
                            if (e.key === 'Escape') setRenamingFile(null);
                          }}
                          className="px-2 py-0.5 rounded bg-black/80 border border-[var(--accent-primary,#38bdf8)] text-xs text-white outline-none"
                        />
                      ) : (
                        <span className="text-xs font-bold text-white truncate group-hover:text-[var(--accent-primary,#38bdf8)] transition-colors">
                          {entry.name}
                        </span>
                      )}
                      <span className="text-[10px] font-mono text-white/40">
                        {entry.faces_count ? `${entry.faces_count} faces` : '.fsz archive'}
                      </span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button
                      type="button"
                      onClick={() => handleLoad(entry)}
                      disabled={busy}
                      className="px-2.5 py-1 rounded-lg bg-[var(--accent-primary,#38bdf8)]/20 hover:bg-[var(--accent-primary,#38bdf8)] text-[var(--accent-primary,#38bdf8)] hover:text-black font-semibold text-xs transition-colors"
                    >
                      Load
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setRenamingFile(entry.filename);
                        setRenameVal(entry.name);
                      }}
                      title="Rename faceset"
                      className="w-7 h-7 rounded-lg flex items-center justify-center text-white/50 hover:text-white hover:bg-white/10 text-xs"
                    >
                      ✎
                    </button>

                    <button
                      type="button"
                      onClick={() => handleDelete(entry)}
                      title="Delete faceset from disk"
                      className="w-7 h-7 rounded-lg flex items-center justify-center text-red-400/60 hover:text-red-400 hover:bg-red-500/10 text-xs"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="col-span-full flex flex-col items-center justify-center p-8 text-center text-white/40">
              <span className="text-2xl mb-1">📂</span>
              <p className="text-xs font-medium">No faceset archives found</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
