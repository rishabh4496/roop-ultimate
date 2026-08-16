import React, { useEffect, useState, useRef } from 'react';
import { getJSON, postJSON, postFiles, fileUrl } from '../api';
import { Button, Card } from './ui';
import { confirmDialog } from './confirm';
import { Icon } from '../icons';
import OutputCompare from './OutputCompare';

export default function Gallery({ notify, setSettings, setTab }) {
  const [files, setFiles] = useState([]);
  const [outputPath, setOutputPath] = useState('');
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all'); // 'all', 'video', 'image'
  const [sortBy, setSortBy] = useState('new'); // 'new' | 'old' | 'big' | 'name'
  const [viewMode, setViewMode] = useState('grid'); // 'grid' | 'list'
  const [selectedFiles, setSelectedFiles] = useState(new Set());
  const [busyFile, setBusyFile] = useState(''); // tracking loading reuse actions
  // Run-history entries keyed by output basename → "how was this file made?"
  const [historyByName, setHistoryByName] = useState({});
  // The two files being compared, as [nameA, nameB]. Driven from the same
  // selection the bulk actions use, so "pick two, compare them" needs no
  // separate mode to enter.
  const [comparePair, setComparePair] = useState(null);

  const fetchOutputs = async () => {
    setLoading(true);
    try {
      const res = await getJSON('/api/output');
      setFiles(res.files || []);
      setOutputPath(res.output_path || '');
    } catch (e) {
      notify(e.message, 'error');
    } finally {
      setLoading(false);
    }
    // History is best-effort — the gallery works fine without it.
    try {
      const h = await getJSON('/api/history');
      const map = {};
      (h.entries || []).forEach((entry) => {
        (entry.outputs || []).forEach((name) => { if (!map[name]) map[name] = entry; });
      });
      setHistoryByName(map);
    } catch { /* no history yet */ }
  };

  // Re-apply the exact settings a past run used, then jump to the Face Swap tab.
  const loadRunSettings = (entry) => {
    if (!setSettings || !entry?.settings) return;
    setSettings((s) => ({ ...(s || {}), ...entry.settings }));
    notify(`Loaded the settings this file was rendered with (${new Date(entry.time * 1000).toLocaleString()})`);
    if (setTab) setTab('faceswap');
  };

  /* eslint-disable react-hooks/exhaustive-deps -- intentional: fetch once on mount */
  useEffect(() => {
    fetchOutputs();
  }, []);
  /* eslint-enable react-hooks/exhaustive-deps */

  const revealFolder = async () => {
    try {
      await postJSON('/api/reveal', {});
      notify('Opened output directory');
    } catch (e) {
      notify(e.message, 'error');
    }
  };

  const revealFile = async (name) => {
    try {
      const fullPath = `${outputPath}/${name}`;
      await postJSON('/api/reveal', { path: fullPath });
      notify(`Revealed ${name}`);
    } catch (e) {
      notify(e.message, 'error');
    }
  };

  const deleteFile = async (name) => {
    if (!(await confirmDialog({ title: 'Delete output?', message: `Delete “${name}”? This removes the file from the output folder.`, confirmLabel: 'Delete', danger: true }))) return;
    try {
      await postJSON('/api/output/delete', { name });
      notify(`Deleted ${name}`);
      setFiles((prev) => prev.filter((f) => f.name !== name));
      // Drop it from the selection too, or the bulk count keeps counting a
      // file that is already gone and the next bulk delete 404s on it.
      setSelectedFiles((prev) => {
        if (!prev.has(name)) return prev;
        const next = new Set(prev);
        next.delete(name);
        return next;
      });
    } catch (e) {
      notify(e.message, 'error');
    }
  };

  const toggleSelect = (name) => {
    setSelectedFiles((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  // Acts on the VISIBLE rows only, in both directions. Replacing the whole set
  // contradicted the rest of this view: the selection deliberately survives a
  // change of filter or search, so selecting all under one filter used to throw
  // away everything picked under another — and "Deselect All" wiped selections
  // that were not even on screen.
  const toggleSelectAll = () => {
    const visible = filteredFiles.map((f) => f.name);
    setSelectedFiles((prev) => {
      const next = new Set(prev);
      if (allVisibleSelected) visible.forEach((n) => next.delete(n));
      else visible.forEach((n) => next.add(n));
      return next;
    });
  };

  const bulkDelete = async () => {
    // Only ever the files the user can currently SEE. The selection survives a
    // change of filter or search, so acting on the raw set would delete files
    // that scrolled out of view under a filter — with a count that did not
    // match what is on screen.
    const list = [...selectedVisible];
    if (list.length === 0) return;
    if (!(await confirmDialog({
      title: `Delete ${list.length} output files?`,
      message: `Permanently delete ${list.length} selected files from the output directory?`,
      confirmLabel: `Delete ${list.length} files`,
      danger: true
    }))) return;

    const deleted = [];
    const failed = [];
    for (const name of list) {
      try {
        await postJSON('/api/output/delete', { name });
        deleted.push(name);
      } catch (e) {
        failed.push(name);
        notify(`Failed to delete ${name}: ${e.message}`, 'error');
      }
    }
    // A file that failed to delete is still there — it has to stay on screen,
    // and stay selected, or the only sign of the failure is a toast that fades.
    const gone = new Set(deleted);
    setFiles((prev) => prev.filter((f) => !gone.has(f.name)));
    setSelectedFiles((prev) => new Set([...prev].filter((n) => !gone.has(n))));
    if (deleted.length) {
      notify(`Deleted ${deleted.length} file${deleted.length === 1 ? '' : 's'}`
             + (failed.length ? ` · ${failed.length} could not be deleted` : ''));
    }
  };

  const reuseAsTarget = async (name) => {
    setBusyFile(name);
    notify(`Preparing to load ${name} as target...`, 'info');
    try {
      const fullPath = `${outputPath}/${name}`;
      const url = fileUrl(fullPath);
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], name, { type: blob.type });
      
      await postFiles('/api/target/add', [file]);
      notify(`Loaded ${name} into face swap targets queue!`);
    } catch (e) {
      notify(`Failed to reuse as target: ${e.message}`, 'error');
    } finally {
      setBusyFile('');
    }
  };

  const reuseAsSource = async (name) => {
    setBusyFile(name);
    notify(`Extracting faces from ${name}...`, 'info');
    try {
      const fullPath = `${outputPath}/${name}`;
      const url = fileUrl(fullPath);
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], name, { type: blob.type });
      
      const result = await postFiles('/api/source/add', [file]);
      if (result.source_faces && result.source_faces.length > 0) {
        notify(`Successfully extracted ${result.source_faces.length} faces!`);
      } else {
        notify('No faces detected in this output file.', 'error');
      }
    } catch (e) {
      notify(`Failed to reuse as source: ${e.message}`, 'error');
    } finally {
      setBusyFile('');
    }
  };

  const copySelectedPaths = () => {
    const paths = selectedVisible.map((name) => `${outputPath}/${name}`).join('\n');
    navigator.clipboard.writeText(paths);
    notify(`Copied ${selectedVisible.length} output path(s) to clipboard!`);
  };

  // Filter & Search files
  const filteredFiles = files.filter((f) => {
    const matchesSearch = f.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter =
      filterType === 'all' ||
      (filterType === 'video' && f.kind === 'video') ||
      (filterType === 'image' && f.kind === 'image');
    return matchesSearch && matchesFilter;
  }).sort((a, b) => {
    if (sortBy === 'old') return (a.mtime || 0) - (b.mtime || 0);
    if (sortBy === 'big') return (b.size || 0) - (a.size || 0);
    if (sortBy === 'name') return a.name.localeCompare(b.name);
    return (b.mtime || 0) - (a.mtime || 0); // 'new'
  });

  const totalSize = filteredFiles.reduce((acc, f) => acc + (f.size || 0), 0);

  // What "selected" means everywhere in this view: selected AND on screen.
  const selectedVisible = filteredFiles
    .filter((f) => selectedFiles.has(f.name))
    .map((f) => f.name);
  const allVisibleSelected =
    filteredFiles.length > 0 && selectedVisible.length === filteredFiles.length;

  const getFormatDate = (timestamp) => {
    try {
      return new Date(timestamp * 1000).toLocaleString();
    } catch {
      return '';
    }
  };

  const fmtSize = (bytes) => {
    if (!bytes && bytes !== 0) return '';
    if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 ** 3)).toFixed(2)} GB`;
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 ** 2)).toFixed(1)} MB`;
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  };

  return (
    <div className="space-y-6">
      {/* Upper header action bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white/90">Outputs</h2>
          <p className="text-sm text-white/50">Browse, manage, and reuse files from the output folder.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={fetchOutputs} disabled={loading}>
            Refresh
          </Button>
          <Button variant="primary" onClick={revealFolder}>
            Reveal Folder
          </Button>
        </div>
      </div>

      {/* Filter and Search capsule */}
      <Card className="p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="w-full md:w-auto flex-1 max-w-md">
          <input
            type="text"
            placeholder="Search outputs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-3 py-2 rounded-xl glass-input text-white text-sm focus:outline-none"
          />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex gap-1 bg-black/20 p-1 rounded-xl border border-white/5">
            <button
              type="button"
              onClick={() => setViewMode('grid')}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
                viewMode === 'grid' ? 'bg-[var(--accent)] text-white shadow' : 'text-white/60 hover:text-white'
              }`}
              title="Grid View"
            >
              Grid
            </button>
            <button
              type="button"
              onClick={() => setViewMode('list')}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
                viewMode === 'list' ? 'bg-[var(--accent)] text-white shadow' : 'text-white/60 hover:text-white'
              }`}
              title="List View"
            >
              ☰ List
            </button>
          </div>

          <div className="flex gap-1.5 bg-black/20 p-1 rounded-xl border border-white/5">
            {['all', 'video', 'image'].map((t) => (
              <button
                key={t}
                onClick={() => setFilterType(t)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${
                  filterType === t
                    ? 'bg-[#E94560] text-white shadow-[0_2px_8px_rgba(233,69,96,0.3)]'
                    : 'text-white/60 hover:text-white'
                }`}
              >
                {t}s
              </button>
            ))}
          </div>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            title="Sort outputs"
            className="px-3 py-2 rounded-xl glass-input text-white text-xs font-bold focus:outline-none cursor-pointer"
          >
            <option value="new" className="bg-[#121420]">Newest first</option>
            <option value="old" className="bg-[#121420]">Oldest first</option>
            <option value="big" className="bg-[#121420]">Largest first</option>
            <option value="name" className="bg-[#121420]">Name (A–Z)</option>
          </select>
        </div>
      </Card>

      {!loading && files.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 text-mini text-white/45 -mt-2 px-1">
          <div className="flex items-center gap-2">
            <span className="font-bold text-white/60">{filteredFiles.length}</span> shown
            {filteredFiles.length !== files.length && <span>of {files.length}</span>}
            <span className="text-white/20">·</span>
            <span className="font-bold text-white/60">{fmtSize(totalSize)}</span> total
          </div>

          {/* Bulk Selection Actions */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={toggleSelectAll}
              className="px-2 py-1 rounded bg-white/5 hover:bg-white/10 text-white/70 text-micro font-semibold transition-colors"
            >
              {allVisibleSelected ? 'Deselect All' : 'Select All'}
            </button>
            {selectedVisible.length === 2 && (
              <button
                type="button"
                onClick={() => setComparePair([...selectedVisible])}
                className="px-2.5 py-1 rounded bg-[var(--accent)]/20 hover:bg-[var(--accent)]/30 border border-[var(--accent)]/40 text-white text-micro font-bold transition-all animate-fade-in flex items-center gap-1"
                title="Compare these two renders side by side"
              >
                <span>Compare A/B</span>
              </button>
            )}
            {selectedVisible.length > 0 && (
              <button
                type="button"
                onClick={copySelectedPaths}
                className="px-2.5 py-1 rounded bg-white/10 hover:bg-white/20 border border-white/10 text-white text-micro font-bold transition-all flex items-center gap-1"
                title="Copy output paths to clipboard"
              >
                <span>Copy Paths ({selectedVisible.length})</span>
              </button>
            )}
            {selectedVisible.length > 0 && (
              <button
                type="button"
                onClick={bulkDelete}
                className="px-2.5 py-1 rounded bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-200 text-micro font-bold transition-all animate-fade-in flex items-center gap-1"
              >
                <span>Delete Selected ({selectedVisible.length})</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 gap-3">
          <div className="h-8 w-8 rounded-full border-4 border-white/10 border-t-[#E94560] animate-spin" />
          <span className="text-white/40 text-sm font-medium">Scanning outputs folder...</span>
        </div>
      )}

      {/* Empty state */}
      {!loading && filteredFiles.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 border-2 border-dashed border-white/10 rounded-2xl">
          <Icon.outputs size={34} className="mb-2 text-white/20" />
          <span className="text-white/50 text-sm">No files found matching the criteria.</span>
        </div>
      )}

      {/* Grid or List view */}
      {!loading && filteredFiles.length > 0 && (
        viewMode === 'grid' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredFiles.map((file) => {
              const absolutePath = `${outputPath}/${file.name}`;
              const srcUrl = fileUrl(absolutePath);

              return (
                <VideoHoverCard
                  key={file.name}
                  file={file}
                  srcUrl={srcUrl}
                  dateStr={getFormatDate(file.mtime)}
                  sizeStr={fmtSize(file.size)}
                  onDelete={() => deleteFile(file.name)}
                  onReveal={() => revealFile(file.name)}
                  onReuseTarget={() => reuseAsTarget(file.name)}
                  onReuseSource={() => reuseAsSource(file.name)}
                  historyEntry={historyByName[file.name]}
                  onLoadSettings={loadRunSettings}
                  isBusy={busyFile === file.name}
                  isSelected={selectedFiles.has(file.name)}
                  onToggleSelect={() => toggleSelect(file.name)}
                />
              );
            })}
          </div>
        ) : (
          <div className="rounded-2xl glass-panel border border-white/10 overflow-hidden shadow-2xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-white/80 border-collapse">
                <thead>
                  <tr className="border-b border-white/10 bg-black/40 text-white/45 uppercase font-mono text-nano tracking-wider">
                    <th className="p-3 w-10 text-center">
                      <input
                        type="checkbox"
                        checked={allVisibleSelected}
                        onChange={toggleSelectAll}
                        className="rounded accent-[var(--accent)] cursor-pointer"
                      />
                    </th>
                    <th className="p-3">File</th>
                    <th className="p-3">Type</th>
                    <th className="p-3">Size</th>
                    <th className="p-3">Date</th>
                    <th className="p-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {filteredFiles.map((file) => {
                    const absolutePath = `${outputPath}/${file.name}`;
                    const srcUrl = fileUrl(absolutePath);
                    const isSel = selectedFiles.has(file.name);
                    const historyEntry = historyByName[file.name];
                    return (
                      <tr key={file.name} className={`hover:bg-white/[0.04] transition-colors ${isSel ? 'bg-[var(--accent)]/10' : ''}`}>
                        <td className="p-3 text-center">
                          <input
                            type="checkbox"
                            checked={isSel}
                            onChange={() => toggleSelect(file.name)}
                            className="rounded accent-[var(--accent)] cursor-pointer"
                          />
                        </td>
                        <td className="p-3 font-semibold text-white/90 truncate max-w-xs flex items-center gap-2">
                          {file.kind === 'video'
                            ? <Icon.film size={15} className="text-white/40" />
                            : <Icon.still size={15} className="text-white/40" />}
                          <a href={srcUrl} target="_blank" rel="noreferrer" className="hover:text-[var(--accent)] truncate" title={file.name}>
                            {file.name}
                          </a>
                        </td>
                        <td className="p-3 text-white/50 font-mono text-micro uppercase">{file.kind}</td>
                        <td className="p-3 text-white/60 font-mono text-mini tabular-nums">{fmtSize(file.size)}</td>
                        <td className="p-3 text-white/50 text-mini whitespace-nowrap">{getFormatDate(file.mtime)}</td>
                        <td className="p-3 text-right">
                          <div className="flex items-center justify-end gap-1.5">
                            {historyEntry && (
                              <button
                                type="button"
                                onClick={() => loadRunSettings(historyEntry)}
                                className="px-2 py-1 rounded bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/30 text-amber-300 text-micro font-bold"
                                title="Re-apply run settings"
                              >
                                Preset
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={() => reuseAsTarget(file.name)}
                              disabled={busyFile === file.name}
                              className="px-2 py-1 rounded bg-white/5 hover:bg-white/15 text-white/80 text-micro font-bold"
                              title="Reuse as Target"
                            >
                              Target
                            </button>
                            <button
                              type="button"
                              onClick={() => reuseAsSource(file.name)}
                              disabled={busyFile === file.name}
                              className="px-2 py-1 rounded bg-white/5 hover:bg-white/15 text-white/80 text-micro font-bold"
                              title="Extract Face as Source"
                            >
                              Source
                            </button>
                            <button
                              type="button"
                              onClick={() => revealFile(file.name)}
                              className="p-1 rounded bg-white/5 hover:bg-white/15 text-white/70"
                              title="Reveal File"
                              aria-label={`Reveal ${file.name} in the file manager`}
                            >
                              <Icon.reveal size={14} />
                            </button>
                            <button
                              type="button"
                              onClick={() => deleteFile(file.name)}
                              className="p-1 rounded bg-red-500/10 hover:bg-red-500/20 text-red-400"
                              title="Delete File"
                              aria-label={`Delete ${file.name}`}
                            >
                              <Icon.trash size={14} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )
      )}

      {comparePair && (() => {
        // Resolve by name each render: a delete or a refresh while the viewer is
        // open must close it rather than leave it pointing at a missing file.
        const [aName, bName] = comparePair;
        const a = files.find((f) => f.name === aName);
        const b = files.find((f) => f.name === bName);
        if (!a || !b) { setComparePair(null); return null; }
        return (
          <OutputCompare
            a={a}
            b={b}
            aUrl={fileUrl(`${outputPath}/${a.name}`)}
            bUrl={fileUrl(`${outputPath}/${b.name}`)}
            historyA={historyByName[a.name]}
            historyB={historyByName[b.name]}
            onSwap={() => setComparePair([bName, aName])}
            onClose={() => setComparePair(null)}
          />
        );
      })()}
    </div>
  );
}

function VideoHoverCard({ file, srcUrl, dateStr, sizeStr, onDelete, onReveal, onReuseTarget, onReuseSource, historyEntry, onLoadSettings, isBusy, isSelected, onToggleSelect }) {
  const [hovered, setHovered] = useState(false);
  const videoRef = useRef(null);
  const cardRef = useRef(null);
  const isVideo = file.kind === 'video';

  // Only spin up a <video> (and its metadata fetch / decoder) once the card is
  // near the viewport. Otherwise opening a large Outputs folder would mount a
  // decoder for every file at once and stutter. `inView` latches true so a card
  // that's been seen doesn't reload while scrolling — bounding the initial
  // stampede is what matters. Images already use native loading="lazy".
  const [inView, setInView] = useState(false);
  useEffect(() => {
    if (!isVideo || inView) return;
    const el = cardRef.current;
    if (!el) return;
    if (typeof IntersectionObserver === 'undefined') { setInView(true); return; }
    const obs = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { setInView(true); obs.disconnect(); }
    }, { rootMargin: '300px' });
    obs.observe(el);
    return () => obs.disconnect();
  }, [isVideo, inView]);

  useEffect(() => {
    if (!videoRef.current) return;
    if (hovered) {
      videoRef.current.play().catch(() => {});
    } else {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  }, [hovered]);

  return (
    <Card
      // A media tile is exactly the surface tilt was built for: it is a
      // picture, not a form, so depth reads as depth and there are no controls
      // for the movement to fight.
      elevation="hero"
      className={`tap overflow-hidden border flex flex-col group/card transition-all ${
        isSelected ? 'border-[var(--accent)] bg-[var(--accent)]/5 shadow-[0_0_20px_rgba(233,69,96,0.2)]' : 'border-white/5 hover:border-[#E94560]/40 hover:shadow-[0_12px_36px_rgba(233,69,96,0.15)]'
      }`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* File preview box */}
      <div ref={cardRef} className="relative aspect-video bg-black/45 flex items-center justify-center overflow-hidden border-b border-white/5 shrink-0 select-none">
        {onToggleSelect && (
          <div className="absolute top-2 left-2 z-20">
            <input
              type="checkbox"
              checked={!!isSelected}
              onChange={onToggleSelect}
              className="h-4 w-4 rounded accent-[var(--accent)] cursor-pointer shadow-lg"
              title="Select file"
            />
          </div>
        )}
        {isVideo ? (
          inView ? (
            <>
              <video
                ref={videoRef}
                src={srcUrl}
                muted
                loop
                playsInline
                preload="metadata"
                className="w-full h-full object-contain pointer-events-none transition-transform duration-500 group-hover/card:scale-102"
                onError={(e) => {
                  e.target.style.display = 'none';
                  if (e.target.nextElementSibling) e.target.nextElementSibling.style.display = 'flex';
                }}
              />
              <div className="hidden w-full h-full flex-col items-center justify-center text-white/30 text-nano p-2 text-center bg-black/50">
                <Icon.film size={24} className="mb-1 opacity-50" />
                <span className="truncate w-full">{file.name}</span>
              </div>
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-white/15"><Icon.film size={30} /></div>
          )
        ) : (
          <>
            <img
              src={srcUrl}
              alt={file.name}
              loading="lazy"
              className="w-full h-full object-contain pointer-events-none transition-transform duration-500 group-hover/card:scale-105"
              onError={(e) => {
                e.target.style.display = 'none';
                if (e.target.nextElementSibling) e.target.nextElementSibling.style.display = 'flex';
              }}
            />
            <div className="hidden w-full h-full flex-col items-center justify-center text-white/30 text-nano p-2 text-center bg-black/50">
              <Icon.still size={24} className="mb-1 opacity-50" />
              <span className="truncate w-full">{file.name}</span>
            </div>
          </>
        )}

        {/* Video badge */}
        {isVideo && (
          <span className="absolute top-2 right-2 bg-black/70 px-2 py-0.5 rounded text-micro font-semibold text-white tracking-wider pointer-events-none z-10 flex items-center gap-1 border border-white/5">
            Video
          </span>
        )}

        {/* Action hover mask overlay */}
        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover/card:opacity-100 transition-opacity flex flex-col items-center justify-center gap-2 p-4">
          <div className="flex gap-2 w-full max-w-[200px]">
            <Button
              size="sm"
              variant="primary"
              onClick={onReuseTarget}
              disabled={isBusy}
              className="flex-1 justify-center whitespace-nowrap"
            >
              Set Target
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={onReuseSource}
              disabled={isBusy}
              className="flex-1 justify-center whitespace-nowrap"
            >
              Extract Faces
            </Button>
          </div>
          <div className="flex gap-2 w-full max-w-[200px]">
            <Button
              size="sm"
              variant="secondary"
              onClick={onReveal}
              disabled={isBusy}
              className="flex-1 justify-center"
            >
              Reveal
            </Button>
            <a
              href={srcUrl}
              download={file.name}
              className="flex-1 rounded-xl font-bold border border-white/5 bg-white/10 hover:bg-white/15 text-white backdrop-blur-md px-3 py-1.5 text-xs flex items-center justify-center"
            >
              ⬇ Get
            </a>
          </div>
          {historyEntry && (
            <div className="w-full max-w-[200px]">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => onLoadSettings(historyEntry)}
                disabled={isBusy}
                title={`Re-apply the exact settings this file was rendered with (${historyEntry.settings?.swap_model || ''} · ${historyEntry.settings?.selected_enhancer || 'no enhancer'})`}
                className="w-full justify-center whitespace-nowrap"
              >
                Load run settings
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* File details panel */}
      <div className="p-4 flex-1 flex flex-col justify-between gap-2 min-w-0">
        <div className="min-w-0">
          <h4
            className="text-sm font-bold text-white/90 truncate cursor-pointer hover:text-[#E94560]"
            title={file.name}
            onClick={onReveal}
          >
            {file.name}
          </h4>
          <span className="text-micro font-mono text-white/45 block mt-0.5">{dateStr}{sizeStr ? ` · ${sizeStr}` : ''}</span>
        </div>
        <div className="flex justify-between items-center shrink-0">
          <span className="text-micro uppercase font-semibold tracking-wider text-[var(--accent)]/80">
            {file.kind}
          </span>
          <button
            type="button"
            onClick={onDelete}
            disabled={isBusy}
            className="text-mini font-bold text-white/45 hover:text-red-400 cursor-pointer transition-colors"
          >
            Delete
          </button>
        </div>
      </div>
    </Card>
  );
}
