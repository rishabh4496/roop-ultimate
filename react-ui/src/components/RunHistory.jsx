import React, { useEffect, useMemo, useState } from 'react';
import { getJSON, postJSON, fileUrl } from '../api';
import { Button, Card, MotionIcon } from './ui';
import { confirmDialog } from './confirm';
import { LABELS, CHIP_KEYS, fmtDur, fmtVal, primitives, diffSettings } from './settingsDiff';
import { Icon } from '../icons';

/**
 * Run History — a browsable record of every completed swap.
 *
 * The live DiagnosticsPanel answers "how is THIS run going" and then vanishes
 * the moment the job ends. This view is the durable other half: each finished
 * run is kept (backend run_history.json) with the exact settings it used, how
 * long it took, and its average throughput — so "was this slower than last
 * time?" and "what did I change between these two renders?" become one glance
 * instead of guesswork. Any two runs can be diffed field-by-field, and a run's
 * settings can be re-loaded straight back into the Face Swap tab.
 */

// The settings vocabulary (labels, chip keys, diff) is shared with the Outputs
// tab's A/B comparison — see components/settingsDiff.js.
const fmtRel = (t) => {
  const d = (Date.now() - t * 1000) / 1000;
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  const days = Math.floor(d / 86400);
  if (days < 7) return `${days}d ago`;
  return new Date(t * 1000).toLocaleDateString();
};

export default function RunHistory({ notify, setSettings, setTab }) {
  const [entries, setEntries] = useState([]);
  const [outputPath, setOutputPath] = useState('');
  const [existing, setExisting] = useState({}); // basename -> kind, for still-present outputs
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [compare, setCompare] = useState([]); // up to two run ids
  const [presets, setPresets] = useState([]);

  const load = async () => {
    setLoading(true);
    try {
      const h = await getJSON('/api/history');
      setEntries(h.entries || []);
    } catch (e) {
      notify?.(e.message, 'error');
    } finally {
      setLoading(false);
    }
    // Output listing is best-effort: it lets us show a thumbnail and flag runs
    // whose file has since been deleted. History works fully without it.
    try {
      const o = await getJSON('/api/output');
      setOutputPath(o.output_path || '');
      const map = {};
      (o.files || []).forEach((f) => { map[f.name] = f.kind; });
      setExisting(map);
    } catch { /* no outputs endpoint / folder */ }
  };

  /* eslint-disable react-hooks/exhaustive-deps -- fetch once on mount */
  useEffect(() => { load(); }, []);
  useEffect(() => {
    getJSON('/api/export/presets').then((r) => setPresets(r.presets || [])).catch(() => {});
  }, []);
  /* eslint-enable react-hooks/exhaustive-deps */

  const toggleCompare = (id) => {
    setCompare((cur) => {
      if (cur.includes(id)) return cur.filter((x) => x !== id);
      if (cur.length >= 2) return [cur[1], id]; // keep the most recent two picks
      return [...cur, id];
    });
  };

  const loadRunSettings = (entry) => {
    if (!setSettings || !entry?.settings) return;
    setSettings((s) => ({ ...(s || {}), ...entry.settings }));
    notify?.(`Loaded the settings from this run (${new Date(entry.time * 1000).toLocaleString()})`);
    setTab?.('faceswap');
  };

  const revealOutput = async (name) => {
    try {
      await postJSON('/api/reveal', { path: `${outputPath}/${name}` });
    } catch (e) {
      notify?.(e.message, 'error');
    }
  };

  const exportRun = async (name, presetKey) => {
    try {
      const res = await postJSON('/api/export/apply', { source: name, preset: presetKey });
      notify?.(`Exported ${res.name}`);
    } catch (e) {
      notify?.(e.message, 'error');
    }
  };

  const deleteEntry = async (entry) => {
    if (!(await confirmDialog({
      title: 'Remove from history?',
      message: 'This deletes only the history record. The output file itself is kept.',
      confirmLabel: 'Remove', danger: true,
    }))) return;
    try {
      await postJSON('/api/history/delete', { id: entry.id });
      setEntries((prev) => prev.filter((e) => e.id !== entry.id));
      setCompare((cur) => cur.filter((x) => x !== entry.id));
    } catch (e) {
      notify?.(e.message, 'error');
    }
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter((e) => {
      if ((e.outputs || []).some((n) => n.toLowerCase().includes(q))) return true;
      const p = primitives(e.settings);
      return Object.values(p).some((v) => fmtVal(v).toLowerCase().includes(q));
    });
  }, [entries, query]);

  // Roll-up figures for the header. Only runs that recorded a duration count
  // toward render time / throughput, so a fresh install with old entries reads
  // honestly instead of showing zeros.
  const summary = useMemo(() => {
    const timed = entries.filter((e) => e.duration_s > 0);
    const totalTime = timed.reduce((a, e) => a + e.duration_s, 0);
    const totalFrames = entries.reduce((a, e) => a + (e.frames || 0), 0);
    const fpsVals = entries.filter((e) => e.fps > 0).map((e) => e.fps);
    const avgFps = fpsVals.length ? fpsVals.reduce((a, b) => a + b, 0) / fpsVals.length : 0;
    return { count: entries.length, totalTime, totalFrames, avgFps, medianFps: median(fpsVals) };
  }, [entries]);

  const cmpA = entries.find((e) => e.id === compare[0]);
  const cmpB = entries.find((e) => e.id === compare[1]);

  const exportTelemetryCSV = () => {
    try {
      const headers = ['ID', 'Date', 'Output Name', 'Duration (s)', 'Frames', 'FPS', 'Enhancer', 'Swap Model'];
      const rows = entries.map((e) => [
        e.id,
        new Date(e.time * 1000).toISOString(),
        `"${e.outputs?.[0] || ''}"`,
        e.duration_s || 0,
        e.frames || 0,
        e.fps || 0,
        `"${e.settings?.selected_enhancer || 'None'}"`,
        `"${e.settings?.swap_model || 'inswapper'}"`,
      ]);

      const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `roop_telemetry_history_${Date.now()}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      notify?.('Exported Telemetry History (.csv)', 'success');
    } catch (err) {
      notify?.('Failed to export CSV: ' + err.message, 'error');
    }
  };

  const recentRunsForGraph = useMemo(() => {
    return entries.slice(0, 12).reverse();
  }, [entries]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white/90">Run History & Performance Telemetry</h2>
          <p className="text-sm text-white/50">Every completed swap, with the settings and throughput it ran at. Pick two to compare.</p>
        </div>
        <div className="flex items-center gap-2">
          {entries.length > 0 && (
            <Button variant="secondary" size="sm" onClick={exportTelemetryCSV}>
              <Icon.download size={14} className="mr-1" /> Export CSV
            </Button>
          )}
          <Button variant="secondary" size="sm" onClick={load} disabled={loading}>Refresh</Button>
        </div>
      </div>

      {/* Visual Performance Telemetry Graph */}
      {recentRunsForGraph.length > 0 && (
        <Card className="p-4 space-y-2 bg-black/40 border border-white/10">
          <div className="flex items-center justify-between text-xs">
            <span className="font-bold text-white/90 flex items-center gap-2">
              <MotionIcon icon={Icon.meter} size="sm" variant="emerald" /> Recent Runs Throughput (FPS)
            </span>
            <span className="text-micro text-white/40 font-mono">Last 12 Runs</span>
          </div>

          <div className="flex items-end gap-2 h-24 pt-2">
            {recentRunsForGraph.map((r, idx) => {
              const maxFps = Math.max(...recentRunsForGraph.map((x) => x.fps || 1), 60);
              const hPct = Math.min(100, Math.max(12, ((r.fps || 1) / maxFps) * 100));
              return (
                <div key={idx} className="flex-1 flex flex-col items-center gap-1 group relative">
                  {/* Tooltip */}
                  <div className="absolute bottom-full mb-1 hidden group-hover:flex flex-col items-center bg-black/90 p-1.5 rounded-lg border border-white/10 text-nano text-white z-20 whitespace-nowrap shadow-xl">
                    <span className="font-bold">{r.outputs?.[0] || `Run #${idx + 1}`}</span>
                    <span className="text-emerald-400 font-mono">{r.fps ? r.fps.toFixed(1) : 0} FPS</span>
                    <span className="text-white/50">{r.frames || 1} frames</span>
                  </div>

                  <div
                    className="w-full bg-emerald-500/30 hover:bg-emerald-400 border border-emerald-500/50 rounded-t transition-all"
                    style={{ height: `${hPct}%` }}
                  />
                  <span className="text-nano font-mono text-white/40 truncate max-w-full">
                    {r.fps ? Math.round(r.fps) : '—'}
                  </span>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Summary strip */}
      {entries.length > 0 && (
        <div className="rounded-xl border border-white/[0.07] bg-black/30 flex flex-wrap items-stretch divide-x divide-white/[0.06]">
          <Sum label="Runs" value={summary.count} />
          <Sum label="Total render time" value={fmtDur(summary.totalTime) || '—'} />
          <Sum label="Frames rendered" value={summary.totalFrames ? summary.totalFrames.toLocaleString() : '—'} />
          <Sum label="Median throughput" value={summary.medianFps ? `${summary.medianFps.toFixed(1)} fps` : '—'}
               tone="text-emerald-400" />
        </div>
      )}

      {/* Compare panel */}
      {cmpA && cmpB && <ComparePanel a={cmpA} b={cmpB} onClear={() => setCompare([])} onLoad={loadRunSettings} />}

      {/* Search */}
      {entries.length > 0 && (
        <input
          type="text"
          placeholder="Search by output name or a setting value…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full md:max-w-md px-3 py-2 rounded-xl glass-input text-white text-sm focus:outline-none"
        />
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-20 gap-3">
          <div className="h-8 w-8 rounded-full border-4 border-white/10 border-t-[var(--accent)] animate-spin" />
          <span className="text-white/40 text-sm font-medium">Loading run history…</span>
        </div>
      )}

      {!loading && entries.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 border-2 border-dashed border-white/10 rounded-2xl text-center">
          <Icon.history size={34} className="mb-2 text-white/20" />
          <span className="text-white/60 text-sm font-medium">No runs recorded yet</span>
          <span className="text-white/35 text-xs mt-1">Finish a swap and it will appear here with its settings and timing.</span>
        </div>
      )}

      {!loading && entries.length > 0 && filtered.length === 0 && (
        <div className="text-center text-white/40 text-sm py-10">No runs match “{query}”.</div>
      )}

      {/* Run list */}
      {!loading && filtered.length > 0 && (
        <div className="space-y-3">
          {filtered.map((entry) => (
            <RunRow
              key={entry.id}
              entry={entry}
              outputPath={outputPath}
              existing={existing}
              presets={presets}
              selected={compare.includes(entry.id)}
              compareFull={compare.length >= 2 && !compare.includes(entry.id)}
              onToggleCompare={() => toggleCompare(entry.id)}
              onLoad={() => loadRunSettings(entry)}
              onReveal={revealOutput}
              onExport={exportRun}
              onDelete={() => deleteEntry(entry)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function Sum({ label, value, tone = 'text-white/85' }) {
  return (
    <div className="px-4 py-2.5 min-w-0">
      <div className="text-nano font-semibold uppercase tracking-[0.14em] text-white/45">{label}</div>
      <div className={`font-mono text-lead font-bold tabular-nums truncate ${tone}`}>{value}</div>
    </div>
  );
}

function Chip({ k, v }) {
  return (
    <span className="inline-flex items-baseline gap-1 px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.07] text-micro">
      <span className="text-white/35">{LABELS[k] || k}</span>
      <span className="font-semibold text-white/70">{fmtVal(v)}</span>
    </span>
  );
}

function RunRow({ entry, outputPath, existing, presets, selected, compareFull, onToggleCompare, onLoad, onReveal, onExport, onDelete }) {
  const outputs = entry.outputs || [];
  // Pick the first output that still exists on disk for the thumbnail.
  const liveName = outputs.find((n) => existing[n]);
  const kind = liveName ? existing[liveName] : null;
  const allDeleted = outputs.length > 0 && !liveName;
  const [exporting, setExporting] = useState(false);

  const handleExportChange = async (e) => {
    const presetKey = e.target.value;
    e.target.value = '';
    if (!presetKey || !liveName) return;
    setExporting(true);
    try {
      await onExport(liveName, presetKey);
    } finally {
      setExporting(false);
    }
  };

  const p = primitives(entry.settings);
  const chips = CHIP_KEYS.filter((k) => p[k] !== undefined && p[k] !== '' && p[k] !== null);

  return (
    <Card className={`p-3 border transition-colors ${selected ? 'border-[var(--accent)]/50 bg-[var(--accent)]/[0.04]' : 'border-white/[0.06]'}`}>
      <div className="flex gap-3">
        {/* Thumbnail */}
        <div className="relative w-32 h-20 shrink-0 rounded-lg overflow-hidden bg-black/45 border border-white/5 grid place-items-center">
          {liveName ? (
            kind === 'video' ? (
              <video src={fileUrl(`${outputPath}/${liveName}`)} muted playsInline preload="metadata"
                     className="w-full h-full object-contain" />
            ) : (
              <img src={fileUrl(`${outputPath}/${liveName}`)} alt={liveName} loading="lazy"
                   className="w-full h-full object-contain" />
            )
          ) : (
            <span className="text-white/20">{allDeleted ? <Icon.trash size={22} /> : <Icon.film size={22} />}</span>
          )}
        </div>

        {/* Body */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-semibold text-white/85 text-sm truncate max-w-[22rem]" title={outputs.join(', ')}>
                  {outputs[0] || 'run'}
                </span>
                {outputs.length > 1 && (
                  <span className="text-micro text-white/45">+{outputs.length - 1} more</span>
                )}
                {allDeleted && (
                  <span className="text-nano uppercase tracking-wide text-amber-400/70 border border-amber-400/20 rounded px-1 py-0.5">file removed</span>
                )}
              </div>
              <div className="mt-0.5 flex items-center gap-2 text-mini text-white/45 font-mono">
                <span title={new Date(entry.time * 1000).toLocaleString()}>{fmtRel(entry.time)}</span>
                {entry.duration_s > 0 && (<><span className="text-white/15">·</span><span>{fmtDur(entry.duration_s)}</span></>)}
                {entry.fps > 0 && (
                  <>
                    <span className="text-white/15">·</span>
                    <span className={entry.fps < 6 ? 'text-amber-400/80' : 'text-emerald-400/80'}>{entry.fps.toFixed(1)} fps</span>
                  </>
                )}
                {entry.frames > 0 && (<><span className="text-white/15">·</span><span>{entry.frames.toLocaleString()} frames</span></>)}
              </div>
            </div>
          </div>

          {/* Setting chips */}
          <div className="mt-2 flex flex-wrap gap-1.5">
            {chips.length ? chips.map((k) => <Chip key={k} k={k} v={p[k]} />)
              : <span className="text-mini text-white/45">no settings recorded</span>}
          </div>

          {/* Actions */}
          <div className="mt-2.5 flex items-center gap-1.5 flex-wrap">
            <button
              type="button"
              onClick={onToggleCompare}
              disabled={compareFull}
              title={compareFull ? 'Two runs already selected — clear one first' : 'Select this run to compare'}
              className={`px-2.5 py-1 rounded-lg text-mini font-semibold border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                selected
                  ? 'bg-[var(--accent)]/20 border-[var(--accent)]/40 text-[var(--accent)]'
                  : 'bg-white/[0.04] border-white/10 text-white/55 hover:text-white hover:border-white/20'
              }`}
            >
              {selected ? '✓ Comparing' : '⇄ Compare'}
            </button>
            <button type="button" onClick={onLoad}
                    className="px-2.5 py-1 rounded-lg text-mini font-semibold bg-white/[0.04] border border-white/10 text-white/55 hover:text-white hover:border-white/20 transition-colors">
              ↻ Load settings
            </button>
            {liveName && kind === 'video' && presets.length > 0 && (
              <select
                onChange={handleExportChange}
                disabled={exporting}
                defaultValue=""
                title="Export a crop-to-fill copy for a social aspect ratio"
                className="px-2 py-1 rounded-lg text-mini font-semibold bg-white/[0.04] border border-white/10 text-white/55 hover:text-white hover:border-white/20 transition-colors disabled:opacity-40 cursor-pointer"
              >
                <option value="" disabled>{exporting ? 'Exporting…' : 'Export…'}</option>
                {presets.map((p) => <option key={p.key} value={p.key} className="bg-[#121420]">{p.label}</option>)}
              </select>
            )}
            {liveName && (
              <button type="button" onClick={() => onReveal(liveName)}
                      className="px-2.5 py-1 rounded-lg text-mini font-semibold bg-white/[0.04] border border-white/10 text-white/55 hover:text-white hover:border-white/20 transition-colors">
                Reveal
              </button>
            )}
            <button type="button" onClick={onDelete}
                    className="px-2.5 py-1 rounded-lg text-mini font-semibold bg-white/[0.02] border border-white/5 text-white/45 hover:text-red-300 hover:border-red-500/30 transition-colors ml-auto">
              Remove
            </button>
          </div>
        </div>
      </div>
    </Card>
  );
}

function ComparePanel({ a, b, onClear, onLoad }) {
  // Field-by-field diff: the union of primitive keys across both runs, split
  // into those that differ (the interesting part) and those that match.
  const { changed, same } = useMemo(() => {
    return diffSettings(a.settings, b.settings);
  }, [a, b]);

  const [showSame, setShowSame] = useState(false);

  const Head = ({ e, side }) => (
    <div className="min-w-0">
      <div className="text-nano font-semibold uppercase tracking-[0.14em] text-white/45">{side}</div>
      <div className="font-semibold text-white/85 text-sm truncate" title={(e.outputs || []).join(', ')}>
        {(e.outputs || [])[0] || 'run'}
      </div>
      <div className="mt-0.5 flex items-center gap-2 text-micro font-mono text-white/45">
        <span>{new Date(e.time * 1000).toLocaleString()}</span>
        {e.duration_s > 0 && <span>· {fmtDur(e.duration_s)}</span>}
        {e.fps > 0 && <span className="text-emerald-400/80">· {e.fps.toFixed(1)} fps</span>}
      </div>
      <button type="button" onClick={() => onLoad(e)}
              className="mt-1.5 px-2 py-0.5 rounded-md text-micro font-semibold bg-white/[0.05] border border-white/10 text-white/60 hover:text-white transition-colors">
        ↻ Load these settings
      </button>
    </div>
  );

  return (
    <Card className="p-4 border border-[var(--accent)]/25 bg-[var(--accent)]/[0.03]">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="grid grid-cols-2 gap-4 flex-1 min-w-0">
          <Head e={a} side="Run A" />
          <Head e={b} side="Run B" />
        </div>
        <button type="button" onClick={onClear}
                className="shrink-0 text-white/40 hover:text-white text-lg leading-none px-1"
                title="Close comparison" aria-label="Close comparison">✕</button>
      </div>

      {/* Speed delta */}
      {a.fps > 0 && b.fps > 0 && (
        <div className="mb-3 text-mini font-mono text-white/50">
          {b.fps === a.fps ? 'Same throughput.' : (
            <>Run B is <span className={b.fps > a.fps ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'}>
              {b.fps > a.fps ? `${((b.fps / a.fps - 1) * 100).toFixed(0)}% faster` : `${((1 - b.fps / a.fps) * 100).toFixed(0)}% slower`}
            </span> than Run A ({a.fps.toFixed(1)} → {b.fps.toFixed(1)} fps).</>
          )}
        </div>
      )}

      {/* Changed settings */}
      <div className="text-nano font-semibold uppercase tracking-[0.14em] text-white/45 mb-1.5">
        {changed.length} setting{changed.length === 1 ? '' : 's'} differ
      </div>
      {changed.length === 0 ? (
        <div className="text-note text-white/45">These two runs used identical settings.</div>
      ) : (
        <div className="rounded-lg border border-white/[0.07] overflow-hidden divide-y divide-white/[0.05]">
          {changed.map(({ k, va, vb }) => (
            <div key={k} className="grid grid-cols-[1fr_1fr_1fr] gap-2 px-3 py-1.5 text-mini font-mono items-baseline">
              <span className="text-white/40 truncate" title={k}>{LABELS[k] || k}</span>
              <span className="text-amber-300/80 truncate text-right">{fmtVal(va)}</span>
              <span className="text-emerald-300/80 truncate text-right">{fmtVal(vb)}</span>
            </div>
          ))}
        </div>
      )}

      {same.length > 0 && (
        <div className="mt-2">
          <button type="button" onClick={() => setShowSame((v) => !v)}
                  className="text-micro font-semibold text-white/45 hover:text-white/60 transition-colors">
            {showSame ? '▾ Hide' : '▸ Show'} {same.length} matching setting{same.length === 1 ? '' : 's'}
          </button>
          {showSame && (
            <div className="mt-1.5 rounded-lg border border-white/[0.05] overflow-hidden divide-y divide-white/[0.04]">
              {same.map(({ k, va }) => (
                <div key={k} className="grid grid-cols-[1fr_2fr] gap-2 px-3 py-1 text-mini font-mono items-baseline">
                  <span className="text-white/30 truncate" title={k}>{LABELS[k] || k}</span>
                  <span className="text-white/50 truncate text-right">{fmtVal(va)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
