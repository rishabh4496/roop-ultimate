import React, { useEffect, useMemo, useState } from 'react';
import { History as HistoryIcon, RotateCcw, Trash2, FolderOpen, GitCompare, Film, Timer, Gauge, Clock } from 'lucide-react';
import { getJSON, postJSON } from '@engine/api';
import { useStudio } from '../engine/studio';
import { Panel, SplitText, Meteors, NumberTicker } from '../fx/effects';
import { notify } from '../ui/controls';

// ── History: every finished run, its settings, and what changed between two ──
// /api/history (run_history.json). "Apply settings" merges a run's settings into the current ones
// (saved like any other change). Pick two runs to see exactly which settings differ.

const dur = (s) => (s == null ? '—' : s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`);
const when = (t) => {
  if (!t) return '';
  const d = typeof t === 'number' ? new Date(t * (t < 1e12 ? 1000 : 1)) : new Date(t);
  return Number.isNaN(d.getTime()) ? String(t) : d.toLocaleString();
};
const KEY_SETTINGS = ['swap_model', 'selected_enhancer', 'mask_engine', 'detector_engine', 'face_detection_mode', 'subsample_upscale'];
const show = (v) => (v === undefined ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v));

export default function History() {
  const [entries, setEntries] = useState(null);
  const [outputPath, setOutputPath] = useState('');
  const [presets, setPresets] = useState([]);
  const [compare, setCompare] = useState([]);
  const load = async () => {
    try { setEntries((await getJSON('/api/history')).entries || []); } catch (e) { notify(e.message, 'error'); setEntries([]); }
    try { setOutputPath((await getJSON('/api/output')).output_path || ''); } catch { /* reveal just stays off */ }
  };
  useEffect(() => {
    load();
    getJSON('/api/export/presets').then((r) => setPresets(r.presets || [])).catch(() => setPresets([]));
  }, []);
  const list = (entries || []).slice().reverse();
  const pair = compare.map((id) => list.find((e) => e.id === id)).filter(Boolean);
  const diff = useMemo(() => {
    if (pair.length !== 2) return [];
    const [a, b] = pair.map((e) => e.settings || {});
    return Array.from(new Set([...Object.keys(a), ...Object.keys(b)]))
      .filter((k) => JSON.stringify(a[k]) !== JSON.stringify(b[k])).sort();
  }, [compare, entries]); // eslint-disable-line react-hooks/exhaustive-deps

  const apply = (e) => { useStudio.getState().setSettings(e.settings || {}); notify('Settings from that run applied'); };
  const remove = async (e) => {
    try { await postJSON('/api/history/delete', { id: e.id }); setEntries((x) => x.filter((y) => y.id !== e.id)); setCompare((c) => c.filter((id) => id !== e.id)); }
    catch (err) { notify(err.message, 'error'); }
  };
  const reveal = async (name) => { try { await postJSON('/api/reveal', { path: `${outputPath}/${name}` }); } catch (err) { notify(err.message, 'error'); } };
  const exportAs = async (name, preset) => {
    try { const r = await postJSON('/api/export/apply', { source: name, preset }, { timeout: 0 }); notify(`Exported ${r.name}`); }
    catch (err) { notify(err.message, 'error'); }
  };
  const toggleCompare = (id) => setCompare((c) => (c.includes(id) ? c.filter((x) => x !== id) : [...c, id].slice(-2)));
  const totals = list.reduce((a, e) => ({ frames: a.frames + (e.frames || 0), time: a.time + (e.duration_s || 0) }), { frames: 0, time: 0 });

  return (
    <div className="mx-auto max-w-[1400px] space-y-3" data-testid="history">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight"><SplitText text="History" className="gradient-text" /></h1>
          <p className="text-[13px] text-[var(--fg-dim)]">{list.length} run{list.length === 1 ? '' : 's'} · <NumberTicker value={totals.frames} /> frames · {dur(totals.time)} rendering. Tick two runs to compare their settings.</p>
        </div>
      </div>

      {pair.length === 2 && (
        <Panel title="Compare" accent={`${diff.length} setting${diff.length === 1 ? '' : 's'} differ`} beam data-testid="history-compare"
          actions={<button type="button" className="btn btn-ghost btn-sm" onClick={() => setCompare([])}>Clear</button>}>
          {diff.length === 0 ? <div className="text-[12.5px] text-[var(--fg-dim)]">Identical settings.</div> : (
            <div className="max-h-[300px] overflow-y-auto">
              <table className="w-full text-left font-mono text-[11.5px]">
                <thead><tr className="text-[var(--fg-faint)]"><th className="py-1 pr-3 font-medium">setting</th><th className="py-1 pr-3 font-medium">{when(pair[0].time)}</th><th className="py-1 font-medium">{when(pair[1].time)}</th></tr></thead>
                <tbody>{diff.map((k) => (
                  <tr key={k} className="border-t border-[var(--rule)]">
                    <td className="py-1 pr-3 text-[var(--fg-dim)]">{k}</td>
                    <td className="max-w-[260px] truncate py-1 pr-3 text-[var(--accent)]" title={show(pair[0].settings?.[k])}>{show(pair[0].settings?.[k])}</td>
                    <td className="max-w-[260px] truncate py-1 text-[var(--accent-2)]" title={show(pair[1].settings?.[k])}>{show(pair[1].settings?.[k])}</td>
                  </tr>))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      )}

      {entries === null ? (
        <div className="space-y-2">{Array.from({ length: 4 }, (_, i) => <div key={i} className="tile h-24 animate-pulse" />)}</div>
      ) : list.length === 0 ? (
        <Panel bodyClass="relative flex min-h-[260px] flex-col items-center justify-center gap-2 overflow-hidden text-center">
          <Meteors count={10} />
          <HistoryIcon size={30} strokeWidth={1.5} className="relative text-[var(--accent-2)]" />
          <div className="relative text-[15px] font-semibold">No runs yet</div>
          <div className="relative text-[12.5px] text-[var(--fg-dim)]">Every finished render is recorded here with its settings.</div>
        </Panel>
      ) : (
        <ol className="space-y-2" data-testid="history-list">
          {list.map((e) => {
            const picked = compare.includes(e.id);
            return (
              <li key={e.id} className={`tile p-3 ${picked ? 'selected' : ''}`}>
                <div className="flex flex-wrap items-start gap-3">
                  <label className="mt-0.5 flex cursor-pointer items-center gap-2" title="Compare">
                    <input type="checkbox" className="h-4 w-4 accent-[var(--accent)]" checked={picked} onChange={() => toggleCompare(e.id)} aria-label="Compare this run" />
                  </label>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="truncate text-[13.5px] font-semibold text-white">{e.target || (e.outputs || [])[0] || `run ${e.id}`}</span>
                      <span className="font-mono text-[11px] text-[var(--fg-faint)]"><Clock size={11} className="mr-1 inline" />{when(e.time)}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      <span className="chip"><Film size={11} /> <b>{e.frames ?? '—'}</b> frames</span>
                      <span className="chip"><Gauge size={11} /> <b>{e.fps ? Number(e.fps).toFixed(1) : '—'}</b> fps</span>
                      <span className="chip"><Timer size={11} /> {dur(e.duration_s)}</span>
                      {KEY_SETTINGS.filter((k) => e.settings?.[k] != null).map((k) => <span key={k} className="chip" title={k}>{String(e.settings[k])}</span>)}
                    </div>
                    {(e.outputs || []).length > 0 && (
                      <div className="mt-2 flex flex-wrap items-center gap-1.5">
                        {e.outputs.map((name) => (
                          <span key={name} className="flex items-center gap-1 rounded-lg border border-[var(--rule)] bg-black/40 py-0.5 pl-2 pr-0.5 font-mono text-[11px]">
                            <a className="max-w-[260px] truncate text-[var(--accent-2)] hover:underline" href={`/outputs/${encodeURIComponent(name)}`} target="_blank" rel="noreferrer">{name}</a>
                            {outputPath && <button type="button" className="btn btn-ghost btn-sm btn-icon !h-6 !w-6" aria-label={`Reveal ${name}`} title="Show in folder" onClick={() => reveal(name)}><FolderOpen size={12} /></button>}
                            {presets.length > 0 && (
                              <select className="select !h-6 !py-0 !pl-1.5 !pr-6 text-[11px]" style={{ width: 'auto' }} value="" aria-label={`Export ${name}`} onChange={(ev) => { if (ev.target.value) exportAs(name, ev.target.value); }}>
                                <option value="">export…</option>
                                {presets.map((p) => <option key={p.key || p.id || p.name} value={p.key || p.id || p.name}>{p.label || p.name || p.key}</option>)}
                              </select>
                            )}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    {e.settings && <button type="button" className="btn btn-sm" onClick={() => apply(e)} title="Use these settings again"><RotateCcw size={13} /> Apply settings</button>}
                    <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Delete from history" title="Delete from history" onClick={() => remove(e)}><Trash2 size={14} /></button>
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      )}
      {compare.length === 1 && <div className="text-center text-[12px] text-[var(--fg-faint)]"><GitCompare size={12} className="mr-1 inline" />Tick one more run to compare.</div>}
    </div>
  );
}
