import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Play, Square, Loader2, RefreshCw, Trash2, Undo2, Check, X, Cpu, Fingerprint, Wand2 } from 'lucide-react';
import { getJSON, postJSON } from '@engine/api';
import { useStudio } from '../engine/studio';
import { useRun } from '../engine/run';
import { Panel } from '../fx/effects';
import { notify } from '../ui/controls';

// Settings → Tuning: the engine's own measuring tools. Every panel is a thin view of a route that does
// real work; nothing here invents a number. Deliberately NOT here: UltraMax "compile"
// (/api/enhancer/ultramax/compile sleeps ~1 s and reports success without building anything, and
// nothing in the render reads its settings) - see docs/CHANGELOG.md 2026-11-28.

// Routes spelled out, never interpolated: test_default_client_capability.py reads them as text.
const BENCH_ROUTE = { apply: '/api/benchmark/apply', decline: '/api/benchmark/decline', revert: '/api/benchmark/revert' };
const TUNE_ROUTE = { start: '/api/autotune/start', cancel: '/api/autotune/cancel', revert: '/api/autotune/revert' };

const usePoll = (fn, ms, on) => {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!on) return undefined;
    const id = setInterval(() => { if (!document.hidden) ref.current(); }, ms);
    return () => clearInterval(id);
  }, [ms, on]);
};

const Stat = ({ k, v, tone }) => (
  <div className="rounded-xl border border-[var(--rule)] bg-black/30 p-2">
    <div className="font-mono text-[10px] uppercase tracking-wide text-[var(--fg-faint)]">{k}</div>
    <div className="mt-0.5 text-[15px] font-semibold tnum" style={{ color: tone || 'var(--fg)' }}>{v}</div>
  </div>
);

// ── Benchmark: a standardized clip through the user's pipeline, then a recommendation ──────────
export function BenchmarkPanel() {
  const processing = useRun((r) => r.progress.processing);
  const [prompt, setPrompt] = useState(null);
  const [faces, setFaces] = useState('1');
  const [mode, setMode] = useState('quick');
  const [prog, setProg] = useState(null);
  const [report, setReport] = useState(null);
  const [busy, setBusy] = useState('');
  const load = useCallback(async () => {
    try {
      const [p, r] = await Promise.all([getJSON('/api/benchmark/prompt'), getJSON('/api/benchmark/result')]);
      setPrompt(p); setFaces(p.default_faces || '1'); setMode(p.default_mode || 'quick');
      setReport(r?.ready ? r : null);
      if (r?.running) setProg(await getJSON('/api/benchmark/progress'));
    } catch (e) { notify(`Benchmark: ${e.message}`, 'error'); }
  }, []);
  useEffect(() => { load(); }, [load]);
  const running = !!prog?.running;
  usePoll(async () => {
    const p = await getJSON('/api/benchmark/progress').catch(() => null);
    if (!p) return;
    setProg(p);
    if (!p.running) { const r = await getJSON('/api/benchmark/result').catch(() => null); if (r?.ready) setReport(r); }
  }, 1000, running);

  const start = async () => {
    setBusy('start');
    try { await postJSON('/api/benchmark/start', { faces, mode, persist: true }); setReport(null); setProg(await getJSON('/api/benchmark/progress')); }
    catch (e) { notify(e.message, 'error'); } finally { setBusy(''); }
  };
  const act = async (verb, label) => {
    setBusy(verb);
    try {
      const r = await postJSON(BENCH_ROUTE[verb], verb === 'apply' ? { run_id: report?.run_id, recommended_settings: report?.recommended_settings } : { run_id: report?.run_id });
      notify(r?.message || label);
      if (verb === 'apply' || verb === 'revert') useStudio.getState().boot().catch(() => {});
      if (verb === 'apply') setReport((x) => (x ? { ...x, applied: true } : x));
    } catch (e) { notify(e.message, 'error'); } finally { setBusy(''); }
  };
  const changed = (report?.comparison || []).filter((r) => r.changed);
  return (
    <Panel title="Benchmark" accent={running ? 'running' : report ? `score ${report.score}` : undefined} beam={running} data-testid="benchmark">
      <p className="mb-2 text-[11.5px] text-[var(--fg-dim)]">Runs a standardized clip through your current pipeline, finds the bottleneck, and recommends settings. Nothing changes until you apply.</p>
      {prompt && !running && (
        <div className="grid grid-cols-2 gap-2">
          <label className="text-[11.5px] text-[var(--fg-dim)]">Faces per frame
            <select className="select mt-1" value={faces} onChange={(e) => setFaces(e.target.value)} data-testid="bench-faces">
              {(prompt.face_choices || []).map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
            </select>
          </label>
          <label className="text-[11.5px] text-[var(--fg-dim)]">Length
            <select className="select mt-1" value={mode} onChange={(e) => setMode(e.target.value)} data-testid="bench-mode">
              {(prompt.mode_choices || []).map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
            </select>
          </label>
          {prompt.model_summary && <div className="col-span-2 truncate font-mono text-[10.5px] text-[var(--fg-faint)]" title={prompt.model_summary}>{prompt.model_summary}</div>}
          {(prompt.warnings || []).map((w) => <div key={w} className="col-span-2 text-[11.5px] text-[var(--warn)]">{w}</div>)}
          <button type="button" className="btn btn-primary col-span-2" disabled={!!busy || processing || prompt.can_run === false} onClick={start} data-testid="bench-start"
            title={processing ? 'A render is running: a benchmark would measure a busy GPU' : undefined}>
            {busy === 'start' ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} Run benchmark
          </button>
        </div>
      )}
      {running && (
        <div data-testid="bench-progress">
          <div className="flex items-center justify-between text-[12px]"><span className="truncate text-white">{prog.status || prog.phase}</span><span className="font-mono tnum text-[var(--accent-2)]">{Math.round(prog.progress_pct || 0)}%</span></div>
          <div className="bar mt-1.5"><i style={{ width: `${prog.progress_pct || 0}%` }} /></div>
          <div className="mt-2 grid grid-cols-3 gap-2">
            <Stat k="fps" v={(prog.current_fps || 0).toFixed(1)} />
            <Stat k="frame" v={`${prog.frame || 0}/${prog.total_frames || 0}`} />
            <Stat k="VRAM" v={`${Math.round(prog.vram_pct || 0)}%`} />
          </div>
          <button type="button" className="btn btn-danger btn-sm mt-2" onClick={() => postJSON('/api/benchmark/cancel', {}).then(() => notify('Benchmark cancelled', 'info')).catch((e) => notify(e.message, 'error'))} data-testid="bench-cancel"><Square size={12} /> Cancel</button>
        </div>
      )}
      {report && !running && (
        <div className="mt-3 space-y-2" data-testid="bench-report">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat k="score" v={report.score} tone="var(--accent-2)" />
            <Stat k="avg fps" v={Number(report.average_fps || 0).toFixed(1)} />
            <Stat k="peak VRAM" v={`${Math.round((report.peak_vram_mb || 0) / 1024 * 10) / 10} GB`} />
            <Stat k="bottleneck" v={report.bottleneck || '—'} tone="var(--warn)" />
          </div>
          {report.badge && <div className="text-[12px] text-[var(--fg-dim)]"><b className="text-white">{report.badge}</b>{report.badge_detail ? ` · ${report.badge_detail}` : ''}</div>}
          {changed.length > 0 ? (
            <table className="w-full text-left font-mono text-[11px]">
              <thead className="text-[var(--fg-faint)]"><tr><th className="py-1 font-normal">setting</th><th className="font-normal">now</th><th className="font-normal">recommended</th></tr></thead>
              <tbody>{changed.map((r) => (
                <tr key={r.key || r.setting} className="border-t border-[var(--rule)]" title={r.note || undefined}>
                  <td className="py-1 text-white">{r.setting}</td><td className="text-[var(--fg-dim)]">{r.current}</td><td className="text-[var(--ok)]">{r.recommended}{r.requires_restart ? ' *' : ''}</td>
                </tr>
              ))}</tbody>
            </table>
          ) : <div className="text-[12px] text-[var(--fg-dim)]">Your current settings are already what the benchmark recommends.</div>}
          {(report.warnings || []).map((w) => <div key={w} className="text-[11.5px] text-[var(--warn)]">{w}</div>)}
          <div className="flex flex-wrap gap-2">
            <button type="button" className="btn btn-primary btn-sm" disabled={!!busy || report.applied || !changed.length} onClick={() => act('apply', 'Recommended settings applied')} data-testid="bench-apply"><Check size={13} /> {report.applied ? 'Applied' : 'Apply'}</button>
            <button type="button" className="btn btn-sm" disabled={!!busy} onClick={() => act('decline', 'Kept your settings')} data-testid="bench-decline"><X size={13} /> Keep mine</button>
            <button type="button" className="btn btn-ghost btn-sm" disabled={!!busy} onClick={() => { if (window.confirm('Undo every setting a benchmark has changed, back to the shipped defaults for those settings?')) act('revert', 'Benchmark changes reverted'); }} data-testid="bench-revert"><Undo2 size={13} /> Revert benchmark changes</button>
          </div>
        </div>
      )}
    </Panel>
  );
}

// ── Auto-tune: replays the last render's setup on a short stretch, arm by arm ────────────────────
export function AutotunePanel() {
  const [st, setSt] = useState(null);
  const [busy, setBusy] = useState('');
  const load = useCallback(async () => { try { setSt(await getJSON('/api/autotune')); } catch (e) { notify(`Auto-tune: ${e.message}`, 'error'); } }, []);
  useEffect(() => { load(); }, [load]);
  const running = !!st?.progress?.running;
  usePoll(load, 1500, running);
  const post = async (verb, ok) => {
    setBusy(verb);
    try { const r = await postJSON(TUNE_ROUTE[verb], {}); notify(r?.status === 'nothing_to_revert' ? 'Nothing to revert' : ok); await load(); if (verb === 'revert') useStudio.getState().boot().catch(() => {}); }
    catch (e) { notify(e.message, 'error'); } finally { setBusy(''); }
  };
  const res = st?.result;
  const p = st?.progress;
  return (
    <Panel title="Auto-tune" accent={running ? 'running' : undefined} beam={running} data-testid="autotune">
      <p className="mb-2 text-[11.5px] text-[var(--fg-dim)]">Replays your last render on a short stretch of it with each provider and batch size, confirms the fastest that still swaps every face, and applies it.</p>
      {!st && <div className="bar indeterminate"><i /></div>}
      {st && !running && (
        <div className="space-y-2">
          <div className="text-[12px] text-[var(--fg-dim)]">{st.ready ? <>Ready on <b className="text-white">{st.target}</b> ({st.available_frames} frames)</> : <span className="text-[var(--warn)]">{st.reason || 'Not ready'}</span>}</div>
          <div className="flex flex-wrap gap-2">
            <button type="button" className="btn btn-primary btn-sm" disabled={!st.ready || !!busy} onClick={() => post('start', 'Auto-tune started')} data-testid="autotune-start"><Wand2 size={13} /> Start auto-tune</button>
            {res?.applied && Object.keys(res.applied).length > 0 && (
              <button type="button" className="btn btn-ghost btn-sm" disabled={!!busy} onClick={() => post('revert', 'Auto-tune changes reverted')} data-testid="autotune-revert"><Undo2 size={13} /> Revert</button>
            )}
          </div>
        </div>
      )}
      {running && (
        <div data-testid="autotune-progress">
          <div className="text-[12px] text-white">{p.status || p.phase}{p.arm || p.current_arm ? ` · ${p.arm || p.current_arm}` : ''}</div>
          {p.arms_total ? <div className="bar mt-1.5"><i style={{ width: `${(p.arms_done / p.arms_total) * 100}%` }} /></div> : <div className="bar indeterminate mt-1.5"><i /></div>}
          <button type="button" className="btn btn-danger btn-sm mt-2" onClick={() => post('cancel', 'Auto-tune cancelled')} data-testid="autotune-cancel"><Square size={12} /> Cancel</button>
        </div>
      )}
      {res && !running && (
        <div className="mt-2 rounded-xl border border-[var(--rule)] p-2 text-[12px]" data-testid="autotune-result">
          <div>Last run: <b style={{ color: res.status === 'complete' ? 'var(--ok)' : 'var(--warn)' }}>{res.status}</b>{res.target ? ` on ${res.target}` : ''}{res.finished ? ` · ${String(res.finished).replace('T', ' ')}` : ''}</div>
          {res.winner && <div className="mt-0.5 text-[var(--fg-dim)]">Fastest: <b className="text-white">{res.winner.arm}</b> at {res.winner.fps} fps{res.winner.changed ? ' (applied)' : ' (already your setting)'}</div>}
          {res.error && <div className="mt-0.5 text-[var(--danger)]">{res.error}</div>}
        </div>
      )}
    </Panel>
  );
}

// ── Recognition backbone for the identity embedding API ───────────────────────────────────────
export function RecognitionPanel() {
  const [cur, setCur] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState('default');
  const [provider, setProvider] = useState('app');
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    try {
      const [m, c] = await Promise.all([getJSON('/api/recognition/models'), getJSON('/api/recognition/current')]);
      setModels(m.models || []); setCur(c); setModel(c.selection?.model || 'default'); setProvider(c.selection?.provider || 'app');
    } catch (e) { notify(`Recognition: ${e.message}`, 'error'); }
  }, []);
  useEffect(() => { load(); }, [load]);
  if (!cur) return <Panel title="Identity embedding" data-testid="recognition"><div className="bar indeterminate"><i /></div></Panel>;
  const adv = cur.advice || {};
  const providers = cur.providers || [];
  const spec = models.find((m) => m.key === model);
  const chosen = providers.find((p) => p.value === provider);
  const dirty = model !== cur.selection?.model || provider !== cur.selection?.provider;
  const active = cur.active;
  const apply = async () => {
    setBusy(true);
    try {
      // May download the model and build a TensorRT engine: minutes, not seconds.
      const r = await postJSON('/api/recognition/set', { model_name: model, provider }, { timeout: 15 * 60 * 1000 });
      if (r?.after) setCur(r.after);
      setModels((await getJSON('/api/recognition/models')).models || []);
      notify(r?.message || 'Applied', r?.degraded ? 'warn' : 'success');
    } catch (e) { notify(e.message, 'error'); } finally { setBusy(false); }
  };
  return (
    <Panel title="Identity embedding" data-testid="recognition">
      <p className="mb-2 text-[11.5px] text-[var(--fg-dim)]">The backbone for the identity embedding API (Face Bank pooling, tracking helpers). Who is who in renders is the Recognizer setting.</p>
      <div className="rounded-xl border border-[var(--rule)] bg-black/25 p-2 text-[12px]">
        <div className="flex flex-wrap items-center gap-2"><span className="badge">{adv.label}</span><span className="text-[var(--fg-dim)]">{adv.device_name || 'No GPU detected'}</span></div>
        {adv.strategy && <div className="mt-1 text-[var(--fg-dim)]">Suggested: <b className="text-white">{providers.find((p) => p.value === adv.provider)?.label || adv.provider}</b> · {adv.strategy}</div>}
        {adv.provider && provider !== adv.provider && <button type="button" className="btn btn-ghost btn-sm mt-1" disabled={busy} onClick={() => setProvider(adv.provider)}>Use the suggested provider</button>}
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <label className="text-[11.5px] text-[var(--fg-dim)]">Model
          <select className="select mt-1" value={model} disabled={busy} onChange={(e) => setModel(e.target.value)} data-testid="recog-model">
            {models.map((m) => <option key={m.key} value={m.key}>{m.display_name}{m.downloaded ? '' : ' (downloads)'}</option>)}
          </select>
        </label>
        <label className="text-[11.5px] text-[var(--fg-dim)]">Acceleration
          <select className="select mt-1" value={provider} disabled={busy} onChange={(e) => setProvider(e.target.value)} data-testid="recog-provider">
            {providers.map((p) => <option key={p.value} value={p.value}>{p.available ? p.label : `${p.label} (unavailable)`}</option>)}
          </select>
        </label>
      </div>
      {spec?.description && <div className="mt-1 text-[11.5px] text-[var(--fg-faint)]">{spec.description}{spec.dim ? ` · ${spec.dim}-d` : ''}</div>}
      {chosen && !chosen.available && <div className="mt-1 text-[11.5px] text-[var(--warn)]">{chosen.reason || 'Unavailable on this machine'}: the engine would fall back.</div>}
      <div className="mt-2 flex items-center gap-2">
        <button type="button" className="btn btn-primary btn-sm" disabled={busy || !dirty} onClick={apply} data-testid="recog-apply">{busy ? <Loader2 size={13} className="animate-spin" /> : <Fingerprint size={13} />} {busy ? 'Loading (may download or build)…' : 'Apply'}</button>
        <span className="text-[11.5px] text-[var(--fg-faint)]">{active ? `Loaded: ${active.model || active.model_name} on ${(active.active_providers?.[0] || 'CPU').replace('ExecutionProvider', '')}${active.degraded ? ' (CPU fallback)' : ''}` : 'Nothing loaded yet'}</span>
      </div>
    </Panel>
  );
}

// ── TensorRT engine cache ───────────────────────────────────────────────────────────────────────
const TRT_STATUS = {
  ready: ['Engines built', 'var(--ok)'], not_built: ['Not built yet: the next render builds them (2-18 min cold)', 'var(--warn)'],
  not_loaded: ['TensorRT selected, no session loaded yet', 'var(--fg-dim)'], not_tensorrt: ['Not running on TensorRT', 'var(--fg-faint)'],
};
const KIND_TONE = { active: 'var(--ok)', stale: 'var(--warn)', unknown: 'var(--fg-faint)', other: 'var(--fg-faint)' };

export function TrtCachePanel() {
  const [d, setD] = useState(null);
  const [busy, setBusy] = useState('');
  const load = useCallback(async () => { try { setD(await getJSON('/api/trt_cache', { timeout: 20000 })); } catch (e) { notify(`TensorRT cache: ${e.message}`, 'error'); } }, []);
  useEffect(() => { load(); }, [load]);
  const clear = async (scope) => {
    const msg = scope === 'all'
      ? 'Clear ALL TensorRT engines?\n\nEvery engine is rebuilt on the next render: 2-18 minutes before the first frame, per model set.'
      : `Clear ${d?.stale_mb ?? 0} MB of stale engines (built for settings or versions this app no longer uses)?\n\nThe active namespace is kept.`;
    if (!window.confirm(msg)) return;
    setBusy(scope);
    try {
      const r = await postJSON('/api/trt_cache/clear', { scope }, { timeout: 0 });
      notify(`Freed ${r.freed_mb} MB${r.failed?.length ? `, ${r.failed.length} in use (retry after a restart)` : ''}`);
      if (r.after) setD(r.after); else load();
    } catch (e) { notify(e.message, 'error'); } finally { setBusy(''); }
  };
  if (!d) return <Panel title="TensorRT engines" data-testid="trt-cache"><div className="bar indeterminate"><i /></div></Panel>;
  const [label, tone] = TRT_STATUS[d.status] || [d.status, 'var(--fg-dim)'];
  return (
    <Panel title="TensorRT engines" accent={`${d.total_mb} MB`} data-testid="trt-cache"
      actions={<button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Refresh" onClick={load} disabled={!!busy}><RefreshCw size={13} /></button>}>
      <div className="flex flex-wrap items-center gap-2 text-[12px]"><Cpu size={13} style={{ color: tone }} /><span style={{ color: tone }} data-testid="trt-status">{label}</span>{d.active_engines > 0 && <span className="text-[var(--fg-faint)]">· {d.active_engines} active engine(s)</span>}</div>
      <div className="mt-2 max-h-48 overflow-y-auto">
        <table className="w-full text-left font-mono text-[11px]">
          <tbody>
            {(d.namespaces || []).filter((r) => r.size_mb > 0).map((r) => (
              <tr key={r.name} className="border-t border-[var(--rule)]" data-testid="trt-ns">
                <td className="py-1 pr-2" style={{ color: KIND_TONE[r.kind] }}>{r.kind}</td>
                <td className="max-w-[16rem] truncate pr-2 text-[var(--fg-dim)]" title={r.name}>{r.name}</td>
                <td className="pr-2 text-right text-[var(--fg-dim)]">{r.engines}</td>
                <td className="text-right text-[var(--fg-dim)]">{r.size_mb} MB</td>
              </tr>
            ))}
            {d.loose_mb > 0 && <tr className="border-t border-[var(--rule)]"><td className="py-1 text-[var(--fg-faint)]">other</td><td className="text-[var(--fg-faint)]" colSpan={2}>loose files ({d.loose_engines})</td><td className="text-right text-[var(--fg-dim)]">{d.loose_mb} MB</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <button type="button" className="btn btn-sm" disabled={!!busy || !d.active_namespace || !(d.stale_mb > 0)} onClick={() => clear('stale')} data-testid="trt-clear-stale">
          {busy === 'stale' ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />} Clear stale ({d.stale_mb} MB)
        </button>
        <button type="button" className="btn btn-danger btn-sm" disabled={!!busy} onClick={() => clear('all')} data-testid="trt-clear-all">
          {busy === 'all' ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />} Clear all
        </button>
      </div>
      {!d.active_namespace && d.status !== 'not_tensorrt' && <div className="mt-1 text-[11px] text-[var(--fg-faint)]">"Clear stale" unlocks after a TensorRT preview or render, once the engine knows which namespace is live.</div>}
    </Panel>
  );
}

export default function TuningPanels() {
  return (
    <div className="grid gap-3 lg:grid-cols-2" data-testid="tuning">
      <BenchmarkPanel />
      <AutotunePanel />
      <TrtCachePanel />
      <RecognitionPanel />
    </div>
  );
}

