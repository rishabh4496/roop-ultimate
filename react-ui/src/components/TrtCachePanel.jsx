import React, { useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../api';
import { Button } from './ui';
import { confirmDialog } from './confirm';

// TensorRT engine cache: what the running backend builds into, what is stale,
// and clearing it. Every number comes from /api/trt_cache (routes_trt_cache.py).

const STATUS = {
  ready: ['Engines built', 'text-emerald-300 border-emerald-500/40 bg-emerald-500/15'],
  not_built: ['Not built yet: the next render builds them (2-18 min cold)', 'text-amber-300 border-amber-500/40 bg-amber-500/15'],
  not_loaded: ['TensorRT is selected but no session has loaded yet', 'text-white/70 border-white/20 bg-white/5'],
  not_tensorrt: ['Not running on TensorRT', 'text-white/50 border-white/10 bg-white/5'],
};

const KIND = {
  active: 'text-emerald-300',
  stale: 'text-amber-300',
  unknown: 'text-white/50',
  other: 'text-white/40',
};

export default function TrtCachePanel({ notify }) {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState('');

  const load = useCallback(async () => {
    try {
      setData(await getJSON('/api/trt_cache', { timeout: 20000 }));
    } catch (e) {
      notify?.(`TensorRT cache: ${e.message || e}`, 'error');
    }
  }, [notify]);

  useEffect(() => { load(); }, [load]);

  const clear = async (scope) => {
    const ok = await confirmDialog({
      title: scope === 'all' ? 'Clear ALL TensorRT engines?' : 'Clear stale engines?',
      message: scope === 'all'
        ? 'Every engine is rebuilt on the next render: 2-18 minutes before the first frame, per model set. Engines in use stay locked until a restart.'
        : `Removes ${data?.stale_mb ?? 0} MB of engines built for settings or versions this app no longer uses. The active namespace is kept.`,
      confirmLabel: 'Clear',
      danger: scope === 'all',
    });
    if (!ok) return;
    setBusy(scope);
    try {
      const res = await postJSON('/api/trt_cache/clear', { scope });
      notify?.(`Freed ${res.freed_mb} MB${res.failed?.length ? `, ${res.failed.length} in use (retry after a restart)` : ''}`, 'success');
      setData(res.after);
    } catch (e) {
      notify?.(e.message || String(e), 'error');
    } finally {
      setBusy('');
    }
  };

  if (!data) return <p className="text-xs text-white/40">Reading the engine cache…</p>;
  const [statusText, statusCls] = STATUS[data.status] || [data.status, STATUS.not_tensorrt[1]];

  return (
    <div className="flex flex-col gap-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className={`px-2 py-0.5 rounded-md border text-nano font-semibold ${statusCls}`}>{statusText}</span>
        {data.active_engines > 0 && <span className="text-white/60">{data.active_engines} engine(s) in the active namespace</span>}
        <span className="text-white/40">{data.total_mb} MB on disk</span>
      </div>
      <div className="max-h-40 overflow-auto rounded-md border border-white/10">
        <table className="w-full text-nano">
          <tbody>
            {data.namespaces.filter((r) => r.size_mb > 0).map((r) => (
              <tr key={r.name} className="border-b border-white/5">
                <td className={`px-2 py-1 font-semibold ${KIND[r.kind] || ''}`}>{r.kind}</td>
                <td className="px-2 py-1 text-white/60 truncate max-w-[18rem]" title={r.name}>{r.name}</td>
                <td className="px-2 py-1 text-right text-white/60">{r.engines}</td>
                <td className="px-2 py-1 text-right text-white/60">{r.size_mb} MB</td>
                <td className="px-2 py-1 text-white/40">{r.modified}</td>
              </tr>
            ))}
            {data.loose_mb > 0 && (
              <tr><td className="px-2 py-1 text-white/40">other</td><td className="px-2 py-1 text-white/50" colSpan={2}>loose files at the cache root ({data.loose_engines} engine(s))</td><td className="px-2 py-1 text-right text-white/60">{data.loose_mb} MB</td><td /></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="secondary" onClick={load} disabled={!!busy}>Refresh</Button>
        <Button size="sm" variant="secondary" onClick={() => clear('stale')}
          disabled={!!busy || !data.active_namespace || !(data.stale_mb > 0)}>
          {busy === 'stale' ? 'Clearing…' : `Clear stale (${data.stale_mb} MB)`}
        </Button>
        <Button size="sm" variant="stop" onClick={() => clear('all')} disabled={!!busy}>
          {busy === 'all' ? 'Clearing…' : 'Clear all'}
        </Button>
      </div>
      {!data.active_namespace && data.status !== 'not_tensorrt' && (
        <p className="text-nano text-white/40">"Clear stale" unlocks after a TensorRT preview or render, once the backend knows which namespace is live.</p>
      )}
    </div>
  );
}
