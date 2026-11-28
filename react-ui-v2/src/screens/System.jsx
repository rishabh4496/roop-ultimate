import React, { useCallback, useEffect, useState } from 'react';
import { RefreshCw, Trash2, Loader2, Cpu, CloudDownload, ShieldAlert } from 'lucide-react';
import { getJSON, postJSON } from '@engine/api';
import { Panel } from '../fx/effects';
import { notify } from '../ui/controls';

// Settings' Storage and System tabs: what the engine keeps on disk (and what is safe to remove),
// the machine it runs on, and whether an update is available. All read-only except one explicit,
// confirmed delete of one item the engine itself classified SAFE_TO_DELETE.

export const formatBytes = (n) => {
  const v = Number(n) || 0;
  if (v < 1024) return `${v} B`;
  const u = ['KB', 'MB', 'GB', 'TB'];
  let x = v / 1024; let i = 0;
  while (x >= 1024 && i < u.length - 1) { x /= 1024; i += 1; }
  return `${x.toFixed(x >= 100 ? 0 : 1)} ${u[i]}`;
};

const CLASS_TONE = { SAFE_TO_DELETE: 'var(--ok)', REVIEW_BEFORE_DELETE: 'var(--warn)', PROTECTED: 'var(--accent-2)' };
const CLASS_LABEL = { SAFE_TO_DELETE: 'safe to delete', REVIEW_BEFORE_DELETE: 'review first', PROTECTED: 'protected' };

export function StoragePanel() {
  const [review, setReview] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState('');
  const [filter, setFilter] = useState('all');
  const load = useCallback(async () => {
    setLoading(true); setError('');
    // The review walks the model, cache, output and project trees under one lock. Measured 11-28 on
    // the 4070 install (~58 GB of models + caches, 739 logs): minutes, not seconds, and requests queue
    // behind each other. So no client timeout (1.0's 45 s fails there); the bar says it is working.
    try { setReview(await getJSON('/api/storage', { timeout: 0 })); } catch (e) { setError(e.message); } finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);

  const items = Array.isArray(review?.items) ? review.items : [];
  const safe = items.filter((i) => i.classification === 'SAFE_TO_DELETE');
  const shown = filter === 'all' ? items : items.filter((i) => i.classification === filter);
  const remove = async (item) => {
    if (!window.confirm(`Delete ${item.relative_path} (${formatBytes(item.size_bytes)})?\n\nThe engine can regenerate it, but this cannot be undone.`)) return;
    setBusy(item.id);
    try {
      await postJSON('/api/storage/delete', { item_id: item.id, confirm: true }, { timeout: 0 });
      notify(`Deleted ${item.relative_path}`);
      await load();
    } catch (e) { notify(e.message, 'error'); } finally { setBusy(''); }
  };

  return (
    <Panel title="Storage" accent={review ? `${items.length} items` : undefined} data-testid="storage"
      actions={<button type="button" className="btn btn-ghost btn-sm" onClick={load} disabled={loading} data-testid="storage-refresh">
        {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />} Rescan</button>}>
      {error && <div className="rounded-xl border border-[var(--rule-strong)] p-2.5 text-[12.5px] text-[var(--danger)]" role="alert">{error}</div>}
      {loading && (
        <div className="mb-3">
          <div className="bar indeterminate"><i /></div>
          <div className="mt-1.5 text-[11.5px] text-[var(--fg-faint)]" data-testid="storage-scanning">Scanning models, caches, outputs and logs. On a large install this takes a few minutes.</div>
        </div>
      )}
      {review && (
        <>
          {review.active_work && (
            <div className="mb-3 flex items-start gap-2 rounded-xl border border-[color-mix(in_oklab,var(--warn)_35%,transparent)] p-2.5 text-[12px] text-[var(--warn)]">
              <ShieldAlert size={14} className="mt-0.5 shrink-0" /> Work is in progress ({(review.active_reasons || []).join(', ') || 'a render'}), so items it uses are protected.
            </div>
          )}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {[
              ['Safe to delete', `${safe.length}`, 'var(--ok)'],
              ['Reclaimable', formatBytes(safe.reduce((n, i) => n + (Number(i.size_bytes) || 0), 0)), 'var(--ok)'],
              ['Review first', `${items.filter((i) => i.classification === 'REVIEW_BEFORE_DELETE').length}`, 'var(--warn)'],
              ['Protected', `${items.filter((i) => i.classification === 'PROTECTED').length}`, 'var(--accent-2)'],
            ].map(([k, v, tone]) => (
              <div key={k} className="rounded-xl border border-[var(--rule)] bg-black/30 p-2.5">
                <div className="font-mono text-[10px] uppercase tracking-wide text-[var(--fg-faint)]">{k}</div>
                <div className="mt-0.5 text-[15px] font-semibold tnum" style={{ color: tone }}>{v}</div>
              </div>
            ))}
          </div>
          {Array.isArray(review.categories) && review.categories.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5" data-testid="storage-categories">
              {review.categories.map((c) => (
                <span key={c.name} className="chip !h-7" title={c.evidence || undefined}>{c.name} · {formatBytes(c.size_bytes)}</span>
              ))}
            </div>
          )}
          <div className="mt-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Filter storage items">
            {[['all', 'All'], ['SAFE_TO_DELETE', 'Safe'], ['REVIEW_BEFORE_DELETE', 'Review'], ['PROTECTED', 'Protected']].map(([v, l]) => (
              <button key={v} type="button" role="tab" aria-selected={filter === v} className={`chip !h-7 ${filter === v ? 'on' : ''}`} onClick={() => setFilter(v)}>{l}</button>
            ))}
          </div>
          {/* Content-visibility keeps an 800-item review (739 logs on the 4070 install) cheap to scroll. */}
          <div className="mt-2 max-h-[52vh] space-y-2 overflow-y-auto pr-1 [&>article]:[content-visibility:auto] [&>article]:[contain-intrinsic-size:auto_92px]">
            {shown.map((item) => (
              <article key={item.id} className="rounded-xl border border-[var(--rule)] bg-black/25 p-2.5" data-testid="storage-item">
                <div className="flex items-start gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="selectable break-all font-mono text-[11.5px] text-white">{item.relative_path}</div>
                    <div className="mt-0.5 flex flex-wrap gap-x-3 font-mono text-[10.5px] text-[var(--fg-faint)]">
                      <span>{item.category}</span><span>{formatBytes(item.size_bytes)}</span><span>{item.is_directory ? 'folder' : 'file'}</span>
                      <span>regenerable: {item.regenerable ? 'yes' : 'no'}</span>
                      {item.referenced && <span style={{ color: 'var(--warn)' }}>referenced ×{item.reference_count}</span>}
                    </div>
                  </div>
                  <span className="badge shrink-0" style={{ color: CLASS_TONE[item.classification] }}>{CLASS_LABEL[item.classification] || item.classification}</span>
                </div>
                {item.reason && <p className="mt-1.5 text-[12px] text-[var(--fg-dim)]">{item.reason}</p>}
                {item.classification === 'SAFE_TO_DELETE' && (
                  <button type="button" className="btn btn-danger btn-sm mt-2" disabled={!!busy} onClick={() => remove(item)} data-testid="storage-delete">
                    {busy === item.id ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />} Delete this item
                  </button>
                )}
              </article>
            ))}
            {!shown.length && <div className="py-6 text-center text-[12px] text-[var(--fg-faint)]">No items here. Files the engine does not know about are outside this review on purpose.</div>}
          </div>
          {review.deletion_policy && <p className="mt-2 text-[11px] text-[var(--fg-faint)]">{review.deletion_policy}</p>}
        </>
      )}
    </Panel>
  );
}

const Row = ({ k, v }) => (
  <div className="flex items-baseline justify-between gap-3 border-b border-[var(--rule)] py-1.5 last:border-0">
    <span className="text-[12px] text-[var(--fg-dim)]">{k}</span>
    <span className="selectable truncate text-right font-mono text-[11.5px] text-white" title={String(v ?? '')}>{v ?? '—'}</span>
  </div>
);
const gb = (v) => (Number.isFinite(Number(v)) ? `${Number(v).toFixed(1)} GB` : null);

export function HardwarePanel() {
  const [hw, setHw] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { getJSON('/api/system/hardware', { timeout: 30000 }).then(setHw).catch((e) => setError(e.message)); }, []);
  // The engine and the mock name a few fields differently (gpu_name / gpu, *_available_gb / *_free_gb).
  const rows = hw ? [
    ['GPU', hw.gpu_name || hw.gpu],
    ['Architecture', [hw.architecture, hw.compute_capability && `sm ${hw.compute_capability}`].filter(Boolean).join(' · ') || null],
    ['VRAM', hw.vram_total_gb != null ? `${gb(hw.vram_available_gb ?? hw.vram_free_gb) || '?'} free of ${gb(hw.vram_total_gb)}` : null],
    ['Driver / CUDA', [hw.driver_version, hw.cuda_version && `CUDA ${hw.cuda_version}`].filter(Boolean).join(' · ') || null],
    ['TensorRT', hw.tensorrt_version || (hw.tensorrt_available === false ? 'not available' : null)],
    ['ONNX Runtime', hw.onnxruntime_version],
    ['Hardware encode', Array.isArray(hw.nvenc_codecs) ? hw.nvenc_codecs.join(', ') || 'none' : null],
    ['Precision', Array.isArray(hw.tensor_core_capabilities) ? hw.tensor_core_capabilities.join(', ') : null],
    ['CPU', hw.cpu_name],
    ['Cores', (hw.cpu_physical_cores ?? hw.cpu_cores) != null ? `${hw.cpu_physical_cores ?? hw.cpu_cores} cores · ${hw.cpu_logical_cores ?? hw.cpu_threads} threads${hw.cpu_performance_cores ? ` (${hw.cpu_performance_cores}P + ${hw.cpu_efficiency_cores}E)` : ''}` : null],
    ['RAM', hw.ram_total_gb != null ? `${gb(hw.ram_available_gb ?? hw.ram_free_gb) || '?'} free of ${gb(hw.ram_total_gb)}` : null],
    ['Tier', hw.vram_tier || hw.active_profile],
    ['Platform', hw.platform],
  ] : [];
  return (
    <Panel title="This machine" data-testid="hardware">
      {error && <div className="text-[12.5px] text-[var(--danger)]" role="alert">{error}</div>}
      {!hw && !error && <div className="bar indeterminate"><i /></div>}
      {hw && <div className="flex items-center gap-2 pb-1 text-[var(--accent-2)]"><Cpu size={15} /><span className="text-[13px] font-semibold text-white">{hw.gpu_name || hw.gpu || 'GPU'}</span></div>}
      {rows.filter(([, v]) => v != null && v !== '').map(([k, v]) => <Row key={k} k={k} v={v} />)}
    </Panel>
  );
}

// update_manager's classes: SAFE, REQUIRES REVIEW, UNVERIFIED, INCOMPATIBLE.
const UPDATE_TONE = { SAFE: 'var(--ok)', 'REQUIRES REVIEW': 'var(--warn)', UNVERIFIED: 'var(--warn)', INCOMPATIBLE: 'var(--danger)' };

export function UpdatePanel() {
  const [u, setU] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const check = useCallback(async (refresh = false) => {
    setLoading(true); setError('');
    try { setU(await getJSON(`/api/update/check${refresh ? '?refresh=true' : ''}`, { timeout: 60000 })); } catch (e) { setError(e.message); } finally { setLoading(false); }
  }, []);
  useEffect(() => { check(false); }, [check]);
  const short = (sha) => (sha ? String(sha).slice(0, 7) : '—');
  return (
    <Panel title="Updates" data-testid="updates"
      actions={<button type="button" className="btn btn-ghost btn-sm" onClick={() => check(true)} disabled={loading} data-testid="update-check">
        {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />} Check now</button>}>
      {error && <div className="text-[12.5px] text-[var(--danger)]" role="alert">{error}</div>}
      {!u && !error && <div className="bar indeterminate"><i /></div>}
      {u && (
        <>
          <div className="flex items-center gap-2">
            <CloudDownload size={16} style={{ color: u.available ? 'var(--accent-2)' : 'var(--fg-faint)' }} />
            <span className="text-[13.5px] font-semibold text-white" data-testid="update-status">{u.available ? 'An update is available' : 'Up to date'}</span>
            <span className="badge ml-auto" style={{ color: UPDATE_TONE[u.classification] || 'var(--fg-dim)' }}>{u.classification}</span>
          </div>
          <Row k="Installed" v={`${short(u.current?.sha)}${u.current?.branch ? ` · ${u.current.branch}` : ''}`} />
          {u.available && <Row k="Available" v={`${short(u.candidate_sha)}${u.candidate_date ? ` · ${String(u.candidate_date).slice(0, 10)}` : ''}`} />}
          {u.available && u.candidate_manifest && !u.candidate_manifest.present && (
            <p className="mt-2 text-[12px] text-[var(--warn)]">The newer commit carries no compatibility manifest: nothing has checked it on any machine yet.</p>
          )}
          {(u.reasons || []).length > 0 && (
            <ul className="mt-2 list-disc space-y-0.5 pl-4 text-[12px] text-[var(--fg-dim)]">{u.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
          )}
          <p className="mt-2 text-[11px] text-[var(--fg-faint)]">Updates install from Pinokio's Update button; the browser cannot start one.{u.cached ? ' (cached result)' : ''}</p>
        </>
      )}
    </Panel>
  );
}

export function SystemPanels() {
  return (
    <div className="grid gap-3 lg:grid-cols-2" data-testid="system">
      <HardwarePanel />
      <UpdatePanel />
    </div>
  );
}

