import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { getJSON, postJSON } from '../api';
import { confirmDialog } from './confirm';
import { Icon } from '../icons';

const CLASS_LABELS = {
  SAFE_TO_DELETE: 'Safe to delete',
  REVIEW_BEFORE_DELETE: 'Review before delete',
  PROTECTED: 'Protected',
  UNKNOWN: 'Unknown',
};

const CLASS_STYLES = {
  SAFE_TO_DELETE: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300',
  REVIEW_BEFORE_DELETE: 'bg-amber-500/15 border-amber-500/30 text-amber-300',
  PROTECTED: 'bg-blue-500/15 border-blue-500/30 text-blue-300',
  UNKNOWN: 'bg-white/10 border-white/15 text-white/60',
};

function formatBytes(value) {
  let size = Number(value) || 0;
  for (const unit of ['B', 'KB', 'MB', 'GB', 'TB']) {
    if (size < 1024 || unit === 'TB') return `${size >= 10 || unit === 'B' ? size.toFixed(0) : size.toFixed(1)} ${unit}`;
    size /= 1024;
  }
  return '0 B';
}

function Classification({ value }) {
  return <span className={`inline-flex rounded-full border px-2 py-0.5 text-nano font-bold ${CLASS_STYLES[value] || CLASS_STYLES.UNKNOWN}`}>
    {CLASS_LABELS[value] || value || 'Unknown'}
  </span>;
}

export default function StorageManager({ notify }) {
  const [review, setReview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      // /api/storage walks the model, cache, output and project trees. Measured
      // at 6.4s warm on this machine and longer on the first call after a boot,
      // when the page-cache is cold and the backend is still loading models —
      // so a 10s deadline reported "Request timed out" on exactly the visit
      // where the panel is first opened. The scan is bounded work, not a poll;
      // waiting for it is correct, and a spurious failure is not.
      setReview(await getJSON('/api/storage', { timeout: 45000 }));
      setError('');
    } catch (cause) {
      setError(cause.message || 'Storage review is unavailable');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const deletable = useMemo(
    () => (review?.items || []).filter((item) => item.classification === 'SAFE_TO_DELETE'),
    [review],
  );

  const remove = async (item) => {
    if (item.classification !== 'SAFE_TO_DELETE' || item.referenced) return;
    const accepted = await confirmDialog({
      title: 'Delete this storage item?',
      message: `${item.relative_path} (${formatBytes(item.size_bytes)}) will be removed. It can be regenerated, but this action is explicit and cannot be undone by Roop.`,
      confirmLabel: 'Delete item',
      danger: true,
    });
    if (!accepted) return;
    setBusy(item.id);
    try {
      await postJSON('/api/storage/delete', { item_id: item.id, confirm: true });
      notify(`Deleted ${item.relative_path}`, 'info');
      await refresh();
    } catch (cause) {
      notify(cause.message || 'Storage item was not deleted', 'error');
      await refresh();
    } finally {
      setBusy('');
    }
  };

  return <section className="mt-8 rounded-2xl glass-panel p-5" aria-labelledby="storage-manager-title">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <div className="flex items-center gap-2"><Icon.reveal size={16} /><h3 id="storage-manager-title" className="text-sm font-bold text-white">Storage manager</h3></div>
        <p className="mt-1 max-w-3xl text-xs text-white/45">Review known application-owned disk usage. Classification comes from verified roots and live project/queue references; there is no automatic cleanup.</p>
      </div>
      <button type="button" onClick={refresh} disabled={loading} className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs font-semibold text-white/65 hover:text-white disabled:opacity-40">
        <Icon.refresh size={13} /> {loading ? 'Scanning…' : 'Refresh review'}
      </button>
    </div>

    {review?.active_work && <div className="mt-4 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200" role="status">
      Cleanup is restricted while active or resumable work exists: {(review.active_reasons || []).join('; ')}
    </div>}
    {error && <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-200" role="alert">{error}</div>}

    {review && <>
      <div className="mt-5 grid grid-cols-2 gap-2 md:grid-cols-4">
        <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><span className="text-nano uppercase text-white/40">Safe</span><strong className="mt-1 block text-sm text-emerald-300">{deletable.length} items</strong></div>
        <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><span className="text-nano uppercase text-white/40">Safe total</span><strong className="mt-1 block text-sm text-emerald-300">{formatBytes(deletable.reduce((sum, item) => sum + item.size_bytes, 0))}</strong></div>
        <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><span className="text-nano uppercase text-white/40">Reviewed items</span><strong className="mt-1 block text-sm text-amber-300">{(review.items || []).filter((item) => item.classification === 'REVIEW_BEFORE_DELETE').length}</strong></div>
        <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><span className="text-nano uppercase text-white/40">Protected</span><strong className="mt-1 block text-sm text-blue-300">{(review.items || []).filter((item) => item.classification === 'PROTECTED').length}</strong></div>
      </div>

      <div className="mt-5 space-y-3">
        {(review.items || []).map((item) => <article key={item.id} className="rounded-xl border border-white/10 bg-black/15 p-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="break-all text-xs font-semibold text-white/85">{item.relative_path}</div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-nano text-white/45"><span>{item.category}</span><span>{formatBytes(item.size_bytes)}</span><span>{item.is_directory ? 'directory' : 'file'}</span></div>
            </div>
            <Classification value={item.classification} />
          </div>
          <p className="mt-2 text-xs leading-relaxed text-white/55">{item.reason}</p>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-nano text-white/45">
            <span>Regenerable: <strong className="text-white/70">{item.regenerable ? 'yes' : 'no'}</strong></span>
            <span>Referenced: <strong className={item.referenced ? 'text-amber-300' : 'text-white/70'}>{item.referenced ? `yes (${item.reference_count})` : 'no'}</strong></span>
          </div>
          {item.classification === 'SAFE_TO_DELETE' && <button type="button" onClick={() => remove(item)} disabled={busy === item.id} className="mt-3 rounded-lg border border-red-400/30 bg-red-500/10 px-3 py-1.5 text-xs font-bold text-red-200 hover:bg-red-500/20 disabled:opacity-40">{busy === item.id ? 'Deleting…' : 'Delete this item'}</button>}
        </article>)}
        {(review.items || []).length === 0 && <p className="rounded-xl border border-white/10 px-3 py-4 text-xs text-white/45">No known disk items were found. Unknown drive-wide files are intentionally outside this manager.</p>}
      </div>
      <p className="mt-4 text-nano text-white/35">{review.deletion_policy}</p>
    </>}
  </section>;
}
