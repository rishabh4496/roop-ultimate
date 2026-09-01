import React, { useCallback, useEffect, useState } from 'react';
import { deleteStorageItem, getStorageReview } from '../api';
import { useNotifications } from '../state/appState';
import { useTheme } from '../theme/ThemeProvider';
import { useOperationsStatus } from '../workflow/useOperationsStatus';
import { Badge, Button, Card, Notice, Select } from '../components/primitives';

function display(value) {
  return value == null || value === '' ? 'UNKNOWN' : String(value);
}

function size(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value)) return 'UNKNOWN';
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}

function EnvironmentEvidence({ status }) {
  const runtime = status.runtime;
  const hardware = status.hardware;
  const profile = status.profile;
  const observed = Boolean(runtime) && hardware?.available !== false;
  const badge = status.loading ? 'Checking' : observed ? 'Observed' : status.error ? 'Unavailable' : 'Unknown';
  const tone = observed ? 'success' : status.error ? 'danger' : 'neutral';
  const vram = runtime?.vram || {};
  const profileCount = Array.isArray(profile?.stages) ? profile.stages.length : null;
  return <Card>
    <div className="v2-card-heading"><div><span className="v2-eyebrow">Environment health evidence</span><h3>Local runtime diagnostics</h3></div><Badge tone={tone}>{badge}</Badge></div>
    {status.error && <Notice tone="danger" title="Some diagnostics are unavailable">{status.error}. Values below remain backend-reported; V2 does not fill missing measurements.</Notice>}
    <div className="v2-evidence-grid">
      <div><span>Runtime status</span><strong>{display(runtime?.status?.message || runtime?.status?.code)}</strong></div>
      <div><span>Application</span><strong>{display(status.meta?.git_version)}</strong></div>
      <div><span>Provider</span><strong>{display(runtime?.provider)}</strong></div>
      <div><span>GPU</span><strong title={display(runtime?.gpu)}>{display(runtime?.gpu)}</strong></div>
      <div><span>VRAM</span><strong>{vram.used_gb !== 'UNKNOWN' && vram.total_gb !== 'UNKNOWN' ? `${display(vram.used_gb)} / ${display(vram.total_gb)} GB` : 'UNKNOWN'}</strong></div>
      <div><span>Profile stages</span><strong>{profile?.enabled === false ? 'Disabled' : display(profileCount)}</strong></div>
    </div>
    <p className="v2-muted v2-operations-note">This is live API evidence from <code>/api/runtime/state</code>, <code>/api/system/hardware</code>, and <code>/api/system/profile</code>. The full child-process updater health worker remains a Pinokio/CLI operation and is not claimed here.</p>
    <Button size="sm" onClick={status.refresh} disabled={status.loading}>{status.loading ? 'Refreshing...' : 'Refresh evidence'}</Button>
  </Card>;
}

function UpdateCenter() {
  return <Card>
    <div className="v2-card-heading"><div><span className="v2-eyebrow">Update center</span><h3>Compatibility-gated launcher updates</h3></div><Badge tone="neutral">Pinokio-managed</Badge></div>
    <Notice title="No browser update endpoint is verified">Use the Pinokio Update action to invoke <code>update.js</code> and its manifest-gated <code>app/update_manager.py</code> flow. V2 does not present a fake “latest” result or silently install dependencies, models, CUDA, ONNX Runtime, TensorRT, Python, FFmpeg, or drivers.</Notice>
    <div className="v2-boundary-row"><span>Browser update check</span><strong>Unavailable</strong></div>
    <div className="v2-boundary-row"><span>Rollback scope</span><strong>Source/configuration only</strong></div>
  </Card>;
}

function StorageReview() {
  const { notify } = useNotifications();
  const [review, setReview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [confirmId, setConfirmId] = useState('');
  const [deleting, setDeleting] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setReview(await getStorageReview());
      setError('');
    } catch (cause) {
      setError(cause.message || 'Storage review unavailable');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh().catch(() => {}); }, [refresh]);

  const remove = async (item) => {
    if (item.classification !== 'SAFE_TO_DELETE' || item.referenced) return;
    setDeleting(item.id);
    try {
      await deleteStorageItem(item.id);
      setConfirmId('');
      notify('Storage item deleted after server revalidation', 'success');
      await refresh();
    } catch (cause) {
      setError(cause.message || 'Storage deletion failed');
      notify(cause.message || 'Storage deletion failed', 'danger');
      await refresh();
    } finally {
      setDeleting('');
    }
  };

  const items = review?.items || [];
  return <Card className="v2-storage-card">
    <div className="v2-card-heading"><div><span className="v2-eyebrow">Cleanup</span><h3>Storage review</h3></div><Badge tone="neutral">{loading ? 'Checking' : `${items.length} items`}</Badge></div>
    <p className="v2-muted">Only the application-owned review is shown. Protected projects, checkpoints, models, dependencies, environments, outputs, and referenced files cannot be deleted here.</p>
    {error && <Notice tone="danger" title="Storage review failed">{error}</Notice>}
    {review?.active_work && <Notice title="Cleanup is restricted">Active or resumable work is present. Safe candidates are protected until that work is no longer active.</Notice>}
    {!loading && !items.length && <div className="v2-empty-picker">No known storage items were reported.</div>}
    <div className="v2-storage-list">{items.map((item) => {
      const canDelete = item.classification === 'SAFE_TO_DELETE' && !item.referenced;
      const isConfirming = confirmId === item.id;
      return <div className="v2-storage-row" key={item.id}>
        <div className="v2-storage-main"><div className="v2-queue-title"><strong>{item.relative_path || item.path}</strong><Badge tone={item.classification === 'PROTECTED' ? 'neutral' : item.classification === 'SAFE_TO_DELETE' ? 'success' : 'accent'}>{display(item.classification)}</Badge></div><small>{display(item.category)} · {size(item.size_bytes)} · {item.regenerable ? 'Regenerable' : 'Not regenerable'} · {item.referenced ? `Referenced (${item.reference_count || 1})` : 'Not referenced'}</small><p>{display(item.reason)}</p></div>
        <div className="v2-storage-actions">{canDelete && !isConfirming && <Button size="sm" variant="danger" onClick={() => setConfirmId(item.id)}>Delete</Button>}{isConfirming && <><Button size="sm" onClick={() => setConfirmId('')} disabled={deleting === item.id}>Cancel</Button><Button size="sm" variant="danger" onClick={() => remove(item)} disabled={deleting === item.id}>{deleting === item.id ? 'Deleting...' : 'Confirm delete'}</Button></>}</div>
      </div>;
    })}</div>
    <Button size="sm" onClick={refresh} disabled={loading}>{loading ? 'Refreshing...' : 'Refresh storage review'}</Button>
  </Card>;
}

export default function SettingsScreen() {
  const { theme, themes, setTheme } = useTheme();
  const status = useOperationsStatus();
  return <div className="v2-screen">
    <div className="v2-page-heading"><div><span className="v2-eyebrow">Runtime and maintenance</span><h2>Settings</h2><p>Presentation preferences and verified operational boundaries stay visible in one place.</p></div></div>
    <div className="v2-grid v2-grid-two">
      <Card><div className="v2-card-heading"><div><span className="v2-eyebrow">Theme engine</span><h3>Choose a shared visual language</h3></div></div><Select label="Theme" value={theme} onChange={(event) => setTheme(event.target.value)} options={Object.entries(themes).map(([value, item]) => ({ value, label: item.label }))} hint="All seven themes use the same primitives and layout." /><div className="v2-theme-grid">{Object.entries(themes).map(([id, item]) => <Button key={id} variant={theme === id ? 'primary' : 'secondary'} size="sm" onClick={() => setTheme(id)} aria-pressed={theme === id}>{item.label}</Button>)}</div></Card>
      <EnvironmentEvidence status={status} />
      <UpdateCenter />
      <StorageReview />
    </div>
  </div>;
}
