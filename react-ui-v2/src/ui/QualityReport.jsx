import React, { useEffect, useState } from 'react';
import { Gauge, Loader2, RefreshCw, AlertTriangle } from 'lucide-react';
import { postJSON } from '@engine/api';

// The engine's post-render quality report (POST /api/quality/analyze, routes_quality.py): it samples
// up to 24 frames of an output, re-detects the face, and scores identity likeness against the loaded
// source, sharpness, temporal stability (video) and detection coverage, weighted into a grade A-E.
// Heuristic, measured on the OUTPUT - not a claim about the settings. Run on demand: it takes the GPU.

const GRADE = { A: 'var(--ok)', B: 'var(--ok)', C: 'var(--warn)', D: 'var(--warn)', E: 'var(--danger)' };
const tone = (s) => (s >= 80 ? 'var(--ok)' : s >= 60 ? 'var(--accent-2)' : s >= 40 ? 'var(--warn)' : 'var(--danger)');
const VERDICT = { A: 'Excellent', B: 'Good', C: 'Acceptable', D: 'Needs work', E: 'Poor: review the settings' };

function Bar({ label, score, detail }) {
  const v = Math.max(0, Math.min(100, Number(score) || 0));
  return (
    <div>
      <div className="flex items-baseline justify-between text-[11.5px]">
        <span className="text-[var(--fg-dim)]">{label}</span>
        <span className="font-mono tnum" style={{ color: tone(v) }}>{v.toFixed(0)}<span className="text-[var(--fg-faint)]"> / 100{detail ? ` · ${detail}` : ''}</span></span>
      </div>
      <div className="bar mt-1"><i style={{ width: `${v}%`, background: tone(v) }} /></div>
    </div>
  );
}

/** path: the output's absolute path (from /api/output's absolute_path, or progress.output.path). */
export default function QualityReport({ path, compact = false }) {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  useEffect(() => { setData(null); setErr(''); }, [path]);
  const run = async () => {
    setBusy(true); setErr('');
    try {
      const r = await postJSON('/api/quality/analyze', { path }, { timeout: 0 });
      if (r?.ok) setData(r); else setErr(r?.message || 'No face detected in the output');
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  if (!path) return null;
  if (!data && !busy && !err) {
    return (
      <button type="button" className={`btn btn-ghost btn-sm ${compact ? '' : 'w-full'}`} onClick={run} data-testid="quality-run">
        <Gauge size={13} /> Quality report
      </button>
    );
  }
  if (busy) return <div className="flex items-center gap-2 py-1.5 text-[11.5px] text-[var(--fg-dim)]" data-testid="quality-busy"><Loader2 size={13} className="animate-spin" /> Measuring likeness, sharpness and stability…</div>;
  if (err) {
    return (
      <div className="flex items-center gap-2 py-1 text-[11.5px] text-[var(--warn)]" data-testid="quality-error">
        <AlertTriangle size={13} /> <span className="min-w-0 flex-1 truncate" title={err}>{err}</span>
        <button type="button" className="btn btn-ghost btn-sm" onClick={run}>Retry</button>
      </div>
    );
  }
  const m = data.metrics || {};
  const g = m.grade || '?';
  return (
    <div className="space-y-2 rounded-xl border border-[var(--rule)] bg-black/30 p-2.5" data-testid="quality-report">
      <div className="flex items-center gap-3">
        <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full border-2 text-lg font-bold" style={{ borderColor: GRADE[g] || 'var(--fg-faint)', color: GRADE[g] || 'var(--fg)' }} data-testid="quality-grade">{g}</span>
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-semibold text-white">{VERDICT[g] || 'Scored'} · {Number(m.overall_score || 0).toFixed(0)}</div>
          <div className="text-[11px] text-[var(--fg-faint)]">{data.is_video ? `${data.sampled} frames sampled` : 'Single image'}{data.has_source ? '' : ' · no source loaded, likeness skipped'}</div>
        </div>
        <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Re-run the quality report" title="Re-run" onClick={run}><RefreshCw size={13} /></button>
      </div>
      {m.identity_score !== undefined && <Bar label="Identity likeness" score={m.identity_score} detail={m.identity_similarity != null ? `cos ${m.identity_similarity}` : ''} />}
      {m.temporal_stability !== undefined && <Bar label="Temporal stability" score={m.temporal_stability} />}
      <Bar label="Sharpness" score={m.sharpness_score} />
      {data.is_video && m.detection_rate !== undefined && <Bar label="Face found in" score={m.detection_rate} detail="of sampled frames" />}
      <div className="text-[10.5px] text-[var(--fg-faint)]">Heuristic: faces are re-detected in the output. Likeness needs the source loaded.</div>
    </div>
  );
}
