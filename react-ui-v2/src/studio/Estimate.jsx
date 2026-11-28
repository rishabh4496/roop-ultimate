import React, { useState } from 'react';
import { Timer, Lightbulb, Loader2, Check, X } from 'lucide-react';
import useRuntimeEstimate from '@engine/components/faceswap/useRuntimeEstimate';
import { postJSON } from '@engine/api';
import { useStudio } from '../engine/studio';
import { useRun } from '../engine/run';
import { inOutOf } from '../engine/payload';
import { notify } from '../ui/controls';

// ── Pre-run estimate ──
// 1.0's useRuntimeEstimate: ms/frame MEASURED for these settings on past completed runs
// (roop.runtime_calib, keyed on a settings signature), blended with a heuristic while the data is
// thin, and the heuristic alone before any run. Labelled with which it is: a guess never reads as a
// measurement.
const fmt = (ms) => {
  if (!Number.isFinite(ms) || ms <= 0) return '—';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m ${String(s % 60).padStart(2, '0')}s`;
};

export function RunEstimate() {
  const settings = useStudio((s) => s.settings) || {};
  const target = useStudio((s) => s.targets[s.selectedTarget]);
  const faces = useStudio((s) => s.preview?.faces?.length || 0);
  const processing = useRun((r) => r.progress.processing);
  const { a, b } = inOutOf(target);
  const frames = target ? b - a + 1 : 0;
  const est = useRuntimeEstimate({ settings, estFrames: frames, faceCount: faces || undefined, processing, hasTargets: !!target, threads: settings.max_threads });
  if (!target || frames <= 1 || processing) return null;
  const learned = est.estLearned;
  return (
    <div className="mb-2 flex items-center gap-1.5 font-mono text-[11px] text-[var(--fg-dim)]" data-testid="run-estimate"
      title={learned ? `Measured ${Math.round(est.calibEst.ms_per_frame)} ms/frame over ${est.calibEst.samples} past run(s) with these settings` : 'A rough guess from the settings: no finished run with these settings yet'}>
      <Timer size={12} /> about <b className="text-white">{fmt(est.estTotalMs)}</b> for {frames} frames
      <span style={{ color: learned ? 'var(--ok)' : 'var(--fg-faint)' }}>· {learned ? `learned from ${est.calibEst.samples} run${est.calibEst.samples === 1 ? '' : 's'}` : 'rough guess'}</span>
    </div>
  );
}

// ── Clip advisor ──
// POST /api/advisor samples up to 12 frames of the target (face count and size, detection coverage,
// motion, brightness) and returns only the settings it would CHANGE, each with its reason. Nothing
// is applied server-side; the user applies what they accept.
export function ClipAdvisor() {
  const s = useStudio();
  const processing = useRun((r) => r.progress.processing);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState({});
  if (!s.targets.length) return null;
  const ask = async () => {
    setBusy(true); setDone({});
    try { setRes(await postJSON('/api/advisor', { index: s.selectedTarget, settings: s.settings }, { timeout: 0 })); }
    catch (e) { notify(e.message, 'error'); } finally { setBusy(false); }
  };
  const recs = (res?.recommendations || []).filter((r) => r && r.key);
  const apply = (list) => {
    const patch = Object.fromEntries(list.map((r) => [r.key, r.value]));
    s.setSettings(patch);
    setDone((d) => ({ ...d, ...Object.fromEntries(list.map((r) => [r.key, true])) }));
    notify(`Applied ${list.length} suggestion${list.length === 1 ? '' : 's'}`);
  };
  return (
    <div className="mt-2" data-testid="advisor">
      {!res && (
        <button type="button" className="btn btn-ghost btn-sm w-full" disabled={busy || processing} onClick={ask} data-testid="advisor-run"
          title="Look at a dozen frames of this clip and suggest settings for it">
          {busy ? <Loader2 size={13} className="animate-spin" /> : <Lightbulb size={13} />} {busy ? 'Looking at the clip…' : 'Suggest settings for this clip'}
        </button>
      )}
      {res && (
        <div className="rounded-xl border border-[var(--rule)] bg-black/25 p-2 text-[12px]" data-testid="advisor-result">
          <div className="flex items-center gap-2">
            <Lightbulb size={13} className="text-[var(--accent-2)]" />
            <span className="min-w-0 flex-1 font-semibold text-white">{recs.length ? `${recs.length} suggestion${recs.length === 1 ? '' : 's'}` : 'No changes suggested'}</span>
            {recs.some((r) => !done[r.key]) && <button type="button" className="btn btn-sm" onClick={() => apply(recs.filter((r) => !done[r.key]))} data-testid="advisor-apply-all"><Check size={12} /> Apply all</button>}
            <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Close the suggestions" onClick={() => setRes(null)}><X size={12} /></button>
          </div>
          {res.stats && (
            <div className="mt-1 font-mono text-[10.5px] text-[var(--fg-faint)]">
              {res.stats.sampled_frames} frames · faces in {res.stats.detection_coverage}% · ~{res.stats.median_faces} per frame · size {res.stats.min_face_size_pct}-{res.stats.max_face_size_pct}%{res.stats.motion != null ? ` · motion ${res.stats.motion}` : ''}
            </div>
          )}
          {res.message && !recs.length && <div className="mt-1 text-[var(--fg-dim)]">{res.message}</div>}
          <ul className="mt-1.5 space-y-1">
            {recs.map((r) => (
              <li key={r.key} className="flex items-start gap-2" data-testid="advisor-rec">
                <span className="min-w-0 flex-1"><b className="font-mono text-white">{r.key}</b> → <span className="font-mono text-[var(--ok)]">{JSON.stringify(r.value)}</span><span className="block text-[11px] text-[var(--fg-dim)]">{r.reason}</span></span>
                <button type="button" className="btn btn-ghost btn-sm shrink-0" disabled={done[r.key]} onClick={() => apply([r])}>{done[r.key] ? 'Applied' : 'Apply'}</button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
