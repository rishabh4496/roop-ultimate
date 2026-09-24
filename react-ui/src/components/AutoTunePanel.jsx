import React, { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON } from '../api';
import { Button } from './ui';

// Auto-tune: replays the LAST render on a short stretch of its own target and
// measures provider x cross-frame swap batch, then NVENC presets. Screening
// (100 frames, each arm twice) only ranks; a setting is saved only if it beats
// the current one in both halves of a 600-frame A/B/B/A AND by more than the
// current setting's own run-to-run spread. The protocol lives in
// roop/benchmark/autotune.py; this panel only starts it and shows it.

const pct = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${v}%`);

export default function AutoTunePanel({ notify, onSettingsApplied }) {
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState('');
  const poll = useRef(null);
  const wasRunning = useRef(false);

  const load = useCallback(async () => {
    try {
      const s = await getJSON('/api/autotune', { timeout: 15000 });
      setStatus(s);
      const running = !!s?.progress?.running;
      if (wasRunning.current && !running) onSettingsApplied?.();
      wasRunning.current = running;
    } catch {
      // backend restarting; the next poll retries
    }
  }, [onSettingsApplied]);

  useEffect(() => {
    load();
    poll.current = setInterval(load, 2000);
    return () => clearInterval(poll.current);
  }, [load]);

  const act = async (what, path) => {
    setBusy(what);
    try {
      const res = await postJSON(path, {});
      if (what === 'revert') notify?.('Restored the settings auto-tune replaced', 'success');
      if (what === 'start') notify?.(`Auto-tune started: baseline ${res.baseline}, ${res.threads} workers`, 'info');
      load();
    } catch (e) {
      notify?.(e.message || String(e), 'error');
    } finally {
      setBusy('');
    }
  };

  if (!status) return <p className="text-xs text-white/40">Checking auto-tune…</p>;
  const prog = status.progress;
  const running = !!prog?.running;
  const result = status.result;

  return (
    <div className="flex flex-col gap-3 text-xs">
      <p className="text-white/50">
        Measures CUDA vs TensorRT and swap batch 1/2/4/8 on your last render
        {status.target ? <> (<span className="text-white/80">{status.target}</span>)</> : null}, then NVENC p1-p7.
        Nothing is saved unless it beats your current settings in a 600-frame counterbalanced run by more than the noise.
        Takes roughly 15-30 minutes; one render at a time, so the Start button is locked meanwhile.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {!running ? (
          <Button size="sm" onClick={() => act('start', '/api/autotune/start')} disabled={!status.ready || !!busy}>
            {busy === 'start' ? 'Starting…' : 'Run auto-tune'}
          </Button>
        ) : (
          <Button size="sm" variant="stop" onClick={() => act('cancel', '/api/autotune/cancel')} disabled={!!busy}>Cancel</Button>
        )}
        {!running && result?.applied && Object.keys(result.applied).length > 0 && (
          <Button size="sm" variant="secondary" onClick={() => act('revert', '/api/autotune/revert')} disabled={!!busy}>Revert applied</Button>
        )}
        {!status.ready && !running && status.reason && <span className="text-amber-300/80">{status.reason}</span>}
      </div>

      {running && (
        <div className="rounded-md border border-white/10 p-2 flex flex-col gap-1">
          <div className="flex justify-between text-white/70">
            <span>{prog.phase}: {prog.status}</span>
            <span>{prog.arms_done}/{prog.arms_total} arms{prog.last_fps ? ` · last ${prog.last_fps.toFixed(2)} fps` : ''}</span>
          </div>
          <div className="h-1.5 rounded bg-white/10 overflow-hidden">
            <div className="h-full fill-accent" style={{ width: `${Math.min(100, (100 * prog.arms_done) / Math.max(1, prog.arms_total))}%` }} />
          </div>
          <pre className="max-h-32 overflow-auto text-nano text-white/50 whitespace-pre-wrap">{(prog.log || []).slice(-12).join('\n')}</pre>
        </div>
      )}

      {!running && result && (
        <div className="flex flex-col gap-2">
          <div className="text-white/70">
            Last run {result.finished} on {result.target || 'the last render'} · {result.status}
            {result.winner && <> · kept <span className="text-white">{result.winner.arm}</span> at {result.winner.fps} fps{result.winner.changed ? ' (new)' : ' (unchanged)'}</>}
            {result.applied && Object.keys(result.applied).length > 0 && <> · saved {Object.entries(result.applied).map(([k, v]) => `${k}=${v}`).join(', ')}</>}
            {result.restart_required && <span className="text-amber-300"> · restart to switch provider</span>}
          </div>
          {result.confirm_meets_600_rule === false && (
            <p className="text-amber-300/80">The target was shorter than 600 frames, so the confirmation is shorter than the acceptance rule. Treat this as screening.</p>
          )}
          {Object.entries(result.provider_notes || {}).map(([p, why]) => <p key={p} className="text-white/40">{p}: {why}</p>)}
          {result.screen && (
            <table className="w-full text-nano">
              <thead><tr className="text-white/40 text-left"><th className="px-1">arm</th><th className="px-1 text-right">screen fps</th><th className="px-1 text-right">swaps</th><th className="px-1">note</th></tr></thead>
              <tbody>
                {result.screen.map((r) => (
                  <tr key={r.arm} className="border-t border-white/5">
                    <td className="px-1 text-white/80">{r.arm}{r.arm === result.baseline ? ' (current)' : ''}</td>
                    <td className="px-1 text-right">{r.mean_fps}</td>
                    <td className="px-1 text-right">{r.min_swaps}</td>
                    <td className="px-1 text-white/40">{r.excluded || (result.finalists?.includes(r.arm) ? 'finalist' : '')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {(result.confirm || []).map((c) => (
            <p key={c.arm} className={c.accepted ? 'text-emerald-300' : 'text-white/50'}>
              {c.arm} vs current at {result.confirm_frames} frames: {pct(c.improvement_pct)} (noise {c.noise_pct}%). {c.reason}
            </p>
          ))}
          {result.encoder && (
            <div>
              <p className="text-white/60">
                {result.encoder.codec} at cq {result.encoder.cq}: picked <span className="text-white">{result.encoder.picked || '–'}</span>, the best preset that encodes at ≥ {result.encoder.need_fps} fps (2× the render).
                {!result.encoder.applies && ` Not saved: your codec is ${result.encoder.configured_codec}, which does not use NVENC presets.`}
              </p>
              <table className="w-full text-nano">
                <tbody>
                  {result.encoder.rows.map((r) => (
                    <tr key={r.preset} className="border-t border-white/5">
                      <td className="px-1 text-white/80">{r.preset}</td>
                      <td className="px-1 text-right">{r.fps ?? 'failed'} fps{r.decode_bound ? ' (decode-bound: at least this)' : ''}</td>
                      <td className="px-1 text-right">{r.mbps ?? '–'} Mbps</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
