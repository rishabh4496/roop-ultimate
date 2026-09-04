import React, { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from '../motion';
import { getJSON, postJSON } from '../api';
import { Card, Button, MotionIcon } from './ui';
import { Icon } from '../icons';

// The optimization benchmark: pre-run modal -> live gauges -> results screen ->
// accept/decline.
//
// This component RENDERS; it does not compute. Score, badge, comparison rows
// and presets all arrive from roop/benchmark/ui_dashboard.py so that the panel,
// the CLI (`--benchmark`) and the API agree by construction. A number derived
// here would be a second implementation with its own bugs, and the results
// screen is the worst place in the app to have one: it is a persuasion surface,
// so an unmeasured value shown here reads as a fact.
//
// Note the separate BenchmarkPanel.jsx: that drives roop/bench.py, a different
// (stage-cost and pool-curve) benchmark. Both are reachable; they answer
// different questions.

const POLL_MS = 1000;

const fmt = (v, d = 1) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');
const pct = (v) => (typeof v === 'number' && isFinite(v) ? `${Math.round(v)}%` : '—');

const TONE_CLASS = {
  good: 'state-ok',
  warn: 'state-warn',
  critical: 'state-err',
  neutral: 'state-info',
};

/* ── small primitives ─────────────────────────────────────────────────── */

function Gauge({ label, value, unit, sub, tone }) {
  return (
    <div className="p-3 rounded-xl bg-white/[0.04] border border-white/8 min-w-0">
      <span className="text-nano text-white/40 block truncate">{label}</span>
      <span className={`text-xl font-bold tabular-nums block ${tone ? `ink-${tone}` : 'text-white'}`}>
        {value}
        {unit ? <span className="text-xs font-normal text-white/40 ml-1">{unit}</span> : null}
      </span>
      {sub ? <span className="text-nano text-white/35 block truncate">{sub}</span> : null}
    </div>
  );
}

function LoadBar({ label, value, max = 100, tone = 'accent' }) {
  const filled = Math.max(0, Math.min(100, (Number(value) || 0) / max * 100));
  return (
    <div className="min-w-0">
      <div className="flex justify-between text-nano text-white/45 mb-1">
        <span className="truncate">{label}</span>
        <span className="tabular-nums">{pct(value)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/8 overflow-hidden">
        <motion.div
          className={`h-full rounded-full bg-${tone}`}
          animate={{ width: `${filled}%` }}
          transition={{ duration: 0.35 }}
        />
      </div>
    </div>
  );
}

// A dependency-free sparkline. The series is bounded server-side, so this
// cannot grow without limit no matter how long the run is.
function Spark({ series = [], height = 40, tone = 'accent' }) {
  if (!series.length) {
    return <div style={{ height }} className="rounded-lg bg-white/[0.03]" />;
  }
  const max = Math.max(...series, 0.0001);
  const min = Math.min(...series, 0);
  const span = max - min || 1;
  const step = 100 / Math.max(1, series.length - 1);
  const points = series
    .map((v, i) => `${(i * step).toFixed(2)},${(100 - ((v - min) / span) * 100).toFixed(2)}`)
    .join(' ');
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none"
      style={{ height }} className="w-full rounded-lg bg-white/[0.03]">
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth="1.5"
        vectorEffect="non-scaling-stroke" className={`ink-${tone}`} />
    </svg>
  );
}

/* ── 1. pre-benchmark modal ───────────────────────────────────────────── */

function PreBenchmarkModal({ prompt, busy, onStart, onClose }) {
  const [faces, setFaces] = useState(prompt?.default_faces || '1');
  const [mode, setMode] = useState(prompt?.default_mode || 'quick');
  if (!prompt) return null;

  return (
    <motion.div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <motion.div className="w-full max-w-lg"
        initial={{ scale: 0.96, y: 8 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.96, y: 8 }}>
        <Card className="p-5 space-y-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-base font-bold">Run Optimization Benchmark</h3>
              <p className="text-nano text-white/45 mt-0.5">
                Measures this machine with your current models, then recommends settings.
              </p>
            </div>
            <button onClick={onClose} className="text-white/40 hover:text-white shrink-0"
              aria-label="Close">
              <MotionIcon><Icon.close /></MotionIcon>
            </button>
          </div>

          {/* The models are LOCKED for the run and read back from the pipeline,
              not echoed from this UI — a benchmark that silently substitutes a
              cheaper model is measuring a different application. */}
          <div className="p-3 rounded-lg bg-white/[0.04] border border-white/8">
            <span className="text-nano text-white/40 block mb-1">Models locked for this run</span>
            <span className="text-xs text-white/80">{prompt.model_summary}</span>
          </div>

          {(prompt.warnings || []).map((w, i) => (
            <div key={i} className="state-warn text-nano px-2 py-1.5 rounded">{w}</div>
          ))}

          <div>
            <span className="text-nano text-white/45 block mb-1.5">Target Face Complexity</span>
            <div className="grid grid-cols-3 gap-2">
              {(prompt.face_choices || []).map((c) => (
                <button key={c.value} onClick={() => setFaces(c.value)} title={c.detail}
                  className={`p-2 rounded-lg border text-xs text-left transition
                    ${faces === c.value
                      ? 'border-accent bg-accent/12 text-white'
                      : 'border-white/10 bg-white/[0.03] text-white/60 hover:border-white/25'}`}>
                  {c.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <span className="text-nano text-white/45 block mb-1.5">Benchmark Mode</span>
            <div className="grid grid-cols-2 gap-2">
              {(prompt.mode_choices || []).map((c) => (
                <button key={c.value} onClick={() => setMode(c.value)}
                  className={`p-2.5 rounded-lg border text-left transition
                    ${mode === c.value
                      ? 'border-accent bg-accent/12'
                      : 'border-white/10 bg-white/[0.03] hover:border-white/25'}`}>
                  <span className="text-xs font-semibold block">{c.label}</span>
                  <span className="text-nano text-white/40 block mt-0.5">{c.detail}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="flex gap-2 pt-1">
            <Button onClick={() => onStart(faces, mode)} disabled={busy || !prompt.can_run}
              className="flex-1">
              {busy ? 'Starting…' : 'Start Benchmark'}
            </Button>
            <Button variant="ghost" onClick={onClose}>Cancel</Button>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  );
}

/* ── 2. live progress ─────────────────────────────────────────────────── */

function ProgressScreen({ snap, onCancel }) {
  const eta = snap.eta_sec;
  return (
    <Card className="p-4 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <span className="text-sm font-bold block truncate">{snap.status}</span>
          <span className="text-nano text-white/40">
            frame {snap.frame} / {snap.total_frames}
            {snap.frames_remaining ? ` · ${snap.frames_remaining} remaining` : ''}
            {typeof eta === 'number' && isFinite(eta) ? ` · ~${Math.round(eta)}s left` : ''}
          </span>
        </div>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>

      <div className="h-1.5 rounded-full bg-white/8 overflow-hidden">
        <motion.div className="h-full rounded-full bg-accent"
          animate={{ width: `${Math.max(0, Math.min(100, snap.progress_pct || 0))}%` }}
          transition={{ duration: 0.3 }} />
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <Gauge label="Current FPS" value={fmt(snap.current_fps, 2)} />
        <Gauge label="Average FPS" value={fmt(snap.average_fps, 2)} />
        <Gauge label="VRAM" value={fmt(snap.vram_used_mb, 0)} unit="MiB"
          sub={snap.vram_total_mb ? `${pct(snap.vram_pct)} of ${fmt(snap.vram_total_mb, 0)} MiB` : ''} />
        <Gauge label="Elapsed" value={fmt(snap.elapsed_sec, 0)} unit="s" />
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <div>
          <span className="text-nano text-white/40 block mb-1">FPS</span>
          <Spark series={snap.fps_series || []} />
        </div>
        <div>
          <span className="text-nano text-white/40 block mb-1">VRAM (MiB)</span>
          <Spark series={snap.vram_series || []} tone="warn" />
        </div>
      </div>

      <div className="grid sm:grid-cols-2 gap-3">
        <LoadBar label="CPU load" value={snap.cpu_pct} />
        {/* null means "never sampled", which is not the same as 0% — it is
            rendered as unavailable rather than as an idle GPU. */}
        {snap.gpu_pct === null || snap.gpu_pct === undefined
          ? <div className="text-nano text-white/35 self-center">GPU load not sampled on this system</div>
          : <LoadBar label="GPU load" value={snap.gpu_pct} tone="warn" />}
      </div>

      {snap.logs?.length ? (
        <pre className="text-nano text-white/40 max-h-24 overflow-y-auto whitespace-pre-wrap">
          {snap.logs.slice(-8).join('\n')}
        </pre>
      ) : null}
    </Card>
  );
}

/* ── 3. results dashboard ─────────────────────────────────────────────── */

function ComparisonTable({ rows }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10">
      <table className="w-full text-xs">
        <thead>
          <tr className="bg-white/[0.05] text-white/45">
            <th className="text-left font-medium px-3 py-2">Setting</th>
            <th className="text-left font-medium px-3 py-2">Current Value</th>
            <th className="text-left font-medium px-3 py-2">Recommended (Optimal)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.key || i} className="border-t border-white/8">
              <td className="px-3 py-2 text-white/70">
                {r.setting}
                {r.requires_restart ? (
                  <span className="state-info text-nano px-1 py-0.5 rounded ml-1.5">restart</span>
                ) : null}
              </td>
              <td className="px-3 py-2 tabular-nums text-white/55">{r.current}</td>
              <td className="px-3 py-2 tabular-nums">
                <span className={r.changed ? 'text-white font-semibold' : 'text-white/45'}>
                  {r.recommended}
                </span>
                {r.delta ? <span className="ink-ok ml-1.5">{r.delta}</span> : null}
                {/* A projection is labelled as one. This is the cell where an
                    estimate would otherwise read as a measurement. */}
                {r.evidence && r.evidence !== 'measured' ? (
                  <span className="state-warn text-nano px-1 py-0.5 rounded ml-1.5">{r.evidence}</span>
                ) : null}
                {r.note ? (
                  <span className="text-nano text-white/35 block mt-0.5">{r.note}</span>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultsDashboard({ report, onApply, onDecline, onRevert, busy, notice }) {
  const [preset, setPreset] = useState('balanced');
  const presets = report.presets || {};
  const presetOrder = ['max_throughput', 'balanced', 'stable_low_power'];

  return (
    <Card className="p-4 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-bold">Benchmark Results</h3>
          <span className="text-nano text-white/40">
            {report.workload?.face_label || report.workload?.name || ''} ·{' '}
            {report.frames_processed} frames
          </span>
        </div>
        <span className={`${TONE_CLASS[report.badge_tone] || 'state-info'} text-xs font-bold px-2.5 py-1 rounded`}>
          {report.badge}
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <div className="p-3 rounded-xl bg-accent/10 border border-accent/25 col-span-2 sm:col-span-1">
          <span className="text-nano text-white/45 block">Performance Score</span>
          <span className="text-3xl font-black tabular-nums ink-accent">{report.score}</span>
        </div>
        <Gauge label="Average FPS" value={fmt(report.average_fps, 2)} />
        <Gauge label="1% Low FPS" value={fmt(report.p1_low_fps, 2)}
          sub="the stutter floor" />
        <Gauge label="Peak VRAM" value={fmt(report.peak_vram_mb, 0)} unit="MiB" />
      </div>

      {/* The index states its own denominator. A score with no stated basis is
          a number nobody can check. */}
      <p className="text-nano text-white/35">{report.score_basis}</p>

      <div>
        <span className="text-nano text-white/45 block mb-1">{report.badge_detail}</span>
        {(report.bottleneck_evidence || []).map((e, i) => (
          <span key={i} className="text-nano text-white/35 block">· {e}</span>
        ))}
      </div>

      <ComparisonTable rows={report.comparison || []} />

      <div>
        <span className="text-nano text-white/45 block mb-1.5">Preset</span>
        <div className="grid grid-cols-3 gap-2">
          {presetOrder.filter((k) => presets[k]).map((key) => (
            <button key={key} onClick={() => setPreset(key)}
              title={presets[key].rationale}
              className={`p-2.5 rounded-lg border text-left transition
                ${preset === key
                  ? 'border-accent bg-accent/12'
                  : 'border-white/10 bg-white/[0.03] hover:border-white/25'}`}>
              <span className="text-xs font-semibold block">{presets[key].name || key}</span>
              {presets[key].measured_fps ? (
                <span className="text-nano text-white/40 block tabular-nums">
                  {fmt(presets[key].measured_fps, 2)} FPS
                </span>
              ) : null}
              {presets[key].constraint_violated ? (
                <span className="state-warn text-nano px-1 rounded mt-1 inline-block">
                  constraint not met
                </span>
              ) : null}
            </button>
          ))}
        </div>
      </div>

      {(report.warnings || []).map((w, i) => (
        <div key={i} className="state-warn text-nano px-2 py-1.5 rounded">{w}</div>
      ))}

      {notice ? <div className="state-ok text-nano px-2 py-1.5 rounded">{notice}</div> : null}

      <div className="flex flex-wrap gap-2 pt-1">
        <Button onClick={() => onApply(preset)} disabled={busy} className="flex-1 min-w-[12rem]">
          Apply Recommended Settings
        </Button>
        <Button variant="ghost" onClick={onDecline} disabled={busy}>
          Decline / Keep Current
        </Button>
        {/* Scoped to the settings the benchmark writes — not an app reset. */}
        <Button variant="ghost" onClick={onRevert} disabled={busy}
          title="Restore the shipped defaults for the settings this benchmark can change">
          Revert to Default
        </Button>
      </div>
    </Card>
  );
}

/* ── the panel ────────────────────────────────────────────────────────── */

export default function OptimizationBenchmark() {
  const [prompt, setPrompt] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [snap, setSnap] = useState(null);
  const [report, setReport] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const timer = useRef(null);

  const stopPolling = useCallback(() => {
    if (timer.current) { clearInterval(timer.current); timer.current = null; }
  }, []);

  // A run outlives this component: the session is process-wide, so on mount we
  // adopt whatever is already in flight rather than showing an idle panel over
  // a running benchmark.
  const poll = useCallback(async () => {
    try {
      const s = await getJSON('/api/benchmark/progress', { timeout: 8000 });
      setSnap(s);
      // Stop on `running` going false, not on `done || error`: a CANCELLED run
      // is terminal with neither set, and waiting for one of them polls a
      // finished session forever. `running` is set synchronously by start(),
      // so there is no window where it reads false before the worker begins.
      if (!s.running) {
        stopPolling();
        if (s.done) {
          const r = await getJSON('/api/benchmark/result', { timeout: 8000 });
          if (r.ready) setReport(r);
        }
        if (s.error) setError(s.error);
        if (s.cancelled) setNotice('Benchmark cancelled. Nothing was changed.');
      }
    } catch (e) {
      stopPolling();
      setError(e.message || 'Lost contact with the backend.');
    }
  }, [stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    poll();
    timer.current = setInterval(poll, POLL_MS);
  }, [poll, stopPolling]);

  useEffect(() => {
    (async () => {
      try {
        const s = await getJSON('/api/benchmark/progress', { timeout: 8000 });
        setSnap(s);
        if (s.running) startPolling();
        else if (s.done) {
          const r = await getJSON('/api/benchmark/result', { timeout: 8000 });
          if (r.ready) setReport(r);
        }
      } catch { /* the panel is still usable; Open will surface the error */ }
    })();
    return stopPolling;
  }, [startPolling, stopPolling]);

  const openModal = async () => {
    setError(''); setNotice('');
    try {
      setPrompt(await getJSON('/api/benchmark/prompt', { timeout: 15000 }));
      setShowModal(true);
    } catch (e) {
      setError(e.message || 'Could not read the active models.');
    }
  };

  const start = async (faces, mode) => {
    setBusy(true); setError(''); setNotice(''); setReport(null);
    try {
      await postJSON('/api/benchmark/start', { faces, mode });
      setShowModal(false);
      startPolling();
    } catch (e) {
      setError(e.message || 'Could not start the benchmark.');
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    try { await postJSON('/api/benchmark/cancel', {}); } catch { /* it ends either way */ }
  };

  const apply = async () => {
    setBusy(true); setError('');
    try {
      const r = await postJSON('/api/benchmark/apply', { run_id: report?.run_id });
      setNotice(r.message || 'Applied.');
      setReport((prev) => (prev ? { ...prev, applied: true } : prev));
    } catch (e) {
      setError(e.message || 'Could not apply the settings.');
    } finally { setBusy(false); }
  };

  const revert = async () => {
    setBusy(true); setError('');
    try {
      const r = await postJSON('/api/benchmark/revert', {});
      setNotice(r.message || 'Restored defaults.');
    } catch (e) {
      setError(e.message || 'Could not restore the defaults.');
    } finally { setBusy(false); }
  };

  const decline = async () => {
    setBusy(true); setError('');
    try {
      const r = await postJSON('/api/benchmark/decline', { run_id: report?.run_id });
      setNotice(r.notice || r.message || '');
    } catch (e) {
      setError(e.message || 'Could not save the run.');
    } finally { setBusy(false); }
  };

  const running = !!snap?.running;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-bold">Optimization Benchmark</h2>
          <p className="text-nano text-white/40">
            Measures this machine with your current models and recommends settings.
          </p>
        </div>
        <Button onClick={openModal} disabled={running}>
          {running ? 'Running…' : 'Run Benchmark'}
        </Button>
      </div>

      {error ? <div className="state-err text-nano px-2 py-1.5 rounded">{error}</div> : null}

      {running && snap ? <ProgressScreen snap={snap} onCancel={cancel} /> : null}

      {!running && report ? (
        <ResultsDashboard report={report} onApply={apply} onDecline={decline}
          onRevert={revert} busy={busy} notice={notice} />
      ) : null}

      <AnimatePresence>
        {showModal ? (
          <PreBenchmarkModal prompt={prompt} busy={busy} onStart={start}
            onClose={() => setShowModal(false)} />
        ) : null}
      </AnimatePresence>
    </div>
  );
}
