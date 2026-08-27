import React, { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON } from '../api';
import { MotionIcon } from './ui';
import { Icon } from '../icons';

// The hardware benchmark's UI. Split out of Settings.jsx because what the
// backend now returns is a report rather than three numbers: a per-stage cost
// table, a pool-scaling curve per stage, a thread curve per workload mode, a
// TensorRT-vs-CUDA comparison and an encode/decode pass.
//
// The old card showed "12 Threads / 18698.7 FPS" in green with a "✓ Verified 0
// Errors / Max GPU Speed" badge. Both were invented: nothing verified anything,
// and the FPS was a batched-matmul rate with no relationship to frames. A
// benchmark that cannot be argued with is not much use, so every recommendation
// here is shown next to the curve it came from.

const PROFILES = [
  { id: 'quick', label: 'Quick', hint: '~4 min · thread count only, pools left alone' },
  { id: 'full', label: 'Full', hint: '~7 min · pool sweep, TensorRT vs CUDA, batched swap, encode/decode' },
];

const num = (v, d = 1) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');

const PROFILE_STRESS = { id: 'stress', label: 'Stress', hint: '15+ min · 2+ faces/frame, sustained repeated composite, stability metrics' };

function Chip({ tone = 'info', children, title }) {
  return (
    <span title={title}
      className={`state-${tone} text-nano font-bold px-1.5 py-0.5 rounded whitespace-nowrap`}>
      {children}
    </span>
  );
}

function Tile({ label, value, sub, tone, title }) {
  return (
    <div className="p-2.5 rounded-lg bg-white/[0.04] border border-white/8 min-w-0"
      title={title}>
      <span className="text-nano text-white/40 block truncate">{label}</span>
      <span className={`text-sm font-bold block tabular-nums ${tone ? `ink-${tone}` : 'text-white'}`}>
        {value}
      </span>
      {sub && <span className="text-nano text-white/35 block truncate">{sub}</span>}
    </div>
  );
}

// A curve is drawn rather than tabulated because the SHAPE is the finding: a
// flat line past four threads and a line still climbing at sixteen call for
// different settings, and a row of numbers hides which one you have.
function Curve({ points, pick, unit }) {
  const keys = Object.keys(points || {}).map(Number).sort((a, b) => a - b);
  if (!keys.length) return null;
  const max = Math.max(...keys.map((k) => points[String(k)] || 0)) || 1;
  return (
    <div className="flex items-end gap-1 h-16">
      {keys.map((k) => {
        const v = points[String(k)] || 0;
        const on = String(k) === String(pick);
        return (
          <div key={k} className="flex-1 flex flex-col items-center gap-1 min-w-0">
            <span className={`text-nano tabular-nums ${on ? 'ink-accent font-bold' : 'text-white/35'}`}>
              {num(v, v < 10 ? 1 : 0)}
            </span>
            <div className="w-full rounded-sm transition-all duration-500"
              style={{
                height: `${Math.max(3, (v / max) * 38)}px`,
                background: on ? 'var(--accent)' : 'color-mix(in oklab, var(--accent) 28%, transparent)',
              }} />
            <span className={`text-nano tabular-nums ${on ? 'text-white/70 font-bold' : 'text-white/30'}`}>
              {k}
            </span>
          </div>
        );
      })}
      {unit && <span className="text-nano text-white/25 self-end pb-3 pl-1 shrink-0">{unit}</span>}
    </div>
  );
}

function Foldout({ title, count, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-white/8 bg-black/25">
      <button type="button" onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full flex items-center justify-between px-3 py-2 text-micro font-bold text-white/60 hover:text-white transition-colors">
        <span>{title}{count != null && <span className="text-white/30 font-medium"> · {count}</span>}</span>
        <Icon.expand size={11} className={open ? 'rotate-180 transition-transform' : 'transition-transform'} />
      </button>
      {open && <div className="px-3 pb-3 overflow-x-auto">{children}</div>}
    </div>
  );
}

export default function BenchmarkPanel({ saved, currentSettings, onResult, notify }) {
  const [profile, setProfile] = useState('full');
  const [faces, setFaces] = useState(2);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(saved || null);
  const pollRef = useRef(null);

  useEffect(() => { if (saved) setReport(saved); }, [saved]);

  // Reattach to a benchmark already in flight. Without this, switching tabs (or
  // Pinokio reloading the frontend, which it does on tab switch) leaves a run
  // going on the backend with no UI attached and no way to cancel it.
  useEffect(() => {
    let live = true;
    getJSON('/api/settings/benchmark_status')
      .then((st) => { if (live && st && st.running) { setStatus(st); setRunning(true); } })
      .catch(() => {});
    return () => { live = false; };
  }, []);

  const poll = useCallback(async () => {
    try {
      const st = await getJSON('/api/settings/benchmark_status');
      setStatus(st);
      if (!st.running) {
        setRunning(false);
        if (st.result) {
          setReport(st.result);
          if (onResult) onResult(st.result);
          if (notify) notify('Benchmark complete — ' + (st.result.summary || ''), 'success');
        } else if (st.error) {
          if (notify) notify('Benchmark failed: ' + st.error, 'error');
        }
      }
    } catch {
      // The backend restarts on some settings changes; the next tick retries.
    }
  }, [onResult, notify]);

  useEffect(() => {
    if (!running) return undefined;
    pollRef.current = setInterval(poll, 1000);
    return () => clearInterval(pollRef.current);
  }, [running, poll]);

  const start = async () => {
    try {
      setStatus(null);
      const requestedFaces = Math.max(1, Number(faces) || 1);
      const res = await postJSON('/api/settings/benchmark_threads', {
        profile,
        faces_per_frame: profile === 'stress' ? Math.max(2, requestedFaces) : requestedFaces,
      });
      if (res && res.status === 'started') {
        setRunning(true);
        if (notify) notify(`Benchmark started (${profile}, about ${Math.round((res.est_sec || 300) / 60)} min)`, 'info');
      } else if (notify) {
        notify(res?.message || 'Could not start', 'warning');
      }
    } catch (e) {
      if (notify) notify('Benchmark failed to start: ' + e.message, 'error');
    }
  };

  const cancel = async () => {
    try {
      await postJSON('/api/settings/benchmark_cancel', {});
      if (notify) notify('Cancelling…', 'warning');
    } catch (e) {
      if (notify) notify(e.message, 'error');
    }
  };

  const rec = report?.recommend || {};
  const dev = report?.device || {};
  const pending = report?.applied?.pending_restart || {};
  const pendingKeys = Object.keys(pending);
  const codecChange = report?.applied?.applied_now?.output_video_codec;
  const workload = report?.workload || {};
  const measured = report?.settings_measured || {};
  const trtTuning = measured.trt_tuning || dev.trt_tuning || {};
  const cpuTuning = measured.cpu_tuning || dev.cpu_tuning || {};
  const current = currentSettings || {};
  const settingMismatch = report?.status === 'success' && report?.version >= 2 && [
    ['provider', current.provider],
    ['swap_model', current.swap_model],
    ['enhancer', current.selected_enhancer],
    ['mask_engine', current.mask_engine],
    ['detector_engine', current.detector_engine],
    ['subsample_upscale', current.subsample_upscale],
    ['perf_trt_pool', current.perf_trt_pool],
    ['perf_detmask_pool', current.perf_detmask_pool],
    ['perf_detector_pool', current.perf_detector_pool],
    ['perf_batch_swap', current.perf_batch_swap],
  ].some(([key, value]) => value && measured[key] && value !== measured[key]);

  return (
    <div className="col-span-full p-4 rounded-xl bg-white/[0.03] border border-white/10 space-y-3 mt-2">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-3 border-b border-white/10">
        <div className="flex items-center gap-3 min-w-0">
          <MotionIcon icon={Icon.cpu} size="md" variant="accent" animate={running ? 'spin' : false} />
          <div className="min-w-0">
            <span className="text-xs font-bold text-white block">Hardware benchmark</span>
            <p className="text-nano text-white/50 mt-0.5">
              Measures the active models and provider. Stress adds a sustained multi-face
              run so short peaks cannot hide memory or thermal instability.
            </p>
            <div className="flex flex-wrap items-center gap-1.5 mt-1.5 text-nano text-white/40">
              <span>Current settings:</span>
              <Chip tone="info">{current.provider || 'provider ?'}</Chip>
              <Chip tone="info">swap {current.swap_model || '?'}</Chip>
              <Chip tone="info">enhancer {current.selected_enhancer || '?'}</Chip>
              <Chip tone="info">mask {current.mask_engine || '?'}</Chip>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <label className="flex items-center gap-1.5 text-nano text-white/45" title="Synthetic faces per benchmark frame; stress requires at least two.">
            <span>Faces/frame</span>
            <input type="number" min={profile === 'stress' ? 2 : 1} max="16" step="0.5"
              value={faces} onChange={(e) => setFaces(e.target.value)} disabled={running}
              className="w-14 rounded-md bg-black/30 border border-white/10 px-1.5 py-1.5 text-right text-white/80 tabular-nums" />
          </label>
          <div className="flex rounded-lg overflow-hidden border border-white/10">
            {[...PROFILES, PROFILE_STRESS].map((p) => (
              <button key={p.id} type="button" title={p.hint}
                onClick={() => setProfile(p.id)} disabled={running}
                aria-pressed={profile === p.id}
                className={`px-2.5 py-1.5 text-nano font-bold transition-colors disabled:opacity-50 ${
                  profile === p.id ? 'state-accent' : 'text-white/45 hover:text-white/80'}`}>
                {p.label}
              </button>
            ))}
          </div>
          <button type="button" onClick={running ? cancel : start}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-micro font-bold transition-all shrink-0 ${
              running ? 'state-danger' : 'fill-accent'}`}>
            <Icon.refresh size={13} className={running ? 'animate-spin' : ''} />
            {running ? 'Cancel' : 'Run benchmark'}
          </button>
        </div>
      </div>

      {running && <LiveProgress status={status} />}

      {/* A report saved by the previous benchmark has the same `status:
          "success"` and the same `best_threads`, but none of the tables below —
          so it would render as a card full of dashes. Say what it is instead.
          Its thread counts are still live in `resolve_threads` until this is
          re-run, which is the reason to re-run it. */}
      {!running && report && report.status === 'success' && report.version !== 2 && (
        <div className="state-warn rounded-lg px-3 py-2 text-nano font-medium flex items-start gap-2">
          <Icon.warning size={12} className="mt-0.5 shrink-0" />
          <span>
            Saved by the old benchmark, which measured a synthetic tensor workload
            rather than this pipeline — its {report.best_threads?.standard ?? '?'}/
            {report.best_threads?.enhanced ?? '?'}/{report.best_threads?.heavy ?? '?'} thread
            profile is still what runs. Re-run to replace it with a measurement.
          </span>
        </div>
      )}

      {!running && report && report.status === 'success' && report.version === 2 && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-micro font-bold ink-accent uppercase tracking-wider truncate">
              {dev.gpu_name || report.gpu_name || 'GPU'} · {dev.total_vram_gb ?? report.total_vram_gb} GB
              {dev.cpu_physical ? ` · ${dev.cpu_physical}c/${dev.cpu_logical}t` : ''}
            </span>
            <span className="text-nano text-white/35 tabular-nums">
              {report.profile} profile · {num(report.duration_sec, 0)}s · {report.ran_at}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-2">
            <Tile label="Threads · standard" value={rec.threads?.standard ?? '—'}
              sub={`${num(report.thread_curve?.standard?.[String(rec.threads?.standard)], 1)} f/s`} tone="accent" />
            <Tile label="Threads · enhanced" value={rec.threads?.enhanced ?? '—'}
              sub={`${num(report.thread_curve?.enhanced?.[String(rec.threads?.enhanced)], 1)} f/s`} tone="accent" />
            <Tile label="Threads · heavy" value={rec.threads?.heavy ?? '—'}
              sub={`${num(report.thread_curve?.heavy?.[String(rec.threads?.heavy)], 1)} f/s`} tone="accent" />
            <Tile label="Swapper pool" value={rec.pools?.trt_pool ?? '—'} sub="ROOP_TRT_POOL" />
            <Tile label="Detect/mask pool" value={rec.pools?.detmask_pool ?? '—'} sub="ROOP_DETMASK_POOL" />
            <Tile label="Provider"
              value={rec.provider?.recommend || dev.provider || '—'}
              sub={!rec.provider ? 'not compared'
                : rec.provider.stages_compared === 0
                  // Every stage was refused by the CUDA EP, so there is no
                  // margin to quote — and "—% over CUDA" would read as a
                  // missing measurement rather than the emphatic result it is.
                  ? `CUDA refused all ${rec.provider.stages_cuda_refused} stages`
                  : `${num(rec.provider.margin_pct, 0)}% over CUDA`
                    + (rec.provider.stages_cuda_refused
                      ? ` · ${rec.provider.stages_cuda_refused} refused` : '')} />
            {/* The encoder used to be measured and then never shown, so the one
                table with a 4x spread in it had no recommendation attached. It
                is deliberately NOT the fastest row: encoding streams alongside
                the swap, so anything already ahead of the pipeline is free
                speed nobody collects, paid for in file size. */}
            <Tile label="Encoder" value={rec.encoder || '—'}
              title={rec.encoder_reason}
              sub={rec.encoder_fastest && rec.encoder_fastest !== rec.encoder
                ? `${rec.encoder_fastest} is faster, not needed`
                : 'fastest measured'} />
          </div>

          <div className="flex flex-wrap items-center gap-1.5 text-nano text-white/45">
            <span className="text-white/60 font-semibold">Measured workload:</span>
            <Chip tone="info">{workload.kind || 'calibration'}</Chip>
            <Chip tone="info">{num(workload.faces_per_frame, 1)} faces/frame</Chip>
            {workload.composite_reps > 1 && <Chip tone="info">{workload.composite_reps} composite repeats</Chip>}
            {workload.measure_sec && <Chip tone="info">{num(workload.measure_sec, 0)}s timed pass</Chip>}
            {trtTuning.builder_optimization_level != null && <Chip tone="info">
              TRT b{trtTuning.builder_optimization_level} · aux {trtTuning.auxiliary_streams}
              {trtTuning.cuda_graph ? ' · CUDA graph' : ''}
            </Chip>}
            {cpuTuning.opencv_kernel_threads != null && <Chip tone="info">
              CPU OpenCV {cpuTuning.opencv_kernel_threads} · ORT block {cpuTuning.ort_dynamic_block_base}
            </Chip>}
          </div>

          {settingMismatch && (
            <div className="state-warn rounded-lg px-3 py-2 text-nano font-medium flex items-start gap-2">
              <Icon.warning size={12} className="mt-0.5 shrink-0" />
              <span>This saved benchmark was measured with different model settings. Re-run it before applying recommendations.</span>
            </div>
          )}

          {rec.encoder_reason && (
            <p className="text-nano text-white/40 leading-relaxed">
              <b className="text-white/60">Encoder:</b> {rec.encoder_reason}.
            </p>
          )}

          {/* The codec is the one applied setting that changes the OUTPUT FILE
              rather than the speed it is produced at, so it is called out by
              name and separately from the restart list — a change of container
              codec discovered later, in a finished render, is not a good way to
              find out. It is live immediately: `_run_swap` re-reads it from
              config on every run, unlike the env-backed knobs below. */}
          {codecChange && (
            <div className="state-info rounded-lg px-3 py-2 text-nano font-medium flex items-start gap-2">
              <Icon.info size={12} className="mt-0.5 shrink-0" />
              <span>
                Video codec switched to <b>{codecChange}</b> — live from your next
                render, no restart. Change it back in Settings → Output if you
                would rather keep the previous one.
              </span>
            </div>
          )}

          {pendingKeys.length > 0 && (
            <div className="state-warn rounded-lg px-3 py-2 text-nano font-medium flex items-start gap-2">
              <Icon.warning size={12} className="mt-0.5 shrink-0" />
              <span>
                Saved, but read at startup — <b>restart the app</b> for {pendingKeys.join(', ')}.
                Thread counts are already live.
              </span>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-2">
            {['standard', 'enhanced', 'heavy'].map((m) => (
              <div key={m} className="p-2.5 rounded-lg bg-black/25 border border-white/8">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-nano font-bold text-white/55 uppercase tracking-wide">{m}</span>
                  <Chip tone="accent">{rec.threads?.[m]} threads</Chip>
                </div>
                <Curve points={report.thread_curve?.[m]} pick={rec.threads?.[m]} />
                <span className="text-nano text-white/25 block mt-0.5">frames/s vs worker threads</span>
              </div>
            ))}
          </div>

          <Foldout title="Per-stage cost" count={`${report.stages?.length || 0} stages`} defaultOpen>
            <StageTable stages={report.stages} />
          </Foldout>

          {report.provider_ab?.length > 0 && (
            <Foldout title="TensorRT vs CUDA" count={`${report.provider_ab.length} models`}>
              <table className="w-full text-nano tabular-nums">
                <thead className="text-white/35">
                  <tr><th className="text-left font-bold py-1">stage</th>
                    <th className="text-right font-bold">TRT ms</th>
                    <th className="text-right font-bold">CUDA ms</th>
                    <th className="text-right font-bold">TRT gain</th></tr>
                </thead>
                <tbody className="text-white/70">
                  {report.provider_ab.map((r) => (
                    <tr key={r.stage} className="border-t border-white/5">
                      <td className="py-1 pr-2">{r.stage}</td>
                      <td className="text-right">{num(r.trt_ms, 2)}</td>
                      <td className="text-right" colSpan={r.error ? 2 : 1}>
                        {r.error
                          ? <span className="ink-warn">will not run on CUDA here — {r.error}</span>
                          : (
                            <>
                              {num(r.cuda_ms, 2)}
                              {r.cuda_fell_back && (
                                <span className="ink-warn" title={`onnxruntime dropped the CUDA provider for this model and ran it on ${r.cuda_provider}, so this is not a CUDA time`}>
                                  {' '}({r.cuda_provider})
                                </span>
                              )}
                            </>
                          )}
                      </td>
                      {!r.error && (
                        <td className={`text-right font-bold ${r.trt_speedup >= 1 ? 'ink-ok' : 'ink-danger'}`}>
                          {num(r.trt_speedup, 2)}×
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Foldout>
          )}

          {(report.io?.encode?.length > 0 || report.io?.decode?.length > 0) && (
            <Foldout title="Encode / decode">
              <table className="w-full text-nano tabular-nums">
                <tbody className="text-white/70">
                  {(report.io.encode || []).map((r) => (
                    <tr key={'e' + r.name} className="border-t border-white/5">
                      <td className="py-1">encode</td><td className="pr-2">{r.name}</td>
                      <td className="text-right font-bold">{num(r.fps, 1)} fps</td>
                      <td className="text-right text-white/40">{num(r.mb, 1)} MB</td>
                    </tr>
                  ))}
                  {(report.io.decode || []).map((r) => (
                    <tr key={'d' + r.name} className="border-t border-white/5">
                      <td className="py-1">decode</td><td className="pr-2">{r.name}</td>
                      <td className="text-right font-bold">{num(r.fps, 1)} fps</td><td />
                    </tr>
                  ))}
                </tbody>
              </table>
              {(report.io.notes || []).map((n, i) => (
                <p key={i} className="text-nano text-white/35 mt-1">{n}</p>
              ))}
            </Foldout>
          )}

          {report.pool_ab?.knob && (
            <p className="text-nano text-white/40">
              <b className="text-white/60">Pool check:</b>{' '}
              {report.pool_ab.reason
                ? `${report.pool_ab.knob} was left at ${rec.pools?.[report.pool_ab.knob]} — ${report.pool_ab.reason}` +
                  (report.pool_ab.needed_gb ? ` (needs ~${report.pool_ab.needed_gb} GB, ${report.pool_ab.free_gb} GB free)` : '')
                : `${report.pool_ab.knob} ${report.pool_ab.knee_pools?.[report.pool_ab.knob]} → ${report.pool_ab.wider_pools?.[report.pool_ab.knob]} on a whole frame at ` +
                  `${report.pool_ab.threads} threads: ${num(report.pool_ab.knee_fps, 2)} vs ${num(report.pool_ab.wider_fps, 2)} f/s — kept ` +
                  `${report.pool_ab.winner === 'wider' ? report.pool_ab.wider_pools?.[report.pool_ab.knob] : report.pool_ab.knee_pools?.[report.pool_ab.knob]}.`}
              {' '}A stage can plateau on its own and still pay off in a frame by staying off the global lock, so the knee is checked end to end.
            </p>
          )}

          {rec.serialised_stages?.length > 0 && (
            <p className="text-nano text-white/40 leading-relaxed">
              <b className="ink-warn">Serialised under TensorRT:</b>{' '}
              {rec.serialised_stages.join(', ')} — these have no session pool, so each takes
              the global GPU lock and blocks every other stage while it runs. That is why the
              thread curve flattens where it does; picking a pooled alternative (CodeFormer or
              RestoreFormer++ rather than GPEN/GFPGAN) is worth more than more threads.
            </p>
          )}

          {report.composite_metrics && Object.keys(report.composite_metrics).length > 0 && (
            <Foldout title="Sustained stability" count="repeated composite runs">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                {Object.entries(report.composite_metrics).map(([mode, points]) => {
                  const rows = Object.entries(points || {});
                  // Prefer the recommended thread point; object order is not a
                  // reliable indication of the selected workload.
                  const widest = points?.[String(rec.threads?.[mode])] || rows[rows.length - 1]?.[1];
                  return (
                    <Tile key={mode} label={`${mode} stability`} value={widest ? `${num(widest.median_fps, 1)} f/s` : 'â€”'}
                      sub={widest ? `${num(widest.min_fps, 1)}–${num(widest.max_fps, 1)} · spread ${num(widest.spread_pct, 0)}%` : 'No repeated runs'} />
                  );
                })}
              </div>
              <p className="text-nano text-white/35 mt-2">Thread recommendations use the median repeated run; spread exposes thermal throttling, paging, or scheduler instability.</p>
            </Foldout>
          )}

          {report.warnings?.length > 0 && (
            <p className="text-nano text-white/35">Not measured: {report.warnings.join('; ')}</p>
          )}
        </div>
      )}

      {!running && !report && (
        <p className="text-nano text-white/35">
          Never run on this machine. Until it is, thread counts come from a VRAM
          heuristic and the pools from a table of VRAM tiers.
        </p>
      )}
    </div>
  );
}

function StageTable({ stages }) {
  if (!stages?.length) return null;
  const total = stages.reduce((a, s) => a + (s.ms_frame || 0), 0) || 1;
  return (
    <table className="w-full text-nano tabular-nums">
      <thead className="text-white/35">
        <tr>
          <th className="text-left font-bold py-1">stage</th>
          <th className="text-right font-bold">ms/call</th>
          <th className="text-right font-bold">×/frame</th>
          <th className="text-right font-bold">ms/frame</th>
          <th className="text-left font-bold pl-2">share</th>
          <th className="text-right font-bold">pool</th>
          <th className="text-right font-bold">MB each</th>
        </tr>
      </thead>
      <tbody className="text-white/70">
        {stages.map((s) => (
          <tr key={s.key} className="border-t border-white/5">
            <td className="py-1 pr-2">
              <span className="text-white/80">{s.label}</span>
              {s.error
                ? <span className="ink-danger ml-1">· {s.error}</span>
                : !s.pooled && <span className="ink-warn ml-1">· no pool</span>}
              {/* The scaling curve is the evidence for the pool column beside
                  it. Without it "pool 2" is an assertion; with it you can see
                  that 4 measured no faster, or that it measured slower. */}
              {s.scaling?.length > 1 && (
                <span className="block text-white/30">
                  {s.scaling.map((r) => `${r.n}×${num(r.calls_s, 0)}`).join(' → ')} calls/s
                </span>
              )}
            </td>
            <td className="text-right">{num(s.ms_call, 2)}</td>
            <td className="text-right text-white/40">{num(s.calls_per_frame, 2)}</td>
            <td className="text-right font-bold text-white">{num(s.ms_frame, 1)}</td>
            <td className="pl-2 w-24">
              <div className="h-1.5 rounded-full bg-white/8 overflow-hidden">
                <div className="h-full rounded-full"
                  style={{ width: `${Math.min(100, ((s.ms_frame || 0) / total) * 100)}%`,
                    background: 'var(--accent)' }} />
              </div>
            </td>
            <td className="text-right">{s.best_n}</td>
            <td className="text-right text-white/40">{num(s.vram_per_instance_mb, 0)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function LiveProgress({ status }) {
  const pct = Math.min(100, Math.max(0, status?.progress || 0));
  const el = status?.elapsed_sec || 0;
  const tot = status?.total_sec || 0;
  const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between text-micro gap-2">
        <span className="font-semibold text-white/70 truncate">{status?.status_msg || 'Starting…'}</span>
        <span className="font-mono text-white/45 shrink-0">{mmss(el)} / ~{mmss(tot)}</span>
      </div>
      <div className="h-2 w-full bg-black/50 rounded-full overflow-hidden border border-white/10">
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: 'var(--accent)' }} />
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <Tile label="Stage" value={status?.current_mode || '—'} />
        <Tile label="Concurrency" value={`${status?.current_threads || 0}×`} />
        <Tile label="Throughput" value={num(status?.current_fps, 1)} sub="calls or frames / s" />
        <Tile label="VRAM free" value={`${num(status?.current_vram_gb, 1)} GB`} />
      </div>
      <div className="h-40 bg-black/70 border border-white/10 rounded-lg p-2.5 font-mono text-nano text-white/60 overflow-y-auto flex flex-col-reverse">
        <div>
          {(status?.logs || []).map((l, i) => (
            <div key={i} className="whitespace-pre-wrap leading-relaxed">{l}</div>
          ))}
          {!status?.logs?.length && <div className="text-white/25 italic">Loading models…</div>}
        </div>
      </div>
    </div>
  );
}
