import React, { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON } from '../api';
import { AnimatedNumber, Button, Section } from './ui';
import { Icon } from '../icons';

const WORKLOAD_OPTIONS = [
  { value: '1', mode: 'solo', label: '1 Face (Solo)', desc: 'Single face swap pipeline throughput' },
  { value: '2', mode: 'duo', label: '2 Faces (Duo)', desc: 'Dual concurrent tracking and swap load' },
  { value: '3', mode: 'crowd', label: '3+ Faces (Crowd)', desc: 'Multi-face detection, sorting & memory stress' },
];

const MODE_OPTIONS = [
  { value: 'quick', label: 'Quick Benchmark (~30s)', frames: 90, desc: 'Fast optimization probe for rapid iteration' },
  { value: 'full', label: 'Full Evaluation (~90s)', frames: 300, desc: 'Sustained pacing, frame jitter & thermal retention' },
];

const BADGE_STYLES = {
  good: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400',
  warn: 'bg-amber-500/15 border-amber-500/30 text-amber-300',
  critical: 'bg-rose-500/15 border-rose-500/30 text-rose-400',
  neutral: 'bg-blue-500/15 border-blue-500/30 text-blue-300',
};

function formatTimestamp(isoStr) {
  if (!isoStr) return 'Unknown';
  try {
    const d = new Date(isoStr);
    return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
  } catch {
    return String(isoStr);
  }
}

export default function BenchmarkPanel({ notify, onSettingsApplied }) {
  const [activeTab, setActiveTab] = useState('runner'); // 'runner' | 'history'
  const [prompt, setPrompt] = useState(null);
  const [_loadingPrompt, setLoadingPrompt] = useState(true);
  const [selectedFaces, setSelectedFaces] = useState('1');
  const [selectedMode, setSelectedMode] = useState('quick');
  const [allowLossy, setAllowLossy] = useState(false);

  // Active run and progress state
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [selectedPresetKey, setSelectedPresetKey] = useState('balanced');
  const [busyAction, setBusyAction] = useState(null); // 'start' | 'cancel' | 'apply' | 'decline' | 'revert'
  const [error, setError] = useState('');

  // Stored profiles
  const [profiles, setProfiles] = useState([]);
  const [loadingProfiles, setLoadingProfiles] = useState(false);

  const pollRef = useRef(null);

  // Fetch initial prompt and status
  const fetchPrompt = useCallback(async () => {
    setLoadingPrompt(true);
    try {
      const data = await getJSON('/api/benchmark/prompt', { timeout: 15000 });
      setPrompt(data);
      if (data?.default_faces) setSelectedFaces(data.default_faces);
      if (data?.default_mode) setSelectedMode(data.default_mode);
      setError('');
    } catch (err) {
      setError(err.message || 'Benchmark service unavailable');
    } finally {
      setLoadingPrompt(false);
    }
  }, []);

  // Fetch existing or completed result
  const fetchResult = useCallback(async () => {
    try {
      const res = await getJSON('/api/benchmark/result', { timeout: 15000 });
      if (res && res.ready) {
        setResult(res);
      }
    } catch {
      // not ready or idle
    }
  }, []);

  // Fetch saved history profiles
  const fetchProfiles = useCallback(async () => {
    setLoadingProfiles(true);
    try {
      const data = await getJSON('/api/benchmark/profiles?limit=20', { timeout: 15000 });
      setProfiles(data?.profiles || []);
    } catch {
      setProfiles([]);
    } finally {
      setLoadingProfiles(false);
    }
  }, []);

  useEffect(() => {
    fetchPrompt();
    fetchResult();
    fetchProfiles();
  }, [fetchPrompt, fetchResult, fetchProfiles]);

  // Clean up poll on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Polling loop for active benchmark
  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const snap = await getJSON('/api/benchmark/progress', { timeout: 5000 });
        setProgress(snap);
        if (snap.running) {
          setRunning(true);
        } else if (snap.done || snap.cancelled || snap.error || snap.phase === 'complete' || snap.phase === 'failed') {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setRunning(false);
          setBusyAction(null);

          if (snap.error) {
            setError(snap.error);
            notify?.(snap.error, 'error');
          } else if (snap.done || snap.phase === 'complete') {
            notify?.('Benchmark completed successfully', 'success');
            fetchResult();
            fetchProfiles();
          }
        }
      } catch {
        // Polling blip
      }
    }, 750);
  }, [fetchResult, fetchProfiles, notify]);

  // Start benchmark handler
  const handleStart = async () => {
    setError('');
    setBusyAction('start');
    try {
      const resp = await postJSON('/api/benchmark/start', {
        faces: selectedFaces,
        mode: selectedMode,
        persist: true,
      });

      if (resp?.status === 'started') {
        setRunning(true);
        setResult(null);
        setProgress({
          running: true,
          phase: 'prepare',
          status: 'Initializing engine & loading models...',
          frame: 0,
          total_frames: resp.frames || 90,
          progress_pct: 0,
          current_fps: 0,
          average_fps: 0,
        });
        startPolling();
      } else {
        throw new Error(resp?.message || 'Could not start benchmark');
      }
    } catch (err) {
      setError(err.message || 'Failed to start benchmark');
      notify?.(err.message || 'Failed to start benchmark', 'error');
      setBusyAction(null);
      setRunning(false);
    }
  };

  // Cancel handler
  const handleCancel = async () => {
    setBusyAction('cancel');
    try {
      await postJSON('/api/benchmark/cancel', {});
      notify?.('Benchmark cancel requested', 'info');
    } catch (err) {
      notify?.(err.message || 'Could not cancel benchmark', 'error');
    } finally {
      setBusyAction(null);
    }
  };

  // Apply recommendation handler
  const handleApply = async (customSettings = null) => {
    setBusyAction('apply');
    try {
      const toApply = customSettings || (result?.presets?.[selectedPresetKey]?.recommended_settings) || result?.recommended_settings;
      const resp = await postJSON('/api/benchmark/apply', {
        recommended_settings: toApply,
        run_id: result?.run_id,
        allow_lossy_temp_frames: allowLossy,
      });

      if (resp?.status === 'applied') {
        notify?.(resp.message || 'Recommended settings applied to configuration', 'success');
        setResult((prev) => (prev ? { ...prev, applied: true } : prev));
        fetchProfiles();
        onSettingsApplied?.();
      } else {
        throw new Error(resp?.message || 'Failed to apply settings');
      }
    } catch (err) {
      notify?.(err.message || 'Failed to apply settings', 'error');
    } finally {
      setBusyAction(null);
    }
  };

  // Decline recommendation handler
  const handleDecline = async () => {
    setBusyAction('decline');
    try {
      const resp = await postJSON('/api/benchmark/decline', { run_id: result?.run_id });
      notify?.(resp.message || 'Recommendation declined; current settings preserved', 'info');
      setResult((prev) => (prev ? { ...prev, applied: false } : prev));
      fetchProfiles();
    } catch (err) {
      notify?.(err.message || 'Decline failed', 'error');
    } finally {
      setBusyAction(null);
    }
  };

  // Revert to factory defaults handler
  const handleRevert = async () => {
    setBusyAction('revert');
    try {
      const resp = await postJSON('/api/benchmark/revert', {});
      notify?.(resp.message || 'Reverted benchmark-owned settings to defaults', 'success');
      fetchResult();
      fetchProfiles();
      onSettingsApplied?.();
    } catch (err) {
      notify?.(err.message || 'Failed to revert settings', 'error');
    } finally {
      setBusyAction(null);
    }
  };

  // Apply a historical profile
  const handleApplyProfile = async (profile) => {
    try {
      const resp = await postJSON('/api/benchmark/profiles/apply', {
        run_id: profile.run_id,
        recommended_settings: profile.recommended_settings,
        allow_lossy_temp_frames: allowLossy,
      });
      notify?.(resp.message || `Applied profile ${profile.run_id}`, 'success');
      fetchProfiles();
      onSettingsApplied?.();
    } catch (err) {
      notify?.(err.message || 'Could not apply profile', 'error');
    }
  };



  return (
    <div id="benchmark-panel" className="w-full mb-6">
      <Section
        title="Hardware Benchmark & Auto-Tune"
        icon={Icon.meter}
        collapsible
        defaultOpen={true}
        action={
          <div className="flex items-center gap-2">
            <div className="flex rounded-xl bg-white/[0.04] p-1 border border-white/10 text-mini">
              <button
                type="button"
                id="benchmark-tab-runner"
                onClick={() => setActiveTab('runner')}
                className={`px-3 py-1 rounded-lg transition-colors font-semibold ${
                  activeTab === 'runner' ? 'bg-[var(--accent)] text-white' : 'text-white/60 hover:text-white'
                }`}
              >
                Benchmark Suite
              </button>
              <button
                type="button"
                id="benchmark-tab-history"
                onClick={() => {
                  setActiveTab('history');
                  fetchProfiles();
                }}
                className={`px-3 py-1 rounded-lg transition-colors font-semibold flex items-center gap-1.5 ${
                  activeTab === 'history' ? 'bg-[var(--accent)] text-white' : 'text-white/60 hover:text-white'
                }`}
              >
                <Icon.history size={12} />
                <span>Profiles ({profiles.length})</span>
              </button>
            </div>
          </div>
        }
      >
        {/* Error notification banner if any */}
        {error && (
          <div id="benchmark-error-banner" className="flex items-start gap-2.5 p-3 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs">
            <Icon.warning size={16} className="shrink-0 mt-0.5" />
            <div className="flex-1 leading-relaxed">{error}</div>
          </div>
        )}

        {/* ── TAB 1: RUNNER & DASHBOARD ──────────────────────────────── */}
        {activeTab === 'runner' && (
          <div className="space-y-4">
            {/* Top Overview Bar: Hardware Profile & Active Models */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Hardware context */}
              <div id="benchmark-hardware-card" className="p-3.5 rounded-xl bg-white/[0.02] border border-white/10 flex flex-col justify-between">
                <div>
                  <div className="text-micro font-semibold uppercase tracking-wider text-white/40 mb-1 flex items-center gap-1.5">
                    <Icon.cpu size={12} className="text-[var(--accent)]" />
                    <span>Target Hardware Profile</span>
                  </div>
                  <div className="text-sm font-semibold text-white/90">
                    {result?.device?.gpu_name || prompt?.gpu_name || 'NVIDIA Acceleration Tier'}
                  </div>
                  <div className="text-mini text-white/50 mt-0.5">
                    VRAM: {result?.device?.vram_total_mb ? `${Math.round(result.device.vram_total_mb / 1024)} GB` : 'Hardware Autodetect'} · Cores: {result?.device?.cpu_logical_cores || 'Multi-core'}
                  </div>
                </div>
                <div className="mt-3 flex items-center gap-2">
                  <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-nano font-bold uppercase tracking-wide border ${
                    running ? 'bg-amber-500/15 border-amber-500/30 text-amber-300' :
                    result ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300' :
                    'bg-white/10 border-white/15 text-white/60'
                  }`}>
                    <span className={`h-1.5 w-1.5 rounded-full ${running ? 'bg-amber-400 animate-pulse' : result ? 'bg-emerald-400' : 'bg-white/40'}`} />
                    {running ? 'Benchmarking in progress' : result ? 'Evaluation Ready' : 'Idle & Ready'}
                  </span>
                </div>
              </div>

              {/* Active Pipeline Models */}
              <div id="benchmark-models-card" className="p-3.5 rounded-xl bg-white/[0.02] border border-white/10">
                <div className="text-micro font-semibold uppercase tracking-wider text-white/40 mb-1.5 flex items-center gap-1.5">
                  <Icon.wand size={12} className="text-[var(--accent)]" />
                  <span>Pipeline Models Locked For Test</span>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between py-0.5 border-b border-white/[0.04]">
                    <span className="text-white/40">Face Swapper</span>
                    <span className="font-mono text-white/85 font-medium">{prompt?.active_models?.swapper || result?.active_models?.swapper || 'Inswapper / RealSwap'}</span>
                  </div>
                  <div className="flex justify-between py-0.5 border-b border-white/[0.04]">
                    <span className="text-white/40">Enhancer / Restorer</span>
                    <span className="font-mono text-white/85 font-medium">{prompt?.active_models?.enhancer || result?.active_models?.enhancer || 'None / GPEN'}</span>
                  </div>
                  <div className="flex justify-between py-0.5">
                    <span className="text-white/40">Mask & Alignment</span>
                    <span className="font-mono text-white/85 font-medium">{prompt?.active_models?.mask_engine || result?.active_models?.mask_engine || 'RealityUX / Box'}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Warnings callout from backend */}
            {prompt?.warnings && prompt.warnings.length > 0 && !running && (
              <div id="benchmark-warnings-box" className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/25 space-y-1">
                <div className="text-micro font-bold uppercase tracking-wider text-amber-300 flex items-center gap-1.5">
                  <Icon.warning size={14} />
                  <span>Pre-Benchmark Notice</span>
                </div>
                {prompt.warnings.map((w, i) => (
                  <p key={i} className="text-xs text-amber-200/90 leading-relaxed">• {w}</p>
                ))}
              </div>
            )}

            {/* Configuration & Trigger Controls (when not actively measuring) */}
            {!running && (
              <div id="benchmark-setup-controls" className="p-4 rounded-xl bg-white/[0.03] border border-white/10 space-y-3.5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div>
                    <h4 className="text-sm font-semibold text-white/90">Benchmark Configuration</h4>
                    <p className="text-xs text-white/50">Simulates real video frame processing on isolated synthetic workloads to measure true sustained FPS and hardware limits.</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      id="benchmark-start-btn"
                      variant="primary"
                      onClick={handleStart}
                      disabled={busyAction === 'start' || prompt?.can_run === false}
                      className="whitespace-nowrap flex items-center gap-1.5 px-5 py-2"
                    >
                      <Icon.play size={14} />
                      <span>{busyAction === 'start' ? 'Starting...' : 'Run Benchmark'}</span>
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                  {/* Face Complexity Choice */}
                  <div>
                    <label className="block text-xs font-medium text-white/70 mb-1.5">Workload Complexity (Target Faces)</label>
                    <div className="grid grid-cols-3 gap-2">
                      {WORKLOAD_OPTIONS.map((opt) => (
                        <button
                          key={opt.value}
                          type="button"
                          id={`benchmark-workload-${opt.value}`}
                          onClick={() => setSelectedFaces(opt.value)}
                          className={`px-3 py-2.5 rounded-xl border text-left transition-all ${
                            selectedFaces === opt.value
                              ? 'bg-[var(--accent)]/15 border-[var(--accent)] text-white shadow-sm'
                              : 'bg-white/[0.03] border-white/10 text-white/60 hover:border-white/20 hover:text-white/80'
                          }`}
                        >
                          <div className="text-xs font-bold">{opt.label}</div>
                          <div className="text-nano text-white/45 mt-0.5 line-clamp-1">{opt.desc}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Mode / Duration Choice */}
                  <div>
                    <label className="block text-xs font-medium text-white/70 mb-1.5">Evaluation Duration</label>
                    <div className="grid grid-cols-2 gap-2">
                      {MODE_OPTIONS.map((m) => (
                        <button
                          key={m.value}
                          type="button"
                          id={`benchmark-mode-${m.value}`}
                          onClick={() => setSelectedMode(m.value)}
                          className={`px-3 py-2.5 rounded-xl border text-left transition-all ${
                            selectedMode === m.value
                              ? 'bg-[var(--accent)]/15 border-[var(--accent)] text-white shadow-sm'
                              : 'bg-white/[0.03] border-white/10 text-white/60 hover:border-white/20 hover:text-white/80'
                          }`}
                        >
                          <div className="text-xs font-bold">{m.label}</div>
                          <div className="text-nano text-white/45 mt-0.5">{m.frames} frames measured</div>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── ACTIVE BENCHMARK RUN TELEMETRY HUD ────────────────────────── */}
            {running && progress && (
              <div id="benchmark-active-hud" className="p-4 rounded-xl bg-black/40 border border-white/15 backdrop-blur-md space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--accent)] opacity-75" />
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-[var(--accent)]" />
                    </span>
                    <span className="text-sm font-bold text-white tracking-wide">
                      {progress.phase === 'prepare' ? 'Preparing Workload & Allocating Contexts...' : 'Measuring Execution Pipeline...'}
                    </span>
                  </div>
                  <Button
                    id="benchmark-cancel-btn"
                    variant="secondary"
                    size="sm"
                    onClick={handleCancel}
                    disabled={busyAction === 'cancel'}
                    className="flex items-center gap-1.5 text-rose-300 hover:text-rose-200 border-rose-500/30"
                  >
                    <Icon.stop size={12} />
                    <span>Cancel</span>
                  </Button>
                </div>

                {/* Progress Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono text-white/70">
                    <span>Frame {progress.frame || 0} / {progress.total_frames || 90}</span>
                    <span className="font-bold text-white">{Math.round(progress.progress_pct || 0)}%</span>
                  </div>
                  <div className="w-full h-2.5 rounded-full bg-white/10 overflow-hidden relative">
                    <div
                      className="h-full bg-gradient-to-r from-[var(--accent)] to-cyan-400 transition-all duration-300 rounded-full"
                      style={{ width: `${Math.min(100, Math.max(0, progress.progress_pct || 0))}%` }}
                    />
                  </div>
                </div>

                {/* Live Real-time Telemetry Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                  <div className="p-3 rounded-xl bg-white/[0.04] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">Live Throughput</div>
                    <div className="text-lg font-mono font-bold text-white mt-1">
                      <AnimatedNumber value={progress.current_fps || 0} decimals={1} suffix=" FPS" />
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">Average: {Number(progress.average_fps || 0).toFixed(1)} FPS</div>
                  </div>

                  <div className="p-3 rounded-xl bg-white/[0.04] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">VRAM Footprint</div>
                    <div className="text-lg font-mono font-bold text-cyan-400 mt-1">
                      {progress.vram_used_mb ? `${Math.round(progress.vram_used_mb)} MB` : 'Monitoring...'}
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">
                      {progress.vram_pct ? `${Math.round(progress.vram_pct)}% of card` : 'Allocation steady'}
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-white/[0.04] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">GPU Load</div>
                    <div className="text-lg font-mono font-bold text-emerald-400 mt-1">
                      {progress.gpu_pct != null ? `${Math.round(progress.gpu_pct)}%` : 'Active'}
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">Engine saturation</div>
                  </div>

                  <div className="p-3 rounded-xl bg-white/[0.04] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">Time Remaining</div>
                    <div className="text-lg font-mono font-bold text-white/90 mt-1">
                      {progress.eta_sec != null ? `${Math.ceil(progress.eta_sec)}s` : 'Calculating...'}
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">Elapsed: {Math.floor(progress.elapsed_sec || 0)}s</div>
                  </div>
                </div>

                {/* Status message stream */}
                {progress.status && (
                  <div className="text-xs text-white/60 font-mono bg-white/[0.02] px-3 py-2 rounded-lg border border-white/5 truncate">
                    &gt; {progress.status}
                  </div>
                )}
              </div>
            )}

            {/* ── DASHBOARD REPORT (FINISHED EVALUATION) ───────────────────── */}
            {result && !running && (
              <div id="benchmark-results-dashboard" className="space-y-4 pt-2">
                {/* Score & Verdict Banner */}
                <div className="p-4 rounded-xl bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/15 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                  <div className="flex items-center gap-4">
                    {/* Score Circle */}
                    <div className="h-16 w-16 rounded-2xl bg-black/50 border border-white/15 flex flex-col items-center justify-center shrink-0 shadow-lg">
                      <span className="text-2xl font-black text-white font-mono leading-none">{result.score || 0}</span>
                      <span className="text-nano text-white/40 font-semibold uppercase mt-0.5">Score</span>
                    </div>

                    <div>
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold border ${
                          BADGE_STYLES[result.badge_tone] || BADGE_STYLES.neutral
                        }`}>
                          {result.badge || 'Evaluated'}
                        </span>
                        <span className="text-xs text-white/40 font-mono">
                          {formatTimestamp(result.timestamp)}
                        </span>
                        {result.applied && (
                          <span className="px-2 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-nano font-bold uppercase">
                            Applied
                          </span>
                        )}
                      </div>
                      <p className="text-sm font-medium text-white/80 leading-snug">
                        {result.badge_detail || result.bottleneck || 'Optimal balance achieved.'}
                      </p>
                    </div>
                  </div>

                  {/* Quick Action Buttons */}
                  <div className="flex items-center gap-2 w-full md:w-auto justify-end">
                    <Button
                      id="benchmark-apply-btn"
                      variant="primary"
                      onClick={() => handleApply()}
                      disabled={busyAction === 'apply'}
                      className="px-4 py-2 flex items-center gap-1.5"
                    >
                      <Icon.done size={14} />
                      <span>{busyAction === 'apply' ? 'Applying...' : 'Apply Recommended'}</span>
                    </Button>
                    <Button
                      id="benchmark-decline-btn"
                      variant="secondary"
                      onClick={handleDecline}
                      disabled={busyAction === 'decline'}
                      className="px-3 py-2 text-white/60 hover:text-white"
                      title="Keep existing settings"
                    >
                      <span>Decline</span>
                    </Button>
                    <Button
                      id="benchmark-revert-btn"
                      variant="secondary"
                      onClick={handleRevert}
                      disabled={busyAction === 'revert'}
                      className="px-3 py-2 text-white/60 hover:text-white"
                      title="Revert to factory stock defaults"
                    >
                      <Icon.reset size={13} />
                    </Button>
                  </div>
                </div>

                {/* Key Telemetry Metrics Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">Average FPS</div>
                    <div className="text-lg font-mono font-bold text-white mt-1">
                      {Number(result.average_fps || 0).toFixed(2)}
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">Sustained rate</div>
                  </div>

                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">1% Low FPS</div>
                    <div className="text-lg font-mono font-bold text-cyan-400 mt-1">
                      {Number(result.p1_low_fps || 0).toFixed(2)}
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">Frame pacing floor</div>
                  </div>

                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">Avg Latency</div>
                    <div className="text-lg font-mono font-bold text-white/90 mt-1">
                      {Math.round(result.avg_latency_ms || 0)} ms
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">Per-frame transit</div>
                  </div>

                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">P99 Latency</div>
                    <div className="text-lg font-mono font-bold text-amber-300 mt-1">
                      {Math.round(result.p99_latency_ms || 0)} ms
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">Worst-case spike</div>
                  </div>

                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">Peak VRAM</div>
                    <div className="text-lg font-mono font-bold text-purple-300 mt-1">
                      {Math.round(result.peak_vram_mb || 0)} MB
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">Max memory hold</div>
                  </div>

                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/10">
                    <div className="text-micro text-white/40 uppercase tracking-wider font-semibold">Thermal Retention</div>
                    <div className={`text-lg font-mono font-bold mt-1 ${
                      result.thermal?.throttling_detected ? 'text-rose-400' : 'text-emerald-400'
                    }`}>
                      {result.thermal?.retention_pct != null ? `${result.thermal.retention_pct}%` : '100%'}
                    </div>
                    <div className="text-nano text-white/40 mt-0.5">
                      {result.thermal?.throttling_detected ? 'Throttling noted' : 'Stable thermals'}
                    </div>
                  </div>
                </div>

                {/* Bottleneck Evidence Checklist (if present) */}
                {result.bottleneck_evidence && result.bottleneck_evidence.length > 0 && (
                  <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/10 space-y-1.5">
                    <div className="text-micro font-bold uppercase tracking-wider text-white/40 flex items-center gap-1.5">
                      <Icon.search size={12} className="text-[var(--accent)]" />
                      <span>Bottleneck Diagnostic Evidence</span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                      {result.bottleneck_evidence.map((item, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-white/70">
                          <span className="text-[var(--accent)] font-bold shrink-0">•</span>
                          <span>{item}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Preset Selector Tabs */}
                {result.presets && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="text-xs font-semibold uppercase tracking-wider text-white/50">Tuning Profiles</div>
                      <label className="flex items-center gap-1.5 text-micro text-white/50 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={allowLossy}
                          onChange={(e) => setAllowLossy(e.target.checked)}
                          className="rounded bg-white/10 border-white/20 text-[var(--accent)] focus:ring-0"
                        />
                        <span>Allow lossy temporary frames (JPEG / NVDEC boost)</span>
                      </label>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                      {[
                        { key: 'balanced', label: 'Balanced (Recommended)', desc: 'Best throughput with safe VRAM headroom and rock-solid stability' },
                        { key: 'max_throughput', label: 'Max Throughput', desc: 'Widest execution thread and pool limits for maximum rendering speed' },
                        { key: 'stable_low_power', label: 'Stable & Cool', desc: 'Lower power and thermal footprint, ideal for laptop workstations' },
                      ].map((p) => {
                        const presetData = result.presets[p.key];
                        if (!presetData) return null;
                        const isSelected = selectedPresetKey === p.key;
                        return (
                          <div
                            key={p.key}
                            onClick={() => setSelectedPresetKey(p.key)}
                            className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                              isSelected
                                ? 'bg-[var(--accent)]/10 border-[var(--accent)] shadow-md ring-1 ring-[var(--accent)]/30'
                                : 'bg-white/[0.02] border-white/10 hover:border-white/20'
                            }`}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className={`text-xs font-bold ${isSelected ? 'text-white' : 'text-white/80'}`}>
                                {p.label}
                              </span>
                              {p.key === 'balanced' && (
                                <span className="px-1.5 py-0.2 rounded text-nano bg-[var(--accent)]/20 text-[var(--accent)] font-semibold uppercase">
                                  Default
                                </span>
                              )}
                            </div>
                            <p className="text-micro text-white/50 leading-relaxed">{p.desc}</p>
                            <div className="mt-2 text-micro font-mono text-white/70 space-y-0.5">
                              <div>Threads: {presetData.threads || presetData.recommended_settings?.execution_threads || 'Auto'}</div>
                              <div>Format: {presetData.temp_format || 'Auto'}</div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Settings Comparison Table */}
                {result.comparison && result.comparison.length > 0 && (
                  <div className="rounded-xl border border-white/10 overflow-hidden bg-white/[0.02]">
                    <div className="px-3.5 py-2.5 bg-white/[0.04] border-b border-white/10 flex items-center justify-between">
                      <span className="text-xs font-bold text-white/80">Settings Comparison & Delta</span>
                      <span className="text-micro text-white/40">Differences highlight tuning improvements</span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead>
                          <tr className="border-b border-white/10 text-white/45 text-micro uppercase tracking-wider">
                            <th className="py-2.5 px-3.5">Setting</th>
                            <th className="py-2.5 px-3.5">Current Value</th>
                            <th className="py-2.5 px-3.5 text-cyan-300">Recommended</th>
                            <th className="py-2.5 px-3.5">Impact / Evidence</th>
                            <th className="py-2.5 px-3.5 text-right">Requires Restart</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/[0.04]">
                          {result.comparison.map((row, i) => (
                            <tr key={i} className={`hover:bg-white/[0.02] ${row.changed ? 'bg-[var(--accent)]/[0.03]' : ''}`}>
                              <td className="py-2 px-3.5 font-medium text-white/90">
                                <div>{row.setting}</div>
                                <div className="text-nano font-mono text-white/40">{row.key}</div>
                              </td>
                              <td className="py-2 px-3.5 font-mono text-white/60">
                                {String(row.current)}
                              </td>
                              <td className="py-2 px-3.5 font-mono font-bold text-cyan-400">
                                {String(row.recommended)}
                              </td>
                              <td className="py-2 px-3.5 text-white/70 text-micro">
                                {row.note || row.evidence || '—'}
                              </td>
                              <td className="py-2 px-3.5 text-right">
                                {row.requires_restart ? (
                                  <span className="inline-flex px-1.5 py-0.5 rounded text-nano bg-amber-500/15 border border-amber-500/30 text-amber-300 font-semibold">
                                    Restart
                                  </span>
                                ) : (
                                  <span className="inline-flex px-1.5 py-0.5 rounded text-nano bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-semibold">
                                    Instant
                                  </span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── TAB 2: PROFILES HISTORY ────────────────────────────────── */}
        {activeTab === 'history' && (
          <div id="benchmark-history-panel" className="space-y-3">
            <div className="flex items-center justify-between text-xs text-white/50 mb-1">
              <span>Saved Optimization Profiles and historical benchmark runs</span>
              <button
                type="button"
                onClick={fetchProfiles}
                className="flex items-center gap-1 text-white/70 hover:text-white"
              >
                <Icon.refresh size={11} />
                <span>Refresh</span>
              </button>
            </div>

            {loadingProfiles ? (
              <div className="py-8 text-center text-xs text-white/40">Loading saved profiles...</div>
            ) : profiles.length === 0 ? (
              <div className="py-8 text-center text-xs text-white/40 border border-dashed border-white/10 rounded-xl">
                No optimization profiles stored yet. Run a benchmark to record baseline hardware metrics.
              </div>
            ) : (
              <div className="space-y-2">
                {profiles.map((p, i) => (
                  <div
                    key={p.run_id || i}
                    className="p-3.5 rounded-xl bg-white/[0.02] border border-white/10 hover:border-white/20 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-white font-mono">{p.score || 0} pts</span>
                        <span className="text-xs font-semibold text-white/80">{p.workload || 'General Workload'}</span>
                        <span className="text-micro text-white/40 font-mono">{formatTimestamp(p.timestamp)}</span>
                        {p.applied && (
                          <span className="px-1.5 py-0.2 rounded text-nano bg-emerald-500/20 text-emerald-300 font-bold uppercase">
                            Applied
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-micro text-white/60 font-mono">
                        <span>Avg: <strong className="text-white">{p.avg_fps || 0} FPS</strong></span>
                        <span>1% Low: <strong className="text-cyan-300">{p.p1_low_fps || 0} FPS</strong></span>
                        <span>Swapper: {p.active_models?.swapper || 'realswap'}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => handleApplyProfile(p)}
                        className="flex items-center gap-1"
                      >
                        <Icon.done size={12} />
                        <span>Apply Profile</span>
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Section>
    </div>
  );
}
