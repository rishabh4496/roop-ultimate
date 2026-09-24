import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  Layers, Play, Pause, Square, Trash2, ArrowUp, ArrowDown, FolderOpen,
  Volume2, VolumeX, Bell, BellOff, Power, ChevronUp, ChevronDown, RefreshCw
} from 'lucide-react';
import { postJSON } from '../../api';
import useQueue, {
  jobState,
  QUEUE_STATE_LABEL,
  QUEUE_STATE_CLASS,
  ACTIVE_STATES,
  TERMINAL_STATES,
} from '../faceswap/useQueue';
import { useTelemetryStore } from '../../store/telemetryStore';

/**
 * High-performance 100-frame rolling moving average ETA tracker.
 */
class Rolling100FrameEtaTracker {
  constructor(windowFrames = 100) {
    this.windowFrames = windowFrames;
    this.samples = [];
  }

  reset() {
    this.samples = [];
  }

  addSample(frame) {
    const now = performance.now();
    const f = Number(frame);
    if (!Number.isFinite(f) || f < 0) return;
    if (this.samples.length && f < this.samples[this.samples.length - 1].frame) {
      this.samples = [];
    }
    this.samples.push({ time: now, frame: f });

    while (this.samples.length > 2 && (f - this.samples[0].frame) > this.windowFrames) {
      this.samples.shift();
    }
  }

  getFps() {
    if (this.samples.length < 2) return null;
    const first = this.samples[0];
    const last = this.samples[this.samples.length - 1];
    const dt = (last.time - first.time) / 1000.0;
    const df = last.frame - first.frame;
    if (dt <= 0 || df <= 0) return null;
    return df / dt;
  }

  getEtaSeconds(totalFrames) {
    const total = Number(totalFrames);
    if (!total || total <= 0 || !this.samples.length) return null;
    const currentFrame = this.samples[this.samples.length - 1].frame;
    const remaining = Math.max(0, total - currentFrame);
    if (remaining === 0) return 0;
    const fps = this.getFps();
    if (!fps || fps <= 0) return null;
    return remaining / fps;
  }
}

/**
 * Synthesize a clean dual-tone render completion chime via Web Audio API.
 * Requires zero external audio files.
 */
function playCompletionChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const now = ctx.currentTime;

    // First Tone: D5 (587.33 Hz)
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(587.33, now);
    gain1.gain.setValueAtTime(0, now);
    gain1.gain.linearRampToValueAtTime(0.25, now + 0.04);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(now);
    osc1.stop(now + 0.35);

    // Second Tone: A5 (880.0 Hz)
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(880.0, now + 0.12);
    gain2.gain.setValueAtTime(0, now + 0.12);
    gain2.gain.linearRampToValueAtTime(0.3, now + 0.16);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.55);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now + 0.12);
    osc2.stop(now + 0.6);
  } catch {
    // Autoplay or audio context permission restrictions
  }
}

/**
 * Format seconds into human readable time (MM:SS or HH:MM:SS).
 */
function formatDuration(sec) {
  if (sec === null || sec === undefined || !Number.isFinite(sec)) return '--:--';
  const total = Math.max(0, Math.round(sec));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  if (h > 0) return `${pad(h)}:${pad(m)}:${pad(s)}`;
  return `${pad(m)}:${pad(s)}`;
}

/**
 * Advanced Render Queue Drawer with 100-frame rolling ETA, multi-job prioritize,
 * post-render audio/desktop notification, and auto-shutdown controls.
 */
export default function RenderQueueDrawer({
  isOpen = true,
  onToggleOpen,
  notify,
  className = '',
}) {
  // ── HOOK DECLARATIONS AT TOP ──────────────────────────────────────────────
  const {
    jobs, running, paused, start, pause, resume, stop,
    clear, remove, reorder, retry, cancel, busy,
  } = useQueue({ notify });

  const [audioNotify, setAudioNotify] = useState(true);
  const [desktopNotify, setDesktopNotify] = useState(false);
  const [autoShutdown, setAutoShutdown] = useState(false);
  const [activeJobDetails, setActiveJobDetails] = useState({
    fps: 0,
    etaSeconds: null,
    currentFrame: 0,
    totalFrames: 0,
    progress: 0,
  });

  const etaTrackerRef = useRef(new Rolling100FrameEtaTracker(100));
  const prevRunningRef = useRef(running);

  // Active / current job
  const activeJob = useMemo(() => {
    return jobs.find((j) => ACTIVE_STATES.includes(jobState(j))) || null;
  }, [jobs]);

  // Request desktop notification permission when toggled
  const handleToggleDesktopNotification = useCallback(async () => {
    if (!desktopNotify) {
      if (typeof window !== 'undefined' && 'Notification' in window) {
        try {
          const perm = await Notification.requestPermission();
          if (perm === 'granted') {
            setDesktopNotify(true);
            if (notify) notify('Desktop notifications enabled', 'success');
          } else {
            if (notify) notify('Notification permission was denied', 'warning');
          }
        } catch {
          if (notify) notify('Notifications not supported in this browser', 'warning');
        }
      }
    } else {
      setDesktopNotify(false);
    }
  }, [desktopNotify, notify]);

  // Telemetry rolling 100-frame ETA calculation
  useEffect(() => {
    const unsub = useTelemetryStore.subscribe((state) => {
      const run = state?.run || {};
      const current = Number(run.current_frame) || 0;
      const total = Number(run.total_frames) || 0;
      const progress = Number(run.progress) || 0;

      if (current > 0) {
        etaTrackerRef.current.addSample(current);
      }

      const rollingFps = etaTrackerRef.current.getFps() || Number(run.fps) || 0;
      const rollingEta = etaTrackerRef.current.getEtaSeconds(total);

      setActiveJobDetails({
        fps: rollingFps,
        etaSeconds: rollingEta,
        currentFrame: current,
        totalFrames: total,
        progress,
      });
    });

    return () => unsub();
  }, []);

  // Post-render completion listener
  useEffect(() => {
    if (prevRunningRef.current && !running && jobs.length > 0) {
      const allDone = jobs.every((j) => TERMINAL_STATES.includes(jobState(j)));
      if (allDone) {
        // Play Audio Chime
        if (audioNotify) {
          playCompletionChime();
        }

        // Send Desktop Notification
        if (desktopNotify && typeof window !== 'undefined' && 'Notification' in window && Notification.permission === 'granted') {
          try {
            new Notification('Render Queue Complete', {
              body: `All ${jobs.length} jobs finished successfully.`,
              icon: '/favicon.ico',
            });
          } catch {
            // Ignored
          }
        }

        // Auto-shutdown if armed
        if (autoShutdown) {
          if (notify) notify('Queue complete: executing auto-shutdown sequence...', 'warning');
          postJSON('/api/system/shutdown', {}).catch(() => {});
        }
      }
    }
    prevRunningRef.current = running;
  }, [running, jobs, audioNotify, desktopNotify, autoShutdown, notify]);

  // Prioritize job: Move Up
  const handleMoveUp = (index) => {
    if (index <= 0) return;
    const newJobs = [...jobs];
    const temp = newJobs[index - 1];
    newJobs[index - 1] = newJobs[index];
    newJobs[index] = temp;
    reorder(newJobs.map((j) => j.id));
  };

  // Prioritize job: Move Down
  const handleMoveDown = (index) => {
    if (index >= jobs.length - 1) return;
    const newJobs = [...jobs];
    const temp = newJobs[index + 1];
    newJobs[index + 1] = newJobs[index];
    newJobs[index] = temp;
    reorder(newJobs.map((j) => j.id));
  };

  // Open output folder in OS file manager
  const handleRevealOutput = async (job) => {
    try {
      const targetPath = job.output_path || job.output || '';
      await postJSON('/api/reveal', { path: targetPath });
      if (notify) notify('Opened output directory in file explorer', 'success');
    } catch (e) {
      if (notify) notify(e.message || 'Could not open folder', 'error');
    }
  };

  // Speed multiplier vs 30fps real-time
  const speedMultiplier = useMemo(() => {
    const fps = activeJobDetails.fps || 0;
    if (fps <= 0) return '0.0x real-time (0 FPS)';
    const mult = (fps / 30.0).toFixed(1);
    return `${mult}x real-time (${fps.toFixed(1)} FPS)`;
  }, [activeJobDetails.fps]);

  return (
    <div
      className={`flex flex-col border-t border-white/10 bg-[#0A0B10]/95 backdrop-blur-xl transition-all duration-200 select-none ${
        isOpen ? 'h-72' : 'h-10'
      } ${className}`}
    >
      {/* ── DRAWER HEADER / SUMMARY BAR ───────────────────────────────────── */}
      <div className="flex items-center justify-between px-4 h-10 border-b border-white/10 bg-white/[0.02] shrink-0">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggleOpen}
            aria-label={isOpen ? 'Collapse render queue drawer' : 'Expand render queue drawer'}
            className="flex items-center gap-2 text-white/90 hover:text-white transition-colors"
          >
            <Layers size={15} className="text-rose-400" aria-hidden="true" />
            <span className="text-compact font-semibold">Render Queue</span>
            <span className="text-nano px-1.5 py-0.5 rounded bg-white/10 text-white/70">
              {jobs.length} {jobs.length === 1 ? 'Job' : 'Jobs'}
            </span>
            {isOpen ? <ChevronDown size={14} className="text-white/40" aria-hidden="true" /> : <ChevronUp size={14} className="text-white/40" aria-hidden="true" />}
          </button>

          {/* Active Status Badge */}
          {running && (
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-yellow-500/15 border border-yellow-500/30 text-yellow-300 text-micro font-medium animate-pulse">
              <RefreshCw size={11} className="animate-spin" aria-hidden="true" />
              <span>Processing Batch</span>
            </div>
          )}
        </div>

        {/* Global Controls & Post-Render Toggles */}
        <div className="flex items-center gap-3">
          {/* Post-Render Options */}
          <div className="flex items-center gap-1.5 border-r border-white/10 pr-3">
            {/* Audio Chime Toggle */}
            <button
              type="button"
              onClick={() => setAudioNotify(!audioNotify)}
              aria-label={audioNotify ? 'Disable render complete audio chime' : 'Enable render complete audio chime'}
              title="Play chime on batch completion"
              className={`p-1.5 rounded transition-colors ${
                audioNotify ? 'text-rose-400 bg-rose-500/10' : 'text-white/40 hover:text-white/70'
              }`}
            >
              {audioNotify ? <Volume2 size={13} aria-hidden="true" /> : <VolumeX size={13} aria-hidden="true" />}
            </button>

            {/* Desktop Notification Toggle */}
            <button
              type="button"
              onClick={handleToggleDesktopNotification}
              aria-label={desktopNotify ? 'Disable desktop completion notifications' : 'Enable desktop completion notifications'}
              title="Show system desktop notification on finish"
              className={`p-1.5 rounded transition-colors ${
                desktopNotify ? 'text-sky-400 bg-sky-500/10' : 'text-white/40 hover:text-white/70'
              }`}
            >
              {desktopNotify ? <Bell size={13} aria-hidden="true" /> : <BellOff size={13} aria-hidden="true" />}
            </button>

            {/* Auto-Shutdown Toggle */}
            <button
              type="button"
              onClick={() => setAutoShutdown(!autoShutdown)}
              aria-label={autoShutdown ? 'Disable auto-shutdown on completion' : 'Enable auto-shutdown on completion'}
              title="Shutdown system when queue finishes"
              className={`flex items-center gap-1 px-1.5 py-1 rounded text-nano font-medium transition-colors ${
                autoShutdown
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  : 'text-white/40 hover:text-white/70'
              }`}
            >
              <Power size={11} aria-hidden="true" />
              <span>Shutdown</span>
            </button>
          </div>

          {/* Queue Execution Actions */}
          <div className="flex items-center gap-1.5">
            {running ? (
              paused ? (
                <button
                  type="button"
                  onClick={resume}
                  disabled={busy}
                  aria-label="Resume queue execution"
                  className="flex items-center gap-1 px-2.5 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-mini font-medium transition-colors"
                >
                  <Play size={12} aria-hidden="true" />
                  <span>Resume</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={pause}
                  disabled={busy}
                  aria-label="Pause queue execution"
                  className="flex items-center gap-1 px-2.5 py-1 rounded bg-amber-600 hover:bg-amber-500 text-white text-mini font-medium transition-colors"
                >
                  <Pause size={12} aria-hidden="true" />
                  <span>Pause</span>
                </button>
              )
            ) : (
              <button
                type="button"
                onClick={start}
                disabled={busy || jobs.length === 0}
                aria-label="Start rendering queue"
                className="flex items-center gap-1 px-3 py-1 rounded bg-rose-600 hover:bg-rose-500 text-white text-mini font-medium transition-colors disabled:opacity-40"
              >
                <Play size={12} aria-hidden="true" />
                <span>Start Queue</span>
              </button>
            )}

            {running && (
              <button
                type="button"
                onClick={stop}
                disabled={busy}
                aria-label="Stop queue"
                className="p-1.5 rounded bg-white/5 hover:bg-white/10 text-rose-400 hover:text-rose-300 transition-colors"
              >
                <Square size={13} aria-hidden="true" />
              </button>
            )}

            <button
              type="button"
              onClick={clear}
              disabled={busy || running || jobs.length === 0}
              aria-label="Clear all jobs in queue"
              className="p-1.5 rounded text-white/40 hover:text-white/80 hover:bg-white/10 transition-colors disabled:opacity-30"
            >
              <Trash2 size={13} aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>

      {/* ── EXPANDED DRAWER BODY (TASK ITEMS & METRICS) ────────────────────── */}
      {isOpen && (
        <div className="flex-1 flex overflow-hidden">
          {/* Active Job Telemetry Strip */}
          {activeJob && (
            <div className="w-80 border-r border-white/10 p-3.5 flex flex-col justify-between bg-white/[0.015] shrink-0">
              <div className="flex flex-col gap-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-micro font-semibold uppercase tracking-wider text-white/50">
                    Live Active Render
                  </span>
                  <span className="text-nano px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-300 font-mono font-medium">
                    ACTIVE
                  </span>
                </div>

                <div className="flex flex-col">
                  <span className="text-compact font-semibold text-white/95 truncate">
                    {activeJob.name || activeJob.target || 'Processing Task'}
                  </span>
                  <span className="text-nano text-white/40 font-mono">
                    ID: {activeJob.id}
                  </span>
                </div>

                {/* Progress bar */}
                <div className="flex flex-col gap-1 mt-1">
                  <div className="flex items-center justify-between text-nano font-mono text-white/70">
                    <span>{Math.round(activeJobDetails.progress * 100)}%</span>
                    <span>
                      {activeJobDetails.currentFrame} / {activeJobDetails.totalFrames} frames
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                    <div
                      style={{ width: `${Math.min(100, Math.max(0, activeJobDetails.progress * 100))}%` }}
                      className="h-full bg-gradient-to-r from-rose-500 to-amber-500 rounded-full transition-all duration-300"
                    />
                  </div>
                </div>

                {/* Rolling 100-Frame ETA & Speed */}
                <div className="grid grid-cols-2 gap-2 pt-2 border-t border-white/5 text-mini">
                  <div className="flex flex-col">
                    <span className="text-nano text-white/40">Rolling 100f ETA:</span>
                    <span className="font-mono font-semibold text-emerald-400">
                      {formatDuration(activeJobDetails.etaSeconds)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-nano text-white/40">Render Speed:</span>
                    <span className="font-mono text-white/80 truncate">
                      {speedMultiplier}
                    </span>
                  </div>
                </div>
              </div>

              {/* Encoder Details */}
              <div className="flex items-center justify-between p-2 rounded-lg bg-white/5 text-nano font-mono text-white/60">
                <span>Encoder: NVENC (hevc_nvenc)</span>
                <span>1080p60</span>
              </div>
            </div>
          )}

          {/* Jobs List Table */}
          <div className="flex-1 overflow-y-auto p-3 scrollbar-thin">
            {jobs.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-white/40">
                <Layers size={32} className="mb-2 text-white/20" aria-hidden="true" />
                <span className="text-plain font-medium text-white/60">Queue is Empty</span>
                <span className="text-mini text-white/40 max-w-sm mt-0.5">
                  Add render jobs from Face Swap or Batch Swap to process multiple targets sequentially.
                </span>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {jobs.map((job, idx) => {
                  const state = jobState(job);
                  const isActive = ACTIVE_STATES.includes(state);
                  const isDone = state === 'COMPLETED';
                  const isFailed = state === 'FAILED';
                  const stateLabel = QUEUE_STATE_LABEL[state] || state;
                  const stateClass = QUEUE_STATE_CLASS[state] || 'text-white/60 bg-white/5 border-white/10';

                  return (
                    <div
                      key={`job-${job.id || idx}`}
                      className={`flex items-center justify-between p-2.5 rounded-xl border transition-all ${
                        isActive
                          ? 'bg-white/[0.05] border-yellow-500/30 shadow-md'
                          : 'bg-white/[0.02] border-white/10 hover:border-white/20'
                      }`}
                    >
                      {/* Priority Controls & Order */}
                      <div className="flex items-center gap-1.5">
                        <span className="text-nano font-mono text-white/30 w-5 text-center">
                          #{idx + 1}
                        </span>
                        <div className="flex flex-col">
                          <button
                            type="button"
                            onClick={() => handleMoveUp(idx)}
                            disabled={idx === 0 || running}
                            aria-label={`Prioritize job ${job.id || idx + 1} up`}
                            className="p-0.5 text-white/30 hover:text-white disabled:opacity-20"
                          >
                            <ArrowUp size={11} aria-hidden="true" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleMoveDown(idx)}
                            disabled={idx === jobs.length - 1 || running}
                            aria-label={`Prioritize job ${job.id || idx + 1} down`}
                            className="p-0.5 text-white/30 hover:text-white disabled:opacity-20"
                          >
                            <ArrowDown size={11} aria-hidden="true" />
                          </button>
                        </div>
                      </div>

                      {/* Job Metadata & File Name */}
                      <div className="flex-1 min-w-0 px-3 flex flex-col">
                        <div className="flex items-center gap-2">
                          <span className="text-mini font-semibold text-white/90 truncate">
                            {job.name || job.target || `Render Job #${idx + 1}`}
                          </span>
                          <span className={`text-nano font-medium px-2 py-0.5 rounded border ${stateClass}`}>
                            {stateLabel}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-nano text-white/40 font-mono mt-0.5">
                          <span>{job.resolution || '1920x1080'}</span>
                          <span>&bull;</span>
                          <span>{job.encoder || 'NVENC'}</span>
                          <span>&bull;</span>
                          <span>{job.swap_model || 'realswap'}</span>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-1">
                        {isDone && (
                          <button
                            type="button"
                            onClick={() => handleRevealOutput(job)}
                            aria-label={`Open output folder for job ${job.name || idx + 1}`}
                            title="Reveal in file explorer"
                            className="p-1.5 rounded hover:bg-white/10 text-white/60 hover:text-white transition-colors"
                          >
                            <FolderOpen size={14} aria-hidden="true" />
                          </button>
                        )}

                        {isActive && (
                          <button
                            type="button"
                            onClick={() => cancel(job.id)}
                            aria-label={`Cancel active job ${job.name || idx + 1}`}
                            title="Cancel job"
                            className="p-1.5 rounded hover:bg-rose-500/20 text-rose-400 transition-colors"
                          >
                            <Square size={13} aria-hidden="true" />
                          </button>
                        )}

                        {isFailed && (
                          <button
                            type="button"
                            onClick={() => retry(job.id)}
                            aria-label={`Retry failed job ${job.name || idx + 1}`}
                            title="Retry job"
                            className="p-1.5 rounded hover:bg-white/10 text-amber-400 transition-colors"
                          >
                            <RefreshCw size={13} aria-hidden="true" />
                          </button>
                        )}

                        {!isActive && (
                          <button
                            type="button"
                            onClick={() => remove(job.id)}
                            disabled={running}
                            aria-label={`Remove job ${job.name || idx + 1} from queue`}
                            title="Remove from queue"
                            className="p-1.5 rounded hover:bg-white/10 text-white/40 hover:text-rose-400 transition-colors disabled:opacity-20"
                          >
                            <Trash2 size={13} aria-hidden="true" />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
