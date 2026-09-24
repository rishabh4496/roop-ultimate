import OutputVideoPlayer from './OutputVideoPlayer';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { postJSON, API } from '../api';
import { AnimatedNumber, Button } from './ui';
import { Icon } from '../icons';
import { motion, spring } from '../motion';
import { confirmDialog } from './confirm';
import QualityReport from './QualityReport';
import ProcessingDock from './faceswap/ProcessingDock';
import ProcessingTerminal from './faceswap/ProcessingTerminal';
import DiagnosticsPanel from './faceswap/DiagnosticsPanel';
import RunModelsPanel from './faceswap/RunModelsPanel';
import LiveProcessingPeek from './faceswap/LiveProcessingPeek';
import useTelemetry from './faceswap/useTelemetry';
import { useThrottledSelector, shallowEqual } from '../store/liveSubscribe';
import { UI_HZ, framesOf, etaMsOf } from '../store/telemetryStore';
import { outputMediaUrl, outputSource } from './outputUrl';
import useRenderLite from './faceswap/useRenderLite';
import { lastPreview } from './faceswap/lastPreview';
import { fmtTime } from './faceswap/utils';

/**
 * The Processing tab.
 *
 * All of this used to live inside the Face Swap tab, which meant a run took
 * that tab over: the settings sidebar, the asset rail, the timeline and the
 * preview controls were all hidden for the duration, and everything you had set
 * up disappeared behind a progress panel until the render finished. Watching a
 * run and preparing the next one were the same screen, so you could only do one
 * of them.
 *
 * Now they are two screens. Face Swap keeps its full layout at all times, and
 * this tab — which only exists once a run has started — is the one that gets
 * taken over by the render. Starting a job switches here automatically; the tab
 * stays after the run ends (so the finished log and the output are still
 * readable) and disappears once you navigate away from it.
 */
export default function Processing({ progress, settings, notify, setTab,
                                     desktopAlerts, onToggleDesktopAlerts,
                                     onPauseRun, onResumeRun, onStopRun,
                                     controlBusy = '' }) {
  const p = settings || {};
  const processing = !!progress.processing;
  const pauseRequested = !!progress.pause_requested;
  const stopping = !!progress.stop_requested || controlBusy === 'stop';

  const [terminalExpanded, setTerminalExpanded] = useState(false);

  const telemetry = useTelemetry();
  const {
    renderLite, mode: renderLiteMode, label: renderLiteLabel,
    hint: renderLiteHint, toggleRenderLite,
  } = useRenderLite(processing);
  // The alert itself is App's — it is the only component always mounted, so a
  // run finishing while this tab is closed still announces itself. This tab
  // only hosts the TOGGLE for it, which is why both arrive as props.

  // The knobs that decide a run's speed and look, shown alongside the live
  // diagnostics so a screenshot of a slow or wrong-looking run says what
  // produced it.
  //
  // The MODEL names are deliberately not here any more. They moved to
  // RunModelsPanel, which reads them from the pipeline's own runtime snapshot
  // rather than from `settings` — and while both were on screen they could
  // disagree, because this list re-renders the moment a control is touched in
  // another tab while the run keeps whatever it started with. Two panels
  // captioned "swapper" showing two different swappers is worse than one panel.
  // What is left is the behaviour flags, which have no runtime equivalent.
  const runConfigSummary = useMemo(() => ([
    ['face mode', p.face_detection_mode || '—'],
    ['det size', p.default_det_size ? 'auto' : (p.face_detector_size || '—')],
    ['swap steps', String(p.num_swap_steps ?? 1)],
    ['tracking', p.track_identities ? 'on' : 'off'],
    ['temporal', p.temporal_detection ? 'on' : 'off'],
    ['stabilize', p.stabilize_face || p.stabilize_enhancer || p.stabilize_mask ? 'on' : 'off'],
    ['threads', String(p.max_threads ?? '—')],
    ['codec', p.output_video_codec || '—'],
  ]), [p.face_detection_mode, p.default_det_size, p.face_detector_size,
       p.num_swap_steps, p.track_identities, p.temporal_detection,
       p.stabilize_face, p.stabilize_enhancer, p.stabilize_mask,
       p.max_threads, p.output_video_codec]);

  // The fast readouts (ring, percentage, frame counter, elapsed, ETA, the
  // pipeline rail, the peek, diagnostics) live in the *Live components at the
  // bottom of this file, which subscribe to the telemetry store themselves.
  // What this component still needs is the FINISHED view's summary, which
  // reads `progress` as it stood at the end: App folds the fast fields into
  // `progress` on every structural edge and on each poll.
  const rawProg = Number(progress.progress);
  const prog = Number.isFinite(rawProg) ? Math.min(1, Math.max(0, rawProg)) : 0;
  // The live pieces record the run's age here as it ticks, so "Took 41m" is
  // still answerable after the clock has stopped. The backend's own duration
  // wins when the poll has delivered it.
  const elapsedRef = useRef(0);
  const backendDurationMs = progress.duration_s ? progress.duration_s * 1000 : 0;
  const elapsedMs = backendDurationMs || elapsedRef.current;

  const pause = onPauseRun || (async () => { try { await postJSON('/api/pause', {}); notify('Pause requested; waiting for a safe checkpoint', 'info'); } catch (e) { notify(e.message, 'error'); } });
  const resume = onResumeRun || (async () => { try { await postJSON('/api/resume', {}); notify('Resumed'); } catch (e) { notify(e.message, 'error'); } });

  // Confirmation guard before cancelling a job to prevent accidental abortion
  const stop = useCallback(async () => {
    if (!(await confirmDialog({
      title: 'Stop job?',
      message: 'Stop the active job? The partial output so far is finalized and kept.',
      confirmLabel: 'Stop',
      danger: true,
    }))) return;

    if (onStopRun) {
      onStopRun();
    } else {
      try {
        await postJSON('/api/stop', {});
        notify('Stopping…', 'info');
      } catch (e) {
        notify(e.message, 'error');
      }
    }
  }, [onStopRun, notify]);

  const out = !processing ? progress.output : null;
  // Versioned by the file itself (outputUrl.js). `|| Date.now()` here was
  // evaluated on every render, so any re-render made a new URL and remounted
  // the player, throwing away its buffer and its position.
  const outUrl = outputMediaUrl(out);
  const isVideoOutput = out?.kind === 'video' || /\.(mp4|mkv|mov|webm|avi)$/i.test(out?.path || '');
  const revealOutput = async () => {
    try { await postJSON('/api/reveal', { path: out?.path }); }
    catch (e) { notify(e.message, 'error'); }
  };


  // How the finished run ended. The edge alone does not say — a run stopped by
  // hand and one that crashed at 90% both just stop — so the error field and
  // how far it got decide the wording.
  const failed = !processing && !!progress.error;
  const completed = !processing && !progress.error && prog >= 0.99;

  return (
    <div className="w-full space-y-4">

      {/* ── Run bar ─────────────────────────────────────────────────────────
          Sticky, so the percentage and the stop control stay reachable however
          far down the terminal is scrolled.

          The backdrop follows --bg-gradient's own surface rather than a
          hardcoded `bg-neutral-950/70`. That literal is a near-black wash, which
          is invisible on the dark themes it was written for and a dark smear
          across the top of the panel on every LIGHT one — and the content
          scrolling underneath it still showed through, because 70% of black over
          a white page is grey, not "the page". */}
      <div className="sticky top-20 z-30 pb-3 backdrop-blur-md"
           style={{ background: 'linear-gradient(to bottom, var(--card-bg) 0%, var(--card-bg) 72%, transparent 100%)' }}>
        {processing ? (
          <RunBarLive
            startedAtS={progress.started_at}
            paused={!!progress.paused}
            pauseRequested={pauseRequested}
            stopping={stopping}
            error={progress.error}
            controlBusy={controlBusy}
            pause={pause}
            resume={resume}
            stop={stop}
            elapsedRef={elapsedRef}
          />
        ) : (
          /* ── The run is over ─────────────────────────────────────────────
             The tab stays until you leave it, because the log, the numbers and
             the file that came out of it are all still worth reading. */
          <div className="rounded-2xl glass-panel px-5 py-4 flex flex-wrap items-center justify-between gap-4 shadow-xl border border-white/5 w-full">
            <div className="flex items-center gap-3.5 min-w-0">
              <span className={`grid place-items-center h-11 w-11 rounded-xl border shrink-0 ${
                failed ? 'bg-red-500/10 border-red-500/25 text-red-400'
                : completed ? 'bg-emerald-500/10 border-emerald-500/25 text-emerald-400'
                : 'bg-amber-500/10 border-amber-500/25 text-amber-400'}`}>
                {failed ? <Icon.error size={20} /> : completed ? <Icon.success size={20} /> : <Icon.stop size={18} />}
              </span>
              <div className="min-w-0">
                <div className="text-sm font-bold text-white">
                  {failed ? 'Run failed' : completed ? 'Run complete' : 'Run stopped'}
                </div>
                <div className="text-xs text-white/45 truncate max-w-[52ch]">
                  {progress.error
                    || (elapsedMs > 0 ? `Took ${fmtTime(elapsedMs)}${prog > 0 ? ` · reached ${Math.round(prog * 100)}%` : ''}` : (progress.desc || 'Idle'))}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {out?.path && !failed && (
                <a href={outUrl} download={out.path.split(/[\\/]/).pop()}
                  className="inline-block px-3 py-1.5 rounded-xl text-sm bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white font-bold transition-colors">⬇ Download</a>
              )}
              {out?.path && !failed && <Button size="sm" variant="secondary" onClick={revealOutput}>Open folder</Button>}
              <Button size="sm" variant="secondary" onClick={() => setTab('faceswap')}>Back to Face Swap</Button>
            </div>
          </div>
        )}
      </div>

      {/* ── The stage ───────────────────────────────────────────────────────
          Responsive scrollable container so the terminal and controls are never
          cut off on constrained viewports or laptop displays. */}
      {processing ? (
        <div className="relative min-h-[620px] rounded-2xl overflow-y-auto overflow-x-hidden custom-scrollbar processing-stage flex flex-col items-center select-none px-4 sm:px-6 py-4">
          <div className="relative w-full max-w-[1900px] min-h-0 flex flex-col gap-3.5">

            <RunHeadlineLive
              p={p}
              startedAtS={progress.started_at}
              paused={!!progress.paused}
              pauseRequested={pauseRequested}
              stopping={stopping}
              elapsedRef={elapsedRef}
            />

            {/* Processing action control dock */}
            <ProcessingDock
              paused={progress.paused}
              pauseRequested={pauseRequested}
              stopping={stopping}
              controlBusy={controlBusy}
              onTogglePause={() => (progress.paused ? resume() : pause())}
              onCancelJob={stop}
              desktopAlerts={desktopAlerts}
              onToggleDesktopAlerts={onToggleDesktopAlerts}
              renderLite={renderLite}
              renderLiteMode={renderLiteMode}
              renderLiteLabel={renderLiteLabel}
              renderLiteHint={renderLiteHint}
              onToggleRenderLite={toggleRenderLite}
            />

            {/* Live processing frame peek & diagnostics */}
            {!terminalExpanded && (
              <RunPanelsLive
                settings={settings}
                runtime={progress.runtime}
                telemetry={telemetry}
                runConfigSummary={runConfigSummary}
                startedAtS={progress.started_at}
                paused={!!progress.paused}
                pauseRequested={pauseRequested}
                stopping={stopping}
                elapsedRef={elapsedRef}
              />
            )}

            {/* Live terminal feed — mirrors what the real console prints */}
            <ProcessingTerminal
              log={progress.log || []}
              parts={progress.parts || []}
              statusLine={progress.status_line || progress.desc}
              runtime={progress.runtime}
              paused={progress.paused}
              expanded={terminalExpanded}
              onToggleExpand={() => setTerminalExpanded((v) => !v)}
              className={`w-full transition-all duration-300 ${terminalExpanded ? 'min-h-[560px]' : 'min-h-[320px]'}`}
              bodyClass={terminalExpanded ? 'h-[520px] lg:h-[600px]' : 'h-[260px] lg:h-[300px]'}
            />

            {progress.error && <div className="text-xs text-red-400 font-semibold text-center">{progress.error}</div>}
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* The output itself, so a finished run does not have to be chased
              into another tab to be looked at. */}
          {out?.path && !failed && (
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
              <div className="rounded-2xl glass-panel p-5 shadow-2xl border border-white/5 space-y-3 min-w-0">
                <div className="text-mini uppercase tracking-[0.14em] text-white/45 font-semibold">Output</div>
                {isVideoOutput
                  ? <OutputVideoPlayer src={outUrl} renderKey={out?.path || out?.url} source={outputSource(out)} className="w-full max-h-[52vh] rounded-xl border border-white/5" />
                  : <img src={outUrl} alt="Render output" className="w-full max-h-[52vh] object-contain rounded-xl border border-white/5" />}
                <QualityReport outputPath={out.path} notify={notify} />
              </div>
              {/* What produced this file. The backend keeps the last run's
                  runtime block until the next run starts, so "which swapper was
                  this?" is answerable while looking at the result rather than
                  only while it is being made. */}
              <RunModelsPanel
                runtime={progress.runtime}
                settings={settings}
                telemetry={telemetry}
                className="self-start"
              />
            </div>
          )}

          {/* The log the run left behind. The backend keeps it until the next
              run starts, which is exactly as long as it is useful. */}
          <ProcessingTerminal
            log={progress.log || []}
            parts={progress.parts || []}
            statusLine={progress.status_line || ''}
            runtime={progress.runtime}
            paused={false}
            className="min-h-[360px]"
            bodyClass="h-[360px]"
          />
        </div>
      )}
    </div>
  );
}

// ── The live half of the tab ──────────────────────────────────────────────
//
// Everything below re-renders on TELEMETRY; Processing above re-renders only on
// structure (a run starting, pausing, ending) and on the slow poll that brings
// the log. Before the split the whole tab — the 250-line terminal included —
// re-rendered on every 4 Hz telemetry frame and on a 1 s clock besides. Each
// piece here subscribes to the telemetry store through useLiveRun at <= UI_HZ,
// and the clock ticks only in here.

const RUN_FIELDS = (s) => ({
  progress: s.run.progress,
  desc: s.run.desc,
  statusLine: s.run.status_line,
  current_frame: s.run.current_frame,
  total_frames: s.run.total_frames,
  eta_s: s.run.eta_s,
  fps: s.run.fps,
  liveSeq: s.run.live_seq,
});

function useLiveRun({ startedAtS, paused, pauseRequested, stopping, elapsedRef }) {
  const run = useThrottledSelector(RUN_FIELDS, { hz: UI_HZ, equals: shallowEqual });

  // A 1 s clock so Elapsed and ETA tick between telemetry frames. It keeps
  // running WHILE PAUSED: `started_at` is the backend's wall clock and does not
  // stop for a pause, so freezing this froze the Elapsed readout while the real
  // number kept climbing, and resuming made it jump by the length of the pause.
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Clamped, because it is drawn straight into a `width:` and a stroke offset.
  // `_progress["progress"]` is written from several places in the backend — the
  // swap loop, the post-swap upscale pass and the interpolation pass each reset
  // it to 0.0 and count up again — and `post_swap.py` caps its own writes at
  // 0.999 while `api.py` sets a flat 1.0 at the end. Unclamped, a value even
  // slightly outside [0,1] produced a negative `strokeDashoffset`, which
  // Chromium renders as a FULL ring: a run at 4% drew a complete circle.
  const rawProg = Number(run.progress);
  const prog = Number.isFinite(rawProg) ? Math.min(1, Math.max(0, rawProg)) : 0;

  // Frame counters from the telemetry frame (numbers, 4 Hz) or the status line.
  // The headline percentage is a fraction of the WHOLE job including the encode
  // tail; "51,203 / 88,483" is the thing people actually read progress from.
  const frames = framesOf(run);

  // Elapsed from the backend's own clock, so a run that was already going when
  // this view mounted keeps its real age. The last value is kept in the
  // parent's ref so the finished view can still say how long the run took.
  const startedAt = startedAtS ? startedAtS * 1000 : null;
  const elapsedMs = startedAt ? Math.max(0, now - startedAt) : 0;
  useEffect(() => {
    if (elapsedMs > 0 && elapsedRef) elapsedRef.current = elapsedMs;
  }, [elapsedMs, elapsedRef]);

  // "Time left" is the terminal's own eta_s wherever one is counting frames, so
  // the two agree by construction; the extrapolation is only the fallback for
  // start-up and the encode tail. A PAUSED run has no ETA: eta_s is
  // frames-remaining over a rate that is now zero, and "Finishes 04:12" over a
  // stopped render is a wall-clock time that is already wrong when drawn.
  const etaMs = !paused && !pauseRequested && !stopping ? etaMsOf(run, elapsedMs) : 0;
  // Why the ETA is not being shown, so the placeholder is never just a shrug.
  const etaNote = stopping ? 'stopping'
    : (paused || pauseRequested) ? 'paused'
    : etaMs > 0 ? null
    : 'estimating…';

  return {
    prog, frames, elapsedMs, etaMs, etaNote, now,
    desc: run.desc, statusLine: run.statusLine, fps: run.fps, liveSeq: run.liveSeq,
  };
}

const radius = 21;
const circumference = radius * 2 * Math.PI;

function RunBarLive({ startedAtS, paused, pauseRequested, stopping, error, controlBusy,
                      pause, resume, stop, elapsedRef }) {
  const { prog, frames, elapsedMs, etaMs, etaNote, desc } = useLiveRun({
    startedAtS, paused, pauseRequested, stopping, elapsedRef,
  });
  const strokeDashoffset = circumference - prog * circumference;
  return (
    <div className="relative overflow-hidden rounded-2xl glass-panel px-5 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4 shadow-xl border border-white/5 w-full">
      {/* Left: circular progress ring & status */}
      <div className="flex items-center gap-3.5 min-w-0">
        <div className={`relative flex items-center justify-center h-14 w-14 select-none shrink-0 rounded-full transition-shadow duration-1000 ${!paused && !stopping ? 'shadow-[0_0_14px_var(--accent-glow)]' : ''}`}>
          <svg className="transform -rotate-90 w-[52px] h-[52px]" viewBox="0 0 48 48">
            <circle stroke="rgba(255, 255, 255, 0.08)" fill="transparent" strokeWidth={3.5} r={radius} cx={24} cy={24} />
            <circle
              className="transition-all duration-500 ease-out"
              stroke="var(--accent)"
              fill="transparent"
              strokeWidth={3.5}
              strokeDasharray={`${circumference} ${circumference}`}
              style={{ strokeDashoffset, filter: 'drop-shadow(0 0 4px var(--accent-glow))' }}
              r={radius}
              cx={24}
              cy={24}
              strokeLinecap="round"
            />
          </svg>
          <AnimatedNumber value={prog * 100} decimals={0} suffix="%" className="absolute text-compact font-extrabold text-white tabular-nums" />
        </div>

        <div className="space-y-0.5 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${stopping ? 'bg-red-400' : paused || pauseRequested ? 'bg-amber-400' : 'bg-[var(--accent)] animate-ping'}`} />
            <span className={`text-mini font-semibold uppercase tracking-[0.14em] ${stopping ? 'text-red-400' : paused || pauseRequested ? 'text-amber-400' : 'text-[var(--accent)]'}`}>
              {stopping ? 'Stopping' : paused ? 'Paused' : pauseRequested ? 'Pause Requested' : 'Processing'}
            </span>
          </div>
          <div className="text-sm font-bold text-white truncate max-w-[340px]">
            {desc || 'Swapping faces…'}
          </div>
          {error && <div className="text-xs text-red-400 font-semibold">{error}</div>}
        </div>
      </div>

      {/* Elapsed / ETA compact readout */}
      <div className="flex items-center gap-3 text-xs font-mono shrink-0">
        <div className="flex flex-col">
          <span className="text-micro uppercase tracking-wider text-white/45 font-bold">Elapsed</span>
          <span className="text-white font-bold tabular-nums whitespace-nowrap">{fmtTime(elapsedMs)}</span>
        </div>
        <div className="h-6 w-px bg-white/10" />
        <div className="flex flex-col">
          <span className="text-micro uppercase tracking-wider text-white/45 font-bold">ETA</span>
          {/* `--:--` read as a clock that had stopped. It is not a time at
              all — it is the absence of one — so it says which. */}
          <span className={`font-bold tabular-nums whitespace-nowrap ${etaMs > 0 ? 'text-emerald-400' : 'text-white/35'}`}>
            {etaMs > 0 ? fmtTime(etaMs) : (etaNote || '—')}
          </span>
        </div>
      </div>

      {/* Right: big icon action buttons */}
      <div className="flex items-center gap-3 shrink-0">
        {paused ? (
          <motion.button type="button" onClick={resume} disabled={stopping || !!controlBusy} title="Resume" aria-label="Resume the run"
            whileHover={{ y: -3, scale: 1.06 }} whileTap={{ scale: 0.92, y: 0 }} transition={spring.snappy}
            className="group flex flex-col items-center gap-1.5 focus:outline-none cursor-pointer">
            <span className="h-11 w-11 rounded-xl flex items-center justify-center bg-emerald-500/15 border border-emerald-500/40 text-emerald-400 transition-colors duration-200 group-hover:bg-emerald-500/25">
              <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor"><path d="M8 5.14v13.72a1 1 0 0 0 1.53.85l10.9-6.86a1 1 0 0 0 0-1.7L9.53 4.29A1 1 0 0 0 8 5.14z" /></svg>
            </span>
            <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 group-hover:text-emerald-400 transition-colors">Resume</span>
          </motion.button>
        ) : (
          <motion.button type="button" onClick={pause} disabled={pauseRequested || stopping || !!controlBusy} title={pauseRequested ? 'Waiting for a safe checkpoint' : 'Pause'} aria-label={pauseRequested ? 'Pause requested' : 'Pause the run'}
            whileHover={{ y: -3, scale: 1.06 }} whileTap={{ scale: 0.92, y: 0 }} transition={spring.snappy}
            className="group flex flex-col items-center gap-1.5 focus:outline-none cursor-pointer">
            <span className="h-11 w-11 rounded-xl flex items-center justify-center bg-amber-500/15 border border-amber-500/40 text-amber-400 transition-colors duration-200 group-hover:bg-amber-500/25">
              <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1.5" /><rect x="14" y="5" width="4" height="14" rx="1.5" /></svg>
            </span>
            <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 group-hover:text-amber-400 transition-colors">{pauseRequested ? 'Requested' : 'Pause'}</span>
          </motion.button>
        )}
        <motion.button type="button" onClick={stop} disabled={stopping} title={stopping ? 'Stopping' : 'Stop'} aria-label={stopping ? 'Stop requested' : 'Stop the run'}
          whileHover={{ y: -3, scale: 1.06 }} whileTap={{ scale: 0.92, y: 0 }} transition={spring.snappy}
          className="group flex flex-col items-center gap-1.5 focus:outline-none cursor-pointer">
          <span className="h-11 w-11 rounded-xl flex items-center justify-center bg-red-500/15 border border-red-500/40 text-red-400 transition-colors duration-200 group-hover:bg-red-500/25">
            <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2.5" /></svg>
          </span>
          <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 group-hover:text-red-400 transition-colors">{stopping ? 'Stopping…' : 'Stop'}</span>
        </motion.button>
      </div>

      {/* Smooth animated progress line along the bottom edge.

          This is the tab's primary progress indicator and was invisible to
          assistive tech: a pair of bare divs with a width. `role` + the
          aria-value* trio is what makes a screen reader able to report the
          run at all, and it is the ONE bar here that measures the whole
          job, so it is the one that carries it. */}
      <div
        className="absolute inset-x-0 bottom-0 h-1 bg-white/[0.04]"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(prog * 100)}
        aria-valuetext={`${Math.round(prog * 100)} percent${
          frames ? `, frame ${frames.done} of ${frames.total}` : ''}${
          etaMs > 0 ? `, ${fmtTime(etaMs)} remaining` : ''}`}
        aria-label="Render progress"
      >
        <div
          className={`h-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] transition-[width] duration-500 ease-out ${paused ? '' : 'progress-bar-animated'}`}
          style={{ width: `${Math.max(2, prog * 100)}%`, boxShadow: '0 0 10px var(--accent-glow)' }}
        />
      </div>
    </div>
  );
}

function RunHeadlineLive({ p, startedAtS, paused, pauseRequested, stopping, elapsedRef }) {
  const { prog, frames, etaMs, etaNote, now, elapsedMs, desc } = useLiveRun({
    startedAtS, paused, pauseRequested, stopping, elapsedRef,
  });
  return (
    <>
      {/* ── Headline ────────────────────────────────────────────────
          One line that answers "where is it, and when is it done". */}
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-mini font-semibold uppercase tracking-[0.18em] ${paused ? 'text-amber-400' : 'text-[var(--accent)]'}`}>
              {paused ? 'Paused' : 'Processing'}
            </span>
            {!paused && <span className="h-px w-8 bg-[var(--accent)]/40" />}
          </div>
          <div className="mt-1 flex items-baseline gap-2.5">
            <AnimatedNumber value={prog * 100} decimals={1} suffix="%"
                            className="font-mono text-display leading-none font-bold tabular-nums text-white" />
            {/* The frame counter, beside the percentage rather than only
                down in the diagnostics strip. The percentage covers the
                whole job including the encode tail, so it moves at a
                different rate from the frames and people read progress
                from the frames. */}
            {frames && (
              <span className="font-mono text-title font-bold tabular-nums text-white/45 whitespace-nowrap">
                {frames.done.toLocaleString()}
                <span className="text-white/25"> / {frames.total.toLocaleString()}</span>
              </span>
            )}
            <span className="text-sm font-medium text-white/55 truncate max-w-[46ch]">
              {desc || 'Swapping faces…'}
            </span>
          </div>
        </div>
        <div className="flex items-stretch gap-5 font-mono">
          <div className="text-right">
            <div className="text-nano font-semibold uppercase tracking-[0.16em] text-white/45">Elapsed</div>
            <div className="text-title font-bold tabular-nums text-white/85">{fmtTime(elapsedMs)}</div>
          </div>
          <div className="w-px bg-white/10" />
          <div className="text-right">
            <div className="text-nano font-semibold uppercase tracking-[0.16em] text-white/45">Time left</div>
            <div className={`text-title font-bold tabular-nums ${etaMs > 0 ? 'text-emerald-400' : 'text-white/35'}`}>
              {etaMs > 0 ? fmtTime(etaMs) : (etaNote || '—')}
            </div>
          </div>
          <div className="w-px bg-white/10" />
          <div className="text-right">
            <div className="text-nano font-semibold uppercase tracking-[0.16em] text-white/45">Finishes</div>
            <div className="text-title font-bold tabular-nums text-white/85">
              {etaMs > 0 ? new Date(now + etaMs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'}
            </div>
          </div>
        </div>
      </div>

      {/* ── Pipeline rail ───────────────────────────────────────────
          Continuous track split into named segments with glowing step markers.

          THE BAR INSIDE THE ACTIVE SEGMENT IS STAGE-LOCAL, not the whole
          run's percentage. It used to be `prog * 100`, which is a fraction
          of the ENTIRE job: at 60% overall the Swap segment showed 60%
          full, and then the moment the run moved on to Combine — a stage
          that is seconds long — that segment ALSO started at ~99% and
          crawled, because it was still being handed the job-wide number.
          Each segment claimed to be a progress bar for its own stage and
          was in fact four copies of the same one. The frame counter is the
          honest source for the stage that has one; a stage without one
          (encode, mux) gets an indeterminate sweep, which says "working,
          length unknown" instead of inventing a number. */}
      {(() => {
        const d = (desc || '').toLowerCase();
        const hasInterp = !!p.interp_after_swap && p.interp_after_swap !== 'off';
        const stages = [
          { key: 'analyze', label: 'Analyze' },
          { key: 'swap', label: 'Swap' },
          ...(p.upscale_after_swap ? [{ key: 'upscale', label: 'Upscale' }] : []),
          ...(hasInterp ? [{ key: 'interp', label: 'Interpolate' }] : []),
          { key: 'combine', label: 'Combine' },
        ];
        let activeKey = 'swap';
        if (/combin|finaliz|encod|audio|mux/.test(d)) activeKey = 'combine';
        else if (/interpolat|rife|minterpolat/.test(d)) activeKey = 'interp';
        else if (/upscal/.test(d)) activeKey = 'upscale';
        else if (/processing frame|swapp/.test(d)) activeKey = 'swap';
        else if (/analy|track|extract|detect|start/.test(d)) activeKey = 'analyze';
        let activeIdx = stages.findIndex((s) => s.key === activeKey);
        if (activeIdx < 0) activeIdx = 1;
        // The counter in the status line restarts per stage ("Upscaling
        // frame 1 / N"), so it IS the stage-local fraction wherever one is
        // printed at all.
        const stageFrac = activeKey === 'combine' ? null : (frames && frames.total > 0
          ? Math.min(1, Math.max(0, frames.done / frames.total))
          : null);
        return (
          <div className="w-full p-2.5 sm:p-3 rounded-2xl bg-white/[0.02] border border-white/10 backdrop-blur-sm shadow-inner">
            <div className="flex items-stretch gap-2">
              {stages.map((s, i) => {
                const state = i < activeIdx ? 'done' : i === activeIdx ? 'active' : 'pending';
                return (
                  <div key={s.key} className="flex-1 min-w-0">
                    <div className={`h-2 rounded-full overflow-hidden transition-all duration-300 ${
                      state === 'pending'
                        ? 'bg-white/[0.06]'
                        : state === 'done'
                          ? 'bg-emerald-500/20'
                          : 'bg-white/[0.08]'
                    }`}>
                      {state === 'done' && <div className="h-full w-full rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]" />}
                      {state === 'active' && (stageFrac != null ? (
                        <div
                          className={`h-full rounded-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] transition-[width] duration-500 ease-out ${paused ? '' : 'progress-bar-animated'}`}
                          style={{ width: `${Math.max(6, stageFrac * 100)}%`, boxShadow: '0 0 10px var(--accent-glow)' }}
                        />
                      ) : (
                        /* No counter for this stage: sweep rather than lie. */
                        <div className="relative h-full w-full overflow-hidden">
                          <div
                            className={`absolute inset-y-0 w-1/3 rounded-full bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent ${paused ? '' : 'preview-indeterminate'}`}
                            style={{ boxShadow: '0 0 10px var(--accent-glow)' }}
                          />
                        </div>
                      ))}
                    </div>
                    <div className={`mt-2 flex items-center gap-1.5 text-micro font-semibold uppercase tracking-[0.14em] truncate transition-colors ${
                      state === 'done'
                        ? 'text-emerald-400'
                        : state === 'active'
                          ? 'text-white drop-shadow-[0_0_8px_var(--accent-glow)]'
                          : 'text-white/40'
                    }`}>
                      {state === 'done' ? (
                        <span className="text-emerald-400 font-bold" aria-hidden="true">✓</span>
                      ) : state === 'active' ? (
                        <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)] shadow-[0_0_6px_var(--accent-glow)] animate-pulse" />
                      ) : (
                        <span className="h-1 w-1 rounded-full bg-white/20" />
                      )}
                      <span className="truncate">{s.label}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}
    </>
  );
}

function RunPanelsLive({ settings, runtime, telemetry, runConfigSummary, startedAtS,
                         paused, pauseRequested, stopping, elapsedRef }) {
  const processing = true;
  const { prog, frames, elapsedMs, etaMs, desc, statusLine, fps, liveSeq } = useLiveRun({
    startedAtS, paused, pauseRequested, stopping, elapsedRef,
  });
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
      <div className="lg:col-span-1 flex flex-col gap-3.5 min-w-0">
        <LiveProcessingPeek
          // The still Face Swap last had on screen, kept across the tab
          // switch — see faceswap/lastPreview. It is only the fallback for
          // the window before the first live frame is published.
          previewSrc={lastPreview.previewSrc}
          rawUrl={lastPreview.rawUrl}
          // HTTP fallback only: with /ws/frames up the peek is pushed the
          // JPEG bytes directly and never fetches this. Keyed on live_seq so
          // the browser refetches when there is a newer frame, not per poll.
          liveSrc={liveSeq ? `${API}/api/live_frame?seq=${liveSeq}` : ''}
          frame={lastPreview.frame}
          maxFrames={lastPreview.maxFrames}
          progressDesc={desc}
          paused={paused}
        />
        {/* WHAT is doing the work, under the frame it is producing.
            Sourced from the pipeline's own runtime snapshot, so it
            shows the models the run actually holds rather than
            whatever Settings has drifted to since it started. */}
        <RunModelsPanel
          runtime={runtime}
          settings={settings}
          telemetry={telemetry}
        />
      </div>
      <div className="lg:col-span-2 min-w-0">
        <DiagnosticsPanel
          desc={statusLine || desc}
          telemetry={telemetry}
          processing={processing}
          paused={paused}
          config={runConfigSummary}
          elapsedMs={elapsedMs}
          etaMs={etaMs}
          prog={prog}
          // The counters straight off the 4 Hz telemetry frame. The
          // panel used to re-parse them out of the status STRING,
          // which only changes as fast as the string is rewritten.
          framesDone={frames?.done}
          framesTotal={frames?.total}
          fps={fps}
        />
      </div>
    </div>
  );
}
