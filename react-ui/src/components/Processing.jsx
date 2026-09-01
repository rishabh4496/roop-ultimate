import React, { useEffect, useMemo, useRef } from 'react';
import { postJSON, API } from '../api';
import { AnimatedNumber, Button } from './ui';
import { Icon } from '../icons';
import { motion, spring } from '../motion';
import QualityReport from './QualityReport';
import ProcessingDock from './faceswap/ProcessingDock';
import ProcessingTerminal from './faceswap/ProcessingTerminal';
import DiagnosticsPanel from './faceswap/DiagnosticsPanel';
import LiveProcessingPeek from './faceswap/LiveProcessingPeek';
import useTelemetry from './faceswap/useTelemetry';
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
                                     desktopAlerts, onToggleDesktopAlerts }) {
  const p = settings || {};
  const processing = !!progress.processing;

  const telemetry = useTelemetry();
  const { renderLite, toggleRenderLite } = useRenderLite(processing);
  // The alert itself is App's — it is the only component always mounted, so a
  // run finishing while this tab is closed still announces itself. This tab
  // only hosts the TOGGLE for it, which is why both arrive as props.

  // The knobs that actually decide a run's speed and look, shown alongside the
  // live diagnostics so a screenshot of a slow or wrong-looking run says what
  // produced it. Mirrors the summary Face Swap used to build for this panel.
  const runConfigSummary = useMemo(() => ([
    ['swapper', p.swap_model || '—'],
    ['detector', p.detector_engine || '—'],
    ['enhancer', p.selected_enhancer && p.selected_enhancer !== 'None' ? p.selected_enhancer : 'off'],
    ['mask', p.mask_engine || '—'],
    ['pixel boost', p.subsample_upscale || '—'],
    ['upscale', p.upscale_after_swap ? (p.upscale_model_after || 'on') : 'off'],
    ['tracking', p.track_identities ? 'on' : 'off'],
    ['temporal', p.temporal_detection ? 'on' : 'off'],
    ['stabilize', p.stabilize_face || p.stabilize_enhancer || p.stabilize_mask ? 'on' : 'off'],
  ]), [p.swap_model, p.detector_engine, p.selected_enhancer, p.mask_engine,
       p.subsample_upscale, p.upscale_after_swap, p.upscale_model_after,
       p.track_identities, p.temporal_detection, p.stabilize_face, p.stabilize_enhancer,
       p.stabilize_mask]);

  const prog = progress.progress || 0;
  const startedAt = progress.started_at ? progress.started_at * 1000 : null;

  // Elapsed comes from the backend's own clock, so a run that was already going
  // when this view mounted keeps its real age. The last live value is kept in a
  // ref so the completed state can still report how long the run took — the
  // clock is gone by then, and "0s" would be a lie about a four-hour render.
  const liveElapsedMs = processing && startedAt ? Date.now() - startedAt : 0;
  const elapsedRef = useRef(0);
  useEffect(() => { if (liveElapsedMs > 0) elapsedRef.current = liveElapsedMs; }, [liveElapsedMs]);
  const elapsedMs = liveElapsedMs || elapsedRef.current;

  // "Time left" is the terminal's own eta_s wherever one is counting frames, so
  // the two agree by construction; the extrapolation is only the fallback for
  // start-up and the encode tail, where nothing is counting. (Deriving it from
  // elapsed × (1 − prog) / prog throughout reads minutes of model loading and
  // pre-pass as if they were swap time, and overshoots by 2×.)
  const etaMs = processing
    ? (typeof progress.eta_s === 'number' && progress.eta_s > 0
        ? progress.eta_s * 1000
        : (prog > 0.01 ? (elapsedMs * (1 - prog)) / prog : 0))
    : 0;

  const stop = async () => { try { await postJSON('/api/stop', {}); notify('Stopping…', 'info'); } catch (e) { notify(e.message, 'error'); } };
  const pause = async () => { try { await postJSON('/api/pause', {}); notify('Paused', 'info'); } catch (e) { notify(e.message, 'error'); } };
  const resume = async () => { try { await postJSON('/api/resume', {}); notify('Resumed'); } catch (e) { notify(e.message, 'error'); } };

  const out = !processing ? progress.output : null;
  const outUrl = out?.path ? `${API}/api/file?path=${encodeURIComponent(out.path)}&t=${progress.progress}` : '';
  const revealOutput = async () => {
    try { await postJSON('/api/reveal', { path: out?.path }); }
    catch (e) { notify(e.message, 'error'); }
  };

  const radius = 21;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - prog * circumference;

  // How the finished run ended. The edge alone does not say — a run stopped by
  // hand and one that crashed at 90% both just stop — so the error field and
  // how far it got decide the wording.
  const failed = !processing && !!progress.error;
  const completed = !processing && !progress.error && prog >= 0.99;

  return (
    <div className="w-full space-y-6">

      {/* ── Run bar ─────────────────────────────────────────────────────────
          Sticky, so the percentage and the stop control stay reachable however
          far down the terminal is scrolled. */}
      <div className="sticky top-20 z-30 pb-3 bg-[#0c0e14]/90 backdrop-blur-md">
        {processing ? (
          <div className="relative overflow-hidden rounded-2xl glass-panel px-5 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4 shadow-xl border border-white/5 w-full">
            {/* Left: circular progress ring & status */}
            <div className="flex items-center gap-3.5 min-w-0">
              <div className={`relative flex items-center justify-center h-14 w-14 select-none shrink-0 rounded-full transition-shadow duration-1000 ${!progress.paused ? 'shadow-[0_0_14px_var(--accent-glow)]' : ''}`}>
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
                  <span className={`h-2 w-2 rounded-full ${progress.paused ? 'bg-amber-400' : 'bg-[var(--accent)] animate-ping'}`} />
                  <span className={`text-mini font-semibold uppercase tracking-[0.14em] ${progress.paused ? 'text-amber-400' : 'text-[var(--accent)]'}`}>
                    {progress.paused ? 'Paused' : 'Processing'}
                  </span>
                </div>
                <div className="text-sm font-bold text-white truncate max-w-[340px]">
                  {progress.desc || 'Swapping faces…'}
                </div>
                {progress.error && <div className="text-xs text-red-400 font-semibold">{progress.error}</div>}
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
                <span className="text-emerald-400 font-bold tabular-nums whitespace-nowrap">{etaMs > 0 ? fmtTime(etaMs) : '--:--'}</span>
              </div>
            </div>

            {/* Right: big icon action buttons */}
            <div className="flex items-center gap-3 shrink-0">
              {progress.paused ? (
                <motion.button type="button" onClick={resume} title="Resume" aria-label="Resume the run"
                  whileHover={{ y: -3, scale: 1.06 }} whileTap={{ scale: 0.92, y: 0 }} transition={spring.snappy}
                  className="group flex flex-col items-center gap-1.5 focus:outline-none">
                  <span className="h-11 w-11 rounded-xl flex items-center justify-center bg-emerald-500/15 border border-emerald-500/40 text-emerald-400 transition-colors duration-200 group-hover:bg-emerald-500/25">
                    <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor"><path d="M8 5.14v13.72a1 1 0 0 0 1.53.85l10.9-6.86a1 1 0 0 0 0-1.7L9.53 4.29A1 1 0 0 0 8 5.14z" /></svg>
                  </span>
                  <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 group-hover:text-emerald-400 transition-colors">Resume</span>
                </motion.button>
              ) : (
                <motion.button type="button" onClick={pause} title="Pause" aria-label="Pause the run"
                  whileHover={{ y: -3, scale: 1.06 }} whileTap={{ scale: 0.92, y: 0 }} transition={spring.snappy}
                  className="group flex flex-col items-center gap-1.5 focus:outline-none">
                  <span className="h-11 w-11 rounded-xl flex items-center justify-center bg-amber-500/15 border border-amber-500/40 text-amber-400 transition-colors duration-200 group-hover:bg-amber-500/25">
                    <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1.5" /><rect x="14" y="5" width="4" height="14" rx="1.5" /></svg>
                  </span>
                  <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 group-hover:text-amber-400 transition-colors">Pause</span>
                </motion.button>
              )}
              <motion.button type="button" onClick={stop} title="Stop" aria-label="Stop the run"
                whileHover={{ y: -3, scale: 1.06 }} whileTap={{ scale: 0.92, y: 0 }} transition={spring.snappy}
                className="group flex flex-col items-center gap-1.5 focus:outline-none">
                <span className="h-11 w-11 rounded-xl flex items-center justify-center bg-red-500/15 border border-red-500/40 text-red-400 transition-colors duration-200 group-hover:bg-red-500/25">
                  <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2.5" /></svg>
                </span>
                <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 group-hover:text-red-400 transition-colors">Stop</span>
              </motion.button>
            </div>

            {/* Smooth animated progress line along the bottom edge */}
            <div className="absolute inset-x-0 bottom-0 h-1 bg-white/[0.04]">
              <div
                className={`h-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] transition-[width] duration-500 ease-out ${progress.paused ? '' : 'progress-bar-animated'}`}
                style={{ width: `${Math.max(2, prog * 100)}%`, boxShadow: '0 0 10px var(--accent-glow)' }}
              />
            </div>
          </div>
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
              {out?.path && (
                <a href={outUrl} download
                  className="inline-block px-3 py-1.5 rounded-xl text-sm bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white font-bold transition-colors">⬇ Download</a>
              )}
              {out?.path && <Button size="sm" variant="secondary" onClick={revealOutput}>Open folder</Button>}
              <Button size="sm" variant="secondary" onClick={() => setTab('faceswap')}>Back to Face Swap</Button>
            </div>
          </div>
        )}
      </div>

      {/* ── The stage ───────────────────────────────────────────────────────
          Fills the viewport: this tab has nothing else on it, and the
          diagnostics want the height. Floored so it stays usable on a short
          window. */}
      {processing ? (
        <div className="relative h-[calc(100vh-230px)] min-h-[620px] rounded-2xl overflow-hidden processing-stage flex flex-col items-center select-none px-4 sm:px-6 py-4">
          {/* h-full + min-h-0 so the console below takes ALL the leftover height
              instead of the whole block floating in a tall box. */}
          <div className="relative h-full w-full max-w-[1900px] min-h-0 flex flex-col gap-3">

            {/* ── Headline ────────────────────────────────────────────────
                One line that answers "where is it, and when is it done". */}
            <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className={`text-mini font-semibold uppercase tracking-[0.18em] ${progress.paused ? 'text-amber-400' : 'text-[var(--accent)]'}`}>
                    {progress.paused ? 'Paused' : 'Processing'}
                  </span>
                  {!progress.paused && <span className="h-px w-8 bg-[var(--accent)]/40" />}
                </div>
                <div className="mt-1 flex items-baseline gap-2.5">
                  <AnimatedNumber value={prog * 100} decimals={1} suffix="%"
                                  className="font-mono text-display leading-none font-bold tabular-nums text-white" />
                  <span className="text-sm font-medium text-white/55 truncate max-w-[46ch]">
                    {progress.desc || 'Swapping faces…'}
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
                  <div className="text-title font-bold tabular-nums text-emerald-400">{etaMs > 0 ? fmtTime(etaMs) : '--:--'}</div>
                </div>
                <div className="w-px bg-white/10" />
                <div className="text-right">
                  <div className="text-nano font-semibold uppercase tracking-[0.16em] text-white/45">Finishes</div>
                  <div className="text-title font-bold tabular-nums text-white/85">
                    {etaMs > 0 ? new Date(Date.now() + etaMs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'}
                  </div>
                </div>
              </div>
            </div>

            {/* ── Pipeline rail ───────────────────────────────────────────
                The stages ARE the progress bar: one continuous track split into
                named segments that fill as the run moves through them. */}
            {(() => {
              const d = (progress.desc || '').toLowerCase();
              const stages = [
                { key: 'analyze', label: 'Analyze' },
                { key: 'swap', label: 'Swap' },
                ...(p.upscale_after_swap ? [{ key: 'upscale', label: 'Upscale' }] : []),
                { key: 'combine', label: 'Combine' },
              ];
              let activeKey = 'swap';
              if (/combin|finaliz|encod|audio|mux/.test(d)) activeKey = 'combine';
              else if (/upscal/.test(d)) activeKey = 'upscale';
              else if (/processing frame|swapp/.test(d)) activeKey = 'swap';
              else if (/analy|track|extract|detect|start/.test(d)) activeKey = 'analyze';
              let activeIdx = stages.findIndex((s) => s.key === activeKey);
              if (activeIdx < 0) activeIdx = 1;
              return (
                <div className="w-full">
                  <div className="flex items-stretch gap-1">
                    {stages.map((s, i) => {
                      const state = i < activeIdx ? 'done' : i === activeIdx ? 'active' : 'pending';
                      return (
                        <div key={s.key} className="flex-1 min-w-0">
                          <div className={`h-1.5 rounded-full overflow-hidden ${state === 'pending' ? 'bg-white/[0.06]' : 'bg-white/[0.08]'}`}>
                            {state === 'done' && <div className="h-full w-full rounded-full bg-emerald-500/70" />}
                            {state === 'active' && (
                              <div className={`h-full rounded-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-hover)] transition-[width] duration-500 ease-out ${progress.paused ? '' : 'progress-bar-animated'}`}
                                   style={{ width: `${Math.max(4, prog * 100)}%` }} />
                            )}
                          </div>
                          <div className={`mt-1.5 flex items-center gap-1.5 text-micro font-semibold uppercase tracking-[0.12em] truncate ${
                            state === 'done' ? 'text-emerald-400/70'
                            : state === 'active' ? 'text-white'
                            : 'text-white/45'}`}>
                            {state === 'done' && <span aria-hidden>✓</span>}
                            <span className="truncate">{s.label}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })()}

            {/* Processing action control dock */}
            <ProcessingDock
              paused={progress.paused}
              onTogglePause={() => (progress.paused ? resume() : pause())}
              onCancelJob={stop}
              desktopAlerts={desktopAlerts}
              onToggleDesktopAlerts={onToggleDesktopAlerts}
              renderLite={renderLite}
              onToggleRenderLite={toggleRenderLite}
            />

            {/* Live processing frame peek & diagnostics */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
              <div className="lg:col-span-1">
                <LiveProcessingPeek
                  // The still Face Swap last had on screen, kept across the tab
                  // switch — see faceswap/lastPreview. It is only the fallback
                  // for the window before the first live frame is published.
                  previewSrc={lastPreview.previewSrc}
                  rawUrl={lastPreview.rawUrl}
                  // Keyed on live_seq, which only changes when the pipeline
                  // publishes a newer frame — so the browser refetches then and
                  // not once per poll.
                  liveSrc={progress.live_seq ? `${API}/api/live_frame?seq=${progress.live_seq}` : ''}
                  frame={lastPreview.frame}
                  maxFrames={lastPreview.maxFrames}
                  progressDesc={progress.desc}
                  paused={progress.paused}
                />
              </div>
              <div className="lg:col-span-2">
                <DiagnosticsPanel
                  desc={progress.desc}
                  telemetry={telemetry}
                  processing={processing}
                  paused={progress.paused}
                  config={runConfigSummary}
                  elapsedMs={elapsedMs}
                  etaMs={etaMs}
                  prog={prog}
                />
              </div>
            </div>

            {/* Live terminal feed — mirrors what the real console prints */}
            <ProcessingTerminal
              log={progress.log || []}
              parts={progress.parts || []}
              statusLine={progress.status_line || progress.desc}
              runtime={progress.runtime}
              paused={progress.paused}
              className="flex-1 min-h-0"
              bodyClass="h-full"
            />

            {progress.error && <div className="text-xs text-red-400 font-semibold text-center">{progress.error}</div>}
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* The output itself, so a finished run does not have to be chased
              into another tab to be looked at. */}
          {out?.path && (
            <div className="rounded-2xl glass-panel p-5 shadow-2xl border border-white/5 space-y-3">
              <div className="text-mini uppercase tracking-[0.14em] text-white/45 font-semibold">Output</div>
              {out.kind === 'video'
                ? <video src={outUrl} controls className="w-full max-h-[52vh] rounded-xl border border-white/5" />
                : <img src={outUrl} alt="Render output" className="w-full max-h-[52vh] object-contain rounded-xl border border-white/5" />}
              <QualityReport outputPath={out.path} notify={notify} />
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
            className="min-h-[320px]"
            bodyClass="h-[320px]"
          />
        </div>
      )}
    </div>
  );
}
