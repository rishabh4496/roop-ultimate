import React, { useEffect, useMemo, useState } from 'react';
import { getJSON, fileUrl } from '../api';
import { Card, Section, Skeleton, AnimatedNumber, MotionIcon } from './ui';
import { Stagger, Reveal, motion, spring } from '../motion';
import { CHIP_KEYS, LABELS, fmtVal, fmtDur } from './settingsDiff';
import { Icon } from '../icons';

// ── Home ──────────────────────────────────────────────────────────────────
// Everything on this page already existed somewhere — the run history, the
// output listing, the queue snapshot, the GPU telemetry the processing HUD
// polls. What did not exist was a view that put them together, so there was no
// answer to "what happened last night, and what is this machine doing now"
// short of visiting three tabs and reading a terminal.
//
// It is deliberately NOT the landing tab. This is a tool people open to do one
// job, and putting a dashboard in front of that job would cost a click on every
// single launch to show information that matters between runs, not before them.

const fmtBytes = (n) => {
  if (!n || n < 0) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  const v = n / 1024 ** i;
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${u[i]}`;
};

const fmtAgo = (ts) => {
  if (!ts) return '—';
  const s = Math.max(0, (Date.now() - ts * 1000) / 1000);
  if (s < 90) return 'just now';
  const m = s / 60;
  if (m < 60) return `${Math.round(m)} min ago`;
  const h = m / 60;
  if (h < 24) return `${Math.round(h)} h ago`;
  const d = h / 24;
  if (d < 7) return `${Math.round(d)} d ago`;
  return new Date(ts * 1000).toLocaleDateString();
};

// One headline number. `tone` promotes a tile to the accent when it is
// reporting something live rather than historical.
const Stat = ({ icon: Ico, label, value, sub, tone, onClick, title }) => {
  const Tag = onClick ? motion.button : motion.div;
  return (
    <Tag
      {...(onClick ? { onClick, type: 'button', whileTap: { scale: 0.97 }, transition: spring.snappy, title } : {})}
      className={`text-left rounded-xl p-3.5 border apple-transition ${
        tone === 'accent'
          ? 'bg-[var(--accent)]/10 border-[var(--accent)]/30'
          : 'bg-white/[0.03] border-white/[0.08]'
      } ${onClick ? 'hover:border-white/25 cursor-pointer' : ''}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <MotionIcon icon={Ico} size="sm" variant={tone === 'accent' ? 'accent' : 'subtle'} />
        <span className="text-nano font-semibold uppercase tracking-[0.12em] text-white/45">{label}</span>
      </div>
      <div className={`text-title font-bold tabular-nums leading-none ${tone === 'accent' ? 'text-[var(--accent)]' : 'text-white/90'}`}>
        {value}
      </div>
      {sub && <div className="text-nano text-white/45 mt-1 truncate">{sub}</div>}
    </Tag>
  );
};

export default function Home({ progress, setTab, setSettings, notify }) {
  const [history, setHistory] = useState(null);   // null = still loading
  const [outputs, setOutputs] = useState(null);
  const [queue, setQueue] = useState(null);
  const [telemetry, setTelemetry] = useState(null);

  useEffect(() => {
    let live = true;
    // Each panel is independent and every one of them is optional: a fresh
    // install has no history, no outputs and no queue, and a CPU-only box has
    // no GPU telemetry. So these settle individually rather than as one
    // Promise.all that a single 404 would take down.
    const pull = (path, set, pick) => getJSON(path, { timeout: 8000 })
      .then((r) => { if (live) set(pick(r)); })
      .catch(() => { if (live) set(pick({})); });

    pull('/api/history', setHistory, (r) => r.entries || []);
    pull('/api/output', setOutputs, (r) => ({ files: r.files || [], path: r.output_path || '' }));
    pull('/api/queue', setQueue, (r) => r.jobs || []);
    pull('/api/system/telemetry', setTelemetry, (r) => (r.gpu ? r : null));

    return () => { live = false; };
  }, []);

  const stats = useMemo(() => {
    const runs = history || [];
    const files = outputs?.files || [];
    const jobs = queue || [];
    return {
      runs: runs.length,
      frames: runs.reduce((a, e) => a + (e.frames || 0), 0),
      renderTime: runs.reduce((a, e) => a + (e.duration_s || 0), 0),
      last: runs[0] || null,
      files: files.length,
      bytes: files.reduce((a, f) => a + (f.size || 0), 0),
      pending: jobs.filter((j) => !['finished', 'failed', 'stopped'].includes(j.status)).length,
    };
  }, [history, outputs, queue]);

  const loading = history === null && outputs === null;

  const rerun = (entry) => {
    if (!entry?.settings) return;
    setSettings((s) => ({ ...(s || {}), ...entry.settings }));
    notify(`Loaded the settings from ${fmtAgo(entry.time)}`);
    setTab('faceswap');
  };

  const vramPct = telemetry?.vram_total
    ? Math.round((telemetry.vram_used / telemetry.vram_total) * 100)
    : null;

  return (
    <Stagger className="space-y-6">
      {/* ── Hero ──────────────────────────────────────────────────────────
          Either an invitation to start, or — when something is already
          running — the state of that, so this page is never stale while the
          most interesting thing in the app is happening. */}
      <Reveal>
        <Card elevation="hero" className="p-6 md:p-7 flex flex-col md:flex-row md:items-center gap-5 justify-between">
          <div className="min-w-0">
            <h2 className="text-title font-bold text-white/95 tracking-tight">
              {progress?.processing
                ? (progress.paused ? 'Paused mid-render' : 'Rendering now')
                : stats.runs > 0 ? 'Ready when you are' : 'Nothing rendered yet'}
            </h2>
            <p className="text-compact text-white/45 mt-1.5 max-w-prose">
              {progress?.processing
                ? (progress.desc || 'Working…')
                : stats.runs > 0
                  ? `${stats.runs} run${stats.runs === 1 ? '' : 's'} so far, ${stats.frames.toLocaleString()} frames rendered in ${fmtDur(stats.renderTime) || 'no time at all'}.`
                  : 'Load a source face and a target, and the first run will show up here.'}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {progress?.processing ? (
              <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-[var(--accent)]/10 border border-[var(--accent)]/25">
                <span className="text-display font-bold text-[var(--accent)] tabular-nums leading-none">
                  <AnimatedNumber value={Math.round((progress.progress || 0) * 100)} suffix="%" />
                </span>
              </div>
            ) : null}
            <motion.button
              type="button"
              onClick={() => setTab('faceswap')}
              whileHover={{ y: -2, scale: 1.03 }}
              whileTap={{ scale: 0.96 }}
              transition={spring.snappy}
              className="px-5 py-3 rounded-xl bg-[var(--accent)] text-white font-bold text-compact shadow-[0_2px_10px_var(--accent-glow)] hover:shadow-[0_4px_18px_var(--accent-glow)] border border-white/10 flex items-center gap-2"
            >
              <Icon.faceswap size={15} />
              {progress?.processing ? 'Go to the run' : 'New swap'}
            </motion.button>
          </div>
        </Card>
      </Reveal>

      {/* ── Stat row ───────────────────────────────────────────────────── */}
      <Reveal>
        {loading ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-[92px]" />)}
          </div>
        ) : (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <Stat
              icon={Icon.history} label="Runs" value={stats.runs}
              sub={stats.last ? `last ${fmtAgo(stats.last.time)}` : 'none yet'}
              onClick={stats.runs ? () => setTab('history') : undefined}
              title="Open run history"
            />
            <Stat
              icon={Icon.meter} label="Last speed"
              value={stats.last?.fps ? `${stats.last.fps} fps` : '—'}
              sub={stats.last?.duration_s ? `over ${fmtDur(stats.last.duration_s)}` : 'no timing recorded'}
            />
            <Stat
              icon={Icon.outputs} label="Outputs" value={stats.files}
              sub={fmtBytes(stats.bytes)}
              onClick={stats.files ? () => setTab('gallery') : undefined}
              title="Open the outputs gallery"
            />
            {stats.pending > 0 ? (
              <Stat
                icon={Icon.queue} label="Queued" value={stats.pending} tone="accent"
                sub="waiting to render"
                onClick={() => setTab('faceswap')}
                title="Open the batch queue"
              />
            ) : (
              <Stat
                icon={Icon.meter} label="GPU"
                value={vramPct === null ? '—' : `${vramPct}%`}
                sub={telemetry?.gpu
                  ? `${telemetry.gpu}${telemetry.vram_total ? ` · ${telemetry.vram_used.toFixed(1)}/${telemetry.vram_total.toFixed(1)} GB` : ''}`
                  : 'no telemetry'}
                tone={vramPct !== null && vramPct > 85 ? 'accent' : undefined}
              />
            )}
          </div>
        )}
      </Reveal>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Recent runs ─────────────────────────────────────────────── */}
        <Reveal>
          <Section
            title="Recent runs"
            icon={Icon.history}
            action={history?.length ? (
              <button type="button" onClick={() => setTab('history')}
                      className="text-nano font-semibold text-white/45 hover:text-white apple-transition flex items-center gap-1">
                All <Icon.expand size={10} />
              </button>
            ) : null}
          >
            {history === null ? (
              <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-14" />)}</div>
            ) : history.length === 0 ? (
              <p className="text-compact text-white/45 py-6 text-center">
                Finished runs are recorded here with the settings they used.
              </p>
            ) : (
              <div className="space-y-2">
                {history.slice(0, 4).map((e) => (
                  <div key={e.id}
                       className="flex items-center gap-3 p-2.5 rounded-xl bg-white/[0.03] border border-white/[0.06] hover:border-white/15 apple-transition group/run">
                    <div className="min-w-0 flex-1">
                      <div className="text-compact font-semibold text-white/85 truncate">
                        {e.outputs?.[0] || 'Untitled run'}
                      </div>
                      <div className="flex flex-wrap items-center gap-1 mt-1">
                        <span className="text-nano text-white/45">{fmtAgo(e.time)}</span>
                        {/* The few settings that most define what a run looked
                            like — the same set the history tab chips. */}
                        {CHIP_KEYS.map((k) => {
                          const v = e.settings?.[k];
                          if (v === undefined || v === null || v === '' || v === false) return null;
                          return (
                            <span key={k} title={LABELS[k] || k}
                                  className="text-nano px-1.5 py-0.5 rounded bg-white/[0.06] text-white/45 truncate max-w-[9rem]">
                              {fmtVal(v)}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => rerun(e)}
                      title="Load these settings into the Face Swap tab"
                      className="shrink-0 px-2.5 py-1.5 rounded-lg text-nano font-bold text-white/50 hover:text-[var(--accent)] bg-white/[0.04] hover:bg-[var(--accent)]/12 border border-white/10 hover:border-[var(--accent)]/30 apple-transition opacity-0 group-hover/run:opacity-100 focus-visible:opacity-100"
                    >
                      Reuse
                    </button>
                  </div>
                ))}
              </div>
            )}
          </Section>
        </Reveal>

        {/* ── Recent outputs ──────────────────────────────────────────── */}
        <Reveal>
          <Section
            title="Latest outputs"
            icon={Icon.outputs}
            action={outputs?.files?.length ? (
              <button type="button" onClick={() => setTab('gallery')}
                      className="text-nano font-semibold text-white/45 hover:text-white apple-transition flex items-center gap-1">
                All <Icon.expand size={10} />
              </button>
            ) : null}
          >
            {outputs === null ? (
              <div className="grid grid-cols-3 gap-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="aspect-video" />)}</div>
            ) : outputs.files.length === 0 ? (
              <p className="text-compact text-white/45 py-6 text-center">
                Rendered files land in the Outputs tab.
              </p>
            ) : (
              <div className="grid grid-cols-3 gap-2">
                {outputs.files.slice(0, 6).map((f) => {
                  const targetPath = f.name.includes('/') || f.name.includes('\\') ? f.name : `${outputs.path}/${f.name}`;
                  const src = fileUrl(targetPath);
                  return (
                    <motion.button
                      key={f.name}
                      type="button"
                      onClick={() => setTab('gallery')}
                      whileHover={{ y: -3 }}
                      transition={spring.snappy}
                      title={f.name}
                      className="relative aspect-video rounded-lg overflow-hidden bg-black/40 border border-white/10 hover:border-[var(--accent)]/50 apple-transition group/out"
                    >
                      {f.kind === 'video' ? (
                        <video
                          src={src}
                          muted
                          preload="metadata"
                          className="w-full h-full object-cover pointer-events-none"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            if (e.target.nextElementSibling) e.target.nextElementSibling.style.display = 'flex';
                          }}
                        />
                      ) : (
                        <img
                          src={src}
                          alt={f.name}
                          loading="lazy"
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            if (e.target.nextElementSibling) e.target.nextElementSibling.style.display = 'flex';
                          }}
                        />
                      )}
                      <div className="hidden absolute inset-0 bg-white/5 flex-col items-center justify-center text-white/30 text-nano pointer-events-none p-2 text-center">
                        <Icon.still size={18} className="mb-1 opacity-50" />
                        <span className="truncate w-full">{f.name}</span>
                      </div>
                      <span className="absolute inset-x-0 bottom-0 px-1.5 py-1 bg-black/70 backdrop-blur-sm text-nano text-white/70 truncate text-left z-10">
                        {f.name}
                      </span>
                    </motion.button>
                  );
                })}
              </div>
            )}
          </Section>
        </Reveal>
      </div>
    </Stagger>
  );
}
