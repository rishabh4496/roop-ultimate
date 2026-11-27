import { useShallow } from 'zustand/react/shallow';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useSystemTelemetryPoller } from '../faceswap/useTelemetry';
import { LiveText } from '../LiveTelemetry';
import { Icon } from '../../icons';
import { fpsBasisWord, useTelemetryStore, videoIoTextOf } from '../../store/telemetryStore';
import { pct, readingOf } from './telemetryReading';

// ── Live hardware telemetry: one pill, one popover ──────────────────────────
//
// This is the ONLY place the app shows GPU / VRAM / CPU. It replaces the
// "Hardware telemetry HUD" banner under the header, the GPU tile on Home and the
// GPU/CPU/RAM strip in Face Swap's Live Telemetry section — three views of one
// poll that each had to be kept honest separately.
//
// Data is the shared /api/system/telemetry poll (faceswap/useTelemetry: one
// request every 3 s for the whole app, however many components ask). What is NOT
// invented: `gpu_util`, `gpu_temp` and `gpu_power` come from nvidia-smi and are
// simply absent without it, in which case the pill shows VRAM instead of a made
// up utilisation figure and the popover omits the rows.
//
// THE PILL ITSELF RE-RENDERS NOTHING. It is mounted on every route for the whole
// session, and e2e/frame-budget.spec.js holds an idle tab to <= 6 React commits a
// second (Settings idled at exactly 6 before this existed). So its numbers are
// written straight into text nodes (LiveText) and its sparkline is drawn from a
// store subscription; React only renders the popover, and only while it is open.

const HISTORY = 60;                                  // samples kept: 3 minutes at the 3 s poll
const history = { primary: [], cpu: [] };            // module-level, so a remount keeps its trail

const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(d));

function push(list, v) {
  if (v === null || v === undefined) return;
  list.push(v);
  if (list.length > HISTORY) list.shift();
}

function Sparkline() {
  const ref = useRef(null);
  useEffect(() => {
    const draw = () => {
      const cv = ref.current;
      if (!cv) return;
      const dpr = window.devicePixelRatio || 1;
      const w = 76;
      const h = 22;
      if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
      const ctx = cv.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#E94560';
      const line = (data, color, fill) => {
        if (data.length < 2) return;
        const step = w / (HISTORY - 1);
        const x0 = w - (data.length - 1) * step;
        ctx.beginPath();
        data.forEach((v, i) => {
          const x = x0 + i * step;
          const y = h - 1 - (Math.min(100, Math.max(0, v)) / 100) * (h - 3);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.4;
        ctx.lineJoin = 'round';
        ctx.stroke();
        if (fill) {
          ctx.lineTo(w, h);
          ctx.lineTo(x0, h);
          ctx.closePath();
          ctx.globalAlpha = 0.16;
          ctx.fillStyle = color;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      };
      line(history.cpu, 'rgba(255,255,255,0.45)', false);
      line(history.primary, accent, true);
    };

    draw();
    // One push + one redraw per NEW sample (the poller writes a fresh object each time). The listener runs
    // inside setState, before any React work, so nothing here costs a commit.
    return useTelemetryStore.subscribe((state, prev) => {
      if (state.system === prev.system) return;
      const r = readingOf(state.system);
      if (r) { push(history.primary, r.primary); push(history.cpu, r.cpu); }
      draw();
    });
  }, []);
  return <canvas ref={ref} aria-hidden="true" className="h-[22px] w-[76px] shrink-0 max-xl:hidden" />;
}

function Meter({ label, value, text, tone = 'var(--accent)' }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-3 text-mini">
        <span className="text-muted">{label}</span>
        <span className="font-mono font-semibold tabular-nums text-white/85">{text}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
        <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${value ?? 0}%`, background: tone }} />
      </div>
    </div>
  );
}

const Row = ({ label, children }) => (
  <div className="flex items-baseline justify-between gap-3 text-mini">
    <span className="text-muted">{label}</span>
    <span className="min-w-0 truncate text-right font-mono font-semibold text-white/85">{children}</span>
  </div>
);

// The popover's contents. A component of its own so that subscribing to the sample (a re-render every 3 s)
// happens only while the popover is open.
function Details({ engine, onOpenDiagnostics, onClose }) {
  const sys = useTelemetryStore(useShallow((s) => s.system));
  const reading = readingOf(sys);
  if (!sys) return <div className="py-4 text-center text-mini text-muted">Connecting to hardware diagnostics…</div>;
  return (
    <>
      <div>
        <div className="truncate text-compact font-bold text-white/90">{sys.gpu || 'CPU only'}</div>
        {sys.gpu_temp !== undefined && (
          <div className="mt-0.5 text-mini text-muted">
            {fmt(sys.gpu_temp)}°C{sys.gpu_power !== undefined ? ` · ${fmt(sys.gpu_power)} W` : ''}
          </div>
        )}
      </div>
      {reading?.gpu !== null && <Meter label="GPU utilisation" value={reading.gpu} text={`${fmt(reading.gpu)}%`} />}
      {sys.vram_total > 0 && (
        <Meter label="VRAM" value={reading.vram} text={`${fmt(sys.vram_used, 1)} / ${fmt(sys.vram_total, 1)} GB`} tone="#34d399" />
      )}
      <Meter label="CPU" value={reading?.cpu ?? 0} text={`${fmt(reading?.cpu)}%`} tone="#fb923c" />
      {sys.ram_total > 0 && (
        <Meter label="System RAM" value={pct(sys.ram_used, sys.ram_total)} text={`${fmt(sys.ram_used, 1)} / ${fmt(sys.ram_total, 1)} GB`} tone="#60a5fa" />
      )}
      <div className="space-y-1 border-t border-white/10 pt-2.5">
        {sys.process_rss > 0 && <Row label="App memory">{fmt(sys.process_rss, 2)} GB</Row>}
        <Row label="Python threads">{fmt(sys.threads)}</Row>
        {engine?.provider && <Row label="Execution provider">{engine.provider}</Row>}
        {engine?.workers && <Row label="Configured workers">{engine.workers}</Row>}
        <Row label="Engine connection">{engine?.connection || '—'}</Row>
      </div>
      {/* The run's own readouts used to sit in Face Swap's Live Telemetry card.
          They are here now, with the hardware, and empty (hidden) when no job
          has registered a backend. The Video I/O line is also where a hardware
          fallback is announced, so it must stay on screen somewhere. */}
      <LiveText
        as="div"
        className="font-mono text-mini leading-snug text-white/60 empty:hidden"
        select={(s) => videoIoTextOf(s.run)}
      />
      <LiveText
        as="div"
        className="font-mono text-mini leading-snug text-white/60 empty:hidden"
        select={(s) => (s.run.fps_now
          ? `Render now: ${s.run.fps_now.toFixed(1)} fps ${fpsBasisWord(s.run.fps_basis)} · ${s.run.frame_ms} ms/frame`
          : '')}
      />
      {onOpenDiagnostics && (
        <button
          type="button"
          onClick={() => { onClose(); onOpenDiagnostics(); }}
          className="w-full rounded-lg border border-white/10 bg-white/[0.04] py-1.5 text-mini font-semibold text-white/80 transition-colors hover:bg-white/[0.08] hover:text-white"
        >
          Open runtime diagnostics
        </button>
      )}
    </>
  );
}

export default function TelemetryPill({ engine, onOpenDiagnostics }) {
  // Keeps the shared poll running; renders nothing when a sample lands.
  useSystemTelemetryPoller(3000);
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);
  const btnRef = useRef(null);

  const close = useCallback(() => { setOpen(false); btnRef.current?.focus(); }, []);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (!wrapRef.current?.contains(e.target)) setOpen(false); };
    // Capture + stopPropagation: Face Swap owns a bare Escape at window level, and
    // the key that dismisses this popover must not also reach it.
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      e.stopPropagation();
      close();
    };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [open, close]);

  // The button's accessible name is its CONTENT (the sr-only prefix + the live text nodes), so it follows the
  // numbers without a React render and without an aria-label that would have to be re-set on each one.
  const part = (pick) => (s) => {
    const r = readingOf(s.system);
    return r ? pick(r) : '';
  };

  return (
    <div ref={wrapRef} className="relative shrink-0">
      <button
        ref={btnRef}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Hardware telemetry (click for details)"
        className="flex h-8 items-center gap-2.5 rounded-lg border border-white/10 bg-black/25 px-2.5 font-mono text-mini font-semibold tabular-nums text-white/80 transition-colors hover:border-white/25 hover:text-white"
      >
        <Icon.cpu size={14} className="text-accent" />
        <span className="sr-only">Hardware telemetry:</span>
        <LiveText select={(s) => (s.system ? part((r) => (r.gpu !== null ? `GPU ${fmt(r.gpu)}%` : ''))(s) : 'GPU —')} className="empty:hidden" />
        <LiveText select={part((r) => `VRAM ${fmt(r.vram)}%`)} className="empty:hidden" />
        <LiveText select={part((r) => `CPU ${fmt(r.cpu)}%`)} className="empty:hidden max-lg:hidden" />
        <Sparkline />
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Hardware telemetry"
          style={{ background: 'linear-gradient(var(--card-bg), var(--card-bg)), var(--bg-base)' }}
          className="glass-panel animate-slide-up absolute right-0 top-full z-50 mt-2 w-[300px] origin-top-right space-y-3 rounded-xl p-3.5 shadow-xl"
        >
          <Details engine={engine} onOpenDiagnostics={onOpenDiagnostics} onClose={() => setOpen(false)} />
        </div>
      )}
    </div>
  );
}
