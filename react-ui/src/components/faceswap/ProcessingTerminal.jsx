import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Icon } from '../../icons';

/**
 * Terminal-style live feed shown inside the Preview box while a job runs.
 *
 * Three things make it readable on a long run:
 *
 *  - PART TABS. A resumable render writes numbered part files into the output
 *    folder (ROOP_RESUME_CHUNK frames each) and merges them at the end. Those
 *    parts are the only thing that survives a crash, so they are the natural
 *    chapters of a run: one tab each, showing the frame range it covers and
 *    whether it is finalized on disk. Every log line is tagged by the backend
 *    with the part that was open when it arrived, so selecting a tab shows that
 *    chapter's history.
 *  - A PINNED STATUS LINE. "Processing 6 412 / 109 376 (20.3 fps)" changes every
 *    poll; as history it is noise that buries the messages worth reading. It is
 *    rewritten in place at the bottom instead, the way a real terminal handles a
 *    progress line, and never enters the log.
 *  - Severity tone + copy, as before.
 */

const fmtMB = (b) => (!b ? '' : b >= 1073741824 ? `${(b / 1073741824).toFixed(1)} GB` : `${Math.round(b / 1048576)} MB`);
const fmtN = (n) => (n == null ? '' : Number(n).toLocaleString());
const SECTION_ORDER = [
  'SYSTEM', 'HARDWARE', 'PROVIDER', 'MODEL', 'PRECISION', 'PROCESSING',
  'POOLING', 'QUEUE', 'PROFILE', 'PERFORMANCE', 'WARNINGS', 'ERRORS',
  'PROJECT', 'CHECKPOINT',
];

const flattenValues = (value, prefix = '', depth = 0) => {
  if (value === null || value === undefined) return [[prefix, 'UNKNOWN']];
  if (Array.isArray(value)) {
    if (!value.length) return [[prefix, 'none']];
    if (value.every((item) => item === null || ['string', 'number', 'boolean'].includes(typeof item))) {
      return [[prefix, value.join(', ')]];
    }
    return [[prefix, `${value.length} item${value.length === 1 ? '' : 's'}`]];
  }
  if (typeof value !== 'object') return [[prefix, String(value)]];
  if (depth >= 2) return [[prefix, '{…}']];
  return Object.entries(value).flatMap(([key, child]) => flattenValues(
    child, prefix ? `${prefix}.${key}` : key, depth + 1,
  ));
};

function RuntimeSection({ name, section }) {
  const values = section?.values || {};
  const rows = flattenValues(values)
    .filter(([key]) => key !== 'items')
    .slice(0, 10);
  const status = section?.status || 'UNKNOWN';
  const statusTone = status === 'AVAILABLE'
    ? 'text-emerald-300'
    : status === 'NOT_APPLICABLE' ? 'text-white/35' : 'text-amber-300';
  return <div className="rounded-lg border border-white/[0.08] bg-black/25 px-2.5 py-2 min-w-0">
    <div className="flex items-center justify-between gap-2">
      <span className="text-nano font-bold tracking-[0.12em] text-white/60">{name}</span>
      <span className={`text-nano font-semibold uppercase ${statusTone}`}>{status}</span>
    </div>
    <div className="mt-1.5 space-y-0.5">
      {rows.length ? rows.map(([key, value]) => (
        <div key={key} className="flex items-baseline justify-between gap-2 text-micro leading-relaxed">
          <span className="min-w-0 truncate text-white/35">{key}</span>
          <span className="min-w-0 truncate text-right text-white/70" title={String(value)}>{String(value)}</span>
        </div>
      )) : <span className="text-micro text-white/35">No structured values</span>}
    </div>
    {section?.source && <div className="mt-1 truncate text-nano text-white/25" title={section.source}>source: {section.source}</div>}
  </div>;
}

export default function ProcessingTerminal({
  log = [], parts = [], statusLine = '', paused, className = '', bodyClass = 'h-40',
  runtime = null,
}) {
  const scrollRef = useRef(null);
  const bottomRef = useRef(null);
  const tabsRef = useRef(null);
  const [tab, setTab] = useState('all');          // 'all' | 'errors' | part index
  const [copied, setCopied] = useState(false);
  const [showReport, setShowReport] = useState(true);

  // A long render has one part per ROOP_RESUME_CHUNK frames, so the strip can
  // hold dozens: keep the newest (the one being written) in view.
  useEffect(() => {
    const el = tabsRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [parts.length]);

  // A part that disappears (new run) must not leave a dead tab selected.
  useEffect(() => {
    if (typeof tab === 'number' && !parts.some((p) => p.index === tab)) setTab('all');
  }, [parts, tab]);

  const lineTone = (msg) => {
    const m = (msg || '').toLowerCase();
    if (m.startsWith('⚠') || /error|fail|abort/.test(m)) return 'text-red-400 font-semibold';
    if (m.startsWith('✓') || /\bdone\b|\bpart \d+ written\b/.test(m)) return 'text-emerald-400 font-semibold';
    if (m.startsWith('▶') || /start/.test(m)) return 'text-[var(--accent)] font-semibold';
    if (/combin|encod|audio|mux|finaliz/.test(m)) return 'text-sky-300/90';
    if (/upscal|interpolat/.test(m)) return 'text-fuchsia-300/85';
    return 'text-white/70';
  };

  const structuredLineTone = (line) => {
    if (line?.level === 'ERROR' || line?.category === 'ERRORS') return 'text-red-400 font-semibold';
    if (line?.level === 'WARNING' || line?.category === 'WARNINGS') return 'text-amber-300 font-semibold';
    return lineTone(line?.msg);
  };

  const isError = (msg) => /error|fail|abort|⚠/i.test(msg || '');

  const shown = useMemo(() => {
    if (tab === 'errors') return log.filter((l) => isError(l.msg));
    if (typeof tab === 'number') return log.filter((l) => (l.part || 0) === tab);
    return log;
  }, [log, tab]);

  const errorCount = useMemo(() => log.filter((l) => isError(l.msg)).length, [log]);
  const activePart = typeof tab === 'number' ? parts.find((p) => p.index === tab) : null;
  const lastShownSeq = shown.length ? shown[shown.length - 1].seq : null;
  const authoritativeStatus = runtime?.status?.message || statusLine;

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    if (nearBottom) bottomRef.current?.scrollIntoView({ block: 'end' });
  }, [shown.length, lastShownSeq, tab]);

  const handleCopy = () => {
    // Copy what is on screen — a part tab is how you hand over the log for the
    // stretch of frames that actually went wrong.
    const text = shown.map((l) => `[${l.t}] ${l.msg}`).join('\n');
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const chip = (active) =>
    `px-2 py-0.5 rounded text-nano font-semibold transition-all whitespace-nowrap ${
      active
        ? 'bg-indigo-600/40 text-indigo-200 border border-indigo-500/50'
        : 'text-neutral-400 border border-transparent hover:text-white hover:bg-white/5'
    }`;

  return (
    <div className={`w-full rounded-xl border border-white/10 bg-black/60 overflow-hidden font-mono flex flex-col ${className}`}>
      {/* Title bar — traffic lights + part tabs + copy button */}
      <div className="shrink-0 flex items-center justify-between gap-3 px-3 py-1.5 border-b border-white/10 bg-white/[0.03]">
        <div className="flex items-center gap-3 min-w-0">
          <span className="flex items-center gap-1.5 shrink-0">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/70" />
          </span>

          {/* Tabs: All · errors (only when there are any) · one per output part.
              The count is spelled out next to "All" because the chips scroll:
              on a narrow box the part chips can sit off-screen, and then a
              missing tab strip is indistinguishable from a backend that is not
              reporting parts at all. */}
          <div ref={tabsRef} className="flex items-center gap-1 overflow-x-auto no-scrollbar">
            <button onClick={() => setTab('all')} className={chip(tab === 'all')} title="Everything this run printed">
              All
            </button>
            {runtime?.sections && (
              <button onClick={() => setShowReport((value) => !value)} className={chip(showReport)} title="Toggle structured runtime report">
                Report
              </button>
            )}
            <span className="shrink-0 px-1 text-nano font-semibold uppercase tracking-[0.14em] text-white/45">
              {parts.length ? `${parts.length} part${parts.length > 1 ? 's' : ''}` : 'no parts yet'}
            </span>
            {errorCount > 0 && (
              <button onClick={() => setTab('errors')} className={`${chip(tab === 'errors')} inline-flex items-center gap-1`} title="Warnings and errors only" aria-label={`Show only the ${errorCount} warnings and errors`}>
                <Icon.warning size={11} /> {errorCount}
              </button>
            )}
            {parts.map((p) => (
              <button
                key={p.index}
                onClick={() => setTab(p.index)}
                className={chip(tab === p.index)}
                title={`Part ${p.index} · frames ${fmtN(p.first)}-${fmtN(p.last)}${
                  p.done ? ` · ${fmtMB(p.bytes)} on disk` : ' · writing…'}`}
              >
                {p.index}
                <span className={p.done ? 'text-emerald-400/90' : 'text-amber-300/90'}>
                  {p.done ? ' ✓' : ' •'}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 px-2 py-0.5 rounded text-micro font-semibold bg-white/5 border border-white/10 text-neutral-300 hover:bg-white/10 hover:text-white transition-all active:scale-95"
            title={typeof tab === 'number' ? `Copy part ${tab}'s log` : 'Copy the log shown'}
          >
            <span>{copied ? '✓ Copied!' : 'Copy'}</span>
          </button>

          <span className="flex items-center gap-1.5 text-micro text-white/45">
            <span className={`h-1.5 w-1.5 rounded-full ${paused ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse'}`} />
            {paused ? 'paused' : 'live'}
          </span>
        </div>
      </div>

      {/* Selected part's header — what this chapter covers */}
      {activePart && (
        <div className="shrink-0 px-3 py-1 border-b border-white/[0.06] bg-white/[0.02] text-micro text-white/45 flex items-center gap-2">
          <span className="text-white/70 font-semibold">part {activePart.index}</span>
          <span>·</span>
          <span>frames {fmtN(activePart.first)}–{fmtN(activePart.last)}</span>
          <span>·</span>
          {activePart.done ? (
            <span className="text-emerald-400/80">finalized{activePart.bytes ? ` · ${fmtMB(activePart.bytes)}` : ''}{activePart.inherited ? ' · resumed' : ''}</span>
          ) : (
            <span className="text-amber-300/80">writing…</span>
          )}
        </div>
      )}

      {runtime?.sections && showReport && (
        <div className="shrink-0 max-h-64 overflow-y-auto border-b border-white/[0.08] bg-white/[0.015] p-2">
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <span className="text-nano font-semibold uppercase tracking-[0.16em] text-white/40">Structured runtime report</span>
            <span className="text-nano text-white/25">authoritative backend state</span>
          </div>
          <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 xl:grid-cols-3">
            {SECTION_ORDER.map((name) => <RuntimeSection key={name} name={name} section={runtime.sections[name]} />)}
          </div>
        </div>
      )}

      {/* Scrolling log body */}
      <div ref={scrollRef} className={`selectable ${bodyClass} min-h-0 overflow-y-auto px-3 py-2 text-mini leading-relaxed scroll-smooth`}>
        {shown.length === 0 ? (
          <div className="text-white/30 italic">
            {log.length === 0 ? 'waiting for output…' : 'nothing logged for this part yet…'}
          </div>
        ) : (
          shown.map((l) => (
            <div key={l.seq} className="flex gap-2 whitespace-pre-wrap break-words">
              <span className="shrink-0 text-white/25 tabular-nums">{l.t}</span>
              <span className="shrink-0 text-white/25">›</span>
              <span className={structuredLineTone(l)}>{l.msg}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Pinned status line — rewritten in place, never scrolled into history */}
      <div className="shrink-0 flex items-center gap-2 px-3 py-1.5 border-t border-white/10 bg-white/[0.03] text-mini">
        <span className="text-white/25">›</span>
        <span className="truncate tabular-nums text-white/80">{authoritativeStatus || (paused ? 'paused' : 'idle')}</span>
        <span className="ml-auto inline-block h-3.5 w-2 shrink-0 bg-[var(--accent)]/80 animate-pulse" />
      </div>
    </div>
  );
}
