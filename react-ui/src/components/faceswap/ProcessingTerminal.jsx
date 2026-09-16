import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Icon } from '../../icons';

/**
 * Terminal-style live feed shown inside the Preview box while a job runs.
 *
 * Polished features:
 *  - PART TABS: Grouping by resume chunk, with visual finalized badges.
 *  - PINNED STATUS LINE: Real-time progress rewritten in place.
 *  - ANSI ESCAPE PARSER: Colors and formatting rendered cleanly without escape code artifacts.
 *  - LOG SEARCH & FILTER: Quick real-time search across live terminal messages.
 *  - LOG COPY & DOWNLOAD: Copy to clipboard with fallback, or download as a standalone .log file.
 *  - AUTO-SCROLL LOCK/UNLOCK: Smart detection with a "Jump to latest" floating pill when viewing history.
 *  - EXPAND / COLLAPSE: Toggle between compact split view and full-height log inspection.
 *  - THEMED HIGHLIGHTS: Adapts to the active workspace theme accents.
 */

const fmtMB = (b) => (!b ? '' : b >= 1073741824 ? `${(b / 1073741824).toFixed(1)} GB` : `${Math.round(b / 1048576)} MB`);
const fmtN = (n) => (n == null ? '' : Number(n).toLocaleString());
const SECTION_ORDER = [
  'SYSTEM', 'HARDWARE', 'PROVIDER', 'MODEL', 'PRECISION', 'PROCESSING',
  'POOLING', 'QUEUE', 'PROFILE', 'PERFORMANCE', 'WARNINGS', 'ERRORS',
  'PROJECT', 'CHECKPOINT',
];

const ANSI_COLOR_MAP = {
  '30': 'text-neutral-500',
  '31': 'text-red-400',
  '32': 'text-emerald-400',
  '33': 'text-amber-300',
  '34': 'text-sky-400',
  '35': 'text-fuchsia-400',
  '36': 'text-cyan-400',
  '37': 'text-white/90',
  '90': 'text-neutral-400',
  '91': 'text-red-300',
  '92': 'text-emerald-300',
  '93': 'text-amber-200',
  '94': 'text-sky-300',
  '95': 'text-fuchsia-300',
  '96': 'text-cyan-300',
  '97': 'text-white',
};

// eslint-disable-next-line no-control-regex
function parseAnsiTokens(rawText) {
  /* eslint-disable no-control-regex */
  if (typeof rawText !== 'string') return [{ text: String(rawText || ''), classes: '' }];
  const text = rawText.replace(/\r/g, '');
  if (!text.includes('\u001b') && !text.includes('\x1b')) {
    return [{ text, classes: '' }];
  }
  // Matches all CSI sequences: SGR (m) and control commands (e.g. 2K, 1A)
  const regex = /(?:\u001b|\x1b)\[([0-9;]*)([a-zA-Z])/g;
  const parts = [];
  let lastIndex = 0;
  let currentClasses = [];
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      const chunk = text.slice(lastIndex, match.index);
      if (chunk) {
        parts.push({ text: chunk, classes: currentClasses.join(' ') });
      }
    }
    const [, codesStr, cmd] = match;
    if (cmd === 'm') {
      const codes = codesStr ? codesStr.split(';') : ['0'];
      for (let i = 0; i < codes.length; i++) {
        const code = codes[i];
        if (code === '0' || code === '') {
          currentClasses = [];
        } else if (code === '1') {
          if (!currentClasses.includes('font-bold')) currentClasses.push('font-bold');
        } else if (code === '2') {
          if (!currentClasses.includes('opacity-60')) currentClasses.push('opacity-60');
        } else if (code === '3') {
          if (!currentClasses.includes('italic')) currentClasses.push('italic');
        } else if (code === '4') {
          if (!currentClasses.includes('underline')) currentClasses.push('underline');
        } else if (code === '39') {
          currentClasses = currentClasses.filter((c) => !c.startsWith('text-'));
        } else if (code === '38' || code === '48') {
          // Skip extended color sequences (38;5;n or 38;2;r;g;b)
          if (codes[i + 1] === '5') i += 2;
          else if (codes[i + 1] === '2') i += 4;
        } else if (ANSI_COLOR_MAP[code]) {
          currentClasses = currentClasses.filter((c) => !c.startsWith('text-'));
          currentClasses.push(ANSI_COLOR_MAP[code]);
        }
      }
    }
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    const chunk = text.slice(lastIndex);
    if (chunk) {
      parts.push({ text: chunk, classes: currentClasses.join(' ') });
    }
  }
  return parts.length ? parts : [{ text: '', classes: '' }];
}

function FormattedLogMessage({ msg, toneClass }) {
  const tokens = useMemo(() => parseAnsiTokens(msg), [msg]);
  if (tokens.length === 1 && !tokens[0].classes) {
    return <span className={toneClass}>{tokens[0].text}</span>;
  }
  return (
    <span className={toneClass}>
      {tokens.map((t, idx) => (
        <span key={idx} className={t.classes || undefined}>{t.text}</span>
      ))}
    </span>
  );
}

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
    ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/25'
    : status === 'NOT_APPLICABLE' ? 'text-white/35 bg-white/5 border-white/10' : 'text-amber-300 bg-amber-500/10 border-amber-500/25';
  return (
    <div className="rounded-xl border border-white/[0.08] bg-black/40 px-3 py-2 min-w-0 shadow-sm backdrop-blur-sm">
      <div className="flex items-center justify-between gap-2 border-b border-white/5 pb-1.5">
        <span className="text-nano font-bold tracking-[0.14em] text-white/70 uppercase">{name}</span>
        <span className={`text-nano font-semibold uppercase px-1.5 py-0.5 rounded border ${statusTone}`}>{status}</span>
      </div>
      <div className="mt-1.5 space-y-0.5">
        {rows.length ? rows.map(([key, value]) => (
          <div key={key} className="flex items-baseline justify-between gap-2 text-micro leading-relaxed">
            <span className="min-w-0 truncate text-white/40">{key}</span>
            <span className="min-w-0 truncate text-right text-white/80 font-mono" title={String(value)}>{String(value)}</span>
          </div>
        )) : <span className="text-micro text-white/35 italic">No structured values</span>}
      </div>
      {section?.source && <div className="mt-1 truncate text-nano text-white/30 border-t border-white/5 pt-1" title={section.source}>source: {section.source}</div>}
    </div>
  );
}

export default function ProcessingTerminal({
  log = [], parts = [], statusLine = '', paused, className = '', bodyClass = 'flex-1 min-h-[140px]',
  runtime = null, expanded = false, onToggleExpand = null,
}) {
  const scrollRef = useRef(null);
  const bottomRef = useRef(null);
  const tabsRef = useRef(null);
  const searchInputRef = useRef(null);
  const [tab, setTab] = useState('all');          // 'all' | 'errors' | part index
  const [copied, setCopied] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);

  const isError = (line) => {
    if (!line) return false;
    if (line.level === 'ERROR' || line.level === 'WARNING') return true;
    if (line.category === 'ERRORS' || line.category === 'WARNINGS') return true;
    return /error|fail|abort|⚠/i.test(line.msg || '');
  };

  const errorCount = useMemo(() => log.filter(isError).length, [log]);

  // A long render has one part per ROOP_RESUME_CHUNK frames, so the strip can
  // hold dozens: keep the newest (the one being written) in view.
  useEffect(() => {
    const el = tabsRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [parts.length]);

  // A part that disappears (new run) or cleared errors must not leave a dead tab selected.
  useEffect(() => {
    if (typeof tab === 'number' && !parts.some((p) => p.index === tab)) setTab('all');
    if (tab === 'errors' && errorCount === 0) setTab('all');
  }, [parts, tab, errorCount]);

  const lineTone = (msg) => {
    const m = (msg || '').toLowerCase();
    if (m.startsWith('⚠') || /error|fail|abort/.test(m)) return 'text-red-400 font-semibold';
    if (m.startsWith('✓') || /\bdone\b|\bpart \d+ written\b/.test(m)) return 'text-emerald-400 font-semibold';
    if (m.startsWith('▶') || /start/.test(m)) return 'text-[var(--accent)] font-semibold';
    if (/combin|encod|audio|mux|finaliz/.test(m)) return 'text-sky-300/90';
    if (/upscal|interpolat/.test(m)) return 'text-fuchsia-300/85';
    return 'text-white/75';
  };

  const structuredLineTone = (line) => {
    if (line?.level === 'ERROR' || line?.category === 'ERRORS') return 'text-red-400 font-semibold';
    if (line?.level === 'WARNING' || line?.category === 'WARNINGS') return 'text-amber-300 font-semibold';
    return lineTone(line?.msg);
  };

  const baseShown = useMemo(() => {
    if (tab === 'errors') return log.filter(isError);
    if (typeof tab === 'number') return log.filter((l) => (l.part || 0) === tab);
    return log;
  }, [log, tab]);

  const shown = useMemo(() => {
    if (!searchQuery.trim()) return baseShown;
    const q = searchQuery.trim().toLowerCase();
    return baseShown.filter((l) => (l.msg || '').toLowerCase().includes(q));
  }, [baseShown, searchQuery]);

  const activePart = typeof tab === 'number' ? parts.find((p) => p.index === tab) : null;
  const lastShownSeq = shown.length ? shown[shown.length - 1].seq : null;
  const authoritativeStatus = runtime?.status?.message || statusLine;

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 36;
    setAutoScroll(nearBottom);
  }, []);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [shown.length, lastShownSeq, tab, autoScroll]);

  const cleanLogText = (msg) => String(msg || '').replace(/(?:\u001b|\x1b)\[[0-9;]*[a-zA-Z]|\r/g, '');

  const handleCopy = () => {
    if (!shown.length) return;
    const text = shown.map((l) => `[${l.t}] ${cleanLogText(l.msg)}`).join('\n');
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }).catch(() => fallbackCopy(text));
    } else {
      fallbackCopy(text);
    }
  };

  const fallbackCopy = (text) => {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
      document.execCommand('copy');
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* ignore */ }
    document.body.removeChild(ta);
  };

  const handleDownload = () => {
    if (!shown.length) return;
    const text = shown.map((l) => `[${l.t}] ${cleanLogText(l.msg)}`).join('\n');
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `roop_render_log_${typeof tab === 'number' ? `part_${tab}` : tab}_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.log`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const chip = (active) =>
    `px-2.5 py-1 rounded-lg text-nano font-semibold transition-all whitespace-nowrap select-none ${
      active
        ? 'bg-[var(--accent)]/25 text-white border border-[var(--accent)]/50 shadow-sm'
        : 'text-neutral-400 border border-transparent hover:text-white hover:bg-white/5'
    }`;

  return (
    <div className={`relative w-full rounded-2xl border border-white/10 bg-black/75 shadow-2xl backdrop-blur-md overflow-hidden font-mono flex flex-col ${className}`}>
      {/* Title bar — traffic lights + part tabs + search + copy/download buttons */}
      <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 px-3.5 py-2 border-b border-white/10 bg-white/[0.03]">
        <div className="flex items-center gap-3 min-w-0">
          <span className="flex items-center gap-1.5 shrink-0" aria-hidden="true">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500/80 shadow-[0_0_6px_rgba(239,68,68,0.4)]" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400/80 shadow-[0_0_6px_rgba(251,191,36,0.4)]" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/80 shadow-[0_0_6px_rgba(16,185,129,0.4)]" />
          </span>

          {/* Tabs: All · Report · Errors · Part chapters */}
          <div ref={tabsRef} className="flex items-center gap-1 overflow-x-auto no-scrollbar py-0.5">
            <button onClick={() => setTab('all')} className={chip(tab === 'all')} title="Everything this run printed">
              All
            </button>
            {runtime?.sections && (
              <button
                onClick={() => setShowReport((value) => !value)}
                className={chip(showReport)}
                title="Toggle structured runtime report"
              >
                Report {showReport ? '▾' : '▸'}
              </button>
            )}
            <span className="shrink-0 px-1 text-nano font-semibold uppercase tracking-[0.14em] text-white/45">
              {parts.length ? `${parts.length} part${parts.length > 1 ? 's' : ''}` : 'no parts yet'}
            </span>
            {errorCount > 0 && (
              <button
                onClick={() => setTab('errors')}
                className={`${chip(tab === 'errors')} inline-flex items-center gap-1 text-red-300`}
                title="Warnings and errors only"
                aria-label={`Show only the ${errorCount} warnings and errors`}
              >
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

        {/* Right action tools: search input, copy, download, expand view */}
        <div className="flex items-center gap-2 shrink-0">
          {showSearch ? (
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-lg bg-white/10 border border-white/20 text-xs">
              <Icon.search size={12} className="text-white/40" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') {
                    e.stopPropagation();
                    setSearchQuery('');
                    setShowSearch(false);
                  }
                }}
                placeholder="Filter logs…"
                className="bg-transparent text-white placeholder:text-white/30 text-micro focus:outline-none w-28 sm:w-36 font-mono"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="text-white/40 hover:text-white text-nano"
                  aria-label="Clear search filter"
                >
                  ✕
                </button>
              )}
              <button
                type="button"
                onClick={() => { setShowSearch(false); setSearchQuery(''); }}
                className="text-white/40 hover:text-white text-nano border-l border-white/10 pl-1"
                aria-label="Close search input"
              >
                Done
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => { setShowSearch(true); setTimeout(() => searchInputRef.current?.focus(), 50); }}
              className="flex items-center gap-1 px-2 py-1 rounded-lg text-micro font-semibold bg-white/5 border border-white/10 text-neutral-300 hover:bg-white/10 hover:text-white transition-all active:scale-95"
              title="Filter terminal output"
              aria-label="Search and filter logs"
            >
              <Icon.search size={12} />
              <span className="hidden sm:inline">Filter</span>
            </button>
          )}

          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1 px-2 py-1 rounded-lg text-micro font-semibold bg-white/5 border border-white/10 text-neutral-300 hover:bg-white/10 hover:text-white transition-all active:scale-95"
            title={typeof tab === 'number' ? `Copy part ${tab}'s log` : 'Copy the log shown'}
            aria-label="Copy terminal log"
          >
            <span>{copied ? '✓ Copied!' : 'Copy'}</span>
          </button>

          <button
            type="button"
            onClick={handleDownload}
            className="flex items-center gap-1 px-2 py-1 rounded-lg text-micro font-semibold bg-white/5 border border-white/10 text-neutral-300 hover:bg-white/10 hover:text-white transition-all active:scale-95"
            title="Download log file"
            aria-label="Download terminal log file"
          >
            <Icon.download size={12} />
            <span className="hidden sm:inline">Log</span>
          </button>

          {onToggleExpand && (
            <button
              type="button"
              onClick={onToggleExpand}
              className={`flex items-center gap-1 px-2 py-1 rounded-lg text-micro font-semibold border transition-all active:scale-95 ${
                expanded
                  ? 'bg-[var(--accent)]/20 border-[var(--accent)]/50 text-white'
                  : 'bg-white/5 border-white/10 text-neutral-300 hover:bg-white/10 hover:text-white'
              }`}
              title={expanded ? 'Restore split view' : 'Expand terminal to full view'}
              aria-label={expanded ? 'Restore split view' : 'Expand terminal'}
            >
              <Icon.split size={12} className={expanded ? 'rotate-90' : ''} />
              <span className="hidden sm:inline">{expanded ? 'Split' : 'Expand'}</span>
            </button>
          )}

          <span className="flex items-center gap-1.5 text-micro text-white/50 pl-1 border-l border-white/10">
            <span className={`h-1.5 w-1.5 rounded-full ${paused ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse'}`} />
            <span className="uppercase text-nano font-bold tracking-wider">{paused ? 'paused' : 'live'}</span>
          </span>
        </div>
      </div>

      {/* Selected part's header */}
      {activePart && (
        <div className="shrink-0 px-3.5 py-1.5 border-b border-white/[0.06] bg-white/[0.02] text-micro text-white/45 flex items-center gap-2">
          <span className="text-white/80 font-bold">Chapter {activePart.index}</span>
          <span>·</span>
          <span>frames {fmtN(activePart.first)}–{fmtN(activePart.last)}</span>
          <span>·</span>
          {activePart.done ? (
            <span className="text-emerald-400/90 font-medium">finalized{activePart.bytes ? ` · ${fmtMB(activePart.bytes)}` : ''}{activePart.inherited ? ' · resumed' : ''}</span>
          ) : (
            <span className="text-amber-300/90 font-medium">writing to disk…</span>
          )}
        </div>
      )}

      {/* Search filter matches banner */}
      {searchQuery && (
        <div className="shrink-0 px-3.5 py-1 border-b border-white/10 bg-[var(--accent)]/10 text-nano text-white/70 flex items-center justify-between">
          <span>Filtering by &ldquo;{searchQuery}&rdquo; — {shown.length} matching line{shown.length === 1 ? '' : 's'}</span>
          <button type="button" onClick={() => setSearchQuery('')} className="hover:text-white underline">Clear</button>
        </div>
      )}

      {/* Structured runtime report drawer */}
      {runtime?.sections && showReport && (
        <div className="shrink-0 max-h-72 overflow-y-auto border-b border-white/[0.08] bg-black/50 p-3 custom-scrollbar animate-slide-down">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-nano font-bold uppercase tracking-[0.16em] text-white/60">Structured runtime report</span>
            <span className="text-nano text-white/35 font-mono">authoritative backend state</span>
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {SECTION_ORDER.map((name) => <RuntimeSection key={name} name={name} section={runtime.sections[name]} />)}
          </div>
        </div>
      )}

      {/* Scrolling log body */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className={`selectable ${bodyClass} min-h-0 overflow-y-auto px-3.5 py-2.5 text-mini leading-relaxed custom-scrollbar`}
      >
        {shown.length === 0 ? (
          <div className="text-white/30 italic py-4 text-center">
            {log.length === 0 ? 'waiting for engine output…' : searchQuery ? 'no log entries match the search filter' : 'nothing logged for this chapter yet…'}
          </div>
        ) : (
          shown.map((l) => (
            <div key={l.seq} className="flex gap-2.5 py-0.5 whitespace-pre-wrap break-words hover:bg-white/[0.02] rounded px-1 -mx-1 transition-colors">
              <span className="shrink-0 text-white/30 tabular-nums font-mono text-micro select-none">{l.t}</span>
              <span className="shrink-0 text-white/25 select-none">›</span>
              <FormattedLogMessage msg={l.msg} toneClass={structuredLineTone(l)} />
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Floating auto-scroll snap button */}
      {!autoScroll && shown.length > 0 && (
        <button
          type="button"
          onClick={() => {
            setAutoScroll(true);
            scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
          }}
          className="absolute bottom-12 right-4 z-20 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[var(--accent)] hover:brightness-110 text-white text-nano font-bold shadow-2xl border border-white/20 transition-all active:scale-95 animate-bounce cursor-pointer"
          aria-label="Scroll to latest output"
        >
          <span>↓ Jump to latest</span>
        </button>
      )}

      {/* Pinned status line — rewritten in place, never scrolled into history */}
      <div className="shrink-0 flex items-center gap-2 px-3.5 py-2 border-t border-white/10 bg-black/40 text-mini">
        <span className="text-[var(--accent)] font-bold">›</span>
        <span className="truncate tabular-nums text-white/90 font-semibold">{authoritativeStatus || (paused ? 'paused' : 'idle')}</span>
        <span className="ml-auto inline-block h-3.5 w-2 shrink-0 bg-[var(--accent)] shadow-[0_0_6px_var(--accent-glow)] animate-pulse" />
      </div>
    </div>
  );
}
