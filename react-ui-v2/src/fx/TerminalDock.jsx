import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getJSON } from '@engine/api';
import { useTelemetryStore, selectRenderProgress } from '@engine/store/telemetryStore';
import { useFxStore, TRANSITIONS } from './fxStore';
import { termLog, termLines, onTermLines, clearTermLines } from './termFeed';
import { FILTERS, foldEngineLog, matchesFilter } from './termModel';

// ── The terminal dock ───────────────────────────────────────────────────────
//
// A tail -f of everything happening, docked to the bottom of the workspace: the ENGINE's own log
// (the backend's rolling 250-line tail from /api/progress, merged by `seq` so a line is printed
// once however many polls carry it) interleaved with the CLIENT's (navigation, toasts, connection
// changes - termFeed.js). Collapsed it is one line: the latest entry typing itself out beside a
// live throughput ticker. Expanded it is a resizable terminal with severity filters and a prompt
// that takes a handful of UI commands (help, goto, fx, clear, palette, filter).
//
// It never starts, stops or changes a job: the prompt's commands are navigation and appearance
// only. 1.0's Processing terminal is still where a run's log lives, unchanged.

const DOM_CAP = 400;
const IDLE_POLL_MS = 4000;

const GLYPH = { info: '·', ok: '✓', warn: '!', error: '✗', perf: '▲', cmd: '❯' };

function Line({ line, fresh, typewriter }) {
  const chars = Math.min(140, line.msg.length + 4);
  return (
    <div
      className={`v2-term-line lvl-${line.level}${fresh && typewriter ? ' is-typing' : ''}`}
      style={fresh && typewriter ? { '--chars': chars } : undefined}
      data-origin={line.origin}
    >
      <span className="v2-term-t">{line.t}</span>
      <span className="v2-term-glyph" aria-hidden="true">{GLYPH[line.level] || '·'}</span>
      <span className="v2-term-src">{line.src}</span>
      <span className="v2-term-msg selectable">{line.msg}</span>
    </div>
  );
}

function Ticker({ processing }) {
  // Primitives only: a selector returning a fresh object would re-render on every store write.
  const fraction = useTelemetryStore((s) => selectRenderProgress(s).fraction);
  const label = useTelemetryStore((s) => selectRenderProgress(s).label);
  const fps = useTelemetryStore((s) => Number(s.run?.fps_now ?? s.run?.fps) || 0);
  if (!processing) return <span className="v2-term-tick is-idle">idle · engine standing by</span>;
  const pct = fraction == null ? '' : `${Math.round(fraction * 100)}%`;
  return (
    <span className="v2-term-tick">
      <span className="v2-dot-live" aria-hidden="true" />
      {label || 'rendering'} {pct && <b>{pct}</b>} {fps > 0 && <span>· {fps.toFixed(1)} fps</span>}
    </span>
  );
}

export default function TerminalDock({ progress, onNavigate, tabs, onOpenPalette }) {
  const open = useFxStore((s) => s.terminalOpen);
  const height = useFxStore((s) => s.terminalHeight);
  const typewriter = useFxStore((s) => s.typewriter);
  const crt = useFxStore((s) => s.crt);
  const setFx = useFxStore((s) => s.setFx);
  const toggleFx = useFxStore((s) => s.toggleFx);

  const [uiLines, setUiLines] = useState(termLines);
  const [filter, setFilter] = useState('ALL');
  const [follow, setFollow] = useState(true);
  const [input, setInput] = useState('');
  const historyRef = useRef([]);
  const histIdx = useRef(-1);
  const engineRef = useRef(new Map());
  const [engineVersion, setEngineVersion] = useState(0);
  const mountedAt = useRef(Date.now());
  const scrollRef = useRef(null);

  useEffect(() => onTermLines(setUiLines), []);

  // The dock is fixed to the window bottom; the screens and the nav dock read --term-reserve to
  // keep clear of it.
  useEffect(() => {
    document.documentElement.style.setProperty('--term-reserve', `${open ? height : 33}px`);
    return () => document.documentElement.style.removeProperty('--term-reserve');
  }, [open, height]);

  // Fold an engine log tail in by seq (a line arrives once however many polls repeat it).
  const ingest = useCallback((log) => {
    if (foldEngineLog(engineRef.current, log)) setEngineVersion((v) => v + 1);
  }, []);

  useEffect(() => { ingest(progress?.log); }, [progress?.log, ingest]);

  // While idle and open, App's poll is off - read the tail ourselves (GET only, nothing written).
  useEffect(() => {
    if (!open || progress?.processing) return undefined;
    let alive = true;
    const tick = async () => {
      try {
        const pr = await getJSON('/api/progress', { timeout: 6000 });
        if (alive) ingest(pr?.log);
      } catch { /* the connection banner already says so */ }
    };
    tick();
    const h = setInterval(tick, IDLE_POLL_MS);
    return () => { alive = false; clearInterval(h); };
  }, [open, progress?.processing, ingest]);

  const all = useMemo(() => {
    const merged = [...engineRef.current.values(), ...uiLines];
    merged.sort((a, b) => a.at - b.at);
    return merged;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- engineVersion is the engine map's revision
  }, [uiLines, engineVersion]);

  const shown = useMemo(() => all.filter((l) => matchesFilter(l, filter)).slice(-DOM_CAP), [all, filter]);
  const counts = useMemo(() => {
    const c = { warn: 0, error: 0, perf: 0 };
    all.forEach((l) => { if (c[l.level] !== undefined) c[l.level] += 1; });
    return c;
  }, [all]);
  const latest = all[all.length - 1];

  useEffect(() => {
    if (!open || !follow) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [shown, open, follow]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    if (atBottom !== follow) setFollow(atBottom);
  };

  // Ctrl+` toggles the dock, as in an editor.
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === '`') {
        e.preventDefault();
        toggleFx('terminalOpen');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggleFx]);

  // Drag the top edge to resize.
  const startResize = (e) => {
    e.preventDefault();
    const y0 = e.clientY;
    const h0 = height;
    const move = (ev) => setFx({ terminalHeight: Math.max(140, Math.min(window.innerHeight * 0.8, h0 + (y0 - ev.clientY))) });
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const copyAll = async () => {
    const text = shown.map((l) => `${l.t} [${l.level}] ${l.src}: ${l.msg}`).join('\n');
    try { await navigator.clipboard.writeText(text); termLog(`copied ${shown.length} lines`, 'ok', 'term'); }
    catch { termLog('clipboard unavailable in this window', 'warn', 'term'); }
  };

  const run = (raw) => {
    const cmd = raw.trim();
    if (!cmd) return;
    historyRef.current = [cmd, ...historyRef.current].slice(0, 50);
    histIdx.current = -1;
    termLog(cmd, 'cmd', 'you');
    const [name, ...args] = cmd.split(/\s+/);
    const arg = args.join(' ').toLowerCase();
    switch (name.toLowerCase()) {
      case 'help':
        termLog('commands: help · goto <tab> · tabs · palette · filter <all|engine|ui|warn|error|perf> · fx <name> [on|off] · transition <style> · clear', 'info', 'term');
        termLog('fx names: ambient parallax grain crt typewriter sunray boot', 'info', 'term');
        break;
      case 'tabs':
        termLog(tabs.map((t) => t.id).join('  '), 'info', 'term');
        break;
      case 'goto': case 'cd': case 'open': {
        const hit = tabs.find((t) => t.id === arg || t.label.toLowerCase() === arg || t.label.toLowerCase().startsWith(arg));
        if (hit) onNavigate(hit.id);   // App logs the navigation itself (one line, not two)
        else termLog(`no tab "${arg}" — try: tabs`, 'warn', 'term');
        break;
      }
      case 'palette':
        onOpenPalette?.();
        break;
      case 'filter': {
        const f = arg.toUpperCase();
        if (FILTERS.includes(f)) setFilter(f); else termLog(`filters: ${FILTERS.join(' ').toLowerCase()}`, 'warn', 'term');
        break;
      }
      case 'fx': {
        const [key, state] = args;
        const valid = ['ambient', 'parallax', 'grain', 'crt', 'typewriter', 'sunray', 'boot'];
        if (!valid.includes(key)) { termLog(`fx names: ${valid.join(' ')}`, 'warn', 'term'); break; }
        const next = state === 'on' ? true : state === 'off' ? false : !useFxStore.getState()[key];
        setFx({ [key]: next });
        termLog(`fx ${key} ${next ? 'on' : 'off'}`, 'ok', 'fx');
        break;
      }
      case 'transition': {
        const t = TRANSITIONS.find((x) => x.id === arg);
        if (t) { setFx({ transition: t.id }); termLog(`transition → ${t.label}`, 'ok', 'fx'); }
        else termLog(`styles: ${TRANSITIONS.map((x) => x.id).join(' ')}`, 'warn', 'term');
        break;
      }
      case 'clear': case 'cls':
        engineRef.current = new Map();
        setEngineVersion((v) => v + 1);
        clearTermLines();
        break;
      default:
        termLog(`${name}: command not found — type help`, 'error', 'term');
    }
  };

  const onKeyDown = (e) => {
    if (e.key === 'Enter') { run(input); setInput(''); }
    else if (e.key === 'ArrowUp') {
      const h = historyRef.current;
      if (!h.length) return;
      histIdx.current = Math.min(h.length - 1, histIdx.current + 1);
      setInput(h[histIdx.current]);
      e.preventDefault();
    } else if (e.key === 'ArrowDown') {
      histIdx.current = Math.max(-1, histIdx.current - 1);
      setInput(histIdx.current < 0 ? '' : historyRef.current[histIdx.current]);
      e.preventDefault();
    } else if (e.key === 'Escape') {
      setFx({ terminalOpen: false });
    }
  };

  return (
    <section
      className={`v2-term ${open ? 'is-open' : 'is-closed'} ${crt ? 'is-crt' : ''}`}
      style={open ? { '--term-h': `${height}px` } : undefined}
      aria-label="Terminal"
      data-testid="v2-terminal"
    >
      {open && <div className="v2-term-resize" onPointerDown={startResize} aria-hidden="true" />}
      <header className="v2-term-bar">
        <button type="button" className="v2-term-toggle" onClick={() => toggleFx('terminalOpen')}
          aria-expanded={open} aria-label={open ? 'Collapse terminal (Ctrl `)' : 'Expand terminal (Ctrl `)'}
          title="Terminal (Ctrl `)">
          <span className="v2-term-dots" aria-hidden="true"><i /><i /><i /></span>
          <span className="v2-term-title">roop<span className="text-[var(--fx-sky)]">@</span>ultimate<span className="opacity-50">:~$</span> tail -f engine.log</span>
        </button>

        {!open && latest && (
          <span key={latest.id} className={`v2-term-latest lvl-${latest.level}${typewriter ? ' is-typing' : ''}`}
            style={{ '--chars': Math.min(120, latest.msg.length + 2) }} title={latest.msg}>
            {latest.msg}
          </span>
        )}

        {open && (
          <div className="v2-term-filters" role="group" aria-label="Filter">
            {FILTERS.map((f) => (
              <button key={f} type="button" onClick={() => setFilter(f)} aria-pressed={filter === f}
                className={`v2-chip ${filter === f ? 'is-on' : ''}`}>
                {f.toLowerCase()}
                {f === 'WARN' && counts.warn > 0 && <b>{counts.warn}</b>}
                {f === 'ERROR' && counts.error > 0 && <b>{counts.error}</b>}
                {f === 'PERF' && counts.perf > 0 && <b>{counts.perf}</b>}
              </button>
            ))}
          </div>
        )}

        <span className="ml-auto flex items-center gap-1.5">
          <Ticker processing={!!progress?.processing} />
          {open && (
            <>
              <button type="button" className="v2-chip" onClick={() => toggleFx('crt')} aria-pressed={crt} title="CRT scanlines">crt</button>
              <button type="button" className="v2-chip" onClick={copyAll} title="Copy the shown lines">copy</button>
              <button type="button" className="v2-chip" onClick={() => run('clear')} title="Clear">clear</button>
            </>
          )}
        </span>
      </header>

      {open && (
        <>
          <div ref={scrollRef} onScroll={onScroll} className="v2-term-body" role="log" aria-live="off">
            {shown.length === 0 && <div className="v2-term-empty">no output yet — type <b>help</b></div>}
            {shown.map((l, i) => (
              <Line key={l.id} line={l} typewriter={typewriter}
                fresh={l.at > mountedAt.current + 300 && i >= shown.length - 6} />
            ))}
          </div>
          {!follow && (
            <button type="button" className="v2-term-follow" onClick={() => setFollow(true)}>↓ follow</button>
          )}
          <label className="v2-term-prompt">
            <span aria-hidden="true">❯</span>
            <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={onKeyDown}
              spellCheck={false} autoComplete="off" aria-label="Terminal command" placeholder="help" />
            <span className="v2-caret" aria-hidden="true" />
          </label>
        </>
      )}
    </section>
  );
}
