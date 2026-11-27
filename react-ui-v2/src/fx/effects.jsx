// React UI 2.0's effect components, written in-house in the style of react-bits / Magic UI
// (spotlight card, number ticker, split text, meteors, marquee, magnifying dock, animated beam).
// No code from either library is used. Each one honours prefers-reduced-motion (CSS in app.css,
// or the `reduced` checks below).
import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { prefersReducedMotion } from './fxStore';

// ── Panel: a hairline card whose light follows the pointer ────────────────
export function Panel({ title, accent, actions, className = '', bodyClass = 'p-3', beam = false, children, ...rest }) {
  const ref = useRef(null);
  const onMove = (e) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty('--mx', `${e.clientX - r.left}px`);
    el.style.setProperty('--my', `${e.clientY - r.top}px`);
  };
  const onLeave = () => { ref.current?.style.setProperty('--mx', '-999px'); };
  return (
    <section ref={ref} onPointerMove={onMove} onPointerLeave={onLeave}
      className={`panel ${beam ? 'beam' : ''} ${className}`} {...rest}>
      {(title || actions) && (
        <header className="panel-head">
          {title && <h2 className="panel-title">{title}{accent && <> · <b>{accent}</b></>}</h2>}
          {actions && <div className="ml-auto flex items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

// ── NumberTicker: counts to its value with a spring-ish ease ─────────────
export function NumberTicker({ value, decimals = 0, className = '', suffix = '' }) {
  const [shown, setShown] = useState(Number(value) || 0);
  const from = useRef(Number(value) || 0);
  useEffect(() => {
    const target = Number(value) || 0;
    if (prefersReducedMotion()) { setShown(target); from.current = target; return undefined; }
    const start = performance.now();
    const a = from.current;
    let raf = 0;
    const step = (now) => {
      const t = Math.min(1, (now - start) / 700);
      const e = 1 - Math.pow(1 - t, 4);
      const v = a + (target - a) * e;
      setShown(v);
      if (t < 1) raf = requestAnimationFrame(step); else from.current = target;
    };
    raf = requestAnimationFrame(step);
    return () => { cancelAnimationFrame(raf); from.current = target; };
  }, [value]);
  return <span className={`tnum ${className}`}>{shown.toFixed(decimals)}{suffix}</span>;
}

// ── SplitText: letters rise in, staggered (re-runs when the text changes) ─
export function SplitText({ text, className = '', stagger = 22 }) {
  const chars = useMemo(() => Array.from(String(text ?? '')), [text]);
  return (
    <span className={className} aria-label={text} key={text}>
      {chars.map((c, i) => (
        <span key={i} aria-hidden="true" className="split-char"
          style={{ animationDelay: `${i * stagger}ms`, '--p': `${chars.length > 1 ? (i / (chars.length - 1)) * 100 : 0}%` }}>
          {c === ' ' ? ' ' : c}
        </span>
      ))}
    </span>
  );
}

// ── Meteors: a few streaks across an empty state ─────────────────────────
export function Meteors({ count = 10 }) {
  const items = useMemo(() => Array.from({ length: count }, (_, i) => ({
    left: `${(i * 97) % 100}%`, delay: `${(i * 0.73) % 5}s`, dur: `${3 + ((i * 1.31) % 4)}s`,
  })), [count]);
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      {items.map((m, i) => (
        <span key={i} className="meteor" style={{ left: m.left, animationDelay: m.delay, animationDuration: m.dur }} />
      ))}
    </div>
  );
}

// ── Marquee: an endless strip that pauses on hover ───────────────────────
export function Marquee({ children, seconds = 40, className = '' }) {
  return (
    <div className={`marquee ${className}`} style={{ '--marquee-s': `${seconds}s` }}>
      <div className="marquee-track">
        {children}
        <span aria-hidden="true" className="contents">{children}</span>
      </div>
    </div>
  );
}

// ── Dock: icons magnify toward the pointer (macOS-style) ─────────────────
const BASE = 42;
const MAX = 64;
const RANGE = 130;

export function Dock({ items, active, onSelect }) {
  const [mx, setMx] = useState(null);
  const [tip, setTip] = useState(null);
  const reduced = prefersReducedMotion();
  const nodes = useRef({});
  // Centres measured ONCE, at rest size, when the pointer arrives: measuring the grown icons
  // would feed each frame's magnification back into the next (a visible jitter).
  const centres = useRef({});
  const measure = () => {
    const out = {};
    Object.entries(nodes.current).forEach(([id, el]) => {
      if (!el) return;
      const r = el.getBoundingClientRect();
      out[id] = r.left + r.width / 2;
    });
    centres.current = out;
  };
  const sizeFor = (id) => {
    const c = centres.current[id];
    if (reduced || mx == null || c == null) return BASE;
    const d = Math.abs(mx - c);
    if (d > RANGE) return BASE;
    return BASE + (MAX - BASE) * Math.cos((d / RANGE) * (Math.PI / 2));
  };
  return (
    <nav aria-label="Main" className="dock-wrap">
      <div className="dock" onPointerEnter={measure} onPointerMove={(e) => setMx(e.clientX)}
        onPointerLeave={() => { setMx(null); setTip(null); }}>
        {items.map((it) => {
          if (it.sep) return <span key={it.key} className="dock-sep" aria-hidden="true" />;
          const size = sizeFor(it.id);
          const Ico = it.icon;
          return (
            <button key={it.id} type="button" ref={(el) => { nodes.current[it.id] = el; }}
              className={`dock-item ${active === it.id ? 'active' : ''}`}
              style={{ width: size, height: size, transition: 'width .12s var(--ease), height .12s var(--ease)' }}
              aria-label={it.label} aria-current={active === it.id ? 'page' : undefined}
              onPointerEnter={() => setTip(it.id)} onFocus={() => setTip(it.id)} onBlur={() => setTip(null)}
              onClick={() => onSelect(it.id, it)} data-testid={`dock-${it.id}`}>
              <Ico size={Math.round(size * 0.42)} strokeWidth={1.8} />
              {it.badge > 0 && <span className="dock-badge">{it.badge > 9 ? '9+' : it.badge}</span>}
              {active === it.id && <span className="dock-dot" aria-hidden="true" />}
              {tip === it.id && <span className="dock-tip" role="tooltip">{it.label}{it.kbd && <span className="ml-1.5 text-[var(--fg-faint)]">{it.kbd}</span>}</span>}
            </button>
          );
        })}
      </div>
    </nav>
  );
}

// ── PipelineBeams: nodes joined by beams of light that flow while running ─
// Positions are measured from the DOM, so the beams follow any layout. `live` speeds the flow and
// lights the nodes; idle, a slow faint pulse shows the path.
export function PipelineBeams({ nodes, live = false, className = '' }) {
  const wrap = useRef(null);
  const refs = useRef([]);
  const [paths, setPaths] = useState([]);
  const [box, setBox] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const measure = () => {
      const w = wrap.current;
      if (!w) return;
      const r = w.getBoundingClientRect();
      setBox({ w: r.width, h: r.height });
      const pts = refs.current.map((el) => {
        if (!el) return null;
        const b = el.getBoundingClientRect();
        return { x: b.left - r.left + b.width / 2, y: b.top - r.top + b.height / 2, half: b.width / 2 };
      });
      const out = [];
      for (let i = 0; i < pts.length - 1; i += 1) {
        const a = pts[i]; const b = pts[i + 1];
        if (!a || !b) continue;
        const x1 = a.x + a.half; const x2 = b.x - b.half;
        const mid = (x1 + x2) / 2;
        const bend = (i % 2 ? 1 : -1) * 10;
        out.push(`M ${x1} ${a.y} C ${mid} ${a.y + bend}, ${mid} ${b.y - bend}, ${x2} ${b.y}`);
      }
      setPaths(out);
    };
    measure();
    const ro = new ResizeObserver(measure);
    if (wrap.current) ro.observe(wrap.current);
    return () => ro.disconnect();
  }, [nodes.length]);
  return (
    <div ref={wrap} className={`relative ${className}`}>
      <svg aria-hidden="true" className="pointer-events-none absolute inset-0" width={box.w} height={box.h}>
        <defs>
          <linearGradient id="beam-grad" x1="0" x2="1">
            <stop offset="0" stopColor="var(--accent-2)" stopOpacity="0" />
            <stop offset="0.5" stopColor="var(--accent-2)" />
            <stop offset="1" stopColor="var(--accent)" />
          </linearGradient>
        </defs>
        {paths.map((d, i) => (
          <g key={i}>
            <path d={d} className="beam-path" />
            <path d={d} className={`beam-flow ${live ? '' : 'idle'}`} stroke="url(#beam-grad)" style={{ animationDelay: `${i * 0.25}s` }} />
          </g>
        ))}
      </svg>
      <div className="relative flex items-center justify-between gap-2">
        {nodes.map((n, i) => {
          const Ico = n.icon;
          return (
            <div key={n.id} className="flex flex-col items-center gap-1.5 min-w-0">
              <div ref={(el) => { refs.current[i] = el; }} className={`pipe-node ${live ? 'live' : ''}`} title={n.title}>
                <Ico size={17} strokeWidth={1.8} />
              </div>
              <div className="max-w-[86px] truncate text-center text-[10.5px] font-medium text-[var(--fg-dim)]" title={n.value}>{n.label}</div>
              {n.value && <div className="max-w-[86px] truncate text-center font-mono text-[10px] text-[var(--accent-2)]" title={n.value}>{n.value}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
