import React, { useState, useEffect, useRef, useMemo } from 'react';
import { rankCommands, groupBySection } from './commandPaletteModel';

export default function CommandPalette({ open, onClose, commands }) {
  const [query, setQuery] = useState('');
  const [sel, setSel] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  useEffect(() => {
    if (open) { setQuery(''); setSel(0); setTimeout(() => inputRef.current?.focus(), 20); }
  }, [open]);

  const filtered = useMemo(() => rankCommands(commands, query), [query, commands]);

  useEffect(() => { if (sel >= filtered.length) setSel(0); }, [filtered.length, sel]);

  const run = (c) => { onClose(); setTimeout(() => c.run(), 0); };

  const onKeyDown = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(filtered.length - 1, s + 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
    else if (e.key === 'Enter') { e.preventDefault(); if (filtered[sel]) run(filtered[sel]); }
    else if (e.key === 'Escape') { e.preventDefault(); onClose(); }
  };

  // Keep the highlighted row scrolled into view.
  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-idx="${sel}"]`);
    el?.scrollIntoView({ block: 'nearest' });
  }, [sel]);

  if (!open) return null;

  // Group filtered results by section, preserving score order.
  const groups = groupBySection(filtered);

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[12vh] px-4 bg-black/50 backdrop-blur-sm animate-slide-up" onMouseDown={onClose}>
      <div
        className="w-full max-w-xl rounded-2xl glass-panel border border-white/10 shadow-2xl overflow-hidden"
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="flex items-center gap-3 px-4 py-3 border-b border-white/10">
          <span className="text-muted text-lg">⌘</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => { setQuery(e.target.value); setSel(0); }}
            placeholder="Search actions, themes, settings…"
            className="flex-1 bg-transparent text-white text-sm placeholder-white/30 focus:outline-none"
          />
          <kbd className="text-nano font-mono text-muted bg-white/5 px-1.5 py-0.5 rounded border border-white/10">ESC</kbd>
        </div>
        <div ref={listRef} className="max-h-[52vh] overflow-y-auto py-2">
          {filtered.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-muted">No matching commands</div>
          ) : (
            groups.map((g) => (
              <div key={g.section} className="mb-1">
                <div className="px-4 py-1 text-nano font-semibold uppercase tracking-[0.14em] text-muted">{g.section}</div>
                {g.items.map(({ c, i }) => (
                  <button
                    key={c.id}
                    data-idx={i}
                    type="button"
                    onMouseEnter={() => setSel(i)}
                    onClick={() => run(c)}
                    className={`w-full flex items-center gap-3 px-4 py-2 text-left transition-[background-color,transform] duration-150 ease-out motion-reduce:transition-none active:scale-[0.99] ${i === sel ? 'bg-[var(--accent)]/15 translate-x-0.5' : 'hover:bg-white/5'}`}
                  >
                    {/* `icon` is an icon COMPONENT from icons.jsx (it used to
                        be an emoji string). Rendering it in a fixed-width,
                        centred box keeps every title in the list on the same
                        left edge regardless of the glyph's own width — the
                        thing emoji could never be relied on to do. */}
                    <span className="w-6 grid place-items-center shrink-0 text-muted">
                      {c.icon ? <c.icon size={16} /> : <span className="h-1 w-1 rounded-full bg-current" />}
                    </span>
                    <span className="flex-1 min-w-0">
                      <span className="block text-sm font-semibold text-white/90 truncate">{c.title}</span>
                      {c.subtitle && <span className="block text-mini text-muted truncate">{c.subtitle}</span>}
                    </span>
                    {i === sel && <kbd className="text-nano font-mono text-white/50 bg-white/5 px-1.5 py-0.5 rounded border border-white/10 shrink-0">↵</kbd>}
                  </button>
                ))}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
