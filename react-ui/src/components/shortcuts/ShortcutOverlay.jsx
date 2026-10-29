import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ACTIONS, CATEGORIES } from '../../shortcuts/actions';
import { formatCombo } from '../../shortcuts/combo';
import { allConflicts, bindingsOf, isOverridden } from '../../shortcuts/keymap';
import { useShortcutStore } from '../../store/shortcutStore';
import { Icon } from '../../icons';
import KeyChips from './KeyChips';

// The workstation shortcut sheet: `?` or Ctrl+/ from anywhere.
//
// It READS the live keymap (defaults + the user's remaps), so it is the cheat-sheet that cannot go stale: the two it
// replaces - App's six-line modal and Face Swap's hand-typed "Pro Keyboard Shortcuts" HUD - were prose, and the HUD had
// already drifted from the code (it listed `H`, which the registry no longer binds, and missed `P`).
//
// Searching matches the action, its section AND the key text, so "split" finds Split view and "ctrl" finds every
// Ctrl combo. A key the user has remapped is marked, and an action sharing its key with another is flagged: a
// stored keymap can clash after an upgrade adds an action.

export default function ShortcutOverlay({ onOpenSettings }) {
  const open = useShortcutStore((s) => s.overlayOpen);
  const overrides = useShortcutStore((s) => s.overrides);
  const close = useShortcutStore((s) => s.closeOverlay);
  const [query, setQuery] = useState('');
  const searchRef = useRef(null);
  const returnTo = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    returnTo.current = document.activeElement;
    setQuery('');
    const t = requestAnimationFrame(() => searchRef.current?.focus());
    // Capture phase: Escape closes THIS dialog and nothing behind it (a roto session, the compare wipe).
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      e.preventDefault();
      e.stopPropagation();
      close();
    };
    window.addEventListener('keydown', onKey, true);
    return () => {
      cancelAnimationFrame(t);
      window.removeEventListener('keydown', onKey, true);
      try { returnTo.current?.focus?.(); } catch { /* the opener is gone */ }
    };
  }, [open, close]);

  const clashing = useMemo(() => new Set(allConflicts(overrides).flatMap((c) => c.actions)), [overrides]);

  const sections = useMemo(() => {
    const q = query.trim().toLowerCase();
    return CATEGORIES.map((cat) => {
      const rows = ACTIONS.filter((a) => a.category === cat).filter((a) => {
        if (!q) return true;
        const keys = bindingsOf(a.id, overrides).map((c) => `${c} ${formatCombo(c)}`).join(' ').toLowerCase();
        return `${a.label} ${cat}`.toLowerCase().includes(q) || keys.includes(q);
      });
      return { cat, rows };
    }).filter((s) => s.rows.length > 0);
  }, [query, overrides]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[70] flex items-center justify-center p-4"
      // Only a click on the backdrop itself closes it (not one that bubbled up from inside the dialog).
      onClick={(e) => { if (e.target === e.currentTarget) close(); }}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        data-testid="shortcut-overlay"
        className="bg-[#121216] border border-white/10 rounded-2xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl animate-scale-in text-white"
      >
        <div className="flex items-center gap-3 border-b border-white/10 px-5 py-3">
          <h3 className="text-base font-bold flex items-center gap-2 text-accent shrink-0">
            <Icon.shortcuts size={18} /> Keyboard shortcuts
          </h3>
          <input
            ref={searchRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search actions or keys…"
            aria-label="Search shortcuts"
            className="flex-1 min-w-0 px-3 py-1.5 rounded-lg glass-input text-compact text-white focus:outline-none"
          />
          <button type="button" onClick={close} aria-label="Close keyboard shortcuts" className="text-muted hover:text-white font-bold px-1">✕</button>
        </div>

        <div className="overflow-y-auto px-5 py-4">
          {sections.length === 0 ? (
            <p className="py-8 text-center text-xs text-muted">No shortcut matches “{query}”.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5">
              {sections.map(({ cat, rows }) => (
                <section key={cat} aria-label={cat} className="space-y-1.5 break-inside-avoid">
                  <h4 className="font-bold text-accent text-xs uppercase tracking-wider">{cat}</h4>
                  {rows.map((a) => (
                    <div key={a.id} className="flex items-center justify-between gap-3 text-xs" data-action={a.id}>
                      <span className="text-white/70 flex items-center gap-1.5 min-w-0">
                        <span className="truncate">{a.label}</span>
                        {isOverridden(a.id, overrides) && (
                          <span title="You changed this shortcut" className="h-1.5 w-1.5 rounded-full shrink-0 bg-[var(--mod-dot,#22d3ee)]" />
                        )}
                        {clashing.has(a.id) && (
                          <span className="text-nano font-bold uppercase text-amber-300 shrink-0" title="Shares its key with another action">clash</span>
                        )}
                      </span>
                      <KeyChips combos={bindingsOf(a.id, overrides)} />
                    </div>
                  ))}
                </section>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-white/10 px-5 py-2.5 text-xs text-muted">
          <span>Esc closes. <kbd className="font-mono bg-white/10 px-1.5 rounded">?</kbd> or <kbd className="font-mono bg-white/10 px-1.5 rounded">Ctrl + /</kbd> toggles.</span>
          {onOpenSettings && (
            <button
              type="button"
              onClick={() => { close(); onOpenSettings(); }}
              className="px-3 py-1 rounded-lg border border-white/10 text-white/80 hover:text-white hover:border-white/25 apple-transition"
            >
              Customize in Settings
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
