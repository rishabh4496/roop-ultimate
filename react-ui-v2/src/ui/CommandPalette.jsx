import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Search, CornerDownLeft } from 'lucide-react';
import { postJSON } from '@engine/api';
import { useStudio } from '../engine/studio';
import { useRun } from '../engine/run';
import { readiness, buildQueueJob } from '../engine/payload';
import { notify } from './controls';

// ── Command palette: Ctrl+K (Cmd+K) ──
// Every screen, every Settings page, and the actions people reach for, by name. Typing filters by
// subsequence ("strt rnd" finds "Start render"); arrows move, Enter runs, Escape closes. An action
// that cannot run right now is listed with the reason instead of being hidden.

const score = (q, text) => {
  if (!q) return 1;
  const t = text.toLowerCase();
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  let total = 0;
  for (const w of words) {
    const at = t.indexOf(w);
    if (at >= 0) { total += 10 - Math.min(9, at); continue; }
    let i = 0;
    for (const ch of w) { i = t.indexOf(ch, i); if (i < 0) return 0; i += 1; }
    total += 1;
  }
  return total;
};

export function useCommands({ screens, go, toggleTerminal }) {
  const studio = useStudio();
  const p = useRun((r) => r.progress);
  const control = useRun((r) => r.control);
  const start = useRun((r) => r.start);
  const why = readiness(studio);
  return useMemo(() => {
    const list = screens.map((s) => ({ id: `go-${s.id}`, group: 'Go to', label: s.label, hint: s.kbd, run: () => go(s.id) }));
    [['storage', 'Storage'], ['tuning', 'Tuning (benchmark, auto-tune, TensorRT, recognition)'], ['system', 'System and updates']].forEach(([tab, label]) => {
      list.push({ id: `settings-${tab}`, group: 'Settings', label, run: () => { window.location.hash = `#/settings?tab=${tab}`; } });
    });
    list.push(
      { id: 'run-start', group: 'Render', label: 'Start render', disabled: p.processing ? 'a render is running' : why || '', run: () => { go('studio'); start(); } },
      { id: 'run-queue', group: 'Render', label: 'Add the current setup to the batch', disabled: why || '', run: async () => { const job = buildQueueJob(studio); try { await postJSON('/api/queue/add', job); notify(`Queued “${job.target_name}”`); } catch (e) { notify(e.message, 'error'); } } },
      { id: 'run-pause', group: 'Render', label: p.paused ? 'Resume render' : 'Pause render', disabled: p.processing ? '' : 'no render is running', run: () => control(p.paused ? 'resume' : 'pause') },
      { id: 'run-stop', group: 'Render', label: 'Stop render', disabled: p.processing ? '' : 'no render is running', run: () => control('stop') },
      { id: 'view-swap', group: 'Viewport', label: 'Show the swap', run: () => { go('studio'); studio.setPreviewMode?.('swap'); } },
      { id: 'view-original', group: 'Viewport', label: 'Show the original', run: () => { go('studio'); studio.setPreviewMode?.('original'); } },
      { id: 'view-compare', group: 'Viewport', label: 'Compare (wipe)', run: () => { go('studio'); studio.setPreviewMode?.('compare'); } },
      { id: 'people-scan', group: 'People', label: 'Scan the clip for people (auto-capture)', disabled: studio.targets.length ? '' : 'no target loaded', run: () => { go('studio'); if (!studio.targetFaces.length || window.confirm('Scan the clip and capture everyone from scratch?\n\nThis replaces the people you already have.')) studio.autoCapture(); } },
      { id: 'terminal', group: 'Window', label: 'Toggle the terminal', hint: 'Ctrl `', run: toggleTerminal },
    );
    return list;
  }, [screens, go, toggleTerminal, studio, p.processing, p.paused, why, start, control]);
}

export default function CommandPalette({ commands }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState('');
  const [at, setAt] = useState(0);
  const input = useRef(null);
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'k') { e.preventDefault(); setOpen((o) => !o); setQ(''); setAt(0); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  useEffect(() => { if (open) setTimeout(() => input.current?.focus(), 0); }, [open]);
  const shown = useMemo(() => commands
    .map((c) => ({ c, s: score(q, `${c.group} ${c.label}`) }))
    .filter((x) => x.s > 0)
    .sort((x, y) => y.s - x.s)
    .map((x) => x.c), [commands, q]);
  if (!open) return null;
  const run = (c) => {
    if (!c) return;
    if (c.disabled) { notify(`${c.label}: ${c.disabled}`, 'warn'); return; }
    setOpen(false);
    c.run();
  };
  return (
    <div className="fixed inset-0 z-[150] grid place-items-start justify-center bg-black/60 pt-[12vh] backdrop-blur-[2px]" onMouseDown={(e) => { if (e.target === e.currentTarget) setOpen(false); }}>
      <div className="panel w-[min(620px,92vw)] overflow-hidden" role="dialog" aria-modal="true" aria-label="Command palette" data-testid="palette">
        <div className="flex items-center gap-2 border-b border-[var(--rule)] px-3">
          <Search size={15} className="text-[var(--fg-faint)]" />
          <input ref={input} className="h-12 flex-1 bg-transparent text-[14px] text-white outline-none" placeholder="Type a screen or an action…" value={q}
            role="combobox" aria-expanded="true" aria-controls="palette-list" aria-activedescendant={shown[at] ? `cmd-${shown[at].id}` : undefined}
            onChange={(e) => { setQ(e.target.value); setAt(0); }} data-testid="palette-input"
            onKeyDown={(e) => {
              if (e.key === 'Escape') { e.preventDefault(); setOpen(false); }
              else if (e.key === 'ArrowDown') { e.preventDefault(); setAt((i) => Math.min(shown.length - 1, i + 1)); }
              else if (e.key === 'ArrowUp') { e.preventDefault(); setAt((i) => Math.max(0, i - 1)); }
              else if (e.key === 'Enter') { e.preventDefault(); run(shown[at]); }
            }} />
          <kbd className="rounded border border-[var(--rule-strong)] px-1.5 font-mono text-[10px] text-[var(--fg-faint)]">Esc</kbd>
        </div>
        <ul id="palette-list" role="listbox" className="max-h-[52vh] overflow-y-auto p-1.5">
          {shown.map((c, i) => (
            <li key={c.id} id={`cmd-${c.id}`} role="option" aria-selected={i === at} aria-disabled={!!c.disabled || undefined}
              onMouseEnter={() => setAt(i)} onMouseDown={(e) => { e.preventDefault(); run(c); }} data-testid="palette-item"
              className={`flex cursor-pointer items-center gap-3 rounded-lg px-2.5 py-2 text-[13px] ${i === at ? 'bg-white/[0.07]' : ''} ${c.disabled ? 'opacity-50' : ''}`}>
              <span className="w-20 shrink-0 font-mono text-[10.5px] uppercase tracking-wide text-[var(--fg-faint)]">{c.group}</span>
              <span className="min-w-0 flex-1 truncate text-white">{c.label}</span>
              {c.disabled && <span className="truncate text-[11px] text-[var(--fg-faint)]">{c.disabled}</span>}
              {c.hint && !c.disabled && <span className="font-mono text-[10.5px] text-[var(--fg-faint)]">{c.hint}</span>}
              {i === at && !c.disabled && <CornerDownLeft size={12} className="text-[var(--fg-faint)]" />}
            </li>
          ))}
          {!shown.length && <li className="px-3 py-6 text-center text-[12.5px] text-[var(--fg-faint)]">Nothing matches “{q}”.</li>}
        </ul>
      </div>
    </div>
  );
}
