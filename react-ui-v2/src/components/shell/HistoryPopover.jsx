import React, { useEffect, useId, useRef, useState } from 'react';
import { Icon } from '../../icons';
import { useHistoryStore } from '../../store/historyStore';
import { ago } from '../../history/historyEngine';

// ── The history stack, from the header ─────────────────────────────────────
//
// Undo / redo buttons and a popover listing the last 50 actions. Clicking a past row time-travels to
// that state in one step: the store applies it to every channel, and the existing preview effects
// (which watch the same settings the rows describe) coalesce the follow-up frame request, so no
// refresh is triggered from here.
//
// Rows are newest first. Rows ahead of the current state (what Redo would bring back) are shown
// dimmed rather than hidden, so a time-travel to the past is visibly undoable.
const BTN = 'flex h-8 items-center gap-1.5 rounded-lg px-2 text-mini font-semibold transition-colors';
const IDLE = 'text-white/65 hover:bg-white/10 hover:text-white';
const OFF = 'text-white/25 cursor-not-allowed';

const clock = (ts) => {
  try { return new Date(ts).toLocaleTimeString([], { hour12: false }); } catch { return ''; }
};

export default function HistoryPopover() {
  const entries = useHistoryStore((s) => s.entries);
  const index = useHistoryStore((s) => s.index);
  const canUndo = useHistoryStore((s) => s.canUndo);
  const canRedo = useHistoryStore((s) => s.canRedo);
  const undo = useHistoryStore((s) => s.undo);
  const redo = useHistoryStore((s) => s.redo);
  const jumpTo = useHistoryStore((s) => s.jumpTo);
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false); };
    window.addEventListener('pointerdown', onDown, true);
    return () => {
      window.removeEventListener('pointerdown', onDown, true);
    };
  }, [open]);

  const now = Date.now();
  const rows = entries.map((e, i) => ({ e, i })).reverse();
  const current = entries[index];

  return (
    <div ref={rootRef} className="relative" role="presentation" onKeyDown={(e) => { if (e.key === 'Escape' && open) { e.stopPropagation(); setOpen(false); } }}>
      <div role="group" aria-label="Edit history" className="flex items-center gap-0.5 rounded-xl border border-white/10 bg-white/[0.03] p-0.5">
        <button
          type="button"
          onClick={() => undo()}
          disabled={!canUndo}
          title="Undo (Ctrl/⌘ Z)"
          aria-label="Undo"
          className={`${BTN} ${canUndo ? IDLE : OFF}`}
          data-testid="history-undo"
        >
          <Icon.undo size={14} />
        </button>
        <button
          type="button"
          onClick={() => redo()}
          disabled={!canRedo}
          title="Redo (Ctrl/⌘ Y or Ctrl/⌘ Shift Z)"
          aria-label="Redo"
          className={`${BTN} ${canRedo ? IDLE : OFF}`}
          data-testid="history-redo"
        >
          <Icon.redo size={14} />
        </button>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls={open ? panelId : undefined}
          title="History stack: the last 50 actions"
          aria-label={`History, ${entries.length > 0 ? entries.length - 1 : 0} actions`}
          className={`${BTN} ${IDLE} ${open ? 'bg-white/10 text-white' : ''}`}
          data-testid="history-toggle"
        >
          <Icon.history size={14} />
          <span className="tabular-nums max-2xl:hidden">{Math.max(0, entries.length - 1)}</span>
        </button>
      </div>

      {open && (
        <div
          id={panelId}
          role="region"
          aria-label="History stack"
          className="absolute right-0 top-full z-50 mt-2 w-80 overflow-hidden rounded-xl border border-white/10 bg-[#10131c]/95 shadow-2xl backdrop-blur-xl"
          data-testid="history-panel"
        >
          <div className="flex items-center justify-between border-b border-white/[0.07] px-3 py-2">
            <span className="text-mini font-bold text-white/85">History</span>
            <span className="text-micro text-white/55">{current ? `at: ${current.label}` : ''}</span>
          </div>
          {entries.length <= 1 ? (
            <p className="px-3 py-4 text-mini text-white/55">Nothing to undo yet. Change a setting, paint a mask or draw a roto shape and it appears here.</p>
          ) : (
            <ul className="max-h-96 overflow-y-auto py-1" aria-label="Actions, newest first">
              {rows.map(({ e, i }) => {
                const isCurrent = i === index;
                const future = i > index;
                return (
                  <li key={e.id}>
                    <button
                      type="button"
                      onClick={() => { jumpTo(i); }}
                      aria-current={isCurrent ? 'step' : undefined}
                      title={`Go to: ${e.label}`}
                      className={`flex w-full items-baseline gap-2 px-3 py-1.5 text-left text-mini transition-colors hover:bg-white/10 ${
                        isCurrent ? 'bg-[var(--accent)]/15 text-white' : future ? 'text-white/40 italic' : 'text-white/75'
                      }`}
                      data-history-row={i}
                    >
                      <span className="min-w-0 flex-1 truncate">{e.label}</span>
                      <span className="shrink-0 font-mono text-nano tabular-nums text-white/45" title={ago(e.ts, now)}>{clock(e.ts)}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
