import React, { useMemo } from 'react';
import { Icon } from '../../icons';
import InspectorGroup from './InspectorGroup';
import { useHistoryStore } from '../../store/historyStore';
import { ago } from '../../history/historyEngine';
import { KIND_META, buildRows, positionText, stepsText } from '../../history/historyStackModel';

// ── The History Timeline Stack (Stage 12), in the inspector ─────────────────
//
// Every recorded change as a clickable state: a slider tweak, a brush stroke, an expression edit, a curve keyframe, a
// roto shape, a slot recall. Click a row to time-travel to it (the store applies it to every channel in one step).
//
// NON-DESTRUCTIVE BRANCHING. Undo three steps and make a new edit and the three steps you left are not deleted: they stay
// as a branch off the step you forked from, listed right under it. "Switch" makes that branch the current path, and the
// path you were on is parked in turn, so a switch can always be undone by switching back. Rows ahead of the cursor (what
// Redo would bring back) are dimmed, not hidden.
//
// It reads the same store as the header popover, so the two never disagree.

const clock = (ts) => {
  try { return new Date(ts).toLocaleTimeString([], { hour12: false }); } catch { return ''; }
};

const icoBtn = 'inline-flex h-7 items-center gap-1 rounded-lg border border-white/10 bg-white/[0.04] px-2 text-mini font-semibold text-white/80 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-35';

export default function HistoryStackPanel() {
  const entries = useHistoryStore((s) => s.entries);
  const index = useHistoryStore((s) => s.index);
  const branches = useHistoryStore((s) => s.branches);
  const canUndo = useHistoryStore((s) => s.canUndo);
  const canRedo = useHistoryStore((s) => s.canRedo);
  const undo = useHistoryStore((s) => s.undo);
  const redo = useHistoryStore((s) => s.redo);
  const jumpTo = useHistoryStore((s) => s.jumpTo);
  const switchBranch = useHistoryStore((s) => s.switchBranch);

  const rows = useMemo(() => buildRows(entries, index, branches), [entries, index, branches]);
  const now = Date.now();

  return (
    <div className="space-y-2" data-testid="history-stack">
      <div className="flex flex-wrap items-center gap-1.5">
        <button type="button" className={icoBtn} onClick={() => undo()} disabled={!canUndo} data-testid="history-stack-undo" title="Undo (Ctrl/⌘ Z)">
          <Icon.undo size={13} /> Undo
        </button>
        <button type="button" className={icoBtn} onClick={() => redo()} disabled={!canRedo} data-testid="history-stack-redo" title="Redo (Ctrl/⌘ Y or Ctrl/⌘ Shift Z)">
          <Icon.redo size={13} /> Redo
        </button>
        <span className="ml-auto text-micro tabular-nums text-white/55" data-testid="history-stack-position">{positionText(entries, index)}</span>
      </div>

      {entries.length <= 1 ? (
        <p className="text-mini text-white/55">Nothing recorded yet. Move a slider, paint a mask stroke, key a curve or draw a roto shape and it is listed here.</p>
      ) : (
        <ol className="max-h-72 space-y-0.5 overflow-y-auto pr-0.5" aria-label="History states, newest first">
          {rows.map((r) => {
            const meta = KIND_META[r.kind] || KIND_META.other;
            return (
              <li key={r.id}>
                <button
                  type="button"
                  onClick={() => jumpTo(r.i)}
                  aria-current={r.current ? 'step' : undefined}
                  data-history-stack-row={r.i}
                  data-kind={r.kind}
                  title={r.base ? 'Back to where this session started' : `Go to: ${r.label}`}
                  className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-mini transition-colors hover:bg-white/10 ${
                    r.current ? 'bg-[var(--accent)]/15 text-white ring-1 ring-[var(--accent)]/40' : r.future ? 'italic text-white/40' : 'text-white/75'}`}
                >
                  <span
                    className="inline-flex w-11 shrink-0 justify-center rounded px-1 py-0.5 text-nano font-bold uppercase tracking-wide"
                    style={{ color: meta.color, background: `${meta.color}22` }}
                    title={meta.title}
                  >
                    {r.base ? 'Start' : meta.chip}
                  </span>
                  <span className="min-w-0 flex-1 truncate">{r.label}</span>
                  <span className="shrink-0 font-mono text-nano tabular-nums text-white/45" title={ago(r.ts, now)}>{clock(r.ts)}</span>
                </button>
                {r.branches.length > 0 && (
                  <ul className="ml-5 mt-0.5 space-y-0.5 border-l border-dashed border-white/15 pl-2" aria-label={`Branches from: ${r.label}`}>
                    {r.branches.map((b) => (
                      <li key={b.id}>
                        <button
                          type="button"
                          onClick={() => switchBranch(b.id)}
                          data-history-branch={b.id}
                          title={`Switch to this branch. The path you are on is kept as a branch, so you can switch back.`}
                          className="flex w-full items-center gap-2 rounded-lg px-2 py-1 text-left text-micro text-white/60 transition-colors hover:bg-white/10 hover:text-white"
                        >
                          <span className="rounded bg-white/10 px-1 py-0.5 text-nano font-bold uppercase tracking-wide text-white/70">Branch</span>
                          <span className="min-w-0 flex-1 truncate">{b.label}</span>
                          <span className="shrink-0 tabular-nums text-white/45">{stepsText(b.count)}</span>
                          <span className="shrink-0 font-semibold text-[var(--accent)]">Switch</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

/** The collapsible inspector card around the stack. Closed by default; its badge counts the steps on the current path. */
export function HistoryInspectorGroup() {
  const steps = useHistoryStore((s) => Math.max(0, s.entries.length - 1));
  const branchCount = useHistoryStore((s) => s.branches.length);
  const badge = steps === 0 ? undefined : `${steps}${branchCount ? ` · ${branchCount}⑂` : ''}`;
  return (
    <InspectorGroup id="history-stack" title="History" icon={Icon.history} badge={badge}>
      <HistoryStackPanel />
    </InspectorGroup>
  );
}
