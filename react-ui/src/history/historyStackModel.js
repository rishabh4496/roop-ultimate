// Stage 12: the pure half of the History Timeline Stack in the inspector. No React.
//
// The store keeps the current path as a list of entries plus the parked branches (historyEngine). A row is one entry as
// the panel shows it: what changed (a kind, from the channels the entry moved), where it sits relative to the cursor, and
// the branches that grew from it. Rows come newest first, like the popover, so the top of the list is "now".

import { entryKind } from './historyEngine';

// kind -> short chip text and a colour token. The colours are CSS colours, not Tailwind classes, because the panel
// sets them inline (a Tailwind class built from a variable would be purged).
export const KIND_META = Object.freeze({
  parameter: { chip: 'Param', color: '#38bdf8', title: 'A parameter changed' },
  expression: { chip: 'Expr', color: '#f472b6', title: 'An expression / gaze / mouth setting changed' },
  brush: { chip: 'Brush', color: '#fbbf24', title: 'An occlusion-mask brush stroke' },
  mask: { chip: 'Mask', color: '#fb923c', title: 'An occlusion mask was applied or removed' },
  curve: { chip: 'Curve', color: '#34d399', title: 'A parameter curve keyframe' },
  roto: { chip: 'Roto', color: '#a78bfa', title: 'A roto polygon' },
  pin: { chip: 'Pin', color: '#60a5fa', title: 'An identity pin' },
  slot: { chip: 'Slot', color: '#22d3ee', title: 'A snapshot slot was recalled' },
  other: { chip: 'Edit', color: '#94a3b8', title: 'A change' },
});

/**
 * @param entries   the current path, oldest first (historyStore.entries)
 * @param index     the cursor: the entry the live state equals
 * @param branches  historyEngine.branchesOf(...)
 * @returns rows, newest first: { i, id, label, ts, kind, current, future, base, branches }
 *   `base` is the first entry ("Initial state"): shown, never the target of anything but a jump.
 */
export function buildRows(entries, index, branches = []) {
  const byFork = new Map();
  for (const b of branches) {
    if (!byFork.has(b.forkIndex)) byFork.set(b.forkIndex, []);
    byFork.get(b.forkIndex).push(b);
  }
  return (entries || []).map((e, i) => ({
    i,
    id: e.id,
    label: e.label,
    ts: e.ts,
    kind: i === 0 ? 'other' : entryKind(e),
    current: i === index,
    future: i > index,
    base: i === 0,
    branches: byFork.get(i) || [],
  })).reverse();
}

/** "3 steps" / "1 step". */
export const stepsText = (n) => `${n} step${n === 1 ? '' : 's'}`;

/** What the panel's header says about where the cursor is. */
export function positionText(entries, index) {
  const n = Math.max(0, (entries?.length || 0) - 1);
  if (n === 0) return 'No edits yet';
  return `Step ${index} of ${n}`;
}
