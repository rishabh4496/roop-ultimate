// Stage 23: the pure half of the undo/redo history. No React, no zustand, no DOM.
//
// A history is a capped list of ENTRIES, each holding the whole tracked state at that point as a
// map { channelId: value }. Values are immutable (they are React state), so an entry costs one
// reference per channel rather than a clone: 50 entries of a 400-key settings object is 50 pointers,
// not 50 copies. That is what makes "capped at 50 to keep RAM tiny" true even for a mask-stroke
// channel whose value is an array of vector strokes.
//
//   entries[0 .. n-1]   oldest .. newest; `index` is the entry the live state currently equals
//   redo tail           entries after `index`; a commit made while index < n-1 discards it
//
// Every function takes a history and returns a NEW one (or null when there is nothing to do), so
// the store can compare references and the tests can replay sequences without any fixture.

export const HISTORY_CAP = 50;

// Two changes to the SAME thing inside this window are one entry: arrow-keying a slider, or a
// stream of wheel ticks, fires dozens of changes with no pointer transaction around them.
export const COALESCE_MS = 700;

const makeEntry = (id, label, ts, state, before, coalesceKey) => ({
  id, label, ts, state, before, coalesceKey: coalesceKey || null,
});

export function createHistory(state, { cap = HISTORY_CAP, now = 0, label = 'Initial state' } = {}) {
  return {
    cap: Math.max(2, cap | 0),
    index: 0,
    nextId: 1,
    entries: [makeEntry(0, label, now, state, null, null)],
    // Stage 12: a new edit made while the cursor is behind the tip used to DELETE the redo tail. It is now parked here
    // as a branch: { id, forkId (the entry it grew from), entries (the tail), ts, label }. switchBranch() brings one back.
    branches: [],
    nextBranchId: 1,
  };
}

// How many parked branches are kept. Entries are references to immutable state, so a branch costs a few pointers per
// entry; this bounds the count, not the memory.
export const MAX_BRANCHES = 12;

const tailBranch = (id, forkId, tail) => ({
  id, forkId, entries: tail, ts: tail[tail.length - 1].ts, label: tail[0].label,
});

/**
 * Drop branches whose fork point is gone. A fork is reachable when its entry is on the main path or inside a branch that
 * is itself reachable (a branch of a branch), decided to a fixpoint. The ring cap removing old entries is what strands one.
 */
export function pruneBranches(entries, branches) {
  if (!branches || branches.length === 0) return [];
  const known = new Set(entries.map((e) => e.id));
  const kept = [];
  let pending = branches.slice();
  for (let changed = true; changed;) {
    changed = false;
    const next = [];
    for (const b of pending) {
      if (known.has(b.forkId)) {
        kept.push(b);
        for (const e of b.entries) known.add(e.id);
        changed = true;
      } else next.push(b);
    }
    pending = next;
  }
  kept.sort((a, b) => a.id - b.id);
  return kept.length > MAX_BRANCHES ? kept.slice(kept.length - MAX_BRANCHES) : kept;
}

/** True when two state maps are equal channel by channel. `eq[channel]` overrides Object.is. */
export function sameState(a, b, eq = {}) {
  if (a === b) return true;
  const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
  for (const k of keys) {
    const same = eq[k] || Object.is;
    if (!same(a?.[k], b?.[k])) return false;
  }
  return true;
}

export const head = (h) => h.entries[h.index];
export const canUndo = (h) => h.index > 0;
export const canRedo = (h) => h.index < h.entries.length - 1;

/**
 * Record `state` as the new head.
 *
 * `describe(before, after)` is called to name the entry, and is called AGAIN when a later change is
 * merged into this entry, with `before` still the state the entry started from. That is what turns
 * a drag from 0.50 through 0.61 to 0.78 into "0.50 -> 0.78" instead of "0.61 -> 0.78".
 *
 * Returns the same history when `state` equals the head (a no-op commit never makes an entry).
 */
export function commit(h, state, {
  describe = () => 'Changed', label = null, now = 0, coalesceKey = null,
  coalesceMs = COALESCE_MS, eq = {},
} = {}) {
  const cur = head(h);
  if (sameState(cur.state, state, eq)) return h;
  const nameFor = (before) => label || describe(before, state) || 'Changed';

  // Merge into the newest entry when it is the same kind of change, recent, and we are at the tip.
  if (
    coalesceKey && h.index > 0 && h.index === h.entries.length - 1
    && cur.coalesceKey === coalesceKey && now - cur.ts <= coalesceMs
  ) {
    const before = cur.before;
    // Merged back to where the entry started: the net change is nothing, so the entry goes away.
    if (before && sameState(before, state, eq)) {
      const entries = h.entries.slice(0, -1);
      return { ...h, entries, index: entries.length - 1 };
    }
    const merged = makeEntry(cur.id, label || describe(before, state) || cur.label, now, state, before, coalesceKey);
    const entries = h.entries.slice(0, -1).concat(merged);
    return { ...h, entries };
  }

  const kept = h.entries.slice(0, h.index + 1);
  // A new action ends the REDO path, but no longer destroys it: the tail is parked as a branch off the entry we are
  // leaving, and switchBranch() can bring it back (non-destructive branching).
  const tail = h.entries.slice(h.index + 1);
  let branches = h.branches || [];
  let nextBranchId = h.nextBranchId || 1;
  if (tail.length) {
    branches = branches.concat(tailBranch(nextBranchId, cur.id, tail));
    nextBranchId += 1;
  }
  const entry = makeEntry(h.nextId, nameFor(cur.state), now, state, cur.state, coalesceKey);
  let entries = kept.concat(entry);
  if (entries.length > h.cap) entries = entries.slice(entries.length - h.cap);   // ring: oldest falls off
  return {
    ...h, entries, index: entries.length - 1, nextId: h.nextId + 1, nextBranchId, branches: pruneBranches(entries, branches),
  };
}

/**
 * The branches that can be switched to from the current path, each with where it forks: [{ id, forkId, forkIndex,
 * count, label, ts }], earliest fork first. A branch whose fork lies inside ANOTHER branch is kept but not offered until
 * that branch is the main path again.
 */
export function branchesOf(h) {
  const out = [];
  for (const b of h.branches || []) {
    const forkIndex = h.entries.findIndex((e) => e.id === b.forkId);
    if (forkIndex < 0) continue;
    out.push({ id: b.id, forkId: b.forkId, forkIndex, count: b.entries.length, label: b.label, ts: b.ts });
  }
  return out.sort((a, b) => a.forkIndex - b.forkIndex || a.id - b.id);
}

/**
 * Make branch `branchId` the main path. The entries that were the main path after the fork are parked as a new branch
 * (so switching is itself non-destructive and reversible), and the cursor lands on the switched-to branch's tip.
 * Returns null when the id is unknown or its fork is not on the current path.
 */
export function switchBranch(h, branchId) {
  const b = (h.branches || []).find((x) => x.id === branchId);
  if (!b) return null;
  const p = h.entries.findIndex((e) => e.id === b.forkId);
  if (p < 0) return null;
  const oldTail = h.entries.slice(p + 1);
  let branches = h.branches.filter((x) => x.id !== branchId);
  let nextBranchId = h.nextBranchId;
  if (oldTail.length) {
    branches = branches.concat(tailBranch(nextBranchId, b.forkId, oldTail));
    nextBranchId += 1;
  }
  let entries = h.entries.slice(0, p + 1).concat(b.entries);
  if (entries.length > h.cap) entries = entries.slice(entries.length - h.cap);
  const index = entries.length - 1;
  const next = { ...h, entries, index, nextBranchId, branches: pruneBranches(entries, branches) };
  return { h: next, state: entries[index].state, entry: entries[index] };
}

/** Patch the state of the head (and the one before it) without making an entry. For late channels. */
export function adopt(h, channel, value) {
  const patch = (e) => (e ? { ...e, state: { ...e.state, [channel]: value } } : e);
  const entries = h.entries.slice();
  entries[h.index] = patch(entries[h.index]);
  return { ...h, entries };
}

const moveTo = (h, index) => {
  if (index < 0 || index >= h.entries.length || index === h.index) return null;
  return { h: { ...h, index }, state: h.entries[index].state, entry: h.entries[index] };
};

export const undo = (h) => moveTo(h, h.index - 1);
export const redo = (h) => moveTo(h, h.index + 1);
export const jump = (h, index) => moveTo(h, index);

// ── Naming a change ────────────────────────────────────────────────────────

export function formatValue(v) {
  if (v === null || v === undefined || v === '') return 'none';
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return String(v);
    return Number.isInteger(v) ? String(v) : v.toFixed(2);
  }
  if (typeof v === 'string') return v.length > 24 ? `${v.slice(0, 21)}…` : v;
  if (Array.isArray(v)) return `${v.length} item${v.length === 1 ? '' : 's'}`;
  return '…';
}

const sameValue = (a, b) => {
  if (Object.is(a, b)) return true;
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    try { return JSON.stringify(a) === JSON.stringify(b); } catch { return false; }
  }
  return false;
};

/** Keys whose values differ between two flat settings objects. */
export function changedKeys(before, after) {
  const keys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
  const out = [];
  for (const k of keys) if (!sameValue(before?.[k], after?.[k])) out.push(k);
  return out;
}

export function settingsEqual(a, b) {
  if (a === b) return true;
  return changedKeys(a, b).length === 0;
}

/** "Changed Restore Ultra Blend (0.50 -> 0.78)" and its multi-key forms. */
export function describeSettingsChange(before, after, labelOf = (k) => k) {
  const keys = changedKeys(before, after);
  if (keys.length === 0) return null;
  if (keys.length === 1) {
    const k = keys[0];
    const a = before?.[k];
    const b = after?.[k];
    if (typeof b === 'boolean' && typeof a !== 'object') return `${b ? 'Turned on' : 'Turned off'} ${labelOf(k)}`;
    return `Changed ${labelOf(k)} (${formatValue(a)} -> ${formatValue(b)})`;
  }
  if (keys.length <= 3) return `Changed ${keys.map(labelOf).join(', ')}`;
  return `Changed ${keys.length} settings`;
}

const zoneCount = (byMedia) => Object.values(byMedia || {}).reduce((n, z) => n + (Array.isArray(z) ? z.length : 0), 0);

export function describeRotoChange(before, after) {
  const a = zoneCount(before);
  const b = zoneCount(after);
  if (b > a) return 'Added Roto Polygon';
  if (b < a) return 'Deleted Roto Polygon';
  return 'Edited Roto Polygon';
}

export function describePinsChange(before, after) {
  const a = zoneCount(before);
  const b = zoneCount(after);
  if (b > a) return 'Pinned Identity';
  if (b < a) return 'Unpinned Identity';
  return 'Edited Identity Pin';
}

export function describeStrokesChange(before, after) {
  const a = before?.length || 0;
  const b = after?.length || 0;
  if (b === 0 && a > 0) return 'Cleared Occlusion Mask';
  if (b > a) return after[b - 1]?.erase ? 'Erased Occlusion Mask Stroke' : 'Painted Occlusion Mask Stroke';
  if (b < a) return 'Removed Occlusion Mask Stroke';
  return 'Edited Occlusion Mask';
}

export function describeAppliedMaskChange(before, after) {
  if (after?.dataUrl && !before?.dataUrl) return 'Applied Occlusion Mask';
  if (!after?.dataUrl && before?.dataUrl) return 'Removed Applied Mask';
  return 'Re-applied Occlusion Mask';
}

// ── What an entry changed (Stage 12) ────────────────────────────────────────

// Settings that shape the performance rather than the picture: a change to ONLY these reads as an expression edit.
const EXPRESSION_KEY = /expression|gaze|viseme|blink|brow|pupil|mouth|lip/i;

/** The channels whose value differs between an entry and the state it started from. */
export function changedChannels(entry) {
  const before = entry?.before;
  if (!before) return [];
  const out = [];
  for (const id of new Set([...Object.keys(before), ...Object.keys(entry.state || {})])) {
    if (!sameValue(before[id], entry.state?.[id])) out.push(id);
  }
  return out;
}

/**
 * A coarse kind for a history row, so the stack can tell a slider tweak from a brush stroke at a glance:
 * 'parameter' | 'expression' | 'brush' | 'mask' | 'curve' | 'roto' | 'pin' | 'slot' | 'other'.
 * An entry that moved several channels takes the first in that priority; the recall of a snapshot slot is its own kind.
 */
export function entryKind(entry) {
  if (!entry) return 'other';
  if (/^Recalled Slot/.test(entry.label || '')) return 'slot';
  const ch = changedChannels(entry);
  if (ch.length === 0) return 'other';
  if (ch.includes('maskStrokes')) return 'brush';
  if (ch.includes('mask')) return 'mask';
  if (ch.includes('curves')) return 'curve';
  if (ch.includes('roto')) return 'roto';
  if (ch.includes('pins')) return 'pin';
  if (ch.includes('settings')) {
    const keys = changedKeys(entry.before.settings, entry.state.settings);
    return keys.length > 0 && keys.every((k) => EXPRESSION_KEY.test(k)) ? 'expression' : 'parameter';
  }
  return 'other';
}

/** "9s ago" / "4m ago" for the popover; callers pass `now` so it is testable. */
export function ago(ts, now) {
  const s = Math.max(0, Math.round((now - ts) / 1000));
  if (s < 5) return 'just now';
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.round(m / 60)}h ago`;
}
