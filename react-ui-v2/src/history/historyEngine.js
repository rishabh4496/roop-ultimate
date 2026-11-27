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
  };
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

  const kept = h.entries.slice(0, h.index + 1);          // a new action ends the redo branch
  const entry = makeEntry(h.nextId, nameFor(cur.state), now, state, cur.state, coalesceKey);
  let entries = kept.concat(entry);
  if (entries.length > h.cap) entries = entries.slice(entries.length - h.cap);   // ring: oldest falls off
  return { ...h, entries, index: entries.length - 1, nextId: h.nextId + 1 };
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

/** "9s ago" / "4m ago" for the popover; callers pass `now` so it is testable. */
export function ago(ts, now) {
  const s = Math.max(0, Math.round((now - ts) / 1000));
  if (s < 5) return 'just now';
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.round(m / 60)}h ago`;
}
