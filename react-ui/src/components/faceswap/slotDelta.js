// Stage 12: the pure half of the A | B | C | D snapshot bank's Quad compare. No React, no DOM.
//
// A slot (store/slotStore.js) is { id, state, bitmap, frameKey, label, ts, width, height }: `state` is the tracked recipe
// at the moment of capture ({ channelId: value }, the same map a history entry holds). Two slots differ where their
// `settings` differ (the sliders and toggles), and where their other channels differ (curves, brush strokes, an applied
// mask, roto shapes, identity pins). This module says exactly where, in the labels the rest of the UI uses.

import { changedKeys, formatValue } from '../../history/historyEngine';
import { isRecipeKey, settingLabel } from '../../history/historyChannels';

export const SLOT_ORDER = Object.freeze(['A', 'B', 'C', 'D']);

// The non-settings channels that can differ between two slots, and how a difference reads.
const CHANNEL_NAME = Object.freeze({
  curves: 'Parameter curves',
  maskStrokes: 'Occlusion brush strokes',
  mask: 'Applied occlusion mask',
  roto: 'Roto shapes',
  pins: 'Identity pins',
});

const same = (a, b) => {
  if (Object.is(a, b)) return true;
  try { return JSON.stringify(a) === JSON.stringify(b); } catch { return false; }
};

/** The recipe settings a slot kept ({} for an empty slot). */
export const slotSettings = (slot) => (slot?.state?.settings && typeof slot.state.settings === 'object' ? slot.state.settings : {});

/**
 * The settings that differ between two slots, as rows { key, label, from, to, fromText, toText }.
 * Only RECIPE keys count: a slot is a look, not a machine (theme, provider and server keys are never part of a recall, so
 * they are never reported as a difference either). Sorted by label so the list does not reshuffle between renders.
 */
export function settingDeltas(base, other) {
  const a = slotSettings(base);
  const b = slotSettings(other);
  return changedKeys(a, b)
    .filter(isRecipeKey)
    .map((key) => ({
      key, label: settingLabel(key), from: a[key], to: b[key], fromText: formatValue(a[key]), toText: formatValue(b[key]),
    }))
    .sort((x, y) => x.label.localeCompare(y.label));
}

/** Names of the non-settings channels that differ between two slots (curves, strokes, ...). */
export function channelDeltas(base, other) {
  const out = [];
  for (const [id, name] of Object.entries(CHANNEL_NAME)) {
    const inA = base?.state && id in base.state;
    const inB = other?.state && id in other.state;
    if (!inA && !inB) continue;
    if (!same(base?.state?.[id], other?.state?.[id])) out.push(name);
  }
  return out;
}

/** The frame a slot's picture is of, from its frameKey (`<media>_<frame>`), or null. */
export function slotFrame(slot) {
  const m = /_(\d+)$/.exec(String(slot?.frameKey || ''));
  return m ? Number(m[1]) : null;
}

/** "14:03:09" */
export function slotClock(ts) {
  if (!Number.isFinite(Number(ts)) || !ts) return '';
  try { return new Date(ts).toLocaleTimeString([], { hour12: false }); } catch { return ''; }
}

/** The slot the others are compared against: the requested one if it is filled, else the first filled slot, else null. */
export function pickBaseline(slots, requested = null) {
  if (requested && slots?.[requested]) return requested;
  return SLOT_ORDER.find((id) => slots?.[id]) || null;
}

/**
 * Everything the Quad compare shows, per quadrant.
 *
 * @param slots       { A, B, C, D } (null for empty) - the bank
 * @param baselineId  the slot the others are measured against (null = the first filled one)
 * @param currentFrameKey the media + frame on screen, to say which pictures are of another moment
 * @returns an array of four { id, slot, empty, baseline, hasPicture, ts, clock, frame, otherFrame, deltas, channels, same }
 *   `same` is true for a filled, non-baseline slot whose recipe is identical to the baseline's (the picture should match).
 */
export function buildQuad(slots, baselineId = null, currentFrameKey = null) {
  const baseId = pickBaseline(slots, baselineId);
  const base = baseId ? slots[baseId] : null;
  return SLOT_ORDER.map((id) => {
    const slot = slots?.[id] || null;
    const isBase = id === baseId;
    const deltas = slot && base && !isBase ? settingDeltas(base, slot) : [];
    const channels = slot && base && !isBase ? channelDeltas(base, slot) : [];
    return {
      id,
      slot,
      empty: !slot,
      baseline: isBase,
      hasPicture: !!slot?.bitmap,
      ts: slot?.ts || 0,
      clock: slotClock(slot?.ts),
      frame: slotFrame(slot),
      otherFrame: !!(slot?.frameKey && currentFrameKey && slot.frameKey !== currentFrameKey),
      deltas,
      channels,
      same: !!slot && !isBase && !!base && deltas.length === 0 && channels.length === 0,
    };
  });
}

/**
 * The comparison as a table: one row per setting that is not the same in every filled slot, one value per slot.
 * [{ key, label, values: { A, B, C, D } (display text, '' for an empty slot), differs: Set of slot ids that differ from the baseline }]
 */
export function diffTable(slots, baselineId = null) {
  const baseId = pickBaseline(slots, baselineId);
  const filled = SLOT_ORDER.filter((id) => slots?.[id]);
  if (filled.length < 2) return [];
  const keys = new Set();
  for (const id of filled) {
    if (id === baseId) continue;
    for (const d of settingDeltas(slots[baseId], slots[id])) keys.add(d.key);
  }
  return [...keys].map((key) => {
    const values = {};
    const differs = new Set();
    for (const id of SLOT_ORDER) {
      const s = slots?.[id];
      values[id] = s ? formatValue(slotSettings(s)[key]) : '';
      if (s && id !== baseId && !same(slotSettings(slots[baseId])[key], slotSettings(s)[key])) differs.add(id);
    }
    return { key, label: settingLabel(key), values, differs };
  }).sort((x, y) => x.label.localeCompare(y.label));
}

/** A short line for a quadrant: "3 settings differ from A" and the first few names. */
export function deltaSummary(q, baselineId, max = 4) {
  if (q.empty) return { headline: 'Empty slot', names: [], more: 0 };
  if (q.baseline) return { headline: 'Baseline', names: [], more: 0 };
  const n = q.deltas.length + q.channels.length;
  if (n === 0) return { headline: `Same recipe as ${baselineId}`, names: [], more: 0 };
  const names = [...q.deltas.map((d) => `${d.label}: ${d.fromText} → ${d.toText}`), ...q.channels.map((c) => `${c} differ`)];
  return {
    headline: `${n} difference${n === 1 ? '' : 's'} from ${baselineId}`,
    names: names.slice(0, max),
    more: Math.max(0, names.length - max),
  };
}
