/**
 * markerModel.ts: chapter / keyframe markers with a colour tag and a free-text note.
 *
 * Where a marker IS stays what it always was: a sorted list of 1-based UI frame numbers under
 * `roop_markers_<clip>` (the legacy Timeline reads it, useWorkstationShortcuts' `M` key writes it, and
 * the legacy reader drops anything that is not a bare number, so richer entries there would be
 * erased the next time it saved). The colour and the note live in a SIDECAR keyed by frame under
 * `roop_marker_meta_<clip>`. Moving a marker therefore moves the number AND re-keys its sidecar entry;
 * a marker the sidecar has never seen is just a default-coloured, note-less marker.
 *
 * Frames in this module's public functions are 0-based timeline frames; `toUi` / `fromUi` convert
 * to the stored 1-based numbers at the one boundary.
 */

import type { ChapterMarker } from '../timeline/timelineSnap';

export interface MarkerColor {
  id: string;
  hex: string;
  label: string;
}

export const MARKER_COLORS: readonly MarkerColor[] = Object.freeze([
  { id: 'sky', hex: '#38bdf8', label: 'Chapter' },
  { id: 'emerald', hex: '#34d399', label: 'Approved' },
  { id: 'amber', hex: '#fbbf24', label: 'Review' },
  { id: 'rose', hex: '#fb7185', label: 'Problem' },
  { id: 'violet', hex: '#a78bfa', label: 'Note' },
]);

export const DEFAULT_MARKER_COLOR = MARKER_COLORS[0].hex;
export const MAX_NOTE_CHARS = 280;

export interface MarkerMeta {
  color: string;
  note: string;
}

/** Sidecar keyed by the stored (1-based) frame number as a string. */
export type MarkerMetaMap = Record<string, MarkerMeta>;

export const markersKey = (clip?: string | null) => `roop_markers_${clip || 'default'}`;
export const markerMetaKey = (clip?: string | null) => `roop_marker_meta_${clip || 'default'}`;

export const toUi = (frame0: number) => Math.max(1, Math.round(frame0) + 1);
export const fromUi = (frame1: number) => Math.max(0, Math.round(frame1) - 1);

const isHex = (v: unknown): v is string => typeof v === 'string' && /^#[0-9a-fA-F]{6}$/.test(v);

export function sanitizeFrames(raw: unknown, maxFrames = Infinity): number[] {
  if (!Array.isArray(raw)) return [];
  const set = new Set<number>();
  for (const n of raw) {
    if (typeof n === 'number' && Number.isFinite(n) && n >= 1 && n <= maxFrames) set.add(Math.round(n));
  }
  return [...set].sort((a, b) => a - b);
}

export function sanitizeMeta(raw: unknown): MarkerMetaMap {
  const out: MarkerMetaMap = {};
  if (!raw || typeof raw !== 'object') return out;
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (!/^\d+$/.test(k) || !v || typeof v !== 'object') continue;
    const o = v as Record<string, unknown>;
    out[k] = {
      color: isHex(o.color) ? o.color : DEFAULT_MARKER_COLOR,
      note: typeof o.note === 'string' ? o.note.slice(0, MAX_NOTE_CHARS) : '',
    };
  }
  return out;
}

/** The first line of the note is the marker's label on the ruler; a bare marker is "Marker". */
export function markerLabel(meta: MarkerMeta | undefined): string {
  const first = (meta?.note || '').split('\n')[0].trim();
  return first || 'Marker';
}

export function buildMarkers(frames1: ReadonlyArray<number>, meta: MarkerMetaMap): ChapterMarker[] {
  return frames1.map((f1) => {
    const m = meta[String(f1)];
    return {
      id: `marker_${f1}`,
      frame: fromUi(f1),
      label: markerLabel(m),
      color: m?.color || DEFAULT_MARKER_COLOR,
      note: m?.note || '',
    };
  });
}

export interface MarkerState {
  frames: number[];
  meta: MarkerMetaMap;
}

/** Add a marker at 0-based `frame0`. A frame that already has one is left alone. */
export function addMarker(state: MarkerState, frame0: number, maxFrames: number, init?: Partial<MarkerMeta>): MarkerState {
  const f1 = toUi(frame0);
  if (f1 > maxFrames || state.frames.includes(f1)) return state;
  const frames = [...state.frames, f1].sort((a, b) => a - b);
  const meta = { ...state.meta, [String(f1)]: { color: init?.color && isHex(init.color) ? init.color : DEFAULT_MARKER_COLOR, note: (init?.note || '').slice(0, MAX_NOTE_CHARS) } };
  return { frames, meta };
}

/**
 * Move the marker at `fromFrame0` to `toFrame0`, carrying its colour and note. Refuses (returns the
 * state unchanged) to land on a frame that already has a marker: two markers on one frame would
 * share a sidecar key and one of them would silently lose its note.
 */
export function moveMarker(state: MarkerState, fromFrame0: number, toFrame0: number, maxFrames: number): MarkerState {
  const from = toUi(fromFrame0);
  const to = Math.max(1, Math.min(maxFrames, toUi(toFrame0)));
  if (from === to || !state.frames.includes(from) || state.frames.includes(to)) return state;
  const frames = state.frames.map((f) => (f === from ? to : f)).sort((a, b) => a - b);
  const meta = { ...state.meta };
  if (meta[String(from)]) {
    meta[String(to)] = meta[String(from)];
    delete meta[String(from)];
  }
  return { frames, meta };
}

export function updateMarker(state: MarkerState, frame0: number, patch: Partial<MarkerMeta>): MarkerState {
  const f1 = toUi(frame0);
  if (!state.frames.includes(f1)) return state;
  const cur = state.meta[String(f1)] || { color: DEFAULT_MARKER_COLOR, note: '' };
  const next: MarkerMeta = {
    color: patch.color && isHex(patch.color) ? patch.color : cur.color,
    note: typeof patch.note === 'string' ? patch.note.slice(0, MAX_NOTE_CHARS) : cur.note,
  };
  return { frames: state.frames, meta: { ...state.meta, [String(f1)]: next } };
}

/** `marker_<frame1>` (see buildMarkers) -> the 0-based frame it was built for, or -1. */
export function markerIdToFrame0(id: string): number {
  const m = /^marker_(\d+)$/.exec(id);
  return m ? fromUi(Number(m[1])) : -1;
}

export interface MarkerEdit {
  /** target 0-based frame (a drag) */
  frame?: number;
  color?: string;
  note?: string;
}

/**
 * One edit from the UI: optionally move the marker named by `id`, then optionally recolour / re-note
 * it. A refused move (onto another marker) leaves the marker where it was but still applies the
 * colour / note, so a drag that ends on a neighbour does not throw the user's text away.
 */
export function editMarker(state: MarkerState, id: string, edit: MarkerEdit, maxFrames: number): MarkerState {
  const from = markerIdToFrame0(id);
  if (from < 0) return state;
  let next = state;
  let at = from;
  if (edit.frame !== undefined && Math.round(edit.frame) !== from) {
    const target = Math.max(0, Math.min(maxFrames - 1, Math.round(edit.frame)));
    const moved = moveMarker(next, from, target, maxFrames);
    if (moved !== next) {
      next = moved;
      at = target;
    }
  }
  if (edit.color !== undefined || edit.note !== undefined) next = updateMarker(next, at, { color: edit.color, note: edit.note });
  return next;
}

export function removeMarker(state: MarkerState, frame0: number): MarkerState {
  const f1 = toUi(frame0);
  if (!state.frames.includes(f1)) return state;
  const meta = { ...state.meta };
  delete meta[String(f1)];
  return { frames: state.frames.filter((f) => f !== f1), meta };
}

/**
 * Replace the frame list (the `M` key and the legacy view write it). Notes of markers that are gone
 * are dropped; markers that appeared keep the defaults.
 */
export function withFrames(state: MarkerState, frames1: ReadonlyArray<number>, maxFrames: number): MarkerState {
  const frames = sanitizeFrames(frames1, maxFrames);
  const meta: MarkerMetaMap = {};
  for (const f of frames) if (state.meta[String(f)]) meta[String(f)] = state.meta[String(f)];
  return { frames, meta };
}
