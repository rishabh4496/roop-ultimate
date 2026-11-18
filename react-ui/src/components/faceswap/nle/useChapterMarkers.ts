/**
 * useChapterMarkers.ts: the single owner of a clip's markers (frame list + colour/note sidecar).
 *
 * One owner matters because there used to be two: useWorkstationShortcuts kept a number[] of its own
 * for the `M` key while NleTimeline kept an in-memory list for its `+ Marker` button, so a marker
 * made one way never showed up the other way. FaceSwap now builds this once and hands the frame list
 * to useWorkstationShortcuts (`markers` + `onMarkersChange`) and the rich markers to NleTimeline.
 *
 * Storage is per clip and per viewer (localStorage, every access guarded): see markerModel.ts for why
 * the frame list keeps its legacy key and the colour/note live beside it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ChapterMarker } from '../timeline/timelineSnap';
import {
  type MarkerMeta,
  type MarkerState,
  addMarker,
  buildMarkers,
  editMarker,
  markerIdToFrame0,
  markerMetaKey,
  markersKey,
  removeMarker,
  sanitizeFrames,
  sanitizeMeta,
  withFrames,
} from './markerModel';

const EMPTY: MarkerState = { frames: [], meta: {} };

function readState(clip: string): MarkerState {
  try {
    const frames = sanitizeFrames(JSON.parse(localStorage.getItem(markersKey(clip)) || '[]'));
    const meta = sanitizeMeta(JSON.parse(localStorage.getItem(markerMetaKey(clip)) || '{}'));
    return { frames, meta };
  } catch {
    return EMPTY;
  }
}

function writeState(clip: string, state: MarkerState): void {
  try {
    if (state.frames.length) localStorage.setItem(markersKey(clip), JSON.stringify(state.frames));
    else localStorage.removeItem(markersKey(clip));
    const live = Object.keys(state.meta).filter((k) => state.frames.includes(Number(k)));
    if (live.length) {
      const trimmed = Object.fromEntries(live.map((k) => [k, state.meta[k]]));
      localStorage.setItem(markerMetaKey(clip), JSON.stringify(trimmed));
    } else {
      localStorage.removeItem(markerMetaKey(clip));
    }
  } catch {
    /* private mode / quota: markers are a convenience layered on the clip */
  }
}

export interface ChapterMarkers {
  markers: ChapterMarker[];
  /** sorted 1-based UI frame numbers: what useWorkstationShortcuts and the legacy Timeline read */
  frames: number[];
  add: (marker: Pick<ChapterMarker, 'frame'> & Partial<ChapterMarker>) => void;
  /** frame changed -> moved (note and colour travel with it); colour / note changed -> updated */
  update: (marker: ChapterMarker) => void;
  remove: (id: string) => void;
  /** the `M` key and the legacy view write a bare frame list */
  setFrames: (frames1: number[]) => void;
}

export function useChapterMarkers(clipKey: string | null | undefined, maxFrames: number): ChapterMarkers {
  const clip = clipKey || 'default';
  const [state, setState] = useState<MarkerState>(() => readState(clip));
  const stateRef = useRef(state);
  stateRef.current = state;
  const maxRef = useRef(maxFrames);
  maxRef.current = maxFrames;

  // Re-read when the clip changes. maxFrames is deliberately NOT a dependency: it arrives a tick after
  // the clip does, and clamping the stored list against the previous clip's length would write the
  // damage back (the same trap useSegments documents).
  useEffect(() => {
    setState(readState(clip));
  }, [clip]);

  const commit = useCallback(
    (next: MarkerState) => {
      if (next === stateRef.current) return;
      stateRef.current = next;
      setState(next);
      writeState(clip, next);
    },
    [clip],
  );

  const add = useCallback<ChapterMarkers['add']>(
    (marker) => commit(addMarker(stateRef.current, marker.frame, maxRef.current, { color: marker.color, note: marker.note })),
    [commit],
  );

  const update = useCallback<ChapterMarkers['update']>(
    (marker) => commit(editMarker(stateRef.current, marker.id, { frame: marker.frame, color: marker.color, note: marker.note }, maxRef.current)),
    [commit],
  );

  const remove = useCallback<ChapterMarkers['remove']>(
    (id) => {
      const f0 = markerIdToFrame0(id);
      if (f0 >= 0) commit(removeMarker(stateRef.current, f0));
    },
    [commit],
  );

  const setFrames = useCallback<ChapterMarkers['setFrames']>(
    (frames1) => commit(withFrames(stateRef.current, frames1, Math.max(maxRef.current, 1))),
    [commit],
  );

  const markers = useMemo(() => buildMarkers(state.frames, state.meta), [state]);
  return { markers, frames: state.frames, add, update, remove, setFrames };
}

export type { MarkerMeta };
