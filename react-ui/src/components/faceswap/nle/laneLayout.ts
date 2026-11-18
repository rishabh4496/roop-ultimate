/**
 * laneLayout.ts: per-lane header state (hide, lock, height) for the NLE track stack.
 *
 * What each toggle MEANS, lane by lane, so none of them is a button that only looks wired:
 *   hidden  the lane collapses to a thin header strip and its body is not rendered (nothing is drawn,
 *           fetched or hit-tested for it). On the audio lane the same control is MUTE (see audioMuted).
 *   locked  edits on that lane are refused: segment blocks cannot be added or moved, markers cannot be
 *           dragged, keyframes cannot be added / moved / deleted. Reading and seeking still work.
 *   height  compact / normal / tall: the lane's pixel height from LANE_HEIGHTS.
 * The face-track lanes keep their own long-standing meanings (lock = Range Lock, eye = swap on/off),
 * which live in NleTimeline because they call back into the studio project.
 */

export type LaneId = 'v1' | 'health' | 'segments' | 'a1' | 'kf' | 'faces';
export type LaneHeight = 'compact' | 'normal' | 'tall';

export interface LaneState {
  hidden: boolean;
  locked: boolean;
  height: LaneHeight;
}

export type LaneStates = Record<LaneId, LaneState>;

export const LANE_IDS: readonly LaneId[] = Object.freeze(['v1', 'health', 'segments', 'a1', 'kf', 'faces']);
export const HEIGHT_ORDER: readonly LaneHeight[] = Object.freeze(['compact', 'normal', 'tall']);

/** Body height in px per lane and size. A collapsed (hidden) lane is COLLAPSED_PX regardless. */
export const LANE_HEIGHTS: Record<LaneId, Record<LaneHeight, number>> = {
  v1: { compact: 30, normal: 52, tall: 88 },
  health: { compact: 14, normal: 22, tall: 38 },
  segments: { compact: 26, normal: 38, tall: 58 },
  a1: { compact: 30, normal: 46, tall: 84 },
  kf: { compact: 28, normal: 38, tall: 60 },
  faces: { compact: 30, normal: 44, tall: 68 },
};

export const COLLAPSED_PX = 20;

export const LANE_LABELS: Record<LaneId, string> = {
  v1: 'Video',
  health: 'Face coverage',
  segments: 'Swap segments',
  a1: 'Audio',
  kf: 'Keyframes',
  faces: 'Face tracks',
};

export function defaultLaneStates(): LaneStates {
  const out = {} as LaneStates;
  for (const id of LANE_IDS) out[id] = { hidden: false, locked: false, height: 'normal' };
  return out;
}

export function nextHeight(height: LaneHeight): LaneHeight {
  return HEIGHT_ORDER[(HEIGHT_ORDER.indexOf(height) + 1) % HEIGHT_ORDER.length];
}

export function laneBodyPx(id: LaneId, state: LaneState): number {
  return state.hidden ? COLLAPSED_PX : LANE_HEIGHTS[id][state.height];
}

export type LaneAction =
  | { type: 'toggleHidden'; id: LaneId }
  | { type: 'toggleLocked'; id: LaneId }
  | { type: 'cycleHeight'; id: LaneId }
  | { type: 'reset' };

export function reduceLanes(states: LaneStates, action: LaneAction): LaneStates {
  if (action.type === 'reset') return defaultLaneStates();
  const cur = states[action.id];
  if (!cur) return states;
  if (action.type === 'toggleHidden') return { ...states, [action.id]: { ...cur, hidden: !cur.hidden } };
  if (action.type === 'toggleLocked') return { ...states, [action.id]: { ...cur, locked: !cur.locked } };
  return { ...states, [action.id]: { ...cur, height: nextHeight(cur.height) } };
}

/** Accepts anything a previous version or a hand edit may have left in storage. */
export function sanitizeLaneStates(raw: unknown): LaneStates {
  const base = defaultLaneStates();
  if (!raw || typeof raw !== 'object') return base;
  for (const id of LANE_IDS) {
    const v = (raw as Record<string, unknown>)[id];
    if (!v || typeof v !== 'object') continue;
    const o = v as Record<string, unknown>;
    base[id] = {
      hidden: o.hidden === true,
      locked: o.locked === true,
      height: HEIGHT_ORDER.includes(o.height as LaneHeight) ? (o.height as LaneHeight) : 'normal',
    };
  }
  return base;
}

export const LANES_STORAGE_KEY = 'roop_nle_lanes_v1';

/** localStorage is a per-viewer convenience: every access is guarded and the page works without it. */
export function loadLaneStates(storage: Pick<Storage, 'getItem'> | null | undefined): LaneStates {
  try {
    const raw = storage?.getItem(LANES_STORAGE_KEY);
    return sanitizeLaneStates(raw ? JSON.parse(raw) : null);
  } catch {
    return defaultLaneStates();
  }
}

export function saveLaneStates(storage: Pick<Storage, 'setItem'> | null | undefined, states: LaneStates): void {
  try {
    storage?.setItem(LANES_STORAGE_KEY, JSON.stringify(states));
  } catch {
    /* private mode / quota: the layout just does not persist */
  }
}
