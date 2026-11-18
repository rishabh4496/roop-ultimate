/**
 * transportModel.ts: the pure rules behind the transport deck.
 *
 * The deck drives the SAME state the keyboard drives (FaceSwap's isPlaying / playbackRate / isLooping,
 * useWorkstationShortcuts' stepFrame and markers), so nothing here owns playback; it only decides
 * where a button should take the playhead.
 */

export const PLAYBACK_RATES: readonly number[] = Object.freeze([0.25, 0.5, 1, 2, 4]);

export const formatRate = (rate: number): string => `${rate}×`;

/** Nearest allowed rate (a stale or hand-edited value falls onto the ladder instead of showing none). */
export function snapRate(rate: number): number {
  if (!Number.isFinite(rate)) return 1;
  let best = PLAYBACK_RATES[0];
  for (const r of PLAYBACK_RATES) if (Math.abs(r - rate) < Math.abs(best - rate)) best = r;
  return best;
}

/** One notch up (+1) or down (-1) the rate ladder, stopping at the ends. */
export function shiftRate(rate: number, dir: -1 | 1): number {
  const idx = PLAYBACK_RATES.indexOf(snapRate(rate));
  return PLAYBACK_RATES[Math.max(0, Math.min(PLAYBACK_RATES.length - 1, idx + dir))];
}

/** Move `delta` frames, clamped to the clip. */
export function stepTo(frame: number, delta: number, maxFrames: number): number {
  return Math.max(0, Math.min(Math.max(0, Math.floor(maxFrames) - 1), Math.round(frame + delta)));
}

/**
 * Previous / next entry of `frames` strictly before / after `current`, or null when there is none.
 * `frames` need not be sorted.
 */
export function neighbourFrame(frames: ReadonlyArray<number>, current: number, dir: -1 | 1): number | null {
  let best: number | null = null;
  for (const f of frames) {
    if (dir > 0 ? f > current : f < current) {
      if (best === null || (dir > 0 ? f < best : f > best)) best = f;
    }
  }
  return best;
}

export interface LoopRegion {
  in: number;
  out: number;
}

/** A well-formed region inside the clip (in <= out). */
export function loopRegion(trimIn: number, trimOut: number, maxFrames: number): LoopRegion {
  const last = Math.max(0, Math.floor(maxFrames) - 1);
  const a = Math.max(0, Math.min(last, Math.round(Math.min(trimIn, trimOut))));
  const b = Math.max(0, Math.min(last, Math.round(Math.max(trimIn, trimOut))));
  return { in: a, out: b };
}

/** Frames inside a region, both ends included (the readout beside the In/Out handles). */
export function regionLength(region: LoopRegion): number {
  return region.out - region.in + 1;
}

/** The frame a looping player should show next: wrap to `in` once it passes `out`. */
export function advanceWithLoop(frame: number, region: LoopRegion, loop: boolean): number {
  if (loop && frame > region.out) return region.in;
  return frame;
}

/** `00:00:05:12` style duration for the region readout ("12 f" under a second). */
export function formatDuration(frames: number, fps: number): string {
  const fr = Math.max(1, Math.round(fps));
  const f = Math.max(0, Math.round(frames));
  const secs = Math.floor(f / fr);
  const rem = f % fr;
  const mm = Math.floor(secs / 60);
  const ss = secs % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  if (secs === 0) return `${rem} f`;
  return `${mm > 0 ? `${mm}:${pad(ss)}` : `${ss}`}s ${pad(rem)}f`;
}
