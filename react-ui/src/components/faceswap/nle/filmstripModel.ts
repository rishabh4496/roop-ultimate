/**
 * filmstripModel.ts: which stills the V1 lane shows at a given zoom.
 *
 * Two properties the previous slot picker (`round(viewStart + (i + 0.5) * span / n)`) did not have:
 *
 *  1. Density follows PIXELS. The stride (frames per still) is the smallest 1-2-5 step that keeps a
 *     still at least `thumbPx` wide, so zooming in 4x shows ~4x as many distinct stills until every
 *     frame has its own, and zooming out merges them again.
 *  2. Slots sit on a fixed GRID (multiples of the stride), not at offsets from the view start. Panning
 *     one frame used to move every slot's centre by a frame, which missed the still cache and
 *     re-requested all of them; on a grid the same pan reuses every still it already has and only
 *     the strip's new edge is fetched.
 *
 * Frames here are 0-based timeline frames. The server's /api/target/preview(_grid) take 1-based UI
 * frames: use `toServerFrame` at the request boundary (the old hook did not, so every still showed
 * the frame BEFORE the one it was labelled with).
 */

export interface FilmSlot {
  slotIndex: number;
  /** 0-based frame whose still this slot shows (the slot's first frame) */
  frame: number;
  /** frames the slot spans */
  stride: number;
  leftPct: number;
  widthPct: number;
}

export interface FilmstripOptions {
  viewStart: number;
  viewEnd: number;
  widthPx: number;
  maxFrames: number;
  thumbPx?: number;
  maxSlots?: number;
}

/** 1-2-5 strides: 1, 2, 5, 10, 20, 50, 100, ... */
export function nearestStride(minFrames: number): number {
  const need = Math.max(1, Math.ceil(minFrames));
  let decade = 1;
  while (decade < need) {
    for (const m of [1, 2, 5]) {
      if (decade * m >= need) return decade * m;
    }
    decade *= 10;
  }
  return decade;
}

export function filmstripSlots({ viewStart, viewEnd, widthPx, maxFrames, thumbPx = 100, maxSlots = 48 }: FilmstripOptions): FilmSlot[] {
  if (!(widthPx > 0) || maxFrames <= 1) return [];
  const span = Math.max(1, viewEnd - viewStart);
  const pxPerFrame = widthPx / span;
  const stride = nearestStride(thumbPx / pxPerFrame);
  const last = maxFrames - 1;

  const slots: FilmSlot[] = [];
  const firstK = Math.floor(Math.max(0, viewStart) / stride);
  for (let k = firstK; k * stride <= Math.min(viewEnd, last) && slots.length < maxSlots; k++) {
    const frame = k * stride;
    slots.push({
      slotIndex: k,
      frame,
      stride,
      leftPct: ((frame - viewStart) / span) * 100,
      widthPct: (Math.min(stride, maxFrames - frame) / span) * 100,
    });
  }
  return slots;
}

/** 0-based timeline frame -> the 1-based frame number the preview endpoints take. */
export const toServerFrame = (frame0: number): number => Math.max(1, Math.round(frame0) + 1);
