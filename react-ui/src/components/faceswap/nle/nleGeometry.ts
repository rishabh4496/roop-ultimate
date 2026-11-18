/**
 * nleGeometry.ts: the one place the NLE maps between pixels and frames.
 *
 * Every lane is a `[header | lane area]` row. The header is a fixed-width column, so a pointer's
 * x only means a frame once the header has been subtracted. NleTimeline used to divide by the
 * WHOLE row's width while it drew every overlay with `calc(14rem + (100% - 14rem) * r)`: the two
 * halves disagreed, so a click landed on a different frame than the playhead it created (a click on
 * frame 0 read as ~18% of the clip; the error shrinks to nothing at the right edge, which is why
 * testing near the end of a clip never showed it). Both directions go through this module now.
 */

/** Width of the left track-header column in CSS px. Every row's header takes this width from here. */
export const LANE_HEADER_PX = 256;

export interface LaneRect {
  left: number;
  width: number;
}

/** The lane area of a row: the row's rect minus the header column. */
export function laneAreaRect(row: LaneRect, headerPx: number = LANE_HEADER_PX): LaneRect {
  const header = Math.max(0, headerPx);
  return { left: row.left + header, width: Math.max(0, row.width - header) };
}

/** 0..1 position of `frame` inside [viewStart, viewEnd] (not clamped: callers cull off-screen). */
export function frameToRatio(frame: number, viewStart: number, viewEnd: number): number {
  const span = viewEnd - viewStart;
  if (!(span > 0)) return 0;
  return (frame - viewStart) / span;
}

/** CSS `left` for something sitting on a lane at `frame`, relative to the ROW (header included). */
export function frameToRowLeft(
  frame: number,
  viewStart: number,
  viewEnd: number,
  headerPx: number = LANE_HEADER_PX,
): string {
  const r = frameToRatio(frame, viewStart, viewEnd);
  return `calc(${headerPx}px + (100% - ${headerPx}px) * ${r})`;
}

/** Pixel x of `frame` inside a lane area `widthPx` wide. */
export function frameToLaneX(frame: number, viewStart: number, viewEnd: number, widthPx: number): number {
  return frameToRatio(frame, viewStart, viewEnd) * widthPx;
}

/**
 * Frame under a pointer. `area` is the LANE AREA rect (use laneAreaRect on a row rect).
 * Rounded to the nearest frame and clamped to [0, maxFrames - 1].
 */
export function clientXToFrame(
  clientX: number,
  area: LaneRect | null | undefined,
  viewStart: number,
  viewEnd: number,
  maxFrames: number,
): number {
  const last = Math.max(0, Math.floor(maxFrames) - 1);
  if (!area || !(area.width > 0)) return Math.max(0, Math.min(last, Math.round(viewStart)));
  const ratio = Math.max(0, Math.min(1, (clientX - area.left) / area.width));
  const raw = viewStart + ratio * (viewEnd - viewStart);
  return Math.max(0, Math.min(last, Math.round(raw)));
}

/** True when `clientX` is over the lane area (not over the header column). */
export function isOverLaneArea(clientX: number, area: LaneRect | null | undefined): boolean {
  return !!area && clientX >= area.left && clientX <= area.left + area.width;
}
