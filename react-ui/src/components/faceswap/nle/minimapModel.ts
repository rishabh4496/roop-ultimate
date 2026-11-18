/**
 * minimapModel.ts: drag arithmetic for the overview navigator.
 *
 * The navigator draws the whole clip with a window over the part the lanes show. Dragging the window
 * pans, dragging its edges zooms, clicking outside it recentres it. All three keep the window a
 * valid [start, end] inside [0, maxFrames - 1] with at least `minSpan` frames.
 */

export type MinimapMode = 'move' | 'resize-left' | 'resize-right' | 'jump';
export type ViewWindow = readonly [number, number];

const clampInt = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(n)));

/** Which part of the navigator a pointer at `ratio` (0..1 of the clip) is on. */
export function hitMinimap(
  ratio: number,
  view: ViewWindow,
  maxFrames: number,
  widthPx: number,
  edgePx = 7,
): MinimapMode {
  const last = Math.max(1, maxFrames - 1);
  const left = (view[0] / last) * widthPx;
  const right = (view[1] / last) * widthPx;
  const x = ratio * widthPx;
  // Narrow windows: the edges must not swallow the whole body, or the window could never be moved.
  const edge = Math.min(edgePx, Math.max(2, (right - left) / 3));
  if (x < left - edge || x > right + edge) return 'jump';
  if (Math.abs(x - left) <= edge) return 'resize-left';
  if (Math.abs(x - right) <= edge) return 'resize-right';
  return 'move';
}

export interface MinimapDrag {
  mode: MinimapMode;
  /** pointer ratio when the drag began */
  startRatio: number;
  /** window when the drag began */
  startView: ViewWindow;
  /** pointer ratio now */
  ratio: number;
  maxFrames: number;
  minSpan?: number;
}

export function dragMinimap({ mode, startRatio, startView, ratio, maxFrames, minSpan = 4 }: MinimapDrag): [number, number] {
  const last = Math.max(1, maxFrames - 1);
  const span = Math.max(1, startView[1] - startView[0]);
  const floorSpan = Math.min(minSpan, last);

  if (mode === 'jump') {
    const center = ratio * last;
    const start = clampInt(center - span / 2, 0, last - span);
    return [start, start + span];
  }
  const delta = (ratio - startRatio) * last;
  if (mode === 'move') {
    const start = clampInt(startView[0] + delta, 0, last - span);
    return [start, start + span];
  }
  if (mode === 'resize-left') {
    const start = clampInt(startView[0] + delta, 0, startView[1] - floorSpan);
    return [start, startView[1]];
  }
  const end = clampInt(startView[1] + delta, startView[0] + floorSpan, last);
  return [startView[0], end];
}

/**
 * The visible window after the clip's last frame changes (`prevLast` -> `nextLast`).
 *
 * A window that showed the WHOLE clip ("fit") keeps showing the whole clip: Extras learns a clip's length
 * only after its metadata loads, so the window was created while the clip was 1 frame long and the old
 * clamp left it at [0, 1] for good (a click at the middle of the lane read as frame 1, markers past frame 1
 * were not drawn). A zoomed window is clamped into the new clip instead of being reset under the user.
 */
export function reconcileView(prev: ViewWindow, prevLast: number, nextLast: number, minSpan = 1): [number, number] {
  const last = Math.max(0, nextLast);
  const wasFit = prev[0] <= 0 && prev[1] >= prevLast;
  if (wasFit) return [0, last];
  const start = Math.max(0, Math.min(prev[0], Math.max(0, last - minSpan)));
  const end = Math.max(start + minSpan, Math.min(prev[1], last));
  return [start, Math.min(end, last)];
}

/**
 * Zoom slider <-> visible span, on a log scale so every notch is the same zoom RATIO (a linear slider
 * would spend nearly all its travel between "whole clip" and "a few thousand frames").
 *   0   = the whole clip (`last` frames across)
 *   100 = `minSpan` frames across
 */
export function spanToZoom(span: number, last: number, minSpan = 4): number {
  const lo = Math.max(2, Math.min(minSpan, last));
  const hi = Math.max(lo + 1, last);
  const t = (Math.log(hi) - Math.log(Math.max(lo, Math.min(hi, span)))) / (Math.log(hi) - Math.log(lo));
  return Math.round(Math.max(0, Math.min(1, t)) * 100);
}

export function zoomToSpan(zoom: number, last: number, minSpan = 4): number {
  const lo = Math.max(2, Math.min(minSpan, last));
  const hi = Math.max(lo + 1, last);
  const t = Math.max(0, Math.min(100, zoom)) / 100;
  return Math.round(Math.exp(Math.log(hi) * (1 - t) + Math.log(lo) * t));
}

/** Window after a wheel notch over the navigator: zoom about the pointer (factor < 1 zooms in). */
export function wheelMinimap(view: ViewWindow, ratio: number, factor: number, maxFrames: number, minSpan = 4): [number, number] {
  const last = Math.max(1, maxFrames - 1);
  const span = Math.max(1, view[1] - view[0]);
  const nextSpan = clampInt(span * factor, Math.min(minSpan, last), last);
  const pivot = ratio * last;
  // Keep the frame under the pointer where it is within the window when it is inside the window,
  // otherwise zoom about the window centre.
  const inside = pivot >= view[0] && pivot <= view[1];
  const anchor = inside ? (pivot - view[0]) / span : 0.5;
  const focus = inside ? pivot : (view[0] + view[1]) / 2;
  const start = clampInt(focus - anchor * nextSpan, 0, last - nextSpan);
  return [start, start + nextSpan];
}
