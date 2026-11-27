// Pure view-window maths for the multi-track timeline. No React, no DOM: every function here is
// a number-in/number-out step so a 100-frame clip and a 100,000-frame clip cost the same, and so
// the render checks can run it under node.
//
// A VIEW is {start, end}: the inclusive 1-based frame range the lanes show. A frame's position on
// the lane is its LEFT-EDGE-free centre point, x(f) = (f - start) / (end - start) * width, so the
// playhead at frame f and the centre of that frame's cell are the same pixel. A frame "cell" is
// [f - 0.5, f + 0.5] in the same units, which is what the frame-by-frame view draws.

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));

// A cell narrower than this has no room for its `#00412` label, so the ruler never labels closer.
export const FRAME_CELL_MIN_PX = 56;
// The frame-by-frame view: zoom stops when one frame is as wide as one filmstrip still (16:9 at the
// track's height), so every cell holds its own picture, its number and nothing else.
export const FRAME_VIEW_PX = 140;
// Below this many px per frame the lanes are in "overview" density and draw no per-frame detail.
export const FRAME_CELL_DETAIL_PX = 14;

// Label steps a human would choose, in seconds and in frames.
const NICE_SECONDS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
const NICE_FRAMES = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000];

/** The smallest window the view may shrink to: the frame-by-frame view, FRAME_VIEW_PX per frame. */
export function minSpan(trackW, maxFrames) {
  const cap = Math.max(1, maxFrames - 1);
  const want = Math.max(4, Math.floor((trackW || 0) / FRAME_VIEW_PX));
  return Math.min(cap, want);
}

/** A degenerate or missing window is the whole clip. */
export function normalizeView(view, maxFrames) {
  const hi = Math.max(1, maxFrames);
  if (!view || !(view.end > view.start)) return { start: 1, end: hi };
  const start = clamp(Math.round(view.start), 1, hi);
  const end = clamp(Math.round(view.end), start, hi);
  return end > start ? { start, end } : { start: 1, end: hi };
}

export const viewSpan = (view) => Math.max(1, view.end - view.start);
export const isZoomed = (view, maxFrames) => viewSpan(view) < maxFrames - 1;
export const pxPerFrame = (trackW, view) => (trackW || 0) / viewSpan(view);

export const xAtFrame = (f, trackW, view) => ((f - view.start) / viewSpan(view)) * trackW;
export const frameAtX = (x, trackW, view, maxFrames) =>
  clamp(Math.round(view.start + (x / Math.max(1, trackW)) * viewSpan(view)), 1, Math.max(1, maxFrames));

/** Place a window of `span` frames starting at `start`, kept inside the clip. */
export function placeView(start, span, maxFrames) {
  const s = clamp(Math.round(start), 1, Math.max(1, maxFrames - span));
  return { start: s, end: Math.min(maxFrames, s + span) };
}

/**
 * Zoom so that `anchorFrame` stays under the same pixel. factor > 1 zooms out, < 1 zooms in.
 * `anchorRatio` is where that frame sits across the lane (0..1); omit it to keep the frame where
 * it already is, which is what a zoom anchored at the playhead or the cursor wants.
 */
export function zoomView(view, maxFrames, trackW, factor, anchorFrame, anchorRatio) {
  const v = normalizeView(view, maxFrames);
  const span = viewSpan(v);
  const lo = minSpan(trackW, maxFrames);
  const hi = Math.max(lo, maxFrames - 1);
  let next = clamp(Math.round(span * factor), lo, hi);
  // Rounding can swallow a small factor entirely (span 30, factor 0.99 -> 30); a zoom request
  // that asked for movement must move by at least one frame or the wheel feels dead.
  if (next === span && factor < 1 && span > lo) next = span - 1;
  if (next === span && factor > 1 && span < hi) next = span + 1;
  const ratio = anchorRatio ?? clamp((anchorFrame - v.start) / span, 0, 1);
  return placeView(anchorFrame - ratio * next, next, maxFrames);
}

export function panView(view, maxFrames, deltaFrames) {
  const v = normalizeView(view, maxFrames);
  return placeView(v.start + deltaFrames, viewSpan(v), maxFrames);
}

/** Centre the window on a frame without changing its width. */
export function centerView(view, maxFrames, frame) {
  const v = normalizeView(view, maxFrames);
  const span = viewSpan(v);
  return placeView(frame - span / 2, span, maxFrames);
}

/** The tightest window that shows [start, end] whole, never narrower than the zoom floor. */
export function fitRange(start, end, maxFrames, trackW) {
  const floor = minSpan(trackW, maxFrames);
  const a = clamp(Math.min(start, end), 1, Math.max(1, maxFrames));
  const b = clamp(Math.max(start, end), 1, Math.max(1, maxFrames));
  const span = Math.min(Math.max(b - a, floor), Math.max(1, maxFrames - 1));
  return placeView((a + b) / 2 - span / 2, span, maxFrames);
}

/** Bring a frame into the window only if it fell off an edge (playhead follow while playing). */
export function followFrame(view, maxFrames, frame, lead = 0.25) {
  const v = normalizeView(view, maxFrames);
  if (frame >= v.start && frame <= v.end) return v;
  return placeView(frame - viewSpan(v) * lead, viewSpan(v), maxFrames);
}

// ── zoom slider (logarithmic: equal travel = equal zoom ratio) ───────────────

/** 0 = whole clip, 1 = frame-by-frame. */
export function zoomFraction(view, maxFrames, trackW) {
  const hi = Math.max(2, maxFrames - 1);
  const lo = Math.max(1, minSpan(trackW, maxFrames));
  if (hi <= lo) return 0;
  return clamp(Math.log(hi / viewSpan(view)) / Math.log(hi / lo), 0, 1);
}

export function spanAtFraction(t, maxFrames, trackW) {
  const hi = Math.max(2, maxFrames - 1);
  const lo = Math.max(1, minSpan(trackW, maxFrames));
  if (hi <= lo) return hi;
  return Math.round(hi * Math.pow(lo / hi, clamp(t, 0, 1)));
}

// ── time formatting ─────────────────────────────────────────────────────────

const pad = (n, w = 2) => String(Math.max(0, Math.trunc(n))).padStart(w, '0');
const nominalFps = (fps) => Math.max(1, Math.round(fps || 25));

/** Frame 1 is timecode 0. Non-drop: the frame field counts at the NOMINAL (rounded) rate. */
export function fmtTimecode(frame, fps) {
  const base = nominalFps(fps);
  const idx = Math.max(0, Math.round(frame) - 1);
  const totalSeconds = Math.floor(idx / base);
  return `${pad(totalSeconds / 3600)}:${pad((totalSeconds % 3600) / 60)}:${pad(totalSeconds % 60)}:${pad(idx % base)}`;
}

/** `#00412` — the 1-based frame number, padded to the clip's own digit count (at least 5). */
export function fmtFrameLabel(frame, maxFrames) {
  const digits = Math.max(5, String(Math.max(1, maxFrames)).length);
  return `#${pad(frame, digits)}`;
}

/** m:ss under an hour, h:mm:ss above it. */
export function fmtSeconds(frame, fps) {
  const total = Math.max(0, frame - 1) / (fps || 25);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = Math.floor(total % 60);
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

// ── adaptive ruler ──────────────────────────────────────────────────────────

/**
 * Pick the ruler's label mode and tick positions for the current window.
 *
 *   frames    >= 14 px per frame: labels are exact frame numbers (#00412)
 *   timecode  sub-second spacing:  HH:MM:SS:FF
 *   seconds   anything wider:      m:ss / h:mm:ss
 *
 * Ticks are pinned to absolute multiples of `step` from frame 1, so panning slides the clip under
 * them instead of the labels crawling. The loop bound is the number of ticks ON SCREEN, never the
 * clip length — a 100,000-frame clip fully zoomed out walks a few dozen iterations.
 */
export function rulerTicks(view, maxFrames, fps, trackW, labelPx = 92) {
  const v = normalizeView(view, maxFrames);
  const span = viewSpan(v);
  const ppf = trackW / span;
  const base = nominalFps(fps);
  if (!trackW || maxFrames < 2) return { mode: 'seconds', step: base, major: [], minor: [] };

  let mode;
  let step;
  if (ppf >= FRAME_CELL_DETAIL_PX) {
    mode = 'frames';
    const need = Math.ceil((FRAME_CELL_MIN_PX + 8) / ppf);
    step = NICE_FRAMES.find((n) => n >= need) ?? need;
  } else {
    const want = Math.max(2, Math.floor(trackW / labelPx));
    const secondsPerLabel = span / base / want;
    if (secondsPerLabel < 1) {
      mode = 'timecode';
      const raw = Math.max(1, Math.ceil(span / want));
      step = NICE_FRAMES.find((n) => n >= raw) ?? raw;
    } else {
      mode = 'seconds';
      const sec = NICE_SECONDS.find((s) => s >= secondsPerLabel) ?? NICE_SECONDS[NICE_SECONDS.length - 1];
      step = Math.max(1, sec * base);
    }
  }

  const minorPer = mode === 'frames' ? 1 : 4;
  const minorStep = Math.max(1, step / minorPer);
  const first = 1 + Math.floor((v.start - 1) / step) * step;
  const major = [];
  const minor = [];
  for (let f = first; f <= v.end + step; f += step) {
    if (f >= v.start && f <= v.end) major.push(f);
    if (minorPer > 1) {
      for (let k = 1; k < minorPer; k++) {
        const mf = Math.round(f + minorStep * k);
        if (mf >= v.start && mf <= v.end && mf <= maxFrames) minor.push(mf);
      }
    }
  }
  return { mode, step, major, minor };
}

export function labelFor(mode, frame, fps, maxFrames) {
  if (mode === 'frames') return fmtFrameLabel(frame, maxFrames);
  if (mode === 'timecode') return fmtTimecode(frame, fps);
  return fmtSeconds(frame, fps);
}

// ── thumbnail slots ─────────────────────────────────────────────────────────

/**
 * Where the filmstrip puts its stills: one slot per `slotW` pixels, each naming the frame it should
 * show. Once a frame is wider than a slot (frame-by-frame view) there is one slot per frame, so the
 * thumbnail sits exactly in its own cell. Returns at most ceil(trackW / slotW) + 1 slots however
 * long the clip is.
 */
export function thumbSlots(view, maxFrames, trackW, slotW) {
  const v = normalizeView(view, maxFrames);
  const ppf = trackW / viewSpan(v);
  const out = [];
  if (!trackW || slotW <= 0) return out;
  if (ppf >= slotW) {
    for (let f = v.start; f <= v.end; f++) out.push({ frame: f, x: xAtFrame(f, trackW, v) - ppf / 2, w: ppf });
    return out;
  }
  const n = Math.max(1, Math.ceil(trackW / slotW));
  const w = trackW / n;
  const seen = new Set();
  for (let i = 0; i < n; i++) {
    const centre = v.start + ((i + 0.5) * w / trackW) * viewSpan(v);
    const frame = clamp(Math.round(centre), 1, maxFrames);
    out.push({ frame, x: i * w, w, dup: seen.has(frame) });
    seen.add(frame);
  }
  return out;
}
