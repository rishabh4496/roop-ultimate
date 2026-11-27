// The pure half of the on-canvas face context menu (Stage 25): where the menu goes, what a face's badge says, the
// geometry of a crop, and the maths behind "Auto-Match Skin Tone". No React, no DOM - the canvas reads live in
// faceSample.js and hand this module plain pixel arrays.

import { trackLabel } from './identityLock';

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const finite = (v) => Number.isFinite(Number(v));

// ── placement ────────────────────────────────────────────────────────────────

export const MENU_MARGIN = 8;

/**
 * Where a menu of size w x h opens for a click at (x, y), kept inside a vw x vh viewport. All in ONE coordinate space
 * (the caller passes visual pixels - what getBoundingClientRect and window.innerWidth both speak - and converts the
 * result to layout pixels, so the app-level CSS zoom cannot push the menu off-screen).
 *
 * It opens at the pointer, down and to the right. Where that would cross the right / bottom edge it opens to the left
 * of / above the pointer instead (so the pointer never ends up under the menu's corner), and it is finally clamped so
 * no edge crosses the margin. A menu taller than the viewport is pinned to the top and given `maxHeight`.
 */
export function clampMenuPosition({ x, y, w, h, vw, vh, margin = MENU_MARGIN }) {
  const room = (total) => Math.max(0, total - 2 * margin);
  const flipX = x + w > vw - margin;
  const flipY = y + h > vh - margin;
  const rawLeft = flipX ? x - w : x;
  const rawTop = flipY ? y - h : y;
  const left = clamp(rawLeft, margin, Math.max(margin, vw - w - margin));
  const top = clamp(rawTop, margin, Math.max(margin, vh - h - margin));
  return { left, top, flipX, flipY, maxHeight: h > room(vh) ? room(vh) : null, maxWidth: w > room(vw) ? room(vw) : null };
}

// ── the badge ────────────────────────────────────────────────────────────────

/**
 * What a hovered face says about itself: which track it is and which identity it is wearing.
 * `role` is faceRoles.faceRole(...).role ('target' | 'swapped' | 'bystander'); `personId` the captured person (or null).
 * There is no match score here on purpose: the preview does not return one per face, and a made-up number is worse
 * than none.
 */
export function faceBadgeText({ faceIndex, personId, sourceName, role, excluded = false }) {
  const who = personId !== null && personId !== undefined && personId !== '' ? trackLabel(personId) : `Face ${faceIndex + 1}`;
  if (excluded) return `${who} → excluded (never swapped)`;
  if (role === 'bystander' || !role) return `${who} → not swapped`;
  return `${who} → ${sourceName || 'source not in gallery'}`;
}

// ── crops ────────────────────────────────────────────────────────────────────

/**
 * The rectangle to cut out of a frame for a face, in the frame's own pixels: the box grown by `margin` of its size
 * on every side (a detector needs the hair, chin and ears around a face to find it again), kept inside the frame.
 * `bbox` is in imgDim space; `natural` is the real size of the image being cropped (they differ whenever the decoded
 * frame is a proxy).
 */
export function cropRectFor(bbox, imgDim, natural, margin = 0.6) {
  if (!bbox || bbox.length < 4 || !imgDim?.w || !imgDim?.h || !natural?.w || !natural?.h) return null;
  const sx = natural.w / imgDim.w;
  const sy = natural.h / imgDim.h;
  const [x0, y0, x1, y1] = bbox;
  const bw = (x1 - x0) * sx;
  const bh = (y1 - y0) * sy;
  if (!(bw > 4 && bh > 4)) return null;
  const left = clamp(Math.floor(x0 * sx - bw * margin), 0, natural.w - 1);
  const top = clamp(Math.floor(y0 * sy - bh * margin), 0, natural.h - 1);
  const right = clamp(Math.ceil(x1 * sx + bw * margin), left + 1, natural.w);
  const bottom = clamp(Math.ceil(y1 * sy + bh * margin), top + 1, natural.h);
  return { x: left, y: top, w: right - left, h: bottom - top };
}

// ── skin tone ────────────────────────────────────────────────────────────────

// The backend's own warmth gain (procmgr_color._apply_skin_photometric_controls): in OpenCV's 8-bit LAB, a warmth of
// w (-1..+1) adds 1.5 * w to a* and 4.0 * w to b* on skin. The calibration below solves for the w that undoes a
// measured shift, so the number it writes is in the units the renderer already uses.
export const WARMTH_GAIN = Object.freeze({ a: 1.5, b: 4.0 });
export const MIN_SKIN_SAMPLES = 80;
// Below this share of the measured shift lying along the warmth direction, a warmth change would mostly move colour
// the OTHER way (a pure green / magenta error is almost orthogonal to it): the answer is declined, not applied.
export const MIN_EXPLAINED = 0.3;
// A chroma difference under this (8-bit LAB units, both axes) is below what anyone can see on skin.
export const NEGLIGIBLE_SHIFT = 0.75;

/**
 * Regions of a face worth sampling for skin: the forehead and both cheeks - flat, usually unshadowed, and free of the
 * eyes, brows, lips, nostrils and hairline that would skew a median. With the five keypoints (eyeL, eyeR, nose,
 * mouthL, mouthR) the regions follow the face's pose and scale; without, they are fractions of the box.
 * Rectangles are {x, y, w, h} in imgDim space.
 */
export function faceRois(bbox, kps = null) {
  if (!bbox || bbox.length < 4) return [];
  const [x0, y0, x1, y1] = bbox;
  const bw = x1 - x0;
  const bh = y1 - y0;
  if (!(bw > 4 && bh > 4)) return [];
  const rect = (cx, cy, w, h) => ({ x: cx - w / 2, y: cy - h / 2, w, h });
  const pts = Array.isArray(kps) && kps.length >= 5 && kps.slice(0, 5).every((p) => Array.isArray(p) && finite(p[0]) && finite(p[1]));
  if (pts) {
    const [eL, eR, , mL, mR] = kps;
    const iod = Math.hypot(eR[0] - eL[0], eR[1] - eL[1]);
    if (iod > 6) {
      // The eyes' line is not always level (a tilted head): the forehead sits "above" along the face's own up axis.
      const ex = (eL[0] + eR[0]) / 2;
      const ey = (eL[1] + eR[1]) / 2;
      // Unit vector from the eyes towards the forehead: the eye line (eL -> eR) turned a quarter-turn "up" (y is down).
      const ux = (eR[1] - eL[1]) / iod;
      const uy = -(eR[0] - eL[0]) / iod;
      const fx = ex + ux * iod * 0.62;
      const fy = ey + uy * iod * 0.62;
      return [
        rect(fx, fy, iod * 0.5, iod * 0.22),
        rect((eL[0] + mL[0]) / 2 - iod * 0.04, (eL[1] + mL[1]) / 2 + iod * 0.02, iod * 0.3, iod * 0.3),
        rect((eR[0] + mR[0]) / 2 + iod * 0.04, (eR[1] + mR[1]) / 2 + iod * 0.02, iod * 0.3, iod * 0.3),
      ].filter((r) => r.w >= 2 && r.h >= 2);
    }
  }
  return [
    rect(x0 + bw * 0.5, y0 + bh * 0.16, bw * 0.34, bh * 0.1),
    rect(x0 + bw * 0.24, y0 + bh * 0.6, bw * 0.2, bh * 0.16),
    rect(x0 + bw * 0.76, y0 + bh * 0.6, bw * 0.2, bh * 0.16),
  ];
}

const srgbToLinear = (c) => { const v = c / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
const labF = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);

/** sRGB (0-255) to OpenCV's 8-bit LAB: L scaled to 0-255, a* and b* offset by 128. */
export function rgbToLab8(r, g, b) {
  const R = srgbToLinear(r);
  const G = srgbToLinear(g);
  const B = srgbToLinear(b);
  const X = (0.412453 * R + 0.35758 * G + 0.180423 * B) / 0.950456;
  const Y = 0.212671 * R + 0.71516 * G + 0.072169 * B;
  const Z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / 1.088754;
  const fy = labF(Y);
  return [(116 * fy - 16) * 255 / 100, 500 * (labF(X) - fy) + 128, 200 * (fy - labF(Z)) + 128];
}

/** The backend's skin evidence (procmgr_color._skin_region_mask): YCrCb ranges, and not near-black. */
export function isSkinPixel(r, g, b) {
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  const cr = (r - y) * 0.713 + 128;
  const cb = (b - y) * 0.564 + 128;
  return cr >= 118 && cr <= 195 && cb >= 72 && cb <= 150 && Math.max(r, g, b) >= 25;
}

/**
 * Median LAB a* / b* (and L) of the skin pixels in a list of ImageData-like { data (RGBA), width, height } patches.
 * A histogram median, so it is O(pixels) and exact for 8-bit LAB. Returns { a, b, L, n }; n is the skin pixel count.
 */
export function skinStats(patches) {
  const hist = { a: new Uint32Array(256), b: new Uint32Array(256), L: new Uint32Array(256) };
  let n = 0;
  for (const patch of patches || []) {
    const d = patch?.data;
    if (!d) continue;
    for (let i = 0; i + 3 < d.length; i += 4) {
      if (d[i + 3] < 128 || !isSkinPixel(d[i], d[i + 1], d[i + 2])) continue;
      const [L, a, b] = rgbToLab8(d[i], d[i + 1], d[i + 2]);
      hist.L[clamp(Math.round(L), 0, 255)] += 1;
      hist.a[clamp(Math.round(a), 0, 255)] += 1;
      hist.b[clamp(Math.round(b), 0, 255)] += 1;
      n += 1;
    }
  }
  if (!n) return { a: 0, b: 0, L: 0, n: 0 };
  const median = (h) => {
    let acc = 0;
    for (let i = 0; i < 256; i++) { acc += h[i]; if (acc * 2 >= n) return i; }
    return 255;
  };
  return { a: median(hist.a), b: median(hist.b), L: median(hist.L), n };
}

/**
 * The warmth change that moves the swapped skin's chroma onto the plate's. `plate` is the target frame's skin, `swap`
 * the swapped result's (both skinStats output, same regions of the same face). The measured shift (da, db) is fitted
 * with the renderer's warmth direction (1.5, 4.0) by least squares. Two things are reported beside the answer:
 *   residual   what a warmth slider cannot move (a green / magenta tint - the colour-transfer mode's job)
 *   explained  the share of the shift's energy the fit accounts for, 0..1
 * and `verdict` says what to do with it: 'apply', 'matched' (already within NEGLIGIBLE_SHIFT) or 'tint' (the error is
 * mostly off the warm axis, so applying the least-squares answer would trade a small error for a different one).
 * Returns null when either side has too few skin pixels to trust.
 */
export function warmthDeltaFor(plate, swap) {
  if (!plate || !swap || plate.n < MIN_SKIN_SAMPLES || swap.n < MIN_SKIN_SAMPLES) return null;
  const da = plate.a - swap.a;
  const db = plate.b - swap.b;
  const { a: ga, b: gb } = WARMTH_GAIN;
  const w = (ga * da + gb * db) / (ga * ga + gb * gb);
  const residual = Math.hypot(da - ga * w, db - gb * w);
  const size = Math.hypot(da, db);
  const explained = size > 1e-9 ? clamp(1 - (residual * residual) / (size * size), 0, 1) : 1;
  let verdict = 'apply';
  if (Math.abs(da) < NEGLIGIBLE_SHIFT && Math.abs(db) < NEGLIGIBLE_SHIFT) verdict = 'matched';
  else if (explained < MIN_EXPLAINED) verdict = 'tint';
  return {
    da, db, verdict,
    warmthDelta: verdict === 'apply' ? Math.round(clamp(w * 100, -200, 200) * 10) / 10 : 0,
    residual: Math.round(residual * 100) / 100,
    explained: Math.round(explained * 1000) / 1000,
    plate, swap,
  };
}

/** The warmth to write: the current slider plus the measured change, kept inside the slider's range. */
export function nextWarmth(current, delta) {
  return Math.round(clamp((finite(current) ? Number(current) : 0) + delta, -100, 100));
}
