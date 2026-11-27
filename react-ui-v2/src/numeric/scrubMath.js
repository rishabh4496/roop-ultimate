// The arithmetic behind dragging a label to change a number. No React, no DOM: the hook in useScrub.js feeds it
// pixel deltas and the modifier keys held at that instant, and gets back the value.
//
//   plain drag   one `step` per SCRUB_PX_PER_STEP pixels - or, on a wide range, the pace that crosses the whole
//                range in SCRUB_FULL_RANGE_PX, whichever is faster. A 0..1 slider is therefore 400 px end to end,
//                a 0..4096 field is 600 px, and neither needs a second drag to get anywhere.
//   Shift        x10, for crossing a range quickly.
//   Alt          x0.1, snapped to MICRO_STEP (0.001), for the last digits. A control whose own step is a whole
//                number (a thread count) keeps its step: a fraction of a thread is not a value.
//   both         cancel to x1, so releasing one key mid-drag never jumps the value.
//
// The drag is INCREMENTAL: every pointer move adds its own delta, scaled by the modifiers held for that move, to an
// accumulator. Computing "total distance x current multiplier" instead would make the value lurch the instant Shift
// is pressed or released, because the whole distance travelled so far would be re-priced.

export const SCRUB_PX_PER_STEP = 4;
export const SCRUB_FULL_RANGE_PX = 600;
export const COARSE = 10;
export const FINE = 0.1;
export const MICRO_STEP = 0.001;
/** Pointer travel before a press becomes a drag; below it the press is a click. */
export const SCRUB_THRESHOLD_PX = 3;

export const clampNum = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Remove float drift (0.1 + 0.2) before the value reaches settings state. */
export const snapTo = (v, step, origin = 0) => Number((origin + Math.round((v - origin) / step) * step).toFixed(6));

/** Value change per pixel of travel at the plain pace. */
export function unitsPerPixel(min, max, step) {
  const range = Number.isFinite(max - min) ? max - min : 0;
  return Math.max(step / SCRUB_PX_PER_STEP, range / SCRUB_FULL_RANGE_PX);
}

/** x10 / x0.1 / x1 for the modifier keys held. */
export function scrubMultiplier({ shiftKey = false, altKey = false } = {}) {
  if (shiftKey && altKey) return 1;
  if (shiftKey) return COARSE;
  if (altKey) return FINE;
  return 1;
}

/** The grid the value is snapped to: the control's own step, or MICRO_STEP under Alt when that is finer. */
export function snapStep(step, { altKey = false, shiftKey = false } = {}) {
  if (altKey && !shiftKey && !(Number.isInteger(step) && step >= 1)) return Math.min(step, MICRO_STEP);
  return step;
}

export function beginScrub(value, { min = 0, max = 1, step = 0.01 } = {}) {
  const v = clampNum(Number(value) || 0, min, max);
  return { acc: v, value: v, min, max, step };
}

/** Apply `dx` pixels with the given modifiers; returns the next state (state.value is what to show). */
export function advanceScrub(state, dx, mods = {}) {
  const { min, max, step } = state;
  const acc = clampNum(state.acc + dx * unitsPerPixel(min, max, step) * scrubMultiplier(mods), min, max);
  const grid = snapStep(step, mods);
  // The grid starts at `min` so a control whose range does not begin on a multiple of its step lands on values the
  // native input would also accept.
  const value = clampNum(snapTo(acc, grid, Number.isFinite(min) ? min : 0), min, max);
  return { ...state, acc, value };
}

/** How many decimals a value needs to read back exactly, up to 3 (what Alt can produce). */
export function decimalsFor(value, base) {
  const s = String(Number(Number(value).toFixed(3)));
  const i = s.indexOf('.');
  return Math.max(base, i < 0 ? 0 : Math.min(3, s.length - i - 1));
}
