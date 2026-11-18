// The geometry of the studio scrubber's bar. No React, no DOM: `.render-check/stage5-scrubber-check.mjs` runs it under node.

/** The fine step the keyboard (Shift+Arrow) uses on a fractional control; an integer control keeps its own step. */
export const STEP_FINE = 0.01;

const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** Decimals a value shows at rest, from the control's step: 0.05 -> 2, 1 -> 0, 0.001 -> 3. */
export function scrubberDecimals(step) {
  const s = String(step);
  const i = s.indexOf('.');
  const fromStep = i < 0 ? 0 : s.length - i - 1;
  return Math.max(fromStep, step < 1 ? 2 : 0);
}

/**
 * Where the fill runs, as fractions of the bar (0..1).
 *
 * A range that straddles zero fills OUTWARD FROM ZERO (an erode / dilate radius of -12 fills the left of the zero line, +12 the
 * right of it); every other range fills from the left edge. `pct` is where the value sits (the bright leading edge) and
 * `pctDefault` where the default sits (the tick), or null with no default. A degenerate range (min >= max, or not finite)
 * draws nothing rather than NaN.
 */
export function fillExtent(value, min, max, defaultValue = null) {
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) return { from: 0, to: 0, pct: 0, pctDefault: null, bipolar: false };
  const frac = (v) => clamp01((v - min) / span);
  const pct = frac(Number(value));
  const bipolar = min < 0 && max > 0;
  const origin = bipolar ? frac(0) : 0;
  return {
    from: Math.min(origin, pct),
    to: Math.max(origin, pct),
    pct,
    pctDefault: defaultValue === null || defaultValue === undefined || !Number.isFinite(Number(defaultValue)) ? null : frac(Number(defaultValue)),
    bipolar,
  };
}
