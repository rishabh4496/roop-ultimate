// Hover scrubbing for video thumbnails: the pointer's horizontal position across the card IS the playhead.
// Pure helpers; the card does the DOM work (one rAF-coalesced write per frame, no React state per move).

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** 0..1 across the card for a pointer x. A zero-width card is 0, not NaN. */
export const scrubRatio = (clientX, rect) => (rect && rect.width > 0 ? clamp((clientX - rect.left) / rect.width, 0, 1) : 0);

/**
 * The time to seek to. Stops one frame (~40 ms) short of the end: a seek to exactly `duration` lands on
 * the ended state, which blanks some decoders instead of showing the last picture.
 */
export const scrubTime = (ratio, duration) => {
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  return clamp(ratio, 0, 1) * Math.max(0, duration - 0.04);
};

/** m:ss.t for the little time readout. */
export const fmtClock = (t) => {
  const s = Math.max(0, Number.isFinite(t) ? t : 0);
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}.${Math.floor((s * 10) % 10)}`;
};
