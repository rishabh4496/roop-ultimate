// J / K / L shuttle, the way every NLE does it.
//
//   L      play forward at 1x; pressed again while going forward: 2x, then 4x
//   J      the same, backwards
//   K      pause (and reset the speed ladder)
//   J while going forward (or L while going backward) reverses at 1x, it does not "slow down":
//          that is what an editor expects, and it is the cheapest rule to hold in your head.
//
// Forward shuttle rides the buffered player (the frame buffer is what makes 2x/4x possible);
// reverse has no buffer to ride, so the Timeline steps the playhead on a clock. This module is the
// state machine and the clock arithmetic, with no React in it.

export const SHUTTLE_SPEEDS = [1, 2, 4];

export const IDLE = Object.freeze({ dir: 0, level: 0 });

/** Next shuttle state for a J / K / L key press. Unknown keys return the state unchanged. */
export function shuttleKey(state, key) {
  const k = String(key || '').toLowerCase();
  if (k === 'k') return IDLE;
  const dir = k === 'l' ? 1 : k === 'j' ? -1 : 0;
  if (!dir) return state;
  if (state.dir === dir) return { dir, level: Math.min(state.level + 1, SHUTTLE_SPEEDS.length - 1) };
  return { dir, level: 0 };
}

export const shuttleRate = (state) => (state.dir ? SHUTTLE_SPEEDS[state.level] : 0);

/** "◀◀ 2×" / "▶ 1×" / "" for the transport read-out. */
export function shuttleLabel(state) {
  if (!state.dir) return '';
  return `${state.dir < 0 ? '◀' : '▶'} ${SHUTTLE_SPEEDS[state.level]}×`;
}

/**
 * Advance the reverse clock. `carry` is the fractional frame left over from the last tick, so a
 * 60 Hz clock at 25 fps does not round every step to zero or to one. Returns the new frame, the
 * new carry, and whether the start of the clip was reached.
 */
export function reverseTick({ frame, carry = 0 }, dtMs, rate, fps, floor = 1) {
  const exact = carry + (Math.max(0, dtMs) / 1000) * rate * (fps || 25);
  const whole = Math.floor(exact);
  const next = Math.max(floor, frame - whole);
  return { frame: next, carry: exact - whole, atStart: next <= floor };
}

/** Frames to move for a step key: `,` `.` and the arrows move 1, Shift moves 10. */
export const stepSize = (shiftKey) => (shiftKey ? 10 : 1);
