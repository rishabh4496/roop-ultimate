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

// Stage 9: the ladder runs 0.25× to 4×. A fresh press starts at 1× (FRESH_LEVEL), so L and J still feel the same
// at first; each further press goes faster, and Shift+J / Shift+L step down toward 0.25×.
export const SHUTTLE_SPEEDS = [0.25, 0.5, 1, 2, 4];
export const FRESH_LEVEL = SHUTTLE_SPEEDS.indexOf(1);

export const IDLE = Object.freeze({ dir: 0, level: 0 });

/**
 * Next shuttle state for a J / K / L key press. Unknown keys return the state unchanged.
 * `slow` is the Shift variant: one step slower in the direction already running, or a fresh 0.25× start.
 */
export function shuttleKey(state, key, { slow = false } = {}) {
  const k = String(key || '').toLowerCase();
  if (k === 'k') return IDLE;
  const dir = k === 'l' ? 1 : k === 'j' ? -1 : 0;
  if (!dir) return state;
  const top = SHUTTLE_SPEEDS.length - 1;
  if (slow) {
    if (state.dir === dir) return { dir, level: Math.max(state.level - 1, 0) };
    return { dir, level: 0 };
  }
  if (state.dir === dir) return { dir, level: Math.min(state.level + 1, top) };
  return { dir, level: FRESH_LEVEL };
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

/**
 * What the shuttle -> player sync should do this render, given the shuttle state at the LAST sync.
 *
 *   forward shuttle   set the player to the shuttle's rate (and start it if it is stopped)
 *   leaving a shuttle back to 1x, once, on the transition
 *   idle -> idle      nothing: the rate the user chose (0.25x ... 4x on the transport deck) stays
 *
 * The effect this replaces ran `setPlaybackRate(1)` on EVERY render where no shuttle was active, and it
 * depends on `isPlaying`, so picking 0.5x and pressing Play flipped isPlaying, re-ran the effect and
 * silently put the rate back to 1x: the speed selector could never take effect.
 */
export function shuttleSync(prev, state, isPlaying) {
  if (state.dir > 0) return { rate: shuttleRate(state), play: !isPlaying };
  if (state.dir === 0 && prev.dir !== 0) return { rate: 1, play: false };
  return null;
}

/**
 * Should a change of the player's play state cancel the shuttle? A forward shuttle ends when playback STOPS
 * (a true -> false transition: somebody paused it), a reverse shuttle ends when playback STARTS.
 *
 * The check this replaces was `!isPlaying && dir > 0`, a LEVEL test. Pressing L on a stopped player sets the
 * shuttle forward in the same commit that asks the player to start, and in that commit `isPlaying` is still
 * false, so the shuttle was cancelled the moment it began: from a stopped player L only ever played at 1x and
 * the 2x / 4x ladder could not be climbed (it worked only if you were already playing).
 */
export function shuttleEndsOnPlayChange(wasPlaying, isPlaying, state) {
  if (state.dir > 0) return !!wasPlaying && !isPlaying;
  if (state.dir < 0) return !!isPlaying;
  return false;
}

/** Frames to move for a step key: `,` `.` and the arrows move 1, Shift moves 10. */
export const stepSize = (shiftKey) => (shiftKey ? 10 : 1);
