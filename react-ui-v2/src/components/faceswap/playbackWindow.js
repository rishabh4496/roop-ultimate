// ── Which frame the playback prefetcher should ask for next ───────────────
//
// Pure, so .render-check/player-check.mjs can test it without a browser.
//
//   cur        the playhead
//   start/end  the In/Out points (inclusive)
//   ahead      look-ahead, in frames
//   isLooping  whether playback wraps from `end` back to `start`
//   has(f)     true when frame f is buffered OR already on its way
//
// Returns the lowest frame in [cur, cur + ahead] (clamped to `end`) that is
// missing, else — only when looping AND the look-ahead runs past `end` — the
// lowest missing frame of the wrap-around stretch [start, start + overflow],
// else null.
//
// The wrap scan used to run at overflow 0 as well, where its range is
// [start, start] — and `start` has long been played and evicted. So every time
// the look-ahead was full it asked for the clip's FIRST frame again, which
// seeks the server's one decoder back to the start for a chunk that is evicted
// on the next tick. On b1.mp4 that was 164 socket streams and 1,715 frames
// delivered for 290 played (2026-09-24); after the fix, 1 stream and 408
// frames (290 played + the 120-frame look-ahead). The HTTP chunk path had the
// same loop but its buffer seldom filled, so it rarely fired there.
export function nextNeededFrame({ cur, start, end, ahead, isLooping, has }) {
  for (let f = cur; f <= Math.min(end, cur + ahead); f++) {
    if (!has(f)) return f;
  }
  const overflow = isLooping ? Math.max(0, cur + ahead - end) : 0;
  if (overflow > 0) {
    for (let f = start; f <= Math.min(end, start + overflow); f++) {
      if (!has(f)) return f;
    }
  }
  return null;
}
