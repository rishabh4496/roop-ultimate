import { useEffect, useRef, useState } from 'react';
import { API } from '../../api';

// The whole playback surface of the Face Swap timeline: the play/loop/rate
// controls' state, the rolling frame buffer behind them, and the rAF clock that
// drives the playhead.
//
// This is lifted verbatim out of FaceSwap. Every ref below was already private
// to it -- nothing outside read playBufRef, playFetchRef, playBufIdxRef,
// targetsRef, playWroteRef, playSeekRef or stalledRef, and clearPlayBuffer was
// only ever called from the effects here. What the rest of the panel actually
// uses is the eight values returned at the bottom.
//
// The declaration ORDER matters and is preserved: the seek-detector effect must
// run before the main loop effect, which relies on it having already queued the
// current playhead (see the note about rewinding past the out point).

/**
 * @param frame      current playhead (the loop both reads and writes this)
 * @param setFrame   playhead writer
 * @param selTarget  index of the target being played
 * @param maxFrames  total frames in that target
 * @param targets    the target list; mirrored into a ref so the loop can read
 *                   In/Out points live without re-binding on every drag
 */
export default function usePlaybackBuffer({ frame, setFrame, selTarget, maxFrames, targets }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(1); // 0.25 | 0.5 | 1 | 2 | 4

  // Buffered, video-style playback: a rolling window of upcoming frames decoded
  // into blob URLs so the playhead can run at real time instead of refetching
  // each frame from the server per tick.
  const playBufRef = useRef(new Map());     // frame -> object URL
  const playFetchRef = useRef(new Set());   // frames currently in flight
  const playBufIdxRef = useRef('');         // signature of the video the buffer was built for
  // Latest `targets`, readable from inside the playback rAF loop, which must not
  // re-bind on it: every In/Out drag and unrelated target-state update would
  // otherwise tear playback down and restart it.
  const targetsRef = useRef([]);
  // Seek plumbing. The playback loop owns the playhead and writes `frame` itself
  // every time it paints, so it cannot simply watch `frame` for seeks -- it would
  // see its own writes. `playWroteRef` is the last value the LOOP wrote (set
  // synchronously at write time, so it is already current when the resulting
  // commit runs its effects); anything else arriving in `frame` therefore came
  // from outside -- a timeline drag, the frame box, an arrow key, a step/jump
  // button -- and is handed to the loop through `playSeekRef`.
  const playWroteRef = useRef(null);
  const playSeekRef = useRef(null);
  const [bufferedSrc, setBufferedSrc] = useState(null);
  // True while the playhead is held on a frame that hasn't arrived yet.
  const [playStalled, setPlayStalled] = useState(false);
  const stalledRef = useRef(false);

  // ── Buffered, video-style playback ──────────────────────────────────────
  // Each frame is a fresh server JPEG (a video seek + decode), which can't keep
  // up with real time — a naive setInterval(setFrame) makes the browser refetch
  // per tick and the video stutters through "cut" frames. Instead we prefetch a
  // rolling window of upcoming frames into decoded blob URLs and drive the
  // playhead off a requestAnimationFrame clock, only advancing onto frames that
  // are already buffered. The result plays back continuously like a real player.
  const clearPlayBuffer = () => {
    for (const url of playBufRef.current.values()) URL.revokeObjectURL(url);
    playBufRef.current.clear();
    playFetchRef.current.clear();
  };
  useEffect(() => clearPlayBuffer, []);  // revoke any buffered blobs on unmount
  useEffect(() => { targetsRef.current = targets; });
  // Detect a playhead move that did NOT come from the playback loop and queue it
  // as a seek. Previously the loop just overwrote it on the next tick, so
  // dragging the timeline or typing a frame during playback looked like nothing
  // happened at all.
  useEffect(() => {
    if (!isPlaying) { playSeekRef.current = null; return; }
    if (frame === playWroteRef.current) return;   // our own write, echoed back
    playSeekRef.current = frame;
  }, [frame, isPlaying]);

  useEffect(() => {
    if (!isPlaying) {
      clearPlayBuffer();
      playBufIdxRef.current = '';
      setBufferedSrc(null);
      stalledRef.current = false;
      setPlayStalled(false);
      return;
    }
    const idx = selTarget;
    const fps = targetsRef.current[idx]?.fps || 25;
    // The timeline is 1-based (a fresh target reports start_frame 0), so clamp
    // the loop/start point to 1 — otherwise looping jumps to "Frame 0" and
    // double-shows the first frame. Read live from the ref each tick so
    // dragging In/Out during playback takes effect without restarting the loop.
    // `|| maxFrames`, NOT `?? maxFrames`: the backend reports end_frame 0 for a
    // target whose trim has not been initialised yet (ProcessEntry starts at 0),
    // and ?? only substitutes null/undefined — so 0 survived as a real Out point,
    // making `end` 1 and every play press stop on the first tick. The timeline's
    // own In/Out display already used ||, so the bar showed the full range while
    // the player thought the clip was one frame long.
    const bounds = () => {
      const t = targetsRef.current[idx];
      const s = Math.max(1, t?.start_frame || 1);
      return { start: s, end: Math.max(s, t?.end_frame || maxFrames) };
    };
    let { start, end } = bounds();
    const frameDur = 1000 / (fps * (playbackRate || 1));
    // Lead is measured in TIME, not frames: a 60fps clip drains the buffer twice
    // as fast as a 30fps one, so a fixed 120-frame look-ahead is two seconds of
    // cover on one and one second on the other. Chunk size scales with it for
    // the same reason — the per-request overhead has to be amortised over enough
    // frames to keep up with the drain rate (raising CHUNK is the documented
    // lever for slow playback; parallelising requests is not, because
    // out-of-order arrivals break the decoder's sequential fast path).
    const AHEAD = Math.max(120, Math.round(fps * 2.5));
    const BEHIND = 8;    // frames retained behind before eviction
    const CHUNK = Math.max(16, Math.min(48, Math.round(fps / 2)));

    // Keep the buffered frames across speed/loop changes (they're still valid);
    // only discard and rebuild when the effect is (re)bound to a different video.
    const sig = `${idx}:${maxFrames}`;
    if (playBufIdxRef.current !== sig) {
      clearPlayBuffer();
      setBufferedSrc(null);
      playBufIdxRef.current = sig;
    }
    // Press play with the playhead parked on (or past) the out point and a real
    // player rewinds and plays again — it does not sit there. Without this the
    // loop stopped on its very first tick, so after a clip had played through
    // once the button appeared dead: it flipped to Pause and straight back.
    let cur = frame >= end ? start : Math.max(start, Math.min(frame, end));
    // The seek-detector runs first on this same commit (it is declared above and
    // shares the isPlaying dep) and will have queued the CURRENT playhead as an
    // external seek. That is the position we just decided to move away from, so
    // clear it — otherwise the first tick seeks straight back to the out point
    // and stops, undoing the rewind above.
    playSeekRef.current = null;
    let rendered = -1;
    let cancelled = false;
    let rafId = null;
    let lastTs = null;
    let acc = 0;

    // Prefetch stays SINGLE-FLIGHT and strictly ascending: the server decodes the
    // next in-order frame cheaply but re-seeks (seconds per frame for long-GOP
    // video) whenever a request breaks sequence, so overlapping requests would
    // arrive out of order and force a seek per frame.
    //
    // But one frame per request also means one HTTP round trip, one FastAPI
    // dispatch and one capture-lock acquire PER FRAME, and since the playhead
    // refuses to advance onto an unbuffered frame, that round trip — not the
    // ~1 ms decode — becomes the frame rate. The buffer can never build a lead
    // either: it fills at exactly the speed it drains. That is why playback ran
    // far below the clip's real fps however fast the machine is.
    // /api/target/preview_seq returns CHUNK consecutive frames in one
    // length-prefixed response, so the fixed per-request cost is paid once per
    // CHUNK frames while the decode stays strictly sequential.
    let inFlight = 0;

    const nextNeeded = () => {
      for (let f = cur; f <= Math.min(end, cur + AHEAD); f++) {
        if (!playBufRef.current.has(f) && !playFetchRef.current.has(f)) return f;
      }
      // When looping and the look-ahead overruns the out point, warm the
      // wrap-around frames near `start` so the loop seam doesn't stall.
      if (isLooping) {
        const overflow = Math.max(0, cur + AHEAD - end);
        for (let f = start; f <= Math.min(end, start + overflow); f++) {
          if (!playBufRef.current.has(f) && !playFetchRef.current.has(f)) return f;
        }
      }
      return null;
    };

    // Split the length-prefixed body: [4-byte BE length][JPEG] repeated.
    const splitChunk = (buf) => {
      const view = new DataView(buf);
      const out = [];
      let off = 0;
      while (off + 4 <= buf.byteLength) {
        const len = view.getUint32(off);
        off += 4;
        if (len <= 0 || off + len > buf.byteLength) break;
        out.push(new Blob([buf.slice(off, off + len)], { type: 'image/jpeg' }));
        off += len;
      }
      return out;
    };

    const fetchChunk = (fr) => {
      // Never run past the out point in one request — beyond it the frames are
      // outside the retained window and would be evicted the moment they land.
      const n = Math.max(1, Math.min(CHUNK, end - fr + 1));
      for (let i = 0; i < n; i++) playFetchRef.current.add(fr + i);
      inFlight++;
      const done = () => {
        for (let i = 0; i < n; i++) playFetchRef.current.delete(fr + i);
        inFlight--;
      };
      fetch(`${API}/api/target/preview_seq?index=${idx}&start=${fr}&count=${n}&width=960`)
        .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject()))
        .then((buf) => {
          if (cancelled) return;
          splitChunk(buf).forEach((blob, i) => {
            const f = fr + i;
            if (!playBufRef.current.has(f)) playBufRef.current.set(f, URL.createObjectURL(blob));
          });
        })
        .catch(() => {})
        .finally(done);
    };

    const pump = () => {
      // Evict frames outside the retained window (and outside the loop wrap set).
      const overflow = isLooping ? Math.max(0, cur + AHEAD - end) : 0;
      for (const k of [...playBufRef.current.keys()]) {
        const keep = (k >= cur - BEHIND && k <= cur + AHEAD) ||
                     (overflow > 0 && k >= start && k <= start + overflow);
        if (!keep) {
          URL.revokeObjectURL(playBufRef.current.get(k));
          playBufRef.current.delete(k);
        }
      }
      // Top up the single in-flight sequential request.
      if (inFlight < 1) {
        const fr = nextNeeded();
        if (fr !== null) fetchChunk(fr);
      }
    };

    const tick = (ts) => {
      if (cancelled) return;
      if (lastTs === null) lastTs = ts;
      acc += ts - lastTs;
      lastTs = ts;

      // Honour In/Out points moved during playback.
      ({ start, end } = bounds());

      // ── External seek ────────────────────────────────────────────────────
      // Someone moved the playhead from outside the loop (see playSeekRef).
      // Adopt the new position and drop the look-ahead — it is all frames the
      // playhead has just left, and keeping it would stall the prefetcher, which
      // only ever requests the lowest unbuffered frame ahead of `cur`.
      if (playSeekRef.current !== null) {
        const want = Math.max(start, Math.min(playSeekRef.current, end));
        playSeekRef.current = null;
        if (want !== cur) {
          cur = want;
          rendered = -1;
          acc = 0;
          clearPlayBuffer();
        }
      }

      pump();
      // Advance whole frames for the elapsed time, but never onto a frame that
      // isn't buffered yet — hold there (buffering) so playback never skips.
      let guard = 0;
      let stop = false;
      let waiting = false;
      while (acc >= frameDur && guard++ < 240) {
        let next;
        if (cur >= end) {
          if (isLooping) { next = start; }
          else { stop = true; break; }
        } else {
          next = cur + 1;
        }
        if (!playBufRef.current.has(next)) {
          acc = Math.min(acc, frameDur);
          waiting = true;   // held on an unbuffered frame — that's buffering
          break;
        }
        acc -= frameDur;
        cur = next;
      }
      // Surface the hold. Without it a slow buffer is indistinguishable from a
      // dead button: the playhead just sits there with the Pause icon showing.
      if (waiting !== stalledRef.current) {
        stalledRef.current = waiting;
        setPlayStalled(waiting);
      }
      // Paint the current frame before honouring a non-loop stop, so playback
      // ends showing the out point rather than one frame short of it.
      if (cur !== rendered) {
        const url = playBufRef.current.get(cur);
        // playWroteRef must be set BEFORE setFrame so the seek-detection effect
        // that runs on this commit already sees it and doesn't treat our own
        // advance as an external seek.
        if (url) { setBufferedSrc(url); playWroteRef.current = cur; setFrame(cur); rendered = cur; }
      }
      if (stop) { setIsPlaying(false); return; }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => { cancelled = true; if (rafId) cancelAnimationFrame(rafId); };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `frame`/`targets` are read through refs on purpose: the loop writes `frame` itself every tick, and re-binding on `targets` restarted playback on every In/Out drag.
  }, [isPlaying, isLooping, playbackRate, selTarget, maxFrames]);

  return {
    isPlaying, setIsPlaying,
    isLooping, setIsLooping,
    playbackRate, setPlaybackRate,
    bufferedSrc, playStalled,
  };
}
