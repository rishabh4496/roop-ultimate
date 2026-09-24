import { useEffect, useMemo, useRef, useState } from 'react';
import { API } from '../../api';
import { frameSocket } from '../../transport/frameSocket';
import { UI_HZ } from '../../store/telemetryStore';
import { nextNeededFrame } from './playbackWindow';

// The whole playback surface of the Face Swap timeline: the play/loop/rate
// controls' state, the rolling frame buffer behind them, and the rAF clock that
// drives the playhead.
//
// WHAT CHANGED, AND WHY (2026-09-24)
//
//  * Frames are no longer React state. Each played frame used to be
//    `setBufferedSrc(blobUrl)` + `setFrame(n)` — two commits of the entire Face
//    Swap panel (~4,000 lines) per frame, 24-60 times a second, before the
//    stage could even START decoding the picture. Now a frame is handed to
//    `playbackSource` subscribers (a <FastCanvasPlayer> layered over the stage)
//    as JPEG bytes: decode and draw happen in a worker, and React sees nothing.
//  * The playhead (`setFrame`) is written at most UI_HZ (10) times a second
//    while playing, plus once, exactly, where playback stops. The timeline,
//    the frame box and every effect keyed on `frame` therefore run at 10 Hz
//    instead of at the clip's frame rate.
//  * The buffer holds raw JPEG BYTES (ArrayBuffers), not blob URLs. A blob URL
//    had to be fetched back and decoded at display time; the bytes go straight
//    to createImageBitmap, and there is nothing to revoke.
//  * Frames come over /ws/frames as a credit-metered PLAY stream when the
//    socket is up: one sequential decode on the server, refilled BEFORE the
//    buffer runs dry, instead of one HTTP request per 16-48 frames with a
//    round trip in between. /api/target/preview_seq stays as the fallback and
//    is used whenever the socket is not open (or drops mid-stream).
//
// The declaration ORDER matters and is preserved: the seek-detector effect must
// run before the main loop effect, which relies on it having already queued the
// current playhead (see the note about rewinding past the out point).

// Longest the playhead may go without being written to React while playing.
const UI_WRITE_MS = 1000 / UI_HZ;

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

  // Rolling window of upcoming frames as encoded JPEG bytes.
  const playBufRef = useRef(new Map());     // frame -> ArrayBuffer
  const playFetchRef = useRef(new Set());   // frames currently in flight
  const playBufIdxRef = useRef('');         // signature of the video the buffer was built for
  // Latest `targets`, readable from inside the playback rAF loop, which must not
  // re-bind on it: every In/Out drag and unrelated target-state update would
  // otherwise tear playback down and restart it.
  const targetsRef = useRef([]);
  // Seek plumbing. The playback loop owns the playhead and writes `frame` itself,
  // so it cannot simply watch `frame` for seeks -- it would see its own writes.
  // `playWroteRef` is the last value the LOOP wrote (set synchronously at write
  // time, so it is already current when the resulting commit runs its effects);
  // anything else arriving in `frame` came from outside -- a timeline drag, the
  // frame box, an arrow key, a step/jump button -- and is handed to the loop
  // through `playSeekRef`.
  const playWroteRef = useRef(null);
  const playSeekRef = useRef(null);
  // Where the loop actually IS, which runs ahead of `frame` by up to one UI
  // write. A rate/loop toggle re-binds the loop; it resumes from here rather
  // than from the (up to 100 ms stale) React playhead.
  const playCurRef = useRef(null);
  const selTargetRef = useRef(selTarget);
  selTargetRef.current = selTarget;
  // True while the playhead is held on a frame that hasn't arrived yet.
  const [playStalled, setPlayStalled] = useState(false);
  const stalledRef = useRef(false);
  // Which transport fed the last frame: 'socket' | 'http' | ''. Diagnostic.
  const viaRef = useRef('');
  // Counters for verification (tests/frame_transport_ab.py reads them through
  // window.__roopPlayback): an external seek throws the buffer away, so a
  // seek count that climbs during plain playback is a bug, not a statistic.
  const statsRef = useRef({ seeks: 0, streams: 0, chunks: 0, presented: 0, uiWrites: 0 });

  // ── The frame bus ───────────────────────────────────────────────────────
  // `subscribe(cb)` delivers {bytes, frame, width, height} per presented frame,
  // and {clear: true} when whatever is on screen should be dropped (playback
  // (re)starting, the target changing, the stage having caught up). A new
  // subscriber gets the frame currently up, so a view remounted mid-playback
  // (split view toggled) is not blank until the next frame.
  // Built once: `playbackSource` and `clearPlaybackFrame` are stable, so they
  // can sit in dependency arrays without re-running anything.
  const { playbackSource, emit, clearPlaybackFrame } = useMemo(() => {
    const listeners = new Set();
    let last = null;
    const send = (msg) => {
      last = msg.clear ? null : msg;
      for (const cb of listeners) {
        try { cb(msg); } catch { /* a consumer fault must not stop playback */ }
      }
    };
    return {
      playbackSource: {
        subscribe(cb) {
          listeners.add(cb);
          if (last) cb(last);
          return () => listeners.delete(cb);
        },
        get via() { return viaRef.current; },
        get stats() { return { ...statsRef.current, via: viaRef.current }; },
      },
      emit: (msg) => { if (!msg.clear) statsRef.current.presented += 1; send(msg); },
      /** Drop the playback picture (the still underneath is current again). */
      clearPlaybackFrame: () => { if (last) send({ clear: true }); },
    };
  }, []);

  const clearPlayBuffer = () => {
    playBufRef.current.clear();
    playFetchRef.current.clear();
  };
  useEffect(() => clearPlayBuffer, []);
  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    window.__roopPlayback = playbackSource;
    return () => { if (window.__roopPlayback === playbackSource) delete window.__roopPlayback; };
  }, [playbackSource]);
  useEffect(() => { targetsRef.current = targets; });
  // A different clip: whatever the overlay holds belongs to the old one.
  useEffect(() => { clearPlaybackFrame(); }, [selTarget, clearPlaybackFrame]);

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
      playCurRef.current = null;
      stalledRef.current = false;
      setPlayStalled(false);
      return undefined;
    }
    const idx = selTarget;
    const fps = targetsRef.current[idx]?.fps || 25;
    // The timeline is 1-based (a fresh target reports start_frame 0), so clamp
    // the loop/start point to 1 — otherwise looping jumps to "Frame 0" and
    // double-shows the first frame. Read live from the ref each tick so
    // dragging In/Out during playback takes effect without restarting the loop.
    // `|| maxFrames`, NOT `?? maxFrames`: the backend reports end_frame 0 for a
    // target whose trim has not been initialised yet, and ?? only substitutes
    // null/undefined — so 0 survived as a real Out point and every play press
    // stopped on the first tick.
    // `clipEnd` is where the clip REALLY ends when that turns out to be short
    // of what it reported (a stream that ends cleanly before its out point):
    // without it the prefetcher would ask for the missing tail forever.
    let clipEnd = Infinity;
    const bounds = () => {
      const t = targetsRef.current[idx];
      const s = Math.max(1, t?.start_frame || 1);
      return { start: s, end: Math.max(s, Math.min(clipEnd, t?.end_frame || maxFrames)) };
    };
    // Every chunk request currently on the wire, so teardown can cancel them.
    const chunkAborters = new Set();
    let { start, end } = bounds();
    const frameDur = 1000 / (fps * (playbackRate || 1));
    // Lead is measured in TIME, not frames: a 60fps clip drains the buffer twice
    // as fast as a 30fps one. Chunk size scales with it for the same reason.
    const AHEAD = Math.max(120, Math.round(fps * 2.5));
    const BEHIND = 8;    // frames retained behind before eviction
    const CHUNK = Math.max(16, Math.min(48, Math.round(fps / 2)));

    // Keep the buffered frames across speed/loop changes (they're still valid);
    // only discard and rebuild when the effect is (re)bound to a different video.
    const sig = `${idx}:${maxFrames}`;
    const sameVideo = playBufIdxRef.current === sig;
    if (!sameVideo) {
      clearPlayBuffer();
      playBufIdxRef.current = sig;
    }
    // Resume where the LOOP was if this is a re-bind (rate/loop toggle) and
    // nobody has moved the playhead since; otherwise start from `frame`.
    const resumeAt = sameVideo && playCurRef.current != null && frame === playWroteRef.current
      ? playCurRef.current : frame;
    // Press play with the playhead parked on (or past) the out point and a real
    // player rewinds and plays again — it does not sit there.
    let cur = resumeAt >= end ? start : Math.max(start, Math.min(resumeAt, end));
    // The seek-detector runs first on this same commit and will have queued the
    // CURRENT playhead as an external seek. That is the position we just
    // decided to move away from, so clear it.
    playSeekRef.current = null;
    // A fresh start shows the still underneath until its first frame lands,
    // rather than whatever the overlay held from the LAST playback.
    if (!sameVideo || playCurRef.current == null) clearPlaybackFrame();
    let rendered = -1;
    let cancelled = false;
    let rafId = null;
    let lastTs = null;
    let acc = 0;
    let lastUiWrite = -Infinity;

    // Frames are fetched by ONE sequential source at a time — an HTTP chunk or
    // a socket stream — strictly ascending: the server decodes the next in-order
    // frame cheaply but re-seeks (seconds per frame on long-GOP video) whenever
    // a request breaks sequence.
    let inFlight = 0;        // HTTP chunks on the wire (0 or 1)
    let stream = null;       // { handle, next, outstanding }
    // A stream that ended on an ERROR is not retried for this playback: the
    // server said no (not a video, target gone), and reopening every tick would
    // turn that into a request loop. HTTP takes over and reports the same way.
    let socketFailed = false;

    // See playbackWindow.js for the rule (and the loop-wrap bug it fixes).
    const nextNeeded = () => nextNeededFrame({
      cur, start, end, ahead: AHEAD, isLooping,
      has: (f) => playBufRef.current.has(f) || playFetchRef.current.has(f),
    });

    // Split the length-prefixed body: [4-byte BE length][JPEG] repeated.
    const splitChunk = (buf) => {
      const view = new DataView(buf);
      const out = [];
      let off = 0;
      while (off + 4 <= buf.byteLength) {
        const len = view.getUint32(off);
        off += 4;
        if (len <= 0 || off + len > buf.byteLength) break;
        out.push(buf.slice(off, off + len));
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
      statsRef.current.chunks += 1;
      // A chunk is up to 48 server-side video seeks. Stopping playback, seeking
      // away or switching target must stop that work, not merely ignore it.
      const ctrl = new AbortController();
      chunkAborters.add(ctrl);
      const done = () => {
        for (let i = 0; i < n; i++) playFetchRef.current.delete(fr + i);
        inFlight--;
        chunkAborters.delete(ctrl);
      };
      fetch(`${API}/api/target/preview_seq?index=${idx}&start=${fr}&count=${n}&width=960`,
        { signal: ctrl.signal })
        .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error('seq failed'))))
        .then((buf) => {
          if (cancelled) return;
          const parts = splitChunk(buf);
          parts.forEach((bytes, i) => {
            const f = fr + i;
            if (!playBufRef.current.has(f)) playBufRef.current.set(f, bytes);
          });
          // "A short body simply means the clip ended" (preview_seq's own
          // contract). Record it, or the tail is re-requested every tick.
          if (parts.length < n) clipEnd = Math.max(1, fr + parts.length - 1);
          viaRef.current = 'http';
        })
        .catch(() => { /* aborted, or a failed chunk: pump() re-requests it */ })
        .finally(done);
    };

    // ── The socket stream ────────────────────────────────────────────────
    // Forget a stream AND the frames it was credited but never delivered.
    // Those frames are marked in flight; left marked, nextNeeded() skips them
    // forever and playback stalls at the hole.
    const dropStream = () => {
      if (!stream) return;
      for (let f = stream.next; f < stream.next + stream.outstanding; f++) playFetchRef.current.delete(f);
      stream = null;
    };
    const closeStream = () => {
      if (!stream) return;
      stream.handle.close();
      dropStream();
    };
    const openStream = (from, credit) => {
      const s = { handle: null, next: from, outstanding: credit, end };
      s.handle = frameSocket.openStream(
        { index: idx, start: from, end, width: 960, credit },
        {
          onFrame: (m) => {
            if (cancelled || stream !== s) return;
            playFetchRef.current.delete(m.frame);
            if (!playBufRef.current.has(m.frame)) playBufRef.current.set(m.frame, m.bytes);
            s.next = m.frame + 1;
            s.outstanding = Math.max(0, s.outstanding - 1);
            viaRef.current = 'socket';
          },
          // Out point, end of clip, or an error: either way this stream is
          // spent. pump() opens another (or goes to HTTP) if frames are needed.
          onEnd: ({ nextFrame, error }) => {
            if (stream === s) dropStream();
            if (error) socketFailed = true;
            else if (nextFrame <= s.end) clipEnd = Math.max(1, nextFrame - 1);
          },
          onError: () => { if (stream === s) dropStream(); },
        },
      );
      if (!s.handle) return false;
      statsRef.current.streams += 1;
      for (let f = from; f < from + credit; f++) playFetchRef.current.add(f);
      stream = s;
      return true;
    };
    const pumpSocket = () => {
      const want = nextNeeded();
      if (want === null) return true;
      const room = (f) => Math.max(0, Math.min(end, cur + AHEAD) - f + 1);
      if (stream) {
        const tail = stream.next + stream.outstanding;
        if (want !== tail) {
          // Not a continuation (a loop wrap, a hole). Let the stream drain
          // first — single sequential source — then restart it at `want`.
          if (stream.outstanding > 0) return true;
          closeStream();
        } else {
          // Refill at half-empty, so the next credit is on the wire before
          // the server runs out: this is the round trip HTTP chunks paid in
          // full between every chunk.
          if (stream.outstanding > CHUNK / 2) return true;
          const n = Math.min(CHUNK - stream.outstanding, room(tail));
          if (n > 0) {
            for (let f = tail; f < tail + n; f++) playFetchRef.current.add(f);
            stream.outstanding += n;
            stream.handle.grant(n);
          }
          return true;
        }
      }
      const n = Math.min(CHUNK, room(want)) || 1;
      return openStream(want, n);
    };

    const pump = () => {
      // Evict frames outside the retained window (and outside the loop wrap set).
      const overflow = isLooping ? Math.max(0, cur + AHEAD - end) : 0;
      for (const k of [...playBufRef.current.keys()]) {
        const keep = (k >= cur - BEHIND && k <= cur + AHEAD) ||
                     (overflow > 0 && k >= start && k <= start + overflow);
        if (!keep) playBufRef.current.delete(k);
      }
      if (inFlight > 0) return;                         // an HTTP chunk owns the decoder
      if (!socketFailed && frameSocket.isOpen() && pumpSocket()) return;
      if (stream) return;                               // draining; wait for it
      const fr = nextNeeded();
      if (fr !== null) fetchChunk(fr);
    };

    const writePlayhead = (f) => {
      // playWroteRef must be set BEFORE setFrame so the seek-detection effect
      // that runs on this commit already sees it and doesn't treat our own
      // advance as an external seek.
      playWroteRef.current = f;
      statsRef.current.uiWrites += 1;
      setFrame(f);
    };

    const tick = (ts) => {
      if (cancelled) return;
      if (lastTs === null) lastTs = ts;
      acc += ts - lastTs;
      lastTs = ts;

      // Honour In/Out points moved during playback.
      ({ start, end } = bounds());

      // ── External seek ────────────────────────────────────────────────────
      // Adopt the new position and drop the look-ahead — it is all frames the
      // playhead has just left, and keeping it would stall the prefetcher, which
      // only ever requests the lowest unbuffered frame ahead of `cur`.
      if (playSeekRef.current !== null) {
        const want = Math.max(start, Math.min(playSeekRef.current, end));
        playSeekRef.current = null;
        if (want !== cur) {
          statsRef.current.seeks += 1;
          cur = want;
          rendered = -1;
          acc = 0;
          closeStream();
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
      // Present the current frame before honouring a non-loop stop, so playback
      // ends showing the out point rather than one frame short of it.
      if (cur !== rendered) {
        const bytes = playBufRef.current.get(cur);
        if (bytes) {
          emit({ bytes, frame: cur });
          rendered = cur;
          playCurRef.current = cur;
          // The playhead follows at <= UI_HZ; the picture does not wait for it.
          if (ts - lastUiWrite >= UI_WRITE_MS) {
            lastUiWrite = ts;
            writePlayhead(cur);
          }
        }
      }
      if (stop) {
        if (rendered > 0 && playWroteRef.current !== rendered) writePlayhead(rendered);
        setIsPlaying(false);
        return;
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (rafId) cancelAnimationFrame(rafId);
      for (const c of chunkAborters) c.abort();
      chunkAborters.clear();
      closeStream();
      // Land the playhead EXACTLY on the frame on screen: the 10 Hz writes can
      // leave React up to 100 ms behind the picture. Not when the target
      // itself changed — that frame belongs to the clip we just left.
      if (rendered > 0 && playWroteRef.current !== rendered && selTargetRef.current === idx) {
        writePlayhead(rendered);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `frame`/`targets` are read through refs on purpose: the loop writes `frame` itself, and re-binding on `targets` restarted playback on every In/Out drag.
  }, [isPlaying, isLooping, playbackRate, selTarget, maxFrames]);

  return {
    isPlaying, setIsPlaying,
    isLooping, setIsLooping,
    playbackRate, setPlaybackRate,
    playStalled,
    playbackSource,
    clearPlaybackFrame,
  };
}
