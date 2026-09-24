// ── The binary frame socket (/ws/frames) ──────────────────────────────────
//
// One connection for the whole app, opened only while something wants it:
// a LIVE subscriber (the Processing tab's render view), an open PLAY stream
// (timeline playback), or a `hold()` (Face Swap keeps it warm so pressing Play
// does not first wait out a handshake).
//
// SAME CONTRACT AS useTelemetrySocket: the socket is an ACCELERATOR. Every
// consumer has an HTTP path it falls back to — /api/live_frame for the live
// view, /api/target/preview_seq for playback — and uses it whenever
// `isOpen()` is false or a stream is torn down mid-flight. A proxy that blocks
// the Upgrade therefore degrades to exactly the behaviour that shipped before.
//
// Frames arrive as ArrayBuffers (binaryType 'arraybuffer', never Blob: a Blob
// would need an extra async read before the bytes could be transferred to a
// decode worker). See frameProtocol.js for the header.
import {
  FLAG_ERROR, KIND_END, KIND_LIVE, KIND_PLAY, parseFrameMessage,
} from './frameProtocol';

const BACKOFF_MIN_MS = 500;
const BACKOFF_MAX_MS = 15000;
// The server does not heartbeat this socket (it is silent between frames on
// purpose), so the client pings. Silence past the timeout = a half-open TCP
// connection, which never fires onclose by itself.
const PING_MS = 20000;
const SILENCE_TIMEOUT_MS = 50000;
// Close an unused connection after this long, rather than instantly: tab
// switches in Pinokio unmount and remount consumers in quick succession.
const IDLE_CLOSE_MS = 8000;

const socketUrl = () => {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/frames`;
};

const visible = () => typeof document === 'undefined' || document.visibilityState !== 'hidden';

class FrameSocket {
  constructor() {
    this.ws = null;
    this.open = false;
    this.liveSubs = new Set();
    this.streams = new Map();       // id -> { onFrame, onEnd, onError }
    this.statusSubs = new Set();
    this.holds = 0;
    this.nextStreamId = 1;
    this.attempt = 0;
    this.retryTimer = null;
    this.idleTimer = null;
    this.pingTimer = null;
    this.lastRx = 0;
    this.liveOnServer = false;
    this.stats = { live: 0, play: 0, bytes: 0, connects: 0 };
    this._listening = false;
  }

  // ── lifecycle ──────────────────────────────────────────────────────────
  _wanted() { return this.liveSubs.size > 0 || this.streams.size > 0 || this.holds > 0; }

  _listen() {
    if (this._listening || typeof document === 'undefined') return;
    this._listening = true;
    const onVis = () => {
      if (visible()) {
        if (this.retryTimer) { clearTimeout(this.retryTimer); this.retryTimer = null; }
        this.attempt = 0;
        this._ensure();
      }
      this._syncLive();
    };
    document.addEventListener('visibilitychange', onVis);
    window.addEventListener('online', onVis);
  }

  _ensure() {
    this._listen();
    if (this.idleTimer && this._wanted()) { clearTimeout(this.idleTimer); this.idleTimer = null; }
    if (!this._wanted()) { this._scheduleIdleClose(); return; }
    if (this.ws || !visible()) return;
    let ws;
    try {
      ws = new WebSocket(socketUrl());
    } catch {
      this._scheduleRetry();
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => {
      this.open = true;
      this.attempt = 0;
      this.stats.connects += 1;
      this.lastRx = Date.now();
      this.liveOnServer = false;
      this._syncLive();
      this._startPing();
      this._emitStatus();
    };
    ws.onmessage = (e) => this._onMessage(e.data);
    ws.onerror = () => { /* onclose follows and owns the retry */ };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      const wasOpen = this.open;
      this.open = false;
      this.liveOnServer = false;
      this._stopPing();
      // Every in-flight stream is dead. Tell its owner so it can fall back to
      // HTTP for the frames it was promised — waiting for them would stall
      // playback forever.
      const dead = [...this.streams.values()];
      this.streams.clear();
      for (const s of dead) { try { s.onError?.('disconnected'); } catch { /* consumer fault */ } }
      if (wasOpen) this._emitStatus();
      this._scheduleRetry();
    };
  }

  _scheduleRetry() {
    if (this.retryTimer || !this._wanted() || !visible()) return;
    const base = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS * 2 ** this.attempt++);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this._ensure();
    }, base * (0.5 + Math.random() * 0.5));
  }

  _scheduleIdleClose() {
    if (this.idleTimer || !this.ws) return;
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      if (this._wanted() || !this.ws) return;
      const ws = this.ws;
      this.ws = null;
      try { ws.close(); } catch { /* already gone */ }
    }, IDLE_CLOSE_MS);
  }

  _startPing() {
    this._stopPing();
    this.pingTimer = setInterval(() => {
      if (!this.ws || !this.open) return;
      if (Date.now() - this.lastRx > SILENCE_TIMEOUT_MS) {
        try { this.ws.close(); } catch { /* ignore */ }
        return;
      }
      this._send('ping');
    }, PING_MS);
  }

  _stopPing() {
    if (this.pingTimer) { clearInterval(this.pingTimer); this.pingTimer = null; }
  }

  _send(payload) {
    if (!this.ws || !this.open) return false;
    try {
      this.ws.send(typeof payload === 'string' ? payload : JSON.stringify(payload));
      return true;
    } catch {
      return false;
    }
  }

  // LIVE is subscribed on the server only while a subscriber exists AND the
  // tab is visible. The server counts a subscription as a viewer (it keeps the
  // pipeline publishing at the watched cadence), so a hidden tab must let go.
  _syncLive() {
    const want = this.liveSubs.size > 0 && visible();
    if (!this.open || want === this.liveOnServer) return;
    if (this._send({ op: 'live', on: want })) this.liveOnServer = want;
  }

  _onMessage(data) {
    this.lastRx = Date.now();
    if (typeof data === 'string') return;          // pong
    const msg = parseFrameMessage(data);
    if (!msg) return;
    this.stats.bytes += data.byteLength;
    if (msg.kind === KIND_LIVE) {
      this.stats.live += 1;
      for (const cb of this.liveSubs) {
        try { cb(msg); } catch { /* one consumer must not starve the others */ }
      }
      return;
    }
    const s = this.streams.get(msg.stream);
    if (!s) return;                                 // a stream already closed
    if (msg.kind === KIND_PLAY) {
      this.stats.play += 1;
      s.onFrame?.(msg);
    } else if (msg.kind === KIND_END) {
      this.streams.delete(msg.stream);
      s.onEnd?.({ nextFrame: msg.frame, error: (msg.flags & FLAG_ERROR) !== 0 });
      if (!this._wanted()) this._scheduleIdleClose();
    }
  }

  _emitStatus() {
    for (const cb of this.statusSubs) { try { cb(this.open); } catch { /* ignore */ } }
  }

  // ── public API ─────────────────────────────────────────────────────────
  isOpen() { return this.open; }

  /** Subscribe to open/closed transitions. Returns an unsubscribe. */
  onStatus(cb) {
    this.statusSubs.add(cb);
    return () => this.statusSubs.delete(cb);
  }

  /** Keep the connection up without subscribing to anything. */
  hold() {
    this.holds += 1;
    this._ensure();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.holds -= 1;
      this._ensure();
    };
  }

  /**
   * Receive every LIVE frame: cb({frame: seq, width, height, bytes}).
   * `width`/`height` are the SOURCE frame's size (the caption), not the JPEG's.
   */
  subscribeLive(cb) {
    this.liveSubs.add(cb);
    this._ensure();
    this._syncLive();
    return () => {
      this.liveSubs.delete(cb);
      this._syncLive();
      this._ensure();
    };
  }

  /**
   * Open a sequential PLAY stream of target frames [start, end].
   *
   * Returns null when the socket is not open — the caller uses HTTP then.
   * Frames arrive through onFrame({frame, width, height, bytes}) only while
   * the stream holds credit: `grant(n)` adds n. onEnd({nextFrame, error})
   * fires once, at the out point / end of clip / error. onError fires if the
   * socket drops with the stream open. `close()` cancels it server-side.
   */
  openStream({ index, start, end, width = 960, quality = 82, credit = 0 }, handlers) {
    if (!this.open) return null;
    const id = this.nextStreamId++;
    this.streams.set(id, handlers || {});
    if (!this._send({ op: 'play', stream: id, index, start, end, width, quality, credit })) {
      this.streams.delete(id);
      return null;
    }
    return {
      id,
      grant: (n) => {
        if (n > 0 && this.streams.has(id)) this._send({ op: 'credit', stream: id, n });
      },
      close: () => {
        if (!this.streams.delete(id)) return;
        this._send({ op: 'stop', stream: id });
        if (!this._wanted()) this._scheduleIdleClose();
      },
    };
  }

  /**
   * A frame source for <FastCanvasPlayer source={...}>: the LIVE channel.
   * The shape (`subscribe(cb) -> unsubscribe`) is the one every source uses.
   */
  get liveSource() {
    if (!this._liveSource) {
      this._liveSource = { subscribe: (cb) => this.subscribeLive(cb) };
    }
    return this._liveSource;
  }
}

export const frameSocket = new FrameSocket();

if (typeof window !== 'undefined') {
  // For in-browser verification (tests/browser_driver.py reads it over CDP).
  window.__roopFrameSocket = frameSocket;
}
