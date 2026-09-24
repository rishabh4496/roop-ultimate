import React, {
  forwardRef, useEffect, useImperativeHandle, useRef,
} from 'react';
import { GlRenderer } from './glRenderer';
import { FramePipeline, decodeBytes } from './framePipeline';

// ── <FastCanvasPlayer /> — encoded frames straight to the GPU ─────────────
//
// A canvas that shows a stream of JPEG frames (the live render view, timeline
// playback) without React, <img>, base64 or blob URLs anywhere in the path:
//
//   ArrayBuffer ──transfer──▶ worker: createImageBitmap ─▶ back texture
//                                          rAF: swap to front, draw (WebGL)
//
// When the webview can transfer a canvas (`transferControlToOffscreen`, every
// current Chromium) all of that runs in fastPlayer.worker.js and the page's
// per-frame cost is one postMessage. Otherwise the same FramePipeline runs on
// the page with the canvas it has; createImageBitmap still decodes off-thread.
//
// FEEDING IT
//   * `source` — anything with `subscribe(cb) -> unsubscribe`, where cb gets
//     `{ bytes: ArrayBuffer, frame, width, height }` (frameSocket.liveSource,
//     usePlaybackBuffer's playbackSource), or `{ clear: true }` to go
//     transparent. The player COPIES the bytes before
//     transferring them, because a source may keep its buffer (the playback
//     buffer replays frames on loop and seek-back) and a transfer detaches it.
//   * or imperatively through the ref: push(bytes, meta), pushBitmap(bmp, meta).
//
// NOTHING HERE RENDERS PER FRAME. Props that change per frame do not exist;
// `onPresent` is called from the draw path for a caller that wants to know
// which frame is on screen — it is the caller's job to throttle anything it
// does with that (see useRafThrottle / the 10 Hz playhead in usePlaybackBuffer).

const MAX_EDGE = 2560;      // backing-store cap, as PreviewCanvas

const canTransfer = () => typeof window !== 'undefined'
  && typeof HTMLCanvasElement !== 'undefined'
  && typeof HTMLCanvasElement.prototype.transferControlToOffscreen === 'function'
  && typeof Worker !== 'undefined'
  && typeof OffscreenCanvas !== 'undefined';

// A canvas can be transferred ONCE. React StrictMode (on in this app) runs
// every effect mount -> cleanup -> mount against the SAME element, so a host
// torn down in the first cleanup could never be rebuilt. Hosts are therefore
// keyed on the element and released on a zero-delay timer that a re-mount
// cancels.
const hosts = new WeakMap();

function createWorkerHost(canvas, { force2d, onPresented, onReady }) {
  const offscreen = canvas.transferControlToOffscreen();
  const worker = new Worker(new URL('./fastPlayer.worker.js', import.meta.url), { type: 'module' });
  const statsWaiters = [];
  const host = {
    kind: 'worker',
    handlers: { onPresented, onReady },
    push(layer, bytes, seq, meta, mime) {
      worker.postMessage({ type: 'frame', layer, bytes, seq, meta, mime }, [bytes]);
    },
    pushBitmap(layer, bitmap, seq, meta) {
      // Bitmaps are transferable too; the worker's pipeline takes ownership.
      worker.postMessage({ type: 'bitmap', layer, bitmap, seq, meta }, [bitmap]);
    },
    resize(w, h) { worker.postMessage({ type: 'resize', w, h }); },
    mode(mode, split) { worker.postMessage({ type: 'mode', mode, split }); },
    clear(layer) { worker.postMessage({ type: 'clear', layer }); },
    reset(layer) { worker.postMessage({ type: 'reset', layer }); },
    stats() {
      return new Promise((resolve) => {
        statsWaiters.push(resolve);
        worker.postMessage({ type: 'stats' });
      });
    },
    dispose() { worker.postMessage({ type: 'dispose' }); },
  };
  worker.onmessage = (e) => {
    const m = e.data || {};
    if (m.type === 'presented') host.handlers.onPresented?.(m);
    else if (m.type === 'ready') { host.renderKind = m.kind; host.handlers.onReady?.(m.kind); }
    else if (m.type === 'stats') statsWaiters.splice(0).forEach((r) => r({ ...m.stats, kind: m.kind, host: 'worker' }));
  };
  worker.onerror = () => { host.failed = true; };
  worker.postMessage({ type: 'init', canvas: offscreen, force2d: !!force2d }, [offscreen]);
  return host;
}

function createMainHost(canvas, { force2d, onPresented, onReady }) {
  const renderer = new GlRenderer(canvas, { force2d });
  const host = { kind: 'main', handlers: { onPresented, onReady }, renderKind: renderer.kind };
  const pipeline = new FramePipeline({
    renderer,
    decode: decodeBytes,
    schedule: (cb) => requestAnimationFrame(cb),
    onPresented: (info) => host.handlers.onPresented?.(info),
  });
  Object.assign(host, {
    push: (layer, bytes, seq, meta, mime) => pipeline.push(layer, bytes, seq, meta, mime),
    pushBitmap: (layer, bitmap, seq, meta) => pipeline.pushBitmap(layer, bitmap, seq, meta),
    resize: (w, h) => { if (renderer.resize(w, h)) pipeline.requestDraw(); },
    mode: (mode, split) => {
      renderer.setMode(mode);
      if (split !== undefined) renderer.setSplit(split);
      pipeline.requestDraw();
    },
    clear: (layer) => pipeline.clearLayer(layer),
    reset: (layer) => pipeline.resetSequence(layer),
    stats: () => Promise.resolve({ ...pipeline.stats, kind: renderer.kind, host: 'main' }),
    dispose: () => pipeline.dispose(),
  });
  queueMicrotask(() => host.handlers.onReady?.(renderer.kind));
  return host;
}

function acquireHost(canvas, opts) {
  const existing = hosts.get(canvas);
  if (existing && !existing.disposed) {
    if (existing.releaseTimer) { clearTimeout(existing.releaseTimer); existing.releaseTimer = null; }
    existing.handlers = { onPresented: opts.onPresented, onReady: opts.onReady };
    return existing;
  }
  let host = null;
  if (opts.offscreen !== false && canTransfer()) {
    try { host = createWorkerHost(canvas, opts); } catch { host = null; }
  }
  if (!host) host = createMainHost(canvas, opts);
  hosts.set(canvas, host);
  return host;
}

function releaseHost(canvas, host) {
  host.handlers = {};
  host.releaseTimer = setTimeout(() => {
    host.releaseTimer = null;
    host.disposed = true;
    hosts.delete(canvas);
    try { host.dispose(); } catch { /* already gone */ }
  }, 0);
}

const FastCanvasPlayer = forwardRef(function FastCanvasPlayer({
  source = null,
  layer = 0,
  mode = 'single',
  split = 0.5,
  offscreen = true,
  force2d = false,
  onPresent,
  onReady,
  className = '',
  style,
  label = 'Video frame',
}, ref) {
  const canvasRef = useRef(null);
  const hostRef = useRef(null);
  const seqRef = useRef(0);
  const onPresentRef = useRef(onPresent);
  const onReadyRef = useRef(onReady);
  onPresentRef.current = onPresent;
  onReadyRef.current = onReady;

  // Host lifetime = the canvas element's lifetime.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const host = acquireHost(canvas, {
      offscreen,
      force2d,
      onPresented: (info) => onPresentRef.current?.(info),
      onReady: (kind) => onReadyRef.current?.(kind),
    });
    hostRef.current = host;

    // Backing store follows the element's CSS box x devicePixelRatio, capped.
    const fit = (cssW, cssH) => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      let w = cssW * dpr;
      let h = cssH * dpr;
      const s = Math.min(1, MAX_EDGE / Math.max(w, h, 1));
      w = Math.max(1, Math.round(w * s));
      h = Math.max(1, Math.round(h * s));
      host.resize(w, h);
    };
    let ro = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver((entries) => {
        const r = entries[0]?.contentRect;
        if (r && r.width && r.height) fit(r.width, r.height);
      });
      ro.observe(canvas);
    }
    const rect = canvas.getBoundingClientRect();
    if (rect.width && rect.height) fit(rect.width, rect.height);

    return () => {
      ro?.disconnect();
      if (hostRef.current === host) hostRef.current = null;
      releaseHost(canvas, host);
    };
    // offscreen/force2d are decided once per element; changing them would need
    // a new canvas (a transferred canvas cannot come back).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { hostRef.current?.mode(mode, split); }, [mode, split]);

  const pushBytes = (bytes, meta = null, lyr = layer) => {
    const host = hostRef.current;
    if (!host || !bytes || !bytes.byteLength) return false;
    // Copy: the caller may still need its buffer (see the header note).
    const copy = bytes.slice(0);
    host.push(lyr, copy, ++seqRef.current, meta, 'image/jpeg');
    return true;
  };

  // Subscribe to the source, if any.
  useEffect(() => {
    if (!source || typeof source.subscribe !== 'function') return undefined;
    return source.subscribe((f) => {
      // `{clear: true}`: drop the picture (transparent again), e.g. playback
      // handing the stage back to the still underneath.
      if (f?.clear) { hostRef.current?.clear(layer); return; }
      if (!f || !f.bytes) return;
      pushBytes(f.bytes, { frame: f.frame, width: f.width, height: f.height });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, layer]);

  useImperativeHandle(ref, () => ({
    push: (bytes, meta, lyr) => pushBytes(bytes, meta, lyr),
    pushBitmap: (bitmap, meta, lyr = layer) => {
      const host = hostRef.current;
      if (!host) { bitmap?.close?.(); return false; }
      host.pushBitmap(lyr, bitmap, ++seqRef.current, meta);
      return true;
    },
    clear: (lyr = layer) => hostRef.current?.clear(lyr),
    setMode: (m, s) => hostRef.current?.mode(m, s),
    stats: () => hostRef.current?.stats() ?? Promise.resolve(null),
    get renderKind() { return hostRef.current?.renderKind || ''; },
  }));

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={style}
      role="img"
      aria-label={label}
      data-fast-canvas-player=""
    />
  );
});

export default FastCanvasPlayer;
