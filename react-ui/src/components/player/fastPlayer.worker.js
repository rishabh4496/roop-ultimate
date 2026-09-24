// ── <FastCanvasPlayer>'s render thread ────────────────────────────────────
//
// Owns an OffscreenCanvas transferred from the page, and with it the whole
// frame path: decode (createImageBitmap), texture upload and draw. The page's
// only per-frame work is handing over an ArrayBuffer — transferred, not copied
// — so a long React commit on the main thread (the Face Swap panel is ~4,000
// lines) cannot delay a frame, and a frame cannot delay a pointer event.
//
// Protocol (page -> worker):
//   {type:'init', canvas, force2d}         once, with the canvas transferred
//   {type:'resize', w, h}                  backing store in device pixels
//   {type:'frame', layer, bytes, seq, meta, mime}   bytes transferred
//   {type:'bitmap', layer, bitmap, seq, meta}       an already-decoded frame
//   {type:'mode', mode, split}
//   {type:'clear', layer} | {type:'reset', layer} | {type:'stats'} | {type:'dispose'}
// (worker -> page):
//   {type:'ready', kind}  {type:'presented', layer, seq, meta, decodeMs}
//   {type:'stats', stats, kind}
import { GlRenderer } from './glRenderer';
import { FramePipeline, decodeBytes } from './framePipeline';

let renderer = null;
let pipeline = null;

// Dedicated workers get requestAnimationFrame in Chromium when they own an
// OffscreenCanvas, and it runs on the display's vsync. Anything older gets a
// ~60 Hz timer, which still keeps the draw off the page's thread.
const schedule = typeof self.requestAnimationFrame === 'function'
  ? (cb) => self.requestAnimationFrame(cb)
  : (cb) => setTimeout(cb, 16);

self.onmessage = (e) => {
  const m = e.data || {};
  switch (m.type) {
    case 'init': {
      renderer = new GlRenderer(m.canvas, { force2d: !!m.force2d });
      pipeline = new FramePipeline({
        renderer,
        decode: decodeBytes,
        schedule,
        onPresented: (info) => self.postMessage({ type: 'presented', ...info }),
      });
      self.postMessage({ type: 'ready', kind: renderer.kind });
      break;
    }
    case 'resize':
      if (renderer && renderer.resize(m.w, m.h)) pipeline.requestDraw();
      break;
    case 'frame':
      pipeline?.push(m.layer || 0, m.bytes, m.seq, m.meta, m.mime);
      break;
    case 'bitmap':
      if (pipeline) pipeline.pushBitmap(m.layer || 0, m.bitmap, m.seq, m.meta);
      else m.bitmap?.close?.();
      break;
    case 'mode':
      if (renderer) {
        renderer.setMode(m.mode);
        if (m.split !== undefined) renderer.setSplit(m.split);
        pipeline.requestDraw();
      }
      break;
    case 'clear':
      pipeline?.clearLayer(m.layer || 0);
      break;
    case 'reset':
      pipeline?.resetSequence(m.layer || 0);
      break;
    case 'stats':
      self.postMessage({ type: 'stats', stats: pipeline ? { ...pipeline.stats } : null, kind: renderer?.kind });
      break;
    case 'dispose':
      pipeline?.dispose();
      pipeline = null;
      renderer = null;
      self.close();
      break;
    default:
  }
};
