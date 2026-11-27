// ── Frame pipeline: bytes in, one draw per display refresh out ────────────
//
// Owns the path from an encoded frame to the screen, for one <FastCanvasPlayer>.
// It runs in two places with the same code:
//   * inside fastPlayer.worker.js, drawing to an OffscreenCanvas (the normal
//     case: the main thread only ever transfers an ArrayBuffer), and
//   * on the main thread, when the webview cannot transfer a canvas.
//
// THE RULES
//  1. Decode is off the draw path. `createImageBitmap` runs asynchronously and
//     the draw only ever sees a finished bitmap.
//  2. At most ONE decode in flight per layer, and a frame that arrives while
//     one is running REPLACES any frame already waiting. Decodes finish in
//     order, and a burst (a socket that delivered three frames in one task)
//     costs one decode, not three — the two in the middle were never going to
//     be seen.
//  3. Draw on the display's clock. A decoded frame waits in the layer's
//     `pending` slot (the back buffer) until the next animation frame; if a
//     newer one lands first, the older is closed unseen. Nothing draws when
//     nothing changed — there is no perpetual loop burning a core while the
//     stage sits on a still.
//  4. Every ImageBitmap is closed exactly once: on upload (the renderer takes
//     ownership), or when superseded while pending, or on dispose.

export class FramePipeline {
  /**
   * @param {object} o
   * @param {import('./glRenderer').GlRenderer} o.renderer
   * @param {(bytes: ArrayBuffer, mime: string) => Promise<ImageBitmap>} o.decode
   * @param {(cb: () => void) => void} o.schedule    requestAnimationFrame-alike
   * @param {(info: object) => void} [o.onPresented]
   */
  constructor({ renderer, decode, schedule, onPresented }) {
    this.renderer = renderer;
    this.decode = decode;
    this.schedule = schedule;
    this.onPresented = onPresented || (() => {});
    // `gen` is bumped by clearLayer: a decode that started before a clear must
    // not land after it and put the cleared picture back.
    this.layers = [0, 1].map(() => ({ busy: false, queued: null, pending: null, shownSeq: -1, gen: 0 }));
    this.drawQueued = false;
    this.dirty = false;
    this.disposed = false;
    this.stats = { received: 0, decoded: 0, presented: 0, dropped: 0, decodeMs: 0, errors: 0 };
  }

  /** Accept an encoded frame for a layer. `seq` must increase per layer. */
  push(layer, bytes, seq, meta, mime = 'image/jpeg') {
    if (this.disposed) return;
    this.stats.received += 1;
    const L = this.layers[layer];
    const item = { bytes, seq, meta, mime, gen: L.gen };
    if (L.busy) {
      if (L.queued) this.stats.dropped += 1;
      L.queued = item;
      return;
    }
    this._decode(layer, item);
  }

  /** Accept an already-decoded picture (the pipeline takes ownership). */
  pushBitmap(layer, bitmap, seq, meta) {
    if (this.disposed) { bitmap?.close?.(); return; }
    this.stats.received += 1;
    this._setPending(layer, { bitmap, seq, meta, decodeMs: 0, gen: this.layers[layer].gen });
  }

  _decode(layer, item) {
    const L = this.layers[layer];
    L.busy = true;
    const t0 = performance.now();
    this.decode(item.bytes, item.mime).then((bitmap) => {
      const ms = performance.now() - t0;
      if (this.disposed) { bitmap.close?.(); return; }
      this.stats.decoded += 1;
      this.stats.decodeMs = this.stats.decodeMs * 0.9 + ms * 0.1;
      this._setPending(layer, { bitmap, seq: item.seq, meta: item.meta, decodeMs: ms, gen: item.gen });
    }).catch(() => {
      this.stats.errors += 1;
    }).finally(() => {
      L.busy = false;
      if (L.queued && !this.disposed) {
        const next = L.queued;
        L.queued = null;
        this._decode(layer, next);
      }
    });
  }

  _setPending(layer, frame) {
    const L = this.layers[layer];
    if (frame.gen !== L.gen || frame.seq <= L.shownSeq) {
      frame.bitmap?.close?.();
      this.stats.dropped += 1;
      return;
    }
    if (L.pending) {
      L.pending.bitmap?.close?.();
      this.stats.dropped += 1;
    }
    L.pending = frame;
    this.requestDraw();
  }

  /** Redraw on the next animation frame (resize, mode change, new frame). */
  requestDraw() {
    this.dirty = true;
    if (this.drawQueued || this.disposed) return;
    this.drawQueued = true;
    this.schedule(() => this._draw());
  }

  _draw() {
    this.drawQueued = false;
    if (this.disposed || !this.dirty) return;
    this.dirty = false;
    const shown = [];
    for (let i = 0; i < this.layers.length; i++) {
      const L = this.layers[i];
      const f = L.pending;
      if (!f) continue;
      L.pending = null;
      // Ownership passes to the renderer, which closes the bitmap once the
      // pixels are on the GPU.
      if (this.renderer.setLayer(i, f.bitmap, { owned: true })) {
        L.shownSeq = f.seq;
        shown.push({ layer: i, seq: f.seq, meta: f.meta, decodeMs: f.decodeMs });
      }
    }
    this.renderer.draw();
    for (const s of shown) {
      this.stats.presented += 1;
      this.onPresented(s);
    }
  }

  clearLayer(layer) {
    const L = this.layers[layer];
    L.pending?.bitmap?.close?.();
    L.pending = null;
    L.queued = null;
    L.gen += 1;
    this.renderer.clearLayer(layer);
    this.requestDraw();
  }

  /** Forget ordering, so a new source (a seek, a new stream) is not refused as "old". */
  resetSequence(layer) {
    this.layers[layer].shownSeq = -1;
  }

  dispose() {
    this.disposed = true;
    for (const L of this.layers) {
      L.pending?.bitmap?.close?.();
      L.pending = null;
      L.queued = null;
    }
    this.renderer.destroy();
  }
}

/** createImageBitmap from encoded bytes — usable in a worker or on the main thread. */
export function decodeBytes(bytes, mime = 'image/jpeg') {
  return createImageBitmap(new Blob([bytes], { type: mime }));
}
