// ── Main-thread facade over the decoder worker ────────────────────────────
//
// One shared worker for the whole app (frame decodes are already serialised by
// the backend's single video decoder, so a pool would only add copies), with a
// two-step fallback for environments that cannot run it:
//
//   1. worker + createImageBitmap   — decode off-thread, transferable result
//   2. createImageBitmap on main    — still no <img> element, still abortable
//   3. <img>.decode()               — last resort; drawImage accepts an <img>
//
// Every path honours an AbortSignal, because the scrubber's whole design is
// that a request the user has already moved past must stop costing anything —
// including the decode, which is the expensive half on 4K footage.

let worker = null;
let workerBroken = false;
let seq = 0;
const pending = new Map();   // id -> { resolve, reject }

function getWorker() {
  if (workerBroken) return null;
  if (worker) return worker;
  if (typeof Worker === 'undefined' || typeof createImageBitmap === 'undefined') {
    workerBroken = true;
    return null;
  }
  try {
    worker = new Worker(new URL('./decoder.worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (e) => {
      const { id, ok, bitmap, aborted, message } = e.data || {};
      const entry = pending.get(id);
      if (!entry) { if (bitmap) bitmap.close(); return; }   // aborted after send
      pending.delete(id);
      if (ok) entry.resolve(bitmap);
      else entry.reject(aborted
        ? new DOMException('aborted', 'AbortError')
        : new Error(message || 'decode failed'));
    };
    worker.onerror = () => {
      // A worker that fails to boot (a webview with no module-worker support)
      // must not take the preview down with it: fail every pending decode so
      // callers fall through to the main-thread path on their next attempt.
      workerBroken = true;
      for (const [, entry] of pending) entry.reject(new Error('decoder worker failed'));
      pending.clear();
      worker = null;
    };
    return worker;
  } catch {
    workerBroken = true;
    return null;
  }
}

function abortable(signal, onAbort) {
  if (!signal) return () => {};
  if (signal.aborted) { onAbort(); return () => {}; }
  signal.addEventListener('abort', onAbort, { once: true });
  return () => signal.removeEventListener('abort', onAbort);
}

async function decodeOnMainThread(url, signal) {
  if (typeof createImageBitmap === 'function') {
    const res = await fetch(url, { signal, credentials: 'same-origin' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
    return createImageBitmap(blob);
  }
  // No createImageBitmap at all. `drawImage` takes an HTMLImageElement, so the
  // canvas path still works — it just decodes on this thread.
  return new Promise((resolve, reject) => {
    const img = new Image();
    const off = abortable(signal, () => { img.src = ''; reject(new DOMException('aborted', 'AbortError')); });
    img.onload = () => { off(); resolve(img); };
    img.onerror = () => { off(); reject(new Error('image decode failed')); };
    img.src = url;
  });
}

/**
 * Decode `url` to something `ctx.drawImage` accepts (ImageBitmap, or an
 * HTMLImageElement on the deepest fallback).
 *
 * Rejects with an AbortError when `signal` fires — callers treat that as
 * "superseded", not as a failure worth surfacing.
 */
export function decodeFrame(url, { signal } = {}) {
  if (signal?.aborted) return Promise.reject(new DOMException('aborted', 'AbortError'));

  const w = getWorker();
  if (!w) return decodeOnMainThread(url, signal);

  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    const off = abortable(signal, () => {
      if (pending.delete(id)) {
        w.postMessage({ id, type: 'abort' });
        reject(new DOMException('aborted', 'AbortError'));
      }
    });
    const settle = (fn) => (v) => { off(); fn(v); };
    const entry = pending.get(id);
    entry.resolve = settle(resolve);
    entry.reject = settle(reject);
    w.postMessage({ id, type: 'decode', url });
  }).catch((err) => {
    // The worker died mid-flight (see onerror). One retry on the main thread so
    // a single bad boot does not leave the stage permanently blank.
    if (workerBroken && err?.name !== 'AbortError') return decodeOnMainThread(url, signal);
    throw err;
  });
}

/** Release a decoded frame. ImageBitmaps hold GPU/heap memory until closed. */
export function releaseFrame(frame) {
  if (frame && typeof frame.close === 'function') frame.close();
}
