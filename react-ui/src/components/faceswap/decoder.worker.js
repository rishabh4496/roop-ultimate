// ── Frame decoder worker ──────────────────────────────────────────────────
//
// Fetches a frame URL and decodes it to an ImageBitmap OFF the main thread.
//
// Why it is worth a worker: a 4K JPEG costs 15-40 ms to decode, and on the main
// thread that lands inside the same frame budget as React's commit, the
// compositor's paint and the pointer handlers driving a scrub — so the decode
// of frame N is what makes the drag stutter at frame N+1. `createImageBitmap`
// here returns a TRANSFERABLE bitmap: the main thread receives an already-
// decoded, GPU-uploadable object and `drawImage`s it with no decode of its own.
//
// The fetch lives in here too rather than on the main thread, so that an
// abort — the scrubber moving on — cancels the network read AND the decode as
// one, instead of cancelling the request and still paying for a decode nobody
// wants.

/** @type {Map<number, AbortController>} */
const inFlight = new Map();

self.onmessage = async (e) => {
  const { id, type, url } = e.data || {};

  if (type === 'abort') {
    inFlight.get(id)?.abort();
    inFlight.delete(id);
    return;
  }
  if (type !== 'decode') return;

  const ctrl = new AbortController();
  inFlight.set(id, ctrl);
  try {
    const res = await fetch(url, { signal: ctrl.signal, credentials: 'same-origin' });
    if (!res.ok) {
      // A 503 that carries a retry hint is the backend being busy (pre-pass burst) or still starting:
      // pass the hint on so the requester can wait as told instead of counting a failure.
      let body = null;
      if (res.status === 503) { try { body = await res.json(); } catch { body = null; } }
      const header = res.headers.get('retry-after');
      throw Object.assign(new Error(`HTTP ${res.status}`), {
        status: res.status,
        busy: res.status === 503 && !!(body?.busy || body?.retry_after_ms || header),
        retryAfterMs: body?.retry_after_ms ?? null,
        retryAfterHeader: header,
      });
    }
    const blob = await res.blob();
    if (ctrl.signal.aborted) throw new DOMException('aborted', 'AbortError');
    const bitmap = await createImageBitmap(blob);
    if (ctrl.signal.aborted) { bitmap.close(); throw new DOMException('aborted', 'AbortError'); }
    // Transfer, don't clone: a cloned bitmap would be a second full-size copy.
    self.postMessage({ id, ok: true, bitmap, width: bitmap.width, height: bitmap.height },
      [bitmap]);
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      aborted: err?.name === 'AbortError',
      message: String(err?.message || err),
      status: err?.status ?? null,
      busy: err?.busy === true,
      retryAfterMs: err?.retryAfterMs ?? null,
      retryAfterHeader: err?.retryAfterHeader ?? null,
    });
  } finally {
    inFlight.delete(id);
  }
};
