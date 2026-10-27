// ── Frame decoder worker ──────────────────────────────────────────────────
//
// Decodes a frame URL, Blob, or ArrayBuffer to an ImageBitmap OFF the main thread
// using { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } and transfers
// the decoded ImageBitmap back to the main thread via zero-copy transferable objects.

/** @type {Map<number, AbortController>} */
const inFlight = new Map();

self.onmessage = async (e) => {
  const { id, type, url, buffer, blob, mime } = e.data || {};

  if (type === 'abort') {
    inFlight.get(id)?.abort();
    inFlight.delete(id);
    return;
  }
  if (type !== 'decode') return;

  const ctrl = new AbortController();
  inFlight.set(id, ctrl);
  try {
    let sourceBlob;
    if (blob instanceof Blob) {
      sourceBlob = blob;
    } else if (buffer instanceof ArrayBuffer) {
      sourceBlob = new Blob([buffer], { type: mime || 'image/jpeg' });
    } else if (typeof url === 'string' && url) {
      const res = await fetch(url, { signal: ctrl.signal, credentials: 'same-origin' });
      if (!res.ok) {
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
      sourceBlob = await res.blob();
    } else {
      throw new Error('No valid frame source provided');
    }

    if (ctrl.signal.aborted) throw new DOMException('aborted', 'AbortError');

    let bitmap;
    try {
      bitmap = await createImageBitmap(sourceBlob, {
        premultiplyAlpha: 'none',
        colorSpaceConversion: 'none',
      });
    } catch {
      bitmap = await createImageBitmap(sourceBlob);
    }

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
