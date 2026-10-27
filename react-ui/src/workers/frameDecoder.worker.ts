// ── Dedicated Frame Decoder Web Worker ─────────────────────────────────────────
// Decodes image frames off the main thread using createImageBitmap with
// { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } and transfers
// the resulting ImageBitmap back to the main thread via zero-copy transferable objects.

interface DecodeRequest {
  id: number;
  type: 'decode';
  url?: string;
  buffer?: ArrayBuffer;
  blob?: Blob;
  mime?: string;
}

interface AbortRequest {
  id: number;
  type: 'abort';
}

type WorkerMessage = DecodeRequest | AbortRequest;

const inFlight = new Map<number, AbortController>();

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  const data = e.data;
  if (!data) return;

  if (data.type === 'abort') {
    inFlight.get(data.id)?.abort();
    inFlight.delete(data.id);
    return;
  }

  if (data.type !== 'decode') return;

  const { id, url, buffer, blob, mime } = data;
  const ctrl = new AbortController();
  inFlight.set(id, ctrl);

  try {
    let sourceBlob: Blob;

    if (blob instanceof Blob) {
      sourceBlob = blob;
    } else if (buffer instanceof ArrayBuffer) {
      sourceBlob = new Blob([buffer], { type: mime || 'image/jpeg' });
    } else if (typeof url === 'string' && url) {
      const res = await fetch(url, { signal: ctrl.signal, credentials: 'same-origin' });
      if (!res.ok) {
        let body: any = null;
        if (res.status === 503) {
          try {
            body = await res.json();
          } catch {
            body = null;
          }
        }
        const header = res.headers.get('retry-after');
        const err = new Error(`HTTP ${res.status}`) as any;
        err.status = res.status;
        err.busy = res.status === 503 && !!(body?.busy || body?.retry_after_ms || header);
        err.retryAfterMs = body?.retry_after_ms ?? null;
        err.retryAfterHeader = header;
        throw err;
      }
      sourceBlob = await res.blob();
    } else {
      throw new Error('No valid frame source (url, buffer, or blob) provided');
    }

    if (ctrl.signal.aborted) {
      throw new DOMException('aborted', 'AbortError');
    }

    let bitmap: ImageBitmap;
    try {
      bitmap = await createImageBitmap(sourceBlob, {
        premultiplyAlpha: 'none',
        colorSpaceConversion: 'none',
      });
    } catch {
      // Fallback if browser environment does not support decoding options
      bitmap = await createImageBitmap(sourceBlob);
    }

    if (ctrl.signal.aborted) {
      bitmap.close();
      throw new DOMException('aborted', 'AbortError');
    }

    // Zero-copy transfer of ImageBitmap to the caller
    self.postMessage(
      {
        id,
        ok: true,
        bitmap,
        width: bitmap.width,
        height: bitmap.height,
      },
      [bitmap]
    );
  } catch (err: any) {
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
