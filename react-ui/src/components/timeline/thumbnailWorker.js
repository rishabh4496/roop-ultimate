// ── Thumbnail Off-Thread Web Worker ─────────────────────────────────────────
//
// Generates and manages an off-thread OffscreenCanvas tile cache for the Master
// Video filmstrip. Keeps thumbnail decode, resizing, and caching completely off
// the 60 FPS UI rendering thread.

const CACHE_CAPACITY = 300;
const thumbnailCache = new Map(); // frame -> ImageBitmap

/**
 * Procedural fallback thumbnail generator for when video frames are loading
 * or before backend frames arrive, showing clean video slate and frame counter.
 */
function createProceduralTile(frame, width = 80, height = 48) {
  if (typeof OffscreenCanvas === 'undefined') return null;
  const offscreen = new OffscreenCanvas(width, height);
  const ctx = offscreen.getContext('2d');
  if (!ctx) return null;

  // Cinematic dark background with filmstrip slate
  ctx.fillStyle = '#18181B'; // zinc-900
  ctx.fillRect(0, 0, width, height);

  // Gradient shimmer
  const grad = ctx.createLinearGradient(0, 0, width, height);
  grad.addColorStop(0, 'rgba(255, 255, 255, 0.04)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0.4)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  // Top/bottom filmstrip sprocket holes
  ctx.fillStyle = '#09090B';
  const holeW = 6;
  const holeH = 4;
  for (let x = 4; x < width - 4; x += 14) {
    ctx.fillRect(x, 2, holeW, holeH);
    ctx.fillRect(x, height - holeH - 2, holeW, holeH);
  }

  // Border
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

  // Frame stamp
  ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
  ctx.font = 'bold 9px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`#${frame}`, width / 2, height / 2);

  return offscreen.transferToImageBitmap();
}

self.onmessage = async (e) => {
  const { type, frame, blob, width = 80, height = 48 } = e.data || {};

  if (type === 'decode_thumbnail') {
    try {
      let bitmap;
      if (blob) {
        bitmap = await createImageBitmap(blob, {
          resizeWidth: width,
          resizeHeight: height,
          resizeQuality: 'medium',
        });
      } else {
        bitmap = createProceduralTile(frame, width, height);
      }

      if (bitmap) {
        // Enforce cache LRU bounds
        if (thumbnailCache.size >= CACHE_CAPACITY) {
          const firstKey = thumbnailCache.keys().next().value;
          const old = thumbnailCache.get(firstKey);
          old?.close?.();
          thumbnailCache.delete(firstKey);
        }

        // Post transferable bitmap back to main thread
        self.postMessage(
          {
            type: 'thumbnail_ready',
            frame,
            bitmap,
          },
          [bitmap]
        );
      }
    } catch (err) {
      self.postMessage({ type: 'thumbnail_error', frame, error: String(err) });
    }
  } else if (type === 'clear_cache') {
    for (const bmp of thumbnailCache.values()) {
      bmp?.close?.();
    }
    thumbnailCache.clear();
  }
};
