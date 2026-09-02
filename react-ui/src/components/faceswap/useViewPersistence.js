import { useEffect, useRef } from 'react';
import { blobUrlToDataUrl } from './objectUrls';

// ── Remember where you were looking, across a webview reload ──────────────
// Switching Pinokio tabs (React UI ↔ Terminal, Run ↔ Dev) reloads the webview,
// so the Face Swap tab is a fresh mount every time. The rehydrate effect in
// FaceSwap puts faces, targets and job state back from the backend — but two
// things it cannot restore are exactly the two you are looking at: the playhead
// (it forced frame 1) and the RENDERED preview, which is client state, so the
// box fell back to the raw frame. Coming back from the terminal therefore
// looked like the preview had reset itself.
//
// localStorage with a TTL rather than sessionStorage: sessionStorage belongs to
// a browsing context, and if Pinokio recreates the webview instead of reloading
// it there is nothing left to restore from. The TTL is what sessionStorage was
// buying — a view from days ago is not "where I was".

const VIEW_KEY = 'roop_view';
const VIEW_IMG_KEY = 'roop_view_preview';
const VIEW_TTL_MS = 30 * 60 * 1000;

// Above this, skip the image rather than risk a quota error that would take the
// small view entry down with it.
const MAX_IMAGE_BYTES = 3_000_000;

const WRITE_DEBOUNCE_MS = 400;

/**
 * @returns a ref holding `{ target, frame, image }` captured at mount, or null
 *          when there was nothing fresh to restore. A ref, not state, because
 *          the rehydrate callback reads it once and re-rendering on it would
 *          be pointless churn.
 */
export default function useViewPersistence({ selTarget, frame, previewSrc }) {
  const restoredViewRef = useRef(null);

  // Read BOTH halves at mount, before anything can overwrite them. The image
  // has to be captured here rather than looked up later in the rehydrate
  // callback: the writer below is on a debounce, so a slow /api/state would let
  // it run first — with previewSrc still empty — and clear the very entry the
  // callback was about to read.
  useEffect(() => {
    try {
      const v = JSON.parse(localStorage.getItem(VIEW_KEY) || 'null');
      const fresh = v && typeof v === 'object' && Date.now() - (v.t || 0) < VIEW_TTL_MS;
      restoredViewRef.current = fresh
        ? { ...v, image: localStorage.getItem(VIEW_IMG_KEY) || '' }
        : null;
      if (!fresh) localStorage.removeItem(VIEW_IMG_KEY);   // don't hoard a stale frame
    } catch { restoredViewRef.current = null; }
  }, []);

  // Debounced so scrubbing writes once when it settles, not once per frame.
  //
  // `previewSrc` is a BLOB URL now, and a blob URL dies with the document — the
  // one thing this hook exists to survive. So the bytes are read back out and
  // stored as a data URL. That re-serialisation is why the debounce matters
  // more than it used to: it must happen once, when the user settles, never per
  // scrub tick.
  useEffect(() => {
    let cancelled = false;
    const t = setTimeout(async () => {
      const bytes = await blobUrlToDataUrl(previewSrc);
      // The playhead moved while we were re-serialising: this image is of a
      // frame that is no longer on screen, and pairing it with the new frame
      // number is exactly the mismatch the comment below refuses to write.
      if (cancelled) return;
      try {
        // Image FIRST, and dropped outright if it cannot be written. If the
        // view entry were written first and the image then hit the quota, the
        // previous frame's image would still be sitting there — and the restore
        // would pair it with this frame, showing the wrong picture under a
        // frame number that says otherwise. No image is fine; a mismatched one
        // is not.
        if (bytes && bytes.length < MAX_IMAGE_BYTES) {
          try { localStorage.setItem(VIEW_IMG_KEY, bytes); }
          catch { localStorage.removeItem(VIEW_IMG_KEY); }
        } else {
          localStorage.removeItem(VIEW_IMG_KEY);
        }
        localStorage.setItem(VIEW_KEY, JSON.stringify({ target: selTarget, frame, t: Date.now() }));
      } catch { /* storage blocked — the view just won't be restored */ }
    }, WRITE_DEBOUNCE_MS);
    return () => { cancelled = true; clearTimeout(t); };
  }, [selTarget, frame, previewSrc]);

  return restoredViewRef;
}
