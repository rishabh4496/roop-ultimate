import React, {
  forwardRef, useCallback, useEffect, useImperativeHandle, useRef,
} from 'react';
import { decodeFrame, releaseFrame } from './frameDecoder';

// ── The preview stage, as an UNCONTROLLED canvas ──────────────────────────
//
// What this replaces: two <img> layers whose `src` came from React state
// holding a base64 data URL, cross-faded by animating an `opacity` that was
// ALSO React state. Every frame of a scrub therefore committed a re-render of
// the whole Face Swap panel (3,200 lines of JSX, thirty-odd hooks) just to move
// a picture, and every rendered preview pushed a multi-megabyte string through
// that commit. React was being asked to diff a video.
//
// Here the pixels are not React's business at all. `src` changes are consumed
// by an effect that decodes off-thread and paints with `ctx.drawImage`; the
// cross-fade is an alpha ramp inside one rAF loop; the compare wipe is a clip
// rectangle set through an imperative handle. NOTHING in this component calls
// setState during a paint, a fade or a drag — the only re-renders it can have
// are the ones its parent gives it.
//
// Two consequences worth stating because they are easy to undo by accident:
//
//  * <canvas> is uncontrolled by construction. React must never be given a
//    reason to recreate the element (no `key` derived from the frame, no
//    conditional mount), or every source change would drop the backing store
//    and flash black between frames.
//  * decoded frames are ImageBitmaps, which hold memory until `.close()`. They
//    are released on replacement and on unmount, in the same spirit as
//    objectUrls.js — a decoded 4K frame is ~33 MB, and holding one per scrub
//    tick is a faster leak than any blob URL.

// Longest edge the backing store is allowed to take. The stage is ~1000 CSS
// pixels at most, so anything above this is memory spent on detail the display
// cannot show — the browser downscales on composite either way.
const MAX_EDGE = 2560;

// Release only what this component decoded. Caller-supplied frames belong to
// the caller's lifecycle.
const releaseOwned = (layer, img) => { if (img && !layer.external) releaseFrame(img); };

// `object-fit: contain`, done by hand because a canvas has no such property.
const fitBox = (cw, ch, iw, ih) => {
  const s = Math.min(cw / iw, ch / ih);
  const w = iw * s;
  const h = ih * s;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
};

const PreviewCanvas = forwardRef(function PreviewCanvas({
  baseSrc = '',
  topSrc = '',
  // Already-decoded frames, owned by the CALLER (see useThrottledFrameRequest).
  // When given they win over the matching `*Src`, and this component never
  // closes them — closing a bitmap someone else still holds is a use-after-free
  // that shows up as a silently blank stage, not as an error.
  baseFrame = null,
  topFrame = null,
  // 0..100. How much of the stage the TOP layer covers. 100 = fully swapped.
  wipe = 100,
  wipeDir = 'vertical',       // 'vertical' = a left/right curtain
  mode = 'normal',            // 'normal' | 'blend' | 'diff'
  fadeMs = 200,
  className = '',
  style,
  onDimensions,
}, ref) {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);

  // Everything the paint loop reads lives in refs. A prop that reached the loop
  // through a closure would need the loop re-bound on every change, which is
  // the re-render this component exists to avoid.
  const layersRef = useRef({
    base: { src: '', img: null, pending: null, external: false },
    top: { src: '', img: null, pending: null, prev: null, fadeStart: 0, external: false },
  });
  const wipeRef = useRef(wipe);
  const cfgRef = useRef({ wipeDir, mode, fadeMs });
  cfgRef.current = { wipeDir, mode, fadeMs };

  const rafRef = useRef(0);
  const reportedDimRef = useRef('');
  const onDimensionsRef = useRef(onDimensions);
  onDimensionsRef.current = onDimensions;

  // ── Sizing: the canvas gets the FRAME's own dimensions ───────────────────
  //
  // THIS IS LOAD-BEARING LAYOUT, not a performance detail.
  //
  // A <canvas> is a replaced element whose INTRINSIC size is its width/height
  // attributes — defaulting to 300x300. The stage around it is content-sized:
  // the box that holds it has `aspect-ratio` and `max-width/height: 100%` but
  // no definite width, so with an <img> it took the picture's natural size and
  // filled the panel. Swap in a canvas that has not been told otherwise and the
  // whole stage silently collapses to a 300x300 square — the element is there,
  // the paint is correct, and the preview is a thumbnail in the corner.
  //
  // Giving the canvas the frame's own dimensions makes it lay out EXACTLY as
  // the <img> did, which also keeps the face-box overlay correct: those boxes
  // are positioned as percentages of this element's box, so the box has to be
  // the picture and nothing else.
  //
  // Capped on the long edge because the backing store is real memory (a 4K
  // frame is ~33 MB) and the stage is never more than about a thousand CSS
  // pixels wide; the browser downscales on composite exactly as it did for the
  // <img>.
  const sizeToFrame = useCallback((iw, ih) => {
    const canvas = canvasRef.current;
    if (!canvas || !iw || !ih) return false;
    const scale = Math.min(1, MAX_EDGE / Math.max(iw, ih));
    const w = Math.max(1, Math.round(iw * scale));
    const h = Math.max(1, Math.round(ih * scale));
    if (canvas.width === w && canvas.height === h) return false;
    // Assigning either attribute RESETS the backing store, so this must happen
    // before the paint that follows, never between two draws of one frame.
    canvas.width = w;
    canvas.height = h;
    return true;
  }, []);

  const paint = useCallback((ts) => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx) return false;

    const { base, top } = layersRef.current;
    const { wipeDir: dir, mode: m, fadeMs: fade } = cfgRef.current;

    // Geometry comes from whichever layer we have — the two are the same frame
    // at the same resolution, so either answers.
    const anyImg = base.img || top.img || top.prev;
    if (!anyImg) return false;
    const iw = anyImg.width || anyImg.naturalWidth;
    const ih = anyImg.height || anyImg.naturalHeight;
    if (!iw || !ih) return false;
    sizeToFrame(iw, ih);

    const cw = canvas.width;
    const ch = canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;

    // Normally the identity rectangle — the canvas IS the frame's shape. It
    // stops mattering only when the two layers disagree (the swapped frame
    // arriving at a different resolution from the raw one), and then it
    // letterboxes rather than stretching.
    const box = fitBox(cw, ch, iw, ih);
    const dim = `${iw}x${ih}`;
    if (dim !== reportedDimRef.current) {
      reportedDimRef.current = dim;
      // Deferred: onDimensions is a parent setState, and calling it from inside
      // a paint would re-render the tree mid-frame.
      const cb = onDimensionsRef.current;
      if (cb) queueMicrotask(() => cb({ w: iw, h: ih }));
    }

    if (base.img) ctx.drawImage(base.img, box.x, box.y, box.w, box.h);

    // The top layer's own cross-fade: the OUTGOING frame stays up underneath at
    // full alpha while the incoming one comes in over it, so stepping between
    // frames is continuous instead of a blank flash. At fadeMs 0 (playback) it
    // is a straight replacement — no ramp, and no extra rAF.
    let alpha = 1;
    let needsMore = false;
    if (top.img && fade > 0 && top.fadeStart) {
      alpha = Math.min(1, (ts - top.fadeStart) / fade);
      if (alpha < 1) needsMore = true;
      else {
        top.fadeStart = 0;
        if (top.prev) { releaseOwned(top, top.prev); top.prev = null; }
      }
    } else if (top.img && top.prev) {
      releaseOwned(top, top.prev);
      top.prev = null;
    }

    const drawTop = () => {
      if (top.prev && alpha < 1) {
        ctx.globalAlpha = 1;
        ctx.drawImage(top.prev, box.x, box.y, box.w, box.h);
      }
      if (top.img) {
        ctx.globalAlpha = top.prev ? alpha : 1;
        ctx.drawImage(top.img, box.x, box.y, box.w, box.h);
        ctx.globalAlpha = 1;
      }
    };

    const pct = Math.max(0, Math.min(100, wipeRef.current)) / 100;

    if (m === 'diff') {
      ctx.globalCompositeOperation = 'difference';
      drawTop();
      ctx.globalCompositeOperation = 'source-over';
    } else if (m === 'blend') {
      ctx.globalAlpha = pct;
      drawTop();
      ctx.globalAlpha = 1;
    } else if (pct >= 0.999) {
      drawTop();
    } else if (pct > 0.0005) {
      // The curtain. Clipped to the STAGE, not to the image box, so the divider
      // the CompareSlider draws over it lines up with the cut at every size.
      ctx.save();
      ctx.beginPath();
      if (dir === 'horizontal') ctx.rect(0, ch * (1 - pct), cw, ch * pct);
      else ctx.rect(cw * (1 - pct), 0, cw * pct, ch);
      ctx.clip();
      drawTop();
      ctx.restore();
    }

    return needsMore;
  }, [sizeToFrame]);

  const requestPaint = useCallback(() => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame((ts) => {
      rafRef.current = 0;
      // Keep ticking only while a fade is still running. A canvas that is
      // simply displaying a frame costs nothing per rAF — because there is
      // no rAF.
      if (paint(ts)) requestPaint();
    });
  }, [paint]);


  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // NOT `desynchronized`. The low-latency path it selects is for ink-style
    // drawing, buys nothing at video rates, and makes getImageData read back
    // blank — which silently disables the mask export and every pixel check
    // that would tell us this component had stopped painting.
    ctxRef.current = canvas.getContext('2d', { alpha: true });
    requestPaint();
  }, [requestPaint]);

  // ── Source -> pixels ──────────────────────────────────────────────────────
  const loadLayer = useCallback((which, src) => {
    const layer = layersRef.current[which];
    if (layer.src === src) return;
    layer.src = src;

    // Whatever was being decoded for this layer is now superseded. Aborting is
    // not an optimisation here: the decode is the expensive half of a scrub,
    // and the frame it produces would be thrown away on arrival.
    layer.pending?.abort();
    layer.pending = null;

    if (!src) {
      releaseOwned(layer, layer.img);
      layer.img = null;
      requestPaint();
      return;
    }

    const ctrl = new AbortController();
    layer.pending = ctrl;
    decodeFrame(src, { signal: ctrl.signal }).then((img) => {
      if (ctrl.signal.aborted || layer.src !== src) { releaseFrame(img); return; }
      layer.pending = null;
      if (which === 'top') {
        // Hold the outgoing frame for the fade; the paint loop releases it.
        if (layer.prev) releaseOwned(layer, layer.prev);
        layer.prev = layer.img;
        layer.fadeStart = layer.prev ? performance.now() : 0;
      } else {
        releaseOwned(layer, layer.img);
      }
      layer.external = false;
      layer.img = img;
      requestPaint();
    }).catch(() => {
      // AbortError is the normal case (the user moved on). A real decode
      // failure leaves the previous frame up, which is what the <img> path did.
      if (layer.src === src) layer.pending = null;
    });
  }, [requestPaint]);

  // A caller-decoded frame takes over the layer: cancel any decode of our own
  // that is still running for it, and adopt the bitmap without copying it.
  const adopt = useCallback((which, img) => {
    const layer = layersRef.current[which];
    if (layer.img === img) return;
    layer.pending?.abort();
    layer.pending = null;
    layer.src = '';                    // so a later `*Src` change re-decodes
    if (which === 'top') {
      if (layer.prev) releaseOwned(layer, layer.prev);
      layer.prev = layer.img;
      layer.fadeStart = layer.prev ? performance.now() : 0;
    } else {
      releaseOwned(layer, layer.img);
    }
    layer.external = true;
    layer.img = img;
    requestPaint();
  }, [requestPaint]);

  useEffect(() => {
    if (baseFrame) adopt('base', baseFrame);
    else loadLayer('base', baseSrc);
  }, [baseFrame, baseSrc, adopt, loadLayer]);
  useEffect(() => {
    if (topFrame) adopt('top', topFrame);
    else loadLayer('top', topSrc);
  }, [topFrame, topSrc, adopt, loadLayer]);

  // A prop-driven wipe still works (the compare-mode buttons and the auto-swipe
  // set it); the slider DRAG uses the imperative handle below and re-renders
  // nothing at all.
  useEffect(() => { wipeRef.current = wipe; requestPaint(); }, [wipe, requestPaint]);
  useEffect(() => { requestPaint(); }, [mode, wipeDir, fadeMs, requestPaint]);

  // Teardown must return the layers to their INITIAL state, `src` included.
  //
  // Freeing the bitmaps and leaving `layer.src` set is the obvious version and
  // it is wrong: React 19 StrictMode mounts, tears down and mounts again, so on
  // the second mount `loadLayer` sees `layer.src === src`, treats the frame as
  // already loaded, and returns without decoding anything. The images are gone,
  // nothing ever asks for them again, and the stage is blank forever — with no
  // error, no failed request, and a canvas element sitting there at its default
  // size looking entirely correct in the DOM.
  //
  // It only bites in dev (StrictMode) and on any real remount, which is exactly
  // the combination that makes it easy to ship.
  useEffect(() => () => {
    // Cancelling the frame is not enough: `requestPaint` guards on this id to
    // avoid stacking rAFs, so leaving it set makes every later call a no-op and
    // the component never paints again. Same shape as the `src` reset below —
    // free the resource AND the bookkeeping that says it exists.
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
    const layers = layersRef.current;
    layers.base.pending?.abort();
    layers.top.pending?.abort();
    releaseOwned(layers.base, layers.base.img);
    releaseOwned(layers.top, layers.top.img);
    releaseOwned(layers.top, layers.top.prev);
    layers.base = { src: '', img: null, pending: null, external: false };
    layers.top = { src: '', img: null, pending: null, prev: null, fadeStart: 0, external: false };
    reportedDimRef.current = '';
  }, []);

  useImperativeHandle(ref, () => ({
    // Move the curtain with no React work at all — this is the drag path.
    setWipe(pct) { wipeRef.current = pct; requestPaint(); },
    getWipe() { return wipeRef.current; },
    canvas: () => canvasRef.current,
    // Natural pixel size of the frame on screen, for overlay maths.
    naturalSize() {
      const { base, top } = layersRef.current;
      const img = base.img || top.img;
      return img ? { w: img.width || img.naturalWidth, h: img.height || img.naturalHeight } : null;
    },
    repaint: requestPaint,
  }), [requestPaint]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={style}
      role="img"
      aria-label="Preview frame"
    />
  );
});

export default PreviewCanvas;
