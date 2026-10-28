import React, {
  forwardRef, useCallback, useEffect, useImperativeHandle, useRef,
} from 'react';
import { decodeFrame, releaseFrame } from './frameDecoder';
import { drawAmplifiedHeatmap } from './heatmap';
import { drawZebraOverlay } from './falseColorZebra';

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
//
// Ownership is tracked PER IMAGE. It used to be a flag on the layer (`layer.external`), which is
// the wrong granularity: a layer that hands over from a caller's frame to one it decodes itself
// flips the flag first, and the outgoing caller frame (held in `prev` for the cross-fade) is then
// released as if it were ours. That closed bitmaps the caller still owned - the snapshot slots'
// kept frames - and the next paint threw "The image source is detached".
const ownedFrames = new WeakSet();
const releaseOwned = (_layer, img) => {
  if (!img || !ownedFrames.has(img)) return;
  ownedFrames.delete(img);
  releaseFrame(img);
};

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
  wipeAngle = 0,              // 0..360 vector wipe angle in degrees
  mode = 'normal',            // 'normal' | 'blend' | 'diff' | 'heatmap'
  diffGain = 3.0,             // Amplified diff multiplier (1x - 10x)
  diffColorMap = 'turbo',     // 'turbo' | 'inferno'
  fadeMs = 200,
  zebra = false,
  zebraMode = 'zebra',
  faceBbox = null,
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
  const wipeAngleRef = useRef(wipeAngle);
  const diffGainRef = useRef(diffGain);
  const diffColorMapRef = useRef(diffColorMap);
  const cfgRef = useRef({ wipeDir, wipeAngle, mode, diffGain, diffColorMap, fadeMs, zebra, zebraMode });
  cfgRef.current = { wipeDir, wipeAngle, mode, diffGain, diffColorMap, fadeMs, zebra, zebraMode };
  const faceBboxRef = useRef(faceBbox);
  faceBboxRef.current = faceBbox;

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
    const baseW = base.img ? (base.img.width || base.img.naturalWidth) : iw;
    const baseH = base.img ? (base.img.height || base.img.naturalHeight) : ih;
    const baseBox = fitBox(cw, ch, baseW, baseH);
    const dim = `${iw}x${ih}`;
    if (dim !== reportedDimRef.current) {
      reportedDimRef.current = dim;
      // Deferred: onDimensions is a parent setState, and calling it from inside
      // a paint would re-render the tree mid-frame.
      const cb = onDimensionsRef.current;
      if (cb) queueMicrotask(() => cb({ w: iw, h: ih }));
    }

    if (base.img) ctx.drawImage(base.img, baseBox.x, baseBox.y, baseBox.w, baseBox.h);

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

    const getTopBox = (img) => {
      if (!img) return baseBox;
      const tw = img.width || img.naturalWidth;
      const th = img.height || img.naturalHeight;
      if (!tw || !th || (tw === baseW && th === baseH)) return baseBox;
      return fitBox(cw, ch, tw, th);
    };

    // `gain` is the layer's overall opacity (the alpha-blend percentage); the cross-fade's own
    // ramp multiplies it. Setting `ctx.globalAlpha` from the fade alone, as this did, silently
    // overwrote the blend's value, so Blend at 25% showed the swap at 100%.
    const drawTop = (gain = 1) => {
      if (top.prev && alpha < 1) {
        const pbox = getTopBox(top.prev);
        ctx.globalAlpha = gain;
        ctx.drawImage(top.prev, pbox.x, pbox.y, pbox.w, pbox.h);
      }
      if (top.img) {
        const tbox = getTopBox(top.img);
        ctx.globalAlpha = (top.prev ? alpha : 1) * gain;
        ctx.drawImage(top.img, tbox.x, tbox.y, tbox.w, tbox.h);
        ctx.globalAlpha = 1;
      }
    };

    const pct = Math.max(0, Math.min(100, wipeRef.current)) / 100;

    if (m === 'heatmap') {
      if (base.img && (top.img || top.prev)) {
        drawAmplifiedHeatmap(
          ctx,
          base.img,
          top.img || top.prev,
          cw,
          ch,
          diffGainRef.current || 3.0,
          diffColorMapRef.current || 'turbo',
        );
      } else {
        drawTop();
      }
    } else if (m === 'diff') {
      ctx.globalCompositeOperation = 'difference';
      drawTop();
      ctx.globalCompositeOperation = 'source-over';
    } else if (m === 'blend') {
      drawTop(pct);
      ctx.globalAlpha = 1;
    } else if (pct >= 0.999) {
      drawTop();
    } else if (pct > 0.0005) {
      ctx.save();
      ctx.beginPath();
      const angle = wipeAngleRef.current || 0;
      if (angle !== 0) {
        const rad = (angle * Math.PI) / 180;
        const diag = Math.hypot(cw, ch) * 2;
        const nx = Math.cos(rad);
        const ny = Math.sin(rad);
        const d = (pct - 0.5) * Math.hypot(cw, ch);
        const px = cw / 2 + d * nx;
        const py = ch / 2 + d * ny;
        ctx.translate(px, py);
        ctx.rotate(rad);
        ctx.rect(0, -diag, diag, diag * 2);
        ctx.clip();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
      } else if (dir === 'horizontal') {
        ctx.rect(0, ch * (1 - pct), cw, ch * pct);
        ctx.clip();
      } else {
        ctx.rect(cw * (1 - pct), 0, cw * pct, ch);
        ctx.clip();
      }
      drawTop();
      ctx.restore();
    }

    if (cfgRef.current.zebra) {
      drawZebraOverlay(
        ctx,
        cw,
        ch,
        faceBboxRef.current,
        cfgRef.current.zebraMode || 'zebra',
      );
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
      ownedFrames.add(img);
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
  useEffect(() => { wipeAngleRef.current = wipeAngle; requestPaint(); }, [wipeAngle, requestPaint]);
  useEffect(() => { diffGainRef.current = diffGain; requestPaint(); }, [diffGain, requestPaint]);
  useEffect(() => { diffColorMapRef.current = diffColorMap; requestPaint(); }, [diffColorMap, requestPaint]);
  useEffect(() => { requestPaint(); }, [mode, wipeDir, wipeAngle, diffGain, diffColorMap, fadeMs, zebra, zebraMode, faceBbox, requestPaint]);

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
    setWipeAngle(deg) { wipeAngleRef.current = deg; requestPaint(); },
    getWipeAngle() { return wipeAngleRef.current; },
    setDiffGain(gain) { diffGainRef.current = gain; requestPaint(); },
    getDiffGain() { return diffGainRef.current; },
    setMode(newMode) { cfgRef.current.mode = newMode; requestPaint(); },
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
