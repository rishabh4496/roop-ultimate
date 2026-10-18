// Pointer -> mask-canvas mapping for the Mask Brush. Pure, so .render-check/brush-mapping-check.mjs
// can pin it without a browser.
//
// THE BOX IS NOT THE PICTURE. The mask canvas is `absolute inset-0 w-full h-full object-contain`
// at the media's NATIVE size, laid over PreviewCanvas, which has the same object-contain in the
// same box. `canvas.getBoundingClientRect()` is the whole box. When the box's aspect is not the
// media's (an anamorphic clip: the box takes the display aspect from SAR/DAR while the pixels keep
// their own), `object-contain` letterboxes the pixels INSIDE that rect, and a mapping that divides
// by `rect.width` is off in both scale and offset by exactly the letterbox, so the stroke lands
// beside the pointer. The mapping below first finds where the pixels really are.
//
// Everything is in VISUAL pixels (what getBoundingClientRect and clientX/Y report, stage zoom and
// pan and the app-level zoom already applied); the result is in canvas pixels, with the fraction
// kept: a stroke is drawn at sub-pixel positions, never snapped to the pixel grid.

/** Where an `object-fit: contain` image of w x h sits inside `rect` (centered). */
export function containedRect(rect, w, h) {
  const s = Math.min(rect.width / w, rect.height / h);
  const width = w * s;
  const height = h * s;
  return {
    left: rect.left + (rect.width - width) / 2,
    top: rect.top + (rect.height - height) / 2,
    width,
    height,
  };
}

/**
 * One brush sample. `brushSize` is the on-screen DIAMETER in visual pixels (the slider's number),
 * so the canvas radius is half of it times canvas-pixels-per-visual-pixel.
 * Returns null when the box has no size yet.
 */
export function brushSample(rect, canvasW, canvasH, clientX, clientY, brushSize) {
  if (!rect || !rect.width || !rect.height || !canvasW || !canvasH) return null;
  const c = containedRect(rect, canvasW, canvasH);
  const scale = canvasW / c.width;
  return {
    x: (clientX - c.left) * scale,
    y: (clientY - c.top) * scale,
    r: Math.max(0.5, (brushSize / 2) * scale),
    scale,
    // Pointer on the picture, not on a letterbox bar. A stroke may still START outside and
    // wander in, so drawing does not require it; the caller may.
    onImage: clientX >= c.left && clientX <= c.left + c.width && clientY >= c.top && clientY <= c.top + c.height,
  };
}
