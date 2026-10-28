// The mask brush, as DATA.
//
// Painting used to exist only as pixels on two canvases, so there was nothing to undo: removing the
// last stroke meant clearing everything. A stroke is now recorded as the list of dab samples the
// brush laid down, in coordinates normalised to the canvas, and the canvases are what you get by
// replaying the list. That is what the undo history stores (a few hundred numbers per stroke, not a
// 4K bitmap per stroke), and it makes a canvas resize (which wipes a canvas) recoverable too.
//
//   stroke = { erase: bool, feather: 0..1, pts: [[x, y, r, innerR], ...] }   x,y,r,innerR in 0..1 of canvas WIDTH/HEIGHT

const TINT = 'rgba(233, 69, 96, 0.45)';
const TINT_CLEAR = 'rgba(233, 69, 96, 0)';

/** Canvas-pixel sample -> normalised (x,y by their own axis; radii by width, as the canvas is scaled uniformly). */
export function normalizeSample({ x, y, r, innerR }, w, h) {
  return [x / w, y / h, r / w, innerR / w];
}

export function denormalizeSample([x, y, r, innerR], w, h) {
  return { x: x * w, y: y * h, r: r * w, innerR: innerR * w };
}

/**
 * One dab (and the line joining it to the previous one) onto BOTH canvases: the visible tinted one
 * and the white-on-transparent export one that becomes the mask PNG.
 */
export function paintDab(canvas, exportCanvas, sample, prev, erase, feather) {
  const { x, y, r, innerR } = sample;
  for (const [c, isExport] of [[canvas, false], [exportCanvas, true]]) {
    const ctx = c?.getContext('2d');
    if (!ctx) continue;
    ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';

    if (feather > 0.05) {
      const grad = ctx.createRadialGradient(x, y, Math.max(0, innerR), x, y, r);
      if (isExport || erase) {
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
      } else {
        grad.addColorStop(0, TINT);
        grad.addColorStop(1, TINT_CLEAR);
      }
      ctx.fillStyle = grad;
      ctx.strokeStyle = grad;
    } else {
      const color = isExport || erase ? '#ffffff' : TINT;
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
    }

    ctx.lineWidth = r * 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (prev) {
      ctx.beginPath();
      ctx.moveTo(prev.x, prev.y);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function clearCanvases(...canvases) {
  for (const c of canvases) {
    const ctx = c?.getContext('2d');
    if (ctx) {
      ctx.globalCompositeOperation = 'source-over';
      ctx.clearRect(0, 0, c.width, c.height);
    }
  }
}

/** Redraw `strokes` from scratch onto both canvases. */
export function replayStrokes(canvas, exportCanvas, strokes) {
  if (!canvas || !exportCanvas) return;
  clearCanvases(canvas, exportCanvas);
  const w = canvas.width;
  const h = canvas.height;
  if (!w || !h) return;
  for (const stroke of strokes || []) {
    let prev = null;
    for (const pt of stroke.pts) {
      const sample = denormalizeSample(pt, w, h);
      paintDab(canvas, exportCanvas, sample, prev, stroke.erase, stroke.feather);
      prev = { x: sample.x, y: sample.y };
    }
  }
}
