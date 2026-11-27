/**
 * False-Color Luminance & Skin-Tone Zebra Overlay Engine.
 *
 * Professional broadcast-grade IRE luminance zones and exposure zebra overlay.
 * Highlights:
 *  - Clipped highlights (>95 IRE) with 45° diagonal warning zebra stripes
 *  - Crushed shadows (<5 IRE) with -45° diagonal warning zebra stripes
 *  - Target skin-tone exposure band (55-72 IRE) for skin exposure inspection
 */

// Rec.709 Luma weights
const RW = 0.2126;
const GW = 0.7152;
const BW = 0.0722;

let scratchCanvas = null;

function getScratch(w, h) {
  if (typeof document === 'undefined') return null;
  if (!scratchCanvas) {
    scratchCanvas = document.createElement('canvas');
  }
  if (scratchCanvas.width !== w || scratchCanvas.height !== h) {
    scratchCanvas.width = w;
    scratchCanvas.height = h;
  }
  return scratchCanvas;
}

/**
 * Draws the False-Color / Zebra Exposure overlay directly onto target canvas ctx.
 *
 * @param {CanvasRenderingContext2D} ctx Target canvas context
 * @param {number} cw Canvas width
 * @param {number} ch Canvas height
 * @param {Array<number>|null} faceBbox [x1, y1, x2, y2] face bounding box in canvas coordinates
 * @param {string} mode 'zebra' | 'false_color'
 * @param {boolean} roiOnly Whether to restrict analysis strictly to face ROI
 */
export function drawZebraOverlay(ctx, cw, ch, faceBbox = null, mode = 'zebra', roiOnly = false) {
  if (!ctx || cw <= 0 || ch <= 0) return;

  const scratch = getScratch(cw, ch);
  if (!scratch) return;
  const sCtx = scratch.getContext('2d', { willReadFrequently: true });
  if (!sCtx) return;

  // Determine ROI bounds
  let minX = 0;
  let minY = 0;
  let maxX = cw;
  let maxY = ch;

  if (faceBbox && faceBbox.length >= 4) {
    const [bx1, by1, bx2, by2] = faceBbox;
    minX = Math.max(0, Math.min(cw, Math.round(bx1)));
    minY = Math.max(0, Math.min(ch, Math.round(by1)));
    maxX = Math.max(0, Math.min(cw, Math.round(bx2)));
    maxY = Math.max(0, Math.min(ch, Math.round(by2)));
  }

  const rw = maxX - minX;
  const rh = maxY - minY;
  if (rw <= 2 || rh <= 2) return;

  // Read current canvas pixels for the region
  let imgData;
  try {
    imgData = ctx.getImageData(minX, minY, rw, rh);
  } catch {
    return;
  }

  const data = imgData.data;
  const overlayData = sCtx.createImageData(rw, rh);
  const out = overlayData.data;

  // Process pixels
  for (let y = 0; y < rh; y++) {
    const globalY = minY + y;
    for (let x = 0; x < rw; x++) {
      const globalX = minX + x;
      const idx = (y * rw + x) * 4;

      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const a = data[idx + 3];

      if (a < 10) continue;

      // Rec.709 Luminance and IRE calculation: 0..100
      const luma = RW * r + GW * g + BW * b;
      const ire = (luma / 255) * 100;

      if (mode === 'false_color') {
        // Broadcast False-Color IRE spectrum mapping
        if (ire >= 95) {
          // Clipped highlight (>95 IRE) -> Red
          out[idx] = 239;
          out[idx + 1] = 68;
          out[idx + 2] = 68;
          out[idx + 3] = 220;
        } else if (ire >= 85) {
          // Near clipping (85-95 IRE) -> Yellow
          out[idx] = 234;
          out[idx + 1] = 179;
          out[idx + 2] = 8;
          out[idx + 3] = 180;
        } else if (ire >= 55 && ire <= 72) {
          // Target skin-tone sweet spot (55-72 IRE) -> Pink / Green
          out[idx] = 244;
          out[idx + 1] = 114;
          out[idx + 2] = 182;
          out[idx + 3] = 150;
        } else if (ire >= 40 && ire < 55) {
          // Midtones (40-55 IRE) -> Green
          out[idx] = 34;
          out[idx + 1] = 197;
          out[idx + 2] = 94;
          out[idx + 3] = 120;
        } else if (ire <= 5) {
          // Crushed shadow (<5 IRE) -> Purple
          out[idx] = 168;
          out[idx + 1] = 85;
          out[idx + 2] = 247;
          out[idx + 3] = 220;
        } else if (ire <= 18) {
          // Low shadows (5-18 IRE) -> Deep Blue
          out[idx] = 59;
          out[idx + 1] = 130;
          out[idx + 2] = 246;
          out[idx + 3] = 160;
        }
      } else {
        // Standard high-contrast Zebra pattern mode
        if (ire >= 95) {
          // Highlight zebra (45° diagonal stripes): alternating vivid red and white
          const stripe = (globalX + globalY) % 10 < 5;
          out[idx] = stripe ? 239 : 255;
          out[idx + 1] = stripe ? 68 : 255;
          out[idx + 2] = stripe ? 68 : 255;
          out[idx + 3] = 240;
        } else if (ire <= 5) {
          // Shadow zebra (-45° diagonal stripes): alternating deep indigo and black
          const stripe = ((globalX - globalY) % 10 + 10) % 10 < 5;
          out[idx] = stripe ? 99 : 10;
          out[idx + 1] = stripe ? 102 : 10;
          out[idx + 2] = stripe ? 241 : 20;
          out[idx + 3] = 240;
        } else if (ire >= 55 && ire <= 72 && (faceBbox || roiOnly)) {
          // Subtle skin-tone indicator within face ROI: soft green stipple
          if ((globalX + globalY) % 6 === 0) {
            out[idx] = 74;
            out[idx + 1] = 222;
            out[idx + 2] = 128;
            out[idx + 3] = 110;
          }
        }
      }
    }
  }

  // Draw overlay directly onto target canvas
  sCtx.clearRect(0, 0, rw, rh);
  sCtx.putImageData(overlayData, 0, 0);
  ctx.drawImage(scratch, 0, 0, rw, rh, minX, minY, rw, rh);

  // If face ROI is active, draw clean professional bounding corner markers
  if (faceBbox && faceBbox.length >= 4) {
    ctx.save();
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(minX, minY, rw, rh);

    // ROI Label Pill
    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.fillRect(minX, Math.max(0, minY - 16), 110, 16);
    ctx.fillStyle = '#fda4af';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('IRE FACE EXPOSURE', minX + 4, Math.max(0, minY - 16) + 11);
    ctx.restore();
  }
}
