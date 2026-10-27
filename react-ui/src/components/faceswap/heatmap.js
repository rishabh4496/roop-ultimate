// Amplified Difference / Heatmap LUT and Rendering Engine
//
// Computes |Final - Original| * Gain with false-color Turbo/Inferno colormaps
// to immediately spot unintended background shifts or bounding-box seam leaks.

// Google Turbo colormap polynomial approximation (smooth perception, zero banding)
function turboRgb(x) {
  const t = Math.max(0, Math.min(1, x));
  const r = Math.min(255, Math.max(0, Math.round(255 * (0.1357 + t * (4.5974 - t * (42.681 - t * (130.58 - t * (154.49 - t * 59.395))))))));
  const g = Math.min(255, Math.max(0, Math.round(255 * (0.0914 + t * (2.1856 + t * (4.8052 - t * (14.0195 - t * (4.2109 - t * 2.7747))))))));
  const b = Math.min(255, Math.max(0, Math.round(255 * (0.1067 + t * (12.573 - t * (86.19 - t * (231.44 - t * (264.48 - t * 113.25))))))));
  return [r, g, b];
}

// Inferno colormap approximation: dark purple -> magenta -> orange -> yellow
function infernoRgb(x) {
  const t = Math.max(0, Math.min(1, x));
  const r = Math.min(255, Math.max(0, Math.round(255 * (0.0002 + t * (1.1614 + t * (2.0523 - t * (2.3683 - t * (0.1805))))))));
  const g = Math.min(255, Math.max(0, Math.round(255 * (0.0014 + t * (0.0142 + t * (0.4578 + t * (1.5233 - t * 0.9959)))))));
  const b = Math.min(255, Math.max(0, Math.round(255 * (0.0137 + t * (2.5845 - t * (7.6749 + t * (11.025 - t * 5.9458)))))));
  return [r, g, b];
}

// Precomputed 256-entry 32-bit packed RGBA LUTs for sub-millisecond lookup
export const TURBO_LUT_32 = new Uint32Array(256);
export const INFERNO_LUT_32 = new Uint32Array(256);

for (let i = 0; i < 256; i++) {
  const t = i / 255;
  const [tr, tg, tb] = turboRgb(t);
  // Little-endian packed ABGR: (A << 24) | (B << 16) | (G << 8) | R
  TURBO_LUT_32[i] = (255 << 24) | (tb << 16) | (tg << 8) | tr;

  const [ir, ig, ib] = infernoRgb(t);
  INFERNO_LUT_32[i] = (255 << 24) | (ib << 16) | (ig << 8) | ir;
}

// Reusable scratch canvases to avoid per-frame DOM allocation
let scratchCanvasA = null;
let scratchCanvasB = null;

function getScratch(w, h) {
  if (typeof document === 'undefined') return { a: null, b: null };
  if (!scratchCanvasA) {
    scratchCanvasA = document.createElement('canvas');
    scratchCanvasB = document.createElement('canvas');
  }
  if (scratchCanvasA.width !== w || scratchCanvasA.height !== h) {
    scratchCanvasA.width = w;
    scratchCanvasA.height = h;
    scratchCanvasB.width = w;
    scratchCanvasB.height = h;
  }
  return { a: scratchCanvasA, b: scratchCanvasB };
}

/**
 * Computes |top - base| * gain with false-color LUT mapping directly onto target canvas ctx.
 * Runs at 60fps on modern canvases.
 */
export function drawAmplifiedHeatmap(ctx, baseImg, topImg, w, h, gain = 3.0, colormap = 'turbo') {
  if (!ctx || !baseImg || !topImg || !w || !h) return;
  const { a, b } = getScratch(w, h);
  if (!a || !b) return;

  const ctxA = a.getContext('2d', { willReadFrequently: true });
  const ctxB = b.getContext('2d', { willReadFrequently: true });
  if (!ctxA || !ctxB) return;

  ctxA.clearRect(0, 0, w, h);
  ctxB.clearRect(0, 0, w, h);
  ctxA.drawImage(baseImg, 0, 0, w, h);
  ctxB.drawImage(topImg, 0, 0, w, h);

  const imgDataA = ctxA.getImageData(0, 0, w, h);
  const imgDataB = ctxB.getImageData(0, 0, w, h);
  const dataA = imgDataA.data;
  const dataB = imgDataB.data;

  const outputImgData = ctx.createImageData(w, h);
  const out32 = new Uint32Array(outputImgData.data.buffer);
  const lut = colormap === 'inferno' ? INFERNO_LUT_32 : TURBO_LUT_32;

  const len = dataA.length;
  const pixelCount = len >> 2;

  for (let p = 0; p < pixelCount; p++) {
    const idx = p << 2;
    const dr = Math.abs(dataB[idx] - dataA[idx]) * gain;
    const dg = Math.abs(dataB[idx + 1] - dataA[idx + 1]) * gain;
    const db = Math.abs(dataB[idx + 2] - dataA[idx + 2]) * gain;
    // Euclidean magnitude or max delta
    const delta = Math.min(255, Math.round(Math.max(dr, dg, db)));
    out32[p] = lut[delta];
  }

  ctx.putImageData(outputImgData, 0, 0);
}
