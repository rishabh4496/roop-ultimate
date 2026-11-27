// The canvas half of the face context menu: decode the frame on the stage, read the pixels of a face's skin regions,
// cut a crop. The maths that interprets those pixels is faceContextModel.js (pure, tested under node).

import { faceRois, skinStats } from './faceContextModel';

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/**
 * The image behind a stage layer as an ImageBitmap. A bitmap the stage already holds (`frame`) is used as it is and is
 * never closed here - the owner closes it. Otherwise the url is fetched and decoded; `release()` closes only that one.
 */
export async function loadStageImage(frame, src) {
  if (frame && frame.width) return { bitmap: frame, w: frame.width, h: frame.height, release() {} };
  if (!src) throw new Error('there is no image on the stage');
  const res = await fetch(src, { signal: AbortSignal.timeout(20000) });
  if (!res.ok) throw new Error(`could not read the frame (HTTP ${res.status})`);
  const bitmap = await createImageBitmap(await res.blob());
  return { bitmap, w: bitmap.width, h: bitmap.height, release: () => bitmap.close() };
}

/** Median skin chroma of one face in one image. `bbox` / `kps` are in `imgDim` space; the image may be any size. */
export function sampleFaceSkin(img, bbox, kps, imgDim) {
  const sx = img.w / imgDim.w;
  const sy = img.h / imgDim.h;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const patches = [];
  for (const r of faceRois(bbox, kps)) {
    const x = clamp(Math.round(r.x * sx), 0, img.w - 1);
    const y = clamp(Math.round(r.y * sy), 0, img.h - 1);
    const w = clamp(Math.round(r.w * sx), 1, img.w - x);
    const h = clamp(Math.round(r.h * sy), 1, img.h - y);
    // A median does not need every pixel of a big face: a 48 px patch keeps the read cheap at 4K.
    canvas.width = Math.min(48, w);
    canvas.height = Math.min(48, h);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img.bitmap, x, y, w, h, 0, 0, canvas.width, canvas.height);
    patches.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  }
  return skinStats(patches);
}

/** A PNG of `rect` (the frame's own pixels) - lossless and at the frame's full resolution. */
export function cropToPng(img, rect) {
  const canvas = document.createElement('canvas');
  canvas.width = rect.w;
  canvas.height = rect.h;
  canvas.getContext('2d').drawImage(img.bitmap, rect.x, rect.y, rect.w, rect.h, 0, 0, rect.w, rect.h);
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('could not encode the crop'))), 'image/png');
  });
}
