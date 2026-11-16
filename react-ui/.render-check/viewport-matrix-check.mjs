/**
 * viewport-matrix-check.mjs
 * Pure unit tests for Unified ViewportMatrix, LRU Frame Cache, Loupe zero-drift math,
 * and Pose Classification.
 *
 * Run: node .render-check/run-module.mjs viewport-matrix-check.mjs
 */

import {
  ViewportMatrix,
  createViewportMatrix,
  classifyPoseBucket,
  LruFrameCache,
} from '../src/components/faceswap/viewportMatrix.ts';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const near = (a, b, eps = 1e-4) => Math.abs(a - b) <= eps;

console.log('── ViewportMatrix: Zero-Drift Media/Screen Affine Transforms ──');
{
  // 16:9 media (1920x1080) inside a 16:9 container (960x540)
  const vm = createViewportMatrix({
    containerWidth: 960,
    containerHeight: 540,
    mediaWidth: 1920,
    mediaHeight: 1080,
    zoom: 1.0,
    pan: { x: 0, y: 0 },
  });

  const p0 = vm.mediaToScreen(0, 0);
  ok('Top-left (0,0) maps to screen (0,0)', near(p0.x, 0) && near(p0.y, 0), `${p0.x}, ${p0.y}`);

  const pCenter = vm.mediaToScreen(960, 540);
  ok('Center (960, 540) maps to screen (480, 270)', near(pCenter.x, 480) && near(pCenter.y, 270));

  const pBottomRight = vm.mediaToScreen(1920, 1080);
  ok('Bottom-right (1920, 1080) maps to screen (960, 540)', near(pBottomRight.x, 960) && near(pBottomRight.y, 540));

  // Inverse round-trip
  const rev = vm.screenToMedia(480, 270);
  ok('screenToMedia inverse round-trip center', near(rev.x, 960) && near(rev.y, 540) && rev.onMedia);

  const offMedia = vm.screenToMedia(-50, -50);
  ok('Points outside media report onMedia: false', !offMedia.onMedia);
}

console.log('── ViewportMatrix: Pillarboxing & Letterboxing ──');
{
  // Pillarbox: 9:16 portrait media (1080x1920) in 16:9 landscape container (1920x1080)
  const vmPillar = createViewportMatrix({
    containerWidth: 1920,
    containerHeight: 1080,
    mediaWidth: 1080,
    mediaHeight: 1920,
    zoom: 1.0,
    pan: { x: 0, y: 0 },
  });

  // Scale is 1080 / 1920 = 0.5625
  // Contained width = 1080 * 0.5625 = 607.5
  // Contained left = (1920 - 607.5) / 2 = 656.25
  const leftEdge = vmPillar.mediaToScreen(0, 0);
  ok('Pillarboxed left edge matches containedLeft (656.25)', near(leftEdge.x, 656.25) && near(leftEdge.y, 0));

  const rightEdge = vmPillar.mediaToScreen(1080, 0);
  ok('Pillarboxed right edge matches containedLeft + width (1263.75)', near(rightEdge.x, 1263.75));

  // Check clicking in pillarbox black bar
  const clickInPillarBar = vmPillar.screenToMedia(100, 540);
  ok('Click in pillarbox black bar reports onMedia: false', !clickInPillarBar.onMedia);
}

console.log('── ViewportMatrix: Zoom & Pan Affine Transformations ──');
{
  // 2.5x zoom with pan
  const vmZoomed = createViewportMatrix({
    containerWidth: 1000,
    containerHeight: 1000,
    mediaWidth: 1000,
    mediaHeight: 1000,
    zoom: 2.5,
    pan: { x: 150, y: -80 },
  });

  const centerZoomed = vmZoomed.mediaToScreen(500, 500);
  ok('Media center lands at containerCenter + pan (650, 420)', near(centerZoomed.x, 650) && near(centerZoomed.y, 420));

  const revZoomed = vmZoomed.screenToMedia(650, 420);
  ok('Inverse mapping at 2.5x zoom + pan returns exact (500, 500)', near(revZoomed.x, 500) && near(revZoomed.y, 500));
}

console.log('── ViewportMatrix: 3.5x Magnifier Loupe Zero-Drift Math ──');
{
  const vmLoupe = createViewportMatrix({
    containerWidth: 1920,
    containerHeight: 1080,
    mediaWidth: 1920,
    mediaHeight: 1080,
    zoom: 1.0,
    pan: { x: 0, y: 0 },
  });

  const pointerX = 800;
  const pointerY = 600;
  const loupeDiameter = 180;
  const crop = vmLoupe.getLoupeCrop(pointerX, pointerY, loupeDiameter, 3.5);

  ok('Loupe screen center matches pointer coordinates', near(crop.screenCenterX, pointerX) && near(crop.screenCenterY, pointerY));
  ok('Loupe crop center in media space matches pointer', near(crop.sourceX + crop.sourceWidth / 2, pointerX) && near(crop.sourceY + crop.sourceHeight / 2, pointerY));
  ok('Loupe magnification is exactly 3.5', near(crop.magnification, 3.5));
  ok('Loupe source width is loupeDiameter / 3.5 (51.4285)', near(crop.sourceWidth, 180 / 3.5));
}

console.log('── ViewportMatrix: Bounding Box Transformations ──');
{
  const vmBox = createViewportMatrix({
    containerWidth: 1000,
    containerHeight: 1000,
    mediaWidth: 2000,
    mediaHeight: 2000,
    zoom: 1.0,
    pan: { x: 0, y: 0 },
  });

  // Pixel bbox [x0, y0, x1, y1]
  const screenBox = vmBox.mediaBoxToScreenBox([200, 400, 600, 800]);
  // Effective scale = 0.5
  ok('mediaBox [200,400,600,800] maps to screen left=100, top=200, w=200, h=200',
    near(screenBox.left, 100) && near(screenBox.top, 200) && near(screenBox.width, 200) && near(screenBox.height, 200));

  // Normalized 0..1 bbox
  const normBox = vmBox.mediaBoxToScreenBox([0.1, 0.2, 0.3, 0.4]);
  ok('normalized bbox [0.1, 0.2, 0.3, 0.4] maps to screen left=100, top=200, w=200, h=200',
    near(normBox.left, 100) && near(normBox.top, 200) && near(normBox.width, 200) && near(normBox.height, 200));
}

console.log('── ViewportMatrix: Split Comparison Wipe ──');
{
  const vmWipe = createViewportMatrix({
    containerWidth: 1000,
    containerHeight: 500,
    mediaWidth: 2000,
    mediaHeight: 1000,
    zoom: 1.0,
    pan: { x: 0, y: 0 },
  });

  const divider50 = vmWipe.getSplitDivider(50);
  ok('50% wipe maps to mediaX=1000, screenX=500', near(divider50.mediaX, 1000) && near(divider50.screenX, 500));

  const divider75 = vmWipe.getSplitDivider(75);
  ok('75% wipe maps to mediaX=1500, screenX=750', near(divider75.mediaX, 1500) && near(divider75.screenX, 750));
}

console.log('── Pose Classification: Euler Angle Bucketing ──');
{
  const frontal = classifyPoseBucket(5, -4, 2);
  ok('Frontal pose classified correctly', frontal.bucket === 'Frontal' && !frontal.isChallenging);

  const hardLeft = classifyPoseBucket(-52, 10, 0);
  ok('Hard Left pose (> -45 yaw) classified as challenging', hardLeft.bucket === 'Hard Left' && hardLeft.isChallenging);

  const hardRight = classifyPoseBucket(48, -2, 5);
  ok('Hard Right pose (> +45 yaw) classified as challenging', hardRight.bucket === 'Hard Right' && hardRight.isChallenging);

  const lookUp = classifyPoseBucket(0, 32, 0);
  ok('Looking Up pose (> +25 pitch) classified as challenging', lookUp.bucket === 'Looking Up' && lookUp.isChallenging);

  const lookDown = classifyPoseBucket(0, -30, 0);
  ok('Looking Down pose (< -25 pitch) classified as challenging', lookDown.bucket === 'Looking Down' && lookDown.isChallenging);
}

console.log('── LRU Frame Cache: In-Memory Scrubber Cache ──');
{
  const cache = new LruFrameCache(4);
  let closedCount = 0;
  const mockBitmap = (id) => ({
    id,
    width: 640,
    height: 360,
    close: () => { closedCount++; },
  });

  cache.set(10, mockBitmap(10));
  cache.set(11, mockBitmap(11));
  cache.set(12, mockBitmap(12));
  cache.set(13, mockBitmap(13));

  ok('Cache size is 4', cache.size === 4);
  ok('Cache retrieves frame 10', cache.get(10)?.id === 10);

  // Accessing 10 marks it as MRU. Now adding 14 should evict 11 (the oldest unaccessed)
  cache.set(14, mockBitmap(14));
  ok('Frame 11 evicted upon adding frame 14', !cache.has(11));
  ok('Frame 10 retained because it was accessed recently', cache.has(10));
  ok('Evicted bitmap close() was invoked', closedCount === 1);

  // Clear cache
  cache.clear();
  ok('Cache clear empties map', cache.size === 0);
  ok('All remaining bitmaps closed upon clear()', closedCount === 5);
}

console.log(`\nviewport-matrix-check: ${failures === 0 ? 'ALL CHECKS PASSED' : `${failures} CHECKS FAILED`}`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
