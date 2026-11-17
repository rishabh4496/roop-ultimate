import process from 'node:process';
/**
 * Unit verification for PreviewViewport logic and PTS frame / parity integration.
 */
globalThis.window = globalThis;
globalThis.location = { origin: 'http://127.0.0.1:1', protocol: 'http:', host: '127.0.0.1:1' };

const {
  resolveFrameFromTime,
  resolveTimeFromFrame,
  clampFrameIdx,
  frameToTimecode,
  createMaskBrushStroke,
  filterStrokesByFrame,
  createPtsFrameMap,
} = await import('../src/utils/ptsFrameMap.ts');

const {
  ViewportMatrix,
  createViewportMatrix,
  LruFrameCache,
  registerVideoFrameSync,
} = await import('../src/components/faceswap/viewportMatrix.ts');

const { bridgePreviewExportParity } = await import('../src/components/faceswap/runBarSync.ts');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

console.log('── Preview Export Parity Logic ───────────────────────────────');

{
  // 1. Telemetry with tracks_source === 'trajectory_plan'
  const stateVerified = bridgePreviewExportParity({ tracks_source: 'trajectory_plan' }, { available: true, faces: 150 });
  ok('parity is verified when tracks_source === trajectory_plan and summary is available', stateVerified.isParityVerified === true);
  ok('badge tone is verified', stateVerified.badgeTone === 'verified');
  ok('badge text confirms TrajectoryPlan', stateVerified.badgeText.includes('TrajectoryPlan'));

  // 2. Fallback / ad-hoc telemetry
  const stateFallback = bridgePreviewExportParity({ tracks_source: 'none' }, null);
  ok('parity is unverified when tracks_source is none', stateFallback.isParityVerified === false);
  ok('badge tone is idle for none', stateFallback.badgeTone === 'idle');

  // 3. Telemetry with trajectory_plan_active
  const stateActive = bridgePreviewExportParity({ tracks_source: 'trajectory_plan', trajectory_plan_active: true }, null);
  ok('active trajectory plan telemetry confirms verified parity', stateActive.isParityVerified === true);
}

console.log('── VFR-safe PTS Frame Math & Scrubber ────────────────────────');

{
  // Sample 23.976 fps table entries: [[frame, num, den], ...]
  const samplePts = [
    [0, 0, 24000],
    [1, 1001, 24000],
    [2, 2002, 24000],
    [3, 3003, 24000],
    [4, 4004, 24000],
  ];

  const frame0 = resolveFrameFromTime(0.0, samplePts, 23.976);
  ok('time 0.0s resolves to exact integer frame 0', frame0 === 0 && Number.isInteger(frame0));

  const frame2 = resolveFrameFromTime(2002 / 24000, samplePts, 23.976);
  ok('time of frame 2 resolves to exact integer frame 2', frame2 === 2 && Number.isInteger(frame2));

  const time2 = resolveTimeFromFrame(2, samplePts, 23.976);
  ok('frame 2 resolves to exact timestamp 2002/24000', Math.abs(time2 - (2002 / 24000)) < 1e-6);

  const tc = frameToTimecode(24, 24);
  ok('frame 24 at 24fps formats to 00:00:01:00', tc === '00:00:01:00');
}

console.log('── Mask Brush Strokes Exact Integer Frame ────────────────────');

{
  const stroke1 = createMaskBrushStroke({
    frame_idx: 42.7, // float input must be clamped to integer
    tool: 'brush',
    size: 30,
    points: [{ x: 100, y: 100 }],
  }, 100);

  ok('stroke frame_idx is strictly integer', stroke1.frame_idx === 43 && Number.isInteger(stroke1.frame_idx));

  const strokes = [
    stroke1,
    createMaskBrushStroke({ frame_idx: 10, tool: 'brush', size: 20, points: [] }),
    createMaskBrushStroke({ frame_idx: 43, tool: 'eraser', size: 25, points: [] }),
  ];

  const filtered43 = filterStrokesByFrame(strokes, 43);
  ok('filterStrokesByFrame returns only strokes matching exact frame 43', filtered43.length === 2 && filtered43.every((s) => s.frame_idx === 43));
}

console.log('── Unified ViewportMatrix Coordinate Math ────────────────────');

{
  const vm = createViewportMatrix({
    containerWidth: 1280,
    containerHeight: 720,
    mediaWidth: 1920,
    mediaHeight: 1080,
    zoom: 2.0,
    pan: { x: 50, y: -30 },
  });

  // 1. mediaToScreen <-> screenToMedia round-trip
  const screenPt = vm.mediaToScreen(960, 540);
  const mediaPt = vm.screenToMedia(screenPt.x, screenPt.y);
  ok('mediaToScreen / screenToMedia round-trip at zoom 2.0', Math.abs(mediaPt.x - 960) < 1e-4 && Math.abs(mediaPt.y - 540) < 1e-4);
  ok('screenToMedia correctly marks point onMedia: true', mediaPt.onMedia === true);

  // 2. 3.5x Magnifier Loupe crop math
  const loupeCrop = vm.getLoupeCrop(screenPt.x, screenPt.y, 180, 3.5);
  ok('loupeCrop centered at screen coordinates', Math.abs(loupeCrop.screenCenterX - screenPt.x) < 1e-4);
  ok('loupeCrop source center matches media coordinates (960, 540)', Math.abs(loupeCrop.sourceX + loupeCrop.sourceWidth / 2 - 960) < 1e-4);
  ok('loupeCrop magnification is 3.5', loupeCrop.magnification === 3.5);

  // 3. Split Comparison Wipe & Alpha Blend layout
  const wipeLayout = vm.getComparisonLayout('slider', 60, 0);
  ok('comparison layout returns isSplit: true for slider', wipeLayout.isSplit === true);
  ok('comparison layout computes 60% wipePctClamped', wipeLayout.divider.wipePctClamped === 60);
  ok('comparison layout computes mediaX at 60% of 1920 (1152)', Math.abs(wipeLayout.divider.mediaX - 1152) < 1e-4);

  const blendLayout = vm.getComparisonLayout('blend', 40);
  ok('comparison layout returns isBlend: true and blendAlpha 0.4', blendLayout.isBlend === true && Math.abs(blendLayout.blendAlpha - 0.4) < 1e-4);

  // 4. Floating PiP mini-viewport child matrix (Stage 26)
  const pipMatrix = vm.getPipChildMatrix(280, 158);
  ok('pipMatrix inherits media dimensions', pipMatrix.metrics.mediaWidth === 1920 && pipMatrix.metrics.mediaHeight === 1080);
  ok('pipMatrix container dimensions match PiP window', pipMatrix.metrics.containerWidth === 280 && pipMatrix.metrics.containerHeight === 158);

  // 5. Snapshot slots A/B/C/D quadrant layout (Stage 23)
  const quadA = vm.getSlotQuadrant(0);
  const quadD = vm.getSlotQuadrant(3);
  ok('slot A is top-left quadrant', quadA.slotId === 'A' && quadA.screenLeft === 0 && quadA.screenTop === 0);
  ok('slot D is bottom-right quadrant', quadD.slotId === 'D' && quadD.screenLeft === 640 && quadD.screenTop === 360);
}

console.log('── In-Memory ImageBitmap LRU Cache ───────────────────────────');

{
  const cache = new LruFrameCache(3);
  let closedBitmaps = 0;
  const mockBmp = (id) => ({ id, close: () => { closedBitmaps++; } });

  cache.set(1, mockBmp(1));
  cache.set(2, mockBmp(2));
  cache.set(3, mockBmp(3));
  ok('cache holds 3 entries', cache.size === 3);

  // Touch frame 1 to make it most recently used
  cache.get(1);

  // Insert frame 4, evicting frame 2 (oldest)
  cache.set(4, mockBmp(4));
  ok('cache size remains 3 after eviction', cache.size === 3);
  ok('frame 2 was evicted', !cache.has(2));
  ok('evicted bitmap close() was executed', closedBitmaps === 1);
  ok('frame 1 was preserved', cache.has(1));
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
} else {
  console.log('\nAll PreviewViewport & PTS integration checks PASSED');
}
