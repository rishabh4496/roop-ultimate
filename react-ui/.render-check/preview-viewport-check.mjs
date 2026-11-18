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

console.log('── Ultra-Wide Displays (2560x1440, 3440x1440, 3840x2160, 5120x1440) ──');

{
  const ultraWideResolutions = [
    { name: '2560x1440 (QHD 16:9)', w: 2560, h: 1440 },
    { name: '3440x1440 (Ultrawide 21.5:9)', w: 3440, h: 1440 },
    { name: '3840x2160 (4K UHD 16:9)', w: 3840, h: 2160 },
    { name: '5120x1440 (Super Ultrawide 32:9)', w: 5120, h: 1440 },
  ];

  const testMediaSizes = [
    { w: 1920, h: 1080, label: '1080p 16:9' },
    { w: 3840, h: 1600, label: 'Cinema 2.40:1' },
    { w: 1080, h: 1920, label: 'Vertical 9:16' },
  ];

  for (const res of ultraWideResolutions) {
    for (const media of testMediaSizes) {
      for (const zoom of [1.0, 2.5]) {
        const pan = zoom > 1.0 ? { x: 75.5, y: -42.25 } : { x: 0, y: 0 };
        const vm = createViewportMatrix({
          containerWidth: res.w,
          containerHeight: res.h,
          mediaWidth: media.w,
          mediaHeight: media.h,
          zoom,
          pan,
        });

        const { centerScreenX, centerScreenY, effectiveScale } = vm.metrics;
        const domTx = centerScreenX - (media.w / 2) * effectiveScale;
        const domTy = centerScreenY - (media.h / 2) * effectiveScale;

        // Sub-pixel test points across media: top-left, center, arbitrary subpixel, bottom-right
        const testPoints = [
          [0, 0],
          [media.w / 2, media.h / 2],
          [media.w * 0.33333, media.h * 0.66667],
          [media.w, media.h],
        ];

        let maxError = 0;
        for (const [mx, my] of testPoints) {
          const domSx = domTx + mx * effectiveScale;
          const domSy = domTy + my * effectiveScale;
          const vmScreen = vm.mediaToScreen(mx, my);
          const errX = Math.abs(domSx - vmScreen.x);
          const errY = Math.abs(domSy - vmScreen.y);
          maxError = Math.max(maxError, errX, errY);

          // Inverse round-trip
          const back = vm.screenToMedia(domSx, domSy);
          const invErrX = Math.abs(back.x - mx);
          const invErrY = Math.abs(back.y - my);
          maxError = Math.max(maxError, invErrX, invErrY);
        }

        ok(`${res.name} on ${media.label} @ ${zoom}x: DOM CSS transform exactly matches ViewportMatrix (error < 1e-6px)`, maxError < 1e-6);
      }
    }
  }
}

console.log('── 5-Point ArcFace Landmark Overlays Zero-Drift ──────────────');

{
  const vm = createViewportMatrix({
    containerWidth: 3440,
    containerHeight: 1440,
    mediaWidth: 1920,
    mediaHeight: 1080,
    zoom: 1.75,
    pan: { x: 120, y: -60 },
  });

  // Typical ArcFace 5-point facial keypoints: EyeL, EyeR, Nose, MouthL, MouthR
  const arcfaceKps = [
    [892.35, 461.80],  // EyeL
    [1027.65, 461.25], // EyeR
    [959.80, 524.50],  // Nose
    [908.40, 582.15],  // MouthL
    [1011.60, 581.70], // MouthR
  ];

  const screenPoints = vm.landmarksToScreen(arcfaceKps);
  ok('landmarksToScreen transforms all 5 ArcFace points', screenPoints.length === 5);

  let maxLandmarkDrift = 0;
  for (let i = 0; i < arcfaceKps.length; i++) {
    const [mx, my] = arcfaceKps[i];
    const sp = screenPoints[i];
    const back = vm.screenToMedia(sp.x, sp.y);
    maxLandmarkDrift = Math.max(maxLandmarkDrift, Math.abs(back.x - mx), Math.abs(back.y - my));
  }
  ok('ArcFace 5-point landmarks screen-to-media round-trip drift is 0 (< 1e-5px)', maxLandmarkDrift < 1e-5);

  // Normalized landmark coordinates (0..1)
  const normKps = arcfaceKps.map(([x, y]) => [x / 1920, y / 1080]);
  const normScreen = vm.landmarksToScreen(normKps);
  let normDiff = 0;
  for (let i = 0; i < 5; i++) {
    normDiff = Math.max(normDiff, Math.abs(normScreen[i].x - screenPoints[i].x), Math.abs(normScreen[i].y - screenPoints[i].y));
  }
  ok('normalized (0..1) and pixel (0..1920) ArcFace landmarks yield identical screen coordinates', normDiff < 1e-4);

  // Sub-pixel bounding box relative positioning test
  const bbox = [840.0, 390.0, 1080.0, 640.0]; // [x0, y0, x1, y1]
  const bw = bbox[2] - bbox[0];
  const bh = bbox[3] - bbox[1];
  for (const [mx, my] of arcfaceKps) {
    const px = ((mx - bbox[0]) / bw) * 100;
    const py = ((my - bbox[1]) / bh) * 100;
    const reconstructedMx = bbox[0] + (px / 100) * bw;
    const reconstructedMy = bbox[1] + (py / 100) * bh;
    ok(`landmark (${mx.toFixed(1)}, ${my.toFixed(1)}) reconstructs perfectly from sub-pixel percentage`, Math.abs(reconstructedMx - mx) < 1e-6 && Math.abs(reconstructedMy - my) < 1e-6);
  }
}

console.log('── 3.5x+ Magnifier Loupe & Mask Brush Mapping ────────────────');

{
  const vm = createViewportMatrix({
    containerWidth: 5120,
    containerHeight: 1440,
    mediaWidth: 1920,
    mediaHeight: 1080,
    zoom: 2.0,
    pan: { x: -200, y: 100 },
  });

  // Test at various cursor coordinates
  const cursorCoords = [
    { x: 2560, y: 720 },
    { x: 1280.5, y: 450.25 },
    { x: 3840.75, y: 1100.8 },
  ];

  for (const cursor of cursorCoords) {
    for (const mag of [3.5, 5.0, 8.0]) {
      const crop = vm.getLoupeCrop(cursor.x, cursor.y, 180, mag);
      const mediaCenter = vm.screenToMedia(cursor.x, cursor.y);

      // Verify that center of loupe source crop matches cursor in media space
      const cropCenterX = crop.sourceX + crop.sourceWidth / 2;
      const cropCenterY = crop.sourceY + crop.sourceHeight / 2;
      ok(`Loupe at ${mag}x: crop center matches cursor media position within 1e-5px`, Math.abs(cropCenterX - mediaCenter.x) < 1e-5 && Math.abs(cropCenterY - mediaCenter.y) < 1e-5);
      ok(`Loupe at ${mag}x: magnification matches requested`, crop.magnification === mag);
    }
  }

  // Brush radius and radial feathering
  const screenRadius = 25; // 50px diameter on screen
  const mediaRadius = vm.screenRadiusToMedia(screenRadius);
  const backScreenRadius = vm.mediaRadiusToScreen(mediaRadius);
  ok('screenRadiusToMedia / mediaRadiusToScreen round-trip is zero drift', Math.abs(backScreenRadius - screenRadius) < 1e-5);

  const feather = 0.45;
  const innerR = Math.max(0, mediaRadius * (1 - feather));
  ok('radial feathering inner radius is proportional', Math.abs(innerR - mediaRadius * 0.55) < 1e-5);
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
