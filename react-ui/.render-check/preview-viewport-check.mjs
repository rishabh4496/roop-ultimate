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

globalThis.__RENDER_CHECK_FAILURES__ = failures;
if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
} else {
  console.log('\nAll PreviewViewport & PTS integration checks PASSED');
}
