import process from 'node:process';
/**
 * Unit verification for NleTimeline helpers, snapping, zoom-to-cursor, and parameter keyframes.
 */
globalThis.window = globalThis;
globalThis.location = { origin: 'http://127.0.0.1:1', protocol: 'http:', host: '127.0.0.1:1' };

const {
  snapToFrame,
  calculateZoomToCursor,
  generateRulerTicks,
} = await import('../src/components/faceswap/timeline/timelineSnap.ts');

const {
  PARAMETER_CONFIGS,
  createNleKeyframe,
  sortAndNormalizeKeyframes,
  evaluateKeyframeValue,
  duplicateKeyframe,
} = await import('../src/components/faceswap/timeline/parameterKeyframes.ts');

const {
  getTrackColor,
  getFacesetColor,
  TRACK_PALETTE,
  NleTimeline,
} = await import('../src/components/faceswap/NleTimeline.tsx');

const {
  byteTrackRows,
  byteTrackHit,
} = await import('../src/components/faceswap/timeline/laneModel.js');

const {
  drawByteTrackLane,
  BYTE_TRACK_ROW_H,
} = await import('../src/components/faceswap/timeline/lanes.js');

const {
  ByteTrackLane,
} = await import('../src/components/faceswap/timeline/Lanes.jsx');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

console.log('── Magnetic Snapping Logic ────────────────────────────────────');

{
  const targets = [
    { frame: 100, type: 'playhead', label: 'Playhead' },
    { frame: 250, type: 'marker', label: 'Marker 1' },
    { frame: 500, type: 'trim_in', label: 'Trim In' },
  ];

  // Within threshold
  const res1 = snapToFrame(102, targets, 3);
  ok('snaps to frame 100 when candidate is 102 (within 3 frames)', res1.snapped === true && res1.frame === 100);
  ok('identifies snapped target as playhead', res1.snappedTo?.type === 'playhead');

  // Outside threshold
  const res2 = snapToFrame(105, targets, 3);
  ok('does not snap when candidate is 105 (beyond 3 frames)', res2.snapped === false && res2.frame === 105);

  // Snaps to marker
  const res3 = snapToFrame(249, targets, 3);
  ok('snaps to marker at 250 from 249', res3.snapped === true && res3.frame === 250);
}

console.log('── Zoom-to-Cursor Anchor Math ─────────────────────────────────');

{
  const maxFrames = 1000;
  const initialView = [0, 999];

  // Zoom in 2x centered at fraction 0.5 (middle)
  const zoomedMid = calculateZoomToCursor(initialView, 0.5, 0.5, maxFrames, 4);
  ok('zooms in span to ~500 frames', Math.abs((zoomedMid[1] - zoomedMid[0]) - 500) <= 2);
  ok('centers around frame 500', Math.abs((zoomedMid[0] + zoomedMid[1]) / 2 - 500) <= 2);

  // Zoom in centered at fraction 0.2 (frame ~200)
  const zoomed20 = calculateZoomToCursor(initialView, 0.5, 0.2, maxFrames, 4);
  const frameUnderCursor = zoomed20[0] + 0.2 * (zoomed20[1] - zoomed20[0]);
  ok('preserves anchor frame under cursor (~200) during zoom', Math.abs(frameUnderCursor - 200) <= 2);
}

console.log('── Adaptive Ruler Ticks (HH:MM:SS:FF) ─────────────────────────');

{
  // Full view (0 to 1000 frames at 30fps)
  const ticksFull = generateRulerTicks(0, 1000, 30);
  ok('generates ruler ticks across full span', ticksFull.length > 5);
  ok('first tick starts at or after 0', ticksFull[0].frame >= 0);
  ok('formats timecode with HH:MM:SS:FF pattern', /^\d{2}:\d{2}:\d{2}:\d{2}$/.test(ticksFull[0].timecode));

  // Zoomed in view (100 to 120 frames)
  const ticksZoomed = generateRulerTicks(100, 120, 30);
  ok('dense ticks when zoomed into 20 frames', ticksZoomed.length >= 5);
  ok('labels in exact frame format #F when span <= 40', ticksZoomed[0].label.startsWith('#'));
}

console.log('── Parameter Keyframes & Curves ───────────────────────────────');

{
  ok('supports Swap Opacity, Mask Feather, Yaw Erosion, Color Match',
    Boolean(PARAMETER_CONFIGS['Swap Opacity']) &&
    Boolean(PARAMETER_CONFIGS['Mask Feather']) &&
    Boolean(PARAMETER_CONFIGS['Yaw Erosion']) &&
    Boolean(PARAMETER_CONFIGS['Color Match'])
  );

  // Clamping test
  const kfClamped = createNleKeyframe('Swap Opacity', 50, 2.5, 'Linear', 100);
  ok('clamps Swap Opacity max value to 1.0', kfClamped.value === 1.0);
  ok('sets exact integer frame 50', kfClamped.frame_idx === 50);

  // Interpolation: Linear
  const kfsLinear = [
    createNleKeyframe('Swap Opacity', 10, 0.0, 'Linear', 100),
    createNleKeyframe('Swap Opacity', 20, 1.0, 'Linear', 100),
  ];
  const midLinear = evaluateKeyframeValue(kfsLinear, 'Swap Opacity', 15);
  ok('evaluates linear midpoint at exact 0.5', Math.abs(midLinear - 0.5) < 1e-4);

  // Interpolation: Hold
  const kfsHold = [
    createNleKeyframe('Swap Opacity', 10, 0.2, 'Hold', 100),
    createNleKeyframe('Swap Opacity', 20, 0.8, 'Hold', 100),
  ];
  const midHold = evaluateKeyframeValue(kfsHold, 'Swap Opacity', 15);
  ok('evaluates hold segment as constant 0.2 until frame 20', midHold === 0.2);
  const atEndHold = evaluateKeyframeValue(kfsHold, 'Swap Opacity', 20);
  ok('evaluates hold keyframe at frame 20 as 0.8', atEndHold === 0.8);

  // Interpolation: Ease In/Out
  const kfsEase = [
    createNleKeyframe('Swap Opacity', 10, 0.0, 'Ease In/Out', 100),
    createNleKeyframe('Swap Opacity', 30, 1.0, 'Ease In/Out', 100),
  ];
  const midEase = evaluateKeyframeValue(kfsEase, 'Swap Opacity', 20);
  ok('evaluates ease in/out midpoint as 0.5', Math.abs(midEase - 0.5) < 1e-4);
  const q1Ease = evaluateKeyframeValue(kfsEase, 'Swap Opacity', 15); // quarter way
  ok('ease in/out starts slow (quarter value < 0.25)', q1Ease < 0.25);

  // Duplicate / Copy
  const orig = createNleKeyframe('Mask Feather', 10, 25, 'Linear', 100);
  const dup = duplicateKeyframe(orig, 15, 100);
  ok('duplicates keyframe with frame offset 25', dup.frame_idx === 25);
  ok('retains value 25 and param Mask Feather', dup.value === 25 && dup.param === 'Mask Feather');
}

console.log('── Palette & Micro-Indicator Color Logic ──────────────────────');

{
  const col0 = getTrackColor(0);
  const col1 = getTrackColor(1);
  ok('assigns distinct colors from palette', col0 !== col1);
  ok('palette is non-empty', TRACK_PALETTE.length >= 8);

  const fszCol = getFacesetColor('actor_a.fsz');
  ok('derives deterministic faceset color', typeof fszCol === 'string' && fszCol.startsWith('#'));
}

console.log('── ByteTrack Multi-Track Lane Model & Canvas Helpers ──────────');

{
  const rawTracks = [
    { trackId: 0, label: 'Track #1', ranges: [[10, 50], [60, 100]], color: '#38bdf8' },
    { trackId: 1, label: 'Track #2', ranges: [[20, 80]], color: '#f472b6' },
  ];
  const rows = byteTrackRows(rawTracks, 200);
  ok('normalizes raw tracks into 2 byteTrack rows', rows.length === 2);
  ok('row 0 has 2 non-overlapping intervals', rows[0].ranges.length === 2);
  ok('hit-test inside range returns true', byteTrackHit(rows[0], 25) === true);
  ok('hit-test outside range returns false', byteTrackHit(rows[0], 55) === false);

  ok('drawByteTrackLane is callable function', typeof drawByteTrackLane === 'function');
  ok('BYTE_TRACK_ROW_H is defined positive number', typeof BYTE_TRACK_ROW_H === 'number' && BYTE_TRACK_ROW_H > 0);
  ok('ByteTrackLane component is defined', typeof ByteTrackLane === 'function');
}

if (failures > 0) {
  console.error(`\n${failures} checks failed.`);
  process.exit(1);
} else {
  console.log('\nAll NleTimeline logic checks passed cleanly!');
  process.exit(0);
}
