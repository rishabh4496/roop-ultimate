/**
 * Render check for ptsFrameMap.ts:
 * Tests VFR-safe PTS frame resolver, exact frame index math, decoding, binary search,
 * keyframes, brush strokes, and range lock structures.
 */
import assert from 'node:assert';
import {
  decodePtsTable,
  decodePtsTableAsync,
  resolveFrameFromTime,
  resolveTimeFromFrame,
  clampFrameIdx,
  frameToTimecode,
  timecodeToFrame,
  formatSeconds,
  createMaskBrushStroke,
  filterStrokesByFrame,
  createKeyframePoint,
  clampKeyframe,
  normalizeKeyframePoints,
  createRangeLockInterval,
  clampRangeLockInterval,
  PtsFrameMap,
  createPtsFrameMap,
} from '../src/utils/ptsFrameMap.ts';

let failures = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
  } catch (err) {
    failures++;
    console.error(`  ✗ ${name}:`, err.message);
  }
}

async function asyncTest(name, fn) {
  try {
    await fn();
    console.log(`  ✓ ${name}`);
  } catch (err) {
    failures++;
    console.error(`  ✗ ${name}:`, err.message);
  }
}

console.log('Running ptsFrameMap checks...');

// 1. Raw array decoding
test('decodes raw tuple array [[frame_idx, num, den], ...]', () => {
  const raw = [
    [0, 0, 1],
    [1, 1001, 24000],
    [2, 2002, 24000],
  ];
  const decoded = decodePtsTable(raw);
  assert.strictEqual(decoded.length, 3);
  assert.strictEqual(decoded[0].frame_idx, 0);
  assert.strictEqual(decoded[0].pts, 0);
  assert.strictEqual(decoded[1].frame_idx, 1);
  assert.strictEqual(decoded[1].num, 1001);
  assert.strictEqual(decoded[1].den, 24000);
  assert.strictEqual(decoded[1].pts, 1001 / 24000);
});

test('decodes array of timestamp numbers', () => {
  const timestamps = [0.0, 0.03333, 0.06666, 0.1];
  const decoded = decodePtsTable(timestamps);
  assert.strictEqual(decoded.length, 4);
  assert.strictEqual(decoded[2].frame_idx, 2);
  assert.strictEqual(decoded[2].pts, 0.06666);
});

test('decodes JSON string', () => {
  const jsonStr = JSON.stringify([
    [0, 0, 1],
    [1, 1001, 24000],
  ]);
  const decoded = decodePtsTable(jsonStr);
  assert.strictEqual(decoded.length, 2);
  assert.strictEqual(decoded[1].frame_idx, 1);
});

test('decodes base64/zlib encoded string synchronously', () => {
  // Sample generated from python: [[0, 0, 1], [1, 1001, 24000], [2, 2002, 24000]]
  const b64 = 'eNqLjjbQUQAiw1gdhWhDIG1gACSNTAwMDEAiRkC2gYERTCQWAOc5Cac=';
  const decoded = decodePtsTable(b64);
  assert.strictEqual(decoded.length, 3);
  assert.strictEqual(decoded[0].frame_idx, 0);
  assert.strictEqual(decoded[1].frame_idx, 1);
  assert.strictEqual(decoded[1].num, 1001);
  assert.strictEqual(decoded[1].den, 24000);
  assert.strictEqual(decoded[2].frame_idx, 2);
});

await asyncTest('decodes base64/zlib encoded string asynchronously', async () => {
  const b64 = 'eNqLjjbQUQAiw1gdhWhDIG1gACSNTAwMDEAiRkC2gYERTCQWAOc5Cac=';
  const decoded = await decodePtsTableAsync(b64);
  assert.strictEqual(decoded.length, 3);
  assert.strictEqual(decoded[1].frame_idx, 1);
});

// 2. Binary search & VFR-safe resolveFrameFromTime
test('resolves exact frame from time on 23.976 fps clip without drift', () => {
  // 100 frames at 23.976 (1001/24000)
  const entries = [];
  for (let i = 0; i < 100; i++) {
    entries.push([i, i * 1001, 24000]);
  }
  const ptsTable = decodePtsTable(entries);

  for (let i = 0; i < 100; i++) {
    const t = (i * 1001) / 24000;
    // Exact PTS
    assert.strictEqual(resolveFrameFromTime(t, ptsTable), i, `Failed at frame ${i} exact`);
    // Mid-frame PTS (should still resolve to frame i under floor mode)
    if (i < 99) {
      const midT = t + 0.01;
      assert.strictEqual(resolveFrameFromTime(midT, ptsTable), i, `Failed at frame ${i} mid`);
    }
  }
});

test('resolves VFR timestamps with uneven intervals', () => {
  const vfr = [
    { frame_idx: 0, pts: 0.0 },
    { frame_idx: 1, pts: 0.05 },
    { frame_idx: 2, pts: 0.20 }, // large gap
    { frame_idx: 3, pts: 0.22 }, // small gap
  ];
  assert.strictEqual(resolveFrameFromTime(0.0, vfr), 0);
  assert.strictEqual(resolveFrameFromTime(0.049, vfr), 0);
  assert.strictEqual(resolveFrameFromTime(0.05, vfr), 1);
  assert.strictEqual(resolveFrameFromTime(0.15, vfr), 1);
  assert.strictEqual(resolveFrameFromTime(0.20, vfr), 2);
  assert.strictEqual(resolveFrameFromTime(0.21, vfr), 2);
  assert.strictEqual(resolveFrameFromTime(0.22, vfr), 3);
  assert.strictEqual(resolveFrameFromTime(1.00, vfr), 3); // out of bounds clamped to last
});

// 3. resolveTimeFromFrame exact seconds
test('resolves exact seconds from frame index', () => {
  const entries = [
    [0, 0, 1],
    [1, 1001, 24000],
    [2, 2002, 24000],
  ];
  const ptsTable = decodePtsTable(entries);
  assert.strictEqual(resolveTimeFromFrame(0, ptsTable), 0);
  assert.strictEqual(resolveTimeFromFrame(1, ptsTable), 1001 / 24000);
  assert.strictEqual(resolveTimeFromFrame(2, ptsTable), 2002 / 24000);
});

// 4. Clamping and integer helper structures
test('clampFrameIdx ensures safe finite integers', () => {
  assert.strictEqual(clampFrameIdx(3.7), 4);
  assert.strictEqual(clampFrameIdx(-5), 0);
  assert.strictEqual(clampFrameIdx(15, 0, 10), 9);
  assert.strictEqual(clampFrameIdx(NaN), 0);
  assert.strictEqual(clampFrameIdx('42'), 42);
});

test('createMaskBrushStroke clamps frame_idx and sanitizes stroke', () => {
  const stroke = createMaskBrushStroke({
    frame_idx: 12.3,
    size: 25.6,
    points: [{ x: 10, y: 20 }],
  }, 100);
  assert.strictEqual(stroke.frame_idx, 12);
  assert.strictEqual(stroke.size, 26);
  assert.strictEqual(stroke.tool, 'brush');
  assert.strictEqual(stroke.points.length, 1);
});

test('filterStrokesByFrame correctly filters by exact integer frame', () => {
  const strokes = [
    createMaskBrushStroke({ frame_idx: 1, size: 10, points: [] }),
    createMaskBrushStroke({ frame_idx: 2, size: 10, points: [] }),
    createMaskBrushStroke({ frame_idx: 1, size: 20, points: [] }),
  ];
  const filtered = filterStrokesByFrame(strokes, 1);
  assert.strictEqual(filtered.length, 2);
  assert.strictEqual(filterStrokesByFrame(strokes, 3).length, 0);
});

test('createKeyframePoint and normalizeKeyframePoints ensure sorted exact frames', () => {
  const kf1 = createKeyframePoint({ frame_idx: 5.4, value: 0.8 }, 10);
  assert.strictEqual(kf1.frame_idx, 5);

  const list = [
    createKeyframePoint({ frame_idx: 10 }),
    createKeyframePoint({ frame_idx: 2 }),
    createKeyframePoint({ frame_idx: 2, value: 0.5 }), // duplicate frame should be deduplicated
  ];
  const normalized = normalizeKeyframePoints(list, 100);
  assert.strictEqual(normalized.length, 2);
  assert.strictEqual(normalized[0].frame_idx, 2);
  assert.strictEqual(normalized[1].frame_idx, 10);
});

test('createRangeLockInterval and clampRangeLockInterval enforce valid frame intervals', () => {
  const interval = createRangeLockInterval({
    track_id: 1,
    frame_start: 10.2,
    frame_end: 25.8,
  }, 50);
  assert.strictEqual(interval.frame_start, 10);
  assert.strictEqual(interval.frame_end, 26);

  const clamped = clampRangeLockInterval(interval, 15, 21);
  assert.strictEqual(clamped.frame_start, 15);
  assert.strictEqual(clamped.frame_end, 20);
});

test('timecode formatting and parsing roundtrip', () => {
  const tc = frameToTimecode(75, 25);
  assert.strictEqual(tc, '00:00:03:00');
  assert.strictEqual(timecodeToFrame('00:00:03:00', 25), 75);
  assert.strictEqual(formatSeconds(65.123), '01:05.123');
});

// 5. PtsFrameMap class container
test('PtsFrameMap class container exposes methods and properties', () => {
  const map = createPtsFrameMap([
    [0, 0, 1],
    [1, 1001, 24000],
    [2, 2002, 24000],
  ]);
  assert.strictEqual(map.length, 3);
  assert.strictEqual(map.resolveFrameFromTime(1001 / 24000), 1);
  assert.strictEqual(map.resolveTimeFromFrame(2), 2002 / 24000);
  assert.strictEqual(map.hasFrame(1), true);
  assert.strictEqual(map.hasFrame(99), false);
  assert.strictEqual(map.clampFrame(50), 2);
});

globalThis.__RENDER_CHECK_FAILURES__ = failures;
if (failures === 0) {
  console.log('All ptsFrameMap checks passed cleanly!');
} else {
  console.error(`${failures} check(s) failed`);
}
