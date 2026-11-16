/**
 * Render Check: NleTimeline component and interface verification.
 *
 * Verifies:
 *   1. Exports: ByteTrackLane, WarningTick, RangeLockInterval, KeyframePoint, SuggestedFszAssignment, NleTimeline.
 *   2. Support for displaying ByteTrack lanes Track #1..#N.
 *   3. Pose, occlusion, gap warning ticks rendering and time lookup.
 *   4. Suggested .fsz assignments populated from 1 Pre-Scan.
 *   5. Range Lock interval generation and update triggers.
 *   6. Exact integer frame math and PTS frame resolver integration.
 */

import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  NleTimeline,
  TRACK_PALETTE,
  getTrackColor,
} from '../src/components/faceswap/NleTimeline.tsx';
import {
  createPtsFrameMap,
  createRangeLockInterval,
  clampFrameIdx,
  frameToTimecode,
  formatSeconds,
} from '../src/utils/ptsFrameMap.ts';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

console.log('── NleTimeline Interfaces & Component Verification ─────────────────');

// 1. Verify Exports and Helper Functions
ok('NleTimeline component is exported', typeof NleTimeline === 'function');
ok('TRACK_PALETTE is an array with at least 8 colors', Array.isArray(TRACK_PALETTE) && TRACK_PALETTE.length >= 8);
ok('getTrackColor returns correct palette color', getTrackColor(0) === TRACK_PALETTE[0] && getTrackColor(1) === TRACK_PALETTE[1]);
ok('getTrackColor respects customColor', getTrackColor(0, '#123456') === '#123456');

// 2. Mock Data for 1 Pre-Scan outputs
const mockTrackLanes = [
  {
    trackId: 0,
    label: 'Track #1',
    firstFrame: 10,
    lastFrame: 120,
    totalFrames: 111,
    ranges: [[10, 50], [60, 120]],
    sourceIndex: 0,
    facesetId: 'alice_01',
    facesetPath: '/faces/alice.fsz',
    suggestedFsz: 'alice.fsz',
    via: 'pre_scan',
    active: true,
    color: '#38bdf8',
  },
  {
    trackId: 1,
    label: 'Track #2',
    firstFrame: 30,
    lastFrame: 200,
    totalFrames: 171,
    ranges: [[30, 200]],
    sourceIndex: 1,
    facesetId: 'bob_02',
    facesetPath: null,
    suggestedFsz: 'bob_02.fsz',
    via: 'pre_scan',
    active: true,
    color: '#34d399',
  },
];

const mockWarningTicks = [
  {
    frame: 25,
    trackId: 0,
    type: 'pose',
    severity: 'warning',
    message: 'Extreme head angle on Track #1: yaw 48°',
    value: 48,
  },
  {
    frame: 55,
    trackId: 0,
    type: 'gap',
    severity: 'info',
    message: 'Tracking gap between 50 and 60',
    value: 9,
  },
  {
    frame: 80,
    trackId: 1,
    type: 'occlusion',
    severity: 'warning',
    message: 'Heavy occlusion on Track #2: 42% occluded',
    value: 42,
  },
];

const mockSuggestedAssignments = [
  {
    trackId: 0,
    sourceIndex: 0,
    facesetId: 'alice_01',
    facesetPath: '/faces/alice.fsz',
    name: 'Alice',
    suggestedFileName: 'alice.fsz',
    confidence: 1.0,
  },
  {
    trackId: 1,
    sourceIndex: 1,
    facesetId: 'bob_02',
    facesetPath: null,
    name: 'Bob',
    suggestedFileName: 'bob_02.fsz',
    confidence: 0.85,
  },
];

const mockRangeLockIntervals = [
  createRangeLockInterval({
    id: 'rl_0_10_50',
    track_id: 0,
    frame_start: 10,
    frame_end: 50,
    label: 'Track #1 [10–50]',
    faceset_id: 'alice_01',
    source_index: 0,
    color: '#38bdf8',
    locked: true,
  }),
];

// 3. Render Component to Static Markup
const html = renderToStaticMarkup(
  React.createElement(NleTimeline, {
    maxFrames: 300,
    fps: 30,
    frame: 25,
    trackLanes: mockTrackLanes,
    warningTicks: mockWarningTicks,
    suggestedAssignments: mockSuggestedAssignments,
    rangeLockIntervals: mockRangeLockIntervals,
    selectedTrackId: 0,
    trimIn: 5,
    trimOut: 280,
    showWarningTicks: true,
    showSuggestedFsz: true,
  })
);

ok('renders timeline root with testid', html.includes('data-testid="nle-timeline"'));
ok('renders Step 1 Pre-Scan badge with track count', html.includes('1 Pre-Scan') && html.includes('2 Tracks'));
ok('renders Step 2 Range Lock badge with active count', html.includes('2 Range Lock') && html.includes('1 Interval'));
ok('renders Timecode and Frame readout', html.includes('#25') && html.includes('/ 299'));
ok('renders Trim In and Out markers', html.includes('IN:') && html.includes('#5') && html.includes('OUT:') && html.includes('#280'));

// 4. Verify Track Lanes & Suggested .fsz rendering
ok('renders Track #1 lane with testid', html.includes('data-testid="lane-track-0"'));
ok('renders Track #2 lane with testid', html.includes('data-testid="lane-track-1"'));
ok('renders suggested .fsz assignment for Track #1 (alice.fsz)', html.includes('alice.fsz'));
ok('renders suggested .fsz assignment for Track #2 (bob_02.fsz)', html.includes('bob_02.fsz'));
ok('renders LOCKED badge for Range Locked interval', html.includes('LOCKED'));

// 5. Verify Warning Ticks rendering
ok('renders warning ticks section with count (3)', html.includes('Warnings (3)'));
ok('renders pose warning tooltip message', html.includes('Extreme head angle on Track #1'));
ok('renders occlusion warning tooltip message', html.includes('Heavy occlusion on Track #2'));

// 6. Verify Exact Math and PtsFrameMap integration
const ptsTable = [
  [0, 0, 1],
  [1, 1001, 24000],
  [2, 2002, 24000],
  [3, 3003, 24000],
];
const map = createPtsFrameMap(ptsTable, 23.976);
ok('frame 0 resolves to 0 seconds', Math.abs(map.resolveTimeFromFrame(0)) < 1e-6);
ok('exact integer clamping eliminates fractional drift', clampFrameIdx(24.9999, 0, 100) === 25);
ok('frameToTimecode matches SMPTE specs', frameToTimecode(24, 24) === '00:00:01:00');

console.log('');
if (failures === 0) {
  console.log('ALL GREEN: NleTimeline checks passed cleanly!');
  process.exit(0);
} else {
  console.error(`FAILED: ${failures} check(s) failed`);
  process.exit(1);
}
