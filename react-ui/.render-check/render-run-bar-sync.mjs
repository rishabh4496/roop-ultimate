/**
 * Verification check for Run Bar Sync, Batch Matrix Bridge & Preview Export Parity.
 */

import process from 'node:process';

globalThis.window = {
  location: { origin: 'http://127.0.0.1:8000' },
};

const R = await import('../src/components/faceswap/runBarSync.ts');
const BM = await import('../src/components/faceswap/batchMatrix.js');

let passed = 0;
let failed = 0;

function ok(desc, cond, extra) {
  if (cond) {
    passed++;
    console.log(`  PASS  ${desc}`);
  } else {
    failed++;
    console.error(`  FAIL  ${desc}`, extra || '');
  }
}

console.log('── formatTrackLanes & suggested assignments ────────────────────────');
{
  const spans = {
    available: true,
    tracks: [
      {
        track: 0,
        first: 10,
        last: 120,
        frames: 111,
        ranges: [[10, 50], [60, 120]],
        source_index: 0,
        faceset_id: 'face_alice',
        faceset_path: '/faces/alice.fsz',
        via: 'pre_scan',
        reason: 'matched',
      },
      {
        track: 1,
        first: 25,
        last: 90,
        frames: 66,
        ranges: [[25, 90]],
        source_index: null,
        faceset_id: null,
        faceset_path: null,
        via: 'none',
        reason: 'unbound',
      },
    ],
  };

  const sources = [
    { id: 'face_alice', name: 'Alice' },
    { id: 'face_bob', name: 'Bob' },
  ];

  const lanes = R.formatTrackLanes(spans, sources);
  ok('translates 2 track spans into 2 ByteTrack lanes', lanes.length === 2);
  ok('lane 0 label is Track #1 with correct frame bounds', lanes[0].label === 'Track #1' && lanes[0].firstFrame === 10 && lanes[0].lastFrame === 120);
  ok('lane 0 suggestedFsz is alice.fsz', lanes[0].suggestedFsz === 'alice.fsz');
  ok('lane 1 gets suggested assignment from available sources', lanes[1].suggestedFsz === 'Bob.fsz' && lanes[1].sourceIndex === 1);

  const assignments = R.deriveSuggestedAssignments(spans, sources);
  ok('derives suggested assignments for both tracks', assignments.length === 2 && assignments[0].suggestedFileName === 'Alice.fsz');
}

console.log('── extractWarningTicks (pose, occlusion, gaps) ─────────────────────');
{
  const spans = {
    available: true,
    tracks: [
      {
        track: 0,
        first: 1,
        last: 100,
        frames: 80,
        ranges: [[1, 40], [55, 100]], // 14 frame gap
      },
    ],
  };

  const telemetryFaces = [
    {
      frame_idx: 15,
      track_id: 0,
      pose: { yaw: 55, pitch: 10, roll: 0 }, // extreme yaw
      occlusion_coverage_pct: 10,
    },
    {
      frame_idx: 20,
      track_id: 0,
      pose: { yaw: 5, pitch: 5, roll: 0 },
      occlusion_coverage_pct: 65, // heavy occlusion
    },
  ];

  const ticks = R.extractWarningTicks(spans, null, telemetryFaces);
  ok('identifies tracking gap warning tick', ticks.some((t) => t.type === 'gap' && t.frame === 41));
  ok('identifies extreme pose angle tick', ticks.some((t) => t.type === 'pose' && t.frame === 15));
  ok('identifies heavy occlusion tick', ticks.some((t) => t.type === 'occlusion' && t.frame === 20));

  const hardYawTick = ticks.find((t) => t.type === 'pose' && t.frame === 15);
  ok('hard yaw (|yaw| > 50) marked with warning severity', hardYawTick?.severity === 'warning' && hardYawTick?.value > 50);

  const occlusionTick = ticks.find((t) => t.type === 'occlusion' && t.frame === 20);
  ok('foreign occlusion (> 25%) marked with warning severity', occlusionTick?.severity === 'warning' && occlusionTick?.value > 25);
}

console.log('── bridgePreviewExportParity ───────────────────────────────────────');
{
  const telemWithPlan = {
    tracks_source: 'trajectory_plan',
  };
  const summaryActive = {
    available: true,
    faces: 150,
    first_frame: 1,
    last_frame: 200,
    track_ids: [0, 1],
  };

  const verified = R.bridgePreviewExportParity(telemWithPlan, summaryActive);
  ok('parity verified when tracks_source is trajectory_plan and cache active', verified.isParityVerified === true);
  ok('badge indicates verified state', verified.badgeTone === 'verified' && /100% Locked/.test(verified.badgeText));

  const telemNone = { tracks_source: 'none' };
  const unverified = R.bridgePreviewExportParity(telemNone, summaryActive);
  ok('parity unverified when tracks_source is none', unverified.isParityVerified === false);
}

console.log('── Range Lock to Batch Matrix Synchronization ──────────────────────');
{
  const targets = [{ name: 'test_clip.mp4', frames: 300, start_frame: 1, end_frame: 300 }];
  const sources = [{ id: 'src_1', name: 'Face 1' }, { id: 'src_2', name: 'Face 2' }];
  const lanes = [
    {
      trackId: 0,
      label: 'Track #1',
      firstFrame: 10,
      lastFrame: 100,
      totalFrames: 91,
      ranges: [[10, 100]],
      sourceIndex: 0,
      facesetId: 'src_1',
      facesetPath: null,
      suggestedFsz: 'face_1.fsz',
      via: 'pre_scan',
      active: true,
    },
  ];

  const intervals = R.convertTrackLanesToRangeLockIntervals(lanes);
  ok('converts lanes into RangeLockInterval objects', intervals.length === 1 && intervals[0].frame_start === 10 && intervals[0].frame_end === 100);

  const synced = R.syncRangeLockToBatchMatrix({
    targets,
    targetIndex: 0,
    activeIntervals: intervals,
    sourceFacesInfo: sources,
  });

  ok('syncRangeLockToBatchMatrix stages jobs without error', synced.error === null && synced.jobs.length === 1);
  ok('staged job carries target and correct frame bounds', synced.jobs[0].target_name === 'test_clip.mp4' && synced.jobs[0].frame_start === 10 && synced.jobs[0].frame_end === 100);
  ok('staged job includes range_lock metadata', synced.jobs[0].range_lock && synced.jobs[0].range_lock.track_id === 0);
}

console.log('── BatchMatrix stageRangeLockTracks direct ─────────────────────────');
{
  const targets = [{ name: 'clip.mp4', frames: 200 }];
  const sources = [{ id: 's1', name: 'S1' }];
  const res = BM.stageRangeLockTracks({
    targets,
    targetIndex: 0,
    tracks: [{ track: 0, first: 5, last: 50, ranges: [[5, 50]], source_index: 0 }],
    sourceFacesInfo: sources,
  });
  ok('stageRangeLockTracks derives intervals from tracks when activeIntervals omitted', res.jobs.length === 1 && res.jobs[0].frame_start === 5 && res.jobs[0].frame_end === 50);
}

console.log(failed === 0 ? `\nALL GREEN (${passed} passed)` : `\n${failed} checks FAILED (${passed} passed)`);
globalThis.__RENDER_CHECK_FAILURES__ = failed;
if (failed > 0) process.exit(1);
