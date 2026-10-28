/**
 * Stage 22: NLE Timeline Interchange (CMX3600 EDL / OTIO), Standalone VFX Matte Pass Export,
 * and Cryptographic Provenance Render Checks.
 */
import fs from 'node:fs';
import process from 'node:process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  ERR_NO_CUTS,
  ERR_NO_TARGETS,
  parseTimelineCutsText,
  stageTimelineCuts,
} from '../src/components/faceswap/batchMatrix.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

let total = 0;
let passed = 0;

function ok(name, condition) {
  total += 1;
  if (!condition) {
    console.error(`  FAIL  ${name}`);
  } else {
    passed += 1;
    console.log(`  PASS  ${name}`);
  }
}

console.log('── Stage 22: NLE Interchange, VFX Multi-Pass & Provenance Checks ──');

// 1. EDL Parsing in batchMatrix
const sampleEdl = `TITLE: TEST_ROOP_TIMELINE
FCM: NON-DROP FRAME

001  AX       V     C        00:00:00:00 00:01:00:00 00:00:00:00 00:01:00:00
* FROM CLIP NAME: source_reel.mp4
* TRACK: V1

002  SWAP     V     C        00:00:02:00 00:00:05:00 00:00:02:00 00:00:05:00
* FROM CLIP NAME: target_shot.mp4
* TRACK: V2
* SPEAKER: person_0

003  SWAP     V     C        00:00:06:00 00:00:08:15 00:00:06:00 00:00:08:15
* FROM CLIP NAME: target_shot.mp4
* TRACK: V2
* SPEAKER: person_1
`;

const edlCuts = parseTimelineCutsText(sampleEdl, 'timeline.edl');
ok('parseTimelineCutsText parses EDL cuts', Array.isArray(edlCuts) && edlCuts.length === 3);
ok('cut 2 is on Track V2', edlCuts[1]?.track === 'V2');
ok('cut 2 frame_in is 60 frames (00:00:02:00 at 30fps)', edlCuts[1]?.src_in === 60);
ok('cut 2 frame_out is 150 frames (00:00:05:00 at 30fps)', edlCuts[1]?.src_out === 150);
ok('cut 3 speaker is person_1 (id 1)', edlCuts[2]?.speaker_id === 1);

// 2. OTIO Parsing in batchMatrix
const sampleOtio = JSON.stringify({
  OTIO_SCHEMA: "Timeline.1",
  name: "Sample OTIO",
  tracks: {
    OTIO_SCHEMA: "Stack.1",
    children: [
      {
        OTIO_SCHEMA: "Track.1",
        name: "V1_Source",
        children: [
          {
            OTIO_SCHEMA: "Clip.1",
            name: "source_reel.mp4",
            source_range: {
              start_time: { value: 0, rate: 30 },
              duration: { value: 600, rate: 30 }
            }
          }
        ]
      },
      {
        OTIO_SCHEMA: "Track.1",
        name: "V2_FaceSwap",
        children: [
          {
            OTIO_SCHEMA: "Clip.1",
            name: "target_shot.mp4",
            source_range: {
              start_time: { value: 90, rate: 30 },
              duration: { value: 120, rate: 30 }
            },
            metadata: {
              roop: { speaker_cluster_id: "person_2" }
            }
          }
        ]
      }
    ]
  }
});

const otioCuts = parseTimelineCutsText(sampleOtio, 'timeline.otio.json');
ok('parseTimelineCutsText parses OTIO cuts', Array.isArray(otioCuts) && otioCuts.length === 1);
ok('OTIO cut start is 90', otioCuts[0]?.src_in === 90);
ok('OTIO cut end is 210 (90 + 120)', otioCuts[0]?.src_out === 210);
ok('OTIO cut speaker is person_2 (id 2)', otioCuts[0]?.speaker_id === 2);

// 3. stageTimelineCuts Staging Logic
const mockTargets = [
  { name: 'target_shot.mp4', frames: 1000, fps: 30.0 }
];
const mockSources = [
  { name: 'Alice', id: 'face_1' },
  { name: 'Bob', id: 'face_2' },
  { name: 'Charlie', id: 'face_3' }
];
const mockCreatePayload = (mappings, swapMode, overrides) => ({
  mappings,
  swapMode,
  ...overrides
});

const staged = stageTimelineCuts({
  targets: mockTargets,
  targetIndex: 0,
  cuts: edlCuts.filter((c) => c.track === 'V2'),
  sourceFacesInfo: mockSources,
  createJobPayload: mockCreatePayload,
});

ok('stageTimelineCuts produces staged jobs', staged.jobs.length === 2);
ok('first staged job has 1-based start frame 61 (src_in 60)', staged.jobs[0].frame_start === 61);
ok('first staged job has end frame 150', staged.jobs[0].frame_end === 150);
ok('first staged job has total frames 90', staged.jobs[0].total_frames === 90);
ok('first staged job targets target_shot.mp4', staged.jobs[0].target_name === 'target_shot.mp4');
ok('second staged job has speaker_id 1 (Bob)', staged.jobs[1].nle_cut?.speaker_id === 1);
ok('second staged job assigned to Bob', staged.jobs[1].source_name === 'Bob');

// 4. Edge Cases: Empty cuts or empty targets
const emptyCuts = stageTimelineCuts({
  targets: mockTargets,
  targetIndex: 0,
  cuts: [],
  sourceFacesInfo: mockSources,
  createJobPayload: mockCreatePayload,
});
ok('empty cuts returns ERR_NO_CUTS', emptyCuts.error === ERR_NO_CUTS);

const emptyTargets = stageTimelineCuts({
  targets: [],
  targetIndex: 0,
  cuts: edlCuts,
  sourceFacesInfo: mockSources,
  createJobPayload: mockCreatePayload,
});
ok('empty targets returns ERR_NO_TARGETS', emptyTargets.error === ERR_NO_TARGETS);

// 5. Codebase Wiring Checks
const batchSwapCode = fs.readFileSync(path.join(ROOT, 'src/components/BatchSwap.jsx'), 'utf-8');
ok('BatchSwap imports stageTimelineCuts', batchSwapCode.includes('stageTimelineCuts'));
ok('BatchSwap imports parseTimelineCutsText', batchSwapCode.includes('parseTimelineCutsText'));
ok('BatchSwap has Import EDL/OTIO button', batchSwapCode.includes('Import EDL/OTIO') || batchSwapCode.includes('Import EDL / OTIO'));
ok('BatchSwap has NLE Timeline Cut List card', batchSwapCode.includes('NLE Timeline Cut List'));

const faceSwapCode = fs.readFileSync(path.join(ROOT, 'src/components/FaceSwap.jsx'), 'utf-8');
ok('FaceSwap has InspectorGroup for VFX multi-pass', faceSwapCode.includes('id="vfx-multipass"'));
ok('FaceSwap binds vfx_multi_pass toggle', faceSwapCode.includes('vfx_multi_pass'));
ok('FaceSwap binds vfx_export_matte toggle', faceSwapCode.includes('vfx_export_matte'));
ok('FaceSwap binds vfx_export_crops toggle', faceSwapCode.includes('vfx_export_crops'));
ok('FaceSwap has Export EDL button', faceSwapCode.includes('Export EDL (CMX3600)'));
ok('FaceSwap has Export OTIO button', faceSwapCode.includes('Export OTIO (.json)'));

const settingsCode = fs.readFileSync(path.join(ROOT, 'src/components/Settings.jsx'), 'utf-8');
ok('Settings binds vfx_multi_pass', settingsCode.includes("bindToggle('vfx_multi_pass')"));
ok('Settings binds vfx_export_matte', settingsCode.includes("bindToggle('vfx_export_matte')"));
ok('Settings binds vfx_export_crops', settingsCode.includes("bindToggle('vfx_export_crops')"));
ok('Settings binds enable_provenance_signer', settingsCode.includes("bindToggle('enable_provenance_signer')"));
ok('Settings binds embed_provenance_watermark', settingsCode.includes("bindToggle('embed_provenance_watermark')"));
ok('Settings binds provenance_disclosure', settingsCode.includes("bind('provenance_disclosure'"));

console.log(`\n${passed} of ${total} checks passed.`);
if (passed === total) {
  console.log('ALL GREEN\n');
  process.exit(0);
} else {
  console.error('FAILURES DETECTED\n');
  process.exit(1);
}
