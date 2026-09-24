/**
 * CinematicTimeline 60 FPS Verification Suite
 * Tests canvas virtualization mathematics, SMPTE timecodes,
 * Zustand transient store, snapping engine, J-K-L shuttling,
 * In/Out point clamps, and parameter keyframe automation.
 */
import process from 'node:process';
import {
  formatSMPTE,
  useTimelineStore,
} from '../src/components/timeline/timelineStore.js';
import {
  frameToX,
  xToFrame,
  getAdaptiveTickStep,
} from '../src/components/timeline/timelineRenderer.js';

let checks = 0;
let failures = 0;

const ok = (name, cond, detail = '') => {
  checks += 1;
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

console.log('── SMPTE Timecode Formatter ──────────────────────────────');
{
  // 25 FPS checks
  ok('Frame 1 is 00:00:00:00', formatSMPTE(1, 25) === '00:00:00:00');
  ok('Frame 25 is 00:00:00:24', formatSMPTE(25, 25) === '00:00:00:24');
  ok('Frame 26 is 00:00:01:00', formatSMPTE(26, 25) === '00:00:01:00');
  ok('Frame 1501 is 00:01:00:00 (1 minute)', formatSMPTE(1501, 25) === '00:01:00:00');
  ok('Frame 90001 is 01:00:00:00 (1 hour)', formatSMPTE(90001, 25) === '01:00:00:00');

  // Sub-frame rounding
  ok('Sub-frame 26.4 rounds to 00:00:01:00', formatSMPTE(26.4, 25) === '00:00:01:00');
  ok('Sub-frame 26.6 rounds to 00:00:01:01', formatSMPTE(26.6, 25) === '00:00:01:01');

  // 30 FPS check
  ok('30 FPS: frame 31 is 00:00:01:00', formatSMPTE(31, 30) === '00:00:01:00');
}

console.log('── Virtualization Coordinate Transforms ───────────────────');
{
  const zoom = 2.5; // 2.5 px per frame
  const scrollLeft = 100; // px
  const frame = 42;

  // Frame to Pixel
  const x = frameToX(frame, zoom, scrollLeft);
  ok('frameToX formula is correct: (f - 1) * zoom - scrollLeft', x === (42 - 1) * 2.5 - 100);

  // Pixel to Frame
  const fInvert = xToFrame(x, zoom, scrollLeft);
  ok('xToFrame is exact inverse of frameToX', Math.abs(fInvert - frame) < 1e-9);

  // Visible frame slice calculation
  const canvasW = 1000;
  const startF = Math.max(1, Math.floor(xToFrame(0, zoom, scrollLeft)));
  const endF = Math.ceil(xToFrame(canvasW, zoom, scrollLeft));
  ok('Start frame is greater than 1', startF >= 1);
  ok('End frame covers visible width', endF > startF);
  ok('Visible frame range matches canvas width', Math.abs((endF - startF) * zoom - canvasW) < zoom * 2);

  // Adaptive ruler steps
  const stepZoomedIn = getAdaptiveTickStep(10.0); // 10 px/frame -> 10 frames = 100px
  const stepZoomedOut = getAdaptiveTickStep(0.1);  // 0.1 px/frame -> 1000 frames = 100px
  ok('Zoomed in picks small step', stepZoomedIn <= 10);
  ok('Zoomed out picks large step', stepZoomedOut >= 500);
}

console.log('── Zustand Transient Store & Playhead Scrubbing ─────────');
{
  const store = useTimelineStore.getState();

  // Initial state checks
  ok('Initial frame is within bounds', store.frame >= 1 && store.frame <= store.maxFrames);
  ok('Initial In <= Out', store.inPoint <= store.outPoint);

  // Direct setFrame without React diffing
  useTimelineStore.getState().setFrame(150, true);
  ok('setFrame updates frame directly in store', useTimelineStore.getState().frame === 150);

  // Sub-frame scrubbing precision (bypass snapping)
  useTimelineStore.getState().setFrame(150.375, true);
  ok('Sub-frame floating point precision supported', useTimelineStore.getState().frame === 150.375);

  // Bounds clamping
  useTimelineStore.getState().setFrame(-10, true);
  ok('Frame clamped at 1 on lower bound', useTimelineStore.getState().frame === 1);

  useTimelineStore.getState().setFrame(99999, true);
  ok('Frame clamped at maxFrames on upper bound', useTimelineStore.getState().frame === useTimelineStore.getState().maxFrames);
}

console.log('── Snapping Engine ───────────────────────────────────────');
{
  useTimelineStore.setState({
    sceneCuts: [100, 250, 400],
    inPoint: 20,
    outPoint: 500,
    zoom: 2.0, // 2 px per frame -> snap threshold (8px) = 4 frames
    snappingEnabled: true,
  });

  // Scrub near cut at 250: scrub to 252 (distance = 2 frames = 4px <= 8px threshold)
  useTimelineStore.getState().setFrame(252, false);
  ok('Snaps playhead to nearby scene cut at frame 250', useTimelineStore.getState().frame === 250);
  ok('Active snap line recorded', useTimelineStore.getState().activeSnapLine?.frame === 250);

  // Scrub far from cut at 250: scrub to 260 (distance = 10 frames = 20px > 8px)
  useTimelineStore.getState().setFrame(260, false);
  ok('Does not snap when outside threshold', useTimelineStore.getState().frame === 260);

  // Bypass snapping with Shift key (bypassSnapping = true)
  useTimelineStore.getState().setFrame(252, true);
  ok('Bypasses snap when bypassSnapping flag is true', useTimelineStore.getState().frame === 252);
}

console.log('── In / Out Clamps & Trimming ────────────────────────────');
{
  useTimelineStore.setState({ maxFrames: 1000, inPoint: 1, outPoint: 1000 });

  useTimelineStore.getState().setInPoint(50);
  ok('setInPoint clamps correctly', useTimelineStore.getState().inPoint === 50);

  useTimelineStore.getState().setOutPoint(300);
  ok('setOutPoint clamps correctly', useTimelineStore.getState().outPoint === 300);

  // Prevent In > Out
  useTimelineStore.getState().setInPoint(450);
  ok('In-point cannot exceed Out-point', useTimelineStore.getState().inPoint <= 300);

  // In / Out via playhead shortcuts [I] / [O]
  useTimelineStore.getState().setFrame(120, true);
  useTimelineStore.getState().setInPointToCurrent();
  ok('setInPointToCurrent sets In-point to playhead', useTimelineStore.getState().inPoint === 120);

  useTimelineStore.getState().setFrame(280, true);
  useTimelineStore.getState().setOutPointToCurrent();
  ok('setOutPointToCurrent sets Out-point to playhead', useTimelineStore.getState().outPoint === 280);

  useTimelineStore.getState().clearInOutRange();
  ok('clearInOutRange resets to [1, maxFrames]',
    useTimelineStore.getState().inPoint === 1 && useTimelineStore.getState().outPoint === 1000);
}

console.log('── J-K-L Shuttling Engine & Keyboard Step ─────────────────');
{
  useTimelineStore.setState({ isPlaying: false, shuttleSpeed: 0 });

  // L: Forward shuttle cycles: 1x -> 2x -> 4x -> 8x
  useTimelineStore.getState().shuttle('forward');
  ok('L press 1 sets shuttle to +1x', useTimelineStore.getState().shuttleSpeed === 1 && useTimelineStore.getState().isPlaying);

  useTimelineStore.getState().shuttle('forward');
  ok('L press 2 sets shuttle to +2x', useTimelineStore.getState().shuttleSpeed === 2);

  useTimelineStore.getState().shuttle('forward');
  ok('L press 3 sets shuttle to +4x', useTimelineStore.getState().shuttleSpeed === 4);

  useTimelineStore.getState().shuttle('forward');
  ok('L press 4 sets shuttle to +8x', useTimelineStore.getState().shuttleSpeed === 8);

  // K: Pause shuttle
  useTimelineStore.getState().shuttle('pause');
  ok('K pauses shuttle speed to 0 and stops playback',
    useTimelineStore.getState().shuttleSpeed === 0 && !useTimelineStore.getState().isPlaying);

  // J: Reverse shuttle cycles: -1x -> -2x -> -4x -> -8x
  useTimelineStore.getState().shuttle('reverse');
  ok('J press 1 sets shuttle to -1x', useTimelineStore.getState().shuttleSpeed === -1 && useTimelineStore.getState().isPlaying);

  useTimelineStore.getState().shuttle('reverse');
  ok('J press 2 sets shuttle to -2x', useTimelineStore.getState().shuttleSpeed === -2);

  useTimelineStore.getState().shuttle('pause');

  // Step 1 frame
  useTimelineStore.setState({ frame: 50 });
  useTimelineStore.getState().stepFrame(1);
  ok('stepFrame(+1) steps forward by 1 frame', useTimelineStore.getState().frame === 51);

  useTimelineStore.getState().stepFrame(-1);
  ok('stepFrame(-1) steps backward by 1 frame', useTimelineStore.getState().frame === 50);
}

console.log('── Parameter Keyframe Automation Curves ──────────────────');
{
  useTimelineStore.setState({
    curves: [
      {
        id: 'test_curve',
        name: 'Test Curve',
        min: 0.0,
        max: 1.0,
        keyframes: [{ frame: 1, value: 0.5 }],
      },
    ],
  });

  // Add keyframe
  useTimelineStore.getState().addKeyframe('test_curve', 100, 0.85);
  const c1 = useTimelineStore.getState().curves[0];
  ok('Keyframe added successfully', c1.keyframes.length === 2 && c1.keyframes[1].frame === 100 && c1.keyframes[1].value === 0.85);

  // Keyframes sorted chronologically
  useTimelineStore.getState().addKeyframe('test_curve', 50, 0.2);
  const c2 = useTimelineStore.getState().curves[0];
  ok('Keyframes maintain chronological order',
    c2.keyframes[0].frame === 1 && c2.keyframes[1].frame === 50 && c2.keyframes[2].frame === 100);

  // Update keyframe
  useTimelineStore.getState().updateKeyframe('test_curve', 1, 55, 0.4);
  const c3 = useTimelineStore.getState().curves[0];
  ok('updateKeyframe updates frame and value', c3.keyframes[1].frame === 55 && c3.keyframes[1].value === 0.4);

  // Delete keyframe
  useTimelineStore.getState().deleteKeyframe('test_curve', 1);
  const c4 = useTimelineStore.getState().curves[0];
  ok('deleteKeyframe removes target keyframe', c4.keyframes.length === 2 && c4.keyframes[1].frame === 100);
}

console.log(`\n${failures === 0 ? 'ALL GREEN' : 'FAILURES'}: ${checks - failures}/${checks} checks passed`);
process.exit(failures === 0 ? 0 : 1);
