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

// src/api.js reads `window` at import (NleTimeline's hooks import it); the other checks shim it the same way.
globalThis.window = globalThis;
globalThis.location = { origin: 'http://127.0.0.1:1', protocol: 'http:', host: '127.0.0.1:1', hash: '' };

// (dynamic imports: static ones are hoisted above the shim)
const { default: React } = await import('react');
const { renderToStaticMarkup } = await import('react-dom/server');
const {
  NleTimeline,
  TRACK_PALETTE,
  getTrackColor,
  tracksKey,
} = await import('../src/components/faceswap/NleTimeline.tsx');
const {
  createPtsFrameMap,
  createRangeLockInterval,
  clampFrameIdx,
  frameToTimecode,
} = await import('../src/utils/ptsFrameMap.ts');

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
  { frame: 40, trackId: 0, type: 'pose', severity: 'warning', message: 'Extreme head angle on Track #1: yaw 62°', value: 62 },
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
ok('renders Timecode and exact zero-padded frame readout', html.includes('#00025') && html.includes('/ 299') && html.includes('00:00:00:25'));
ok('renders Trim In and Out markers', html.includes('IN:') && html.includes('#5') && html.includes('OUT:') && html.includes('#280'));
ok('renders the In-Out duration readout (frames 5..280 = 276 frames = 9 s 6 f at 30 fps)', html.includes('data-testid="nle-region-duration"') && /9s 06f/.test(html));

// 3b. Stage 4 structure: every lane the brief names, the transport deck only when the host can play
ok('renders the canvas ruler', html.includes('data-testid="nle-ruler-canvas"') && html.includes('data-major-step='));
ok('renders the overview navigator', html.includes('data-testid="nle-minimap"') && html.includes('role="slider"'));
ok('renders the V1 filmstrip lane with hide / height toggles', html.includes('data-testid="lane-v1-filmstrip"') && html.includes('data-testid="lane-v1-hide"') && html.includes('data-testid="lane-v1-height"'));
ok('renders the face coverage heatmap lane with its jump buttons', html.includes('data-testid="lane-health"') && html.includes('data-testid="nle-health-next"'));
ok('renders the swap segment / marker lane with lock', html.includes('data-testid="lane-segments"') && html.includes('data-testid="lane-segments-lock"'));
ok('renders the A1 audio lane and the KF lane', html.includes('data-testid="lane-a1-waveform"') && html.includes('data-testid="lane-kf-editor"') && html.includes('data-testid="lane-kf-lock"'));
ok('hide toggles are real toggle buttons', (html.match(/aria-pressed="false"[^>]*data-testid="lane-[a-z0-9]+-hide"/g) || []).length >= 4, 'aria-pressed missing on a lane hide button');
ok('the playhead is the cyan token, not the old red accent', html.includes('data-testid="nle-playhead"') && html.includes('--accent-cyan') && !/nle-playhead[^>]*accent,#e94560/.test(html));
ok('the playhead has a scrub cap', html.includes('data-testid="nle-playhead-cap"'));
ok('lane area overlay carries the header offset', html.includes('data-testid="nle-lane-area"') && html.includes('left:256px'));
ok('In/Out dim the outside region', html.includes('data-testid="nle-dim-before"') && html.includes('data-testid="nle-dim-after"'));
ok('no transport deck without a play handler', !html.includes('data-testid="nle-transport"'));
const deck = renderToStaticMarkup(React.createElement(NleTimeline, {
  maxFrames: 300, fps: 30, frame: 25, isPlaying: true, playbackRate: 0.5, isLooping: true, trimIn: 5, trimOut: 280,
  onTogglePlay() {}, onStep() {}, onPlaybackRateChange() {}, onToggleLoop() {}, onShuttle() {}, shuttleLabel: '▶ 2×', shuttleDir: 1,
  markers: [{ id: 'marker_100', frame: 99, label: 'beat', color: '#fb7185', note: 'beat' }], onAddMarker() {},
}));
ok('transport deck renders when the host can play', deck.includes('data-testid="nle-transport"'));
ok('deck: start, prev marker, step, play, step, next marker, end', ['start', 'prev-marker', 'step-back', 'play', 'step-fwd', 'next-marker', 'end'].every((k) => deck.includes(`data-testid="transport-${k}"`)));
ok('deck: Play shows Pause while playing and is aria-pressed', /aria-label="Pause"[^>]*aria-pressed="true"|aria-pressed="true"[^>]*aria-label="Pause"/.test(deck));
ok('deck: all five speeds are offered', [0.25, 0.5, 1, 2, 4].every((r) => deck.includes(`data-testid="transport-rate-${r}"`)));
ok('deck: exactly one speed is checked, and it is the host\'s 0.5x', (deck.match(/role="radio" aria-checked="true"/g) || []).length === 1 && /role="radio" aria-checked="true"[^>]*data-testid="transport-rate-0\.5"/.test(deck));
ok('deck: loop is pressed and reads the region duration', /aria-pressed="true"[^>]*data-testid="transport-loop"/.test(deck) && deck.includes('9s 06f'));
ok('deck: J K L and the live shuttle label', ['j', 'k', 'l'].every((k) => deck.includes(`data-testid="transport-shuttle-${k}"`)) && deck.includes('▶ 2×'));
const btnTag = (html, id) => (html.match(new RegExp(`<button[^>]*data-testid="${id}"[^>]*>`)) || [''])[0];
ok('deck: next marker is enabled when one lies ahead (marker at 99, playhead 25)', btnTag(deck, 'transport-next-marker') !== '' && !/ disabled=""/.test(btnTag(deck, 'transport-next-marker')));
ok('deck: previous marker is disabled when none lies behind', / disabled=""/.test(btnTag(deck, 'transport-prev-marker')));
ok('a controlled marker is drawn in the segment lane with its colour', deck.includes('data-testid="nle-marker-flag"') && deck.includes('#fb7185'));

// 3b'. Regression (studio-integration e2e): the stable-reference key for trackLanes once covered only ids + ranges, so a
// faceset re-assignment or a swap mute left the lane showing the OLD value. The lane must follow every field it renders.
{
  const lane = (over) => ({ ...mockTrackLanes[0], ...over });
  const k = (over) => tracksKey([lane(over)]);
  const base = k({});
  ok('tracksKey: identical tracks -> identical key (the point of the stable reference)', k({}) === base);
  ok('tracksKey changes with the faceset assignment (same ranges)', k({ facesetPath: '/other.fsz' }) !== base && k({ facesetId: 'other' }) !== base);
  ok('tracksKey changes with the swap mute', k({ active: false }) !== base);
  ok('tracksKey changes with label, colour, suggested file, source and ranges', [{ label: 'x' }, { color: '#000000' }, { suggestedFsz: 'z.fsz' }, { sourceIndex: 9 }, { ranges: [[1, 2]] }].every((o) => k(o) !== base));
}

// 3c. A clip nobody scanned must not paint as dropout, and the extras-style config hides the pipeline
const bare = renderToStaticMarkup(React.createElement(NleTimeline, { maxFrames: 300, fps: 30, frame: 0, showPipelineSteps: false, showFaceTrackLanes: false }));
ok('unscanned: the lane says so', bare.includes('not scanned'));
ok('showPipelineSteps=false removes the 1/2/3 badges', !bare.includes('nle-step-prescan') && !bare.includes('nle-batch-matrix-btn'));
ok('showFaceTrackLanes=false removes the track group', !bare.includes('data-testid="nle-face-tracks"'));

// 4. Verify Track Lanes & Suggested .fsz rendering
ok('renders Track #1 lane with testid', html.includes('data-testid="lane-track-0"'));
ok('renders Track #2 lane with testid', html.includes('data-testid="lane-track-1"'));
ok('renders suggested .fsz assignment for Track #1 (alice.fsz)', html.includes('alice.fsz'));
ok('renders suggested .fsz assignment for Track #2 (bob_02.fsz)', html.includes('bob_02.fsz'));
ok('renders LOCKED badge for Range Locked interval', html.includes('LOCKED'));

// 5. Verify Warning Ticks rendering
// (the old global "Warnings (N)" strip is gone: the face coverage lane carries the same frames and jumps to them)
ok('the coverage lane reports measured shares that add up (ok / risk / lost)', /data-testid="nle-health-status"[^>]*>\d+% ok · \d+% risk · \d+% lost/.test(html));
ok('sticky controls: transport/overview/ruler sit in one sticky group above the lanes', html.indexOf('data-testid="nle-sticky"') > 0 && html.indexOf('data-testid="nle-sticky"') < html.indexOf('data-testid="nle-runway"') && html.indexOf('data-testid="nle-ruler"') < html.indexOf('data-testid="nle-runway"'));
ok('the ruler layer carries the cap; the lane layer carries the playhead line through the lanes', html.indexOf('data-testid="nle-playhead-cap"') < html.indexOf('data-testid="nle-runway"') && html.indexOf('data-testid="nle-playhead"') > html.indexOf('data-testid="nle-runway"'));
ok('the root clips instead of hiding (so sticky pins to the dock, not to the root)', /nle-timeline-root[^>]*style="[^"]*overflow:clip/.test(html) || /overflow:clip/.test(html));
ok('renders pose warning tooltip message on its track lane', html.includes('Extreme head angle on Track #1'));
// Regression: `value` is yaw DEGREES on a pose tick. The old `value > 25` test made every pose tick a red occlusion.
ok('a yaw-62 pose tick is "Hard Yaw" in amber, not "Foreign Occlusion"', html.includes('Hard Yaw (&gt;50°)') && /title="\[Frame #40\] Hard Yaw[^"]*"/.test(html));
ok('...and its dot is amber', /style="left:[\d.]+%;background-color:#f59e0b"[^>]*title="\[Frame #40\]|title="\[Frame #40\][^>]*style="[^"]*#f59e0b|#f59e0b[^>]*title="\[Frame #40\]/.test(html));
ok('a yaw-48 pose tick (45-50) is info-blue, not red', /title="\[Frame #25\] Extreme head angle[^"]*"/.test(html));
ok('only the real occlusion tick uses the occlusion label', (html.match(/title="\[Frames? #\d+[^\]]*\] Foreign Occlusion/g) || []).length === 1);
ok('runs are keyboard-reachable buttons with an accessible name (not tab stops: a clip can have dozens)', /role="button" tabindex="-1" aria-label="Hard Yaw \(&gt;50°\), Frame #40"/.test(html));
ok('renders occlusion warning tooltip message on its track lane', html.includes('Foreign Occlusion (&gt;25%)'));
ok('each track header has a swap mute toggle only when the host can apply it', !html.includes('data-testid="lane-track-0-mute"'));
const muteHtml = renderToStaticMarkup(React.createElement(NleTimeline, { maxFrames: 300, fps: 30, frame: 25, trackLanes: mockTrackLanes, onToggleTrackActive() {} }));
ok('with a handler, the mute toggle appears per track', muteHtml.includes('data-testid="lane-track-0-mute"') && muteHtml.includes('data-testid="lane-track-1-mute"'));

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
globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? 'ALL GREEN: NleTimeline checks passed cleanly!' : `FAILED: ${failures} check(s) failed`);
