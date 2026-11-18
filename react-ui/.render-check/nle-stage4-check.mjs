/**
 * Stage 4 NLE models: geometry, ruler, face-coverage heatmap, transport, minimap, lane layout, markers.
 *
 *   node .render-check/run-module.mjs nle-stage4-check.mjs
 *
 * These are the pure rules behind the lanes. What they cannot show (that the canvas paints them, that
 * a pointer lands where the playhead is drawn) is checked in a real browser by e2e/nle-stage4.spec.js.
 */
globalThis.window = globalThis;

const geo = await import('../src/components/faceswap/nle/nleGeometry.ts');
const ruler = await import('../src/components/faceswap/nle/rulerModel.ts');
const health = await import('../src/components/faceswap/nle/faceHealth.ts');
const tr = await import('../src/components/faceswap/nle/transportModel.ts');
const mm = await import('../src/components/faceswap/nle/minimapModel.ts');
const lanes = await import('../src/components/faceswap/nle/laneLayout.ts');
const mk = await import('../src/components/faceswap/nle/markerModel.ts');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const section = (t) => console.log(`\n── ${t} ${'─'.repeat(Math.max(2, 64 - t.length))}`);

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Geometry: pixels and frames agree');
{
  const H = geo.LANE_HEADER_PX;
  const row = { left: 100, width: H + 1000 }; // header + 1000 px lane area
  const area = geo.laneAreaRect(row);
  ok('lane area drops the header column', area.left === 100 + H && area.width === 1000);
  ok('left edge of the lane area is the first visible frame', geo.clientXToFrame(area.left, area, 0, 999, 1000) === 0);
  ok('right edge is the last visible frame', geo.clientXToFrame(area.left + 1000, area, 0, 999, 1000) === 999);
  ok('a pointer over the header clamps to the first visible frame', geo.clientXToFrame(150, area, 40, 140, 1000) === 40);
  ok('mid-lane maps to mid-view', geo.clientXToFrame(area.left + 500, area, 200, 400, 1000) === 300);

  // The pre-fix mapping divided by the whole row. Show it was wrong, not just that the new one is right.
  const oldMap = (x, rect, vs, ve, max) => Math.max(0, Math.min(max - 1, Math.round(vs + Math.max(0, Math.min(1, (x - rect.left) / rect.width)) * (ve - vs))));
  // A playhead drawn on frame 0 sits at the lane area's left edge. The old mapping divided by the whole
  // row, header included, so a click exactly there read as frame ~183 of 1000; the error shrinks to 0
  // at the right edge, which is why it went unnoticed when testing near the end of a clip.
  const atZero = area.left;
  ok('old mapping reads a click on frame 0 as > 150 frames late', oldMap(atZero, row, 0, 999, 1000) > 150, `old gave ${oldMap(atZero, row, 0, 999, 1000)}`);
  ok('(the old error is the header column as a share of the row)', Math.abs(oldMap(atZero, row, 0, 999, 1000) - Math.round((H / row.width) * 999)) <= 1);
  ok('new mapping reads it as frame 0', geo.clientXToFrame(atZero, area, 0, 999, 1000) === 0);
  const drawnX = area.left + 0.9 * area.width; // a playhead drawn at 90% of the lane area
  ok('old mapping is still ~20+ frames off at 90%', Math.abs(oldMap(drawnX, row, 0, 999, 1000) - 899) >= 15, `old gave ${oldMap(drawnX, row, 0, 999, 1000)}`);
  ok('new mapping lands on it', Math.abs(geo.clientXToFrame(drawnX, area, 0, 999, 1000) - 899) <= 1,
    `new gave ${geo.clientXToFrame(drawnX, area, 0, 999, 1000)}`);

  // Round trip: draw a frame at its CSS position, click there, get the same frame back (every zoom).
  let bad = 0;
  for (const [vs, ve] of [[0, 999], [100, 220], [500, 504], [0, 39]]) {
    for (let f = vs; f <= ve; f += Math.max(1, Math.floor((ve - vs) / 37))) {
      const x = area.left + geo.frameToLaneX(f, vs, ve, area.width);
      if (geo.clientXToFrame(x, area, vs, ve, 1000) !== f) bad++;
    }
  }
  ok('frame -> pixel -> frame round-trips at every zoom', bad === 0, `${bad} mismatches`);
  ok('frameToRowLeft names the header in its calc()', geo.frameToRowLeft(50, 0, 100).includes(`${H}px`));
  ok('zero-width area returns a safe frame', geo.clientXToFrame(5, { left: 0, width: 0 }, 7, 90, 100) === 7);
  ok('clamps to the clip, not the view', geo.clientXToFrame(10_000, area, 0, 5000, 1000) === 999);
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Ruler: dense enough to read, sparse enough to label');
{
  const span = (vs, ve, w, fps = 25) => ruler.buildRuler({ viewStart: vs, viewEnd: ve, widthPx: w, fps });
  ok('frame label is zero-padded #00412', ruler.formatFrameLabel(412) === '#00412');
  ok('wide frame numbers keep their digits', ruler.formatFrameLabel(1234567) === '#1234567');

  const wide = span(0, 5000, 1400);
  const narrow = span(0, 5000, 420);
  ok('same span, wider window -> finer ticks', wide.minorStep < narrow.minorStep, `${wide.minorStep} vs ${narrow.minorStep}`);

  let violations = 0;
  for (const w of [320, 600, 1000, 1600, 2400, 3600]) {
    for (const [a, b] of [[0, 20], [0, 120], [0, 900], [0, 12_000], [0, 90_000], [33_000, 33_300]]) {
      const L = span(a, b, w);
      const majors = L.ticks.filter((t) => t.major);
      const minPxMajor = majors.length > 1 ? Math.min(...majors.slice(1).map((t, i) => (t.frame - majors[i].frame) * L.pxPerFrame)) : Infinity;
      if (minPxMajor < 104 - 1e-6) violations++;
      if (L.minorStep * L.pxPerFrame < 7 - 1e-6 && L.minorStep !== ruler.stepLadder(25).at(-1)) violations++;
      if (L.majorStep % L.minorStep !== 0) violations++;
      if (L.ticks.some((t) => t.frame < a || t.frame > b)) violations++;
      if (L.ticks.length > 1500) violations++;
    }
  }
  ok('labels never closer than 104 px, ticks never closer than 7 px, majors are multiples of minors', violations === 0, `${violations} violations`);

  const L = span(0, 2000, 1400, 25);
  const m = L.ticks.find((t) => t.major && t.frame === 500);
  ok('major tick carries SMPTE timecode', m?.timecode === '00:00:20:00', `${m?.timecode} (majorStep ${L.majorStep})`);
  const odd = ruler.buildRuler({ viewStart: 410, viewEnd: 420, widthPx: 1200, fps: 25 }).ticks.find((t) => t.frame === 412);
  ok('and the exact frame number', odd?.frameLabel === '#00412' && odd?.timecode === '00:00:16:12', `${odd?.frameLabel} ${odd?.timecode}`);
  ok('minor ticks carry no label text', L.ticks.filter((t) => !t.major).every((t) => t.timecode === '' && t.frameLabel === ''));
  const deep = span(0, 4, 1000);
  ok('at frame-level zoom every frame is a labelled tick', deep.minorStep === 1 && deep.majorStep === 1);
  ok('long-form clip: majors land on whole seconds at 25 fps', span(0, 90_000, 1400).majorStep % 25 === 0);
  ok('degenerate width does not explode', span(0, 100, 0).ticks.length <= 1500 && Number.isFinite(span(0, 0, 0).pxPerFrame));
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Face coverage heatmap');
{
  const base = { maxFrames: 100, trackRanges: [[10, 39], [60, 89]], warnings: [], scanned: true };
  const seg = health.healthSegments(base);
  ok('uncovered frames are lost (rose)', seg.some((s) => s.state === 'lost' && s.start === 0 && s.end === 9));
  ok('covered frames are face (emerald)', seg.some((s) => s.state === 'face' && s.start === 10 && s.end === 39));
  ok('tail gap is lost', seg.at(-1).state === 'lost' && seg.at(-1).end === 99);
  ok('segments tile the clip with no gap or overlap', seg.every((s, i) => s.start === (i ? seg[i - 1].end + 1 : 0)) && seg.at(-1).end === 99);

  const un = health.healthSegments({ ...base, scanned: false });
  ok('no scan -> one unscanned run, never rose', un.length === 1 && un[0].state === 'unscanned' && un[0].end === 99);

  const risky = health.healthSegments({ ...base, warnings: [{ frame: 20, type: 'pose' }, { frame: 21, type: 'occlusion' }, { frame: 70, type: 'pose' }, { frame: 5, type: 'pose' }, { frame: 30, type: 'gap' }] });
  ok('pose + occlusion on a covered frame -> amber', risky.some((s) => s.state === 'risk' && s.start === 20 && s.end === 21));
  ok('a warning on an uncovered frame does not repaint a dropout', health.segmentAt(risky, 5).state === 'lost');
  ok('gap-type warnings are not amber (the dropout already shows them)', health.segmentAt(risky, 30).state === 'face');
  const sum = health.summarizeHealth(risky);
  ok('summary frames add up to the clip', sum.total === 100 && sum.face + sum.risk + sum.lost + sum.unscanned === 100, JSON.stringify(sum));
  ok('summary counts', sum.risk === 3 && sum.lost === 40, JSON.stringify(sum));

  const pf = health.problemFrames(risky);
  ok('problem frames = first frame of each amber / rose run', JSON.stringify(pf) === JSON.stringify([0, 20, 40, 70, 90]), JSON.stringify(pf));
  ok('hit on an amber frame jumps to the run start', health.problemTargetNear(risky, 21) === 20);
  ok('hit on emerald with no tolerance -> nothing', health.problemTargetNear(risky, 25) === null);
  ok('tolerance forgives a pixel-thin miss', health.problemTargetNear(risky, 23, 3) === 20);
  ok('tolerance picks the nearest run', health.problemTargetNear(risky, 38, 5) === 40);
  ok('segmentAt is exact at boundaries', health.segmentAt(risky, 40).state === 'lost' && health.segmentAt(risky, 39).state === 'face');
  ok('empty clip is safe', health.healthSegments({ maxFrames: 0, trackRanges: [], scanned: true }).length === 0);

  // Scale: this recomputes when a scan lands, so it must not be slow on a long clip.
  const t0 = performance.now();
  const big = health.healthSegments({ maxFrames: 600_000, trackRanges: Array.from({ length: 500 }, (_, i) => [i * 1000, i * 1000 + 700]), warnings: Array.from({ length: 4000 }, (_, i) => ({ frame: i * 100 + 5, type: 'pose' })), scanned: true });
  const ms = performance.now() - t0;
  ok(`600k-frame clip, 500 tracks, 4000 warnings: ${ms.toFixed(0)} ms`, ms < 400 && big.length > 100);
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Transport');
{
  ok('five speeds', JSON.stringify(tr.PLAYBACK_RATES) === JSON.stringify([0.25, 0.5, 1, 2, 4]));
  ok('rate label', tr.formatRate(0.25) === '0.25×' && tr.formatRate(4) === '4×');
  ok('snapRate falls onto the ladder', tr.snapRate(0.3) === 0.25 && tr.snapRate(3) === 2 && tr.snapRate(NaN) === 1);
  ok('shiftRate stops at both ends', tr.shiftRate(4, 1) === 4 && tr.shiftRate(0.25, -1) === 0.25 && tr.shiftRate(1, 1) === 2);
  ok('stepTo clamps to the clip', tr.stepTo(0, -1, 100) === 0 && tr.stepTo(99, 1, 100) === 99 && tr.stepTo(50, 10, 100) === 60);
  ok('neighbourFrame next/prev, unsorted input', tr.neighbourFrame([90, 10, 50], 50, 1) === 90 && tr.neighbourFrame([90, 10, 50], 50, -1) === 10);
  ok('neighbourFrame is null at the ends', tr.neighbourFrame([10, 50], 50, 1) === null && tr.neighbourFrame([10, 50], 10, -1) === null);
  const reg = tr.loopRegion(80, 20, 100);
  ok('loop region is ordered and clamped', reg.in === 20 && reg.out === 80 && tr.loopRegion(-5, 500, 100).out === 99);
  ok('region length counts both ends', tr.regionLength(reg) === 61);
  ok('loop wraps once past Out', tr.advanceWithLoop(81, reg, true) === 20 && tr.advanceWithLoop(80, reg, true) === 80 && tr.advanceWithLoop(81, reg, false) === 81);
  ok('duration readout', tr.formatDuration(12, 25) === '12 f' && tr.formatDuration(137, 25) === '5s 12f' && tr.formatDuration(25 * 75 + 3, 25) === '1:15s 03f', `${tr.formatDuration(137, 25)} / ${tr.formatDuration(25 * 75 + 3, 25)}`);
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Minimap navigator');
{
  const max = 1000;
  const view = [200, 400];
  ok('inside the window -> move', mm.hitMinimap(0.3, view, max, 1000) === 'move');
  ok('on the left edge -> resize-left', mm.hitMinimap(200 / 999, view, max, 1000) === 'resize-left');
  ok('on the right edge -> resize-right', mm.hitMinimap(400 / 999, view, max, 1000) === 'resize-right');
  ok('outside -> jump', mm.hitMinimap(0.8, view, max, 1000) === 'jump');
  ok('a narrow window can still be moved (edges do not swallow it)', mm.hitMinimap(0.5, [498, 502], max, 1000) !== 'jump');

  const d = (mode, startRatio, ratio, startView = view) => mm.dragMinimap({ mode, startRatio, ratio, startView, maxFrames: max });
  ok('move keeps the span', (() => { const w = d('move', 0.3, 0.5); return w[1] - w[0] === 200 && w[0] === 400; })());
  ok('move cannot leave the clip', d('move', 0.3, 5)[1] === 999 && d('move', 0.3, -5)[0] === 0);
  ok('resize-left grows the window leftwards', d('resize-left', 0.2, 0.1)[0] === 100 && d('resize-left', 0.2, 0.1)[1] === 400);
  ok('resize-left stops at the minimum span', d('resize-left', 0.2, 5)[1] - d('resize-left', 0.2, 5)[0] === 4);
  ok('resize-right grows rightwards, capped at the end', d('resize-right', 0.4, 0.6)[1] === 600 && d('resize-right', 0.4, 9)[1] === 999);
  ok('jump recentres on the click', (() => { const w = d('jump', 0, 0.9); return Math.abs((w[0] + w[1]) / 2 - 899) <= 1 && w[1] - w[0] === 200; })());
  ok('jump near the end stays inside the clip', d('jump', 0, 1)[1] === 999);
  const full = mm.dragMinimap({ mode: 'jump', startRatio: 0, ratio: 0.5, startView: [0, 999], maxFrames: max });
  ok('a full-clip window cannot be dragged out of range', full[0] === 0 && full[1] === 999);
  const z = mm.wheelMinimap([200, 400], 0.3, 0.5, max);
  ok('wheel zoom halves the span about the pointer', z[1] - z[0] === 100 && z[0] >= 200 && z[1] <= 400, JSON.stringify(z));
  ok('wheel zoom out is capped at the clip', mm.wheelMinimap([0, 900], 0.5, 10, max)[1] === 999);

  // The clip's length arriving AFTER the window was made (Extras: metadata loads a tick later)
  ok('a window made while the clip was 1 frame long widens to the whole clip when the length is known', JSON.stringify(mm.reconcileView([0, 0], 0, 99)) === '[0,99]');
  ok('a fitted window follows a longer clip', JSON.stringify(mm.reconcileView([0, 179], 179, 359)) === '[0,359]');
  ok('a fitted window follows a shorter clip', JSON.stringify(mm.reconcileView([0, 359], 359, 99)) === '[0,99]');
  ok('a ZOOMED window is kept where the user put it', JSON.stringify(mm.reconcileView([40, 80], 179, 359)) === '[40,80]');
  ok('a zoomed window is clamped into a shorter clip, never past its end', (() => { const w = mm.reconcileView([300, 340], 359, 99); return w[1] <= 99 && w[0] >= 0 && w[1] > w[0]; })());
  ok('the window never inverts on a 1-frame clip', (() => { const w = mm.reconcileView([0, 0], 0, 0); return w[0] === 0 && w[1] === 0; })());

  ok('zoom slider: 0 = whole clip, 100 = 4 frames',mm.zoomToSpan(0, 999) === 999 && mm.zoomToSpan(100, 999) === 4);
  ok('zoom slider is monotonic and round-trips', (() => { let prev = Infinity; for (let z = 0; z <= 100; z += 5) { const s = mm.zoomToSpan(z, 5000); if (s > prev) return false; prev = s; if (Math.abs(mm.spanToZoom(s, 5000) - z) > 1) return false; } return true; })());
  ok('zoom slider: midpoint is the geometric mean, not the arithmetic one', Math.abs(mm.zoomToSpan(50, 10_000) - Math.sqrt(10_000 * 4)) < 2, `${mm.zoomToSpan(50, 10_000)}`);
  ok('zoom slider survives a 2-frame clip', mm.spanToZoom(1, 1) >= 0 && mm.zoomToSpan(50, 1) >= 1);
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Lane layout');
{
  const s0 = lanes.defaultLaneStates();
  ok('six lanes, none hidden or locked', lanes.LANE_IDS.length === 6 && lanes.LANE_IDS.every((id) => !s0[id].hidden && !s0[id].locked && s0[id].height === 'normal'));
  const hid = lanes.reduceLanes(s0, { type: 'toggleHidden', id: 'health' });
  ok('hide collapses the body', lanes.laneBodyPx('health', hid.health) === lanes.COLLAPSED_PX && lanes.laneBodyPx('health', s0.health) === lanes.LANE_HEIGHTS.health.normal);
  ok('toggling one lane leaves the others alone', hid.v1 === s0.v1);
  ok('height cycles compact -> normal -> tall -> compact', lanes.nextHeight('compact') === 'normal' && lanes.nextHeight('normal') === 'tall' && lanes.nextHeight('tall') === 'compact');
  const tall = lanes.reduceLanes(s0, { type: 'cycleHeight', id: 'a1' });
  ok('cycleHeight changes the pixel height', lanes.laneBodyPx('a1', tall.a1) === lanes.LANE_HEIGHTS.a1.tall);
  ok('heights are strictly increasing in every lane', lanes.LANE_IDS.every((id) => lanes.LANE_HEIGHTS[id].compact < lanes.LANE_HEIGHTS[id].normal && lanes.LANE_HEIGHTS[id].normal < lanes.LANE_HEIGHTS[id].tall));
  ok('lock toggles', lanes.reduceLanes(s0, { type: 'toggleLocked', id: 'kf' }).kf.locked === true);
  ok('reset restores the defaults', JSON.stringify(lanes.reduceLanes(hid, { type: 'reset' })) === JSON.stringify(s0));
  ok('garbage in storage -> defaults', JSON.stringify(lanes.sanitizeLaneStates('x')) === JSON.stringify(s0) && JSON.stringify(lanes.sanitizeLaneStates({ v1: { height: 'huge', hidden: 'yes' } }).v1) === JSON.stringify(s0.v1));
  const mem = new Map();
  const store = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v) };
  lanes.saveLaneStates(store, tall);
  ok('save then load round-trips', lanes.loadLaneStates(store).a1.height === 'tall');
  const hostile = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } };
  ok('blocked storage neither throws nor loses the defaults', JSON.stringify(lanes.loadLaneStates(hostile)) === JSON.stringify(s0) && (lanes.saveLaneStates(hostile, tall), true));
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Markers: colour + note, same frame list as before');
{
  const maxFrames = 500;
  let st = { frames: [], meta: {} };
  st = mk.addMarker(st, 9, maxFrames, { color: '#fb7185', note: 'Hand covers mouth\nsecond line' });
  ok('stored frame is 1-based UI (frame0 9 -> 10) so the legacy list still reads it', st.frames[0] === 10);
  const built = mk.buildMarkers(st.frames, st.meta);
  ok('timeline frame is 0-based again', built[0].frame === 9);
  ok('label = first line of the note', built[0].label === 'Hand covers mouth' && built[0].note.includes('second line'));
  ok('colour carried', built[0].color === '#fb7185');
  ok('a marker the sidecar never saw is default-coloured and called Marker', mk.buildMarkers([42], {})[0].label === 'Marker' && mk.buildMarkers([42], {})[0].color === mk.DEFAULT_MARKER_COLOR);
  ok('adding on an existing frame is a no-op', mk.addMarker(st, 9, maxFrames) === st);
  ok('adding past the clip is a no-op', mk.addMarker(st, 600, maxFrames) === st);

  const moved = mk.moveMarker(st, 9, 100, maxFrames);
  ok('move carries the note and colour to the new frame', moved.frames[0] === 101 && moved.meta['101'].note.startsWith('Hand') && moved.meta['10'] === undefined);
  const two = mk.addMarker(st, 200, maxFrames);
  ok('move onto another marker is refused (one note would be lost)', mk.moveMarker(two, 9, 200, maxFrames) === two);
  ok('move keeps the list sorted', JSON.stringify(mk.moveMarker(two, 9, 300, maxFrames).frames) === JSON.stringify([201, 301]));
  ok('move clamps into the clip', mk.moveMarker(st, 9, 9999, maxFrames).frames[0] === 500);

  const upd = mk.updateMarker(st, 9, { color: '#34d399', note: 'ok now' });
  ok('update changes colour and note', upd.meta['10'].color === '#34d399' && upd.meta['10'].note === 'ok now');
  ok('update rejects a non-hex colour', mk.updateMarker(st, 9, { color: 'red' }).meta['10'].color === '#fb7185');
  ok('note is capped', mk.updateMarker(st, 9, { note: 'x'.repeat(5000) }).meta['10'].note.length === mk.MAX_NOTE_CHARS);
  const gone = mk.removeMarker(st, 9);
  ok('remove drops the frame and its sidecar entry', gone.frames.length === 0 && Object.keys(gone.meta).length === 0);

  const synced = mk.withFrames(two, [10, 77], maxFrames);
  ok('the M key writes a bare frame list: kept notes stay, removed ones go, new ones get defaults', synced.meta['10'] && !synced.meta['201'] && synced.frames.join() === '10,77');
  ok('sanitizeFrames drops junk and duplicates', mk.sanitizeFrames([5, 5, 'a', -1, 0, 3.4, 9999], 500).join() === '3,5');
  ok('sanitizeMeta drops bad keys and colours', (() => { const m = mk.sanitizeMeta({ 10: { color: 'nope', note: 4 }, x: {} }); return m['10'].color === mk.DEFAULT_MARKER_COLOR && m['10'].note === '' && !m.x; })());
  ok('storage keys are the legacy key + a separate sidecar key', mk.markersKey('clip') === 'roop_markers_clip' && mk.markerMetaKey('clip') === 'roop_marker_meta_clip');
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
section('Pose series -> amber ticks');
{
  globalThis.location = { origin: 'http://127.0.0.1:1', protocol: 'http:', host: '127.0.0.1:1', hash: '' };
  const rb = await import('../src/components/faceswap/runBarSync.ts');
  const series = (frames, yaw, pitch = frames.map(() => 0)) => ({ available: true, frame: frames, yaw, pitch });
  const t = rb.poseWarningsFromSeries(series([0, 10, 20, 30, 40], [0, 44, 46, -70, 10], [0, 0, 0, 0, 31]), 2);
  ok('only samples past 45 deg yaw or 30 deg pitch become ticks (46, -70 yaw; 31 pitch)', t.map((x) => x.frame).join() === '20,30,40', t.map((x) => x.frame).join());
  ok('boundary: exactly 45 deg is NOT extreme', rb.poseWarningsFromSeries(series([0], [45]), 0).length === 0);
  ok('ticks are pose-typed, carry the track and the yaw', t[0].type === 'pose' && t[0].trackId === 2 && t[0].value === 46 && /Track #3/.test(t[0].message));
  ok('severity: > 50 deg is a warning, 45-50 is info', t[1].severity === 'warning' && t[0].severity === 'info');
  ok('span = the median gap between samples (what one sample stands for)', t[0].span === 10);
  ok('unavailable / empty series -> nothing, not a crash', rb.poseWarningsFromSeries(null, 0).length === 0 && rb.poseWarningsFromSeries({ available: false, frame: [], yaw: [], pitch: [] }, 0).length === 0 && rb.poseWarningsFromSeries(series([], []), 0).length === 0);
  ok('a single sample has a span of 1', rb.poseWarningsFromSeries(series([7], [80]), 0)[0].span === 1);

  // ...and through the coverage model: a sample paints its neighbourhood, only where a face was tracked
  const base = { maxFrames: 100, trackRanges: [[0, 99]], scanned: true };
  const seg = health.healthSegments({ ...base, warnings: [{ frame: 50, type: 'pose', span: 5 }] });
  ok('span 5 paints 48..52 amber', seg.some((s) => s.state === 'risk' && s.start === 48 && s.end === 52), JSON.stringify(seg.filter((s) => s.state === 'risk')));
  ok('span 1 paints just the frame', health.healthSegments({ ...base, warnings: [{ frame: 50, type: 'pose', span: 1 }] }).filter((s) => s.state === 'risk').map((s) => [s.start, s.end]).join() === '50,50');
  const edge = health.healthSegments({ maxFrames: 100, trackRanges: [[10, 20]], scanned: true, warnings: [{ frame: 22, type: 'pose', span: 9 }] });
  ok('the halo never repaints a dropout as risk (only tracked frames can be amber)', edge.filter((s) => s.state === 'risk').map((s) => [s.start, s.end]).join() === '18,20', JSON.stringify(edge));
  ok('the halo is clamped to the clip', health.healthSegments({ ...base, warnings: [{ frame: 0, type: 'pose', span: 9 }, { frame: 99, type: 'pose', span: 9 }] }).filter((s) => s.state === 'risk').map((s) => [s.start, s.end]).join() === '0,4,95,99');
}

section('Warning runs: one element per stretch, not per sample');
{
  const runs = (ticks) => health.groupWarningRuns(ticks);
  const dense = Array.from({ length: 200 }, (_, i) => ({ frame: 100 + i, type: 'pose', value: 50 + (i % 7), message: 'turn' }));
  ok('200 consecutive pose samples are ONE run', runs(dense).length === 1 && runs(dense)[0].start === 100 && runs(dense)[0].end === 299 && runs(dense)[0].count === 200);
  ok('the run keeps the peak value and a message', runs(dense)[0].peak === 56 && runs(dense)[0].message === 'turn');
  const two = runs([{ frame: 10, type: 'pose', value: 60 }, { frame: 11, type: 'pose', value: 61 }, { frame: 30, type: 'pose', value: 70 }]);
  ok('a gap starts a new run', two.length === 2 && two[0].end === 11 && two[1].start === 30);
  ok('different types never merge, even adjacent', runs([{ frame: 5, type: 'pose' }, { frame: 6, type: 'occlusion' }]).length === 2);
  const sampled = runs([{ frame: 100, type: 'pose', span: 5 }, { frame: 105, type: 'pose', span: 5 }, { frame: 110, type: 'pose', span: 5 }]);
  ok('decimated samples (span 5, stride 5) whose neighbourhoods touch are one run covering their whole reach', sampled.length === 1 && sampled[0].start === 98 && sampled[0].end === 112, JSON.stringify(sampled));
  ok('unsorted input is fine', runs([{ frame: 50, type: 'pose' }, { frame: 49, type: 'pose' }, { frame: 51, type: 'pose' }]).length === 1);
  ok('empty -> none', runs([]).length === 0);
  const rb = await import('../src/components/faceswap/runBarSync.ts');
  const ticks = [...dense.map((t) => ({ ...t, trackId: 0 })), { frame: 5, type: 'gap', trackId: 0 }, ...dense.map((t) => ({ ...t, trackId: 1 }))];
  ok('the toast counts EVENTS: two tracks\' turns + one gap = 3, not 401', rb.countWarningEvents(ticks) === 3, `${rb.countWarningEvents(ticks)}`);
  const t0 = performance.now();
  const huge = health.groupWarningRuns(Array.from({ length: 200_000 }, (_, i) => ({ frame: i * 3, type: 'pose', span: 3 })));
  ok(`200k ticks group in ${(performance.now() - t0).toFixed(0)} ms`, performance.now() - t0 < 400 && huge.length === 1);
}

section('Shuttle <-> player sync keeps the chosen speed');
{
  const sh = await import('../src/components/faceswap/timeline/shuttle.js');
  // The effect this replaced, as it was: reset to 1x on EVERY render with no shuttle active.
  const oldSync = (state) => (state.dir > 0 ? { rate: sh.shuttleRate(state), play: true } : state.dir === 0 ? { rate: 1, play: false } : null);
  const idle = sh.IDLE;
  ok('(old) idle re-sync forced 1x, which is what clobbered a 0.5x pick on Play', oldSync(idle)?.rate === 1);
  ok('idle -> idle leaves the chosen rate alone', sh.shuttleSync(idle, idle, true) === null && sh.shuttleSync(idle, idle, false) === null);
  const fwd = sh.shuttleKey(idle, 'l');
  ok('L starts a 1x forward shuttle and starts the player', JSON.stringify(sh.shuttleSync(idle, fwd, false)) === '{"rate":1,"play":true}');
  ok('L, L -> 2x', sh.shuttleSync(fwd, sh.shuttleKey(fwd, 'l'), true).rate === 2);
  ok('an already-playing player is not "started" again', sh.shuttleSync(idle, fwd, true).play === false);
  ok('K after a shuttle puts the rate back to 1x, once', JSON.stringify(sh.shuttleSync(fwd, idle, true)) === '{"rate":1,"play":false}');
  const rev = sh.shuttleKey(idle, 'j');
  ok('leaving a reverse shuttle also resets once', sh.shuttleSync(rev, idle, false)?.rate === 1);
  ok('reverse shuttle itself drives the clock, not the forward player', sh.shuttleSync(idle, rev, false) === null);

  // L pressed on a STOPPED player: the shuttle and the play request land in one commit, isPlaying still false
  const oldEnds = (isPlaying, state) => !isPlaying && state.dir > 0;
  ok('(old) a level test cancelled a forward shuttle in the very commit that started it', oldEnds(false, fwd) === true);
  ok('L on a stopped player does NOT end the shuttle (was playing=false, is playing=false)', sh.shuttleEndsOnPlayChange(false, false, fwd) === false);
  ok('...nor when the player then starts (false -> true)', sh.shuttleEndsOnPlayChange(false, true, fwd) === false);
  ok('pausing a running forward shuttle (true -> false) ends it', sh.shuttleEndsOnPlayChange(true, false, fwd) === true);
  ok('an unchanged playing forward shuttle goes on', sh.shuttleEndsOnPlayChange(true, true, fwd) === false);
  ok('a reverse shuttle ends when playback starts, and not while stopped', sh.shuttleEndsOnPlayChange(false, true, rev) === true && sh.shuttleEndsOnPlayChange(false, false, rev) === false);
  ok('nothing to end when idle', sh.shuttleEndsOnPlayChange(true, false, idle) === false);
}

section('Marker edits from the UI');
{
  const max = 500;
  let st = mk.addMarker({ frames: [], meta: {} }, 9, max, { color: '#fb7185', note: 'beat' });
  st = mk.addMarker(st, 200, max, { note: 'second' });
  ok('marker id round-trips to its 0-based frame', mk.markerIdToFrame0('marker_10') === 9 && mk.markerIdToFrame0('nonsense') === -1);
  const moved = mk.editMarker(st, 'marker_10', { frame: 50 }, max);
  ok('drag: frame moves, note and colour travel', moved.frames.join() === '51,201' && moved.meta['51'].note === 'beat' && moved.meta['51'].color === '#fb7185');
  const refused = mk.editMarker(st, 'marker_10', { frame: 200, note: 'edited' }, max);
  ok('drag onto a neighbour is refused but the text edit is kept', refused.frames.join() === '10,201' && refused.meta['10'].note === 'edited');
  const both = mk.editMarker(st, 'marker_10', { frame: 60, color: '#34d399', note: 'moved+edited' }, max);
  ok('move and edit in one call land on the NEW frame', both.meta['61'].color === '#34d399' && both.meta['61'].note === 'moved+edited' && !both.meta['10']);
  ok('unknown id is a no-op', mk.editMarker(st, 'marker_999', { note: 'x' }, max) === st);
  ok('edit without a frame never moves it', mk.editMarker(st, 'marker_10', { note: 'only text' }, max).frames.join() === '10,201');
}

section('Filmstrip: density follows pixels, slots sit on a grid');
{
  const fm = await import('../src/components/faceswap/nle/filmstripModel.ts');
  const slots = (vs, ve, w, max = 10_000) => fm.filmstripSlots({ viewStart: vs, viewEnd: ve, widthPx: w, maxFrames: max });
  ok('1-2-5 strides', [1, 2, 3, 6, 11, 21, 51, 101, 1000, 1001].map(fm.nearestStride).join() === '1,2,5,10,20,50,100,200,1000,2000', [1, 2, 3, 6, 11, 21, 51, 101, 1000, 1001].map(fm.nearestStride).join());
  const wide = slots(0, 9999, 1500);
  const zoomed = slots(0, 2499, 1500);
  // The 1-2-5 ladder moves in steps of 2-2.5x, so a 4x zoom lands between 2.5x and 5x finer.
  const ratio = wide[0].stride / zoomed[0].stride;
  ok('zooming in 4x makes the stills 2.5-5x denser in frames', ratio >= 2.5 && ratio <= 5, `${wide[0].stride} -> ${zoomed[0].stride} (${ratio}x)`);
  ok('a still is never narrower than its target width (100 px)', [wide, zoomed].every((s) => s.every((x) => (x.widthPct / 100) * 1500 >= 99)), JSON.stringify(wide.slice(0, 2)));
  ok('slot frames are multiples of the stride', wide.every((s) => s.frame % s.stride === 0) && zoomed.every((s) => s.frame % s.stride === 0));
  const a = new Set(slots(1000, 3500, 1500).map((s) => s.frame));
  const b = slots(1010, 3510, 1500).map((s) => s.frame);
  ok('panning 10 frames reuses at least all-but-one still (a cache hit, not a refetch)', b.filter((f) => a.has(f)).length >= b.length - 1, `${b.filter((f) => a.has(f)).length}/${b.length}`);
  ok('slot count is bounded by the strip width', wide.length <= Math.ceil(1500 / 100) + 2 && wide.length >= 8, `${wide.length}`);
  ok('slots never point past the last frame', slots(900, 999, 800, 1000).every((s) => s.frame <= 999));
  ok('at frame-level zoom every frame is its own still', slots(0, 9, 1000).every((s) => s.stride === 1) && slots(0, 9, 1000).length === 10);
  ok('server frame number is 1-based', fm.toServerFrame(0) === 1 && fm.toServerFrame(411) === 412);
  ok('no width, no slots', slots(0, 100, 0).length === 0);
}

section('Segment blocks pack into rows');
{
  const sg = await import('../src/components/faceswap/nle/segmentModel.ts');
  const blocks = [
    { id: 'a', start: 0, end: 99, label: 'seg', kind: 'segment' },
    { id: 'b', start: 20, end: 40, label: 'ov', kind: 'override' },
    { id: 'c', start: 100, end: 120, label: 'seg2', kind: 'segment' },
    { id: 'd', start: 30, end: 35, label: 'byp', kind: 'bypass' },
  ];
  const { packed, rows } = sg.packRows(sg.normalizeBlocks(blocks, 1000));
  const row = (id) => packed.find((p) => p.id === id).row;
  ok('overlapping blocks get different rows', row('a') !== row('b') && row('b') !== row('d'));
  ok('a block after the first ends reuses row 0', row('c') === 0);
  ok('row count is what the lane has to fit', rows === 3, `${rows}`);
  ok('no two blocks in one row overlap', packed.every((p) => packed.every((q) => p === q || p.row !== q.row || p.end < q.start || q.end < p.start)));
  ok('out-of-range and reversed blocks are clamped', JSON.stringify(sg.normalizeBlocks([{ id: 'x', start: 900, end: -4, label: '', kind: 'segment' }], 500).map((b) => [b.start, b.end])) === '[[0,499]]');
  ok('NaN blocks are dropped', sg.normalizeBlocks([{ id: 'x', start: NaN, end: 4, label: '', kind: 'segment' }], 500).length === 0);
  ok('blockAt returns the top-most block', sg.blockAt(packed, 32).id === 'd' && sg.blockAt(packed, 25).id === 'b' && sg.blockAt(packed, 500) === null);
}

section('Client media: fps, peaks');
{
  const cm = await import('../src/components/faceswap/nle/clientMedia.ts');
  ok('snapFps snaps 29.9 -> 29.97 and 24.02 -> 24', cm.snapFps(29.93) === 29.97 && cm.snapFps(24.02) === 24);
  ok('an unusual rate is kept, not forced onto the ladder', cm.snapFps(37.2) === 37.2);
  ok('garbage rate -> 30', cm.snapFps(NaN) === 30 && cm.snapFps(-1) === 30);
  const ts = (fps, n, jitter = 0) => Array.from({ length: n }, (_, i) => i / fps + (i % 5 === 0 ? jitter : 0));
  ok('timestamps at 25 fps read as 25', cm.fpsFromTimestamps(ts(25, 20)) === 25);
  ok('timestamps at 29.97 fps read as 29.97', cm.fpsFromTimestamps(ts(30000 / 1001, 20)) === 29.97);
  ok('one dropped frame does not move the median', cm.fpsFromTimestamps([...ts(25, 10), 0.4 + 10 / 25 + 0.04, ...ts(25, 8).map((t) => t + 1)]) === 25);
  ok('too few frames -> no answer (caller keeps the assumed rate)', cm.fpsFromTimestamps([0, 0.04, 0.08]) === null);
  // The pattern a real 25 fps clip produced at 2x playback in Chromium (media times, from the e2e debug run):
  // the display showed only some frames, so most gaps were TWO frames. The median read this as 12.5 fps.
  const measured = [0.04, 0.12, 0.2, 0.24, 0.32, 0.4, 0.48, 0.52, 0.6, 0.68, 0.76, 0.8, 0.88, 0.96, 1.04, 1.08, 1.16, 1.24, 1.32, 1.36];
  const medianRead = (() => { const g = measured.slice(1).map((t, i) => 1 / (t - measured[i])).sort((a, b) => a - b); return cm.snapFps(g[g.length >> 1]); })();
  ok('(old) the median of a skipped-frame stream read as HALF the rate', medianRead !== 25, `${medianRead}`);
  ok('skipped frames: the smallest gap is the frame interval -> 25', cm.fpsFromTimestamps(measured) === 25, `${cm.fpsFromTimestamps(measured)}`);
  ok('every other frame skipped but with some singles still reads 25', cm.fpsFromTimestamps([0, 0.08, 0.16, 0.2, 0.28, 0.36, 0.4, 0.48]) === 25);
  const vfr = [0, 0.033, 0.075, 0.1, 0.145, 0.18, 0.21, 0.26, 0.29, 0.34];
  const vfrRead = cm.fpsFromTimestamps(vfr);
  ok('a variable-rate clip reads its TYPICAL rate (the median gap: 28.6), not its fastest gap (40)', vfrRead > 25 && vfrRead < 31, `${vfrRead}`);
  ok('timestamps in jittered 1 ms units still read 25', cm.fpsFromTimestamps([0, 0.040, 0.080, 0.120, 0.161, 0.200, 0.240, 0.280, 0.321]) === 25);
  const pcm = new Float32Array(48000);
  pcm[24000] = 0.5; pcm[24001] = -1;
  const peaks = cm.pcmPeaks([pcm], 48000, 100);
  ok('peaks: 100/s over 1 s = 100 buckets', peaks.length === 100);
  ok('peaks: a spike lands in its bucket at full scale (|-1| -> 255), the rest is silent', peaks[50] === 255 && peaks.filter((v) => v > 0).length === 1);
  ok('peaks: stereo takes the louder channel', cm.pcmPeaks([new Float32Array([0, 0.2, 0, 0]), new Float32Array([0, 0, 0.8, 0])], 4, 1)[0] === 204);
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll Stage 4 NLE model checks passed');
