/**
 * Stage 11's pure logic (plain node, no browser): the Pen tool's geometry and the identity pins / confidence strip.
 *
 *   node .render-check/run-module.mjs render-stage11-logic.mjs
 *
 * The geometry is the part that must NOT drift from the backend: the outline a user draws has to be the outline
 * app/roop/roto_polygons.py renders. The PY_* constants below were produced by that module (flatten, pin_transform,
 * face_anchors) and are asserted here to the same tolerance the Python tests use.
 */
import * as R from '../src/components/faceswap/rotoModel.js';
import * as L from '../src/components/faceswap/identityLock.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const near = (a, b, tol = 1e-3) => Math.abs(a - b) <= tol;
const near2 = (a, b, tol = 1e-3) => near(a[0], b[0], tol) && near(a[1], b[1], tol);

// ── reference values from app/roop/roto_polygons.py ─────────────────────────
const PY_FLAT_COUNT = 18;
const PY_FLAT_5 = [30.4443359375, -8.056640625];
const PY_FLAT_8 = [50.0, -15.0];
const PY_FLAT_16 = [100.0, 0.0];
const PY_PIN = [[0.9142857193946838, -1.0, 282.1428527832031], [1.0, 0.9142857193946838, -157.42857360839844]];
const PY_PIN_APPLIED = [283.9142761230469, 274.22857666015625];

const curved = [
  { x: 0, y: 0, hin: null, hout: [30, 0] },
  { x: 100, y: 0, hin: [-30, -40], hout: null },
  { x: 50, y: 90, hin: null, hout: null },
];

console.log('geometry parity with the backend');
{
  const flat = R.flatten(curved, 16);
  ok('flatten: same sample count as the Python module', flat.length === PY_FLAT_COUNT, `${flat.length}`);
  ok('flatten: interior Bezier samples match', near2(flat[5], PY_FLAT_5) && near2(flat[8], PY_FLAT_8), JSON.stringify([flat[5], flat[8]]));
  ok('flatten: the curve lands on the next anchor', near2(flat[16], PY_FLAT_16));
  {
    // the segment from the LAST point back to the first is the only curved one
    const wrapCurved = [{ x: 0, y: 0, hin: [-5, 5], hout: null }, { x: 10, y: 0, hin: null, hout: null }, { x: 5, y: 9, hin: null, hout: [8, 0] }];
    const closedPts = R.flatten(wrapCurved, 16);
    const openPts = R.flatten(wrapCurved, 16, false);
    ok('closed: the wrap-around curve is sampled (3 anchors + 15 interior samples)', closedPts.length === 18, String(closedPts.length));
    ok('open: no wrap-around, the last point is the end of the line', openPts.length === 3 && eq(openPts[2], [5, 9]), JSON.stringify(openPts));
  }
  ok('straight polygon flattens to its vertices', eq(R.flatten([R.point(0, 0), R.point(5, 0), R.point(5, 5)]), [[0, 0], [5, 0], [5, 5]]));

  const m = R.pinTransform([200, 205], [200, 240], [260, 230], [225, 262]);
  ok('pinTransform: matrix matches', near(m.a, PY_PIN[0][0]) && near(m.b, PY_PIN[0][1]) && near(m.tx, PY_PIN[0][2], 1e-2)
    && near(m.c, PY_PIN[1][0]) && near(m.d, PY_PIN[1][1]) && near(m.ty, PY_PIN[1][2], 1e-2), JSON.stringify(m));
  ok('pinTransform: a point lands where Python puts it', near2(R.applyTransform(m, [236, 214]), PY_PIN_APPLIED, 1e-2));
  ok('pinTransform: the reference pair maps onto the current pair',
    near2(R.applyTransform(m, [200, 205]), [260, 230], 1e-2) && near2(R.applyTransform(m, [200, 240]), [225, 262], 1e-2));
  ok('pinTransform: degenerate pairs are null', R.pinTransform([1, 1], [1, 1], [0, 0], [5, 5]) === null && R.pinTransform([0, 0], [5, 5], [3, 3], [3, 3]) === null);
}

console.log('anchors from the preview response');
{
  ok('well-formed anchors', eq(R.anchorsOf([[[160, 135], [160, 219]]], 0), { nose: [160, 135], chin: [160, 219] }));
  ok('missing / degenerate / non-finite anchors are null',
    R.anchorsOf([null], 0) === null && R.anchorsOf([], 3) === null && R.anchorsOf(undefined, 0) === null
    && R.anchorsOf([[[1, 1], [1, 1]]], 0) === null && R.anchorsOf([[[1, 1], [NaN, 3]]], 0) === null);
}

console.log('editing');
{
  let z = R.newZone({ id: 'a', feather: 12 });
  [[10, 10], [60, 10], [60, 60], [10, 60]].forEach(([x, y]) => { z = R.addPoint(z, x, y); });
  ok('open zone with four points', z.points.length === 4 && z.closed === false);
  ok('close needs a triangle', R.closeZone(R.newZone()).closed === false && R.closeZone(z).closed === true);
  z = R.closeZone(z);
  z = R.movePoint(z, 2, 70, 70);
  ok('movePoint moves exactly one anchor', z.points[2].x === 70 && z.points[1].x === 60);
  const before = z;
  R.movePoint(z, 1, 0, 0);
  ok('edits return new objects (the old zone is untouched)', z === before && before.points[1].x === 60);
  ok('hitPoint: nearest within tolerance, else -1', R.hitPoint(z, 61, 11, 4) === 1 && R.hitPoint(z, 200, 200, 4) === -1);
  const smooth = R.setHandle(z, 0, 'hout', 15, -5);
  ok('a smooth handle mirrors its opposite', eq(smooth.points[0].hout, [15, -5]) && eq(smooth.points[0].hin, [-15, 5]));
  ok('mirror can be turned off', R.setHandle(z, 0, 'hout', 15, -5, false).points[0].hin === null);
  ok('hitHandle finds the knob at anchor + offset', eq(R.hitHandle(smooth, 25, 5, 3), { i: 0, which: 'hout' }));
  ok('removing a point from a triangle opens it', R.removePoint(R.closeZone(R.addPoint(R.addPoint(R.addPoint(R.newZone(), 0, 0), 9, 0), 9, 9)), 0).closed === false);
  ok('mode and feather are validated', R.setMode(z, 'include').mode === 'include' && R.setMode(z, 'nonsense').mode === 'exclude'
    && R.setFeather(z, 1e9).feather === R.MAX_FEATHER && R.setFeather(z, -5).feather === 0);
  ok('range is ordered and inclusive; null clears', eq(R.setRange(z, [90, 10]).frames, [10, 90]) && R.setRange(R.setRange(z, [1, 2]), null).frames === null);
  ok('activeAt: inclusive bounds, whole clip, disabled',
    R.activeAt(R.setRange(z, [10, 20]), 10) && R.activeAt(R.setRange(z, [10, 20]), 20) && !R.activeAt(R.setRange(z, [10, 20]), 21)
    && R.activeAt(z, 999) && !R.activeAt({ ...z, enabled: false }, 1));
}

console.log('frame-range presets (the toolbar select)');
{
  const z = R.newZone({ id: 'r' });
  ok('presets map to the ranges they name', eq(R.rangeFor('frame', 40, 300), [40, 40]) && eq(R.rangeFor('from', 40, 300), [40, 300])
    && eq(R.rangeFor('until', 40, 300), [1, 40]) && R.rangeFor('clip', 40, 300) === null && R.rangeFor('nonsense', 40, 300) === null);
  ok('and read back as the same preset', R.rangeKind(z) === 'clip' && R.rangeKind(R.setRange(z, [40, 40])) === 'frame'
    && R.rangeKind(R.setRange(z, [40, 300])) === 'from' && R.rangeKind(R.setRange(z, [1, 40])) === 'until');
  ok('"from here" on the last frame is that one frame, never an inverted range', eq(R.rangeFor('from', 300, 300), [300, 300]));
  ok('exclude and include have distinct colours', R.MODE_COLOR.exclude !== R.MODE_COLOR.include);
}

console.log('hit-testing the outline and inserting a point');
{
  let z = R.closeZone(['x', 'y', 'z'].reduce((acc, _, i) => R.addPoint(acc, [0, 100, 50][i], [0, 0, 90][i]), R.newZone()));
  const hit = R.hitSegment(z, 50, 2, 5);
  ok('hitSegment finds the top edge at t = 0.5', hit && hit.seg === 0 && near(hit.t, 0.5, 0.01), JSON.stringify(hit));
  ok('a click far from the outline hits nothing', R.hitSegment(z, 50, 40, 5) === null);
  const grown = R.insertPoint(z, 0, 0.5);
  ok('insertPoint on a straight edge adds a point in place', grown.points.length === 4 && near2([grown.points[1].x, grown.points[1].y], [50, 0]));
  // splitting a curved segment must keep its shape
  let c = R.closeZone({ ...R.newZone(), points: curved.map((p) => ({ ...p })) });
  const split = R.insertPoint(c, 0, 0.5);
  const a = R.flatten(c.points, 32);
  const b = R.flatten(split.points, 16);
  const nearest = (p, line) => Math.min(...line.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])));
  ok('splitting a Bezier keeps its shape (de Casteljau)', b.every((p) => nearest(p, a) < 0.8), `${Math.max(...b.map((p) => nearest(p, a)))}`);
  ok('the split point is ON the original curve', near2([split.points[1].x, split.points[1].y], PY_FLAT_8, 0.01));
  ok('insert wraps: a point on the closing edge goes last', R.insertPoint(z, 2, 0.5).points.length === 4);
}

console.log('point in polygon / area');
{
  const sq = [[0, 0], [10, 0], [10, 10], [0, 10]];
  ok('inside / outside / area', R.pointInPolygon(sq, 5, 5) && !R.pointInPolygon(sq, 15, 5) && R.polygonArea(sq) === 100);
}

console.log('the 2-point pin follows a face');
{
  const ref = { nose: [200, 205], chin: [200, 240] };
  let z = R.closeZone([[230, 200], [260, 200], [260, 230], [230, 230]].reduce((acc, [x, y]) => R.addPoint(acc, x, y), R.newZone()));
  z = R.pinZone(z, ref, 1);
  ok('pinZone records the reference and the source', eq(z.pin, { ref_nose: [200, 205], ref_chin: [200, 240], source_index: 1 }));
  ok('pinZone without anchors changes nothing', R.pinZone(z, null, 1) === z);
  const moved = [{ source_index: 1, nose: [300, 225], chin: [300, 260] }];
  const { outline, tracked } = R.placedOutline(z, moved);
  ok('the zone moved with the face', tracked && near2(outline[0], [330, 220]) && near2(outline[2], [360, 250]), JSON.stringify(outline));
  const rel = (pt, a) => { const ux = (a.chin[0] - a.nose[0]); const uy = (a.chin[1] - a.nose[1]); const l = Math.hypot(ux, uy);
    const dx = pt[0] - a.nose[0]; const dy = pt[1] - a.nose[1]; return [(dx * ux + dy * uy) / l / l, (-dx * uy + dy * ux) / l / l]; };
  const turned = [{ source_index: 1, nose: [260, 230], chin: [225, 262] }];
  const t = R.placedOutline(z, turned);
  ok('rotation + scale: its place in the FACE frame is unchanged',
    near2(rel(t.outline[0], turned[0]), rel(R.flatten(z.points)[0], ref), 1e-3), JSON.stringify([rel(t.outline[0], turned[0]), rel(R.flatten(z.points)[0], ref)]));
  const wrong = [{ source_index: 0, nose: [10, 10], chin: [10, 50] }];
  const far = R.placedOutline(z, [{ source_index: 0, nose: [201, 205], chin: [201, 240] }, { source_index: 1, nose: [500, 205], chin: [500, 240] }]);
  ok('the pin\'s source index outranks proximity', far.tracked && near(far.outline[0][0], 530), JSON.stringify(far.outline[0]));
  ok('a pin with no source follows the nearest nose', (() => {
    const p = { ...z, pin: { ...z.pin, source_index: null } };
    return near(R.placedOutline(p, [{ source_index: 0, nose: [201, 205], chin: [201, 240] }, { source_index: 1, nose: [500, 205], chin: [500, 240] }]).outline[0][0], 231);
  })());
  ok('no face: it stays where it was drawn and says it is untracked',
    (() => { const r = R.placedOutline(z, []); return !r.tracked && near2(r.outline[0], [230, 200]); })());
  ok('an unpinned zone never moves', !R.placedOutline(R.unpinZone(z), wrong).tracked);
}

console.log('serialising for the request');
{
  let z = R.closeZone([[1.234, 2.345], [50, 1], [30, 40]].reduce((acc, [x, y]) => R.addPoint(acc, x, y), R.newZone({ id: 'q', mode: 'include', feather: 7.777 })));
  z = R.setHandle(z, 0, 'hout', 3.14159, 2.71828);
  z = R.pinZone(z, { nose: [10.005, 20], chin: [10, 60] }, 2);
  const open = R.addPoint(R.addPoint(R.newZone({ id: 'open' }), 0, 0), 9, 9);
  const out = R.serialize([z, open, { ...z, id: 'off', enabled: false }]);
  ok('only closed, enabled zones with a triangle go to the backend', out.length === 1 && out[0].id === 'q');
  ok('coordinates are rounded to 0.01 px', out[0].points[0].x === 1.23 && out[0].points[0].hout[0] === 3.14 && out[0].feather === 7.78);
  ok('the pin travels with its source index, rounded', out[0].pin.source_index === 2 && out[0].pin.enabled === true && near(out[0].pin.ref_nose[0], 10.005, 0.006) && out[0].pin.ref_chin[1] === 60);
  ok('UI-only fields are not sent', !('closed' in out[0]) && !('enabled' in out[0]));
  ok('no zones is an empty list', eq(R.serialize(undefined), []) && eq(R.serialize([]), []));
  ok('describe names the zone', R.describe(z, 0) === 'Include 1 · 3 pts · pinned' && R.describe(open, 1).includes('open'));
}

console.log('identity pins');
{
  const pin = L.makePin({ frame: 120, bbox: [100, 50, 220, 200], personId: 1, sourceIndex: 3, faceIndex: 0 });
  ok('a pin from a clicked face', pin && pin.frame === 120 && pin.source_index === 3 && pin.person_id === 'tp_1' && pin.track_id === 2, JSON.stringify(pin));
  ok('the label is Track #N (N = the person number the preview shows)', L.trackLabel(1) === 'Track #2' && L.trackLabel(null) === 'Track');
  ok('no source to lock, a bad box or a bad frame: no pin',
    L.makePin({ frame: 1, bbox: [0, 0, 5, 5], sourceIndex: null }) === null
    && L.makePin({ frame: 1, bbox: [5, 5, 0, 0], sourceIndex: 0 }) === null
    && L.makePin({ frame: NaN, bbox: [0, 0, 5, 5], sourceIndex: 0 }) === null
    && L.makePin({ frame: 1, bbox: [0, 0, 5], sourceIndex: 0 }) === null);
  let pins = L.addPin([], pin);
  pins = L.addPin(pins, L.makePin({ frame: 40, bbox: [0, 0, 50, 50], personId: 0, sourceIndex: 0 }));
  ok('pins are kept in frame order', pins.map((p) => p.frame).join() === '40,120');
  ok('pinning the same person on the same frame again replaces, not duplicates',
    L.addPin(pins, L.makePin({ frame: 120, bbox: [90, 40, 230, 210], personId: 1, sourceIndex: 3 })).length === 2);
  ok('removePin', L.removePin(pins, pin.id).length === 1);
  ok('pinFor finds the pin for a source near a frame', L.pinFor(pins, 122, 3, 5)?.id === pin.id && L.pinFor(pins, 200, 3, 5) === null);
  ok('serializePins drops UI-only fields', !('face_index' in L.serializePins(pins)[0]) && L.serializePins(pins)[0].source_index === 0);
}

console.log('confidence strip');
{
  const points = [[1, 0.9, 0], [31, 0.15, 1], [61, 0.7, 0], [91, 0.45, 0]];
  const cells = L.laneCells(points, 100);
  ok('one cell per sample, tiling the clip without gaps or overlap',
    cells.length === 4 && cells[0].start === 1 && cells.every((c, i) => i === 0 || c.start === cells[i - 1].end + 1) && cells.at(-1).end === 91,
    JSON.stringify(cells.map((c) => [c.start, c.end])));
  ok('the amber sample is amber, others by strength', cells[1].tone === 'amber' && cells[0].tone === 'ok' && cells[3].tone === 'fair');
  ok('a lane with no samples draws nothing', eq(L.laneCells([], 100), []) && eq(L.laneCells(undefined, 100), []));
  ok('junk samples are ignored', L.laneCells([[1, 0.5, 0], null, ['x', 1], [5, NaN, 0]], 10).length === 1);
  const amber = [31, 150, 400];
  ok('next amber walks forward and wraps', L.nextAmber(amber, 1) === 31 && L.nextAmber(amber, 31) === 150 && L.nextAmber(amber, 400) === 31);
  ok('previous amber walks back and wraps', L.nextAmber(amber, 150, -1) === 31 && L.nextAmber(amber, 31, -1) === 400);
  ok('no amber frames: null', L.nextAmber([], 5) === null && L.nextAmber(undefined, 5, -1) === null);
  ok('allAmber merges lanes, deduplicated and sorted', eq(L.allAmber([{ amber: [31, 9] }, { amber: [9, 100] }, {}]), [9, 31, 100]));
}

console.log(failures === 0 ? '\nAll Stage 11 logic checks passed.' : `\n${failures} Stage 11 logic check(s) FAILED.`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
